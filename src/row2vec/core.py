"""
Row2Vec: Core functionality
"""

import random
import time
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import tensorflow as tf
from sklearn.base import BaseEstimator
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler, StandardScaler, normalize

if TYPE_CHECKING:
    from tensorflow.keras.callbacks import Callback, EarlyStopping
else:
    try:
        from tensorflow.keras.callbacks import Callback, EarlyStopping
    except ImportError:
        Callback = object
        EarlyStopping = object
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.models import Model

from .config import EmbeddingConfig
from .logging import Row2VecLogger, get_logger
from .pipeline_builder import build_adaptive_pipeline
from .utils import create_dataframe_schema


class Row2VecTrainingCallback(Callback):  # type: ignore[misc]
    """Keras callback for Row2Vec training progress logging."""

    def __init__(self, logger: Row2VecLogger):
        super().__init__()
        self.logger = logger

    def on_epoch_end(self, epoch: int, logs: Any = None) -> None:
        """Called at the end of each epoch."""
        if logs is None:
            logs = {}

        # Extract metrics
        loss = logs.get("loss", 0.0)
        val_loss = logs.get("val_loss")

        # Remove loss from additional metrics to avoid duplication
        additional_metrics = {
            k: v for k, v in logs.items() if k not in ["loss", "val_loss"]
        }

        self.logger.log_epoch_metrics(
            epoch=epoch,
            loss=loss,
            val_loss=val_loss,
            additional_metrics=additional_metrics if additional_metrics else None,
        )


def _scale_embeddings(
    df_in: pd.DataFrame,
    method: str,
    rng: tuple[float, float] | None,
) -> pd.DataFrame:
    """Apply scaling to embeddings using sklearn scalers where possible."""
    if method == "none":
        return df_in

    values: npt.NDArray[np.float64] = df_in.values.astype(float, copy=False)

    if method == "minmax":
        # Default range if not provided
        if rng is None:
            rng = (0.0, 1.0)
        feature_range: tuple[float, float] = (float(rng[0]), float(rng[1]))

        # Use sklearn MinMaxScaler
        scaler = MinMaxScaler(feature_range=feature_range)
        try:
            scaled_values = scaler.fit_transform(values)
            return pd.DataFrame(scaled_values, columns=df_in.columns, index=df_in.index)
        except ValueError as e:
            # Handle constant columns with more informative error message
            if "Constant" in str(e) or "constant" in str(e):
                # Find which columns are constant
                for j in range(values.shape[1]):
                    col = values[:, j]
                    if np.max(col) == np.min(col):
                        col_name = (
                            df_in.columns[j]
                            if j < len(df_in.columns)
                            else f"column_{j}"
                        )
                        raise ValueError(
                            f"MinMax scaling undefined for constant column '{col_name}' "
                            f"(all values are {np.min(col)}). Consider removing this column or using a different scaling method.",
                        ) from e
            raise

    if method == "standard":
        # Use sklearn StandardScaler
        scaler = StandardScaler()
        scaled_values = scaler.fit_transform(values)
        return pd.DataFrame(scaled_values, columns=df_in.columns, index=df_in.index)

    if method == "l2":
        # Use sklearn normalize function
        normalized_values = normalize(values, norm="l2", axis=1)
        return pd.DataFrame(normalized_values, columns=df_in.columns, index=df_in.index)

    if method == "tanh":
        # tanh scaling doesn't have a direct sklearn equivalent, keep manual implementation
        scaled = np.tanh(values)
        return pd.DataFrame(scaled, columns=df_in.columns, index=df_in.index)

    valid_methods = ["none", "minmax", "standard", "l2", "tanh"]
    raise ValueError(
        f"Invalid scale_method '{method}'. Must be one of {valid_methods}. "
        f"See documentation for details on each scaling method.",
    )


def _validate_inputs(
    df: pd.DataFrame,
    embedding_dim: int,
    mode: str,
    reference_column: str | None,
    max_epochs: int,
    batch_size: int,
    dropout_rate: float,
    hidden_units: int | list[int],
    scale_method: str | None,
    scale_range: tuple[float, float] | None,
    # Classical ML parameters
    n_neighbors: int = 15,
    perplexity: float = 30.0,
    min_dist: float = 0.1,
    n_iter: int = 1000,
    # Contrastive learning parameters
    similar_pairs: list[tuple[int, int]] | None = None,
    dissimilar_pairs: list[tuple[int, int]] | None = None,
    auto_pairs: str | None = None,
    contrastive_loss: str = "triplet",
    margin: float = 1.0,
    negative_samples: int = 5,
) -> None:
    """
    Comprehensive input validation for learn_embedding function.

    Args:
        df: Input DataFrame
        embedding_dim: Dimensionality of embedding space
        mode: Learning mode ('unsupervised' or 'target')
        reference_column: Target column for supervised mode
        max_epochs: Maximum training epochs
        batch_size: Training batch size
        dropout_rate: Dropout rate for regularization
        hidden_units: Hidden layer units (int for single layer, list of ints for multiple layers)
        scale_method: Scaling method for embeddings
        scale_range: Range for minmax scaling

    Raises:
        TypeError: If input types are incorrect
        ValueError: If input values are invalid
    """
    # 1. Validate DataFrame
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected pandas DataFrame, got {type(df).__name__}")

    if df.empty:
        raise ValueError(
            "DataFrame cannot be empty. Please provide a non-empty DataFrame.",
        )

    if len(df.columns) == 0:
        raise ValueError("DataFrame must have at least one column.")

    if df.shape[0] < 2:
        raise ValueError(
            f"DataFrame must have at least 2 rows for training. Got {df.shape[0]} rows.",
        )

    # 2. Validate embedding dimension
    if not isinstance(embedding_dim, int):
        raise TypeError(
            f"embedding_dim must be an integer, got {type(embedding_dim).__name__}",
        )

    if embedding_dim <= 0:
        raise ValueError(f"embedding_dim must be positive, got {embedding_dim}")

    # Note: We'll validate embedding_dim against actual feature count after preprocessing
    # since one-hot encoding can significantly increase the feature count

    # 3. Validate mode
    valid_modes = ["unsupervised", "target", "pca", "tsne", "umap", "contrastive"]
    if mode not in valid_modes:
        raise ValueError(f"mode must be one of {valid_modes}, got '{mode}'")

    # 4. Validate reference column for target mode
    if mode == "target":
        if reference_column is None:
            raise ValueError("reference_column is required when mode='target'")

        if not isinstance(reference_column, str):
            raise TypeError(
                f"reference_column must be a string, got {type(reference_column).__name__}",
            )

        if reference_column not in df.columns:
            available_cols = list(df.columns)
            raise ValueError(
                f"reference_column '{reference_column}' not found in DataFrame. "
                f"Available columns: {available_cols}",
            )

        # Check if reference column has valid data for classification
        unique_values = df[reference_column].nunique()
        if unique_values < 2:
            raise ValueError(
                f"reference_column '{reference_column}' must have at least 2 unique values "
                f"for classification. Got {unique_values} unique value(s).",
            )

        if unique_values > 1000:
            raise ValueError(
                f"reference_column '{reference_column}' has too many unique values ({unique_values}). "
                f"Consider using mode='unsupervised' for high-cardinality categorical data.",
            )

    # 4b. Validate classical ML method parameters
    if mode == "tsne":
        if not isinstance(perplexity, int | float) or perplexity <= 0:
            raise ValueError(f"perplexity must be a positive number, got {perplexity}")

        if not isinstance(n_iter, int) or n_iter < 250:
            raise ValueError(f"max_iter must be an integer >= 250, got {n_iter}")

        # t-SNE works best with perplexity < n_samples/3
        max_perplexity = max(5, df.shape[0] / 3)
        if perplexity > max_perplexity:
            raise ValueError(
                f"perplexity ({perplexity}) should be less than {max_perplexity:.1f} "
                f"for dataset with {df.shape[0]} samples",
            )

    if mode == "umap":
        if not isinstance(n_neighbors, int) or n_neighbors <= 0:
            raise ValueError(
                f"n_neighbors must be a positive integer, got {n_neighbors}",
            )

        if not isinstance(min_dist, int | float) or min_dist <= 0:
            raise ValueError(f"min_dist must be a positive number, got {min_dist}")

        if n_neighbors >= df.shape[0]:
            raise ValueError(
                f"n_neighbors ({n_neighbors}) must be less than dataset size ({df.shape[0]})",
            )

    # 5. Validate training parameters
    if not isinstance(max_epochs, int) or max_epochs <= 0:
        raise ValueError(f"max_epochs must be a positive integer, got {max_epochs}")

    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError(f"batch_size must be a positive integer, got {batch_size}")

    # Only validate batch_size for neural methods that actually use it
    if mode in ["unsupervised", "target", "contrastive"] and batch_size > df.shape[0]:
        raise ValueError(
            f"batch_size ({batch_size}) cannot be larger than dataset size ({df.shape[0]})",
        )

    if not isinstance(dropout_rate, int | float) or not (0 <= dropout_rate < 1):
        raise ValueError(
            f"dropout_rate must be a number between 0 and 1, got {dropout_rate}",
        )

    # Validate hidden_units (can be int or list of ints)
    if isinstance(hidden_units, int):
        if hidden_units <= 0:
            raise ValueError(
                f"hidden_units must be a positive integer, got {hidden_units}"
            )
    elif isinstance(hidden_units, list):
        if not hidden_units:
            raise ValueError("hidden_units list cannot be empty")
        if not all(isinstance(h, int) and h > 0 for h in hidden_units):
            raise ValueError(
                f"All hidden_units must be positive integers, got {hidden_units}"
            )
    else:
        raise ValueError(
            f"hidden_units must be an integer or list of integers, got {hidden_units}"
        )

    # 6. Validate scaling parameters
    if scale_method is not None:
        valid_scale_methods = ["none", "minmax", "standard", "l2", "tanh"]
        if scale_method not in valid_scale_methods:
            raise ValueError(
                f"scale_method must be one of {valid_scale_methods}, got '{scale_method}'",
            )

    if scale_range is not None:
        if not isinstance(scale_range, tuple | list) or len(scale_range) != 2:
            raise TypeError("scale_range must be a tuple or list of two numbers")

        low, high = scale_range
        if not isinstance(low, int | float) or not isinstance(high, int | float):
            raise TypeError("scale_range values must be numbers")

        if low >= high:
            raise ValueError(
                f"scale_range low value ({low}) must be less than high value ({high})",
            )

    # 6.1. Validate contrastive learning parameters
    if mode == "contrastive":
        # Validate contrastive loss type
        valid_contrastive_losses = ["triplet", "contrastive"]
        if contrastive_loss not in valid_contrastive_losses:
            raise ValueError(
                f"contrastive_loss must be one of {valid_contrastive_losses}, got '{contrastive_loss}'",
            )

        # Validate margin
        if not isinstance(margin, int | float) or margin <= 0:
            raise ValueError(f"margin must be a positive number, got {margin}")

        # Validate negative_samples
        if not isinstance(negative_samples, int) or negative_samples <= 0:
            raise ValueError(
                f"negative_samples must be a positive integer, got {negative_samples}"
            )

        # Validate auto_pairs if provided
        if auto_pairs is not None:
            valid_auto_pairs = ["cluster", "neighbors", "categorical", "random"]
            if auto_pairs not in valid_auto_pairs:
                raise ValueError(
                    f"auto_pairs must be one of {valid_auto_pairs}, got '{auto_pairs}'",
                )

        # Validate pairs if provided
        if similar_pairs is not None:
            if not isinstance(similar_pairs, list):
                raise TypeError("similar_pairs must be a list of tuples")
            for i, pair in enumerate(similar_pairs):
                if not isinstance(pair, tuple) or len(pair) != 2:
                    raise ValueError(
                        f"similar_pairs[{i}] must be a tuple of 2 integers"
                    )
                if not all(isinstance(idx, int) for idx in pair):
                    raise ValueError(f"similar_pairs[{i}] must contain only integers")
                if not all(0 <= idx < len(df) for idx in pair):
                    raise ValueError(f"similar_pairs[{i}] contains invalid row indices")

        if dissimilar_pairs is not None:
            if not isinstance(dissimilar_pairs, list):
                raise TypeError("dissimilar_pairs must be a list of tuples")
            for i, pair in enumerate(dissimilar_pairs):
                if not isinstance(pair, tuple) or len(pair) != 2:
                    raise ValueError(
                        f"dissimilar_pairs[{i}] must be a tuple of 2 integers"
                    )
                if not all(isinstance(idx, int) for idx in pair):
                    raise ValueError(
                        f"dissimilar_pairs[{i}] must contain only integers"
                    )
                if not all(0 <= idx < len(df) for idx in pair):
                    raise ValueError(
                        f"dissimilar_pairs[{i}] contains invalid row indices"
                    )

        # Check that we have some way to generate pairs
        if similar_pairs is None and dissimilar_pairs is None and auto_pairs is None:
            raise ValueError(
                "For contrastive mode, you must provide either similar_pairs/dissimilar_pairs "
                "or specify auto_pairs strategy",
            )

    # 7. Validate data content
    # Check for all-NaN columns
    nan_cols = df.columns[df.isnull().all()].tolist()
    if nan_cols:
        raise ValueError(f"DataFrame contains columns with all NaN values: {nan_cols}")

    # Check if there's any usable data
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    # Remove reference column from feature columns if in target mode
    if mode == "target" and reference_column in categorical_cols:
        categorical_cols = [col for col in categorical_cols if col != reference_column]
    if mode == "target" and reference_column in numeric_cols:
        numeric_cols = [col for col in numeric_cols if col != reference_column]

    if not numeric_cols and not categorical_cols:
        raise ValueError(
            "DataFrame must contain at least one numeric or categorical column "
            "(excluding reference_column in target mode)",
        )


def _learn_classical_embedding(
    X: npt.NDArray[Any],
    df: pd.DataFrame,
    embedding_dim: int,
    mode: str,
    scale_method: str | None,
    scale_range: tuple[float, float] | None,
    n_neighbors: int,
    perplexity: float,
    min_dist: float,
    n_iter: int,
    seed: int,
    verbose: bool,
    logger: Row2VecLogger | None,
) -> pd.DataFrame:
    """
    Learn embeddings using classical ML methods (PCA, t-SNE, UMAP).

    Args:
        X: Preprocessed feature matrix
        df: Original DataFrame for logging context
        embedding_dim: Target embedding dimensionality
        mode: Method to use ('pca', 'tsne', 'umap')
        scale_method: Post-processing scaling method
        scale_range: Range for scaling
        n_neighbors: UMAP n_neighbors parameter
        perplexity: t-SNE perplexity parameter
        min_dist: UMAP min_dist parameter
        n_iter: t-SNE n_iter parameter
        seed: Random seed
        verbose: Whether to print progress
        logger: Optional logger instance

    Returns:
        DataFrame with embeddings
    """

    if logger:
        logger.log_debug_info(
            f"Starting {mode.upper()} embedding with {embedding_dim} dimensions",
        )
        logger.log_debug_info(f"Input shape: {X.shape}")

    start_time = time.time()

    if mode == "pca":
        if logger:
            logger.log_debug_info(f"Fitting PCA with {embedding_dim} components")

        from sklearn.decomposition import PCA

        model = PCA(n_components=embedding_dim, random_state=seed)
        embeddings = model.fit_transform(X)

        # Log explained variance ratio
        if logger:
            explained_variance = model.explained_variance_ratio_
            total_variance = explained_variance.sum()
            logger.log_debug_info(f"PCA explained variance ratio: {total_variance:.3f}")
            logger.log_debug_info(f"Top 3 components: {explained_variance[:3]}")

    elif mode == "tsne":
        if logger:
            logger.log_debug_info(
                f"Fitting t-SNE with perplexity={perplexity}, max_iter={n_iter}",
            )

        # Check for optimal embedding dimension for t-SNE
        if embedding_dim > 3 and logger:
            logger.log_performance_warning(
                f"t-SNE with embedding_dim={embedding_dim} may not be optimal. "
                "Consider using 2 or 3 dimensions for t-SNE visualization.",
            )

        from sklearn.manifold import TSNE

        # Use exact method for higher dimensions since barnes_hut is limited to <=3D
        method = "exact" if embedding_dim > 3 else "barnes_hut"
        model = TSNE(
            n_components=embedding_dim,
            perplexity=perplexity,
            max_iter=n_iter,
            method=method,
            random_state=seed,
            verbose=1 if verbose else 0,
        )
        embeddings = model.fit_transform(X)

        if logger:
            logger.log_debug_info(f"t-SNE KL divergence: {model.kl_divergence_:.3f}")

    elif mode == "umap":
        try:
            import umap
        except ImportError:
            raise ImportError(
                "UMAP is not installed. Please install it with: pip install umap-learn",
            )

        if logger:
            logger.log_debug_info(
                f"Fitting UMAP with n_neighbors={n_neighbors}, min_dist={min_dist}",
            )

        model = umap.UMAP(
            n_components=embedding_dim,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            random_state=seed,
            verbose=verbose,
        )
        embeddings = model.fit_transform(X)

    else:
        raise ValueError(f"Unsupported classical method: {mode}")

    elapsed_time = time.time() - start_time

    if logger:
        logger.log_debug_info(
            f"{mode.upper()} fitting completed in {elapsed_time:.2f} seconds",
        )
        logger.log_debug_info(f"Output shape: {embeddings.shape}")

    # Create DataFrame
    embedding_df = pd.DataFrame(
        embeddings,
        columns=[f"embedding_{i}" for i in range(embedding_dim)],
    )

    # Apply scaling if requested
    def _scale_embeddings_classical(
        df_in: pd.DataFrame,
        method: str | None,
        rng: tuple[float, float] | None,
    ) -> pd.DataFrame:
        """Apply scaling to embeddings - delegates to main scaling function."""
        if method is None:
            method = "none"
        # Delegate to the main scaling function to avoid code duplication
        return _scale_embeddings(df_in, method, rng)

    final_embeddings = _scale_embeddings_classical(
        embedding_df,
        scale_method,
        scale_range,
    )

    if logger:
        if scale_method and scale_method != "none":
            logger.log_debug_info(f"Applied {scale_method} scaling to embeddings")
        logger.log_debug_info("Classical embedding process completed successfully")

    return final_embeddings


def _generate_contrastive_pairs(
    df: pd.DataFrame,
    similar_pairs: list[tuple[int, int]] | None,
    dissimilar_pairs: list[tuple[int, int]] | None,
    auto_pairs: str | None,
    reference_column: str | None,
    X_processed: npt.NDArray[Any],
    negative_samples: int,
    contrastive_loss: str,
    seed: int,
    logger: Row2VecLogger | None = None,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Generate similarity and dissimilarity pairs for contrastive learning."""
    np.random.seed(seed)

    final_similar_pairs = similar_pairs[:] if similar_pairs else []
    final_dissimilar_pairs = dissimilar_pairs[:] if dissimilar_pairs else []

    if auto_pairs is not None:
        if logger:
            logger.log_debug_info(
                f"Generating automatic pairs using strategy: {auto_pairs}"
            )

        n_samples = len(df)

        if auto_pairs == "cluster":
            # Use clustering to find similar/dissimilar pairs
            n_clusters = min(10, n_samples // 20 + 2)  # Adaptive cluster count
            kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
            cluster_labels = kmeans.fit_predict(X_processed)

            # Generate similar pairs from same clusters
            for cluster_id in range(n_clusters):
                cluster_indices = np.where(cluster_labels == cluster_id)[0]
                if len(cluster_indices) >= 2:
                    # Sample pairs within cluster
                    n_pairs = min(
                        len(cluster_indices) // 2, 50
                    )  # Limit pairs per cluster
                    for _ in range(n_pairs):
                        idx1, idx2 = np.random.choice(cluster_indices, 2, replace=False)
                        final_similar_pairs.append((int(idx1), int(idx2)))

            # Generate dissimilar pairs from different clusters
            for _ in range(min(len(final_similar_pairs) * negative_samples, 500)):
                idx1 = np.random.randint(n_samples)
                # Find a point from a different cluster
                different_cluster_indices = np.where(
                    cluster_labels != cluster_labels[idx1]
                )[0]
                if len(different_cluster_indices) > 0:
                    idx2 = np.random.choice(different_cluster_indices)
                    final_dissimilar_pairs.append((int(idx1), int(idx2)))

        elif auto_pairs == "neighbors":
            # Use k-NN to find similar/dissimilar pairs
            k = min(10, max(2, n_samples // 10))  # Ensure at least k=2
            nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm="auto")
            nbrs.fit(X_processed)
            distances, indices = nbrs.kneighbors(X_processed)

            # Generate similar pairs from nearest neighbors
            for i in range(n_samples):
                # Skip first neighbor (itself) and sample from rest
                neighbors = indices[i][1:]  # Skip self
                n_pairs = min(len(neighbors) // 2, 10)  # Limit pairs per point
                for j in range(n_pairs):
                    neighbor_idx = neighbors[j]
                    final_similar_pairs.append((i, int(neighbor_idx)))

            # Generate dissimilar pairs from distant points
            for _ in range(min(len(final_similar_pairs) * negative_samples, 500)):
                idx1 = np.random.randint(n_samples)
                # Sample from points that are NOT in the k-nearest neighbors
                non_neighbors = np.setdiff1d(np.arange(n_samples), indices[idx1])
                if len(non_neighbors) > 0:
                    idx2 = np.random.choice(non_neighbors)
                    final_dissimilar_pairs.append((int(idx1), int(idx2)))

        elif auto_pairs == "categorical":
            # Use categorical columns to define similarity
            categorical_cols = df.select_dtypes(
                include=["object", "category"]
            ).columns.tolist()
            if reference_column and reference_column in categorical_cols:
                categorical_cols.remove(reference_column)

            if categorical_cols:
                # Use first categorical column for similarity
                cat_col = categorical_cols[0]
                unique_values = df[cat_col].unique()

                # Generate similar pairs within same category
                for value in unique_values:
                    indices = df[df[cat_col] == value].index.tolist()
                    if len(indices) >= 2:
                        n_pairs = min(len(indices) // 2, 50)
                        for _ in range(n_pairs):
                            idx1, idx2 = np.random.choice(indices, 2, replace=False)
                            final_similar_pairs.append((int(idx1), int(idx2)))

                # Generate dissimilar pairs from different categories
                for _ in range(min(len(final_similar_pairs) * negative_samples, 500)):
                    value1 = np.random.choice(unique_values)
                    different_values = unique_values[unique_values != value1]
                    if len(different_values) > 0:
                        value2 = np.random.choice(different_values)
                        indices1 = df[df[cat_col] == value1].index.tolist()
                        indices2 = df[df[cat_col] == value2].index.tolist()
                        if indices1 and indices2:
                            idx1 = np.random.choice(indices1)
                            idx2 = np.random.choice(indices2)
                            final_dissimilar_pairs.append((int(idx1), int(idx2)))
            else:
                # Fallback to random if no categorical columns
                auto_pairs = "random"

        if auto_pairs == "random":
            # Generate random pairs
            n_similar = min(n_samples // 10, 200)  # Adaptive number
            for _ in range(n_similar):
                idx1, idx2 = np.random.choice(n_samples, 2, replace=False)
                final_similar_pairs.append((int(idx1), int(idx2)))

            # Generate random dissimilar pairs
            for _ in range(min(len(final_similar_pairs) * negative_samples, 500)):
                idx1, idx2 = np.random.choice(n_samples, 2, replace=False)
                final_dissimilar_pairs.append((int(idx1), int(idx2)))

    if logger:
        logger.log_debug_info(
            f"Generated {len(final_similar_pairs)} similar pairs and "
            f"{len(final_dissimilar_pairs)} dissimilar pairs",
        )

    return final_similar_pairs, final_dissimilar_pairs


def _create_contrastive_loss_function(loss_type: str, margin: float) -> Any:
    """Create the contrastive loss function."""

    if loss_type == "triplet":

        def triplet_loss(y_true: Any, y_pred: Any) -> Any:
            """Triplet loss: minimize distance between anchor-positive, maximize anchor-negative."""
            # y_pred contains [anchor, positive, negative] embeddings
            # Shape: (batch_size, 3 * embedding_dim)
            embedding_dim = tf.shape(y_pred)[1] // 3

            anchor = y_pred[:, :embedding_dim]
            positive = y_pred[:, embedding_dim : 2 * embedding_dim]
            negative = y_pred[:, 2 * embedding_dim :]

            # Calculate distances
            pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=1)
            neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=1)

            # Triplet loss with margin
            loss = tf.maximum(0.0, pos_dist - neg_dist + margin)
            return tf.reduce_mean(loss)

        return triplet_loss

    if loss_type == "contrastive":

        def contrastive_loss(y_true: Any, y_pred: Any) -> Any:
            """Contrastive loss: minimize distance for similar pairs, maximize for dissimilar."""
            # y_pred contains [anchor, comparison] embeddings
            # Shape should be: (batch_size, 2 * embedding_dim)
            # y_true: 1 for similar pairs, 0 for dissimilar pairs

            # Debug: print shapes
            # tf.print("y_pred shape:", tf.shape(y_pred))
            # tf.print("y_true shape:", tf.shape(y_true))

            embedding_dim = tf.shape(y_pred)[1] // 2

            anchor = y_pred[:, :embedding_dim]
            comparison = y_pred[:, embedding_dim:]

            # Calculate Euclidean distance
            distance = tf.sqrt(
                tf.reduce_sum(tf.square(anchor - comparison), axis=1) + 1e-8
            )

            # Contrastive loss
            similar_loss = y_true * tf.square(distance)
            dissimilar_loss = (1 - y_true) * tf.square(
                tf.maximum(0.0, margin - distance)
            )

            return tf.reduce_mean(similar_loss + dissimilar_loss)

        return contrastive_loss

    raise ValueError(f"Unknown contrastive loss type: {loss_type}")


def _build_contrastive_model(
    input_dim: int,
    embedding_dim: int,
    hidden_units: int | list[int],
    dropout_rate: float,
    loss_type: str,
    margin: float,
    seed: int,
) -> Model:
    """Build the contrastive learning model."""
    tf.random.set_seed(seed)

    # Shared encoder network
    encoder_input = Input(shape=(input_dim,), name="encoder_input")
    x = encoder_input

    # Build hidden layers
    if isinstance(hidden_units, int):
        # Single layer architecture
        x = Dense(hidden_units, activation="relu")(x)
        x = Dropout(dropout_rate)(x)
        x = Dense(hidden_units // 2, activation="relu")(x)
        x = Dropout(dropout_rate)(x)
    else:
        # Multi-layer architecture
        for i, units in enumerate(hidden_units):
            x = Dense(units, activation="relu", name=f"hidden_{i + 1}")(x)
            x = Dropout(dropout_rate)(x)

    embedding_output = Dense(embedding_dim, activation=None, name="embedding")(x)

    encoder = Model(encoder_input, embedding_output, name="encoder")

    if loss_type == "triplet":
        # Triplet network: anchor, positive, negative inputs
        anchor_input = Input(shape=(input_dim,), name="anchor")
        positive_input = Input(shape=(input_dim,), name="positive")
        negative_input = Input(shape=(input_dim,), name="negative")

        anchor_emb = encoder(anchor_input)
        positive_emb = encoder(positive_input)
        negative_emb = encoder(negative_input)

        # Concatenate embeddings for loss calculation
        concat_output = tf.keras.layers.Concatenate()(
            [anchor_emb, positive_emb, negative_emb]
        )

        model = Model(
            inputs=[anchor_input, positive_input, negative_input],
            outputs=concat_output,
            name="triplet_model",
        )

    elif loss_type == "contrastive":
        # Siamese network: two inputs
        input1 = Input(shape=(input_dim,), name="input1")
        input2 = Input(shape=(input_dim,), name="input2")

        emb1 = encoder(input1)
        emb2 = encoder(input2)

        # Concatenate embeddings for loss calculation
        concat_output = tf.keras.layers.Concatenate()([emb1, emb2])

        model = Model(
            inputs=[input1, input2],
            outputs=concat_output,
            name="contrastive_model",
        )

    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

    # Compile model
    loss_fn = _create_contrastive_loss_function(loss_type, margin)
    model.compile(
        optimizer="adam",
        loss=loss_fn,
        metrics=[],
    )

    # Store encoder for later use
    model.encoder = encoder

    return model


def _create_contrastive_dataset(
    X_processed: npt.NDArray[Any],
    similar_pairs: list[tuple[int, int]],
    dissimilar_pairs: list[tuple[int, int]],
    batch_size: int,
    loss_type: str,
    seed: int,
) -> Any:
    """Create a TensorFlow dataset for contrastive learning."""
    np.random.seed(seed)

    all_pairs = []
    all_labels = []

    # Add similar pairs
    for pair in similar_pairs:
        all_pairs.append(pair)
        all_labels.append(1)  # Similar

    # Add dissimilar pairs
    for pair in dissimilar_pairs:
        all_pairs.append(pair)
        all_labels.append(0)  # Dissimilar

    # Shuffle
    combined = list(zip(all_pairs, all_labels, strict=False))
    np.random.shuffle(combined)
    unzipped = list(zip(*combined, strict=False))
    all_pairs = list(unzipped[0])
    all_labels = list(unzipped[1])

    if loss_type == "contrastive":
        # Prepare data for contrastive learning
        input1_data = []
        input2_data = []
        labels_data = []

        for (idx1, idx2), label in zip(all_pairs, all_labels, strict=False):
            input1_data.append(X_processed[idx1])
            input2_data.append(X_processed[idx2])
            labels_data.append(label)

        input1_array = np.array(input1_data, dtype=np.float32)
        input2_array = np.array(input2_data, dtype=np.float32)
        labels_array = np.array(labels_data, dtype=np.float32)

        # Create dataset by yielding batched data
        def data_generator() -> Any:
            n_samples = len(input1_array)
            indices = np.arange(n_samples)

            while True:  # Infinite generator for repeated epochs
                np.random.shuffle(indices)
                for i in range(0, n_samples, batch_size):
                    batch_indices = indices[i : i + batch_size]
                    if len(batch_indices) == 0:
                        continue

                    batch_input1 = input1_array[batch_indices]
                    batch_input2 = input2_array[batch_indices]
                    batch_labels = labels_array[batch_indices]

                    yield ((batch_input1, batch_input2), batch_labels)

        # Convert generator to TensorFlow dataset
        return tf.data.Dataset.from_generator(
            data_generator,
            output_signature=(
                (
                    tf.TensorSpec(shape=(None, X_processed.shape[1]), dtype=tf.float32),
                    tf.TensorSpec(shape=(None, X_processed.shape[1]), dtype=tf.float32),
                ),
                tf.TensorSpec(shape=(None,), dtype=tf.float32),
            ),
        )

    if loss_type == "triplet":
        # Prepare data for triplet learning (anchor, positive, negative)
        anchor_data = []
        positive_data = []
        negative_data = []

        # Convert similar/dissimilar pairs to triplets
        similar_dict: dict[int, list[int]] = {}
        for idx1, idx2 in similar_pairs:
            if idx1 not in similar_dict:
                similar_dict[idx1] = []
            similar_dict[idx1].append(idx2)
            if idx2 not in similar_dict:
                similar_dict[idx2] = []
            similar_dict[idx2].append(idx1)

        dissimilar_list = list(dissimilar_pairs)

        # Generate triplets: (anchor, positive, negative)
        for anchor_idx, positives in similar_dict.items():
            if len(positives) == 0:
                continue

            # For each anchor, create multiple triplets
            num_triplets = min(len(positives), 5)  # Limit triplets per anchor
            for _ in range(num_triplets):
                positive_idx = np.random.choice(positives)

                # Find a negative sample (from dissimilar pairs or random)
                negative_candidates = []
                for d_pair in dissimilar_list:
                    if d_pair[0] == anchor_idx:
                        negative_candidates.append(d_pair[1])
                    elif d_pair[1] == anchor_idx:
                        negative_candidates.append(d_pair[0])

                if len(negative_candidates) == 0:
                    # Random negative sampling
                    negative_idx = np.random.randint(len(X_processed))
                    while negative_idx == anchor_idx or negative_idx in positives:
                        negative_idx = np.random.randint(len(X_processed))
                else:
                    negative_idx = np.random.choice(negative_candidates)

                anchor_data.append(X_processed[anchor_idx])
                positive_data.append(X_processed[positive_idx])
                negative_data.append(X_processed[negative_idx])

        anchor_array = np.array(anchor_data, dtype=np.float32)
        positive_array = np.array(positive_data, dtype=np.float32)
        negative_array = np.array(negative_data, dtype=np.float32)

        # Create dataset by yielding batched data
        def triplet_data_generator() -> Any:
            n_samples = len(anchor_array)
            indices = np.arange(n_samples)

            while True:  # Infinite generator for repeated epochs
                np.random.shuffle(indices)
                for i in range(0, n_samples, batch_size):
                    batch_indices = indices[i : i + batch_size]
                    if len(batch_indices) == 0:
                        continue

                    batch_anchor = anchor_array[batch_indices]
                    batch_positive = positive_array[batch_indices]
                    batch_negative = negative_array[batch_indices]

                    # Dummy labels (not used in triplet loss)
                    batch_labels = np.zeros(len(batch_indices), dtype=np.float32)

                    yield ((batch_anchor, batch_positive, batch_negative), batch_labels)

        # Convert generator to TensorFlow dataset
        return tf.data.Dataset.from_generator(
            triplet_data_generator,
            output_signature=(
                (
                    tf.TensorSpec(shape=(None, X_processed.shape[1]), dtype=tf.float32),
                    tf.TensorSpec(shape=(None, X_processed.shape[1]), dtype=tf.float32),
                    tf.TensorSpec(shape=(None, X_processed.shape[1]), dtype=tf.float32),
                ),
                tf.TensorSpec(shape=(None,), dtype=tf.float32),
            ),
        )

    raise ValueError(
        f"Unknown loss type: {loss_type}. Supported types: 'contrastive', 'triplet'"
    )


def learn_embedding(
    df: pd.DataFrame,
    embedding_dim: int = 10,
    mode: str = "unsupervised",  # "unsupervised", "target", "pca", "tsne", "umap", "contrastive"
    reference_column: str | None = None,
    max_epochs: int = 50,
    batch_size: int = 64,
    dropout_rate: float = 0.2,
    hidden_units: int | list[int] = 128,
    early_stopping: bool = True,
    seed: int = 1305,
    verbose: bool = False,
    scale_method: str | None = None,
    scale_range: tuple[float, float] | None = None,
    log_level: str = "INFO",
    log_file: str | None = None,
    enable_logging: bool = True,
    # Classical ML method parameters
    n_neighbors: int = 15,  # for UMAP
    perplexity: float = 30.0,  # for t-SNE
    min_dist: float = 0.1,  # for UMAP
    n_iter: int = 1000,  # for t-SNE
    # Contrastive learning parameters
    similar_pairs: list[tuple[int, int]] | None = None,
    dissimilar_pairs: list[tuple[int, int]] | None = None,
    auto_pairs: str | None = None,  # "cluster", "neighbors", "categorical", "random"
    contrastive_loss: str = "triplet",  # "triplet", "contrastive"
    margin: float = 1.0,
    negative_samples: int = 5,
    # Preprocessing configuration
    config: EmbeddingConfig | None = None,
) -> pd.DataFrame:
    """
    Learns a low-dimensional embedding from a pandas DataFrame.

    Note:
        Current version supports numeric and categorical features. Textual and temporal
        features are not directly supported - please preprocess them yourself using
        appropriate tools (e.g., BERT-like embeddings for text, temporal libraries for
        time series). Support for these feature types is planned for future versions.

    Args:
        df (pd.DataFrame): The input DataFrame containing numeric and categorical features.
        embedding_dim (int): The dimensionality of the embedding space.
        mode (str): Embedding method - 'unsupervised' (autoencoder), 'target' (supervised),
                   'pca' (Principal Component Analysis), 'tsne' (t-SNE), 'umap' (UMAP),
                   or 'contrastive' (contrastive learning).
        reference_column (str): The target column for 'target' mode.
        max_epochs (int): The maximum number of training epochs (neural methods only).
        batch_size (int): The batch size for training (neural methods only).
        dropout_rate (float): The dropout rate for regularization (neural methods only).
        hidden_units (Union[int, list[int]]): Hidden layer configuration - single int for one layer
                     or list of ints for multiple layers (neural methods only).
        early_stopping (bool): Whether to use early stopping (neural methods only).
        seed (int): A random seed for reproducibility.
        verbose (bool): Whether to print training progress.
        scale_method (str, optional): Scaling method for embeddings. Options:
                     'none', 'minmax', 'standard', 'l2', 'tanh'.
        scale_range (tuple, optional): Range for minmax scaling. Default: (0, 1).
        log_level (str): Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR').
        log_file (str, optional): File path for logging output.
        enable_logging (bool): Whether to enable structured logging.
        n_neighbors (int): Number of neighbors for UMAP (default: 15).
        perplexity (float): Perplexity parameter for t-SNE (default: 30.0).
        min_dist (float): Minimum distance for UMAP (default: 0.1).
        n_iter (int): Number of iterations for t-SNE (default: 1000).
        similar_pairs (list[tuple[int, int]], optional): List of (row_idx1, row_idx2) pairs
                     that should have similar embeddings (for contrastive mode).
        dissimilar_pairs (list[tuple[int, int]], optional): List of (row_idx1, row_idx2) pairs
                     that should have dissimilar embeddings (for contrastive mode).
        auto_pairs (str, optional): Strategy for automatic pair generation. Options:
                     'cluster' (cluster-based), 'neighbors' (k-NN based),
                     'categorical' (same category values), 'random' (random sampling).
        contrastive_loss (str): Contrastive loss function. Options: 'triplet', 'contrastive'.
        margin (float): Margin parameter for contrastive loss functions (default: 1.0).
        negative_samples (int): Number of negative samples per positive pair (default: 5).
        config (EmbeddingConfig, optional): Configuration for preprocessing and model behavior.
                     If None, intelligent defaults are used based on data analysis.

    Returns:
        pd.DataFrame: A DataFrame containing the learned embeddings.

    Raises:
        ValueError: If input validation fails or unsupported mode is specified.
        TypeError: If input types are incorrect.
    """

    # === LOGGING SETUP ===
    logger = None
    if enable_logging:
        logger = get_logger(
            name="row2vec.learn_embedding",
            level=log_level,
            log_file=log_file,
            include_performance=True,
            include_memory=True,
        )

        # Log training start with configuration (basic info only before validation)
        config_dict = {
            "mode": mode,
            "embedding_dim": embedding_dim,
            "max_epochs": max_epochs,
            "batch_size": batch_size,
            "dropout_rate": dropout_rate,
            "hidden_units": hidden_units,
            "early_stopping": early_stopping,
            "scale_method": scale_method,
        }

        # Add data shape only if df is actually a DataFrame
        if hasattr(df, "shape"):
            config_dict["data_shape"] = df.shape

        logger.start_training(**config_dict)

    # === INPUT VALIDATION ===
    _validate_inputs(
        df,
        embedding_dim,
        mode,
        reference_column,
        max_epochs,
        batch_size,
        dropout_rate,
        hidden_units,
        scale_method,
        scale_range,
        n_neighbors,
        perplexity,
        min_dist,
        n_iter,
        similar_pairs,
        dissimilar_pairs,
        auto_pairs,
        contrastive_loss,
        margin,
        negative_samples,
    )

    # Resolve scaling behavior defaults
    if scale_method is None:
        scale_method = "minmax" if scale_range is not None else "none"

    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    # Set TF seeds and prefer deterministic ops when available
    try:  # TensorFlow 2.13+
        from tensorflow.config import experimental as tf_config_exp

        tf_config_exp.enable_op_determinism(True)
    except Exception:
        pass

    # === DATA PREPROCESSING ===
    preprocessing_start_time = time.time()

    # Initialize configuration with intelligent defaults if none provided
    if config is None:
        config = EmbeddingConfig()

    # Prepare target variable for supervised preprocessing
    y: pd.Series | None = None
    num_classes: int = 0
    target_series: pd.Series | None = None

    if mode == "target":
        # reference_column validation already done in _validate_inputs
        assert reference_column is not None  # Type hint for mypy
        # Remove rows with missing target values to avoid pandas cat.codes == -1
        non_null_mask = df[reference_column].notnull()
        if not non_null_mask.all():
            dropped = int((~non_null_mask).sum())
            if logger:
                logger.log_debug_info(
                    f"Dropping {dropped} rows with missing '{reference_column}' for target mode",
                )
            # Work on a filtered copy for the remainder of the pipeline/training
            df = df.loc[non_null_mask].reset_index(drop=True)

        target_series = df[reference_column]
        y = target_series.astype("category").cat.codes
        num_classes = len(np.unique(y))

        if logger:
            logger.log_debug_info(
                f"Target mode: {num_classes} classes in '{reference_column}'",
            )

    # Build intelligent preprocessing pipeline
    preprocessor, analysis_report = build_adaptive_pipeline(
        df=df,
        target=target_series,
        config=config,
        mode=mode,
    )

    if logger:
        # Log intelligent preprocessing decisions
        processing_steps = [
            f"Dataset analyzed: {analysis_report['dataset_shape'][0]} rows, {analysis_report['dataset_shape'][1]} columns",
            f"Missing data: {analysis_report['missing_percentage']:.1f}%",
            f"Numeric features: {analysis_report['numeric_columns']}",
            f"Categorical features: {analysis_report['categorical_columns']}",
            f"Categorical encoding strategy: {config.preprocessing.categorical_encoding_strategy}",
            f"Memory usage: {analysis_report['memory_usage_mb']:.1f} MB",
        ]
        logger.log_data_preprocessing(df.shape, processing_steps)

    # Apply preprocessing
    train_df = df.drop(columns=[reference_column]) if mode == "target" else df
    X: npt.NDArray[Any] | Any = preprocessor.fit_transform(train_df)

    # Initialize for type checkers
    y_train: pd.Series | None = None
    y_test: pd.Series | None = None

    # Ensure X is a dense array for TensorFlow compatibility, especially for autoencoders
    if hasattr(X, "toarray"):
        X = X.toarray()

    preprocessing_time = time.time() - preprocessing_start_time
    if logger:
        logger.log_preprocessing_result(
            df.shape, (X.shape[0], X.shape[1]), preprocessing_time
        )

        # Performance warnings
        if X.shape[1] > 1000:
            logger.log_performance_warning(
                f"High-dimensional input ({X.shape[1]} features) may slow training. "
                "Consider feature selection or dimensionality reduction.",
            )

        if df.shape[0] > 100000:
            logger.log_performance_warning(
                f"Large dataset ({df.shape[0]} rows) detected. "
                "Consider using larger batch_size or distributed training for better performance.",
            )

        if embedding_dim > X.shape[1] * 0.8:
            logger.log_performance_warning(
                f"Embedding dimension ({embedding_dim}) is close to input features ({X.shape[1]}). "
                "This may lead to overfitting.",
            )

    # Validate embedding dimension against actual feature count after preprocessing
    actual_feature_count = X.shape[1]
    if embedding_dim > actual_feature_count:
        error_msg = (
            f"embedding_dim ({embedding_dim}) cannot be larger than the number of features "
            f"after preprocessing ({actual_feature_count}). Consider reducing embedding_dim or adding more features."
        )
        if logger:
            logger.log_validation_issue(error_msg)
        raise ValueError(error_msg)

    # === CLASSICAL ML METHODS ===
    if mode in ["pca", "tsne", "umap"]:
        return _learn_classical_embedding(
            X,
            df,
            embedding_dim,
            mode,
            scale_method,
            scale_range,
            n_neighbors,
            perplexity,
            min_dist,
            n_iter,
            seed,
            verbose,
            logger,
        )

    # === CONTRASTIVE LEARNING ===
    if mode == "contrastive":
        if logger:
            logger.log_debug_info("Starting contrastive learning training")

        # Generate pairs for contrastive learning
        similar_pairs_final, dissimilar_pairs_final = _generate_contrastive_pairs(
            df,
            similar_pairs,
            dissimilar_pairs,
            auto_pairs,
            reference_column,
            X,
            negative_samples,
            contrastive_loss,
            seed,
            logger,
        )

        if logger:
            logger.log_debug_info(f"Using {contrastive_loss} loss with margin={margin}")

        # Build contrastive model
        model = _build_contrastive_model(
            X.shape[1],
            embedding_dim,
            hidden_units,
            dropout_rate,
            contrastive_loss,
            margin,
            seed,
        )

        # Create training dataset
        train_dataset = _create_contrastive_dataset(
            X,
            similar_pairs_final,
            dissimilar_pairs_final,
            batch_size,
            contrastive_loss,
            seed,
        )

        # Calculate steps per epoch
        total_pairs = len(similar_pairs_final) + len(dissimilar_pairs_final)
        steps_per_epoch = max(1, total_pairs // batch_size)

        # Train the model
        callbacks = []
        if early_stopping:
            callbacks.append(
                EarlyStopping(monitor="loss", patience=5, restore_best_weights=True)
            )

        if logger:
            callbacks.append(Row2VecTrainingCallback(logger))

        history = model.fit(
            train_dataset,
            steps_per_epoch=steps_per_epoch,
            epochs=max_epochs,
            verbose=1 if verbose else 0,
            callbacks=callbacks,
        )

        # Extract embeddings using the encoder
        embedded_values = model.encoder.predict(X, verbose=0)
        embedding_df = pd.DataFrame(
            embedded_values,
            columns=[f"embedding_{i}" for i in range(embedding_dim)],
        )

        # Apply scaling
        final_embeddings = _scale_embeddings(embedding_df, scale_method, scale_range)

        # Log completion
        if logger:
            epochs_trained = len(history.history["loss"])
            final_loss = history.history["loss"][-1]
            logger.end_training(final_loss, epochs_trained)
            logger.log_embedding_stats(final_embeddings)
            logger.log_completion("Contrastive learning completed successfully!")

        return final_embeddings

    # === NEURAL NETWORK METHODS ===
    if mode == "target":
        X_train: npt.NDArray[Any]
        X_test: npt.NDArray[Any]
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=seed,
        )
    else:
        X_train, X_test = train_test_split(X, test_size=0.2, random_state=seed)

    # === MODEL ARCHITECTURE ===
    input_layer = Input(shape=(X_train.shape[1],))

    # Build encoder layers
    x = input_layer
    if isinstance(hidden_units, int):
        # Single layer architecture
        x = Dense(hidden_units, activation="relu")(x)
        x = Dropout(dropout_rate)(x)
    else:
        # Multi-layer architecture
        for i, units in enumerate(hidden_units):
            x = Dense(units, activation="relu", name=f"encoder_hidden_{i + 1}")(x)
            x = Dropout(dropout_rate)(x)

    encoded = Dense(embedding_dim, activation="linear", name="embedding")(x)

    if mode == "target":
        output = Dense(num_classes, activation="softmax")(encoded)
        model = Model(inputs=input_layer, outputs=output)
        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
    else:
        # Build decoder layers (mirror of encoder)
        decoded = encoded
        if isinstance(hidden_units, int):
            # Single layer architecture
            decoded = Dense(hidden_units, activation="relu")(decoded)
            decoded = Dropout(dropout_rate)(decoded)
        else:
            # Multi-layer architecture (reverse order)
            for i, units in enumerate(reversed(hidden_units)):
                decoded = Dense(
                    units, activation="relu", name=f"decoder_hidden_{i + 1}"
                )(decoded)
                decoded = Dropout(dropout_rate)(decoded)

        decoded = Dense(X_train.shape[1], activation="linear")(decoded)
        model = Model(inputs=input_layer, outputs=decoded)
        model.compile(optimizer="adam", loss="mse")

    if logger:
        # Create a string representation of the model
        import io
        import sys

        # Capture model summary
        string_buffer = io.StringIO()
        original_stdout = sys.stdout
        sys.stdout = string_buffer
        try:
            model.summary()
            model_summary = string_buffer.getvalue()
        finally:
            sys.stdout = original_stdout

        logger.log_model_architecture(model_summary)
        logger.log_debug_info(
            "Model compiled",
            {
                "mode": mode,
                "input_shape": X_train.shape,
                "embedding_dim": embedding_dim,
                "hidden_units": hidden_units,
                "dropout_rate": dropout_rate,
            },
        )

    model_callbacks: list[Callback] = []
    if early_stopping:
        model_callbacks.append(
            EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        )

    # Add logging callback if logger is enabled
    if logger:
        model_callbacks.append(Row2VecTrainingCallback(logger))

    # === MODEL TRAINING ===
    history = model.fit(
        X_train,
        y_train if mode == "target" else X_train,
        validation_data=(X_test, y_test if mode == "target" else X_test),
        epochs=max_epochs,
        batch_size=batch_size,
        verbose=int(verbose),
        callbacks=model_callbacks,
    )

    # Log training completion
    if logger:
        final_loss = history.history["loss"][-1]
        epochs_trained = len(history.history["loss"])
        logger.end_training(final_loss, epochs_trained)

        if epochs_trained < max_epochs:
            logger.log_early_stopping(epochs_trained - 1, "EarlyStopping triggered")

    # === EMBEDDING EXTRACTION ===
    encoder: Model = Model(
        inputs=model.input,
        outputs=model.get_layer("embedding").output,
    )
    final_embedded_values: npt.NDArray[Any] = encoder.predict(X, verbose=0)
    final_embedding_df: pd.DataFrame = pd.DataFrame(
        final_embedded_values,
        columns=[f"embedding_{i}" for i in range(embedding_dim)],
    )

    if mode == "target":
        final_embedding_df["category"] = (
            df[reference_column].astype("category").cat.codes
        )
        grouped: pd.DataFrame = final_embedding_df.groupby("category").mean()
        # Apply scaling after grouping per user preference
        final_embeddings = _scale_embeddings(grouped, scale_method, scale_range)
    else:
        # Apply scaling to row embeddings
        final_embeddings = _scale_embeddings(
            final_embedding_df, scale_method, scale_range
        )

    # Log final embedding statistics
    if logger:
        logger.log_embedding_stats(final_embeddings)
        logger.log_completion()

    return final_embeddings


def learn_embedding_with_model(
    df: pd.DataFrame,
    embedding_dim: int = 10,
    mode: str = "unsupervised",  # "unsupervised", "target", "pca", "tsne", "umap"
    reference_column: str | None = None,
    max_epochs: int = 50,
    batch_size: int = 64,
    dropout_rate: float = 0.2,
    hidden_units: int | list[int] = 128,
    early_stopping: bool = True,
    seed: int = 1305,
    verbose: bool = False,
    scale_method: str | None = None,
    scale_range: tuple[float, float] | None = None,
    log_level: str = "INFO",
    log_file: str | None = None,
    enable_logging: bool = True,
    # Classical ML method parameters
    n_neighbors: int = 15,  # for UMAP
    perplexity: float = 30.0,  # for t-SNE
    min_dist: float = 0.1,  # for UMAP
    n_iter: int = 1000,  # for t-SNE
    # Contrastive learning parameters
    similar_pairs: list[tuple[int, int]] | None = None,
    dissimilar_pairs: list[tuple[int, int]] | None = None,
    auto_pairs: str | None = None,
    negative_samples: int = 5,
    contrastive_loss: str = "triplet",
    margin: float = 1.0,
    # Preprocessing configuration
    config: EmbeddingConfig | None = None,
) -> tuple[
    pd.DataFrame, tf.keras.Model | BaseEstimator, ColumnTransformer, dict[str, Any]
]:
    """
    Extended version of learn_embedding that also returns the model, preprocessor, and training metadata.

    This function is designed for use with the serialization system to capture all necessary
    components for saving and loading trained models.

    Args:
        Same as learn_embedding function

    Returns:
        Tuple of (embeddings, model, preprocessor, metadata)
        - embeddings: DataFrame with learned embeddings
        - model: Trained model (Keras model for neural methods, sklearn estimator for classical)
        - preprocessor: Fitted sklearn ColumnTransformer for data preprocessing
        - metadata: Dictionary containing training metadata and configuration
    """
    start_time = time.time()

    # Initialize logger
    logger = None
    if enable_logging:
        logger = get_logger(
            name="row2vec.learn_embedding_with_model",
            level=log_level,
            log_file=log_file,
        )

    # Store original DataFrame schema for validation
    original_schema = create_dataframe_schema(df)

    # We'll need to essentially duplicate the learn_embedding logic but capture additional info
    # For brevity, I'll implement this by calling the existing function and then reconstructing
    # the model and preprocessor. This is not the most efficient approach, but it maintains
    # compatibility with the existing codebase.

    # First, get the embeddings using the existing function
    embeddings = learn_embedding(
        df=df,
        embedding_dim=embedding_dim,
        mode=mode,
        reference_column=reference_column,
        max_epochs=max_epochs,
        batch_size=batch_size,
        dropout_rate=dropout_rate,
        hidden_units=hidden_units,
        early_stopping=early_stopping,
        seed=seed,
        verbose=verbose,
        scale_method=scale_method,
        scale_range=scale_range,
        log_level=log_level,
        log_file=log_file,
        enable_logging=enable_logging,
        n_neighbors=n_neighbors,
        perplexity=perplexity,
        min_dist=min_dist,
        n_iter=n_iter,
        similar_pairs=similar_pairs,
        dissimilar_pairs=dissimilar_pairs,
        auto_pairs=auto_pairs,
        negative_samples=negative_samples,
        contrastive_loss=contrastive_loss,
        margin=margin,
    )

    # Initialize configuration with intelligent defaults if none provided
    if config is None:
        config = EmbeddingConfig()

    # Prepare target variable for supervised preprocessing
    target_series: pd.Series | None = None
    if mode == "target" and reference_column:
        target_series = df[reference_column]

    # Build intelligent preprocessing pipeline
    preprocessor, analysis_report = build_adaptive_pipeline(
        df=df,
        target=target_series,
        config=config,
        mode=mode,
    )

    # Fit the preprocessor
    train_df = df.drop(columns=[reference_column]) if mode == "target" else df
    X_processed = preprocessor.fit_transform(train_df)

    if hasattr(X_processed, "toarray"):
        X_processed = X_processed.toarray()

    # Now reconstruct the model based on mode
    model: tf.keras.Model | BaseEstimator
    training_history: dict[str, Any] = {}
    final_loss: float | None = None
    epochs_trained: int | None = None

    if mode in ["pca", "tsne", "umap"]:
        # For classical methods, recreate the fitted estimator
        if mode == "pca":
            from sklearn.decomposition import PCA

            model = PCA(n_components=embedding_dim, random_state=seed)
        elif mode == "tsne":
            from sklearn.manifold import TSNE

            model = TSNE(
                n_components=embedding_dim,
                perplexity=perplexity,
                n_iter=n_iter,
                random_state=seed,
                verbose=1 if verbose else 0,
            )
        else:  # umap
            try:
                import umap

                model = umap.UMAP(
                    n_components=embedding_dim,
                    n_neighbors=n_neighbors,
                    min_dist=min_dist,
                    random_state=seed,
                    verbose=verbose,
                )
            except ImportError:
                raise ImportError(
                    "UMAP not installed. Install with: pip install umap-learn"
                )

        # Fit the model
        if mode == "tsne":
            # t-SNE doesn't have a separate fit/transform
            model.fit_transform(X_processed)
        else:
            model.fit(X_processed)

    else:
        # Neural network methods - reconstruct the model
        np.random.seed(seed)
        random.seed(seed)
        tf.random.set_seed(seed)

        input_shape = X_processed.shape[1]

        if mode == "target" and reference_column:
            y = df[reference_column].astype("category").cat.codes
            num_classes = len(np.unique(y))
            X_train, X_test, y_train, y_test = train_test_split(
                X_processed,
                y,
                test_size=0.2,
                random_state=seed,
            )
        elif mode == "contrastive":
            # Generate pairs for contrastive learning
            similar_pairs_final, dissimilar_pairs_final = _generate_contrastive_pairs(
                df,
                similar_pairs,
                dissimilar_pairs,
                auto_pairs,
                reference_column,
                X_processed,
                negative_samples,
                contrastive_loss,
                seed,
                logger,
            )

            if logger:
                logger.log_debug_info(
                    f"Using {contrastive_loss} loss with margin={margin}"
                )

            # Build contrastive model
            model = _build_contrastive_model(
                input_shape,
                embedding_dim,
                hidden_units,
                dropout_rate,
                contrastive_loss,
                margin,
                seed,
            )

            # Create training dataset
            train_dataset = _create_contrastive_dataset(
                X_processed,
                similar_pairs_final,
                dissimilar_pairs_final,
                batch_size,
                contrastive_loss,
                seed,
            )

            # Calculate steps per epoch
            total_pairs = len(similar_pairs_final) + len(dissimilar_pairs_final)
            steps_per_epoch = max(1, total_pairs // batch_size)

            # Train the model
            callbacks = []
            if early_stopping:
                callbacks.append(
                    EarlyStopping(monitor="loss", patience=5, restore_best_weights=True)
                )

            if logger:
                callbacks.append(Row2VecTrainingCallback(logger))

            history = model.fit(
                train_dataset,
                steps_per_epoch=steps_per_epoch,
                epochs=max_epochs,
                verbose=1 if verbose else 0,
                callbacks=callbacks,
            )

            epochs_trained = len(history.history["loss"])

            # Extract embeddings using the encoder
            embeddings = model.encoder.predict(X_processed)

        else:
            X_train, X_test = train_test_split(
                X_processed, test_size=0.2, random_state=seed
            )
            y_train = y_test = None
            num_classes = 0

        # Build and train model (for non-contrastive modes)
        if mode != "contrastive":
            # Build model architecture
            input_layer = Input(shape=(input_shape,))
            encoded = Dense(hidden_units, activation="relu")(input_layer)
            encoded = Dropout(dropout_rate)(encoded)
            encoded = Dense(embedding_dim, activation="linear", name="embedding")(
                encoded
            )

            if mode == "target":
                output = Dense(num_classes, activation="softmax")(encoded)
                model = Model(inputs=input_layer, outputs=output)
                model.compile(
                    optimizer="adam",
                    loss="sparse_categorical_crossentropy",
                    metrics=["accuracy"],
                )
            else:
                decoded = Dense(hidden_units, activation="relu")(encoded)
                decoded = Dropout(dropout_rate)(decoded)
                decoded = Dense(input_shape, activation="linear")(decoded)

                model = Model(inputs=input_layer, outputs=decoded)
                model.compile(optimizer="adam", loss="mse")

            # Train the model
            callbacks = []
            if early_stopping:
                callbacks.append(
                    EarlyStopping(
                        monitor="val_loss", patience=5, restore_best_weights=True
                    )
                )

            history = model.fit(
                X_train,
                y_train if mode == "target" else X_train,
                epochs=max_epochs,
                batch_size=batch_size,
                validation_data=(X_test, y_test if mode == "target" else X_test),
                verbose=1 if verbose else 0,
                callbacks=callbacks,
            )

            epochs_trained = len(history.history["loss"])

            # Extract embeddings
            if mode == "target":
                # For target mode, get embeddings from the encoder part
                encoder = Model(
                    inputs=model.input, outputs=model.get_layer("embedding").output
                )
                embeddings = encoder.predict(X_processed)
            else:
                # For unsupervised mode, get embeddings from the encoder part
                encoder = Model(
                    inputs=model.input, outputs=model.get_layer("embedding").output
                )
                embeddings = encoder.predict(X_processed)

    # Convert embeddings to DataFrame with proper column names
    embedding_columns = [f"embedding_{i}" for i in range(embeddings.shape[1])]
    embeddings_df = pd.DataFrame(embeddings, columns=embedding_columns, index=df.index)

    # Handle target mode grouping
    if mode == "target" and reference_column is not None:
        # Add category codes for grouping
        embeddings_df["category"] = df[reference_column].astype("category").cat.codes
        embeddings = embeddings_df.groupby("category").mean()
    else:
        embeddings = embeddings_df

    # Calculate training time
    training_time = time.time() - start_time

    # Prepare metadata
    metadata = {
        # Training configuration
        "embedding_dim": embedding_dim,
        "mode": mode,
        "reference_column": reference_column,
        "max_epochs": max_epochs,
        "batch_size": batch_size,
        "dropout_rate": dropout_rate,
        "hidden_units": hidden_units,
        "early_stopping": early_stopping,
        "seed": seed,
        "scale_method": scale_method,
        "scale_range": scale_range,
        "n_neighbors": n_neighbors,
        "perplexity": perplexity,
        "min_dist": min_dist,
        "n_iter": n_iter,
        # Training results
        "training_history": training_history,
        "final_loss": final_loss,
        "epochs_trained": epochs_trained,
        "training_time": training_time,
        # Data information
        "original_columns": list(df.columns),
        "preprocessed_feature_names": _get_feature_names(preprocessor),
        "data_shape": df.shape,
        "data_types": {col: str(dtype) for col, dtype in df.dtypes.items()},
        # Schema validation
        "expected_schema": original_schema,
    }

    return embeddings, model, preprocessor, metadata


def _get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """
    Extract feature names from a fitted ColumnTransformer.

    Args:
        preprocessor: Fitted ColumnTransformer

    Returns:
        List of feature names
    """
    try:
        # Try to get feature names if available (sklearn 1.0+)
        if hasattr(preprocessor, "get_feature_names_out"):
            return list(preprocessor.get_feature_names_out())
    except:
        pass

    # Fallback: construct feature names manually
    feature_names = []

    for name, transformer, columns in preprocessor.transformers_:
        if name == "remainder":
            continue

        if hasattr(transformer, "get_feature_names_out"):
            try:
                names = transformer.get_feature_names_out(columns)
                feature_names.extend(names)
            except:
                # Fallback for transformers without proper feature name support
                if (
                    hasattr(transformer, "named_steps")
                    and "onehot" in transformer.named_steps
                ):
                    # For categorical features with OneHot
                    onehot = transformer.named_steps["onehot"]
                    if hasattr(onehot, "categories_"):
                        for i, col in enumerate(columns):
                            for category in onehot.categories_[i]:
                                feature_names.append(f"{col}_{category}")
                else:
                    # For numeric features
                    feature_names.extend([f"{name}_{col}" for col in columns])
        else:
            # Basic fallback
            feature_names.extend([f"{name}_{col}" for col in columns])

    return feature_names
