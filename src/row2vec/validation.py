"""Input validation for :func:`row2vec.learn_embedding` and friends.

Kept in its own module so the public facade and the fitted
:class:`row2vec.model.Row2VecModel` share one definition of what a valid
request looks like.
"""

import pandas as pd

from .utils import categorical_columns, numeric_columns

__all__ = ["validate_inputs"]


def validate_inputs(
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
    """Comprehensive input validation for learn_embedding function.

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
        n_neighbors: Neighbourhood size for UMAP
        perplexity: Perplexity for t-SNE
        min_dist: Minimum distance between points for UMAP
        n_iter: Number of optimisation iterations for t-SNE
        similar_pairs: Explicit (i, j) row pairs that should embed close together
        dissimilar_pairs: Explicit (i, j) row pairs that should embed far apart
        auto_pairs: Strategy for deriving pairs automatically
        contrastive_loss: Contrastive objective ('triplet' or 'contrastive')
        margin: Margin of the contrastive objective
        negative_samples: Number of negative samples drawn per anchor

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
            raise ValueError(f"hidden_units must be a positive integer, got {hidden_units}")
    elif isinstance(hidden_units, list):
        if not hidden_units:
            raise ValueError("hidden_units list cannot be empty")
        if not all(isinstance(h, int) and h > 0 for h in hidden_units):
            raise ValueError(f"All hidden_units must be positive integers, got {hidden_units}")
    else:
        raise ValueError(f"hidden_units must be an integer or list of integers, got {hidden_units}")

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
            raise ValueError(f"negative_samples must be a positive integer, got {negative_samples}")

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
                    raise ValueError(f"similar_pairs[{i}] must be a tuple of 2 integers")
                if not all(isinstance(idx, int) for idx in pair):
                    raise ValueError(f"similar_pairs[{i}] must contain only integers")
                if not all(0 <= idx < len(df) for idx in pair):
                    raise ValueError(f"similar_pairs[{i}] contains invalid row indices")

        if dissimilar_pairs is not None:
            if not isinstance(dissimilar_pairs, list):
                raise TypeError("dissimilar_pairs must be a list of tuples")
            for i, pair in enumerate(dissimilar_pairs):
                if not isinstance(pair, tuple) or len(pair) != 2:
                    raise ValueError(f"dissimilar_pairs[{i}] must be a tuple of 2 integers")
                if not all(isinstance(idx, int) for idx in pair):
                    raise ValueError(f"dissimilar_pairs[{i}] must contain only integers")
                if not all(0 <= idx < len(df) for idx in pair):
                    raise ValueError(f"dissimilar_pairs[{i}] contains invalid row indices")

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
    numeric_cols = numeric_columns(df)
    categorical_cols = categorical_columns(df)

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
