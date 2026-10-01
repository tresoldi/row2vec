"""Contrastive-learning helpers: pair generation, losses, and the siamese model.

These live apart from :mod:`row2vec.core` so that both the public facade and the
fitted :class:`row2vec.model.Row2VecModel` can use them without importing each
other.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

from ._backend import require_tensorflow
from .utils import categorical_columns

if TYPE_CHECKING:
    import pandas as pd
    from tensorflow.keras.models import Model

    from .logging import Row2VecLogger

__all__ = [
    "build_contrastive_model",
    "create_contrastive_dataset",
    "create_contrastive_loss_function",
    "generate_contrastive_pairs",
]


def generate_contrastive_pairs(
    df: pd.DataFrame,
    similar_pairs: list[tuple[int, int]] | None,
    dissimilar_pairs: list[tuple[int, int]] | None,
    auto_pairs: str | None,
    reference_column: str | None,
    X_processed: npt.NDArray[Any],
    negative_samples: int,
    seed: int,
    logger: Row2VecLogger | None = None,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Generate similarity and dissimilarity pairs for contrastive learning."""
    np.random.seed(seed)

    final_similar_pairs = similar_pairs[:] if similar_pairs else []
    final_dissimilar_pairs = dissimilar_pairs[:] if dissimilar_pairs else []

    if auto_pairs is not None:
        if logger:
            logger.log_debug_info(f"Generating automatic pairs using strategy: {auto_pairs}")

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
                    n_pairs = min(len(cluster_indices) // 2, 50)  # Limit pairs per cluster
                    for _ in range(n_pairs):
                        idx1, idx2 = np.random.choice(cluster_indices, 2, replace=False)
                        final_similar_pairs.append((int(idx1), int(idx2)))

            # Generate dissimilar pairs from different clusters
            for _ in range(min(len(final_similar_pairs) * negative_samples, 500)):
                idx1 = np.random.randint(n_samples)
                # Find a point from a different cluster
                different_cluster_indices = np.where(cluster_labels != cluster_labels[idx1])[0]
                if len(different_cluster_indices) > 0:
                    idx2 = np.random.choice(different_cluster_indices)
                    final_dissimilar_pairs.append((int(idx1), int(idx2)))

        elif auto_pairs == "neighbors":
            # Use k-NN to find similar/dissimilar pairs
            k = min(10, max(2, n_samples // 10))  # Ensure at least k=2
            nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm="auto")
            nbrs.fit(X_processed)
            _distances, indices = nbrs.kneighbors(X_processed)

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
            categorical_cols = categorical_columns(df)
            if reference_column and reference_column in categorical_cols:
                categorical_cols.remove(reference_column)

            if categorical_cols:
                # Use first categorical column for similarity
                cat_col = categorical_cols[0]
                unique_values = df[cat_col].unique()

                # Generate similar pairs within same category
                for value in unique_values:
                    indices = np.flatnonzero(df[cat_col].to_numpy() == value).tolist()
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
                        indices1 = np.flatnonzero(df[cat_col].to_numpy() == value1).tolist()
                        indices2 = np.flatnonzero(df[cat_col].to_numpy() == value2).tolist()
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


def create_contrastive_loss_function(loss_type: str, margin: float) -> Any:
    """Create the contrastive loss function."""
    tf = require_tensorflow("mode='contrastive'")

    if loss_type == "triplet":

        def triplet_loss(y_true: Any, y_pred: Any) -> Any:  # noqa: ARG001  (Keras loss signature)
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
            distance = tf.sqrt(tf.reduce_sum(tf.square(anchor - comparison), axis=1) + 1e-8)

            # Contrastive loss
            similar_loss = y_true * tf.square(distance)
            dissimilar_loss = (1 - y_true) * tf.square(tf.maximum(0.0, margin - distance))

            return tf.reduce_mean(similar_loss + dissimilar_loss)

        return contrastive_loss

    raise ValueError(f"Unknown contrastive loss type: {loss_type}")


def build_contrastive_model(
    input_dim: int,
    embedding_dim: int,
    hidden_units: int | list[int],
    dropout_rate: float,
    loss_type: str,
    margin: float,
    seed: int,
) -> Model:
    """Build the contrastive learning model."""
    tf = require_tensorflow("mode='contrastive'")
    from tensorflow.keras.layers import Dense, Dropout, Input
    from tensorflow.keras.models import Model

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
        concat_output = tf.keras.layers.Concatenate()([anchor_emb, positive_emb, negative_emb])

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
    loss_fn = create_contrastive_loss_function(loss_type, margin)
    model.compile(
        optimizer="adam",
        loss=loss_fn,
        metrics=[],
    )

    # Store encoder for later use
    model.encoder = encoder

    return model


def create_contrastive_dataset(
    X_processed: npt.NDArray[Any],
    similar_pairs: list[tuple[int, int]],
    dissimilar_pairs: list[tuple[int, int]],
    batch_size: int,
    loss_type: str,
    seed: int,
) -> Any:
    """Create a TensorFlow dataset for contrastive learning."""
    tf = require_tensorflow("mode='contrastive'")
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

    raise ValueError(f"Unknown loss type: {loss_type}. Supported types: 'contrastive', 'triplet'")
