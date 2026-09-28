"""Public entry points for learning embeddings.

Both functions here are thin facades over :class:`row2vec.model.Row2VecModel`,
which owns every piece of fitted state. Before 0.3.0 this module carried two
near-duplicate 400-line implementations that had quietly diverged on nine
separate points - multi-layer support, embedding scaling, null-target
handling, the t-SNE iteration keyword, output indexing - and
``learn_embedding_with_model`` trained a second model from scratch after
already calling ``learn_embedding``, returning embeddings from one model and
the other model itself.

The keyword signature is kept so existing code and documentation keep working.
What changed is the semantics, and all of it is documented in MIGRATION.md.
"""

import pandas as pd

from .config import EmbeddingConfig
from .model import Row2VecModel, Row2VecTrainingCallback, get_feature_names

__all__ = [
    "Row2VecTrainingCallback",
    "get_feature_names",
    "learn_embedding",
    "learn_embedding_with_model",
]


def _build_config(
    base: EmbeddingConfig | None,
    *,
    embedding_dim: int,
    mode: str,
    reference_column: str | None,
    max_epochs: int,
    batch_size: int,
    dropout_rate: float,
    hidden_units: int | list[int],
    activation: str,
    early_stopping: bool,
    seed: int,
    verbose: bool,
    scale_method: str | None,
    scale_range: tuple[float, float] | None,
    log_level: str,
    log_file: str | None,
    enable_logging: bool,
    n_neighbors: int,
    perplexity: float,
    min_dist: float,
    n_iter: int,
    similar_pairs: list[tuple[int, int]] | None,
    dissimilar_pairs: list[tuple[int, int]] | None,
    auto_pairs: str | None,
    contrastive_loss: str,
    margin: float,
    negative_samples: int,
) -> EmbeddingConfig:
    """Fold loose keyword arguments into an :class:`EmbeddingConfig`.

    When ``base`` is supplied its preprocessing settings are carried over;
    every other field comes from the keyword arguments, which is what the
    caller wrote explicitly.
    """
    config = EmbeddingConfig(
        embedding_dim=embedding_dim,
        mode=mode,
        reference_column=reference_column,
        seed=seed,
        verbose=verbose,
    )

    config.neural.max_epochs = max_epochs
    config.neural.batch_size = batch_size
    config.neural.dropout_rate = dropout_rate
    config.neural.hidden_units = hidden_units
    config.neural.activation = activation
    config.neural.early_stopping = early_stopping

    config.classical.n_neighbors = n_neighbors
    config.classical.perplexity = perplexity
    config.classical.min_dist = min_dist
    config.classical.n_iter = n_iter

    config.contrastive.loss_type = contrastive_loss
    config.contrastive.similar_pairs = similar_pairs
    config.contrastive.dissimilar_pairs = dissimilar_pairs
    config.contrastive.auto_pairs = auto_pairs
    config.contrastive.margin = margin
    config.contrastive.negative_samples = negative_samples

    config.scaling.method = scale_method
    config.scaling.range = scale_range

    config.logging.level = log_level
    config.logging.file = log_file
    config.logging.enabled = enable_logging

    if base is not None:
        config.preprocessing = base.preprocessing

    return config


def learn_embedding(
    df: pd.DataFrame,
    embedding_dim: int = 10,
    mode: str = "unsupervised",
    reference_column: str | None = None,
    max_epochs: int = 50,
    batch_size: int = 64,
    dropout_rate: float = 0.2,
    hidden_units: int | list[int] = 128,
    activation: str = "relu",
    early_stopping: bool = True,
    seed: int = 1305,
    verbose: bool = False,
    scale_method: str | None = None,
    scale_range: tuple[float, float] | None = None,
    log_level: str = "INFO",
    log_file: str | None = None,
    enable_logging: bool = True,
    n_neighbors: int = 15,
    perplexity: float = 30.0,
    min_dist: float = 0.1,
    n_iter: int = 1000,
    similar_pairs: list[tuple[int, int]] | None = None,
    dissimilar_pairs: list[tuple[int, int]] | None = None,
    auto_pairs: str | None = None,
    contrastive_loss: str = "triplet",
    margin: float = 1.0,
    negative_samples: int = 5,
    aggregate_by_reference: bool = False,
    config: EmbeddingConfig | None = None,
) -> pd.DataFrame:
    """Learn an embedding for every row of ``df``.

    Args:
        df (pd.DataFrame): Input data with numeric and/or categorical columns.
        embedding_dim (int): Width of the embedding space.
        mode (str): One of ``"unsupervised"`` (autoencoder), ``"target"``
            (supervised encoder), ``"pca"``, ``"tsne"``, ``"umap"`` or
            ``"contrastive"``.
        reference_column (str, optional): Label column. Required for
            ``mode="target"`` and used by ``auto_pairs="categorical"``.
        max_epochs (int): Training epoch ceiling for neural modes.
        batch_size (int): Training batch size.
        dropout_rate (float): Dropout applied after each hidden layer.
        hidden_units (int | list[int]): Width of the hidden layer, or a list of
            widths for a multi-layer encoder.
        activation (str): Hidden-layer activation. Previously hardcoded to
            ``"relu"`` no matter what was configured.
        early_stopping (bool): Whether to stop once the monitored loss stalls.
        seed (int): Random seed, applied to Python, NumPy and TensorFlow.
        verbose (bool): Whether to let the underlying libraries print progress.
        scale_method (str, optional): ``"none"``, ``"minmax"``, ``"standard"``,
            ``"l2"`` or ``"tanh"``. The fitted scaler is retained, so a saved
            model reproduces these values rather than rescaling against
            whatever batch it is given.
        scale_range (tuple[float, float], optional): Output range for
            ``"minmax"``.
        log_level (str): Logging level.
        log_file (str, optional): Path to write logs to.
        enable_logging (bool): Whether to log at all.
        n_neighbors (int): UMAP neighbourhood size.
        perplexity (float): t-SNE perplexity.
        min_dist (float): UMAP minimum distance.
        n_iter (int): t-SNE iteration count.
        similar_pairs (list[tuple[int, int]], optional): Explicit positive
            pairs, given as *positions* into ``df``.
        dissimilar_pairs (list[tuple[int, int]], optional): Explicit negative
            pairs, given as *positions* into ``df``.
        auto_pairs (str, optional): ``"cluster"``, ``"neighbors"``,
            ``"categorical"`` or ``"random"``.
        contrastive_loss (str): ``"triplet"`` or ``"contrastive"``.
        margin (float): Contrastive loss margin.
        negative_samples (int): Negatives drawn per positive pair.
        aggregate_by_reference (bool): ``mode="target"`` only. When ``True``,
            return one row per distinct reference value rather than one row per
            input row. Defaults to ``False`` so target mode, like every other
            mode, returns a frame indexed by ``df.index``.
        config (EmbeddingConfig, optional): Supplies the preprocessing settings
            (categorical encoding thresholds, imputation, numeric scaling).

    Returns:
        pd.DataFrame: Embeddings in columns ``embedding_0`` through
        ``embedding_{n-1}``, indexed by ``df.index`` so the result joins
        straight back onto ``df``. With ``aggregate_by_reference=True`` the
        index is the distinct reference values instead.

    Raises:
        ValueError: If the inputs are invalid, or if ``mode="target"`` and the
            reference column contains missing values.

    Examples:
        >>> import row2vec
        >>> df = row2vec.generate_synthetic_data(60)
        >>> embeddings = row2vec.learn_embedding(df, mode="pca", embedding_dim=2)
        >>> embeddings.shape
        (60, 2)

        The index is preserved, so the result concatenates cleanly:

        >>> list(embeddings.index) == list(df.index)
        True
    """
    resolved = _build_config(
        config,
        embedding_dim=embedding_dim,
        mode=mode,
        reference_column=reference_column,
        max_epochs=max_epochs,
        batch_size=batch_size,
        dropout_rate=dropout_rate,
        hidden_units=hidden_units,
        activation=activation,
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
        contrastive_loss=contrastive_loss,
        margin=margin,
        negative_samples=negative_samples,
    )
    return Row2VecModel(resolved, aggregate_by_reference=aggregate_by_reference).fit_transform(df)


def learn_embedding_with_model(
    df: pd.DataFrame,
    embedding_dim: int = 10,
    mode: str = "unsupervised",
    reference_column: str | None = None,
    max_epochs: int = 50,
    batch_size: int = 64,
    dropout_rate: float = 0.2,
    hidden_units: int | list[int] = 128,
    activation: str = "relu",
    early_stopping: bool = True,
    seed: int = 1305,
    verbose: bool = False,
    scale_method: str | None = None,
    scale_range: tuple[float, float] | None = None,
    log_level: str = "INFO",
    log_file: str | None = None,
    enable_logging: bool = True,
    n_neighbors: int = 15,
    perplexity: float = 30.0,
    min_dist: float = 0.1,
    n_iter: int = 1000,
    similar_pairs: list[tuple[int, int]] | None = None,
    dissimilar_pairs: list[tuple[int, int]] | None = None,
    auto_pairs: str | None = None,
    contrastive_loss: str = "triplet",
    margin: float = 1.0,
    negative_samples: int = 5,
    aggregate_by_reference: bool = False,
    config: EmbeddingConfig | None = None,
) -> tuple[pd.DataFrame, Row2VecModel]:
    """Learn embeddings and hand back the fitted model that produced them.

    Identical to :func:`learn_embedding` apart from the return value. A single
    training pass produces both; the previous implementation ran a second,
    different training and returned embeddings from one model and the other
    model itself, at roughly 1.8x the cost.

    Args:
        df (pd.DataFrame): Input data.
        embedding_dim (int): Width of the embedding space.
        mode (str): Embedding mode. See :func:`learn_embedding`.
        reference_column (str, optional): Label column for ``mode="target"``.
        max_epochs (int): Training epoch ceiling for neural modes.
        batch_size (int): Training batch size.
        dropout_rate (float): Dropout applied after each hidden layer.
        hidden_units (int | list[int]): Hidden layer width, or widths.
        activation (str): Hidden-layer activation.
        early_stopping (bool): Whether to stop once the loss stalls.
        seed (int): Random seed.
        verbose (bool): Whether to print training progress.
        scale_method (str, optional): Embedding scaling method.
        scale_range (tuple[float, float], optional): Range for ``"minmax"``.
        log_level (str): Logging level.
        log_file (str, optional): Path to write logs to.
        enable_logging (bool): Whether to log at all.
        n_neighbors (int): UMAP neighbourhood size.
        perplexity (float): t-SNE perplexity.
        min_dist (float): UMAP minimum distance.
        n_iter (int): t-SNE iteration count.
        similar_pairs (list[tuple[int, int]], optional): Explicit positive pairs.
        dissimilar_pairs (list[tuple[int, int]], optional): Explicit negatives.
        auto_pairs (str, optional): Automatic pairing strategy.
        contrastive_loss (str): ``"triplet"`` or ``"contrastive"``.
        margin (float): Contrastive loss margin.
        negative_samples (int): Negatives drawn per positive pair.
        aggregate_by_reference (bool): Return one row per reference value.
        config (EmbeddingConfig, optional): Preprocessing settings.

    Returns:
        tuple[pd.DataFrame, Row2VecModel]: The training embeddings, and the
        fitted model. The model embeds new rows with ``.transform(df)`` for
        every mode except ``"tsne"``, which has no out-of-sample extension.

    Examples:
        >>> import row2vec
        >>> df = row2vec.generate_synthetic_data(60)
        >>> embeddings, model = row2vec.learn_embedding_with_model(
        ...     df, mode="pca", embedding_dim=2
        ... )
        >>> model.transform(df.head(2)).shape
        (2, 2)
    """
    resolved = _build_config(
        config,
        embedding_dim=embedding_dim,
        mode=mode,
        reference_column=reference_column,
        max_epochs=max_epochs,
        batch_size=batch_size,
        dropout_rate=dropout_rate,
        hidden_units=hidden_units,
        activation=activation,
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
        contrastive_loss=contrastive_loss,
        margin=margin,
        negative_samples=negative_samples,
    )
    model = Row2VecModel(resolved, aggregate_by_reference=aggregate_by_reference)
    embeddings = model.fit_transform(df)
    return embeddings, model
