"""The fitted model behind every embedding row2vec produces.

Before 0.3.0 no object held fitted state. ``learn_embedding`` built a
preprocessor, fitted it, projected, and dropped every fitted part on the floor;
``learn_embedding_with_model`` then trained a *second*, subtly different model
to hand back; and the scikit-learn adapter re-ran training inside
``transform``. Three separate bugs, one cause.

:class:`Row2VecModel` is that missing object. It owns the fitted preprocessor,
the fitted projector, the encoder view used to read embeddings out of a neural
model, and - the piece that was missing entirely - the fitted embedding scaler.
Everything else in the package is a thin wrapper over it.

Two invariants hold for every mode:

* ``transform`` never fits. It projects with what ``fit`` learned.
* The returned frame carries ``df``'s own index, one row per input row, so it
  joins straight back onto the source data.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, normalize
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.models import Model

from .config import EmbeddingConfig
from .contrastive import (
    build_contrastive_model,
    create_contrastive_dataset,
    generate_contrastive_pairs,
)
from .logging import Row2VecLogger, get_logger
from .pipeline_builder import build_adaptive_pipeline
from .utils import create_dataframe_schema
from .validation import validate_inputs

if TYPE_CHECKING:
    from collections.abc import Hashable

    from sklearn.compose import ColumnTransformer

try:  # pragma: no cover - exercised implicitly wherever Keras is present
    from tensorflow.keras.callbacks import Callback, EarlyStopping
except ImportError:  # pragma: no cover
    Callback = object
    EarlyStopping = object

__all__ = [
    "MODES",
    "ModeSpec",
    "Row2VecModel",
    "Row2VecTrainingCallback",
    "get_feature_names",
]


class Row2VecTrainingCallback(Callback):
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
        additional_metrics = {k: v for k, v in logs.items() if k not in ["loss", "val_loss"]}

        self.logger.log_epoch_metrics(
            epoch=epoch,
            loss=loss,
            val_loss=val_loss,
            additional_metrics=additional_metrics if additional_metrics else None,
        )


# --------------------------------------------------------------------------- #
# Mode registry
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ModeSpec:
    """What the library needs to know about one embedding mode.

    Attributes:
        family (str):
            ``"classical"`` for a scikit-learn style projector, ``"neural"``
            for a Keras model.
        supports_transform (bool):
            Whether the mode can embed rows it was not fitted on. t-SNE cannot:
            it is non-parametric and has no out-of-sample extension.
        requires_reference (bool):
            Whether ``reference_column`` must be supplied.
        no_transform_reason (str):
            Explanation shown when ``transform`` is called on a mode that
            cannot support it.
    """

    family: str
    supports_transform: bool
    requires_reference: bool = False
    no_transform_reason: str = ""


MODES: dict[str, ModeSpec] = {
    "pca": ModeSpec(family="classical", supports_transform=True),
    "umap": ModeSpec(family="classical", supports_transform=True),
    "tsne": ModeSpec(
        family="classical",
        supports_transform=False,
        no_transform_reason=(
            "t-SNE is non-parametric: it has no out-of-sample extension, so a "
            "fitted t-SNE cannot embed rows it was not fitted on. Use "
            "fit_transform() to embed a dataset in one pass, or choose "
            "mode='pca' or mode='umap' if you need to transform new rows."
        ),
    ),
    "unsupervised": ModeSpec(family="neural", supports_transform=True),
    "target": ModeSpec(family="neural", supports_transform=True, requires_reference=True),
    "contrastive": ModeSpec(family="neural", supports_transform=True),
}


# --------------------------------------------------------------------------- #
# Embedding scaling
# --------------------------------------------------------------------------- #


class EmbeddingScaler:
    """Scales embeddings, remembering how so inference matches training.

    ``minmax`` and ``standard`` are *fitted* transformations: applying them
    afresh to an inference batch rescales it against that batch's own extrema,
    which is why a reloaded model used to disagree with the embeddings training
    had returned. ``l2`` and ``tanh`` are pointwise and need no state.

    Args:
        method (str | None):
            One of ``"none"``, ``"minmax"``, ``"standard"``, ``"l2"``,
            ``"tanh"``. ``None`` is treated as ``"none"``.
        value_range (tuple[float, float] | None):
            Output range for ``"minmax"``; defaults to ``(0.0, 1.0)``.

    Examples:
        >>> import pandas as pd
        >>> scaler = EmbeddingScaler("minmax")
        >>> train = pd.DataFrame({"embedding_0": [0.0, 10.0]})
        >>> scaler.fit_transform(train)["embedding_0"].tolist()
        [0.0, 1.0]

        A later batch is scaled against the *training* extrema, not its own:

        >>> scaler.transform(pd.DataFrame({"embedding_0": [5.0]}))["embedding_0"].tolist()
        [0.5]
    """

    _VALID = ("none", "minmax", "standard", "l2", "tanh")

    def __init__(self, method: str | None, value_range: tuple[float, float] | None = None) -> None:
        resolved = "none" if method is None else method
        if resolved not in self._VALID:
            raise ValueError(
                f"Invalid scale_method '{resolved}'. Must be one of {list(self._VALID)}. "
                "See documentation for details on each scaling method.",
            )
        self.method = resolved
        self.value_range = value_range
        self.scaler_: Any = None

    @property
    def is_stateful(self) -> bool:
        """Whether this method needs statistics learned at fit time."""
        return self.method in ("minmax", "standard")

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Learn any needed statistics and scale ``df``."""
        if self.method == "none":
            return df

        values = df.to_numpy(dtype=float, copy=False)

        if self.method == "minmax":
            rng = self.value_range if self.value_range is not None else (0.0, 1.0)
            self.scaler_ = MinMaxScaler(feature_range=(float(rng[0]), float(rng[1])))
            self._raise_on_constant_column(df, values)
            scaled = self.scaler_.fit_transform(values)
        elif self.method == "standard":
            self.scaler_ = StandardScaler()
            scaled = self.scaler_.fit_transform(values)
        else:
            return self.transform(df)

        return pd.DataFrame(scaled, columns=df.columns, index=df.index)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Scale ``df`` with whatever ``fit_transform`` learned."""
        if self.method == "none":
            return df

        values = df.to_numpy(dtype=float, copy=False)

        if self.is_stateful:
            if self.scaler_ is None:
                raise RuntimeError(
                    f"EmbeddingScaler(method={self.method!r}) was never fitted; "
                    "call fit_transform() before transform().",
                )
            scaled = self.scaler_.transform(values)
        elif self.method == "l2":
            scaled = normalize(values, norm="l2", axis=1)
        else:  # tanh
            scaled = np.tanh(values)

        return pd.DataFrame(scaled, columns=df.columns, index=df.index)

    @staticmethod
    def _raise_on_constant_column(df: pd.DataFrame, values: npt.NDArray[Any]) -> None:
        """Give a usable error for the degenerate min-max case."""
        for j in range(values.shape[1]):
            column = values[:, j]
            if np.max(column) == np.min(column):
                name = df.columns[j] if j < len(df.columns) else f"column_{j}"
                raise ValueError(
                    f"MinMax scaling undefined for constant column '{name}' "
                    f"(all values are {np.min(column)}). Consider removing this "
                    "column or using a different scaling method.",
                )


# --------------------------------------------------------------------------- #
# The fitted model
# --------------------------------------------------------------------------- #


class Row2VecModel:
    """A fitted row2vec embedder.

    Holds every part needed to embed new rows the same way the training rows
    were embedded: the fitted preprocessing pipeline, the fitted projector, the
    encoder view for neural modes, and the fitted embedding scaler.

    Args:
        config (EmbeddingConfig, optional):
            Configuration to fit with. Defaults to :class:`EmbeddingConfig`.
        aggregate_by_reference (bool):
            ``mode="target"`` only. When ``True``, ``fit_transform`` returns one
            row per distinct reference value instead of one row per input row.
            Off by default, so target mode carries ``df``'s index like every
            other mode.

    Attributes:
        preprocessor_ (ColumnTransformer):
            The fitted preprocessing pipeline.
        projector_ (object):
            The fitted PCA/UMAP/t-SNE estimator, or the trained Keras model.
        encoder_ (object or None):
            For neural modes, the sub-model mapping processed features to the
            embedding. ``None`` for classical modes.
        embedding_scaler_ (EmbeddingScaler):
            The fitted embedding scaler.
        feature_names_in_ (list[Hashable]):
            Column labels seen during ``fit``.
        metadata (object or None):
            Populated by :mod:`row2vec.serialization` on save/load.

    Examples:
        >>> import row2vec
        >>> df = row2vec.generate_synthetic_data(60)
        >>> model = Row2VecModel(row2vec.EmbeddingConfig(embedding_dim=2, mode="pca"))
        >>> train = model.fit_transform(df)
        >>> list(train.index) == list(df.index)
        True

        ``transform`` reuses the fitted basis rather than refitting:

        >>> model.transform(df.head(1)).shape
        (1, 2)
    """

    def __init__(
        self,
        config: EmbeddingConfig | None = None,
        *,
        aggregate_by_reference: bool = False,
    ) -> None:
        self.config = config if config is not None else EmbeddingConfig()
        self.aggregate_by_reference = aggregate_by_reference

        # Fitted state
        self.preprocessor_: ColumnTransformer | None = None
        self.projector_: Any = None
        self.encoder_: Any = None
        self.embedding_scaler_: EmbeddingScaler | None = None
        self.feature_names_in_: list[Hashable] = []
        self.n_features_in_: int = 0

        # Training record
        self.training_columns_: list[str] = []
        self.training_shape_: tuple[int, int] | None = None
        self.training_dtypes_: dict[str, str] = {}
        self.training_schema_: dict[str, Any] = {}
        self.training_history_: dict[str, Any] = {}
        self.final_loss_: float | None = None
        self.epochs_trained_: int | None = None
        self.training_time_: float | None = None

        # Shape of the matrix the projector actually saw, for logging.
        self._processed_shape: tuple[int, int] | None = None

        # t-SNE has no transform, so its training embedding is kept.
        self._fitted_embedding: pd.DataFrame | None = None

        # Filled in by row2vec.serialization; not needed to embed.
        self.metadata: Any = None

    # -- properties -------------------------------------------------------- #

    @property
    def mode(self) -> str:
        """The embedding mode this model was configured with."""
        return self.config.mode

    @property
    def spec(self) -> ModeSpec:
        """The :class:`ModeSpec` for this model's mode."""
        return MODES[self.config.mode]

    @property
    def is_fitted(self) -> bool:
        """Whether :meth:`fit` has run."""
        return self.preprocessor_ is not None

    # -- public API -------------------------------------------------------- #

    def fit(self, df: pd.DataFrame) -> Row2VecModel:
        """Fit the preprocessor, projector, and embedding scaler on ``df``.

        Args:
            df (pd.DataFrame): Training data.

        Returns:
            Row2VecModel: ``self``, so calls can be chained.
        """
        self._fit_impl(df)
        return self

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit on ``df`` and return its embeddings in one training pass.

        Args:
            df (pd.DataFrame): Training data.

        Returns:
            pd.DataFrame: Embeddings indexed by ``df.index``, unless
            ``aggregate_by_reference`` is set for ``mode="target"``.
        """
        return self._fit_impl(df)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Embed rows using the state learned during :meth:`fit`.

        Args:
            df (pd.DataFrame): Rows to embed. May be a single row.

        Returns:
            pd.DataFrame: Embeddings indexed by ``df.index``.

        Raises:
            RuntimeError: If the model has not been fitted.
            NotImplementedError: If the mode has no out-of-sample extension.
        """
        if not self.is_fitted:
            raise RuntimeError(
                "This Row2VecModel is not fitted yet. Call fit() or "
                "fit_transform() before transform().",
            )
        if not self.spec.supports_transform:
            raise NotImplementedError(
                f"mode={self.config.mode!r} does not support transform(). "
                f"{self.spec.no_transform_reason}",
            )

        raw = self._project(self._preprocess(df), index=df.index)
        assert self.embedding_scaler_ is not None
        return self.embedding_scaler_.transform(raw)

    def predict(self, df: pd.DataFrame, validate_schema: bool = True) -> pd.DataFrame:
        """Embed ``df`` with a saved model.

        Args:
            df (pd.DataFrame): Rows to embed.
            validate_schema (bool): Whether to check ``df`` against the schema
                recorded at training time. Ignored when the model carries no
                metadata, which is the case before it has been saved.

        Returns:
            pd.DataFrame: Embeddings indexed by ``df.index``, identical to what
            training returned for the same rows.
        """
        if validate_schema and self.metadata is not None:
            self.validate_input_schema(df, strict=True)
        return self.transform(df)

    def validate_input_schema(self, df: pd.DataFrame, strict: bool = True) -> bool:
        """Check ``df`` against the schema recorded when the model was saved.

        Args:
            df (pd.DataFrame): Input to validate.
            strict (bool): Raise on mismatch rather than returning ``False``.

        Returns:
            bool: Whether the schema matches.

        Raises:
            ValueError: If ``strict`` and the schema does not match, or no
                schema was recorded.
        """
        from .utils import validate_dataframe_schema

        expected = getattr(self.metadata, "expected_schema", None)
        if not expected:
            if strict:
                raise ValueError("No expected schema defined in model metadata")
            return False

        try:
            validate_dataframe_schema(df, expected)
            return True
        except Exception as exc:
            if strict:
                raise ValueError(f"Schema validation failed: {exc!s}") from exc
            return False

    # -- persistence -------------------------------------------------------- #

    #: Attributes that together constitute a saved model.
    STATE_KEYS = (
        "config",
        "aggregate_by_reference",
        "preprocessor_",
        "projector_",
        "encoder_",
        "embedding_scaler_",
        "feature_names_in_",
        "n_features_in_",
        "training_columns_",
        "training_shape_",
        "training_dtypes_",
        "training_schema_",
        "training_history_",
        "final_loss_",
        "epochs_trained_",
        "training_time_",
    )

    def to_state(self) -> dict[str, Any]:
        """Everything needed to rebuild this model, for pickling.

        Returns:
            dict[str, Any]: The fitted state, keyed by attribute name.
        """
        if not self.is_fitted:
            raise ValueError("Cannot serialize an unfitted Row2VecModel.")
        return {key: getattr(self, key) for key in self.STATE_KEYS}

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> Row2VecModel:
        """Rebuild a fitted model from :meth:`to_state` output.

        Args:
            state (dict[str, Any]): A previously saved state dictionary.

        Returns:
            Row2VecModel: The restored model, ready to ``transform``.
        """
        model = cls(
            state["config"],
            aggregate_by_reference=state.get("aggregate_by_reference", False),
        )
        for key in cls.STATE_KEYS:
            if key in state and key not in ("config", "aggregate_by_reference"):
                setattr(model, key, state[key])
        return model

    # -- fitting ----------------------------------------------------------- #

    def _fit_impl(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run the single training pass and return the training embeddings."""
        config = self.config
        logger = self._make_logger()
        started = time.time()

        validate_inputs(
            df=df,
            embedding_dim=config.embedding_dim,
            mode=config.mode,
            reference_column=config.reference_column,
            max_epochs=config.neural.max_epochs,
            batch_size=config.neural.batch_size,
            dropout_rate=config.neural.dropout_rate,
            hidden_units=config.neural.hidden_units,
            scale_method=config.scaling.method,
            scale_range=config.scaling.range,
            n_neighbors=config.classical.n_neighbors,
            perplexity=config.classical.perplexity,
            min_dist=config.classical.min_dist,
            n_iter=config.classical.n_iter,
            similar_pairs=config.contrastive.similar_pairs,
            dissimilar_pairs=config.contrastive.dissimilar_pairs,
            auto_pairs=config.contrastive.auto_pairs,
            contrastive_loss=config.contrastive.loss_type,
            margin=config.contrastive.margin,
            negative_samples=config.contrastive.negative_samples,
        )

        self._seed_everything(config.seed)

        if logger is not None:
            logger.start_training(
                mode=config.mode,
                embedding_dim=config.embedding_dim,
                seed=config.seed,
                max_epochs=config.neural.max_epochs,
                batch_size=config.neural.batch_size,
                hidden_units=config.neural.hidden_units,
                activation=config.neural.activation,
            )
            logger.log_data_preprocessing(
                df.shape,
                [
                    f"Mode: {config.mode}",
                    f"Categorical encoding strategy: "
                    f"{config.preprocessing.categorical_encoding_strategy}",
                    f"Numeric scaling: {config.preprocessing.numeric_scaling}",
                    f"Missing value handling: {config.preprocessing.handle_missing}",
                    f"Embedding scaling: {config.scaling.method or 'none'}",
                ],
            )

        # The labels themselves: a DataFrame may have integer column names,
        # and comparing str() of them against the real labels never matches.
        self.feature_names_in_ = list(df.columns)
        self.training_columns_ = list(df.columns)
        self.training_shape_ = df.shape
        self.training_dtypes_ = {str(c): str(dt) for c, dt in df.dtypes.items()}
        self.training_schema_ = create_dataframe_schema(df)
        self.embedding_scaler_ = EmbeddingScaler(config.scaling.method, config.scaling.range)

        target_series = self._target_series(df)

        # Build and fit preprocessing. For neural modes the fit happens on the
        # training split only, so the validation loss is not inflated by
        # statistics drawn from the rows it is scored on.
        feature_df = self._feature_frame(df)
        self.preprocessor_, _report = build_adaptive_pipeline(
            df=feature_df,
            target=target_series,
            config=config,
            mode=config.mode,
        )

        preprocessing_started = time.time()
        if self.spec.family == "classical":
            processed = self._fit_preprocessor(feature_df, target_series)
            self._log_preprocessing(logger, df.shape, time.time() - preprocessing_started)
            raw = self._fit_classical(processed, df.index, logger)
        else:
            raw = self._fit_neural(df, feature_df, target_series, logger)
            self._log_preprocessing(logger, df.shape, time.time() - preprocessing_started)

        self.n_features_in_ = len(self.feature_names_in_)
        self.training_time_ = time.time() - started

        scaled = self.embedding_scaler_.fit_transform(raw)
        self._fitted_embedding = scaled

        if logger is not None:
            logger.log_embedding_stats(scaled)
            logger.log_completion()

        if self.config.mode == "target" and self.aggregate_by_reference:
            return self._aggregate(scaled, df)
        return scaled

    def _log_preprocessing(
        self,
        logger: Row2VecLogger | None,
        original_shape: tuple[int, int],
        elapsed: float,
    ) -> None:
        """Report what preprocessing produced, and warn about costly shapes."""
        if logger is None or self._processed_shape is None:
            return

        logger.log_preprocessing_result(original_shape, self._processed_shape, elapsed)

        n_features = self._processed_shape[1]
        if n_features > 1000:
            logger.log_performance_warning(
                f"High-dimensional input ({n_features} features) may slow training. "
                "Consider feature selection or dimensionality reduction.",
            )
        if original_shape[0] > 100000:
            logger.log_performance_warning(
                f"Large dataset ({original_shape[0]} rows) detected. Consider a "
                "larger batch_size or distributed training for better performance.",
            )

    def _fit_preprocessor(
        self,
        feature_df: pd.DataFrame,
        target: pd.Series | None,
    ) -> npt.NDArray[Any]:
        """Fit the preprocessing pipeline and return the dense matrix."""
        assert self.preprocessor_ is not None
        # Passing the target through is what lets supervised encoders (target
        # encoding above all) actually see a target; without it every column's
        # target correlation was computed as zero.
        processed = self._densify(self.preprocessor_.fit_transform(feature_df, target))
        self._processed_shape = (processed.shape[0], processed.shape[1])
        return processed

    def _fit_classical(
        self,
        X: npt.NDArray[Any],
        index: pd.Index,
        logger: Row2VecLogger | None,
    ) -> pd.DataFrame:
        """Fit a PCA/UMAP/t-SNE projector on the whole training matrix."""
        config = self.config
        dim = config.embedding_dim
        self._check_dim_against_features(X.shape[1], logger)

        if config.mode == "pca":
            from sklearn.decomposition import PCA

            self.projector_ = PCA(n_components=dim, random_state=config.seed)
            values = self.projector_.fit_transform(X)
            if logger is not None:
                ratio = self.projector_.explained_variance_ratio_
                logger.log_debug_info(f"PCA explained variance ratio: {ratio.sum():.3f}")
                logger.log_debug_info(f"Top 3 components: {ratio[:3]}")

        elif config.mode == "tsne":
            from sklearn.manifold import TSNE

            if dim > 3 and logger is not None:
                logger.log_performance_warning(
                    f"t-SNE with embedding_dim={dim} may not be optimal. "
                    "Consider using 2 or 3 dimensions for t-SNE visualization.",
                )
            self.projector_ = TSNE(
                n_components=dim,
                perplexity=config.classical.perplexity,
                max_iter=config.classical.n_iter,
                method="exact" if dim > 3 else "barnes_hut",
                random_state=config.seed,
                verbose=1 if config.verbose else 0,
            )
            values = self.projector_.fit_transform(X)
            if logger is not None:
                logger.log_debug_info(
                    f"t-SNE KL divergence: {self.projector_.kl_divergence_:.3f}",
                )

        else:  # umap
            try:
                import umap
            except ImportError:
                raise ImportError(
                    "UMAP is not installed. Please install it with: pip install umap-learn",
                ) from None

            self.projector_ = umap.UMAP(
                n_components=dim,
                n_neighbors=config.classical.n_neighbors,
                min_dist=config.classical.min_dist,
                random_state=config.seed,
                verbose=config.verbose,
            )
            values = self.projector_.fit_transform(X)

        return self._frame(values, index)

    def _fit_neural(
        self,
        df: pd.DataFrame,
        feature_df: pd.DataFrame,
        target: pd.Series | None,
        logger: Row2VecLogger | None,
    ) -> pd.DataFrame:
        """Train a Keras model, fitting preprocessing on the training split only."""
        config = self.config

        if config.mode == "contrastive":
            # Contrastive pairs are drawn over the whole frame, so there is no
            # held-out split to keep clean here.
            X = self._fit_preprocessor(feature_df, target)
            self._check_dim_against_features(X.shape[1], logger)
            self._train_contrastive(df, X, logger)
        else:
            y = None if target is None else target.astype("category").cat.codes
            train_idx, val_idx = train_test_split(
                np.arange(len(feature_df)),
                test_size=0.2,
                random_state=config.seed,
            )

            # Fit preprocessing on the training rows, then apply it to both.
            assert self.preprocessor_ is not None
            train_features = feature_df.iloc[train_idx]
            train_target = None if target is None else target.iloc[train_idx]
            self.preprocessor_.fit(train_features, train_target)

            X_train = self._densify(self.preprocessor_.transform(train_features))
            X_val = self._densify(self.preprocessor_.transform(feature_df.iloc[val_idx]))
            X = self._densify(self.preprocessor_.transform(feature_df))
            self._processed_shape = (X.shape[0], X.shape[1])
            self._check_dim_against_features(X.shape[1], logger)

            y_train = None if y is None else y.to_numpy()[train_idx]
            y_val = None if y is None else y.to_numpy()[val_idx]
            n_classes = 0 if y is None else len(np.unique(y))

            self._train_autoencoder(X_train, X_val, y_train, y_val, n_classes, logger)

        values = self.encoder_.predict(X, verbose=0)
        return self._frame(values, df.index)

    def _train_autoencoder(
        self,
        X_train: npt.NDArray[Any],
        X_val: npt.NDArray[Any],
        y_train: npt.NDArray[Any] | None,
        y_val: npt.NDArray[Any] | None,
        n_classes: int,
        logger: Row2VecLogger | None,
    ) -> None:
        """Build and train the autoencoder (or supervised encoder) model."""
        config = self.config
        neural = config.neural
        activation = neural.activation
        hidden_units = neural.hidden_units
        supervised = config.mode == "target"

        input_layer = Input(shape=(X_train.shape[1],))
        x = input_layer
        units_list = [hidden_units] if isinstance(hidden_units, int) else list(hidden_units)
        for i, units in enumerate(units_list):
            x = Dense(units, activation=activation, name=f"encoder_hidden_{i + 1}")(x)
            x = Dropout(neural.dropout_rate)(x)

        encoded = Dense(config.embedding_dim, activation="linear", name="embedding")(x)

        if supervised:
            output = Dense(n_classes, activation="softmax")(encoded)
            model = Model(inputs=input_layer, outputs=output)
            model.compile(
                optimizer="adam",
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"],
            )
        else:
            decoded = encoded
            for i, units in enumerate(reversed(units_list)):
                decoded = Dense(units, activation=activation, name=f"decoder_hidden_{i + 1}")(
                    decoded
                )
                decoded = Dropout(neural.dropout_rate)(decoded)
            decoded = Dense(X_train.shape[1], activation="linear")(decoded)
            model = Model(inputs=input_layer, outputs=decoded)
            model.compile(optimizer="adam", loss="mse")

        callbacks: list[Any] = []
        if neural.early_stopping:
            callbacks.append(
                EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
            )
        if logger is not None:
            callbacks.append(Row2VecTrainingCallback(logger))

        history = model.fit(
            X_train,
            y_train if supervised else X_train,
            validation_data=(X_val, y_val if supervised else X_val),
            epochs=neural.max_epochs,
            batch_size=neural.batch_size,
            verbose=int(config.verbose),
            callbacks=callbacks,
        )

        self.projector_ = model
        self.encoder_ = Model(
            inputs=model.input,
            outputs=model.get_layer("embedding").output,
        )
        self._record_history(history, logger)

    def _train_contrastive(
        self,
        df: pd.DataFrame,
        X: npt.NDArray[Any],
        logger: Row2VecLogger | None,
    ) -> None:
        """Build and train the siamese/triplet model."""
        config = self.config
        contrastive = config.contrastive

        similar, dissimilar = generate_contrastive_pairs(
            df,
            contrastive.similar_pairs,
            contrastive.dissimilar_pairs,
            contrastive.auto_pairs,
            config.reference_column,
            X,
            contrastive.negative_samples,
            config.seed,
            logger,
        )

        model = build_contrastive_model(
            X.shape[1],
            config.embedding_dim,
            config.neural.hidden_units,
            config.neural.dropout_rate,
            contrastive.loss_type,
            contrastive.margin,
            config.seed,
        )

        dataset = create_contrastive_dataset(
            X,
            similar,
            dissimilar,
            config.neural.batch_size,
            contrastive.loss_type,
            config.seed,
        )

        # The dataset is an endless generator, so an epoch only terminates if
        # it is told how many batches long it is.
        total_pairs = len(similar) + len(dissimilar)
        steps_per_epoch = max(1, total_pairs // config.neural.batch_size)

        callbacks: list[Any] = []
        if config.neural.early_stopping:
            # There is no validation split here, so early stopping watches the
            # training loss.
            callbacks.append(
                EarlyStopping(monitor="loss", patience=5, restore_best_weights=True),
            )
        if logger is not None:
            callbacks.append(Row2VecTrainingCallback(logger))

        history = model.fit(
            dataset,
            steps_per_epoch=steps_per_epoch,
            epochs=config.neural.max_epochs,
            verbose=int(config.verbose),
            callbacks=callbacks,
        )

        self.projector_ = model
        # The shared encoder is the only part that can embed a lone row; the
        # wrapper takes two or three inputs and outputs a concatenation.
        self.encoder_ = model.encoder
        self._record_history(history, logger)

    # -- transform helpers -------------------------------------------------- #

    def _preprocess(self, df: pd.DataFrame) -> npt.NDArray[Any]:
        """Apply the fitted preprocessing pipeline to new rows."""
        assert self.preprocessor_ is not None
        missing = [c for c in self.feature_names_in_ if c not in df.columns]
        if missing and not (
            self.config.mode == "target" and missing == [self.config.reference_column]
        ):
            raise ValueError(
                f"Columns seen during fit are missing from the input: {missing}. "
                f"Expected {self.feature_names_in_}.",
            )
        return self._densify(self.preprocessor_.transform(self._feature_frame(df)))

    def _project(self, X: npt.NDArray[Any], index: pd.Index) -> pd.DataFrame:
        """Map processed features to raw (unscaled) embeddings."""
        if self.spec.family == "neural":
            values = self.encoder_.predict(X, verbose=0)
        else:
            values = self.projector_.transform(X)
        return self._frame(values, index)

    # -- small helpers ------------------------------------------------------ #

    def _feature_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        """The columns actually fed to the preprocessor."""
        if self.config.mode == "target" and self.config.reference_column in df.columns:
            return df.drop(columns=[self.config.reference_column])
        return df

    def _target_series(self, df: pd.DataFrame) -> pd.Series | None:
        """The reference column for target mode, or ``None``.

        Rows with a missing reference value are rejected rather than silently
        dropped: dropping them used to reset the index and change the row count
        under the caller's feet.
        """
        if self.config.mode != "target":
            return None

        column = self.config.reference_column
        assert column is not None
        series = df[column]
        if series.isnull().any():
            n_missing = int(series.isnull().sum())
            raise ValueError(
                f"reference_column '{column}' has {n_missing} missing value(s). "
                "Fill or drop those rows before calling; row2vec will not drop "
                "them for you, because that would change which rows the "
                "returned embeddings correspond to.",
            )
        return series

    def _aggregate(self, embeddings: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        """Average embeddings per distinct reference value."""
        column = self.config.reference_column
        assert column is not None
        grouped = embeddings.groupby(df[column].to_numpy()).mean()
        grouped.index.name = column
        return grouped

    def _frame(self, values: npt.NDArray[Any], index: pd.Index) -> pd.DataFrame:
        """Wrap an embedding matrix, preserving the caller's index."""
        return pd.DataFrame(
            np.asarray(values),
            columns=[f"embedding_{i}" for i in range(self.config.embedding_dim)],
            index=index,
        )

    @staticmethod
    def _densify(X: Any) -> npt.NDArray[Any]:
        """TensorFlow and several sklearn estimators need dense input."""
        if hasattr(X, "toarray"):
            X = X.toarray()
        return np.asarray(X)

    def _check_dim_against_features(self, n_features: int, logger: Row2VecLogger | None) -> None:
        """Reject an embedding wider than the preprocessed feature space."""
        dim = self.config.embedding_dim
        if dim > n_features:
            message = (
                f"embedding_dim ({dim}) cannot be larger than the number of features "
                f"after preprocessing (n_features={n_features}). Consider reducing "
                "embedding_dim or adding more features."
            )
            if logger is not None:
                logger.log_validation_issue(message)
            raise ValueError(message)

        if logger is not None and dim > n_features * 0.8:
            logger.log_performance_warning(
                f"Embedding dimension ({dim}) is close to input features ({n_features}). "
                "This may lead to overfitting.",
            )

    def _record_history(self, history: Any, logger: Row2VecLogger | None) -> None:
        """Keep the training record that used to be declared and never filled."""
        losses = history.history.get("loss", [])
        self.training_history_ = {k: list(v) for k, v in history.history.items()}
        self.final_loss_ = float(losses[-1]) if losses else None
        self.epochs_trained_ = len(losses)

        if logger is not None and self.final_loss_ is not None:
            logger.end_training(self.final_loss_, self.epochs_trained_ or 0)
            if self.epochs_trained_ < self.config.neural.max_epochs:
                logger.log_early_stopping(self.epochs_trained_ - 1, "EarlyStopping triggered")

    def _make_logger(self) -> Row2VecLogger | None:
        """Build the logger this run should use, if logging is enabled."""
        if not self.config.logging.enabled:
            return None
        return get_logger(
            level=self.config.logging.level,
            log_file=self.config.logging.file,
        )

    @staticmethod
    def _seed_everything(seed: int) -> None:
        """Seed every RNG the training path draws from."""
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)
        # Takes no arguments; the old call passed one and swallowed the
        # TypeError, so determinism was never actually enabled.
        tf.config.experimental.enable_op_determinism()


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Extract feature names from a fitted ColumnTransformer.

    Args:
        preprocessor: Fitted ColumnTransformer

    Returns:
        List of feature names
    """
    try:
        # Try to get feature names if available (sklearn 1.0+)
        if hasattr(preprocessor, "get_feature_names_out"):
            return list(preprocessor.get_feature_names_out())
    except Exception:
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
            except Exception:
                # Fallback for transformers without proper feature name support
                if hasattr(transformer, "named_steps") and "onehot" in transformer.named_steps:
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
