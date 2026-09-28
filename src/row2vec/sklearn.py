"""Scikit-learn adapters for Row2Vec embeddings.

These wrap :class:`row2vec.model.Row2VecModel` so row2vec can be dropped into a
``Pipeline`` or a ``GridSearchCV``. Both classes follow the estimator contract:
``fit`` learns and returns ``self``, ``transform`` only projects, and every
constructor argument is a plain attribute of the same name so ``get_params``,
``set_params`` and ``clone`` round-trip.

That was not true before 0.3.0. ``transform`` re-ran the whole training on
whatever frame it was handed, so inside cross-validation each fold learned its
embedding from its own test fold; ``**kwargs`` in ``__init__`` meant ``clone``
silently dropped every nested parameter, so a grid search over them tuned
nothing; and ``fit_transform`` trained the model three times over.
"""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, clone
from sklearn.utils.validation import check_is_fitted

from .config import EmbeddingConfig, create_config_for_mode
from .model import MODES, Row2VecModel

__all__ = ["Row2VecClassifier", "Row2VecTransformer"]


def _as_dataframe(X: Any, feature_names: np.ndarray[Any, Any] | None) -> pd.DataFrame:
    """Coerce sklearn's assorted input types to a DataFrame.

    Args:
        X: Input data: DataFrame, ndarray, or anything array-like.
        feature_names: Column names seen during ``fit``, if fitted.

    Returns:
        pd.DataFrame: ``X`` as a DataFrame with usable column names.

    Raises:
        ValueError: If the column count disagrees with what ``fit`` saw.
        TypeError: If ``X`` cannot be coerced at all.
    """
    if isinstance(X, pd.DataFrame):
        return X.copy()

    if not isinstance(X, np.ndarray):
        try:
            X = np.asarray(X)
        except Exception as exc:
            raise TypeError(f"Cannot convert input to DataFrame: {exc}") from exc

    if X.ndim == 1:
        X = X.reshape(-1, 1)

    if feature_names is not None:
        if X.shape[1] != len(feature_names):
            raise ValueError(
                f"X has {X.shape[1]} features, but transformer was fitted with "
                f"{len(feature_names)} features",
            )
        return pd.DataFrame(X, columns=feature_names)

    return pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])


class Row2VecTransformer(TransformerMixin, BaseEstimator):
    """Scikit-learn transformer producing Row2Vec embeddings.

    ``TransformerMixin`` comes first in the bases deliberately: scikit-learn
    reads the transformer tags off the MRO, and with ``BaseEstimator`` first
    ``check_estimator`` refuses to run at all.

    Args:
        embedding_dim (int): Width of the embedding space.
        mode (str): ``"unsupervised"``, ``"target"``, ``"pca"``, ``"umap"`` or
            ``"contrastive"``. ``"tsne"`` is rejected at ``fit`` time because it
            cannot transform unseen rows, and a transformer that cannot
            transform has no place in a pipeline.
        reference_column (str, optional): Label column for ``mode="target"``.
        max_epochs (int): Training epoch ceiling for neural modes.
        batch_size (int): Training batch size.
        dropout_rate (float): Dropout after each hidden layer.
        hidden_units (int | list[int]): Hidden layer width, or widths.
        activation (str): Hidden-layer activation.
        seed (int): Random seed.
        config (EmbeddingConfig, optional): A complete configuration. When
            given, the other arguments are ignored.

    Attributes:
        model_ (Row2VecModel): The fitted model. ``transform`` projects with it
            rather than refitting.
        config_ (EmbeddingConfig): The configuration actually used.
        n_features_in_ (int): Number of columns seen during ``fit``.
        feature_names_in_ (ndarray): Column names seen during ``fit``.

    Examples:
        >>> import row2vec
        >>> from row2vec.sklearn import Row2VecTransformer
        >>> df = row2vec.generate_synthetic_data(60)
        >>> transformer = Row2VecTransformer(embedding_dim=2, mode="pca")
        >>> transformer.fit_transform(df).shape
        (60, 2)

        A fitted transformer embeds a single unseen row:

        >>> transformer.transform(df.head(1)).shape
        (1, 2)
    """

    def __init__(
        self,
        embedding_dim: int = 10,
        mode: str = "unsupervised",
        reference_column: str | None = None,
        max_epochs: int = 50,
        batch_size: int = 64,
        dropout_rate: float = 0.2,
        hidden_units: int | list[int] = 128,
        activation: str = "relu",
        seed: int = 1305,
        config: EmbeddingConfig | None = None,
    ) -> None:
        # Every argument is stored unchanged under its own name: sklearn's
        # get_params/set_params/clone contract depends on exactly that.
        self.embedding_dim = embedding_dim
        self.mode = mode
        self.reference_column = reference_column
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.dropout_rate = dropout_rate
        self.hidden_units = hidden_units
        self.activation = activation
        self.seed = seed
        self.config = config

    def _build_config(self) -> EmbeddingConfig:
        """Assemble the configuration this transformer should fit with."""
        if self.config is not None:
            return self.config

        config = create_config_for_mode(self.mode)
        config.embedding_dim = self.embedding_dim
        config.seed = self.seed
        config.neural.max_epochs = self.max_epochs
        config.neural.batch_size = self.batch_size
        config.neural.dropout_rate = self.dropout_rate
        config.neural.hidden_units = self.hidden_units
        config.neural.activation = self.activation

        if self.reference_column is not None:
            config.reference_column = self.reference_column

        return config

    def fit(self, X: Any, y: Any = None) -> "Row2VecTransformer":
        """Fit the embedding model.

        Args:
            X: Training data.
            y: Ignored; present for the sklearn signature.

        Returns:
            Row2VecTransformer: ``self``.

        Raises:
            ValueError: If ``mode`` cannot support ``transform``.
        """
        frame = _as_dataframe(X, None)

        mode = self.config.mode if self.config is not None else self.mode
        if mode in MODES and not MODES[mode].supports_transform:
            raise ValueError(
                f"Row2VecTransformer cannot use mode={mode!r}: "
                f"{MODES[mode].no_transform_reason} A transformer must be able "
                "to transform, so use row2vec.learn_embedding() directly for "
                "a one-off embedding.",
            )

        self.n_features_in_ = frame.shape[1]
        self.feature_names_in_ = np.array(frame.columns)
        self.config_ = self._build_config()
        self.model_ = Row2VecModel(self.config_).fit(frame)
        return self

    def transform(self, X: Any) -> np.ndarray[Any, Any]:
        """Project data into the fitted embedding space.

        Args:
            X: Data to transform. A single row is fine.

        Returns:
            ndarray of shape (n_samples, embedding_dim).
        """
        check_is_fitted(self, ["model_"])
        frame = _as_dataframe(X, self.feature_names_in_)
        return self.model_.transform(frame).to_numpy()

    def fit_transform(self, X: Any, y: Any = None, **fit_params: Any) -> np.ndarray[Any, Any]:
        """Fit and transform in a single training pass.

        Args:
            X: Training data.
            y: Ignored; present for the sklearn signature.
            **fit_params: Ignored; present for the sklearn signature.

        Returns:
            ndarray of shape (n_samples, embedding_dim).
        """
        self.fit(X, y)
        frame = _as_dataframe(X, self.feature_names_in_)
        return self.model_.transform(frame).to_numpy()

    def get_feature_names_out(
        self,
        input_features: np.ndarray[Any, Any] | None = None,
    ) -> np.ndarray[Any, Any]:
        """Names of the embedding columns.

        Args:
            input_features: Ignored; present for the sklearn signature.

        Returns:
            ndarray of str: ``row2vec_0`` through ``row2vec_{n-1}``.
        """
        check_is_fitted(self, ["config_"])
        return np.array([f"row2vec_{i}" for i in range(self.config_.embedding_dim)])


class Row2VecClassifier(ClassifierMixin, BaseEstimator):
    """Classify rows using Row2Vec embeddings as features.

    Args:
        embedding_dim (int): Width of the embedding space.
        mode (str): Embedding mode. See :class:`Row2VecTransformer`.
        reference_column (str, optional): Label column for ``mode="target"``.
        classifier (sklearn estimator, optional): Downstream classifier.
            Cloned before fitting, so the caller's instance is left alone.
            Defaults to ``LogisticRegression``.
        seed (int): Random seed.
        embedding_config (EmbeddingConfig, optional): Complete embedding
            configuration; overrides the individual arguments.

    Attributes:
        transformer_ (Row2VecTransformer): The fitted embedder.
        classifier_ (sklearn estimator): The fitted downstream classifier.
        classes_ (ndarray): Class labels seen during ``fit``.

    Examples:
        >>> import row2vec
        >>> from row2vec.sklearn import Row2VecClassifier
        >>> df = row2vec.generate_synthetic_data(80)
        >>> y = df["Country"]
        >>> clf = Row2VecClassifier(embedding_dim=2, mode="pca")
        >>> _ = clf.fit(df.drop(columns=["Country"]), y)
        >>> len(clf.predict(df.drop(columns=["Country"])))
        80
    """

    def __init__(
        self,
        embedding_dim: int = 10,
        mode: str = "pca",
        reference_column: str | None = None,
        classifier: Any = None,
        seed: int = 1305,
        embedding_config: EmbeddingConfig | None = None,
    ) -> None:
        self.embedding_dim = embedding_dim
        self.mode = mode
        self.reference_column = reference_column
        self.classifier = classifier
        self.seed = seed
        self.embedding_config = embedding_config

    def fit(self, X: Any, y: Any) -> "Row2VecClassifier":
        """Fit the embedder, then the downstream classifier on its output.

        Args:
            X: Training data.
            y: Target labels.

        Returns:
            Row2VecClassifier: ``self``.
        """
        from sklearn.linear_model import LogisticRegression

        self.transformer_ = Row2VecTransformer(
            embedding_dim=self.embedding_dim,
            mode=self.mode,
            reference_column=self.reference_column,
            seed=self.seed,
            config=self.embedding_config,
        )

        # Clone rather than fit the caller's object in place.
        if self.classifier is None:
            self.classifier_ = LogisticRegression(random_state=self.seed)
        else:
            self.classifier_ = clone(self.classifier)

        embedded = self.transformer_.fit_transform(X)
        self.classifier_.fit(embedded, y)
        self.classes_ = getattr(self.classifier_, "classes_", np.unique(y))
        return self

    def predict(self, X: Any) -> np.ndarray[Any, Any]:
        """Predict labels for ``X``.

        Args:
            X: Data to classify.

        Returns:
            ndarray of predicted labels.
        """
        check_is_fitted(self, ["transformer_", "classifier_"])
        return self.classifier_.predict(self.transformer_.transform(X))  # type: ignore[no-any-return]

    def predict_proba(self, X: Any) -> np.ndarray[Any, Any]:
        """Predict class probabilities for ``X``.

        Args:
            X: Data to classify.

        Returns:
            ndarray of shape (n_samples, n_classes).
        """
        check_is_fitted(self, ["transformer_", "classifier_"])
        return self.classifier_.predict_proba(self.transformer_.transform(X))  # type: ignore[no-any-return]
