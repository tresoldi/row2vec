"""
Scikit-learn integration for Row2Vec embeddings.

This module provides scikit-learn compatible transformers for Row2Vec,
allowing seamless integration with sklearn pipelines and workflows.
"""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from .api import learn_embedding_v2, learn_embedding_with_model_v2
from .config import EmbeddingConfig, create_config_for_mode


class Row2VecModel:
    """Wrapper for Row2Vec models to provide consistent interface."""

    def __init__(
        self,
        model: Any,
        preprocessor: Any,
        metadata: dict[str, Any],
        config: EmbeddingConfig,
    ):
        self.model = model
        self.preprocessor = preprocessor
        self.metadata = metadata
        self.config = config

    def embed(self, X: pd.DataFrame) -> pd.DataFrame:
        """Generate embeddings for new data."""
        # Use the existing API to generate embeddings with the same config
        return learn_embedding_v2(X, self.config)


class Row2VecTransformer(BaseEstimator, TransformerMixin):  # type: ignore[misc]
    """
    Scikit-learn compatible transformer for Row2Vec embeddings.

    This transformer can be used in sklearn pipelines and follows the
    standard fit/transform API. It internally uses Row2Vec's config-based
    API for flexibility and type safety.

    Parameters
    ----------
    embedding_dim : int, default=10
        Dimensionality of the embedding space.
    mode : str, default="unsupervised"
        Embedding mode. Options: "unsupervised", "target", "pca", "tsne", "umap", "contrastive".
    reference_column : str, optional
        Reference column name for supervised ("target") mode.
    config : EmbeddingConfig, optional
        Pre-configured EmbeddingConfig object. If provided, other parameters are ignored.
    **kwargs
        Additional parameters passed to the embedding configuration.
        These can include nested parameters like neural__max_epochs=100.

    Attributes
    ----------
    config_ : EmbeddingConfig
        The configuration object used for embedding generation.
    model_ : object
        The trained Row2Vec model (if using model-based modes).
    feature_names_in_ : ndarray of shape (n_features,)
        Names of features seen during fit.
    n_features_in_ : int
        Number of features seen during fit.

    Examples
    --------
    >>> from row2vec.sklearn import Row2VecTransformer
    >>> import pandas as pd
    >>>
    >>> # Simple usage
    >>> transformer = Row2VecTransformer(embedding_dim=5, mode="unsupervised")
    >>> X_embedded = transformer.fit_transform(df)
    >>>
    >>> # In a pipeline
    >>> from sklearn.pipeline import Pipeline
    >>> from sklearn.cluster import KMeans
    >>>
    >>> pipeline = Pipeline([
    ...     ('embed', Row2VecTransformer(embedding_dim=10)),
    ...     ('cluster', KMeans(n_clusters=3))
    ... ])
    >>> pipeline.fit(df)
    >>>
    >>> # With configuration object
    >>> from row2vec.config import EmbeddingConfig, NeuralConfig
    >>> config = EmbeddingConfig(
    ...     mode="unsupervised",
    ...     embedding_dim=15,
    ...     neural=NeuralConfig(max_epochs=100, batch_size=32)
    ... )
    >>> transformer = Row2VecTransformer(config=config)
    >>> X_embedded = transformer.fit_transform(df)
    """

    def __init__(
        self,
        embedding_dim: int = 10,
        mode: str = "unsupervised",
        reference_column: str | None = None,
        config: EmbeddingConfig | None = None,
        **kwargs: Any,
    ) -> None:
        self.embedding_dim = embedding_dim
        self.mode = mode
        self.reference_column = reference_column
        self.config = config
        self.kwargs = kwargs
        # Initialize attributes that will be set in fit()
        self.model_wrapper_: Row2VecModel | None = None

    def _create_config(self) -> EmbeddingConfig:
        """Create the embedding configuration."""
        if self.config is not None:
            return self.config

        # Create base config for the mode
        config = create_config_for_mode(self.mode)
        config.embedding_dim = self.embedding_dim

        if self.mode == "target" and self.reference_column is not None:
            config.reference_column = self.reference_column

        # Apply any additional kwargs
        # Handle nested parameters like neural__max_epochs
        for key, value in self.kwargs.items():
            if "__" in key:
                # Handle nested parameters (sklearn convention)
                section, param = key.split("__", 1)
                if hasattr(config, section):
                    section_config = getattr(config, section)
                    if hasattr(section_config, param):
                        setattr(section_config, param, value)
            # Handle top-level parameters
            elif hasattr(config, key):
                setattr(config, key, value)

        return config

    def _validate_input(self, X: Any) -> pd.DataFrame:
        """Validate and convert input to DataFrame."""
        if isinstance(X, np.ndarray):
            # Convert numpy array to DataFrame
            if hasattr(self, "feature_names_in_"):
                if X.shape[1] != len(self.feature_names_in_):
                    raise ValueError(
                        f"X has {X.shape[1]} features, but transformer was fitted with "
                        f"{len(self.feature_names_in_)} features",
                    )
                X = pd.DataFrame(X, columns=self.feature_names_in_)
            else:
                X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
        elif isinstance(X, pd.DataFrame):
            X = X.copy()
        else:
            # Try to convert to numpy array first (handles sparse matrices, etc.)
            try:
                X = np.asarray(X)
                if hasattr(self, "feature_names_in_"):
                    if X.shape[1] != len(self.feature_names_in_):
                        raise ValueError(
                            f"X has {X.shape[1]} features, but transformer was fitted with "
                            f"{len(self.feature_names_in_)} features",
                        )
                    X = pd.DataFrame(X, columns=self.feature_names_in_)
                else:
                    X = pd.DataFrame(
                        X, columns=[f"feature_{i}" for i in range(X.shape[1])]
                    )
            except Exception as e:
                raise TypeError(f"Cannot convert input to DataFrame: {e}")

        return X  # type: ignore[no-any-return]

    def fit(self, X: Any, y: Any = None) -> "Row2VecTransformer":
        """
        Fit the Row2Vec transformer.

        Parameters
        ----------
        X : DataFrame or array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,), optional
            Target values (ignored, exists for sklearn compatibility).

        Returns
        -------
        self : Row2VecTransformer
            Returns the instance itself.
        """
        X = self._validate_input(X)

        # Store input information for sklearn compatibility
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = np.array(X.columns)

        # Create configuration
        self.config_ = self._create_config()

        # For modes that require a model, fit it
        if self.mode in ["unsupervised", "target", "contrastive"]:
            # Train the model and store it
            result = learn_embedding_with_model_v2(X, self.config_)
            embeddings, model, preprocessor, metadata = result
            self.model_wrapper_ = Row2VecModel(
                model, preprocessor, metadata, self.config_
            )
        else:
            # For classical methods, no model storage needed
            self.model_wrapper_ = None

        return self

    def transform(self, X: Any) -> np.ndarray[Any, Any]:
        """
        Transform data to embedding space.

        Parameters
        ----------
        X : DataFrame or array-like of shape (n_samples, n_features)
            Data to transform.

        Returns
        -------
        X_embedded : ndarray of shape (n_samples, embedding_dim)
            Embedded data.
        """
        check_is_fitted(self, ["config_"])

        X = self._validate_input(X)

        if self.model_wrapper_ is not None:
            # Use the fitted model to embed new data
            embeddings = self.model_wrapper_.embed(X)
        else:
            # For classical methods or fallback, apply transformation directly
            embeddings = learn_embedding_v2(X, self.config_)

        return embeddings.values

    def fit_transform(
        self, X: Any, y: Any = None, **fit_params: Any
    ) -> np.ndarray[Any, Any]:
        """
        Fit the transformer and transform the data.

        Parameters
        ----------
        X : DataFrame or array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,), optional
            Target values (ignored, exists for sklearn compatibility).
        **fit_params : dict
            Additional parameters (ignored, exists for sklearn compatibility).

        Returns
        -------
        X_embedded : ndarray of shape (n_samples, embedding_dim)
            Embedded training data.
        """
        X = self._validate_input(X)

        # Store input information for sklearn compatibility
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = np.array(X.columns)

        # Create configuration
        self.config_ = self._create_config()

        # Generate embeddings
        embeddings = learn_embedding_v2(X, self.config_)

        # For neural modes, we might want to store the model for future transforms
        if self.mode in ["unsupervised", "target", "contrastive"]:
            try:
                result = learn_embedding_with_model_v2(X, self.config_)
                embeddings_model, model, preprocessor, metadata = result
                self.model_wrapper_ = Row2VecModel(
                    model, preprocessor, metadata, self.config_
                )
            except:
                # If model creation fails, fall back to no model
                self.model_wrapper_ = None
        else:
            self.model_wrapper_ = None

        return embeddings.values

    def get_feature_names_out(
        self, input_features: np.ndarray[Any, Any] | None = None
    ) -> np.ndarray[Any, Any]:
        """
        Get output feature names for transformation.

        Parameters
        ----------
        input_features : array-like of str or None, default=None
            Not used, exists for sklearn compatibility.

        Returns
        -------
        feature_names_out : ndarray of shape (embedding_dim,), dtype=str
            Feature names for the embedded space.
        """
        check_is_fitted(self, ["config_"])
        return np.array([f"row2vec_{i}" for i in range(self.config_.embedding_dim)])

    def _more_tags(self) -> dict[str, Any]:
        """Return tags for sklearn compatibility."""
        return {
            "requires_y": False,
            "requires_fit": True,
            "X_types": ["2darray"],
            "allow_nan": False,
            "stateless": False,
            "no_validation": False,
        }


class Row2VecClassifier(BaseEstimator):  # type: ignore[misc]
    """
    Scikit-learn compatible classifier using Row2Vec embeddings.

    This combines Row2Vec embedding generation with a downstream classifier,
    making it easy to use embeddings for classification tasks in sklearn pipelines.

    Parameters
    ----------
    embedding_dim : int, default=10
        Dimensionality of the embedding space.
    classifier : sklearn classifier, optional
        The downstream classifier. If None, uses LogisticRegression.
    embedding_config : EmbeddingConfig, optional
        Configuration for embedding generation.
    **embedding_kwargs
        Additional parameters for embedding configuration.

    Examples
    --------
    >>> from row2vec.sklearn import Row2VecClassifier
    >>> from sklearn.ensemble import RandomForestClassifier
    >>>
    >>> # With default classifier
    >>> clf = Row2VecClassifier(embedding_dim=15)
    >>> clf.fit(X_train, y_train)
    >>> predictions = clf.predict(X_test)
    >>>
    >>> # With custom classifier
    >>> clf = Row2VecClassifier(
    ...     embedding_dim=20,
    ...     classifier=RandomForestClassifier(n_estimators=100)
    ... )
    >>> clf.fit(X_train, y_train)
    """

    def __init__(
        self,
        embedding_dim: int = 10,
        classifier: Any = None,
        embedding_config: EmbeddingConfig | None = None,
        **embedding_kwargs: Any,
    ) -> None:
        self.embedding_dim = embedding_dim
        self.classifier = classifier
        self.embedding_config = embedding_config
        self.embedding_kwargs = embedding_kwargs

    def fit(self, X: Any, y: Any) -> "Row2VecClassifier":
        """Fit the embedding and classifier."""
        from sklearn.linear_model import LogisticRegression

        # Set up embedding transformer
        self.embedding_transformer_ = Row2VecTransformer(
            embedding_dim=self.embedding_dim,
            config=self.embedding_config,
            **self.embedding_kwargs,
        )

        # Set up classifier
        if self.classifier is None:
            self.classifier_ = LogisticRegression(random_state=1305)
        else:
            self.classifier_ = self.classifier

        # Fit embedding transformer and transform data
        X_embedded = self.embedding_transformer_.fit_transform(X)

        # Fit classifier on embedded data
        self.classifier_.fit(X_embedded, y)

        return self

    def predict(self, X: Any) -> np.ndarray[Any, Any]:
        """Make predictions on new data."""
        check_is_fitted(self, ["embedding_transformer_", "classifier_"])

        X_embedded = self.embedding_transformer_.transform(X)
        return self.classifier_.predict(X_embedded)  # type: ignore[no-any-return]

    def predict_proba(self, X: Any) -> np.ndarray[Any, Any]:
        """Predict class probabilities."""
        check_is_fitted(self, ["embedding_transformer_", "classifier_"])

        X_embedded = self.embedding_transformer_.transform(X)
        return self.classifier_.predict_proba(X_embedded)  # type: ignore[no-any-return]
