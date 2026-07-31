"""
Row2Vec: Intelligent Categorical Encoding System

This module provides adaptive categorical encoding strategies that automatically
analyze data characteristics and apply optimal encoding methods while maintaining
simplicity for beginners and full control for advanced users.
"""

import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.metrics import mutual_info_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder
from tensorflow import keras
from tensorflow.keras import layers

if TYPE_CHECKING:
    from sklearn.base import BaseEstimator, TransformerMixin
else:
    try:
        from sklearn.base import BaseEstimator, TransformerMixin
    except ImportError:
        BaseEstimator = object
        TransformerMixin = object


@dataclass
class CategoricalEncodingConfig:
    """
    Configuration for intelligent categorical encoding strategies.

    This class provides comprehensive control over how categorical variables are encoded,
    with intelligent defaults that automatically select optimal strategies based on
    data characteristics while allowing expert users to fine-tune every aspect.
    """

    # Core strategy selection
    encoding_strategy: str = "adaptive"
    """Encoding strategy selection. Options:
    - "adaptive": Automatically selects best strategy based on data analysis
    - "onehot": One-hot encoding for all categorical features
    - "target": Target encoding for all categorical features
    - "entity": Entity embeddings for all categorical features
    - "ordinal": Ordinal encoding (assumes natural order)
    - "mixed": Use custom strategies per column (requires custom_strategies)
    """

    # Adaptive strategy thresholds
    onehot_threshold: int = 20
    """Use OneHot encoding if cardinality <= this threshold and correlation is low."""

    target_threshold: int = 100
    """Use target encoding if cardinality is between onehot_threshold and this value."""

    entity_threshold: int = 1000
    """Use entity embeddings if cardinality > target_threshold and <= this value."""

    correlation_threshold: float = 0.1
    """Minimum mutual information score to prefer target/entity over onehot."""

    # Target encoding configuration
    target_smoothing: float = 1.0
    """Bayesian smoothing factor for target encoding. Higher values = more smoothing."""

    target_noise: float = 0.01
    """Gaussian noise standard deviation added to target encodings to prevent overfitting."""

    target_cv_folds: int = 5
    """Number of cross-validation folds for target encoding to prevent data leakage."""

    # Entity embedding configuration
    embedding_dim_ratio: float = 0.5
    """Embedding dimension as ratio of sqrt(cardinality). Controls embedding size."""

    min_embedding_dim: int = 2
    """Minimum embedding dimension for entity embeddings."""

    max_embedding_dim: int = 50
    """Maximum embedding dimension for entity embeddings."""

    entity_epochs: int = 50
    """Number of training epochs for entity embedding networks."""

    entity_batch_size: int = 256
    """Batch size for entity embedding training."""

    # Performance vs accuracy trade-offs
    prefer_speed: bool = True
    """Whether to prefer faster methods over more accurate but slower ones."""

    preserve_interpretability: bool = False
    """Whether to prefer interpretable encodings (OneHot/Ordinal) when possible."""

    # Feature selection
    enable_feature_selection: bool = False
    """Whether to enable automatic feature selection based on importance."""

    feature_importance_threshold: float = 0.01
    """Minimum feature importance score to keep feature (only if enable_feature_selection=True)."""

    # Custom strategies (for mixed mode)
    custom_strategies: dict[str, str] = field(default_factory=dict)
    """Custom encoding strategy for specific columns. Format: {column_name: strategy}"""

    # Advanced options
    handle_unknown: str = "ignore"
    """How to handle unknown categories. Options: 'ignore', 'error', 'infrequent_if_exist'"""

    random_state: int = 42
    """Random state for reproducible results."""


class CategoricalAnalyzer:
    """Analyzes categorical data to recommend optimal encoding strategies."""

    def __init__(self, config: CategoricalEncodingConfig):
        self.config = config

    def analyze_column(
        self, series: pd.Series, target: pd.Series | None = None
    ) -> dict[str, Any]:
        """
        Analyze a categorical column to recommend encoding strategy.

        Parameters
        ----------
        series : pd.Series
            Categorical column to analyze
        target : pd.Series, optional
            Target variable for correlation analysis

        Returns
        -------
        Dict[str, Any]
            Analysis results and strategy recommendation
        """
        # Basic statistics
        cardinality = series.nunique()
        missing_rate = series.isna().mean()
        value_counts = series.value_counts()

        # Distribution analysis
        frequency_entropy = self._calculate_entropy(value_counts)
        imbalance_ratio = (
            value_counts.iloc[0] / len(series) if len(value_counts) > 0 else 0
        )

        # Target correlation analysis
        target_correlation = 0.0
        if target is not None and not target.isna().all():
            try:
                # Calculate mutual information between categorical feature and target
                valid_mask = ~(series.isna() | target.isna())
                if valid_mask.sum() > 10:  # Need enough samples
                    target_correlation = mutual_info_score(
                        series[valid_mask].astype(str),
                        target[valid_mask],
                    )
            except Exception:
                target_correlation = 0.0

        # Strategy recommendation
        recommended_strategy = self._recommend_strategy(
            cardinality,
            target_correlation,
            missing_rate,
            imbalance_ratio,
        )

        # Embedding dimension recommendation (for entity embeddings)
        embedding_dim = self._calculate_embedding_dim(cardinality)

        return {
            "cardinality": cardinality,
            "missing_rate": missing_rate,
            "frequency_entropy": frequency_entropy,
            "imbalance_ratio": imbalance_ratio,
            "target_correlation": target_correlation,
            "recommended_strategy": recommended_strategy,
            "embedding_dim": embedding_dim,
            "reasoning": self._explain_recommendation(
                cardinality,
                target_correlation,
                recommended_strategy,
            ),
        }

    def _calculate_entropy(self, value_counts: pd.Series) -> float:
        """Calculate Shannon entropy of value distribution."""
        if len(value_counts) <= 1:
            return 0.0

        probabilities = value_counts / value_counts.sum()
        return float(-np.sum(probabilities * np.log2(probabilities + 1e-10)))

    def _recommend_strategy(
        self,
        cardinality: int,
        target_correlation: float,
        missing_rate: float,
        imbalance_ratio: float,
    ) -> str:
        """Recommend encoding strategy based on data characteristics."""

        # Handle edge cases
        if cardinality <= 1:
            return "drop"  # Constant column

        if cardinality > self.config.entity_threshold:
            return "entity"  # Too high cardinality, must use embeddings

        # Hybrid intelligent strategy selection
        high_correlation = target_correlation > self.config.correlation_threshold

        if cardinality <= self.config.onehot_threshold:
            # Low cardinality: prefer OneHot unless high correlation
            if high_correlation and not self.config.preserve_interpretability:
                return "target"
            return "onehot"

        if cardinality <= self.config.target_threshold:
            # Medium cardinality: prefer target encoding if correlated
            if high_correlation:
                return "target"
            if self.config.prefer_speed:
                return "onehot"  # Fallback to onehot if speed preferred
            return "entity"  # Use embeddings for better representation

        # High cardinality: prefer entity embeddings
        if high_correlation and not self.config.prefer_speed:
            return "entity"
        return "target"  # Fallback to target encoding

    def _calculate_embedding_dim(self, cardinality: int) -> int:
        """Calculate optimal embedding dimension for entity embeddings."""
        # Rule of thumb: embedding_dim = sqrt(cardinality) * ratio
        dim = int(np.sqrt(cardinality) * self.config.embedding_dim_ratio)
        return int(
            np.clip(dim, self.config.min_embedding_dim, self.config.max_embedding_dim)
        )

    def _explain_recommendation(
        self, cardinality: int, correlation: float, strategy: str
    ) -> str:
        """Provide human-readable explanation for strategy recommendation."""
        explanations = {
            "onehot": f"Low cardinality ({cardinality}) and low target correlation ({correlation:.3f}). OneHot is fast and interpretable.",
            "target": f"Medium cardinality ({cardinality}) with significant target correlation ({correlation:.3f}). Target encoding captures relationships.",
            "entity": f"High cardinality ({cardinality}) requires dense representation. Entity embeddings learn semantic relationships.",
            "drop": f"Constant or near-constant column (cardinality={cardinality}). Consider dropping.",
        }
        return explanations.get(strategy, f"Strategy: {strategy}")


class EntityEmbeddingTrainer:
    """Trains entity embeddings for high-cardinality categorical features."""

    def __init__(self, config: CategoricalEncodingConfig) -> None:
        self.config = config
        self.models_: dict[str, Any] = {}
        self.label_encoders_: dict[str, LabelEncoder] = {}

    def fit_column_embedding(
        self,
        series: pd.Series,
        target: pd.Series | None = None,
        embedding_dim: int = 10,
    ) -> NDArray[Any]:
        """
        Train entity embeddings for a categorical column.

        Parameters
        ----------
        series : pd.Series
            Categorical column to embed
        target : pd.Series, optional
            Target variable for supervised embedding
        embedding_dim : int
            Dimension of embedding vectors

        Returns
        -------
        np.ndarray
            Trained embedding matrix of shape (cardinality, embedding_dim)
        """
        # Prepare data
        valid_mask = ~series.isna()
        valid_series = series[valid_mask]

        # Label encode categories
        label_encoder = LabelEncoder()
        encoded_categories = label_encoder.fit_transform(valid_series)
        cardinality = len(label_encoder.classes_)

        # Store for later use
        column_name = str(series.name) if series.name is not None else "unknown"
        self.label_encoders_[column_name] = label_encoder

        # Build embedding model
        if target is not None and not target.isna().all():
            # Supervised embedding using target variable
            embeddings = self._train_supervised_embedding(
                encoded_categories,
                target[valid_mask],
                cardinality,
                embedding_dim,
            )
        else:
            # Unsupervised embedding using autoencoder
            embeddings = self._train_unsupervised_embedding(
                encoded_categories,
                cardinality,
                embedding_dim,
            )

        return embeddings

    def _train_supervised_embedding(
        self,
        categories: NDArray[Any],
        target: pd.Series,
        cardinality: int,
        embedding_dim: int,
    ) -> NDArray[Any]:
        """Train supervised entity embeddings using target variable."""

        # Determine task type
        if target.dtype in ["object", "category"] or target.nunique() < 20:
            # Classification task
            target_encoder = LabelEncoder()
            y_encoded = target_encoder.fit_transform(target.astype(str))
            n_classes = len(target_encoder.classes_)
            task_type = "classification"
        else:
            # Regression task
            y_encoded = target.values.astype(np.float32)
            n_classes = 1
            task_type = "regression"

        # Build model
        input_layer = keras.Input(shape=(), name="category_input")
        embedding_layer = layers.Embedding(
            input_dim=cardinality,
            output_dim=embedding_dim,
            name="category_embedding",
        )(input_layer)

        # Flatten embedding
        flattened = layers.Flatten()(embedding_layer)

        # Add dense layers for prediction
        hidden = layers.Dense(64, activation="relu")(flattened)
        hidden = layers.Dropout(0.3)(hidden)

        if task_type == "classification":
            output = layers.Dense(n_classes, activation="softmax")(hidden)
            model = keras.Model(inputs=input_layer, outputs=output)
            model.compile(
                optimizer="adam",
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"],
            )
        else:
            output = layers.Dense(1)(hidden)
            model = keras.Model(inputs=input_layer, outputs=output)
            model.compile(
                optimizer="adam",
                loss="mse",
                metrics=["mae"],
            )

        # Train model
        model.fit(
            categories,
            y_encoded,
            epochs=self.config.entity_epochs,
            batch_size=self.config.entity_batch_size,
            verbose=0,
            validation_split=0.2,
        )

        # Extract embeddings
        embedding_model = keras.Model(
            inputs=model.input,
            outputs=model.get_layer("category_embedding").output,
        )

        # Get embeddings for all categories
        all_categories = np.arange(cardinality)
        embeddings = embedding_model.predict(all_categories, verbose=0)

        # Flatten if needed (embedding layer adds dimension)
        if embeddings.ndim == 3:
            embeddings = embeddings.squeeze(axis=1)

        return embeddings

    def _train_unsupervised_embedding(
        self,
        categories: NDArray[Any],
        cardinality: int,
        embedding_dim: int,
    ) -> NDArray[Any]:
        """Train unsupervised entity embeddings using autoencoder approach."""

        # Create one-hot representation for autoencoder
        onehot = np.eye(cardinality)[categories]

        # Build autoencoder
        input_layer = keras.Input(shape=(cardinality,), name="onehot_input")

        # Encoder
        encoded = layers.Dense(
            embedding_dim,
            activation="relu",
            name="embedding_layer",
        )(input_layer)

        # Decoder
        decoded = layers.Dense(
            cardinality,
            activation="softmax",
        )(encoded)

        # Model
        autoencoder = keras.Model(inputs=input_layer, outputs=decoded)
        autoencoder.compile(
            optimizer="adam",
            loss="categorical_crossentropy",
        )

        # Train
        autoencoder.fit(
            onehot,
            onehot,
            epochs=self.config.entity_epochs,
            batch_size=self.config.entity_batch_size,
            verbose=0,
            validation_split=0.2,
        )

        # Extract embeddings
        encoder = keras.Model(
            inputs=autoencoder.input,
            outputs=autoencoder.get_layer("embedding_layer").output,
        )

        # Get embeddings for all categories
        all_onehot = np.eye(cardinality)
        return encoder.predict(all_onehot, verbose=0)


class TargetEncoder:
    """Implements Bayesian target encoding with cross-validation."""

    def __init__(self, config: CategoricalEncodingConfig) -> None:
        self.config = config
        self.encodings_: dict[Any, float] = {}
        self.global_mean_: float | None = None

    def fit_transform(
        self,
        series: pd.Series,
        target: pd.Series,
    ) -> pd.Series:
        """
        Fit target encoder and transform the series.

        Parameters
        ----------
        series : pd.Series
            Categorical column to encode
        target : pd.Series
            Target variable

        Returns
        -------
        pd.Series
            Target-encoded values
        """
        if target is None:
            raise ValueError("Target variable is required for target encoding")

        # Handle missing values in target
        valid_mask = ~(series.isna() | target.isna())
        if valid_mask.sum() == 0:
            raise ValueError("No valid samples for target encoding")

        series_clean = series[valid_mask]
        target_clean = target[valid_mask]

        # Calculate global mean
        self.global_mean_ = float(target_clean.mean())

        # Use cross-validation to prevent overfitting
        kf = KFold(
            n_splits=self.config.target_cv_folds,
            shuffle=True,
            random_state=self.config.random_state,
        )

        encoded_values = np.full(len(series_clean), self.global_mean_ or 0.0)

        for train_idx, val_idx in kf.split(series_clean):
            # Fit on train fold
            train_series = series_clean.iloc[train_idx]
            train_target = target_clean.iloc[train_idx]

            # Calculate category means with Bayesian smoothing
            category_stats = train_target.groupby(train_series).agg(["mean", "count"])

            # Apply Bayesian smoothing
            smoothed_means = {}
            for category in category_stats.index:
                cat_mean = category_stats.loc[category, "mean"]
                cat_count = category_stats.loc[category, "count"]

                # Bayesian smoothing formula
                smoothed_mean = (
                    cat_count * cat_mean
                    + self.config.target_smoothing * (self.global_mean_ or 0.0)
                ) / (cat_count + self.config.target_smoothing)
                smoothed_means[category] = smoothed_mean

            # Apply to validation fold
            val_series = series_clean.iloc[val_idx]
            for i, category in enumerate(val_series):
                val_global_idx = val_idx[i]
                if category in smoothed_means:
                    encoded_values[val_global_idx] = smoothed_means[category]
                else:
                    encoded_values[val_global_idx] = self.global_mean_ or 0.0

        # Store final encodings for transform
        final_stats = target_clean.groupby(series_clean).agg(["mean", "count"])
        self.encodings_ = {}
        for category in final_stats.index:
            cat_mean = final_stats.loc[category, "mean"]
            cat_count = final_stats.loc[category, "count"]

            smoothed_mean = (
                cat_count * cat_mean
                + self.config.target_smoothing * (self.global_mean_ or 0.0)
            ) / (cat_count + self.config.target_smoothing)
            self.encodings_[category] = smoothed_mean

        # Add noise to prevent overfitting
        if self.config.target_noise > 0:
            noise = np.random.normal(
                0,
                self.config.target_noise,
                len(encoded_values),
            )
            encoded_values += noise

        # Create result series with original index
        result = pd.Series(index=series.index, dtype=float)
        result[valid_mask] = encoded_values
        result[~valid_mask] = self.global_mean_  # Fill missing with global mean

        return result

    def transform(self, series: pd.Series) -> pd.Series:
        """Transform new data using fitted encodings."""
        if not self.encodings_:
            raise ValueError("Encoder must be fitted before transform")

        return series.map(self.encodings_).fillna(self.global_mean_)


class CategoricalEncoder(BaseEstimator, TransformerMixin):  # type: ignore[misc]
    """
    Intelligent categorical encoder with adaptive strategy selection.

    This encoder analyzes categorical data characteristics and automatically
    selects optimal encoding strategies while providing full control for
    advanced users.
    """

    def __init__(self, config: CategoricalEncodingConfig | None = None) -> None:
        self.config = config or CategoricalEncodingConfig()
        self.analyzer_ = CategoricalAnalyzer(self.config)
        self.column_strategies_: dict[str, str] = {}
        self.fitted_encoders_: dict[str, Any] = {}
        self.entity_embeddings_: dict[str, Any] = {}
        self.feature_names_in_: list[str] | None = None
        self.feature_names_out_: list[str] | None = None
        self.analysis_report_: dict[str, Any] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "CategoricalEncoder":
        """
        Fit the categorical encoder on training data.

        Parameters
        ----------
        X : pd.DataFrame
            Categorical features to encode
        y : pd.Series, optional
            Target variable for supervised encoding strategies

        Returns
        -------
        self : CategoricalEncoder
            Fitted encoder instance
        """
        # Store feature names
        self.feature_names_in_ = list(X.columns)

        # Analyze each column and determine strategies
        for col in X.columns:
            analysis = self.analyzer_.analyze_column(X[col], y)
            self.analysis_report_[col] = analysis

            # Determine final strategy
            if (
                self.config.encoding_strategy == "mixed"
                and col in self.config.custom_strategies
            ):
                strategy = self.config.custom_strategies[col]
            elif self.config.encoding_strategy == "adaptive":
                strategy = analysis["recommended_strategy"]
            else:
                strategy = self.config.encoding_strategy

            self.column_strategies_[col] = strategy

            # Fit appropriate encoder
            self._fit_column_encoder(col, X[col], y, strategy, analysis)

        # Determine output feature names
        self._calculate_output_features(X)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform categorical data using fitted encoders.

        Parameters
        ----------
        X : pd.DataFrame
            Categorical data to transform

        Returns
        -------
        pd.DataFrame
            Encoded categorical data
        """
        if not self.fitted_encoders_:
            raise ValueError("Encoder must be fitted before transform")

        # Check feature consistency
        if list(X.columns) != self.feature_names_in_:
            raise ValueError(
                f"Feature mismatch. Expected {self.feature_names_in_}, got {list(X.columns)}"
            )

        encoded_data = {}

        for col in X.columns:
            strategy = self.column_strategies_[col]

            if strategy == "drop":
                continue  # Skip dropped columns

            encoded_col = self._transform_column(col, X[col], strategy)

            # Handle multi-dimensional outputs (embeddings, onehot)
            if isinstance(encoded_col, pd.DataFrame):
                for sub_col in encoded_col.columns:
                    encoded_data[sub_col] = encoded_col[sub_col]
            else:
                encoded_data[col] = encoded_col

        result_df = pd.DataFrame(encoded_data, index=X.index)

        # Apply feature selection if enabled
        if self.config.enable_feature_selection:
            result_df = self._apply_feature_selection(result_df)

        return result_df

    def _fit_column_encoder(
        self,
        col: str,
        series: pd.Series,
        target: pd.Series | None,
        strategy: str,
        analysis: dict[str, Any],
    ) -> None:
        """Fit encoder for a specific column based on strategy."""

        if strategy == "drop":
            self.fitted_encoders_[col] = None
            return

        if strategy == "onehot":
            encoder = OneHotEncoder(
                handle_unknown=self.config.handle_unknown,
                sparse_output=False,  # Return dense arrays
            )
            # Fit on reshaped data
            encoder.fit(np.asarray(series.values).reshape(-1, 1))
            self.fitted_encoders_[col] = encoder

        elif strategy == "target":
            if target is None:
                warnings.warn(
                    f"Target encoding requested for {col} but no target provided. Using ordinal encoding.",
                    stacklevel=2,
                )
                encoder = OrdinalEncoder(
                    handle_unknown="use_encoded_value", unknown_value=-1
                )
                encoder.fit(np.asarray(series.values).reshape(-1, 1))
                self.fitted_encoders_[col] = encoder
            else:
                encoder = TargetEncoder(self.config)
                encoder.fit_transform(series, target)  # Fit during transform
                self.fitted_encoders_[col] = encoder

        elif strategy == "entity":
            # Train entity embeddings
            embedding_trainer = EntityEmbeddingTrainer(self.config)
            embedding_dim = analysis.get("embedding_dim", 10)

            embeddings = embedding_trainer.fit_column_embedding(
                series,
                target,
                embedding_dim,
            )

            self.entity_embeddings_[col] = {
                "embeddings": embeddings,
                "label_encoder": embedding_trainer.label_encoders_.get(col),
                "embedding_dim": embedding_dim,
            }
            self.fitted_encoders_[col] = "entity"  # Flag for entity embeddings

        elif strategy == "ordinal":
            encoder = OrdinalEncoder(
                handle_unknown="use_encoded_value", unknown_value=-1
            )
            encoder.fit(np.asarray(series.values).reshape(-1, 1))
            self.fitted_encoders_[col] = encoder

        else:
            raise ValueError(f"Unknown encoding strategy: {strategy}")

    def _transform_column(self, col: str, series: pd.Series, strategy: str) -> Any:
        """Transform a specific column using its fitted encoder."""

        if strategy == "drop":
            return None

        if strategy == "onehot":
            encoder = self.fitted_encoders_[col]
            encoded = encoder.transform(np.asarray(series.values).reshape(-1, 1))

            # Create column names
            if hasattr(encoder, "get_feature_names_out"):
                feature_names = encoder.get_feature_names_out([col])
            else:
                feature_names = [f"{col}_{cat}" for cat in encoder.categories_[0]]

            return pd.DataFrame(encoded, columns=feature_names, index=series.index)

        if strategy == "target":
            encoder = self.fitted_encoders_[col]
            if isinstance(encoder, TargetEncoder):
                return encoder.transform(series)
            # Fallback ordinal encoder
            encoded = encoder.transform(np.asarray(series.values).reshape(-1, 1))
            return pd.Series(encoded.flatten(), index=series.index, name=col)

        if strategy == "entity":
            # Transform using entity embeddings
            embedding_info = self.entity_embeddings_[col]
            label_encoder = embedding_info["label_encoder"]
            embeddings = embedding_info["embeddings"]
            embedding_dim = embedding_info["embedding_dim"]

            # Handle unknown categories
            valid_mask = ~series.isna()
            encoded_series = np.full(len(series), -1, dtype=int)

            if valid_mask.any():
                valid_categories = series[valid_mask]
                try:
                    encoded_valid = label_encoder.transform(valid_categories)
                    encoded_series[valid_mask] = encoded_valid
                except ValueError:
                    # Handle unknown categories
                    known_categories = set(label_encoder.classes_)
                    for i, cat in enumerate(valid_categories):
                        if cat in known_categories:
                            encoded_series[valid_mask.to_numpy()[valid_mask]][i] = (
                                label_encoder.transform([cat])[0]
                            )
                        else:
                            encoded_series[valid_mask.to_numpy()[valid_mask]][i] = (
                                0  # Default to first category
                            )

            # Map to embeddings
            result_data = {}
            for dim in range(embedding_dim):
                col_name = f"{col}_emb_{dim}"
                embedding_values = np.zeros(len(series))

                for i, encoded_cat in enumerate(encoded_series):
                    if encoded_cat >= 0 and encoded_cat < len(embeddings):
                        embedding_values[i] = embeddings[encoded_cat, dim]
                    else:
                        embedding_values[i] = 0.0  # Default value for unknown

                result_data[col_name] = embedding_values

            return pd.DataFrame(result_data, index=series.index)

        if strategy == "ordinal":
            encoder = self.fitted_encoders_[col]
            encoded = encoder.transform(np.asarray(series.values).reshape(-1, 1))
            return pd.Series(encoded.flatten(), index=series.index, name=col)

        raise ValueError(f"Unknown encoding strategy: {strategy}")

    def _calculate_output_features(self, X: pd.DataFrame) -> None:
        """Calculate output feature names after encoding."""
        feature_names = []

        for col in X.columns:
            strategy = self.column_strategies_[col]

            if strategy == "drop":
                continue
            if strategy == "onehot":
                encoder = self.fitted_encoders_[col]
                if hasattr(encoder, "get_feature_names_out"):
                    feature_names.extend(encoder.get_feature_names_out([col]))
                else:
                    feature_names.extend(
                        [f"{col}_{cat}" for cat in encoder.categories_[0]]
                    )
            elif strategy == "entity":
                embedding_dim = self.entity_embeddings_[col]["embedding_dim"]
                feature_names.extend([f"{col}_emb_{i}" for i in range(embedding_dim)])
            else:
                feature_names.append(col)

        self.feature_names_out_ = feature_names

    def _apply_feature_selection(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply feature selection based on importance scores."""
        if self.config.feature_importance_threshold <= 0:
            # Skip feature selection if threshold is 0 or negative
            return df

        # Calculate feature importance using variance for numerical features
        # and chi-square for categorical features
        feature_importances = {}

        for col in df.columns:
            if df[col].dtype in ["object", "category"]:
                # For categorical features, use normalized entropy as importance
                value_counts = df[col].value_counts(normalize=True)
                entropy = -np.sum(
                    value_counts * np.log(value_counts + 1e-10)
                )  # Add small epsilon for numerical stability
                max_entropy = np.log(len(df[col].unique()))
                importance = entropy / max_entropy if max_entropy > 0 else 0.0
            # For numerical features, use normalized variance
            elif df[col].var() == 0:
                importance = 0.0
            else:
                # Type ignore: MyPy cannot infer that df[col] is numeric here
                importance = min(  # type: ignore[type-var]
                    1.0,
                    df[col].var() / (df[col].var() + df[col].mean() ** 2),  # type: ignore[operator,call-overload]
                )

            feature_importances[col] = importance

        # Filter features based on threshold
        selected_features = [
            col
            for col, importance in feature_importances.items()
            if importance >= self.config.feature_importance_threshold
        ]

        if len(selected_features) == 0:
            # If no features meet the threshold, keep the top 50% by importance
            sorted_features = sorted(
                feature_importances.items(), key=lambda x: x[1], reverse=True
            )
            n_keep = max(1, len(sorted_features) // 2)
            selected_features = [col for col, _ in sorted_features[:n_keep]]

        return df[selected_features]

    def get_feature_names_out(
        self, input_features: list[str] | None = None
    ) -> list[str]:
        """Get output feature names for transformation."""
        if self.feature_names_out_ is None:
            raise ValueError("Encoder must be fitted before getting feature names")
        return self.feature_names_out_

    def get_analysis_report(self) -> dict[str, dict[str, Any]]:
        """Get detailed analysis report for all columns."""
        return self.analysis_report_.copy()
