"""
Row2Vec: Intelligent Pipeline Builder

This module provides intelligent pipeline construction that automatically
analyzes data characteristics and builds optimal preprocessing pipelines
with adaptive categorical encoding strategies.
"""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from .categorical_encoding import CategoricalEncoder, CategoricalEncodingConfig
from .config import EmbeddingConfig
from .imputation import AdaptiveImputer, ImputationConfig


class PipelineBuilder:
    """
    Intelligent pipeline builder that analyzes data and constructs optimal
    preprocessing pipelines with adaptive strategies.
    """

    def __init__(self, config: EmbeddingConfig | None = None):
        self.config = config or EmbeddingConfig()
        self.analysis_report_: dict[str, Any] = {}
        self.pipeline_description_: dict[str, Any] = {}

    def build_preprocessing_pipeline(
        self,
        df: pd.DataFrame,
        target: pd.Series | None = None,
        mode: str = "unsupervised",
    ) -> tuple[ColumnTransformer, dict[str, Any]]:
        """
        Build intelligent preprocessing pipeline based on data analysis.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataset to analyze
        target : pd.Series, optional
            Target variable for supervised preprocessing
        mode : str
            Embedding mode that influences preprocessing strategy

        Returns
        -------
        Tuple[ColumnTransformer, Dict[str, Any]]
            Fitted preprocessing pipeline and analysis report
        """
        # Analyze data characteristics
        data_analysis = self._analyze_dataset(df, target)
        self.analysis_report_ = data_analysis

        # Separate numeric and categorical columns
        numeric_cols = self._get_numeric_columns(df)
        categorical_cols = self._get_categorical_columns(df)

        # Remove target column from features if in target mode
        if mode == "target" and target is not None and target.name in df.columns:
            target_name = str(target.name)
            if target_name in numeric_cols:
                numeric_cols.remove(target_name)
            if target_name in categorical_cols:
                categorical_cols.remove(target_name)

        # Build numeric pipeline
        numeric_pipeline = self._build_numeric_pipeline(
            df[numeric_cols] if numeric_cols else pd.DataFrame()
        )

        # Build categorical pipeline
        categorical_pipeline = self._build_categorical_pipeline(
            df[categorical_cols] if categorical_cols else pd.DataFrame(),
            target,
        )

        # Combine pipelines
        transformers = []

        if numeric_cols:
            transformers.append(("numeric", numeric_pipeline, numeric_cols))

        if categorical_cols:
            transformers.append(("categorical", categorical_pipeline, categorical_cols))

        # Create final pipeline
        if not transformers:
            raise ValueError("No valid columns found for preprocessing")

        preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder="drop",  # Drop any remaining columns
            sparse_threshold=0,  # Return dense arrays
        )

        # Store pipeline description
        self.pipeline_description_ = self._describe_pipeline(
            numeric_cols,
            categorical_cols,
            data_analysis,
        )

        return preprocessor, data_analysis

    def _analyze_dataset(
        self,
        df: pd.DataFrame,
        target: pd.Series | None = None,
    ) -> dict[str, Any]:
        """Analyze dataset characteristics to inform pipeline construction."""

        analysis = {
            "dataset_shape": df.shape,
            "total_missing": df.isnull().sum().sum(),
            "missing_percentage": (df.isnull().sum().sum() / df.size) * 100,
            "numeric_columns": len(df.select_dtypes(include=[np.number]).columns),
            "categorical_columns": len(
                df.select_dtypes(include=["object", "category"]).columns
            ),
            "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024 / 1024,
            "has_target": target is not None,
            "target_type": None,
        }

        if target is not None:
            if target.dtype in ["object", "category"] or target.nunique() < 20:
                analysis["target_type"] = "classification"
            else:
                analysis["target_type"] = "regression"

        # Column-specific analysis
        analysis["column_analysis"] = {}
        for col in df.columns:
            col_analysis = {
                "dtype": str(df[col].dtype),
                "missing_count": df[col].isnull().sum(),
                "missing_percentage": (df[col].isnull().sum() / len(df)) * 100,
                "unique_values": df[col].nunique(),
                "memory_usage_mb": df[col].memory_usage(deep=True) / 1024 / 1024,
            }

            if df[col].dtype in ["object", "category"]:
                col_analysis["cardinality"] = df[col].nunique()
                col_analysis["most_frequent"] = (
                    df[col].value_counts().iloc[0]
                    if len(df[col].value_counts()) > 0
                    else 0
                )
                col_analysis["frequency_distribution"] = (
                    df[col].value_counts().head(5).to_dict()
                )
            else:
                col_analysis["mean"] = (
                    df[col].mean() if not df[col].isnull().all() else None
                )
                col_analysis["std"] = (
                    df[col].std() if not df[col].isnull().all() else None
                )
                col_analysis["min"] = (
                    df[col].min() if not df[col].isnull().all() else None
                )
                col_analysis["max"] = (
                    df[col].max() if not df[col].isnull().all() else None
                )

            analysis["column_analysis"][col] = col_analysis

        return analysis

    def _get_numeric_columns(self, df: pd.DataFrame) -> list[str]:
        """Get list of numeric columns."""
        return df.select_dtypes(include=[np.number]).columns.tolist()

    def _get_categorical_columns(self, df: pd.DataFrame) -> list[str]:
        """Get list of categorical columns."""
        return df.select_dtypes(include=["object", "category"]).columns.tolist()

    def _build_numeric_pipeline(self, numeric_df: pd.DataFrame) -> Pipeline:
        """Build preprocessing pipeline for numeric columns."""

        if numeric_df.empty:
            return Pipeline([("passthrough", "passthrough")])

        steps = []

        # Missing value handling
        missing_strategy = self._determine_numeric_missing_strategy(numeric_df)
        if missing_strategy != "none":
            if missing_strategy in ["mean", "median", "most_frequent"]:
                steps.append(("imputer", SimpleImputer(strategy=missing_strategy)))
            else:
                # Use adaptive imputer for more sophisticated strategies
                imputation_config = ImputationConfig(
                    numeric_strategy=missing_strategy,
                    prefer_speed=self.config.preprocessing.numeric_scaling
                    == "standard",
                )
                steps.append(("imputer", AdaptiveImputer(imputation_config)))

        # Scaling
        scaling_method = self.config.preprocessing.numeric_scaling
        if scaling_method == "standard":
            steps.append(("scaler", StandardScaler()))
        elif scaling_method == "minmax":
            steps.append(("scaler", MinMaxScaler()))
        elif scaling_method == "robust":
            steps.append(("scaler", RobustScaler()))
        # "none" means no scaling

        if not steps:
            steps.append(("passthrough", "passthrough"))

        return Pipeline(steps)

    def _build_categorical_pipeline(
        self,
        categorical_df: pd.DataFrame,
        target: pd.Series | None = None,
    ) -> Pipeline:
        """Build preprocessing pipeline for categorical columns."""

        if categorical_df.empty:
            return Pipeline([("passthrough", "passthrough")])

        steps = []

        # Missing value handling for categorical data
        if categorical_df.isnull().any().any():
            imputation_config = ImputationConfig(
                categorical_strategy="mode",  # Simple default for pipeline
                preserve_missing_patterns=False,
            )
            steps.append(("imputer", AdaptiveImputer(imputation_config)))

        # Categorical encoding
        categorical_config = self._build_categorical_config(categorical_df, target)
        steps.append(("encoder", CategoricalEncoder(categorical_config)))

        return Pipeline(steps)

    def _build_categorical_config(
        self,
        categorical_df: pd.DataFrame,
        target: pd.Series | None = None,
    ) -> CategoricalEncodingConfig:
        """Build categorical encoding configuration based on data analysis."""

        # Start with base configuration from preprocessing config
        config = CategoricalEncodingConfig(
            encoding_strategy=self.config.preprocessing.categorical_encoding_strategy,
            onehot_threshold=self.config.preprocessing.categorical_onehot_threshold,
            target_threshold=self.config.preprocessing.categorical_target_threshold,
            entity_threshold=self.config.preprocessing.categorical_entity_threshold,
        )

        # Analyze dataset characteristics to adjust configuration
        total_memory = categorical_df.memory_usage(deep=True).sum() / 1024 / 1024  # MB
        total_cardinality = sum(
            categorical_df[col].nunique() for col in categorical_df.columns
        )

        # Adjust thresholds based on dataset size and memory constraints
        if total_memory > 100:  # Large dataset, be more conservative
            config.onehot_threshold = min(config.onehot_threshold, 10)
            config.target_threshold = min(config.target_threshold, 50)
            config.prefer_speed = True

        if total_cardinality > 1000:  # High total cardinality, prefer entity embeddings
            config.target_threshold = min(config.target_threshold, 50)
            config.entity_threshold = min(config.entity_threshold, 500)

        # Adjust based on target availability
        if target is None:
            # No target available, can't use target encoding
            if config.encoding_strategy == "target":
                config.encoding_strategy = "adaptive"
            # Prefer entity embeddings for unsupervised scenarios
            config.correlation_threshold = 0.05  # Lower threshold since no target

        return config

    def _determine_numeric_missing_strategy(self, numeric_df: pd.DataFrame) -> str:
        """Determine optimal missing value strategy for numeric data."""

        if not numeric_df.isnull().any().any():
            return "none"

        total_missing_pct = (numeric_df.isnull().sum().sum() / numeric_df.size) * 100

        if total_missing_pct < 5:
            return "mean"  # Low missing rate, simple imputation
        if total_missing_pct < 20:
            return "median"  # Moderate missing rate, robust imputation
        return "knn"  # High missing rate, sophisticated imputation

    def _describe_pipeline(
        self,
        numeric_cols: list[str],
        categorical_cols: list[str],
        analysis: dict[str, Any],
    ) -> dict[str, Any]:
        """Create human-readable description of the constructed pipeline."""

        description: dict[str, Any] = {
            "dataset_summary": {
                "shape": analysis["dataset_shape"],
                "missing_percentage": f"{analysis['missing_percentage']:.1f}%",
                "numeric_features": len(numeric_cols),
                "categorical_features": len(categorical_cols),
                "memory_usage": f"{analysis['memory_usage_mb']:.1f} MB",
            },
            "numeric_processing": [],
            "categorical_processing": [],
            "recommendations": [],
        }

        # Numeric processing description
        if numeric_cols:
            missing_strategy = self._determine_numeric_missing_strategy(pd.DataFrame())
            description["numeric_processing"] = [
                f"Missing value imputation: {missing_strategy}",
                f"Scaling method: {self.config.preprocessing.numeric_scaling}",
            ]

        # Categorical processing description
        if categorical_cols:
            description["categorical_processing"] = [
                f"Encoding strategy: {self.config.preprocessing.categorical_encoding_strategy}",
                f"OneHot threshold: ≤{self.config.preprocessing.categorical_onehot_threshold} categories",
                f"Target encoding threshold: ≤{self.config.preprocessing.categorical_target_threshold} categories",
                f"Entity embedding threshold: ≤{self.config.preprocessing.categorical_entity_threshold} categories",
            ]

        # Add recommendations
        if analysis["missing_percentage"] > 20:
            description["recommendations"].append(
                "High missing rate detected. Consider investigating missing patterns.",
            )

        if analysis["memory_usage_mb"] > 500:
            description["recommendations"].append(
                "Large dataset detected. Consider using more memory-efficient encodings.",
            )

        high_cardinality_cols = [
            col
            for col in categorical_cols
            if col in analysis["column_analysis"]
            and analysis["column_analysis"][col].get("cardinality", 0) > 100
        ]
        if high_cardinality_cols:
            description["recommendations"].append(
                f"High-cardinality categorical features detected: {high_cardinality_cols}. "
                "Consider entity embeddings for better performance.",
            )

        return description

    def get_analysis_report(self) -> dict[str, Any]:
        """Get detailed analysis report of the dataset."""
        return self.analysis_report_.copy()

    def get_pipeline_description(self) -> dict[str, Any]:
        """Get human-readable description of the constructed pipeline."""
        return self.pipeline_description_.copy()


def build_adaptive_pipeline(
    df: pd.DataFrame,
    target: pd.Series | None = None,
    config: EmbeddingConfig | None = None,
    mode: str = "unsupervised",
) -> tuple[ColumnTransformer, dict[str, Any]]:
    """
    Build adaptive preprocessing pipeline for Row2Vec.

    This is the main entry point for intelligent pipeline construction.
    It analyzes the dataset and automatically selects optimal preprocessing
    strategies based on data characteristics.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset
    target : pd.Series, optional
        Target variable for supervised preprocessing
    config : EmbeddingConfig, optional
        Configuration for preprocessing. If None, intelligent defaults are used.
    mode : str
        Embedding mode ("unsupervised", "target", etc.)

    Returns
    -------
    Tuple[ColumnTransformer, Dict[str, Any]]
        Preprocessing pipeline and analysis report

    Examples
    --------
    Basic usage with automatic configuration:
    >>> pipeline, report = build_adaptive_pipeline(df)
    >>> X_processed = pipeline.fit_transform(df)

    With custom configuration:
    >>> config = EmbeddingConfig()
    >>> config.preprocessing.categorical_encoding_strategy = "entity"
    >>> pipeline, report = build_adaptive_pipeline(df, target=y, config=config)
    """

    builder = PipelineBuilder(config)
    return builder.build_preprocessing_pipeline(df, target, mode)
