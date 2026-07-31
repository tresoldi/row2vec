"""
Row2Vec: Intelligent Missing Value Imputation System

This module provides adaptive and configurable missing value imputation strategies
that automatically detect patterns and apply appropriate techniques while maintaining
simplicity for beginners and flexibility for advanced users.
"""

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pandas as pd
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.pipeline import Pipeline

if TYPE_CHECKING:
    from sklearn.base import BaseEstimator
else:
    try:
        from sklearn.base import BaseEstimator
    except ImportError:
        BaseEstimator = object

# Enable experimental features
try:
    from sklearn.experimental import enable_iterative_imputer
    from sklearn.impute import IterativeImputer

    ITERATIVE_IMPUTER_AVAILABLE = True
except ImportError:
    ITERATIVE_IMPUTER_AVAILABLE = False


@dataclass
class ImputationConfig:
    """
    Configuration for intelligent missing value imputation strategies.

    This class provides comprehensive control over how missing values are handled,
    with sensible defaults that work well for most datasets while allowing power
    users to fine-tune every aspect of the imputation process.
    """

    # Core strategy selection
    numeric_strategy: str = "adaptive"
    """Numeric imputation strategy. Options:
    - "adaptive": Automatically selects best strategy based on missing percentage
    - "mean": Mean imputation (fastest, good for <10% missing)
    - "median": Median imputation (robust to outliers, good for 10-30% missing)
    - "knn": K-nearest neighbors imputation (better for >30% missing)
    - "iterative": MICE-style iterative imputation (best quality, slowest)
    """

    categorical_strategy: str = "adaptive"
    """Categorical imputation strategy. Options:
    - "adaptive": Automatically selects best strategy based on data characteristics
    - "mode": Most frequent value imputation
    - "constant": Fill with specified constant value
    - "missing_category": Create explicit "Missing" category
    """

    # Performance vs accuracy trade-off
    prefer_speed: bool = True
    """Whether to prefer faster methods over more accurate but slower ones.
    When True, uses simpler strategies by default. When False, prefers
    more sophisticated methods even if they take longer."""

    # Missing data thresholds
    missing_threshold: float = 0.7
    """Columns with more than this fraction of missing values will be flagged.
    Conservative default of 0.7 to avoid dropping useful but sparse columns."""

    row_missing_threshold: float = 0.9
    """Rows with more than this fraction of missing values will be flagged.
    Very conservative default to avoid losing data."""

    # Advanced options for KNN imputation
    knn_neighbors: int = 5
    """Number of neighbors for KNN imputation. Should be odd to avoid ties."""

    # Pattern analysis and preservation
    preserve_missing_patterns: bool = False
    """Whether to preserve missing patterns when they might be informative.

    When True, adds binary indicator columns for originally missing values.
    This is useful when missingness itself carries information (e.g.,
    customers not providing income information might be systematically different).

    Example:
        Original: [1.0, NaN, 3.0] -> After imputation: [1.0, 2.0, 3.0]
        With preservation: adds column [False, True, False] indicating missingness
    """

    missing_indicator_suffix: str = "_was_missing"
    """Suffix for missing indicator columns when preserve_missing_patterns=True."""

    # Automatic detection and warnings
    auto_detect_patterns: bool = True
    """Whether to automatically analyze missing data patterns and adjust strategies."""

    warn_high_missingness: bool = True
    """Whether to warn users about columns/rows with high missing percentages."""

    # Constants for categorical imputation
    categorical_fill_value: str = "Missing"
    """Fill value when using 'constant' strategy for categorical data."""

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        # Validate strategies
        valid_numeric = {"adaptive", "mean", "median", "knn", "iterative"}
        if self.numeric_strategy not in valid_numeric:
            raise ValueError(f"numeric_strategy must be one of: {valid_numeric}")

        valid_categorical = {"adaptive", "mode", "constant", "missing_category"}
        if self.categorical_strategy not in valid_categorical:
            raise ValueError(
                f"categorical_strategy must be one of: {valid_categorical}"
            )

        # Validate thresholds
        if not 0 <= self.missing_threshold <= 1:
            raise ValueError("missing_threshold must be between 0 and 1")
        if not 0 <= self.row_missing_threshold <= 1:
            raise ValueError("row_missing_threshold must be between 0 and 1")

        # Validate KNN parameters
        if self.knn_neighbors < 1:
            raise ValueError("knn_neighbors must be at least 1")


class MissingPatternAnalyzer:
    """Analyzes missing data patterns to inform imputation strategy selection."""

    def __init__(self, config: ImputationConfig) -> None:
        self.config = config

    def analyze(self, df: pd.DataFrame) -> dict[str, Any]:
        """
        Analyze missing data patterns in the DataFrame.

        Args:
            df: Input DataFrame to analyze

        Returns:
            Dict containing analysis results and recommendations
        """
        analysis = {
            "total_missing": df.isnull().sum().sum(),
            "missing_percentage": (df.isnull().sum().sum() / df.size) * 100,
            "columns_with_missing": df.isnull().any().sum(),
            "column_missing_percentages": (df.isnull().sum() / len(df) * 100).to_dict(),
            "rows_with_missing": df.isnull().any(axis=1).sum(),
            "completely_missing_columns": df.columns[df.isnull().all()].tolist(),
            "high_missing_columns": [],
            "recommendations": {},
        }

        # Identify high missing columns
        for col, pct in analysis["column_missing_percentages"].items():
            if pct > self.config.missing_threshold * 100:
                analysis["high_missing_columns"].append(col)

        # Generate column-specific recommendations
        for col in df.columns:
            missing_pct = analysis["column_missing_percentages"][col]
            dtype = df[col].dtype

            if missing_pct == 0:
                continue

            recommendation = self._recommend_strategy(col, missing_pct, dtype, df[col])
            analysis["recommendations"][col] = recommendation

        return analysis

    def _recommend_strategy(
        self, column: str, missing_pct: float, dtype: Any, series: pd.Series
    ) -> dict[str, Any]:
        """Recommend imputation strategy for a specific column."""
        is_numeric = pd.api.types.is_numeric_dtype(dtype)

        recommendation: dict[str, Any] = {
            "missing_percentage": missing_pct,
            "is_numeric": is_numeric,
            "suggested_strategy": None,
            "reasoning": "",
            "alternatives": [],
        }

        if is_numeric:
            if self.config.prefer_speed:
                if missing_pct < 10:
                    recommendation["suggested_strategy"] = "mean"
                    recommendation["reasoning"] = (
                        "Low missingness, mean imputation is fast and effective"
                    )
                elif missing_pct < 30:
                    recommendation["suggested_strategy"] = "median"
                    recommendation["reasoning"] = (
                        "Moderate missingness, median is robust to outliers"
                    )
                else:
                    recommendation["suggested_strategy"] = "knn"
                    recommendation["reasoning"] = (
                        "High missingness, KNN can capture relationships"
                    )
            # Prefer accuracy over speed
            elif missing_pct < 20:
                recommendation["suggested_strategy"] = "median"
                recommendation["reasoning"] = (
                    "Median is robust and accurate for moderate missingness"
                )
            else:
                recommendation["suggested_strategy"] = "knn"
                recommendation["reasoning"] = (
                    "KNN provides better accuracy for high missingness"
                )

            recommendation["alternatives"] = ["mean", "median", "knn", "iterative"]
        else:
            # Categorical data
            unique_count = series.nunique()
            if unique_count < 10:
                recommendation["suggested_strategy"] = "mode"
                recommendation["reasoning"] = (
                    "Few categories, mode imputation works well"
                )
            elif missing_pct > 30:
                recommendation["suggested_strategy"] = "missing_category"
                recommendation["reasoning"] = (
                    "High missingness with many categories, explicit missing category"
                )
            else:
                recommendation["suggested_strategy"] = "mode"
                recommendation["reasoning"] = (
                    "Standard mode imputation for categorical data"
                )

            recommendation["alternatives"] = ["mode", "constant", "missing_category"]

        return recommendation


class AdaptiveImputer(BaseEstimator):  # type: ignore[misc]
    """
    Adaptive imputer that automatically selects and applies appropriate
    imputation strategies based on data characteristics.
    """

    def __init__(self, config: ImputationConfig) -> None:
        self.config = config
        self.analyzer = MissingPatternAnalyzer(config)
        self.analysis_report_: dict[str, Any] | None = None
        self.imputation_pipelines_: dict[str, Any] | None = None
        self.feature_names_in_: list[str] | None = None
        self.missing_indicators_: dict[str, Any] | None = None

    def fit(self, X: pd.DataFrame, y: Any = None) -> "AdaptiveImputer":
        """
        Fit the adaptive imputer to the data.

        Args:
            X: Input DataFrame with potential missing values
            y: Ignored, present for API compatibility

        Returns:
            self: Fitted imputer
        """
        X = self._validate_input(X)
        self.feature_names_in_ = X.columns.tolist()

        # Analyze missing patterns
        if self.config.auto_detect_patterns:
            self.analysis_report_ = self.analyzer.analyze(X)
            if self.config.warn_high_missingness:
                self._warn_about_high_missingness()

        # Create column-specific imputation strategies
        self.imputation_pipelines_ = self._create_imputation_pipelines(X)

        # Fit the pipelines
        for col, pipeline in self.imputation_pipelines_.items():
            if pipeline is not None:
                pipeline.fit(X[[col]])

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the data by applying imputation strategies.

        Args:
            X: Input DataFrame with potential missing values

        Returns:
            DataFrame with missing values imputed
        """
        X = self._validate_input(X)
        result = X.copy()

        # Store missing indicators if requested
        if self.config.preserve_missing_patterns:
            self.missing_indicators_ = {}
            for col in X.columns:
                if X[col].isnull().any():
                    indicator_name = f"{col}{self.config.missing_indicator_suffix}"
                    self.missing_indicators_[indicator_name] = X[col].isnull()

        # Apply column-specific imputation
        if self.imputation_pipelines_:
            for col, pipeline in self.imputation_pipelines_.items():
                if pipeline is not None and col in result.columns:
                    if result[col].isnull().any():
                        imputed_values = pipeline.transform(result[[col]])
                        result[col] = imputed_values.ravel()

        # Add missing indicators if requested
        if self.config.preserve_missing_patterns and self.missing_indicators_:
            for indicator_name, indicator_values in self.missing_indicators_.items():
                result[indicator_name] = indicator_values

        return result

    def fit_transform(
        self, X: pd.DataFrame, y: Any = None, **fit_params: Any
    ) -> pd.DataFrame:
        """Fit the imputer and transform the data in one step."""
        return self.fit(X, y).transform(X)

    def _validate_input(self, X: pd.DataFrame) -> pd.DataFrame:
        """Validate input DataFrame."""
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"Expected pandas DataFrame, got {type(X)}")

        if X.empty:
            raise ValueError("Input DataFrame is empty")

        return X

    def _create_imputation_pipelines(
        self, X: pd.DataFrame
    ) -> dict[str, Pipeline | None]:
        """Create column-specific imputation pipelines."""
        pipelines: dict[str, Pipeline | None] = {}

        for col in X.columns:
            if not X[col].isnull().any():
                pipelines[col] = None  # No imputation needed
                continue

            strategy = self._get_column_strategy(col, X[col])
            pipeline = self._create_column_pipeline(col, strategy, X[col])
            pipelines[col] = pipeline

        return pipelines

    def _get_column_strategy(self, column: str, series: pd.Series) -> str:
        """Determine the imputation strategy for a specific column."""
        if self.analysis_report_ and column in self.analysis_report_["recommendations"]:
            return str(
                self.analysis_report_["recommendations"][column]["suggested_strategy"]
            )

        # Fallback to simple rules if no analysis available
        is_numeric = pd.api.types.is_numeric_dtype(series.dtype)
        missing_pct = (series.isnull().sum() / len(series)) * 100

        if is_numeric:
            if self.config.numeric_strategy == "adaptive":
                if self.config.prefer_speed:
                    return "mean" if missing_pct < 20 else "median"
                return "median" if missing_pct < 30 else "knn"
            return self.config.numeric_strategy
        if self.config.categorical_strategy == "adaptive":
            return "mode" if missing_pct < 30 else "missing_category"
        return self.config.categorical_strategy

    def _create_column_pipeline(
        self, column: str, strategy: str, series: pd.Series
    ) -> Pipeline:
        """Create imputation pipeline for a specific column and strategy."""
        is_numeric = pd.api.types.is_numeric_dtype(series.dtype)

        if is_numeric:
            if strategy == "mean":
                imputer = SimpleImputer(strategy="mean")
            elif strategy == "median":
                imputer = SimpleImputer(strategy="median")
            elif strategy == "knn":
                imputer = KNNImputer(n_neighbors=self.config.knn_neighbors)
            elif strategy == "iterative":
                # Use global import check
                if ITERATIVE_IMPUTER_AVAILABLE:
                    imputer = IterativeImputer(random_state=1305)
                else:
                    warnings.warn(
                        "IterativeImputer not available, falling back to KNN imputation",
                        UserWarning,
                        stacklevel=2,
                    )
                    imputer = KNNImputer(n_neighbors=self.config.knn_neighbors)
            else:
                raise ValueError(f"Unknown numeric strategy: {strategy}")
        # Categorical data
        elif strategy == "mode":
            imputer = SimpleImputer(strategy="most_frequent")
        elif strategy in {"constant", "missing_category"}:
            imputer = SimpleImputer(
                strategy="constant", fill_value=self.config.categorical_fill_value
            )
        else:
            raise ValueError(f"Unknown categorical strategy: {strategy}")

        return Pipeline([("imputer", imputer)])

    def _warn_about_high_missingness(self) -> None:
        """Warn users about potentially problematic missing data patterns."""
        if not self.analysis_report_:
            return

        # Warn about high missing columns
        high_missing = self.analysis_report_["high_missing_columns"]
        if high_missing:
            missing_info = []
            for col in high_missing:
                pct = self.analysis_report_["column_missing_percentages"][col]
                missing_info.append(f"{col} ({pct:.1f}%)")

            warnings.warn(
                f"High missingness detected in columns: {', '.join(missing_info)}. "
                f"Consider investigating these patterns or setting preserve_missing_patterns=True "
                f"if missingness is informative.",
                UserWarning,
                stacklevel=2,
            )

        # Warn about completely missing columns
        completely_missing = self.analysis_report_["completely_missing_columns"]
        if completely_missing:
            warnings.warn(
                f"Columns with all missing values will be dropped: {completely_missing}",
                UserWarning,
                stacklevel=2,
            )

    def get_imputation_report(self) -> dict[str, Any]:
        """
        Get detailed report about the imputation process.

        Returns:
            Dict containing analysis and imputation details
        """
        if self.analysis_report_ is None:
            return {"error": "No analysis performed. Call fit() first."}

        report = self.analysis_report_.copy()

        # Add applied strategies
        applied_strategies = {}
        if self.imputation_pipelines_:
            for col, pipeline in self.imputation_pipelines_.items():
                if pipeline is not None:
                    strategy_name = type(pipeline.named_steps["imputer"]).__name__
                    applied_strategies[col] = strategy_name

        report["applied_strategies"] = applied_strategies
        report["missing_indicators_added"] = (
            list(self.missing_indicators_.keys()) if self.missing_indicators_ else []
        )

        return report


def create_imputation_pipeline(
    config: ImputationConfig | None = None,
) -> AdaptiveImputer:
    """
    Create an adaptive imputation pipeline with intelligent defaults.

    This is the main entry point for users who want simple, automatic
    missing value handling without needing to understand the complexity.

    Args:
        config: Optional ImputationConfig. If None, uses conservative defaults.

    Returns:
        Configured AdaptiveImputer ready for use

    Example:
        >>> imputer = create_imputation_pipeline()
        >>> df_imputed = imputer.fit_transform(df_with_missing)
    """
    if config is None:
        config = ImputationConfig()

    return AdaptiveImputer(config)
