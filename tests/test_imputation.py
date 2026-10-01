"""
Tests for the missing-value imputation system.

Exercises the pattern analyser, the three headline imputation configurations
(default/adaptive, speed-first, accuracy-first), the edge cases that break naive
imputers, and the scikit-learn fit/transform contract.
"""

import numpy as np
import pandas as pd
import pytest

from row2vec import AdaptiveImputer, ImputationConfig, MissingPatternAnalyzer


@pytest.fixture
def missing_data() -> pd.DataFrame:
    """A frame with random, correlated, and pattern-based missingness."""
    np.random.seed(42)

    n_rows = 1000
    df = pd.DataFrame(
        {
            "numeric_1": np.random.normal(50, 15, n_rows),
            "numeric_2": np.random.exponential(2, n_rows),
            "categorical_1": np.random.choice(["A", "B", "C", "D"], n_rows),
            "categorical_2": np.random.choice(["X", "Y", "Z"], n_rows, p=[0.5, 0.3, 0.2]),
            "binary": np.random.choice([0, 1], n_rows),
            "text": [f"text_{i}" for i in range(n_rows)],
        }
    )

    # Missing completely at random.
    df.loc[np.random.random(n_rows) < 0.1, "numeric_1"] = np.nan

    # Missing correlated with the value itself (high values drop out more often).
    high_value_mask = (df["numeric_2"] > df["numeric_2"].quantile(0.8)) & (
        np.random.random(n_rows) < 0.3
    )
    df.loc[high_value_mask, "numeric_2"] = np.nan

    df.loc[np.random.random(n_rows) < 0.05, "categorical_1"] = np.nan

    # Missing conditioned on another column.
    z_mask = (df["categorical_2"] == "Z") & (np.random.random(n_rows) < 0.4)
    df.loc[z_mask, "binary"] = np.nan

    return df


def test_pattern_analysis_reports_missingness(missing_data: pd.DataFrame) -> None:
    """The analyser counts what is missing and recommends a strategy per column."""
    analysis = MissingPatternAnalyzer(ImputationConfig()).analyze(missing_data)

    expected_missing = int(missing_data.isna().sum().sum())
    assert analysis["total_missing"] == expected_missing
    assert 0 < analysis["missing_percentage"] < 100
    assert analysis["columns_with_missing"] > 0

    columns_with_missing = {col for col in missing_data.columns if missing_data[col].isna().any()}
    recommended = {col for col, rec in analysis["recommendations"].items() if rec}
    assert columns_with_missing <= recommended


@pytest.mark.parametrize(
    "config",
    [
        pytest.param(ImputationConfig(), id="default"),
        pytest.param(
            ImputationConfig(
                numeric_strategy="mean",
                categorical_strategy="mode",
                prefer_speed=True,
            ),
            id="speed-first",
        ),
        pytest.param(
            ImputationConfig(
                numeric_strategy="knn",
                categorical_strategy="mode",
                prefer_speed=False,
                knn_neighbors=10,
            ),
            id="accuracy-first",
        ),
    ],
)
def test_imputation_fills_every_gap(missing_data: pd.DataFrame, config: ImputationConfig) -> None:
    """Whatever the strategy, nothing missing survives and the shape is kept."""
    assert missing_data.isna().sum().sum() > 0  # guard: the fixture has gaps

    imputed = AdaptiveImputer(config).fit_transform(missing_data)

    assert imputed.isna().sum().sum() == 0
    assert len(imputed) == len(missing_data)


def test_preserve_missing_patterns_adds_indicator_columns(
    missing_data: pd.DataFrame,
) -> None:
    """`preserve_missing_patterns` keeps the fact that a value was missing."""
    config = ImputationConfig(
        numeric_strategy="knn",
        categorical_strategy="mode",
        preserve_missing_patterns=True,
        knn_neighbors=10,
    )
    imputed = AdaptiveImputer(config).fit_transform(missing_data)

    indicators = [c for c in imputed.columns if c.endswith("_was_missing")]
    assert indicators, "expected at least one _was_missing indicator column"


def test_edge_cases_do_not_crash() -> None:
    """All-missing, near-constant, and high-cardinality columns are survivable."""
    n = 100
    edge_df = pd.DataFrame(
        {
            "all_missing": [np.nan] * n,
            "single_value": [1.0] * (n - 1) + [np.nan],
            "binary_missing": [0, 1, np.nan] * (n // 3) + [0] * (n % 3),
            "high_cardinality": [f"cat_{i}" for i in range(n - 1)] + [np.nan],
        }
    )

    imputed = AdaptiveImputer(ImputationConfig()).fit_transform(edge_df)

    assert len(imputed) == n
    # A column with nothing observed cannot be imputed, and is left alone
    # rather than invented. Everything else must come back complete.
    fillable = [c for c in imputed.columns if c != "all_missing"]
    assert imputed[fillable].isna().sum().sum() == 0
    # The single-valued column's one gap must be filled with that value.
    assert imputed["single_value"].nunique() == 1


def test_fit_transform_matches_separate_fit_then_transform(
    missing_data: pd.DataFrame,
) -> None:
    """The scikit-learn contract: fit().transform() == fit_transform()."""
    separate = AdaptiveImputer(ImputationConfig())
    separate.fit(missing_data)
    from_separate = separate.transform(missing_data)

    combined = AdaptiveImputer(ImputationConfig()).fit_transform(missing_data)

    assert list(from_separate.columns) == list(combined.columns)
    # Values, not just shape: an imputer that learned different statistics on
    # the two paths would pass a shape comparison unchanged.
    pd.testing.assert_frame_equal(from_separate, combined)


def test_transform_applies_to_unseen_data(missing_data: pd.DataFrame) -> None:
    """A fitted imputer fills genuinely unseen rows, with fit-time statistics.

    The previous version sampled its "unseen" data out of the training frame,
    so it shared every column and every missingness pattern - which is why it
    could not notice that columns clean at fit time were never imputed at all.
    """
    imputer = AdaptiveImputer(ImputationConfig())
    imputer.fit(missing_data)

    numeric_cols = missing_data.select_dtypes(include=[np.number]).columns
    assert len(numeric_cols) > 0

    # Fresh rows, with holes in places the training frame did not have them.
    unseen = missing_data.head(5).copy()
    unseen.index = pd.Index([f"new{i}" for i in range(5)])
    for col in numeric_cols:
        unseen[col] = np.nan

    imputed = imputer.transform(unseen)

    assert len(imputed) == 5
    assert imputed[numeric_cols].isna().sum().sum() == 0, (
        "columns that happened to be complete at fit time were left unimputed"
    )
    # Each filled column must be constant: every row got the same learned value.
    for col in numeric_cols:
        assert imputed[col].nunique() == 1
