"""Unit tests for the categorical encoding strategies.

`CategoricalEncoder` picks a strategy per column from that column's
characteristics. These tests pin down both the choice and the encoders it
chooses between.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from row2vec import (
    CategoricalAnalyzer,
    CategoricalEncoder,
    CategoricalEncodingConfig,
    TargetEncoder,
)


@pytest.fixture
def mixed_frame() -> pd.DataFrame:
    """Low-, medium-, and high-cardinality categoricals beside a numeric column."""
    rng = np.random.default_rng(1305)
    n = 200
    return pd.DataFrame(
        {
            "low_card": rng.choice(["a", "b", "c"], n),
            "mid_card": rng.choice([f"m{i}" for i in range(15)], n),
            "high_card": [f"h{i}" for i in range(n)],
            "numeric": rng.normal(0, 1, n),
        }
    )


class TestAnalyzer:
    def test_reports_cardinality_and_a_strategy(self, mixed_frame: pd.DataFrame) -> None:
        analyzer = CategoricalAnalyzer(CategoricalEncodingConfig())

        analysis = analyzer.analyze_column(mixed_frame["low_card"])

        assert analysis["cardinality"] == 3
        assert analysis["recommended_strategy"]

    def test_cardinality_drives_the_recommendation(self, mixed_frame: pd.DataFrame) -> None:
        """A 3-value column and a 200-value column should not be encoded alike."""
        analyzer = CategoricalAnalyzer(CategoricalEncodingConfig())

        low = analyzer.analyze_column(mixed_frame["low_card"])
        high = analyzer.analyze_column(mixed_frame["high_card"])

        assert low["cardinality"] < high["cardinality"]
        assert low["recommended_strategy"] != high["recommended_strategy"]


class TestTargetEncoder:
    def test_encodes_categories_to_target_means(self) -> None:
        series = pd.Series(["a", "a", "b", "b", "a", "b"] * 10)
        target = pd.Series([1.0, 1.0, 0.0, 0.0, 1.0, 0.0] * 10)

        encoder = TargetEncoder(CategoricalEncodingConfig())
        encoded = encoder.fit_transform(series, target)

        assert len(encoded) == len(series)
        assert pd.api.types.is_numeric_dtype(encoded)
        # 'a' always co-occurs with 1.0 and 'b' with 0.0, so 'a' must score higher.
        assert encoded[series == "a"].mean() > encoded[series == "b"].mean()

    def test_unseen_category_falls_back_to_the_global_mean(self) -> None:
        series = pd.Series(["a", "b"] * 20)
        target = pd.Series([1.0, 0.0] * 20)

        encoder = TargetEncoder(CategoricalEncodingConfig())
        encoder.fit_transform(series, target)
        encoded = encoder.transform(pd.Series(["a", "b", "never_seen"]))

        assert not encoded.isna().any()
        assert encoded.iloc[2] == pytest.approx(target.mean(), abs=0.2)


class TestCategoricalEncoder:
    @pytest.mark.parametrize("strategy", ["onehot", "ordinal"])
    def test_explicit_strategies_produce_numeric_output(
        self, mixed_frame: pd.DataFrame, strategy: str
    ) -> None:
        encoder = CategoricalEncoder(CategoricalEncodingConfig(encoding_strategy=strategy))

        encoded = encoder.fit_transform(mixed_frame[["low_card"]])

        assert len(encoded) == len(mixed_frame)
        assert np.asarray(encoded).dtype.kind in "fiu"

    def test_onehot_widens_and_ordinal_does_not(self, mixed_frame: pd.DataFrame) -> None:
        """One-hot adds a column per category; ordinal keeps one column."""
        onehot = CategoricalEncoder(
            CategoricalEncodingConfig(encoding_strategy="onehot")
        ).fit_transform(mixed_frame[["low_card"]])
        ordinal = CategoricalEncoder(
            CategoricalEncodingConfig(encoding_strategy="ordinal")
        ).fit_transform(mixed_frame[["low_card"]])

        assert np.asarray(onehot).shape[1] > np.asarray(ordinal).shape[1]

    def test_transform_after_fit_is_available(self, mixed_frame: pd.DataFrame) -> None:
        encoder = CategoricalEncoder(CategoricalEncodingConfig(encoding_strategy="ordinal"))
        encoder.fit(mixed_frame[["low_card"]])

        encoded = encoder.transform(mixed_frame[["low_card"]].head(10))

        assert len(encoded) == 10

    def test_analysis_report_is_available_after_fit(self, mixed_frame: pd.DataFrame) -> None:
        encoder = CategoricalEncoder(CategoricalEncodingConfig(encoding_strategy="ordinal"))
        encoder.fit(mixed_frame[["low_card", "mid_card"]])

        report = encoder.get_analysis_report()

        assert set(report) >= {"low_card", "mid_card"}


class TestConfigValidation:
    def test_thresholds_are_ordered(self) -> None:
        """The cardinality thresholds must increase, or strategy choice is undefined."""
        config = CategoricalEncodingConfig()

        assert config.onehot_threshold < config.target_threshold
        assert config.target_threshold <= config.entity_threshold
