"""Tests for the categorical encoding pipeline.

This file previously consisted of three print-driven scripts that wrapped their
own bodies in `try/except Exception: print(...)`, so a total failure of the
encoder was reported as a pass. It carried three assertions across 162 lines.
Each test below now asserts the behaviour the corresponding script printed.
"""

import numpy as np
import pandas as pd
import pytest

from row2vec import (
    CategoricalAnalyzer,
    CategoricalEncoder,
    CategoricalEncodingConfig,
    EmbeddingConfig,
    build_adaptive_pipeline,
)


@pytest.fixture
def mixed_frame() -> pd.DataFrame:
    """A frame with low, medium and high cardinality categoricals."""
    rng = np.random.default_rng(42)
    n = 100
    return pd.DataFrame(
        {
            "color": rng.choice(["red", "blue", "green"], n),
            "size": rng.choice(["S", "M", "L", "XL"], n),
            "brand": rng.choice([f"brand_{i}" for i in range(10)], n),
            "price": rng.normal(100, 20, n),
            "rating": rng.normal(4.0, 0.5, n),
        }
    )


class TestStrategySelection:
    """What the analyzer recommends, and why."""

    def test_low_cardinality_gets_onehot(self, mixed_frame: pd.DataFrame) -> None:
        """Three distinct values is well under the one-hot threshold."""
        analyzer = CategoricalAnalyzer(CategoricalEncodingConfig())
        analysis = analyzer.analyze_column(mixed_frame["color"])

        assert analysis["cardinality"] == 3
        assert analysis["recommended_strategy"] == "onehot"

    def test_target_encoding_is_never_chosen_without_a_target(self) -> None:
        """Without a target, target encoding degrades to raw ordinal codes.

        Those codes are unbounded, and before 0.3.0 they went on to dominate
        the embedding. The analyzer must not select a strategy it cannot carry
        out.
        """
        rng = np.random.default_rng(0)
        # 150 levels: the band that used to return "target" unconditionally.
        column = pd.Series([f"id_{i % 150:04d}" for i in range(600)])

        analyzer = CategoricalAnalyzer(CategoricalEncodingConfig())
        assert analyzer.analyze_column(column)["recommended_strategy"] != "target"

        # With a numeric target it becomes available again.
        target = pd.Series(rng.normal(size=600))
        with_target = analyzer.analyze_column(column, target)["recommended_strategy"]
        assert with_target in {"target", "entity", "onehot"}

    def test_constant_column_is_dropped(self) -> None:
        """A column with one value carries no information."""
        analyzer = CategoricalAnalyzer(CategoricalEncodingConfig())
        analysis = analyzer.analyze_column(pd.Series(["same"] * 50))

        assert analysis["recommended_strategy"] == "drop"


class TestEncoderOutput:
    """What the encoder actually produces."""

    def test_onehot_output_is_indicator_columns(self) -> None:
        """One column per category, exactly one of them set per row."""
        frame = pd.DataFrame({"color": ["red", "blue", "green", "red"]})
        encoder = CategoricalEncoder(CategoricalEncodingConfig(encoding_strategy="onehot"))

        encoded = encoder.fit_transform(frame)

        assert encoded.shape == (4, 3)
        np.testing.assert_array_equal(encoded.to_numpy().sum(axis=1), np.ones(4))
        # Rows 0 and 3 are both "red", so they must encode identically.
        np.testing.assert_array_equal(encoded.to_numpy()[0], encoded.to_numpy()[3])

    def test_transform_is_stable_across_calls(self, mixed_frame: pd.DataFrame) -> None:
        """The same rows must encode the same way every time."""
        categorical = mixed_frame[["color", "size", "brand"]]
        encoder = CategoricalEncoder(CategoricalEncodingConfig()).fit(categorical)

        first = encoder.transform(categorical)
        second = encoder.transform(categorical)

        pd.testing.assert_frame_equal(first, second)

    def test_output_width_does_not_depend_on_the_batch(self, mixed_frame: pd.DataFrame) -> None:
        """A batch missing some categories must still encode to full width."""
        categorical = mixed_frame[["color", "size", "brand"]]
        encoder = CategoricalEncoder(CategoricalEncodingConfig()).fit(categorical)

        full_width = encoder.transform(categorical).shape[1]
        single_row_width = encoder.transform(categorical.head(1)).shape[1]

        assert single_row_width == full_width


class TestPipelineIntegration:
    """The encoder as the preprocessing pipeline uses it."""

    def test_pipeline_produces_finite_numeric_features(self, mixed_frame: pd.DataFrame) -> None:
        """Whatever the strategy, the model must receive usable numbers."""
        pipeline, report = build_adaptive_pipeline(mixed_frame, config=EmbeddingConfig())
        transformed = np.asarray(pipeline.fit_transform(mixed_frame), dtype=float)

        assert report["dataset_shape"] == mixed_frame.shape
        assert transformed.shape[0] == len(mixed_frame)
        assert np.isfinite(transformed).all()

    def test_no_encoded_feature_dominates(self, mixed_frame: pd.DataFrame) -> None:
        """Encoded columns must be on the same scale as the numeric ones.

        Without this the embedding is decided by whichever column happens to
        have the widest range, which for ordinal codes is the identifier.
        """
        pipeline, _report = build_adaptive_pipeline(mixed_frame, config=EmbeddingConfig())
        transformed = np.asarray(pipeline.fit_transform(mixed_frame), dtype=float)

        stds = transformed.std(axis=0)
        informative = stds[stds > 0]
        assert informative.size > 0
        assert informative.max() < 10.0

    def test_missing_categories_are_imputed(self) -> None:
        """NaN must not reach the model."""
        frame = pd.DataFrame(
            {
                "category_with_missing": ["A", "B", np.nan, "A", "C", np.nan, "B"],
                "complete_category": ["X", "Y", "Z", "X", "Y", "Z", "X"],
                "numeric": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            }
        )

        pipeline, _report = build_adaptive_pipeline(frame, config=EmbeddingConfig())
        transformed = np.asarray(pipeline.fit_transform(frame), dtype=float)

        assert transformed.shape[0] == len(frame)
        assert not np.isnan(transformed).any()


class TestConfigurationRoundTrip:
    """Encoding settings must survive serialization."""

    def test_preprocessing_settings_round_trip(self) -> None:
        """to_dict/from_dict must preserve every threshold."""
        config = EmbeddingConfig()
        config.preprocessing.categorical_encoding_strategy = "adaptive"
        config.preprocessing.categorical_onehot_threshold = 15
        config.preprocessing.categorical_target_threshold = 50

        restored = EmbeddingConfig.from_dict(config.to_dict())

        assert restored.preprocessing.categorical_encoding_strategy == "adaptive"
        assert restored.preprocessing.categorical_onehot_threshold == 15
        assert restored.preprocessing.categorical_target_threshold == 50
