"""Tests for the individual dimension-selection methods and how they combine."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from row2vec import EmbeddingConfig
from row2vec.auto_dimension import AutoDimensionSelector, auto_select_dimension


@pytest.fixture
def low_rank_frame() -> pd.DataFrame:
    rng = np.random.default_rng(1305)
    latent = rng.normal(size=(150, 3))
    observed = latent @ rng.normal(size=(3, 10)) + rng.normal(scale=0.01, size=(150, 10))
    return pd.DataFrame(observed, columns=[f"f{i}" for i in range(10)])


@pytest.fixture
def selector() -> AutoDimensionSelector:
    return AutoDimensionSelector(random_state=1305)


CANDIDATES = [2, 3, 4, 5]


class TestIntrinsicDimensionality:
    def test_recommends_a_candidate(
        self, selector: AutoDimensionSelector, low_rank_frame: pd.DataFrame
    ) -> None:
        result = selector._intrinsic_dimensionality_method(low_rank_frame, CANDIDATES)

        assert result["recommended_dim"] in CANDIDATES
        assert result["score"] > 0
        assert len(result["reconstruction_errors"]) == len(result["test_dimensions"])

    def test_abstains_on_too_few_rows(self, selector: AutoDimensionSelector) -> None:
        tiny = pd.DataFrame(np.random.default_rng(0).normal(size=(10, 4)))

        result = selector._intrinsic_dimensionality_method(tiny, CANDIDATES)

        assert result["recommended_dim"] is None
        assert "20 rows" in result["failed_reason"]

    def test_abstains_without_numeric_columns(self, selector: AutoDimensionSelector) -> None:
        text = pd.DataFrame({"a": ["x", "y"] * 30})

        assert (
            selector._intrinsic_dimensionality_method(text, CANDIDATES)["recommended_dim"] is None
        )


class TestPerformanceAndClustering:
    def test_clustering_quality_picks_a_dimension(
        self, selector: AutoDimensionSelector, low_rank_frame: pd.DataFrame
    ) -> None:
        config = EmbeddingConfig(mode="pca", embedding_dim=2)

        result = selector._clustering_quality_method(low_rank_frame, config, [2, 3])

        assert result["recommended_dim"] in (2, 3)
        assert result["evaluation_type"] == "clustering_quality"

    def test_performance_without_target_falls_back_to_clustering(
        self, selector: AutoDimensionSelector, low_rank_frame: pd.DataFrame
    ) -> None:
        config = EmbeddingConfig(mode="pca", embedding_dim=2)

        result = selector._performance_based_method(low_rank_frame, config, [2, 3])

        assert result.get("evaluation_type") == "clustering_quality"

    def test_supervised_performance_uses_the_target(self, selector: AutoDimensionSelector) -> None:
        rng = np.random.default_rng(7)
        frame = pd.DataFrame(rng.normal(size=(120, 5)), columns=list("abcde"))
        frame["label"] = np.where(frame["a"] + frame["b"] > 0, "up", "down")
        config = EmbeddingConfig(mode="pca", embedding_dim=2)

        result = selector._performance_based_method(frame, config, [2, 3], target_column="label")

        assert result["evaluation_type"] == "supervised_classification"
        assert result["recommended_dim"] in (2, 3)
        assert result["score"] > 0.5


class TestHeuristicsAndCombination:
    def test_heuristic_rules_scores_agreement(self, selector: AutoDimensionSelector) -> None:
        frame = pd.DataFrame(np.zeros((400, 60)))

        result = selector._heuristic_rules_method(frame, [2, 4, 8, 16])

        assert result["recommended_dim"] in (2, 4, 8, 16)
        assert 0 <= result["score"] <= 1
        assert set(result["heuristics"]) == {
            "sqrt_features",
            "log_samples",
            "features_ratio",
            "sample_ratio",
        }

    def test_combination_ignores_abstentions(self, selector: AutoDimensionSelector) -> None:
        results = {
            "pca_variance": {"recommended_dim": 3, "score": 0.9},
            "intrinsic_dim": {"recommended_dim": None, "score": 0.0},
        }

        assert selector._combine_recommendations(results, CANDIDATES) == 3

    def test_combination_with_no_votes_takes_the_middle(
        self, selector: AutoDimensionSelector
    ) -> None:
        results = {"pca_variance": {"recommended_dim": None, "score": 0.0}}

        assert selector._combine_recommendations(results, CANDIDATES) == CANDIDATES[2]

    def test_dimension_scores_fall_off_with_distance(self, selector: AutoDimensionSelector) -> None:
        results = {"heuristic_rules": {"recommended_dim": 3, "score": 1.0}}

        scores = selector._calculate_dimension_scores(results, [2, 3, 4, 8])

        assert scores[3] > scores[2] == scores[4] > scores[8] == 0.0


def test_auto_select_dimension_end_to_end(low_rank_frame: pd.DataFrame) -> None:
    config = EmbeddingConfig(mode="pca", embedding_dim=2)

    dim, metadata = auto_select_dimension(
        low_rank_frame, config, methods=["pca_variance", "heuristic_rules"], random_state=1305
    )

    assert 2 <= dim <= low_rank_frame.shape[1]
    assert isinstance(metadata, dict)
