"""Tests for automatic embedding-dimension selection.

This module sat at roughly 45% coverage, and the parts that were untested were
the parts that decided the answer: seven blanket handlers that returned the
midpoint of the candidate list with a score of 0.5 whenever anything went
wrong, which is indistinguishable from a real recommendation.

These tests pin the behaviour that replaced them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from row2vec import EmbeddingConfig
from row2vec.auto_dimension import AutoDimensionSelector, auto_select_dimension


@pytest.fixture
def low_rank_frame() -> pd.DataFrame:
    """Twelve columns whose variance really lives in three directions."""
    rng = np.random.default_rng(1305)
    n = 200
    latent = rng.normal(size=(n, 3))
    mixing = rng.normal(size=(3, 12))
    observed = latent @ mixing + rng.normal(scale=0.01, size=(n, 12))
    return pd.DataFrame(observed, columns=[f"f{i}" for i in range(12)])


class TestVarianceKnee:
    """The knee construction that replaced argmax of a second difference."""

    def test_finds_the_true_rank_of_a_low_rank_curve(self) -> None:
        """Three components explain everything; the knee must say so."""
        cumulative = np.array([0.40, 0.70, 0.99, 0.995, 0.997, 0.999, 1.0])

        assert AutoDimensionSelector._variance_knee(cumulative) == 3

    def test_never_exceeds_what_reaches_95_percent(self) -> None:
        """The answer is capped by the components 95% of variance needs."""
        cumulative = np.array([0.50, 0.96, 0.97, 0.98, 0.99, 1.0])

        assert AutoDimensionSelector._variance_knee(cumulative) <= 2

    def test_handles_degenerate_curves(self) -> None:
        """One and two component curves must not raise."""
        assert AutoDimensionSelector._variance_knee(np.array([1.0])) == 1
        assert AutoDimensionSelector._variance_knee(np.array([0.6, 1.0])) == 2
        assert AutoDimensionSelector._variance_knee(np.array([])) == 1

    def test_is_not_constant_across_different_data(self) -> None:
        """The previous elbow returned the same index whatever it was given."""
        early_knee = np.array([0.90, 0.95, 0.97, 0.98, 0.99, 1.0])
        late_knee = np.array([0.20, 0.35, 0.50, 0.65, 0.80, 1.0])

        assert AutoDimensionSelector._variance_knee(
            early_knee
        ) != AutoDimensionSelector._variance_knee(late_knee)


class TestPcaVarianceMethod:
    """PCA-based selection over the full component range."""

    def test_recovers_the_latent_dimension(self, low_rank_frame: pd.DataFrame) -> None:
        """Data built from three latent factors should not need many more."""
        selector = AutoDimensionSelector(methods=["pca_variance"], verbose=False)
        result = selector._pca_variance_method(low_rank_frame, [2, 3, 4, 8, 12])

        assert result["recommended_dim"] in (2, 3, 4)
        assert result["score"] > 0.9

    def test_can_recommend_beyond_the_candidate_count(self) -> None:
        """PCA used to be fitted with n_components = len(candidate_dims).

        With five candidates it could never see past five components, so a
        dataset needing eight was capped at five regardless.
        """
        rng = np.random.default_rng(7)
        frame = pd.DataFrame(rng.normal(size=(200, 20)), columns=[f"f{i}" for i in range(20)])

        selector = AutoDimensionSelector(methods=["pca_variance"], verbose=False)
        result = selector._pca_variance_method(frame, [2, 4, 8, 16, 20])

        # Isotropic noise needs almost every component to reach 95%.
        assert result["recommended_dim"] > 5

    def test_abstains_without_numeric_columns(self) -> None:
        """No numeric data is a reason to abstain, not to guess the midpoint."""
        frame = pd.DataFrame({"a": ["x", "y", "z"], "b": ["p", "q", "r"]})

        selector = AutoDimensionSelector(methods=["pca_variance"], verbose=False)
        result = selector._pca_variance_method(frame, [2, 4, 8])

        assert result["recommended_dim"] is None
        assert result["score"] == 0.0
        assert "numeric" in result["failed_reason"]


class TestAbstention:
    """A method that cannot answer must not vote."""

    def test_failed_result_casts_no_vote(self) -> None:
        """The midpoint fallback happens once, visibly, not per method."""
        selector = AutoDimensionSelector(verbose=False)
        failure = selector._failed_result("pca_variance", "no numeric columns")

        assert failure["recommended_dim"] is None
        assert failure["score"] == 0.0
        assert failure["failed_reason"] == "no numeric columns"

    def test_all_methods_failing_still_returns_a_dimension(self) -> None:
        """The caller gets an answer, but only from the combine step."""
        selector = AutoDimensionSelector(verbose=False)
        all_failed = {
            "pca_variance": selector._failed_result("pca_variance", "nope"),
            "intrinsic_dim": selector._failed_result("intrinsic_dim", "nope"),
        }

        assert selector._combine_recommendations(all_failed, [2, 4, 8]) == 4

    def test_a_successful_method_outvotes_the_fallback(self) -> None:
        """One real recommendation must decide the outcome."""
        selector = AutoDimensionSelector(verbose=False)
        results = {
            "pca_variance": {"recommended_dim": 8, "score": 1.0},
            "intrinsic_dim": selector._failed_result("intrinsic_dim", "nope"),
        }

        assert selector._combine_recommendations(results, [2, 4, 8]) == 8


class TestSeeding:
    """The selector must honour the seed it is given."""

    def test_random_state_is_configurable(self) -> None:
        """It was hardcoded to 1305 in four places, ignoring config.seed."""
        assert AutoDimensionSelector(random_state=99).random_state == 99
        assert AutoDimensionSelector().random_state == 1305


class TestEndToEnd:
    """The public entry point."""

    def test_returns_a_candidate_dimension_and_its_reasoning(
        self, low_rank_frame: pd.DataFrame
    ) -> None:
        """auto_select_dimension answers with a dimension and its metadata."""
        config = EmbeddingConfig(mode="pca", embedding_dim=5)
        dimension, metadata = auto_select_dimension(
            low_rank_frame,
            config,
            methods=["pca_variance", "heuristic_rules"],
            verbose=False,
        )

        assert dimension in metadata["candidate_dimensions"]
        assert dimension >= 2
        assert set(metadata["method_results"]) == {"pca_variance", "heuristic_rules"}
