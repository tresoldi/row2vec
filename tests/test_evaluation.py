"""compare_modes: a fair, held-out comparison of embedding modes."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

import row2vec
from row2vec import evaluation
from row2vec._backend import NeuralBackendMissing


def cell(report: pd.DataFrame, mode: str, column: str) -> Any:
    """One cell of the report, typed loosely (pandas-stubs types ``.loc`` as a union)."""
    return report.loc[mode, column]


@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    return row2vec.generate_synthetic_data(160)


class TestReport:
    def test_shape_and_baseline(self, df: pd.DataFrame) -> None:
        report = row2vec.compare_modes(df, modes=["pca"])

        assert list(report.index) == ["pca", "baseline"]
        assert list(report.columns) == evaluation.COLUMNS
        assert cell(report, "pca", "status") == "ok"
        assert cell(report, "baseline", "status") == "baseline"
        assert cell(report, "baseline", "trustworthiness") == 1.0

    def test_trustworthiness_is_a_probability(self, df: pd.DataFrame) -> None:
        report = row2vec.compare_modes(df, modes=["pca"], embedding_dim=2)
        assert 0.5 < cell(report, "pca", "trustworthiness") <= 1.0

    def test_no_downstream_score_without_a_target(self, df: pd.DataFrame) -> None:
        report = row2vec.compare_modes(df, modes=["pca"])
        assert np.isnan(cell(report, "pca", "downstream_score"))

    def test_categorical_target_is_scored_by_accuracy(self, df: pd.DataFrame) -> None:
        report = row2vec.compare_modes(df, target="Country", modes=["pca"])
        assert cell(report, "pca", "downstream_metric") == "accuracy"
        assert 0.0 <= cell(report, "pca", "downstream_score") <= 1.0
        assert cell(report, "baseline", "downstream_metric") == "accuracy"

    def test_numeric_target_is_scored_by_r2(self, df: pd.DataFrame) -> None:
        report = row2vec.compare_modes(df, target="Sales", modes=["pca"])
        assert cell(report, "pca", "downstream_metric") == "r2"

    def test_is_reproducible(self, df: pd.DataFrame) -> None:
        first = row2vec.compare_modes(df, target="Country", modes=["pca"], seed=7)
        second = row2vec.compare_modes(df, target="Country", modes=["pca"], seed=7)
        pd.testing.assert_frame_equal(
            first.drop(columns="fit_seconds"), second.drop(columns="fit_seconds")
        )

    def test_does_not_modify_the_input(self, df: pd.DataFrame) -> None:
        before = df.copy()
        row2vec.compare_modes(df, target="Country", modes=["pca"])
        pd.testing.assert_frame_equal(df, before)


class TestHeldOutAndNoLeakage:
    def test_target_is_not_a_feature_for_other_modes(
        self, df: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: dict[str, list[str]] = {}
        real = evaluation.learn_embedding_with_model

        def spy(frame: pd.DataFrame, **kwargs: Any) -> Any:
            seen[kwargs["mode"]] = list(frame.columns)
            return real(frame, **kwargs)

        monkeypatch.setattr(evaluation, "learn_embedding_with_model", spy)
        row2vec.compare_modes(df, target="Country", modes=["pca"])

        assert "Country" not in seen["pca"]

    def test_fit_sees_only_training_rows(
        self, df: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        sizes: list[int] = []
        real = evaluation.learn_embedding_with_model

        def spy(frame: pd.DataFrame, **kwargs: Any) -> Any:
            sizes.append(len(frame))
            return real(frame, **kwargs)

        monkeypatch.setattr(evaluation, "learn_embedding_with_model", spy)
        row2vec.compare_modes(df, modes=["pca"], test_size=0.25)

        assert sizes == [len(df) - round(len(df) * 0.25)]

    def test_tsne_is_capped_to_a_sample_and_says_so(self, df: pd.DataFrame) -> None:
        report = row2vec.compare_modes(df, modes=["tsne"], perplexity=10, tsne_max_rows=100)
        assert "a sample of 100 rows" in cell(report, "tsne", "note")

    def test_tsne_cannot_be_held_out_and_says_so(self, df: pd.DataFrame) -> None:
        report = row2vec.compare_modes(df, target="Country", modes=["tsne"], perplexity=10)

        assert cell(report, "tsne", "status") == "ok"
        assert np.isnan(cell(report, "tsne", "downstream_score"))
        assert "out-of-sample" in cell(report, "tsne", "note")
        assert cell(report, "tsne", "trustworthiness") > 0.5


class TestFailureReporting:
    def test_missing_tensorflow_is_unavailable_not_an_error(
        self, df: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        real = evaluation.learn_embedding_with_model

        def no_tf(frame: pd.DataFrame, **kwargs: Any) -> Any:
            if kwargs["mode"] == "unsupervised":
                raise NeuralBackendMissing("Neural training needs TensorFlow")
            return real(frame, **kwargs)

        monkeypatch.setattr(evaluation, "learn_embedding_with_model", no_tf)
        report = row2vec.compare_modes(df, modes=["unsupervised", "pca"])

        assert cell(report, "unsupervised", "status") == "unavailable"
        assert "TensorFlow" in cell(report, "unsupervised", "note")
        assert cell(report, "pca", "status") == "ok"

    def test_one_broken_mode_does_not_hide_the_others(
        self, df: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        real = evaluation.learn_embedding_with_model

        def flaky(frame: pd.DataFrame, **kwargs: Any) -> Any:
            if kwargs["mode"] == "umap":
                raise RuntimeError("boom")
            return real(frame, **kwargs)

        monkeypatch.setattr(evaluation, "learn_embedding_with_model", flaky)
        report = row2vec.compare_modes(df, modes=["umap", "pca"])

        assert cell(report, "umap", "status") == "failed"
        assert cell(report, "umap", "note") == "RuntimeError: boom"
        assert cell(report, "pca", "status") == "ok"


class TestValidation:
    def test_unknown_mode(self, df: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="Unknown mode"):
            row2vec.compare_modes(df, modes=["nope"])

    def test_target_mode_needs_a_target(self, df: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="needs a target"):
            row2vec.compare_modes(df, modes=["target"])

    def test_target_must_be_a_column(self, df: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="not a column"):
            row2vec.compare_modes(df, target="missing", modes=["pca"])

    def test_target_must_not_have_missing_values(self, df: pd.DataFrame) -> None:
        broken = df.copy()
        broken.loc[0, "Country"] = None
        with pytest.raises(ValueError, match="missing value"):
            row2vec.compare_modes(broken, target="Country", modes=["pca"])

    def test_test_size_range(self, df: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="test_size"):
            row2vec.compare_modes(df, modes=["pca"], test_size=1.5)

    def test_default_modes_skip_target_without_a_target(self) -> None:
        assert "target" not in evaluation._resolve_modes(None, None)
        assert "target" in evaluation._resolve_modes(None, "Country")


@pytest.mark.neural
def test_neural_modes_run(df: pd.DataFrame) -> None:
    pytest.importorskip("tensorflow")
    report = row2vec.compare_modes(
        df, target="Country", modes=["unsupervised", "target", "contrastive"], max_epochs=2
    )
    assert list(report.loc[["unsupervised", "target", "contrastive"], "status"]) == ["ok"] * 3


@pytest.mark.neural
def test_baseline_is_reproducible_with_a_learned_encoding() -> None:
    """A high-cardinality column gets entity embeddings; the baseline must not drift."""
    pytest.importorskip("tensorflow")
    rng = np.random.default_rng(0)
    n = 900
    frame = pd.DataFrame(
        {
            "code": [f"c{i % 150:03d}" for i in range(n)],
            "x": rng.normal(size=n),
            "label": rng.choice(["u", "v"], size=n),
        }
    )
    first = row2vec.compare_modes(frame, target="label", modes=["pca"])
    second = row2vec.compare_modes(frame, target="label", modes=["pca"])
    assert cell(first, "baseline", "downstream_score") == cell(
        second, "baseline", "downstream_score"
    )
