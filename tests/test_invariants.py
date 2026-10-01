"""Invariants that hold across every mode, and the sklearn estimator contract.

The 0.2.0 suite asserted `.shape` about 87 times and numeric equality roughly
ten, all of which compared the code to itself. It passed while the index was
wrong for every non-default frame, while `transform` retrained on the data it
was given, and while a saved model returned differently scaled values than
training had.

These tests state the properties directly, so a regression of a *shape* this
suite has not anticipated still fails.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.utils.estimator_checks import parametrize_with_checks

from row2vec import EmbeddingConfig, generate_synthetic_data, learn_embedding
from row2vec.model import MODES, Row2VecModel
from row2vec.sklearn import Row2VecClassifier, Row2VecTransformer

if TYPE_CHECKING:
    from pathlib import Path

# Modes that can embed rows they were not fitted on.
TRANSFORMABLE_MODES = [name for name, spec in MODES.items() if spec.supports_transform]
ALL_MODES = list(MODES)

# Modes whose transform is an exact function of a single row: a linear
# projection, or a forward pass through a network. UMAP is excluded because its
# out-of-sample transform optimises the new points jointly against the training
# graph, so the result legitimately depends on what else is in the batch.
EXACT_TRANSFORM_MODES = [m for m in TRANSFORMABLE_MODES if m != "umap"]


def _config_for(mode: str, dim: int = 2) -> EmbeddingConfig:
    """A cheap configuration for `mode`, usable on a small frame."""
    config = EmbeddingConfig(
        embedding_dim=dim,
        mode=mode,
        reference_column="Country" if mode == "target" else None,
    )
    config.neural.max_epochs = 2
    config.neural.batch_size = 16
    config.logging.enabled = False
    if mode == "contrastive":
        config.contrastive.auto_pairs = "random"
    if mode == "tsne":
        config.classical.perplexity = 5.0
        config.classical.n_iter = 250
    return config


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    """A small synthetic frame with a deliberately non-default index."""
    df = generate_synthetic_data(80, seed=1305)
    df.index = pd.Index([f"row{i:03d}" for i in range(len(df))])
    return df


class TestIndexPreservation:
    """Output must join back onto the input, for every mode."""

    @pytest.mark.parametrize("mode", ALL_MODES)
    def test_index_is_preserved(self, frame: pd.DataFrame, mode: str) -> None:
        """One row out per row in, carrying the caller's own index."""
        embeddings = Row2VecModel(_config_for(mode)).fit_transform(frame)

        assert len(embeddings) == len(frame)
        assert list(embeddings.index) == list(frame.index)

    @pytest.mark.parametrize("mode", ALL_MODES)
    def test_result_concatenates_without_nans(self, frame: pd.DataFrame, mode: str) -> None:
        """The README's join promise, checked literally."""
        embeddings = Row2VecModel(_config_for(mode)).fit_transform(frame)
        joined = pd.concat([frame, embeddings], axis=1)

        assert len(joined) == len(frame)
        assert joined.isnull().sum().sum() == 0


class TestTransformNeverFits:
    """`transform` must project, never learn."""

    @pytest.mark.parametrize("mode", TRANSFORMABLE_MODES)
    def test_transform_is_repeatable(self, frame: pd.DataFrame, mode: str) -> None:
        """Two calls on the same rows must agree exactly."""
        model = Row2VecModel(_config_for(mode)).fit(frame)

        first = model.transform(frame)
        second = model.transform(frame)

        pd.testing.assert_frame_equal(first, second)

    @pytest.mark.parametrize("mode", TRANSFORMABLE_MODES)
    def test_transform_accepts_a_single_row(self, frame: pd.DataFrame, mode: str) -> None:
        """A fitted model must embed one row; refitting cannot."""
        model = Row2VecModel(_config_for(mode)).fit(frame)

        single = model.transform(frame.head(1))

        assert single.shape == (1, 2)
        assert list(single.index) == [frame.index[0]]

    @pytest.mark.parametrize("mode", EXACT_TRANSFORM_MODES)
    def test_subset_matches_the_full_transform(self, frame: pd.DataFrame, mode: str) -> None:
        """Embedding rows 0-4 alone must match embedding them among the rest.

        If `transform` refits, the basis differs per call and this diverges.
        UMAP is excluded by construction, not by convenience - see
        EXACT_TRANSFORM_MODES - and is covered by the correlation check below.
        """
        model = Row2VecModel(_config_for(mode)).fit(frame)

        whole = model.transform(frame).head(5)
        subset = model.transform(frame.head(5))

        np.testing.assert_allclose(subset.to_numpy(), whole.to_numpy(), rtol=1e-5, atol=1e-6)

    def test_umap_subset_is_close_to_the_full_transform(self, frame: pd.DataFrame) -> None:
        """UMAP's transform is approximate, but it must still be the same map.

        A refitting transform would produce an unrelated basis, so the
        correlation would be arbitrary rather than high.
        """
        model = Row2VecModel(_config_for("umap")).fit(frame)

        whole = model.transform(frame).head(20).to_numpy()
        subset = model.transform(frame.head(20)).to_numpy()

        for component in range(whole.shape[1]):
            correlation = abs(np.corrcoef(whole[:, component], subset[:, component])[0, 1])
            assert correlation > 0.9, (
                f"component {component} correlates only {correlation:.3f} between a "
                "subset and the full transform; the map is not being reused"
            )

    def test_tsne_refuses_to_transform(self, frame: pd.DataFrame) -> None:
        """A mode with no out-of-sample extension must say so."""
        model = Row2VecModel(_config_for("tsne")).fit(frame)

        with pytest.raises(NotImplementedError, match="non-parametric"):
            model.transform(frame.head(2))


class TestSavedModelsReproduceTraining:
    """A reloaded model must return what training returned."""

    @pytest.mark.parametrize("scale_method", ["none", "minmax", "standard", "l2", "tanh"])
    def test_round_trip_is_numerically_faithful(
        self, frame: pd.DataFrame, tmp_path: Path, scale_method: str
    ) -> None:
        """Including the embedding scaler, which used to be dropped."""
        from row2vec import load_model, train_and_save_model

        trained, script_path, _binary = train_and_save_model(
            frame,
            tmp_path / f"model_{scale_method}",
            mode="pca",
            embedding_dim=2,
            scale_method=scale_method,
            enable_logging=False,
        )

        reloaded = load_model(script_path).predict(frame)

        np.testing.assert_allclose(reloaded.to_numpy(), trained.to_numpy(), rtol=1e-5, atol=1e-6)
        assert list(reloaded.index) == list(frame.index)


class TestDeterminism:
    """The same seed must give the same answer."""

    @pytest.mark.parametrize("mode", ["pca", "umap"])
    def test_repeated_runs_agree(self, frame: pd.DataFrame, mode: str) -> None:
        """Classical modes are deterministic given a seed."""
        first = learn_embedding(frame, mode=mode, embedding_dim=2, enable_logging=False)
        second = learn_embedding(frame, mode=mode, embedding_dim=2, enable_logging=False)

        np.testing.assert_allclose(first.to_numpy(), second.to_numpy(), rtol=1e-6)

    def test_different_seeds_are_allowed_to_differ(self, frame: pd.DataFrame) -> None:
        """A seed that changes nothing would make the guarantee meaningless."""
        config = _config_for("unsupervised")
        config.seed = 1
        first = Row2VecModel(config).fit_transform(frame)

        config_two = _config_for("unsupervised")
        config_two.seed = 999
        second = Row2VecModel(config_two).fit_transform(frame)

        assert not np.allclose(first.to_numpy(), second.to_numpy())


class TestSklearnContract:
    """The adapters must behave like scikit-learn estimators."""

    @parametrize_with_checks(
        [
            Row2VecTransformer(embedding_dim=2, mode="pca"),
            Row2VecClassifier(embedding_dim=2, mode="pca"),
        ]
    )
    def test_estimator_checks(self, estimator: Any, check: Any) -> None:
        """scikit-learn's own conformance suite.

        Before 0.3.0 this could not even run on the transformer: with
        BaseEstimator ahead of TransformerMixin in the bases, sklearn refused
        with "the transformer_tags tag is not set".
        """
        check(estimator)

    def test_get_params_round_trips_through_clone(self) -> None:
        """Every constructor argument must survive `clone`.

        The previous `**kwargs` signature dropped them silently, so a grid
        search over those keys tuned nothing at all.
        """
        transformer = Row2VecTransformer(
            embedding_dim=3,
            mode="pca",
            seed=99,
            max_epochs=7,
            batch_size=16,
            activation="elu",
        )

        params = clone(transformer).get_params()

        assert params["embedding_dim"] == 3
        assert params["seed"] == 99
        assert params["max_epochs"] == 7
        assert params["batch_size"] == 16
        assert params["activation"] == "elu"

    def test_set_params_accepts_what_get_params_returns(self) -> None:
        """The two must agree on the parameter names."""
        transformer = Row2VecTransformer()
        transformer.set_params(embedding_dim=5, max_epochs=9)

        assert transformer.get_params()["embedding_dim"] == 5
        assert transformer.get_params()["max_epochs"] == 9

    def test_classifier_does_not_fit_the_callers_estimator(self, frame: pd.DataFrame) -> None:
        """The wrapped estimator is cloned, not fitted in place."""
        from sklearn.exceptions import NotFittedError
        from sklearn.linear_model import LogisticRegression
        from sklearn.utils.validation import check_is_fitted

        downstream = LogisticRegression(max_iter=200)
        classifier = Row2VecClassifier(embedding_dim=2, mode="pca", classifier=downstream)
        classifier.fit(frame.drop(columns=["Country"]), frame["Country"])

        with pytest.raises(NotFittedError):
            check_is_fitted(downstream)

    def test_transformer_rejects_tsne_at_fit_time(self, frame: pd.DataFrame) -> None:
        """A transformer that cannot transform has no place in a pipeline."""
        with pytest.raises(ValueError, match="tsne"):
            Row2VecTransformer(embedding_dim=2, mode="tsne").fit(frame)


class TestNoTargetLeakage:
    """Supervised encoders must not hand a row its own label."""

    def test_pipeline_cross_validation_is_not_inflated(self, frame: pd.DataFrame) -> None:
        """A noise target must not be predictable from the embedding.

        With `transform` refitting, each fold's embedding was learned from that
        fold's own test rows, which inflated this score substantially.
        """
        from sklearn.dummy import DummyClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score
        from sklearn.pipeline import Pipeline

        rng = np.random.default_rng(3)
        noise_target = rng.integers(0, 2, size=len(frame))

        pipeline = Pipeline(
            [
                ("embed", Row2VecTransformer(embedding_dim=2, mode="pca")),
                ("clf", LogisticRegression(max_iter=200)),
            ]
        )
        scores = cross_val_score(pipeline, frame, noise_target, cv=3)

        baseline = cross_val_score(
            DummyClassifier(strategy="most_frequent"), frame, noise_target, cv=3
        )

        # Against a target that carries no signal, the pipeline should not beat
        # a constant predictor by a wide margin.
        assert scores.mean() < baseline.mean() + 0.25
