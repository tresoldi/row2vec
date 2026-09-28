"""Regression tests for the data-integrity defects found in the 0.2.0 audit.

Every test here was written *before* its fix and observed to fail, so each one
pins down a specific wrong behaviour rather than merely exercising a code path.
The identifiers in the test names (C1..C9, S1..S3) are the audit's finding IDs;
they are kept so a failure points straight back at the defect it guards.

These tests assert *values*, not shapes. A shape assertion would have passed on
every one of these bugs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest

from row2vec import (
    AdaptiveImputer,
    CategoricalEncoder,
    CategoricalEncodingConfig,
    EmbeddingConfig,
    ImputationConfig,
    TargetEncoder,
    learn_embedding,
    load_model,
    train_and_save_model,
)
from row2vec.config import create_config_for_mode

if TYPE_CHECKING:
    from pathlib import Path

# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def mid_cardinality_frame() -> pd.DataFrame:
    """600 rows whose only informative columns are numeric.

    ``entity_id`` has 150 levels, which lands in the 101..1000 band where the
    adaptive selector used to emit raw, unscaled ordinal codes.
    """
    rng = np.random.default_rng(1305)
    n = 600
    return pd.DataFrame(
        {
            "entity_id": [f"id_{i % 150:04d}" for i in range(n)],
            "x": rng.normal(size=n),
            "y": rng.normal(size=n),
            "z": rng.normal(size=n),
        }
    )


@pytest.fixture
def labelled_frame() -> pd.DataFrame:
    """Small mixed frame with a low-cardinality reference column."""
    rng = np.random.default_rng(7)
    n = 120
    return pd.DataFrame(
        {
            "country": rng.choice(["USA", "Canada", "Mexico", "Brazil"], size=n),
            "a": rng.normal(size=n),
            "b": rng.normal(size=n),
        }
    )


def _ordinal_rank(values: pd.Series) -> np.ndarray:
    """Position of each value in the sorted set of distinct values."""
    order = {v: i for i, v in enumerate(sorted(values.unique()))}
    return values.map(order).to_numpy(dtype=float)


def _abs_corr(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    return float(abs(np.corrcoef(a, b)[0, 1]))


# --------------------------------------------------------------------------- #
# C1 - mid-cardinality categoricals must not dominate the embedding
# --------------------------------------------------------------------------- #


def test_c1_mid_cardinality_column_does_not_dominate_embedding(
    mid_cardinality_frame: pd.DataFrame,
) -> None:
    """An arbitrary ID column must not become the principal component.

    Before the fix the categorical branch had no scaler while numerics were
    standardised, so a 150-level ID arrived as ordinal codes spanning 0..149
    (std ~43) against numeric features of std 1. PCA then returned, almost
    exactly, the alphabetical rank of the ID string.
    """
    embeddings = learn_embedding(
        mid_cardinality_frame, mode="pca", embedding_dim=2, enable_logging=False
    )

    rank = _ordinal_rank(mid_cardinality_frame["entity_id"])
    corr = _abs_corr(embeddings.iloc[:, 0].to_numpy(), rank)

    assert corr < 0.5, (
        f"first component correlates {corr:.4f} with the alphabetical rank of an "
        "arbitrary identifier; the categorical branch is dominating the embedding"
    )


def test_c1_categorical_features_are_scaled_like_numeric_ones(
    mid_cardinality_frame: pd.DataFrame,
) -> None:
    """No encoded feature may enter the model on a wildly different scale."""
    from row2vec.pipeline_builder import build_adaptive_pipeline

    pipeline, _report = build_adaptive_pipeline(mid_cardinality_frame, config=EmbeddingConfig())
    processed = np.asarray(pipeline.fit_transform(mid_cardinality_frame))

    stds = processed.std(axis=0)
    informative = stds[stds > 0]
    assert informative.size > 0

    # Numeric columns are standardised to ~1. Nothing should tower over them.
    assert informative.max() < 10.0, (
        f"max feature std is {informative.max():.3f}; an unscaled encoded column "
        "is entering the model"
    )


# --------------------------------------------------------------------------- #
# C2 / C3 - index handling
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("mode", ["pca", "target"])
def test_c2_c3_output_index_matches_input_index(labelled_frame: pd.DataFrame, mode: str) -> None:
    """Output must carry df's own index, one row per input row.

    ``mode="target"`` used to build its frame with a fresh RangeIndex and then
    group by a label-aligned Series, so on a non-default index every group key
    was NaN and the result came back completely empty.
    """
    df = labelled_frame.copy()
    df.index = pd.Index([f"row{i}" for i in range(len(df))])

    embeddings = learn_embedding(
        df,
        mode=mode,
        embedding_dim=2,
        reference_column="country" if mode == "target" else None,
        max_epochs=2,
        enable_logging=False,
    )

    assert len(embeddings) == len(df), f"{mode} returned {len(embeddings)} of {len(df)} rows"
    assert list(embeddings.index) == list(df.index)


def test_c3_output_joins_back_onto_the_source_frame(labelled_frame: pd.DataFrame) -> None:
    """The README promises a concat-able result; verify it literally."""
    df = labelled_frame.copy()
    df.index = pd.Index([f"row{i}" for i in range(len(df))])

    embeddings = learn_embedding(df, mode="pca", embedding_dim=2, enable_logging=False)
    joined = pd.concat([df, embeddings], axis=1)

    assert len(joined) == len(df), "concat duplicated rows: indices do not align"
    assert joined.isnull().sum().sum() == 0, "concat introduced NaNs: indices do not align"


def test_c2_target_mode_aggregation_is_opt_in(labelled_frame: pd.DataFrame) -> None:
    """The per-category matrix is still reachable, but only on request."""
    per_row = learn_embedding(
        labelled_frame,
        mode="target",
        reference_column="country",
        embedding_dim=2,
        max_epochs=2,
        enable_logging=False,
    )
    assert len(per_row) == len(labelled_frame)

    aggregated = learn_embedding(
        labelled_frame,
        mode="target",
        reference_column="country",
        embedding_dim=2,
        max_epochs=2,
        enable_logging=False,
        aggregate_by_reference=True,
    )
    assert len(aggregated) == labelled_frame["country"].nunique()


# --------------------------------------------------------------------------- #
# C4 - unseen categories must stay local to their own rows
# --------------------------------------------------------------------------- #


def test_c4_unseen_category_does_not_zero_other_rows() -> None:
    """One unknown value must not wipe out the column for every row.

    The entity-embedding transform assigned into ``arr[mask][i]`` - a temporary
    copy - so on an unknown category nothing was written and every row fell
    through to the all-zero default.
    """
    rng = np.random.default_rng(3)
    n = 300
    known = pd.DataFrame(
        {
            "cat": [f"c{i % 30}" for i in range(n)],
            "num": rng.normal(size=n),
        }
    )

    config = CategoricalEncodingConfig(entity_threshold=5)  # force entity embeddings
    encoder = CategoricalEncoder(config)
    encoder.fit(known[["cat"]])

    baseline = np.asarray(encoder.transform(known[["cat"]]), dtype=float)

    with_unseen = known.copy()
    with_unseen.loc[with_unseen.index[0], "cat"] = "totally_new_value"
    perturbed = np.asarray(encoder.transform(with_unseen[["cat"]]), dtype=float)

    all_zero_rows = int((np.abs(perturbed) < 1e-12).all(axis=1).sum())
    assert all_zero_rows < len(perturbed), (
        "a single unseen category zeroed the encoding for every row "
        f"({all_zero_rows}/{len(perturbed)} rows are all zero)"
    )

    # Rows 1..n-1 were untouched, so their encodings must be unchanged.
    np.testing.assert_allclose(
        perturbed[1:],
        baseline[1:],
        rtol=1e-6,
        atol=1e-9,
        err_msg="an unseen value in row 0 changed the encoding of unrelated rows",
    )


# --------------------------------------------------------------------------- #
# C5 - imputation must cover columns that happened to be clean at fit time
# --------------------------------------------------------------------------- #


def test_c5_column_clean_at_fit_time_is_still_imputed() -> None:
    """A column with no NaN during fit must still be imputed at transform.

    The imputer stored ``None`` for such columns, so NaN flowed straight
    through into PCA/Keras and produced NaN embeddings.
    """
    fit_df = pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0, 4.0],  # clean at fit time
            "b": [1.0, np.nan, 3.0, 4.0],
        }
    )
    transform_df = pd.DataFrame(
        {
            "a": [1.0, np.nan, 3.0, 4.0],  # dirty at transform time
            "b": [1.0, 2.0, 3.0, np.nan],
        }
    )

    imputer = AdaptiveImputer(ImputationConfig())
    imputer.fit(fit_df)
    result = imputer.transform(transform_df)

    remaining = int(pd.DataFrame(result).isnull().sum().sum())
    assert remaining == 0, f"{remaining} NaN survived imputation"


def test_c5_transform_output_width_is_stable() -> None:
    """Output width must be pinned at fit time, not derived per batch."""
    fit_df = pd.DataFrame({"a": [1.0, np.nan, 3.0, 4.0], "b": [1.0, 2.0, 3.0, 4.0]})

    imputer = AdaptiveImputer(ImputationConfig(preserve_missing_patterns=True))
    imputer.fit(fit_df)

    clean_batch = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    dirty_batch = pd.DataFrame({"a": [np.nan, 2.0], "b": [np.nan, 4.0]})

    width_clean = np.asarray(imputer.transform(clean_batch)).shape[1]
    width_dirty = np.asarray(imputer.transform(dirty_batch)).shape[1]

    assert width_clean == width_dirty, (
        f"output width depends on the batch ({width_clean} vs {width_dirty}); "
        "downstream estimators will break on the second call"
    )


# --------------------------------------------------------------------------- #
# C6 - transform() must not refit
# --------------------------------------------------------------------------- #


def test_c6_transform_reuses_the_fitted_state(labelled_frame: pd.DataFrame) -> None:
    """``transform`` must project with the basis learned during ``fit``.

    It used to call the full training entry point on whatever it was handed, so
    inside cross-validation every fold learned its embedding from its own test
    fold.
    """
    from row2vec.sklearn import Row2VecTransformer

    train = labelled_frame.iloc[:90]
    test = labelled_frame.iloc[90:]

    fitted_on_train = Row2VecTransformer(embedding_dim=2, mode="pca").fit(train)
    projected = fitted_on_train.transform(test)

    refit_on_test = Row2VecTransformer(embedding_dim=2, mode="pca").fit_transform(test)

    assert not np.allclose(projected, refit_on_test), (
        "transform(test) equals a fresh fit_transform(test): the fit on train "
        "was discarded and the transformer refit on the data it was given"
    )


def test_c6_transform_accepts_a_single_row(labelled_frame: pd.DataFrame) -> None:
    """Single-row inference is the whole point of a fitted transformer."""
    from row2vec.sklearn import Row2VecTransformer

    transformer = Row2VecTransformer(embedding_dim=2, mode="pca").fit(labelled_frame)
    one_row = transformer.transform(labelled_frame.iloc[[0]])

    assert one_row.shape == (1, 2)


# --------------------------------------------------------------------------- #
# C7 - a saved model must reproduce what training returned
# --------------------------------------------------------------------------- #


def test_c7_saved_model_reproduces_training_embeddings(
    labelled_frame: pd.DataFrame, tmp_path: Path
) -> None:
    """``predict`` must apply the same embedding scaler that training applied.

    ``scale_method`` was recorded in the metadata and then never used at
    inference, so a model trained with minmax scaling returned raw, unscaled
    coordinates on reload.
    """
    base_path = tmp_path / "model"
    trained, _script_path, _binary_path = train_and_save_model(
        labelled_frame,
        base_path,
        mode="pca",
        embedding_dim=2,
        scale_method="minmax",
        scale_range=(0.0, 1.0),
        enable_logging=False,
    )

    reloaded = load_model(base_path)
    predicted = reloaded.predict(labelled_frame)

    np.testing.assert_allclose(
        np.asarray(predicted, dtype=float),
        np.asarray(trained, dtype=float),
        rtol=1e-5,
        atol=1e-6,
        err_msg="reloaded model does not reproduce the training embeddings",
    )


def test_c7_metadata_records_the_real_version(labelled_frame: pd.DataFrame, tmp_path: Path) -> None:
    """Metadata must record the running version, not a hardcoded literal."""
    import row2vec

    base_path = tmp_path / "model"
    train_and_save_model(
        labelled_frame,
        base_path,
        mode="pca",
        embedding_dim=2,
        enable_logging=False,
    )

    assert load_model(base_path).metadata.row2vec_version == row2vec.__version__


# --------------------------------------------------------------------------- #
# C8 - target encoding must not leak the row's own label
# --------------------------------------------------------------------------- #


def test_c8_target_encoding_does_not_leak_at_transform() -> None:
    """On a noise target with unique categories the encoding must carry no signal.

    ``fit_transform`` computed leak-free cross-validated values and then threw
    them away; ``transform`` returned the full-data mean per category, which for
    a unique category *is* that row's own label.
    """
    rng = np.random.default_rng(11)
    n = 50
    series = pd.Series([f"cat_{i}" for i in range(n)])  # every category unique
    target = pd.Series(rng.normal(size=n))  # pure noise

    encoder = TargetEncoder(CategoricalEncodingConfig())
    encoder.fit_transform(series, target)
    encoded = encoder.transform(series)

    corr = _abs_corr(encoded.to_numpy(dtype=float), target.to_numpy(dtype=float))
    assert corr < 0.3, (
        f"encoded values correlate {corr:.4f} with a pure-noise target; "
        "the encoder is returning each row's own label"
    )


# --------------------------------------------------------------------------- #
# C9 - contrastive pairing must use positions, not index labels
# --------------------------------------------------------------------------- #


def test_c9_contrastive_pairs_use_positions_not_labels(labelled_frame: pd.DataFrame) -> None:
    """Pair generation indexed a numpy array with pandas index *labels*.

    With a shifted index that raises IndexError; with a merely permuted one it
    silently pairs the wrong rows.
    """
    df = labelled_frame.copy()
    df.index = pd.Index(range(1000, 1000 + len(df)))

    embeddings = learn_embedding(
        df,
        mode="contrastive",
        embedding_dim=2,
        auto_pairs="categorical",
        reference_column="country",
        max_epochs=2,
        enable_logging=False,
    )

    assert len(embeddings) == len(df)
    assert list(embeddings.index) == list(df.index)


# --------------------------------------------------------------------------- #
# S1..S3 - architecture search and config plumbing
# --------------------------------------------------------------------------- #


def test_s1_auto_architecture_works_for_contrastive(labelled_frame: pd.DataFrame) -> None:
    """Architecture search dropped the contrastive settings from each trial.

    Every trial therefore failed, the failures were swallowed, and the caller
    hit ``KeyError: 'hidden_units'`` on an empty best-architecture dict.
    """
    from row2vec.api import learn_embedding_v2

    config = create_config_for_mode("contrastive")
    config.embedding_dim = 2
    config.reference_column = "country"
    config.contrastive.auto_pairs = "categorical"
    config.neural.max_epochs = 2

    embeddings = learn_embedding_v2(labelled_frame, config, auto_architecture=True)

    assert len(embeddings) == len(labelled_frame)
    assert embeddings.shape[1] == 2


def test_s2_max_time_stops_the_search() -> None:
    """``max_time`` compared ``current_time - time.time()``, which is ~0."""
    import time

    from row2vec.architecture_search import ArchitectureSearchConfig, ArchitectureSearcher

    searcher = ArchitectureSearcher(ArchitectureSearchConfig(max_time=1e-6))
    # Pretend the search began ten seconds ago, against a one-microsecond budget.
    # setattr keeps this test type-clean before the attribute exists.
    setattr(searcher, "_start_time", time.time() - 10.0)  # noqa: B010

    assert searcher._should_stop(time.time()) is True


def test_s3_activation_survives_the_config_round_trip() -> None:
    """``NeuralConfig.activation`` was omitted from ``to_dict``.

    The architecture search spent trials distinguishing relu/elu/swish models
    that were byte-identical, because the choice never reached the builder.
    """
    config = EmbeddingConfig()
    config.neural.activation = "elu"

    restored = EmbeddingConfig.from_dict(config.to_dict())

    assert restored.neural.activation == "elu"


def test_s11_from_dict_does_not_mutate_the_caller() -> None:
    """``from_dict`` popped keys straight out of the caller's dictionary."""
    payload = {"embedding_dim": 3, "neural": {"max_epochs": 5}}
    snapshot = {"embedding_dim": 3, "neural": {"max_epochs": 5}}

    EmbeddingConfig.from_dict(payload)

    assert payload == snapshot, "from_dict consumed the caller's dictionary"
