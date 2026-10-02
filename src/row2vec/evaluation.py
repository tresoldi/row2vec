"""Compare embedding modes on your own data.

Which ``mode`` suits a table is an empirical question, and until now the library
could not help answer it. :func:`compare_modes` fits each mode on the same
training rows, embeds the same held-out rows, and scores the result against two
yardsticks that need no ground truth beyond the table itself:

* **Trustworthiness** - do a row's nearest neighbours in the embedding come from
  its nearest neighbours in the preprocessed table? 1.0 means the neighbourhood
  structure survived the projection.
* **Downstream score** - when a ``target`` column is given, how well does a
  k-nearest-neighbour model on the embedding predict it? Accuracy for a
  categorical target, R² for a numeric one. This is the number to trust if
  the embedding is going to feed a model.

Both are measured on rows the embedding was **not** fitted on, so a mode that
memorises its training rows does not win. The exception is t-SNE, which has no
out-of-sample extension: it embeds all rows at once, is scored on trustworthiness
only, and says so in its ``note``.
"""

from __future__ import annotations

import time
import warnings
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from sklearn.manifold import trustworthiness
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

from ._backend import NeuralBackendMissing, neural_available
from .config import EmbeddingConfig
from .core import learn_embedding_with_model
from .model import MODES, Row2VecModel
from .pipeline_builder import build_adaptive_pipeline
from .utils import is_categorical_series

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = ["compare_modes"]

#: Columns of the table :func:`compare_modes` returns.
COLUMNS = [
    "status",
    "trustworthiness",
    "downstream_score",
    "downstream_metric",
    "fit_seconds",
    "note",
]

# A numeric target with at most this many distinct values is treated as a label.
_MAX_CLASSES = 20
_KNN_NEIGHBOURS = 5


def compare_modes(
    df: pd.DataFrame,
    *,
    target: str | None = None,
    modes: Sequence[str] | None = None,
    embedding_dim: int = 2,
    n_neighbors: int = 10,
    test_size: float = 0.25,
    seed: int = 1305,
    tsne_max_rows: int = 1500,
    **learn_kwargs: Any,
) -> pd.DataFrame:
    """Fit several embedding modes on one table and score them side by side.

    Args:
        df (pd.DataFrame): The table to embed.
        target (str, optional): A column to predict from the embedding. It is
            removed from the features for every mode, so no mode sees it as an
            input, except ``mode="target"``, which uses it as its label. Without
            a target only trustworthiness is scored, and ``mode="target"`` is
            skipped.
        modes (Sequence[str], optional): Modes to compare. Defaults to every
            mode that can run (``"target"`` only when ``target`` is given).
        embedding_dim (int): Width of every embedding.
        n_neighbors (int): Neighbourhood size for trustworthiness.
        test_size (float): Share of rows held out for scoring.
        seed (int): Seed for the split and for every mode.
        tsne_max_rows (int): t-SNE embeds a seeded random sample of at most this
            many rows, because its cost grows steeply with the row count (several
            minutes at a few thousand rows). The sample size is given in its
            ``note``.
        **learn_kwargs: Passed to :func:`row2vec.learn_embedding_with_model`
            for every mode, e.g. ``max_epochs=20`` to keep neural modes quick.
            Contrastive mode pairs nearest neighbours (``auto_pairs="neighbors"``)
            unless you pass ``auto_pairs`` or explicit pairs.

    Returns:
        pd.DataFrame: One row per mode, indexed by mode name, with columns
        ``status`` (``"ok"``, ``"unavailable"``, ``"skipped"`` or ``"failed"``),
        ``trustworthiness``, ``downstream_score``, ``downstream_metric``,
        ``fit_seconds`` and ``note``. A ``baseline`` row gives the same
        downstream score on the preprocessed features without any embedding, so
        the numbers have something to be compared with. ``mode="target"`` is
        ``skipped`` for a numeric target, which it would treat as one class per
        value. Modes that need
        TensorFlow are reported as ``unavailable`` when it is not installed;
        a mode that raises is reported as ``failed`` with the error in ``note``.

    Raises:
        ValueError: If ``target`` is not a column, has missing values, or a mode
            is unknown; or if ``modes`` asks for ``"target"`` without a target.

    Examples:
        >>> import row2vec
        >>> df = row2vec.generate_synthetic_data(120)
        >>> report = row2vec.compare_modes(df, target="Country", modes=["pca"])
        >>> list(report.index)
        ['pca', 'baseline']
        >>> report.loc["pca", "status"]
        'ok'
    """
    chosen = _resolve_modes(modes, target)
    # The reference space can contain a learned (entity-embedding) block, so it
    # must be seeded like everything else or the baseline drifts between runs.
    Row2VecModel._seed_everything(seed, neural=neural_available())
    features, y = _split_target(df, target)

    if not 0 < test_size < 1:
        raise ValueError(f"test_size must be between 0 and 1, got {test_size}")
    positions = np.arange(len(df))
    train_pos, test_pos = train_test_split(positions, test_size=test_size, random_state=seed)
    n_neighbors = max(1, min(n_neighbors, len(test_pos) // 2 - 1))

    # One reference space for every mode: the preprocessed table, fitted on the
    # training rows only so the held-out rows stay held out.
    reference = _fit_reference(features.iloc[train_pos], seed)
    ref_train = _dense(reference.transform(features.iloc[train_pos]))
    ref_test = _dense(reference.transform(features.iloc[test_pos]))

    rows: dict[str, dict[str, Any]] = {}
    for mode in chosen:
        rows[mode] = _evaluate_mode(
            mode,
            df=df,
            features=features,
            y=y,
            target=target,
            train_pos=train_pos,
            test_pos=test_pos,
            tsne_max_rows=tsne_max_rows,
            ref_test=ref_test,
            embedding_dim=embedding_dim,
            n_neighbors=n_neighbors,
            seed=seed,
            learn_kwargs=learn_kwargs,
        )

    rows["baseline"] = _baseline_row(ref_train, ref_test, y, train_pos, test_pos)
    return pd.DataFrame.from_dict(rows, orient="index", columns=COLUMNS)


def _resolve_modes(modes: Sequence[str] | None, target: str | None) -> list[str]:
    if modes is None:
        return [m for m in MODES if m != "target" or target is not None]
    unknown = [m for m in modes if m not in MODES]
    if unknown:
        raise ValueError(f"Unknown mode(s) {unknown}; choose from {sorted(MODES)}.")
    if "target" in modes and target is None:
        raise ValueError("mode='target' needs a target column; pass target=...")
    return list(dict.fromkeys(modes))


def _split_target(df: pd.DataFrame, target: str | None) -> tuple[pd.DataFrame, pd.Series | None]:
    if target is None:
        return df, None
    if target not in df.columns:
        raise ValueError(f"target {target!r} is not a column of the table.")
    y = df[target]
    if y.isnull().any():
        raise ValueError(
            f"target {target!r} has {int(y.isnull().sum())} missing value(s); "
            "fill or drop those rows first."
        )
    return df.drop(columns=[target]), y


def _fit_reference(features: pd.DataFrame, seed: int) -> Any:
    pipeline, _report = build_adaptive_pipeline(features, config=EmbeddingConfig(seed=seed))
    return pipeline.fit(features)


def _dense(x: Any) -> np.ndarray[Any, Any]:
    return np.asarray(x.toarray() if hasattr(x, "toarray") else x, dtype=float)


def _evaluate_mode(
    mode: str,
    *,
    df: pd.DataFrame,
    features: pd.DataFrame,
    y: pd.Series | None,
    target: str | None,
    train_pos: np.ndarray[Any, Any],
    test_pos: np.ndarray[Any, Any],
    tsne_max_rows: int,
    ref_test: np.ndarray[Any, Any],
    embedding_dim: int,
    n_neighbors: int,
    seed: int,
    learn_kwargs: dict[str, Any],
) -> dict[str, Any]:
    row: dict[str, Any] = dict.fromkeys(COLUMNS, np.nan)
    row.update(status="ok", downstream_metric="", note="")
    if mode == "target" and y is not None and not _is_classification(y):
        return {
            **row,
            "status": "skipped",
            "note": (
                f"mode='target' embeds the values of a categorical column; {target!r} has "
                f"{y.nunique()} distinct numeric values, so each would be its own class"
            ),
        }
    kwargs = {"enable_logging": False, "seed": seed, **learn_kwargs}
    # `target` mode reads its label from the frame; every other mode must not
    # see the target column at all.
    frame = df if mode == "target" else features
    if mode == "target":
        kwargs["reference_column"] = target
    elif mode == "contrastive":
        # Contrastive mode needs pairs; nearest-neighbour pairs use no labels.
        kwargs.setdefault("auto_pairs", "neighbors")

    started = time.perf_counter()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if mode == "tsne":
                # Non-parametric: embed one set of rows at once and score on it.
                sample = features
                if len(features) > tsne_max_rows:
                    sample = features.sample(n=tsne_max_rows, random_state=seed)
                ref_sample = _dense(_fit_reference(sample, seed).transform(sample))
                started = time.perf_counter()
                emb, _ = learn_embedding_with_model(
                    sample, embedding_dim=embedding_dim, mode=mode, **kwargs
                )
                row["fit_seconds"] = time.perf_counter() - started
                row["trustworthiness"] = _trust(ref_sample, emb.to_numpy(), n_neighbors)
                row["note"] = (
                    "no out-of-sample embedding: no downstream score; "
                    f"scored on {'a sample of ' if len(sample) < len(features) else ''}"
                    f"{len(sample)} rows"
                )
                return row

            emb_train, model = learn_embedding_with_model(
                frame.iloc[train_pos], embedding_dim=embedding_dim, mode=mode, **kwargs
            )
            row["fit_seconds"] = time.perf_counter() - started
            emb_test = model.transform(frame.iloc[test_pos])
    except NeuralBackendMissing as exc:
        return {**row, "status": "unavailable", "note": str(exc)}
    except Exception as exc:
        return {**row, "status": "failed", "note": f"{type(exc).__name__}: {exc}"}

    row["trustworthiness"] = _trust(ref_test, emb_test.to_numpy(), n_neighbors)
    if y is not None:
        score, metric = _downstream(
            emb_train.to_numpy(), y.iloc[train_pos], emb_test.to_numpy(), y.iloc[test_pos]
        )
        row.update(downstream_score=score, downstream_metric=metric)
    return row


def _baseline_row(
    ref_train: np.ndarray[Any, Any],
    ref_test: np.ndarray[Any, Any],
    y: pd.Series | None,
    train_pos: np.ndarray[Any, Any],
    test_pos: np.ndarray[Any, Any],
) -> dict[str, Any]:
    row: dict[str, Any] = dict.fromkeys(COLUMNS, np.nan)
    row.update(
        status="baseline",
        trustworthiness=1.0,
        downstream_metric="",
        note="preprocessed features, no embedding",
    )
    if y is not None:
        score, metric = _downstream(ref_train, y.iloc[train_pos], ref_test, y.iloc[test_pos])
        row.update(downstream_score=score, downstream_metric=metric)
    return row


def _trust(reference: np.ndarray[Any, Any], embedding: np.ndarray[Any, Any], k: int) -> float:
    return float(trustworthiness(reference, embedding, n_neighbors=k))


def _is_classification(y: pd.Series) -> bool:
    return bool(is_categorical_series(y) or y.nunique() <= _MAX_CLASSES)


def _downstream(
    x_train: np.ndarray[Any, Any],
    y_train: pd.Series,
    x_test: np.ndarray[Any, Any],
    y_test: pd.Series,
) -> tuple[float, str]:
    """Fit a k-NN probe on the training embedding, score it on the held-out one."""
    k = min(_KNN_NEIGHBOURS, len(x_train))
    if _is_classification(y_train):
        probe = KNeighborsClassifier(n_neighbors=k).fit(x_train, y_train.astype(str))
        return float(probe.score(x_test, y_test.astype(str))), "accuracy"
    regressor = KNeighborsRegressor(n_neighbors=k).fit(x_train, y_train.astype(float))
    return float(regressor.score(x_test, y_test.astype(float))), "r2"
