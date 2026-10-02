"""Encoding of free-text columns.

Text is opt-in: a column is treated as text only if it is named in
``PreprocessingConfig.text_columns``. By default it is encoded with TF-IDF
followed by a truncated SVD, which needs no extra dependency. A caller can
replace that with any ``callable(list[str]) -> array`` (a sentence-embedding
model, say) through ``PreprocessingConfig.text_encoder``.

A custom callable is code, and a saved model must load without running code
from the file, so it is never written to disk. The model records that it needs
one, and :func:`row2vec.load_model` takes it again as ``text_encoder=``.
"""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

#: What a text hook is: it receives a list of strings and returns an
#: ``(n_strings, dim)`` array.
TextHook = Callable[[list[str]], Any]


class SharedHook:
    """A text hook that scikit-learn's ``clone`` will not deep-copy.

    ``ColumnTransformer`` clones its transformers, and ``clone`` deep-copies any
    parameter that is not an estimator. For a sentence-embedding model that
    would copy the whole model, so the hook is wrapped to be copied by reference.
    """

    def __init__(self, fn: TextHook) -> None:
        self.fn = fn

    def __call__(self, texts: list[str]) -> Any:
        """Call the wrapped hook."""
        return self.fn(texts)

    def __deepcopy__(self, memo: dict[int, Any]) -> "SharedHook":
        """Return ``self``: the hook is shared, never duplicated."""
        return self

    def __copy__(self) -> "SharedHook":
        """Return ``self``: the hook is shared, never duplicated."""
        return self


def _texts(series: pd.Series) -> list[str]:
    """The column as plain strings; missing values become the empty string."""
    return ["" if pd.isna(value) else str(value) for value in series]


class TextEncoder(BaseEstimator, TransformerMixin):
    """Turn text columns into dense feature vectors.

    Each column is encoded on its own. With no ``encoder`` it is TF-IDF followed
    by a truncated SVD to at most ``n_components`` dimensions (fewer when the
    vocabulary or the number of rows is smaller). With an ``encoder`` that
    callable produces the features, and its output width is fixed at fit time.
    Missing values are encoded as the empty string.

    Args:
        n_components (int): Width per column of the default TF-IDF/SVD encoding.
        encoder (callable, optional): ``list[str] -> array of shape (n, dim)``.
        random_state (int): Seed for the SVD.
        batch_size (int): How many strings are sent to ``encoder`` at once.

    Attributes:
        uses_hook_ (bool): Whether fitting used a custom ``encoder``. Such a
            model cannot be reloaded without supplying one again.
        hook_dims_ (list[int]): Output width of the hook, per column.
    """

    def __init__(
        self,
        n_components: int = 16,
        encoder: TextHook | None = None,
        random_state: int = 0,
        batch_size: int = 1024,
    ) -> None:
        self.n_components = n_components
        self.encoder = encoder
        self.random_state = random_state
        self.batch_size = batch_size

    def fit(self, X: Any, y: Any = None) -> "TextEncoder":
        """Fit the vocabulary and projection, or record the hook's output width."""
        frame = pd.DataFrame(X)
        self.__dict__.pop("hook_dims_", None)  # a refit must not check against the last fit
        self.columns_ = [str(c) for c in frame.columns]
        self.uses_hook_ = self.encoder is not None
        self.n_features_in_ = len(self.columns_)
        if self.uses_hook_:
            self.hook_dims_ = [
                int(self._embed(_texts(frame[col]), i).shape[1])
                for i, col in enumerate(frame.columns)
            ]
            return self
        vectorizers: list[TfidfVectorizer] = []
        reducers: list[TruncatedSVD] = []
        for name, col in zip(self.columns_, frame.columns, strict=True):
            # scikit-learn's default pattern drops one-character tokens, which in
            # short tabular text ("size L", "type 7") are often the signal.
            vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
            try:
                tfidf = vectorizer.fit_transform(_texts(frame[col]))
            except ValueError as exc:
                raise ValueError(
                    f"Text column {name!r} has no usable words, so it cannot be encoded: {exc}"
                ) from exc
            k = max(1, min(self.n_components, tfidf.shape[1], tfidf.shape[0]))
            reducer = TruncatedSVD(n_components=k, random_state=self.random_state)
            reducer.fit(tfidf)
            vectorizers.append(vectorizer)
            reducers.append(reducer)
        self.vectorizers_ = vectorizers
        self.reducers_ = reducers
        return self

    def _embed(self, texts: list[str], position: int) -> np.ndarray:
        """Run the hook in batches and check what it returns."""
        assert self.encoder is not None
        step = max(1, int(self.batch_size))
        parts = []
        for start in range(0, len(texts), step):
            chunk = texts[start : start + step]
            out = np.asarray(self.encoder(chunk), dtype=float)
            if out.ndim != 2 or out.shape[0] != len(chunk):
                raise ValueError(
                    f"The text encoder must return an array of shape ({len(chunk)}, dim) "
                    f"for {len(chunk)} strings; got shape {out.shape}."
                )
            parts.append(out)
        if not parts:
            raise ValueError("Cannot encode an empty text column.")
        out = np.vstack(parts)
        if out.shape[1] != parts[0].shape[1] or not np.isfinite(out).all():
            raise ValueError("The text encoder returned a varying width or non-finite values.")
        if hasattr(self, "hook_dims_") and out.shape[1] != self.hook_dims_[position]:
            raise ValueError(
                f"The text encoder returned {out.shape[1]} features for column "
                f"{self.columns_[position]!r}; the model was fitted with "
                f"{self.hook_dims_[position]}."
            )
        return out

    def transform(self, X: Any) -> np.ndarray:
        """Encode the text columns."""
        frame = pd.DataFrame(X)
        blocks: list[np.ndarray] = []
        if self.uses_hook_:
            if self.encoder is None:
                raise ValueError(
                    "This model was trained with a custom text encoder, which is not "
                    "saved. Supply it with load_model(path, text_encoder=...)."
                )
            for i, col in enumerate(frame.columns):
                blocks.append(self._embed(_texts(frame[col]), i))
        else:
            for vectorizer, reducer, col in zip(
                self.vectorizers_, self.reducers_, frame.columns, strict=True
            ):
                blocks.append(
                    np.asarray(reducer.transform(vectorizer.transform(_texts(frame[col]))))
                )
        return np.hstack(blocks) if blocks else np.empty((len(frame), 0))

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        """Names of the output features, ``<column>_text_<i>``."""
        widths = (
            self.hook_dims_
            if self.uses_hook_
            else [reducer.n_components for reducer in self.reducers_]
        )
        names = [
            f"{col}_text_{i}"
            for col, width in zip(self.columns_, widths, strict=True)
            for i in range(width)
        ]
        return np.asarray(names, dtype=object)


def iter_text_encoders(preprocessor: Any) -> Iterator[TextEncoder]:
    """Every distinct :class:`TextEncoder` inside a ``ColumnTransformer``."""
    seen: set[int] = set()
    for attr in ("transformers", "transformers_"):
        for entry in getattr(preprocessor, attr, []):
            candidate = entry[1]
            if isinstance(candidate, TextEncoder) and id(candidate) not in seen:
                seen.add(id(candidate))
                yield candidate


@contextmanager
def detached_hooks(preprocessor: Any) -> Iterator[list[TextEncoder]]:
    """Temporarily remove custom hooks so the preprocessor can be serialised.

    Yields the encoders that had one. Each hook is put back on exit, even if
    serialisation fails.
    """
    held = [(enc, enc.encoder) for enc in iter_text_encoders(preprocessor) if enc.encoder]
    for enc, _ in held:
        enc.encoder = None
    try:
        yield [enc for enc, _ in held]
    finally:
        for enc, hook in held:
            enc.encoder = hook


def attach_hook(preprocessor: Any, hook: TextHook) -> int:
    """Give every hook-dependent text encoder ``hook``; return how many there were."""
    count = 0
    for enc in iter_text_encoders(preprocessor):
        if getattr(enc, "uses_hook_", False):
            enc.encoder = SharedHook(hook)
            count += 1
    return count
