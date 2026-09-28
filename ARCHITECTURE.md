# Row2Vec Architecture

> Status: **describes the delivered architecture as of 0.2.0.**
> This document defines the structure, boundaries, and design principles of
> Row2Vec. New modules should fit within it; changes that move away from it
> should update it in the same PR. Key design decisions are recorded in §8.

---

## 1. Purpose and scope

Row2Vec turns **table rows into vectors**. Given a `DataFrame` of mixed numeric
and categorical columns — with gaps — it produces a dense numeric matrix with one
row per input row.

The scope is deliberately narrow. Row2Vec is *not* a modelling framework, an
AutoML system, or a feature store. It does one conversion, and it does the
unglamorous work that conversion requires:

| Concern | What Row2Vec does |
|---------|-------------------|
| Missing values | Analyses the pattern per column, imputes accordingly |
| Categorical columns | Selects an encoding from the column's cardinality |
| Numeric scale | Standardises before projecting; optionally rescales after |
| Projection | PCA, autoencoder, t-SNE, UMAP, target, contrastive |
| Reuse | Saves the fitted preprocessing *with* the model |

**Design consequence:** the vocabulary throughout is *row*, *column*,
*embedding*, *mode* — never a domain's own terms. A row is whatever the user's
table contains.

---

## 2. Design principles

1. **One entry point.** Every embedding comes from `learn_embedding`. The method
   is a value (`mode=`), not a different function, a different class, or a
   different import. Adding a method must not add a second way to call the
   library.
2. **Preprocessing belongs to the model.** An embedding is only reproducible if
   the encoders, imputers, and scalers that produced it are reproducible too.
   `Row2VecModel` is the object that holds them — the fitted preprocessor, the
   fitted projector, and the fitted embedding scaler — and a saved model
   carries all three, so `predict` on new rows replays exactly what training
   did. `transform` never fits. The exception is `mode="tsne"`, which is
   non-parametric and raises rather than pretending it can embed unseen
   rows.
3. **Sensible defaults over configuration.** Every parameter has a default that
   works. `EmbeddingConfig` exists for when you need to be explicit, not as the
   price of entry.
4. **DRY.** Shared decisions live in exactly one place. Column classification —
   is this column numeric or categorical? — is a single predicate in `utils`,
   not a dtype test repeated in six modules.
5. **YAGNI.** No abstraction, config surface, or extension point ahead of a
   concrete need.
6. **Layout-independent public API.** The public surface is defined by
   `src/row2vec/__init__.py`. Internal reorganisation must not break
   `import row2vec; row2vec.learn_embedding(...)`.
7. **Typed and validated.** Full type hints, `py.typed`, strict mypy. Parameters
   are validated up front with messages that name the offending value, so
   failures surface at the call site rather than deep inside TensorFlow or
   scikit-learn.

---

## 3. Package structure

`src/` layout; the public API is re-exported from `__init__.py`, so file layout
is transparent to users.

```
src/row2vec/
├── __init__.py            # public API surface (stable import path)
├── core.py                # learn_embedding + every mode's implementation
├── api.py                 # config-object entry points (learn_embedding_v2, ...)
├── config.py              # EmbeddingConfig and its component dataclasses
├── utils.py               # dtype classification, synthetic data, schema helpers
├── logging.py             # structured training/progress logging
│
│                          # -- preprocessing --
├── imputation.py          # AdaptiveImputer, MissingPatternAnalyzer
├── categorical_encoding.py# encoder selection, target encoding, entity embeddings
├── pipeline_builder.py    # assembles the above into a ColumnTransformer
│
│                          # -- tuning --
├── auto_dimension.py      # auto_select_dimension
├── architecture_search.py # search_architecture
│
│                          # -- boundaries --
├── serialization.py       # save/load a fitted model with its pipeline
├── cli.py                 # command-line interface
├── sklearn.py             # Row2VecTransformer / Row2VecClassifier
├── pandas.py              # the .row2vec DataFrame accessor
└── py.typed
```

**Rationale for the boundaries**

- `core.py` holds the modes themselves. They share so much preprocessing and
  validation that splitting them per method would mean either duplicating that
  work or inventing an abstraction nobody asked for (YAGNI, §2.5).
- The **preprocessing** modules form a stack: `pipeline_builder` composes
  `imputation` and `categorical_encoding` into a single fitted transformer. That
  transformer is the object `serialization` persists.
- The **tuning** modules are optional extras that call back into the API. They
  are the only modules that run training in a loop.
- The **boundary** modules (`cli`, `sklearn`, `pandas`, `serialization`) are
  adapters. They translate between an external convention and the one entry
  point; none of them contains embedding logic.

### Dependency rule

Dependencies point **inward**. `utils`, `config`, and `logging` are foundations
and import nothing else from the package. Preprocessing imports foundations.
`core` imports preprocessing and foundations. Boundary and tuning modules import
`core`/`api` — and nothing imports *them*.

Concretely: `utils.is_categorical_series` may be called from anywhere;
`utils` may call nothing. If a foundation module ever needs something from
`core`, the design has gone wrong.

---

## 4. Public API contract

The embedding lifecycle:

```python
embeddings = row2vec.learn_embedding(df, mode=..., embedding_dim=...)
```

- The return value is a `DataFrame` with the same index as `df` and
  `embedding_dim` columns. This holds for every mode: `target` mode returns one
  row per input row like the rest, and `aggregate_by_reference=True` asks for
  the per-category matrix instead. Before 0.3.0 the index was a fresh
  `RangeIndex`, so the promise above was true only for input that already had
  one.
- `seed` makes a run reproducible.
- Invalid parameters raise `ValueError`/`TypeError` before any training starts.

**Configuration objects** sit alongside the keyword form for callers who want to
build a configuration once and reuse it:

```python
embeddings = row2vec.learn_embedding_v2(df, EmbeddingConfig(...))
```

`EmbeddingConfig` composes `NeuralConfig`, `ClassicalConfig`,
`ContrastiveConfig`, `ScalingConfig`, `PreprocessingConfig`, and `LoggingConfig`
— one per concern, so a caller sets only what they care about.

**Persistence.** A fitted model can be saved and restored with its preprocessing:

```python
embeddings, script_path, binary_path = row2vec.train_and_save_model(df, base_path=...)
model = row2vec.load_model(script_path)
model.predict(new_rows)
```

The format is deliberately two files: a **readable Python loader script** and a
binary blob. The script documents what was trained and how to load it; the blob
holds the fitted objects. The trade-off is that loading executes the script — see
`SECURITY.md`.

---

## 5. Column classification

Every column takes one of two paths: scaled as numeric, or encoded as
categorical. That decision is made in exactly one place —
`utils.is_categorical_series` — and every module defers to it.

This matters more than it looks. Testing `series.dtype in ("object", "category")`
was correct until pandas 3.0, which infers `str` for text columns; the same test
then routes text into the numeric branch and the first reduction raises. Because
the predicate is centralised, adapting to that was a one-line change rather than
a hunt through six modules.

---

## 6. Documentation architecture

- **Autogenerated API reference** from docstrings via `mkdocstrings` — no
  hand-maintained reference to drift out of date.
- **Narrative guide** (hand-written): concepts, choosing a method, worked tasks.
- **Landing page**: a self-contained MkDocs template (`overrides/home.html`).
- **Executable tutorials** (`docs/tutorial_*.py`, Nhandu) kept as source only;
  rendered HTML is a build artifact and is never committed.
- **Everything is executed.** `tests/test_doctests.py` runs the docstring
  examples and `tests/test_docs_examples.py` runs the README and User Guide code
  blocks, so documentation cannot silently rot.
- **Tooling:** MkDocs-Material + mkdocstrings, published to GitHub Pages from CI
  (see `mkdocs.yml`; build with `make site`).

---

## 7. Versioning & compatibility

The public API is the top-level `row2vec` namespace. The project follows
semantic versioning; while pre-1.0, breaking changes are possible but are
documented in `MIGRATION.md` and flagged in the `CHANGELOG`.

0.2.0 moved the package to a `src/` layout and rebuilt the documentation and
tooling. The import path did not change, so no user code needed editing.

---

## 8. Decisions (resolved)

1. **One function, not one class per method.** `mode=` selects the algorithm.
   The alternative — an estimator class per method — would multiply the surface
   without adding capability, since the preprocessing is shared (§2.1).
2. **`src/` layout.** Adopted in 0.2.0 so that tests run against the installed
   package rather than the working directory, which is what CI and users
   actually exercise (§3).
3. **Preprocessing is persisted with the model.** The alternative — persisting
   only the estimator — makes inference silently inconsistent with training
   whenever the input distribution shifts (§2.2, §4).
4. **Two-file model format.** A readable loader script plus a binary blob, in
   preference to one opaque pickle. The cost is that loading executes code; that
   is documented rather than hidden (§4, `SECURITY.md`).
5. **Column classification centralised in `utils`.** One predicate, deferred to
   everywhere, rather than a dtype test repeated per module (§5).
6. **scikit-learn and pandas integrations are adapters, not the core.** They
   live in their own modules and import inward; the core has no knowledge of
   either convention (§3).
7. **Docs tooling:** MkDocs-Material + mkdocstrings, published to GitHub Pages,
   with every example executed in CI (§6).
