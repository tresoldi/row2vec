# Migration Guide

## Migrating to 0.4.0

Version 0.4.0 is a correctness release. An independent audit of 0.2.0 found
nine data-integrity defects, none of which the test suite caught, because they
changed *values* rather than raising. Fixing them changes what row2vec returns,
so **embeddings produced by 0.4.0 will differ numerically from 0.2.0's for most
inputs**. That is the point: the old numbers were wrong.

Read this section if you have saved models, stored embeddings, or code that
depends on the shape of what `learn_embedding` returns.

### What will break loudly

These raise, so you will find them immediately.

- **TensorFlow is no longer installed by default.** `pip install row2vec` now
  gives PCA, t-SNE and UMAP. The `unsupervised`, `target` and `contrastive`
  modes, architecture search and entity embeddings need
  `pip install "row2vec[neural]"`; without it they raise a
  `NeuralBackendMissing` error (an `ImportError`) that says so. Note that
  `learn_embedding`'s default mode is `unsupervised`, so a bare
  `learn_embedding(df)` needs the extra. Add `row2vec[neural]` to your
  requirements to keep the old behaviour.

- **Models are now a single `.r2v` file, and loading runs no code.** The old
  script-plus-pickle format executed the script when loaded. `save_model` returns
  one path and `train_and_save_model` returns `(embeddings, path)` instead of
  `(embeddings, script_path, binary_path)`. `load_model` on a `.py` or `.pkl`
  path raises `ModelFormatError` without executing it, and the CLI's `--output` /
  `--model` take `.r2v` files. `row2vec.inspect_model(path)` reads the metadata
  without loading the model. `skops` is now a dependency.
- **Saved models must be retrained.** The saved format now carries the whole
  fitted state - preprocessor, projector, encoder and, new in 0.4.0, the fitted
  embedding scaler. Models written by 0.2.0 cannot be loaded.
- **`learn_embedding_with_model` returns two values, not four.** It was
  `(embeddings, model, preprocessor, metadata)`; it is now
  `(embeddings, Row2VecModel)`. The model carries the preprocessor, and
  `row2vec.serialization.describe_model(model)` produces the metadata. The same
  applies to `learn_embedding_with_model_v2`.
- **`Row2VecTransformer` no longer takes `**kwargs`.** Nested parameters such
  as `neural__max_epochs=15` are now explicit arguments: `max_epochs=15`,
  `batch_size=...`, `dropout_rate=...`, `hidden_units=...`, `activation=...`.
  This is what makes `get_params`, `set_params` and `clone` round-trip; with
  `**kwargs` a `GridSearchCV` over those keys silently tuned nothing.
- **`Row2VecTransformer(mode="tsne")` is rejected at `fit`.** t-SNE is
  non-parametric and has no out-of-sample extension, so a t-SNE transformer
  could never transform. Use `learn_embedding(df, mode="tsne")` for a one-off
  embedding. `Row2VecModel.transform` likewise raises for t-SNE;
  `fit_transform` works.
- **`create_config_for_mode("target")` requires `reference_column`.** It used
  to build an invalid config behind a `"__placeholder__"` string and then null
  it. Pass the column:
  `create_config_for_mode("target", reference_column="Country")`.
- **A missing value in `reference_column` is an error.** Target mode used to
  drop those rows and reset the index, quietly changing both the row count and
  which row each embedding referred to. Drop or fill them yourself.

### What changes quietly

These are the ones to check your code against.

- **The output index is now `df`'s own index.** It was a fresh `RangeIndex`, so
  `pd.concat([df, embeddings], axis=1)` silently produced twice the rows and a
  wall of NaN for any frame whose index was not already `0..n-1`. If you worked
  around this by resetting the index, you can stop.
- **`mode="target"` returns one row per input row.** It returned one row per
  distinct reference value. For the per-category matrix, pass
  `aggregate_by_reference=True`:

  ```python
  # 0.2.0 behaviour, now explicit
  country_vectors = row2vec.learn_embedding(
      df,
      mode="target",
      reference_column="Country",
      aggregate_by_reference=True,
  )
  ```

- **Embeddings change numerically**, for three reasons: categorical features
  are now scaled, so a high-cardinality identifier no longer dominates the
  result; target encoding no longer leaks the row's own label into its
  features; and preprocessing is fitted on the training split only, so the
  validation loss that drives early stopping is no longer inflated.
- **`Row2VecTransformer.transform` no longer retrains.** It used to re-run the
  whole training on whatever frame it was handed, so inside `cross_val_score`
  each fold learned its embedding from its own test fold. Cross-validation
  scores computed with 0.2.0 were optimistic; expect them to drop, and to be
  worth something.
- **The CLI writes the index by default.** Output can now be joined back to the
  input rather than matched positionally. Pass `--no-index` for the old
  behaviour.
- **The CLI prints.** It previously exited 1 in silence on every error path.
- **`generate_synthetic_data` no longer seeds the global `random` module.** If
  you relied on that side effect to make unrelated code reproducible, seed it
  yourself.
- **Auto-dimension selection abstains instead of guessing.** Each selection
  method used to answer failure with the midpoint of the candidate list and a
  score of 0.5, indistinguishable from a real recommendation. Failed methods
  now cast no vote.

### What is fixed

No action needed, but worth knowing your results were affected:

- A mid-cardinality categorical column (roughly 100-1000 distinct values) was
  encoded as raw, unscaled ordinal codes while numeric features were
  standardised. With a 150-level identifier, the first principal component
  correlated 1.0000 with the *alphabetical rank* of that identifier.
- A single unseen category at inference zeroed the encoded column for **every**
  row, not just the offending one.
- A column with no missing values during `fit` was never given an imputer, so
  missing values arriving later passed through into the model as NaN.
- A saved model ignored `scale_method`, so `predict` returned values on a
  different scale than training had.
- `mode="contrastive"` with `auto_pairs="categorical"` indexed the feature
  matrix with pandas index *labels*, raising `IndexError` on a shifted index
  and pairing the wrong rows on a permuted one.
- Integer column names raised `KeyError` for any mode.
- `auto_architecture=True` raised `KeyError: 'hidden_units'` on contrastive
  mode, because every trial had its contrastive settings stripped and failed.
- `max_time` on the architecture search was never enforced.
- `NeuralConfig.activation` was dropped when a config was serialised, so the
  architecture search compared models that were byte-identical.
- t-SNE could not run at all on scikit-learn 1.7+.

## Migrating to 0.2.0

Version 0.2.0 restructures the repository and rebuilds the documentation. **The
public API is unchanged** — `import row2vec` and every function and class it
exposes work exactly as before. If you only use the library, you need to change
nothing.

The notes below matter if you develop on Row2Vec, depend on its internals, or
link to its documentation.

### For users

Nothing to do. The package still installs as `row2vec` and imports as
`row2vec`; the move to a `src/` layout is invisible from the outside.

Two behavioural fixes may change what you observe, both in the direction of
working where it previously failed:

- **pandas 3.0 compatibility.** Text columns are inferred as `str` rather than
  `object` in pandas 3.0. Row2Vec classified columns with a dtype test that
  missed this, so under pandas 3 a text column was routed into the numeric
  branch and `learn_embedding` raised `TypeError: Cannot perform reduction
  'mean' with string dtype`. Column classification now goes through a single
  predicate that recognises object, categorical, and string dtypes under both
  pandas 2 and 3.
- **All-missing columns no longer crash the imputer.** `AdaptiveImputer` raised
  `ValueError: Length of values (0) does not match length of index` on a column
  with no observed values. It now leaves such a column unchanged and emits a
  `UserWarning`, since there is no defensible value to impute.

### For contributors

**The package moved to `src/row2vec/`.** Reinstall in editable mode after
pulling:

```bash
pip install -e ".[dev,docs]"
```

Paths in tooling changed accordingly — `mypy` now checks `src/row2vec`, and
`bandit` scans `src/row2vec/`. The `Makefile` targets are updated; use them
rather than remembering paths.

**Dependency declarations consolidated.** `requirements.txt` and
`requirements-dev.txt` are gone; `pyproject.toml` extras are now the single
source of truth:

| Was | Now |
|-----|-----|
| `pip install -r requirements.txt` | `pip install -e .` |
| `pip install -r requirements-dev.txt` | `pip install -e ".[dev]"` |
| (docs dependencies were in `jupyter_book/requirements.txt`) | `pip install -e ".[docs]"` |

**Quality gates are real now.** CI previously *counted* mypy errors and passed
if there were fewer than 50. It now requires a clean `mypy` run, a clean
`ruff check`, `ruff format --check`, and a clean `bandit` scan. `ruff` and
`mypy` are pinned to exact versions so local, pre-commit, and CI runs agree.

**Documentation examples are executed.** `tests/test_doctests.py` runs every
`>>>` example in every docstring, and `tests/test_docs_examples.py` runs every
```python block in `README.md` and `docs/USER_GUIDE.md`. An example referencing
a renamed API now fails the suite. See `CONTRIBUTING.md` for the conventions.

### Documentation moved

| Was | Now |
|-----|-----|
| `evotext.github.io/row2vec` (Jupyter Book) | [`row2vec.tresoldi.org`](https://row2vec.tresoldi.org/) (MkDocs) |
| `docs/API_REFERENCE.md` (hand-written) | [API Reference](https://row2vec.tresoldi.org/reference/), generated from docstrings |
| `docs/LLM_DOCUMENTATION.md` | removed; the User Guide and generated reference cover the same ground |
| `ROADMAP.md` | removed; see [`ARCHITECTURE.md`](ARCHITECTURE.md) for structure and design decisions |
| `jupyter_book/` | removed; the MkDocs site supersedes it |
| `DOCS_DEPLOYMENT.md`, `build_docs.sh` | removed; `make site` builds, CI deploys |
| `check_types.sh` | removed; `make quality` runs mypy |

The repository URL is now `github.com/tresoldi/row2vec`. References to
`evotext/row2vec` were updated throughout.
