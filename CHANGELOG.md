# CHANGELOG

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.4.0] - unreleased

Correctness release, now also a lighter install. An independent audit of 0.2.0 found nine data-integrity
defects that the 278-test suite did not catch, because they changed values
rather than raising. **Embeddings will differ numerically from 0.2.0's** - see
[MIGRATION.md](MIGRATION.md) for everything that changed and why. TensorFlow is
now an optional extra, and `compare_modes` scores modes against each other.

This release was developed as 0.3.0; that version was never published, so its
changes are folded in here.

### Fixed

- **Mid-cardinality categoricals no longer dominate the embedding.** Columns
  with roughly 100–1000 distinct values were encoded as raw, unscaled ordinal
  codes while numeric features were standardised. With a 150-level identifier
  the first principal component correlated 1.0000 with the alphabetical rank of
  that identifier string. The categorical branch is now scaled, and the
  strategy selector no longer picks target encoding when no usable target
  exists.
- **`mode="target"` returns results on any index.** It built its frame with a
  fresh `RangeIndex` and then grouped by a label-indexed Series, so on any
  non-default index every group key was NaN and the result came back empty.
- **Output carries `df`'s index.** `pd.concat([df, embeddings], axis=1)`
  previously duplicated every row for a non-default index.
- **One unseen category no longer zeroes a whole column.** The entity-embedding
  transform assigned into a temporary copy, so an unknown value left every row
  at the all-zero default.
- **Columns clean at fit time are imputed.** Missing values arriving later
  passed straight through into the model as NaN.
- **`Row2VecTransformer.transform` projects instead of retraining.** Inside
  cross-validation each fold previously learned its embedding from its own test
  fold.
- **Saved models reproduce training exactly.** `scale_method` was recorded and
  then never applied at inference.
- **Target encoding no longer leaks the row's own label.** Cross-fitted values
  were computed and discarded.
- **Contrastive pairing uses positions, not index labels**, so a shifted index
  no longer raises `IndexError` and a permuted one no longer pairs wrong rows.
- **Integer column names work.** They raised `KeyError` for every mode.
- **`auto_architecture=True` works for contrastive mode**, and the architecture
  search enforces `max_time`, varies `activation`, scores reconstruction by
  actually reconstructing, and prefers smaller models rather than faster runs.
- **Auto-dimension selection abstains rather than inventing a recommendation**,
  fits PCA over the full component range, uses a real knee construction, and
  cross-validates without leakage.
- **The CLI reports what it did and what went wrong**, rather than exiting 1 in
  silence.
- **t-SNE runs on scikit-learn 1.7+** (`n_iter` → `max_iter`).
- `make quality` is green again; `pandas-stubs` is now pinned like ruff and
  mypy.

### Changed

- **Breaking: models are saved as one `.r2v` file, and loading runs no code.**
  The script-and-pickle format executed the script on load (`exec`) and unpickled
  a blob, so a model file could run anything. A model is now a zip of a JSON
  manifest (versions, metadata, a checksum per member), JSON configuration and
  state, scikit-learn objects written with `skops`, and a neural encoder in
  Keras's own format. skops may only build an exact allow-list of types
  (`row2vec.serialization.TRUSTED_TYPES`), Keras loads with `safe_mode` and only
  `Dense`/`Dropout` layers, and anything else raises `ModelFormatError`. The
  guarantee is against a malicious file, not an unauthenticated one; see
  `SECURITY.md`. `save_model` returns one path, `train_and_save_model` returns
  `(embeddings, path)`, the CLI takes `.r2v` files, and `load_model` refuses
  `.py`/`.pkl` without running them. Neural models store only the encoder, not the
  training network. UMAP models save and reload with identical output. `skops` is
  a new core dependency.
- **Breaking:** `learn_embedding_with_model` returns `(embeddings, model)`
  rather than a four-tuple.
- **Breaking:** `mode="target"` returns one row per input row;
  `aggregate_by_reference=True` asks for the per-category matrix.
- **Breaking:** `Row2VecTransformer` takes explicit parameters instead of
  `**kwargs`, so `get_params`/`set_params`/`clone` round-trip.
- **Breaking:** `Row2VecTransformer(mode="tsne")` and `Row2VecModel.transform`
  for t-SNE raise; t-SNE has no out-of-sample extension.
- **Breaking:** `create_config_for_mode("target")` requires `reference_column`.
- **Breaking:** a missing value in `reference_column` is an error rather than a
  silent row drop.
- **Breaking:** models saved by 0.2.0 cannot be loaded.
- Preprocessing is fitted on the training split only, so validation loss and
  early stopping are no longer inflated.
- The CLI writes the index by default; `--no-index` opts out.
- `generate_synthetic_data` no longer seeds the global `random` module.
- **Breaking:** TensorFlow is no longer a hard dependency. `pip install row2vec`
  gives PCA, t-SNE and UMAP; the autoencoder, target and contrastive modes,
  architecture search, and entity-embedding categorical encoding need
  `pip install "row2vec[neural]"`. Using one without it raises
  `row2vec._backend.NeuralBackendMissing` (an `ImportError`) naming the extra.
- `import row2vec` no longer imports TensorFlow, and fitting PCA, t-SNE or UMAP
  never loads it.
- Without TensorFlow, a high-cardinality categorical column that would get
  entity embeddings uses target encoding when a usable target exists (with a
  warning), and otherwise raises rather than falling back to an unbounded
  encoding.
- `Row2VecTrainingCallback` is built on first use; it is still available as
  `row2vec.model.Row2VecTrainingCallback` and `row2vec.core.Row2VecTrainingCallback`.
- `row2vec[dev]` and `row2vec[all]` include the `neural` extra.

### Added

- `row2vec.inspect_model(path)` reads a saved model's manifest (format and library
  versions, mode, metadata) without deserialising anything else, and
  `row2vec.ModelFormatError`.
- `row2vec.model.Row2VecModel`, the single object holding fitted state: the
  preprocessor, the projector, the encoder, and the embedding scaler. Every
  entry point is now a thin facade over it.
- `aggregate_by_reference` for target mode.
- `activation` as a parameter of `learn_embedding`; it was previously hardcoded
  to `"relu"` whatever the configuration said.
- `--no-index` for the CLI.
- `tests/test_regressions.py` and `tests/test_invariants.py`: a failing-first
  test per audited defect, plus properties checked across every mode and
  scikit-learn's full `check_estimator` conformance suite over both adapters.
- Working `slow` and `neural` pytest markers, so `make test-fast` deselects
  something.
- **`row2vec.compare_modes`**: fits several modes on one table and scores them
  on held-out rows, returning a table of trustworthiness, a downstream k-NN score
  when a `target` column is given, and fit time, alongside a no-embedding
  baseline. The target is withheld from every mode except `mode="target"`;
  t-SNE is scored on all rows without a downstream score because it has no
  out-of-sample extension; modes needing TensorFlow are reported `unavailable`
  rather than raising.
- A `light` CI job that runs the suite without TensorFlow, and
  `tests/test_optional_neural.py`, which checks in a fresh interpreter that the
  classical modes work and the neural ones fail helpfully when TensorFlow is
  blocked.

### Removed

- The duplicate training implementation in `core.py`. `learn_embedding_with_model`
  trained a second, divergent model after already calling `learn_embedding`, at
  roughly 1.8x the cost; `core.py` goes from 1919 lines to 355.
- `row2vec.api._config_to_legacy_params`, the flattening bridge that silently
  dropped `activation` and every contrastive setting.

## [0.2.0] - 2026-07-31

Structural release: the repository, its tooling, and its documentation were
rebuilt to match the layout and conventions of
[freqprob](https://github.com/tresoldi/freqprob), so both projects share one
shape. **The public API is unchanged** — see [MIGRATION.md](MIGRATION.md).

### Added

- **Documentation site.** A MkDocs-Material site published to
  [row2vec.tresoldi.org](https://row2vec.tresoldi.org/), replacing the Jupyter
  Book that was pushed cross-repository into `evotext.github.io`. It is three
  pages: a self-contained landing page (a dry, academic layout with a figure
  illustrating the row-to-vector mapping), a single flowing **User Guide**, and
  an **API Reference generated entirely from docstrings** via mkdocstrings, so
  there is no hand-maintained reference left to drift.
- **All documentation examples are executed in CI.** `tests/test_doctests.py`
  runs every docstring example and `tests/test_docs_examples.py` runs every
  Python code block in the README, User Guide, and contributing guide. Adding
  these immediately surfaced 27 broken docstring examples, all of which are now
  fixed and runnable.
- **`ARCHITECTURE.md`**, recording the package structure, the inward dependency
  rule, the design principles, and the resolved design decisions.
- **`CONTRIBUTING.md`**, **`SECURITY.md`**, **`CITATION.cff`**, and
  **`MIGRATION.md`**.
- **Release automation**: `.github/workflows/release.yml` builds, checks, and
  publishes to PyPI via trusted publishing on a `v*` tag.
- **`.github/dependabot.yml`** and **`codecov.yml`**.
- `Makefile` targets `security`, `site`, `site-serve`, and `tutorials`.
- **New test modules** for the least-covered areas: `tests/test_cli_commands.py`
  drives every CLI subcommand end to end (real files in, real files out), and
  `tests/test_categorical_encoders.py` covers encoder selection and the
  encoders themselves.
- Shared dtype-classification helpers in `row2vec.utils`
  (`is_categorical_series`, `categorical_columns`, `numeric_columns`), so the
  numeric/categorical decision is made in exactly one place.

### Changed

- **`src/` layout.** The package moved from `row2vec/` to `src/row2vec/`. The
  import path is unchanged; contributors should reinstall with
  `pip install -e ".[dev,docs]"`.
- **Quality gates are enforced rather than counted.** The old CI passed if
  `mypy` reported fewer than 50 errors; it now requires a clean `mypy` run over
  the package, tests, and scripts, a clean `ruff check`, `ruff format --check`,
  and a clean `bandit` scan. Reaching that took annotating the test suite,
  narrowing bare `except` clauses, adding exception chaining, and removing dead
  code.
- **Ruff configuration rebuilt** around freqprob's rule set: the blanket
  ignore list shrank from ~50 entries to 9, each with a stated reason.
- **`ruff` and `mypy` are pinned to exact versions** in the `dev` extra and in
  `.pre-commit-config.yaml`, so local, pre-commit, and CI runs cannot drift.
  `mypy` and the standard hygiene hooks are enabled in pre-commit (both were
  previously commented out).
- **CI test matrix widened** to Python 3.10–3.12 on Linux, macOS, and Windows.
- **Coverage raised from 46.58% to 72.17%** (91 tests to 277), and the gate
  raised from 46% to 70%. Coverage is now measured with branch coverage, and
  `codecov.yml` requires 80% on newly changed lines.
- **Dependency declarations consolidated.** `requirements.txt` and
  `requirements-dev.txt` were removed; `pyproject.toml` extras
  (`test`/`docs`/`dev`/`all`) are now the single source of truth.
- **Docstrings converted to Google style** in `pandas.py`, `sklearn.py`,
  `pipeline_builder.py`, and `categorical_encoding.py`, which used NumPy-style
  sections that mkdocstrings would have rendered as raw text.
- **README rewritten** around what the library does and when to use each mode.
- Repository URLs updated from `evotext/row2vec` to `tresoldi/row2vec`.
- Script-style tests that returned booleans and swallowed failures — so they
  could never fail — were rewritten as real pytest tests with assertions
  (`test_contrastive.py`, `test_contrastive_cli.py`, `test_minimal_contrastive.py`,
  `test_cli_unified.py`, `test_imputation.py`).

### Fixed

- **pandas 3.0 compatibility.** Text columns are inferred as `str` rather than
  `object` in pandas 3.0, and Row2Vec's column classification used a dtype test
  that missed this — routing text into the numeric branch, where
  `learn_embedding` failed with `TypeError: Cannot perform reduction 'mean' with
  string dtype`. This broke 88 of 210 tests against pandas 3.0. Classification
  now goes through a single predicate valid under both pandas 2 and 3.
- **All-missing columns no longer crash `AdaptiveImputer`**, which raised
  `ValueError: Length of values (0) does not match length of index`. Such a
  column is now left unchanged with a `UserWarning`, since there is no
  defensible value to impute.
- **Schema validation accepts pandas 3 string dtypes**: `_are_compatible_dtypes`
  did not treat `str` as compatible with `object`/`string`.
- A syntax error in `examples/architecture_search_examples.py`, where a
  truncated string literal had swallowed the following statement.
- `learn_embedding_with_model` and `train_and_save_model` documented their
  parameters as "same as learn_embedding"; every parameter is now described, so
  the generated reference is complete.

### Removed

- `jupyter_book/` and its build scripts, superseded by the MkDocs site.
- `docs/API_REFERENCE.md`, superseded by the generated reference.
- `docs/LLM_DOCUMENTATION.md`, `ROADMAP.md`, `DOCS_DEPLOYMENT.md`,
  `build_docs.sh`, `check_types.sh`, `examples/EXAMPLES_STATUS.md`,
  `requirements.txt`, `requirements-dev.txt`.
- `tests/test_documentation_example.py`, a hand-copied snapshot of documentation
  examples, superseded by `tests/test_docs_examples.py`, which executes the real
  documentation.

## [0.1.0] - 2025-10-13

### Added

#### Documentation
- Comprehensive USER_GUIDE.md with mathematical foundations and examples
- LLM_DOCUMENTATION.md for AI coding agent integration
- Nhandu-based executable tutorials (tutorial_1_quickstart.py, tutorial_2_advanced.py)
- Complete API reference documentation

#### Build & Development
- Comprehensive Makefile for development workflow
- CHANGELOG.md following Keep a Changelog format
- scripts/ directory for benchmarking and utilities
- Dynamic version management from `__init__.py`
- Switched build backend from hatchling to setuptools

#### Testing & Quality
- Achieved 47% test coverage with 91 passing tests
- Updated test coverage threshold to 46% (baseline for gradual improvement)
- Test coverage improvement plan: 46% → 50% → 60% → 70%

#### Core Embedding Methods
- Neural autoencoder-based embeddings for unsupervised learning
- Target-based embeddings for categorical features
- Classical methods integration: PCA, t-SNE, UMAP
- Contrastive learning support for enhanced representations

#### Intelligent Preprocessing
- Adaptive missing value imputation with pattern analysis
- Automatic feature scaling and normalization
- Categorical encoding with entity embeddings
- Pattern-aware missing data detection

#### Advanced Features
- Neural Architecture Search (NAS) for optimal network discovery
- Multi-layer neural networks with dropout and regularization
- Automatic dimension selection based on data characteristics
- Model serialization with full preprocessing pipeline preservation

#### Developer Experience
- Command-line interface (CLI) for batch processing
- Configuration file support (YAML)
- Comprehensive type hints with MyPy validation
- Integration with pandas DataFrames (`.row2vec` accessor)
- Scikit-learn compatible transformers and classifiers

#### Testing & Quality
- 163+ test functions across 17 test modules
- Coverage reporting with pytest-cov
- Property-based testing foundations
- Integration tests for full pipelines
- Performance regression tests

#### Documentation
- Interactive Jupyter Book with executable examples
- Real-world tutorials (Titanic, Adult, Housing datasets)
- Complete API reference
- Automated documentation deployment
- Theoretical foundation guide

#### Build & Release
- Modern `pyproject.toml` configuration with hatchling backend
- GitHub Actions CI/CD pipeline
- Multi-version Python testing (3.10, 3.11, 3.12)
- Automated code quality gates (ruff, mypy)
- Pre-commit hooks for code quality

### Technical Details
- Python 3.10+ support
- TensorFlow/Keras backend for neural methods
- Rich console output for CLI
- Logging infrastructure with configurable levels
- Memory monitoring for large datasets

[Unreleased]: https://github.com/tresoldi/row2vec/compare/v0.4.0...HEAD
[0.4.0]: https://github.com/tresoldi/row2vec/compare/v0.1.0...v0.4.0
[0.2.0]: https://github.com/tresoldi/row2vec/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/tresoldi/row2vec/releases/tag/v0.1.0
