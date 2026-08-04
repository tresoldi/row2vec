# CHANGELOG

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/tresoldi/row2vec/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/tresoldi/row2vec/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/tresoldi/row2vec/releases/tag/v0.1.0
