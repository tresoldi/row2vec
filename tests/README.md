# Row2Vec Tests

Run the suite with `make test`, or with coverage via `make test-cov`. `make
test-fast` runs in parallel and skips anything marked `slow`.

## What is here

| File | Covers |
|------|--------|
| `test_core.py` | `learn_embedding` across every mode, and its validation |
| `test_validation.py` | Parameter validation and the errors it raises |
| `test_config_api.py` | `EmbeddingConfig` and the `*_v2` entry points |
| `test_imputation.py` | `AdaptiveImputer`, `MissingPatternAnalyzer` |
| `test_categorical_encoding.py` | Encoder selection and the encoders themselves |
| `test_contrastive.py`, `test_minimal_contrastive.py` | Contrastive mode |
| `test_architecture_search.py` | `search_architecture` |
| `test_serialization.py` | Saving and loading a model with its pipeline |
| `test_cli_unified.py`, `test_contrastive_cli.py` | The command-line parser |
| `test_integration.py`, `test_integrations.py` | Real datasets; sklearn and pandas integrations |
| `test_performance.py` | Benchmarks and performance regressions |
| `test_logging.py`, `test_utils.py`, `test_fixtures.py` | Supporting modules and fixtures |
| `test_doctests.py` | **Every `>>>` example in every docstring** |
| `test_docs_examples.py` | **Every `python` block in the README and User Guide** |

The last two are what keep the documentation honest: an example that references
a renamed or non-existent API fails the suite instead of shipping. See
`CONTRIBUTING.md` for the conventions they impose on doc authors.

## Markers

Declared in `pyproject.toml`: `slow`, `integration`, `unit`, `neural`, `docs`.
Deselect with, for example, `pytest -m "not slow"`.

## Fixtures

`tests/fixtures/` holds pickled reference outputs used by the regression tests.
Regenerate them deliberately — a changed fixture means a changed result.
