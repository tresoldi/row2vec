# Migration Guide

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
