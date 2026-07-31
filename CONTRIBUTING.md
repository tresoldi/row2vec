# Contributing to Row2Vec

Thanks for your interest in improving Row2Vec! This document explains how to set
up a development environment and the checks your changes are expected to pass.

## Development setup

Row2Vec requires Python 3.10 or newer.

```bash
# Clone your fork and enter the directory
git clone https://github.com/<your-username>/row2vec.git
cd row2vec

# Install the package with all development dependencies
pip install -e ".[dev,docs]"

# (Optional but recommended) install the pre-commit hooks
pre-commit install
```

Installing with the `dev` extra pulls in everything the test suite and the
tooling need: `pytest`, `ruff`, `mypy`, `bandit`, `build`, and the type stubs. If
you only want to run the tests, the lighter `pip install -e ".[test]"` is enough.
The `docs` extra adds MkDocs and mkdocstrings for the documentation site.

The package lives under `src/row2vec/`. An editable install puts it on the path;
there is no need to run anything from the repository root for imports to resolve.

## Quality checks

All of the checks below run in CI. You can run them locally through the
`Makefile` targets:

```bash
make quality   # ruff format --check, ruff check, mypy
make security  # bandit static security analysis
make test      # run the test suite
make test-cov  # run the test suite with a coverage report
make site      # build the documentation site (strict)
```

Or invoke the tools directly:

```bash
ruff format --check .
ruff check .
mypy
bandit -c pyproject.toml -r src/row2vec/
pytest tests/
```

`ruff` and `mypy` are pinned to exact versions in the `dev` extra and in
`.pre-commit-config.yaml`, so local runs, pre-commit, and CI all agree. If you
bump one, bump it in both places in the same commit.

## Documentation is executed

Row2Vec's documentation is kept honest by running it:

- `tests/test_doctests.py` runs every `>>>` example in every docstring.
- `tests/test_docs_examples.py` runs every ```python block in `README.md` and
  `docs/USER_GUIDE.md`.

This means an example that references a renamed or non-existent API fails the
test suite rather than shipping. Two conventions follow from it:

1. **Each block must stand on its own.** Blocks do not share state — import what
   you need inside the block.
2. **Keep examples fast.** Use small frames and few epochs; the whole point is
   that they run on every commit. A block that genuinely cannot run can opt out
   with a `# docs-test: skip` marker on its first line.

The API reference is generated from docstrings by mkdocstrings, so a docstring
is user-facing documentation. Write them in Google style, with a runnable
example where the function's use is not obvious.

## Making changes

1. Create a feature branch off `main`.
2. Make your change, adding or updating tests to cover it.
3. Update `CHANGELOG.md` under the `[Unreleased]` section.
4. If your change alters the structure or a design decision, update
   `ARCHITECTURE.md` in the same pull request.
5. Ensure `make quality`, `make security`, and `make test` all pass.
6. Open a pull request describing the motivation and the change.

## Reporting bugs and requesting features

Please use the [issue tracker](https://github.com/tresoldi/row2vec/issues). For
bug reports, include a minimal reproducible example, the Row2Vec version, your
Python version, and your pandas and TensorFlow versions.

## License

By contributing, you agree that your contributions will be licensed under the
MIT License that covers the project.
