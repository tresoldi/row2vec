"""Execute every docstring example in the package as a doctest.

The API reference is generated from docstrings (mkdocstrings), so their code
examples are user-facing documentation. Running them here in the normal test
suite guarantees they stay correct against the installed code — an example can
never silently rot.

This collects and runs the ``>>>`` examples in every ``row2vec`` submodule. For
a readable per-line report when something fails, run:

    python -m pytest --doctest-modules src/row2vec
"""

from __future__ import annotations

import doctest
import importlib
import pkgutil

import pytest

import row2vec

_NEEDS_TENSORFLOW = "NeuralBackendMissing"


class _RecordingRunner(doctest.DocTestRunner):
    """A doctest runner that remembers the exceptions examples raised."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self.errors: list[str] = []

    def report_unexpected_exception(self, out, test, example, exc_info):  # type: ignore[no-untyped-def]
        self.errors.append(f"{type(exc_info[1]).__name__}: {exc_info[1]}")


def _iter_module_names() -> list[str]:
    """Return the import paths of every submodule under ``row2vec``."""
    names = [row2vec.__name__]
    for info in pkgutil.walk_packages(row2vec.__path__, prefix=f"{row2vec.__name__}."):
        names.append(info.name)
    return sorted(names)


@pytest.mark.docs
@pytest.mark.parametrize("module_name", _iter_module_names())
def test_module_docstring_examples(module_name: str) -> None:
    """Every ``>>>`` example in the module must run and match its output."""
    module = importlib.import_module(module_name)
    runner = _RecordingRunner(verbose=False)
    for test in doctest.DocTestFinder().find(module):
        runner.run(test, out=lambda _text: None)
    results = runner.summarize(verbose=False)
    if results.failed and any(_NEEDS_TENSORFLOW in message for message in runner.errors):
        pytest.skip("examples in this module train a neural model; install row2vec[neural]")
    assert results.failed == 0, (
        f"{results.failed} doctest example(s) failed in {module_name}. "
        f"Reproduce with: python -m pytest --doctest-modules src/row2vec"
    )
