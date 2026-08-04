"""Execute every ``python`` code block in the prose docs.

The README and the docs site are full of runnable examples. This extracts each
fenced ``python`` block from those Markdown files and executes it, so an example
that references a renamed or non-existent API fails the test suite instead of
shipping.

Conventions for doc authors:
* Each ``python`` block must run on its own in a fresh namespace — import what
  it needs. Blocks do not share state.
* Keep them fast: small frames, few epochs. They run on every commit.
* A block that is deliberately illustrative / not meant to run can opt out with
  a marker comment: ``# docs-test: skip``.

For a specific failure, the test id names the file and the 1-based block index.
"""

from __future__ import annotations

import re
import traceback
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DOC_FILES = [
    _REPO_ROOT / "README.md",
    _REPO_ROOT / "docs" / "USER_GUIDE.md",
    _REPO_ROOT / "CONTRIBUTING.md",
]

_FENCE = re.compile(r"^```python\b[^\n]*\n(.*?)^```", re.MULTILINE | re.DOTALL)


def _extract_blocks(path: Path) -> list[str]:
    """Return the ``python`` fenced code blocks in a Markdown file."""
    if not path.exists():
        return []
    return _FENCE.findall(path.read_text(encoding="utf-8"))


def _collect() -> list[tuple[str, int, str]]:
    """Collect (file-label, 1-based index, code) for every runnable block."""
    cases: list[tuple[str, int, str]] = []
    for path in _DOC_FILES:
        for i, block in enumerate(_extract_blocks(path), start=1):
            if "# docs-test: skip" in block:
                continue
            cases.append((path.relative_to(_REPO_ROOT).as_posix(), i, block))
    return cases


_CASES = _collect()


def test_documentation_has_runnable_examples() -> None:
    """Guard against the extractor silently matching nothing."""
    assert len(_CASES) > 10, f"expected many runnable doc blocks, found {len(_CASES)}"


@pytest.mark.docs
@pytest.mark.parametrize(
    ("doc_file", "index", "code"),
    _CASES,
    ids=[f"{f}#block{i}" for f, i, _ in _CASES],
)
def test_doc_code_block_runs(doc_file: str, index: int, code: str) -> None:
    """Every ``python`` block in the prose docs must execute without error."""
    label = f"{doc_file}#block{index}"
    namespace: dict[str, object] = {"__name__": "__doc_example__"}
    try:
        exec(compile(code, label, "exec"), namespace)
    except Exception as exc:  # pragma: no cover - the message is the point
        # A bare `assert` carries no message, so the exception alone says
        # nothing about which line failed. Report the offending line from the
        # traceback — the block is compiled under `label`, so the frames point
        # back into the documentation.
        # The block was compiled from a string, so linecache cannot supply the
        # source; look the line up in `code` by number instead.
        source_lines = code.splitlines()
        failing = [
            f"  line {lineno}: {source_lines[lineno - 1].strip()}"
            for lineno in (
                frame.lineno
                for frame in traceback.extract_tb(exc.__traceback__)
                if frame.filename == label
            )
            if lineno is not None and 0 < lineno <= len(source_lines)
        ]
        where = "\n".join(failing) or "  (no line attributed)"
        pytest.fail(
            f"{label} failed: {type(exc).__name__}: {exc}\n"
            f"failing line(s):\n{where}\n\nfull block:\n{code}"
        )
