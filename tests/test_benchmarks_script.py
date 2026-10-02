"""The benchmark script's reporting, tested without running a benchmark."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from types import ModuleType

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "benchmarks.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("benchmarks_script", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_markdown_has_one_table_per_metric_and_lists_what_did_not_run() -> None:
    results = pd.DataFrame(
        {
            "rows": [100, 100, 100, 200, 200, 200],
            "mode": ["pca", "target", "baseline"] * 2,
            "status": ["ok", "skipped", "baseline"] * 2,
            "trustworthiness": [0.9, None, 1.0, 0.91, None, 1.0],
            "downstream_score": [0.7, None, 0.8, 0.72, None, 0.8],
            "downstream_metric": ["accuracy"] * 6,
            "fit_seconds": [0.1, None, None, 0.2, None, None],
            "note": ["", "numeric target", "", "", "numeric target", ""],
        }
    )

    text = _load().to_markdown(results, dim=4, epochs=5)

    assert text.count("### ") == 4
    assert "| mode | 100 | 200 |" in text
    assert "numeric target" in text
