#!/usr/bin/env python3
"""Benchmark the embedding modes across dataset sizes.

Each size is a random sample of the bundled Adult census data (mixed numeric and
categorical columns, with a categorical target). For every size, ``compare_modes``
fits each mode on a training split and scores it on held-out rows, so the table
reports cost *and* quality together:

* ``fit_seconds``      - wall-clock time to fit the mode,
* ``trustworthiness``  - neighbourhood preservation on held-out rows,
* ``downstream_score`` - k-NN accuracy on the embedding, held out.

Memory is not measured: TensorFlow allocates outside anything Python can see, so
a number here would flatter the neural modes.

Usage:
    python scripts/benchmarks.py --quick --output benchmark_results_quick
    python scripts/benchmarks.py --output benchmark_results

Writes ``results.csv`` and ``results.md`` to the output directory. Timings
depend on the machine; compare modes with each other, not with other hardware.
"""

from __future__ import annotations

import argparse
import platform
import time
from pathlib import Path

import pandas as pd

import row2vec

DATA = Path(__file__).resolve().parent.parent / "data" / "adult.csv"
TARGET = "income"
QUICK_SIZES = [300, 1000]
FULL_SIZES = [500, 1000, 2000, 4000, 8000]
SEED = 1305


def run_benchmarks(sizes: list[int], dim: int, epochs: int, output: Path) -> pd.DataFrame:
    """Run ``compare_modes`` at each size and return one tidy table."""
    adult = pd.read_csv(DATA)
    # One-off costs (numba compilation for UMAP, TensorFlow graph set-up) would
    # otherwise land on whichever size runs first and make it look slowest.
    print("-- warm-up", flush=True)
    row2vec.compare_modes(
        adult.sample(200, random_state=SEED), target=TARGET, embedding_dim=dim, max_epochs=2
    )
    rows = []
    for size in sizes:
        frame = adult.sample(min(size, len(adult)), random_state=SEED).reset_index(drop=True)
        print(f"-- {len(frame)} rows", flush=True)
        report = row2vec.compare_modes(
            frame,
            target=TARGET,
            embedding_dim=dim,
            max_epochs=epochs,
            perplexity=30.0,
        )
        report.insert(0, "mode", report.index)
        report.insert(0, "rows", len(frame))
        rows.append(report.reset_index(drop=True))
        print(report.drop(columns="note").round(3).to_string(), flush=True)

    results = pd.concat(rows, ignore_index=True)
    output.mkdir(parents=True, exist_ok=True)
    results.to_csv(output / "results.csv", index=False)
    (output / "results.md").write_text(to_markdown(results, dim, epochs))
    return results


def _table(frame: pd.DataFrame) -> str:
    """A pipe table, without needing the optional ``tabulate`` package."""
    frame = frame.reset_index()
    cells = frame.astype(object).where(frame.notna(), "-").astype(str)
    lines = ["| " + " | ".join(map(str, frame.columns)) + " |"]
    lines.append("|" + "|".join("---" for _ in frame.columns) + "|")
    lines += ["| " + " | ".join(row) + " |" for row in cells.to_numpy()]
    return "\n".join(lines)


def to_markdown(results: pd.DataFrame, dim: int, epochs: int) -> str:
    """Render the results as Markdown, one table per metric."""
    ok = results[results["status"].isin(["ok", "baseline"])]
    out = [
        f"Run {time.strftime('%Y-%m-%d')} on {platform.platform()}, Python "
        f"{platform.python_version()}, row2vec {row2vec.__version__}. "
        f"Embedding dimension {dim}, {epochs} epochs for neural modes.\n"
    ]
    for column, title in [
        ("fit_seconds", "Fit time (seconds)"),
        ("trustworthiness", "Trustworthiness (held-out rows)"),
        ("downstream_score", "k-NN accuracy on the embedding (held-out rows)"),
    ]:
        table = ok.pivot(index="mode", columns="rows", values=column)
        out.append(f"### {title}\n\n{_table(table.round(3))}\n")
    out.append(
        "t-SNE is scored on at most 1500 rows (it cannot embed unseen rows, and its cost "
        "grows quickly), so its time stops growing past that size.\n"
    )
    skipped = results[~results["status"].isin(["ok", "baseline"])]
    if len(skipped):
        notes = skipped.drop_duplicates("mode")[["mode", "status", "note"]]
        out.append("### Not run\n\n" + _table(notes.set_index("mode")) + "\n")
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark row2vec modes across dataset sizes.",
        epilog="See the module docstring for what is measured.",
    )
    parser.add_argument("--quick", action="store_true", help=f"small sizes only {QUICK_SIZES}")
    parser.add_argument("--sizes", type=int, nargs="+", help="override the row counts")
    parser.add_argument("--dim", type=int, default=4, help="embedding dimension")
    parser.add_argument("--epochs", type=int, default=20, help="epochs for neural modes")
    parser.add_argument("--output", type=Path, default=Path("benchmark_results"))
    args = parser.parse_args()

    sizes = args.sizes or (QUICK_SIZES if args.quick else FULL_SIZES)
    run_benchmarks(sizes, args.dim, args.epochs, args.output)
    print(f"\nWrote {args.output}/results.csv and results.md")


if __name__ == "__main__":
    main()
