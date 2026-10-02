"""Compare embedding modes on the bundled real datasets.

Run from the repository root:

    python examples/compare_modes_real_data.py titanic
    python examples/compare_modes_real_data.py adult
    python examples/compare_modes_real_data.py ames

Each dataset is embedded with every mode that can run, and the table shows how
well each one keeps neighbourhoods (``trustworthiness``) and how well a
nearest-neighbour model on the embedding predicts the target, next to a
``baseline`` that uses the preprocessed features directly.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import row2vec

DATA = Path(__file__).resolve().parent.parent / "data"

DATASETS = {
    # file, target column, rows to sample (None = all), t-SNE perplexity
    "titanic": ("titanic.csv", "Survived", None, 15.0),
    "adult": ("adult.csv", "income", 4000, 30.0),
    "ames": ("ames_housing.csv", "SalePrice", None, 30.0),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("dataset", choices=sorted(DATASETS))
    parser.add_argument("--dim", type=int, default=4, help="embedding dimension")
    parser.add_argument("--epochs", type=int, default=20, help="epochs for neural modes")
    args = parser.parse_args()

    filename, target, sample, perplexity = DATASETS[args.dataset]
    df = pd.read_csv(DATA / filename)
    if "Id" in df.columns:
        df = df.drop(columns="Id")  # a row counter, not a feature
    if sample is not None and len(df) > sample:
        df = df.sample(sample, random_state=1).reset_index(drop=True)
    print(f"{args.dataset}: {len(df)} rows, target {target!r}")

    report = row2vec.compare_modes(
        df,
        target=target,
        embedding_dim=args.dim,
        max_epochs=args.epochs,
        perplexity=perplexity,
    )
    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 80)
    print(report.round(3).to_string())


if __name__ == "__main__":
    main()
