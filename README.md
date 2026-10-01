# Row2Vec

[![CI](https://github.com/tresoldi/row2vec/actions/workflows/quality.yml/badge.svg)](https://github.com/tresoldi/row2vec/actions/workflows/quality.yml)
[![codecov](https://codecov.io/gh/tresoldi/row2vec/branch/main/graph/badge.svg)](https://codecov.io/gh/tresoldi/row2vec)
[![Docs](https://img.shields.io/badge/docs-mkdocs-blue.svg)](https://row2vec.tresoldi.org/)
[![PyPI version](https://badge.fury.io/py/row2vec.svg)](https://badge.fury.io/py/row2vec)
[![Python versions](https://img.shields.io/pypi/pyversions/row2vec.svg)](https://pypi.org/project/row2vec/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

**Turn table rows into vectors.**

Row2Vec takes a `DataFrame` whose columns are a mix of numbers, categories, and
missing values, and returns one dense numeric vector per row. The scaling,
encoding, and imputation are handled for you, and both neural and classical
methods sit behind the same call.

```python
import row2vec

df = row2vec.generate_synthetic_data(200)  # mixed numeric + categorical columns
embeddings = row2vec.learn_embedding(df, mode="pca", embedding_dim=3)

assert embeddings.shape == (200, 3)  # one row in, one vector out
```

That is the whole point: everything downstream of a table — clustering,
nearest-neighbour lookup, a scatter plot, a feature for another model —
presupposes that rows are already points in a metric space. Getting them there
usually means a hand-rolled pipeline of encoders, scalers, and imputers.
Row2Vec is that pipeline, with several well-tested ways to do the projection
itself behind one consistent interface.

## Install

```bash
pip install row2vec
```

## The interface

One function does the work. Swapping methods means changing `mode`; the call
site never changes.

```python
import row2vec

df = row2vec.generate_synthetic_data(150)

pca = row2vec.learn_embedding(df, mode="pca", embedding_dim=2)
tsne = row2vec.learn_embedding(df, mode="tsne", embedding_dim=2, perplexity=10)

assert pca.shape == tsne.shape == (150, 2)
assert list(pca.index) == list(df.index)  # aligned with the input frame
```

The result is a `DataFrame` of the same length as the input, sharing its index,
with one column per embedding dimension — so it joins straight back onto the
original data.

A trained model can be saved with its preprocessing pipeline and reloaded to
embed new rows the same way:

```python
import tempfile
from pathlib import Path

import row2vec

df = row2vec.generate_synthetic_data(100)
base = Path(tempfile.mkdtemp()) / "model"

embeddings, script_path, _binary_path = row2vec.train_and_save_model(
    df, base_path=str(base), mode="pca", embedding_dim=2
)
# load_model() executes the saved .py loader script and unpickles its blob,
# so only load models you trust. See SECURITY.md.
model = row2vec.load_model(script_path)

# The reloaded model reproduces what training returned, scaling included.
assert model.predict(df).shape == embeddings.shape
```

## Choosing a method

| Mode | Use it for | Key parameter |
|------|------------|---------------|
| `pca` | a fast linear baseline with interpretable components | `embedding_dim` |
| `unsupervised` | non-linear structure, via an autoencoder | `hidden_units`, `max_epochs` |
| `tsne` | 2-D/3-D plots that show local structure and clusters | `perplexity` |
| `umap` | preserving local *and* global structure | `n_neighbors`, `min_dist` |
| `target` | supervision from a label column; `aggregate_by_reference=True` for one vector per category | `reference_column` |
| `contrastive` | supervision from pairs known to be alike or unalike | `auto_pairs`, `margin` |

Start with `pca`. It costs nothing to run and tells you whether the structure
you are after is linear; if it isn't, `unsupervised` is the next step.

## What you don't have to write

- **Missing values** — the pattern of missingness is analysed per column and an
  imputation strategy chosen to match (`AdaptiveImputer`).
- **Categorical columns** — encoded by a strategy picked from the column's
  cardinality: one-hot for low cardinality, learned entity embeddings above it,
  and target encoding where a usable target makes it meaningful.
- **Scaling** — numeric *and* encoded columns are brought onto comparable
  scales before the projection, so no single column decides the result, and the
  output can be rescaled (`minmax`, `standard`, `l2`, `tanh`). The scaler is
  kept with the model, so a saved model reproduces its training output.
- **Architecture** — for the neural modes, layer widths and even the embedding
  dimension can be searched rather than guessed (`search_architecture`,
  `auto_select_dimension`).

## Also included

A **command-line interface** for batch work:

```bash
row2vec annotate --input data.csv --output embeddings.csv --mode pca --dim 5
row2vec train --input data.csv --output model.py --mode unsupervised --dim 10
row2vec predict --input new.csv --model model.py --output predictions.csv
```

A **scikit-learn** transformer and classifier (`Row2VecTransformer`,
`Row2VecClassifier`) for use inside a `Pipeline`, and a **pandas accessor**
(`df.row2vec.pca(dim=2)`) for quick interactive work.

## Why Row2Vec

- **One consistent API** across six embedding methods — swap the projection
  without rewriting your preprocessing.
- **Preprocessing travels with the model** — a saved model carries its own
  encoders and imputers, so inference on new data reproduces training exactly.
- **Typed and production-ready** — full type hints (`py.typed`), strict linting
  and type-checking, and a test suite run across Python 3.10–3.12 on Linux,
  macOS, and Windows. Every example in this README and in the User Guide is
  executed by that suite.

## Documentation

- **[Documentation site](https://row2vec.tresoldi.org/)** — user guide and full
  API reference.
- **[User Guide](docs/USER_GUIDE.md)** — concepts, choosing a method, and worked
  examples.
- **[API Reference](https://row2vec.tresoldi.org/reference/)** — every public
  class and function, generated from the source.
- **[Architecture](ARCHITECTURE.md)** — how the package is put together, and why.

## Citation

If you use Row2Vec in academic research, please cite:

```bibtex
@software{tresoldi_row2vec_2026,
  author = {Tresoldi, Tiago},
  title = {Row2Vec: Neural and classical embeddings for tabular data},
  url = {https://github.com/tresoldi/row2vec},
  version = {0.2.0},
  year = {2026}
}
```

## Acknowledgments

This library was originally developed as part of the **"Cultural Evolution of
Texts"** project, led by Michael Dunn at the Department of Linguistics and
Philology, Uppsala University.

## License

MIT — see [LICENSE](LICENSE).
