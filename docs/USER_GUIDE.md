# User Guide

Row2Vec turns **rows into vectors**. You give it a `DataFrame`; it gives you back
a numeric matrix with one row per input row, having dealt with the mixed dtypes,
the missing values, and the scaling on the way.

Almost everything you want to do with a table downstream — cluster it, find the
nearest neighbours of a record, plot it, feed it to a model that only speaks
floats — assumes the rows are already points in a metric space. This guide covers
the concepts, helps you pick a method, and works through the common tasks. Every
code block on this page is executed by the test suite, so the examples stay
correct against the installed version.

> New here? Read **Core concepts** and **Choosing a method**, then jump to the
> section closest to your problem.

---

## Core concepts

### One call for every method

Every embedding is produced by `learn_embedding`. You choose the method with
`mode`; everything else about the call stays the same.

```python
import row2vec

df = row2vec.generate_synthetic_data(120)
embeddings = row2vec.learn_embedding(df, mode="pca", embedding_dim=3)

assert embeddings.shape == (120, 3)
```

Swapping methods means swapping one argument — the call site never changes.

### The output is a DataFrame aligned with your input

The result has the same length and index as the input, so you can put it
straight back next to the original columns.

```python
import pandas as pd

import row2vec

df = row2vec.generate_synthetic_data(50)
embeddings = row2vec.learn_embedding(df, mode="pca", embedding_dim=2)

assert list(embeddings.index) == list(df.index)
combined = pd.concat([df, embeddings], axis=1)
assert len(combined.columns) == len(df.columns) + 2
```

### Mixed dtypes are handled for you

The input can hold numbers, categories, and gaps at once. Numeric columns are
scaled, categorical columns are encoded, and missing values are imputed — you do
not have to prepare any of it.

```python
import numpy as np
import pandas as pd

import row2vec

df = pd.DataFrame(
    {
        "amount": [10.0, 240.0, 35.5, np.nan, 88.0, 12.0, 300.0, 45.0],
        "region": ["north", "south", "north", "east", "south", "east", "north", "south"],
        "tier": ["a", "b", "a", "b", "a", "b", "a", "b"],
    }
)

embeddings = row2vec.learn_embedding(df, mode="pca", embedding_dim=2)

assert embeddings.shape == (8, 2)
assert not embeddings.isna().to_numpy().any()  # no NaN survives into the output
```

### `embedding_dim` is the width of the output

It is the number of columns you get back — how much room the method has to
describe a row. Two or three for plotting; five to fifty as features for another
model.

```python
import row2vec

df = row2vec.generate_synthetic_data(60)

for dim in (2, 5):
    assert row2vec.learn_embedding(df, mode="pca", embedding_dim=dim).shape[1] == dim
```

If you would rather not pick, see
[Choosing the dimension automatically](#choosing-the-dimension-automatically).

---

## Choosing a method

Start from what you want the vectors *for*:

| Your situation | Mode | Key parameter |
|----------------|------|---------------|
| A fast, interpretable baseline | `pca` | `embedding_dim` |
| The structure is non-linear | `unsupervised` | `hidden_units`, `max_epochs` |
| A 2-D picture showing clusters | `tsne` | `perplexity` |
| A 2-D picture that also keeps global layout | `umap` | `n_neighbors`, `min_dist` |
| One vector per *category*, not per row | `target` | `reference_column` |
| You know which rows are alike | `contrastive` | `auto_pairs`, `margin` |

By data characteristics:

| Rows | Structure | Good default |
|------|-----------|--------------|
| Any | Unknown — you are exploring | `pca` first, then `unsupervised` |
| Thousands+ | Non-linear, plenty of data to fit | `unsupervised` |
| Up to a few thousand | You want to *see* it | `umap` (or `tsne`) |
| Any | You have labelled pairs or a grouping column | `contrastive` |

When-to-use, in one line each:

- **`pca`** — linear, deterministic, and instant. Always worth running first: if
  a few components already separate what you care about, stop here.
- **`unsupervised`** — an autoencoder. Captures interactions PCA cannot, at the
  cost of training time and hyperparameters. Wants a few thousand rows.
- **`tsne`** — for visualisation only. Excellent at revealing clusters, but
  distances *between* clusters are not meaningful, and it cannot embed new rows.
- **`umap`** — usually the better plot: faster than t-SNE and keeps more of the
  global arrangement.
- **`target`** — flips the question around: instead of embedding rows, embed the
  *values* of one column by the rows they occur in.
- **`contrastive`** — supervised by pairs. Use it when you know that certain rows
  should be close (same customer, same cluster, same label).

---

## The methods

### PCA

The linear baseline. Fast, deterministic, and its components come out ordered by
how much variance they account for.

```python
import row2vec

df = row2vec.generate_synthetic_data(200)
embeddings = row2vec.learn_embedding(df, mode="pca", embedding_dim=3)

assert embeddings.shape == (200, 3)
# Ordered by explained variance, so the first component is the widest.
assert embeddings.iloc[:, 0].var() >= embeddings.iloc[:, 2].var()
```

### Autoencoder (`unsupervised`)

A neural network trained to reconstruct each row through a narrow bottleneck; the
bottleneck is the embedding. `hidden_units` sets the layers before it — a single
integer for one layer, a list for several.

```python
import row2vec

df = row2vec.generate_synthetic_data(200)
embeddings = row2vec.learn_embedding(
    df,
    mode="unsupervised",
    embedding_dim=4,
    hidden_units=[32, 16],  # two hidden layers
    max_epochs=3,  # kept small for this example; use far more in practice
    verbose=False,
)

assert embeddings.shape == (200, 4)
```

`max_epochs` bounds training; with `early_stopping=True` (the default) it stops
sooner once the reconstruction loss plateaus.

### t-SNE and UMAP

Both exist to be looked at. `perplexity` (t-SNE) and `n_neighbors` (UMAP) control
how large a neighbourhood each point is fitted against — smaller values favour
tight local structure, larger ones a smoother global picture.

```python
import row2vec

df = row2vec.generate_synthetic_data(150)

tsne = row2vec.learn_embedding(df, mode="tsne", embedding_dim=2, perplexity=10)
umap = row2vec.learn_embedding(df, mode="umap", embedding_dim=2, n_neighbors=10, min_dist=0.1)

assert tsne.shape == umap.shape == (150, 2)
```

`perplexity` must be small relative to the number of rows; Row2Vec says so
directly rather than letting scikit-learn fail obscurely.

```python
import pytest

import row2vec

df = row2vec.generate_synthetic_data(30)

with pytest.raises(ValueError, match="should be less than"):
    row2vec.learn_embedding(df, mode="tsne", embedding_dim=2, perplexity=100)
```

### Target-based embeddings

Instead of one vector per row, get one vector per distinct value of a column —
learned from the rows in which that value appears. Useful for turning a
high-cardinality categorical into a small dense feature.

```python
import row2vec

df = row2vec.generate_synthetic_data(200)
country_vectors = row2vec.learn_embedding(
    df,
    mode="target",
    reference_column="Country",
    embedding_dim=2,
    max_epochs=3,
    verbose=False,
)

# One row out per distinct country in, not one per input row.
assert len(country_vectors) == df["Country"].nunique()
```

### Contrastive embeddings

Supervision by example: tell the model which rows should end up close together
and which should not.

```python
import row2vec

df = row2vec.generate_synthetic_data(120)

embeddings = row2vec.learn_embedding(
    df,
    mode="contrastive",
    embedding_dim=3,
    similar_pairs=[(0, 1), (2, 3)],
    dissimilar_pairs=[(0, 50), (1, 60)],
    contrastive_loss="contrastive",
    max_epochs=3,
    verbose=False,
)

assert embeddings.shape == (120, 3)
```

If you don't have pairs to hand, `auto_pairs` derives them: `"categorical"` (rows
sharing a category value are alike), `"cluster"`, `"neighbors"`, or `"random"`.

```python
import row2vec

df = row2vec.generate_synthetic_data(120)

embeddings = row2vec.learn_embedding(
    df,
    mode="contrastive",
    embedding_dim=2,
    auto_pairs="categorical",
    reference_column="Country",
    max_epochs=3,
    verbose=False,
)

assert embeddings.shape == (120, 2)
```

---

## Preparing data

### Missing values

`AdaptiveImputer` analyses the missingness of each column and picks a strategy to
match. `learn_embedding` runs it for you, but you can use it on its own.

```python
import numpy as np
import pandas as pd

from row2vec import AdaptiveImputer, ImputationConfig, MissingPatternAnalyzer

df = pd.DataFrame(
    {
        "age": [25.0, np.nan, 41.0, 33.0, np.nan, 29.0],
        "city": ["rome", "oslo", None, "rome", "oslo", "rome"],
    }
)

analysis = MissingPatternAnalyzer(ImputationConfig()).analyze(df)
assert analysis["total_missing"] == 3

imputed = AdaptiveImputer(ImputationConfig()).fit_transform(df)
assert imputed.isna().sum().sum() == 0
assert len(imputed) == len(df)
```

Choose the trade-off explicitly when you care: `numeric_strategy="mean"` with
`prefer_speed=True` for quick iteration, `numeric_strategy="knn"` when accuracy
matters more than time.

```python
import numpy as np
import pandas as pd

from row2vec import AdaptiveImputer, ImputationConfig

df = pd.DataFrame({"x": [1.0, 2.0, np.nan, 4.0, 5.0, np.nan, 7.0, 8.0]})

fast = AdaptiveImputer(ImputationConfig(numeric_strategy="mean", prefer_speed=True)).fit_transform(
    df
)
accurate = AdaptiveImputer(ImputationConfig(numeric_strategy="knn", knn_neighbors=3)).fit_transform(
    df
)

assert fast.isna().sum().sum() == 0
assert accurate.isna().sum().sum() == 0
```

Set `preserve_missing_patterns=True` to keep the fact that a value *was* missing
as its own feature — often predictive in itself.

### Categorical columns

The encoder picks a strategy from the column's cardinality: one-hot for a handful
of values, ordinal or target encoding as the count grows, learned entity
embeddings for the largest. You can inspect that decision:

```python
import pandas as pd

from row2vec import CategoricalAnalyzer, CategoricalEncodingConfig

df = pd.DataFrame({"colour": ["red", "green", "blue", "red", "green", "blue"]})

analysis = CategoricalAnalyzer(CategoricalEncodingConfig()).analyze_column(df["colour"])

assert analysis["cardinality"] == 3
assert analysis["recommended_strategy"]  # a strategy name, e.g. "onehot"
```

### Scaling the output

`scale_method` rescales the embedding after it is computed — useful when a
downstream model expects a bounded range.

```python
import row2vec

df = row2vec.generate_synthetic_data(100)

scaled = row2vec.learn_embedding(
    df,
    mode="pca",
    embedding_dim=2,
    scale_method="minmax",
    scale_range=(0.0, 1.0),
)

values = scaled.to_numpy()
assert values.min() >= -1e-6
assert values.max() <= 1.0 + 1e-6
```

The options are `"minmax"`, `"standard"`, `"l2"`, `"tanh"`, and `"none"`.

---

## Tuning

### Choosing the dimension automatically

`auto_select_dimension` evaluates candidate widths and recommends one, so you do
not have to guess.

```python
import row2vec

df = row2vec.generate_synthetic_data(150)
recommended_dim, details = row2vec.auto_select_dimension(
    df, methods=["pca_variance"], max_dimension=5
)

assert 1 <= recommended_dim <= 5
assert "method_results" in details
```

### Searching the architecture

For the neural modes, `search_architecture` explores layer counts, widths,
dropout rates, and activations, and returns the best configuration it found.

```python
import row2vec
from row2vec import ArchitectureSearchConfig, EmbeddingConfig, NeuralConfig

df = row2vec.generate_synthetic_data(150)

base_config = EmbeddingConfig(
    mode="unsupervised", embedding_dim=3, neural=NeuralConfig(max_epochs=2)
)
search_config = ArchitectureSearchConfig(
    method="random", max_trials=2, verbose=False, layer_range=(1, 2)
)

best_architecture, _result = row2vec.search_architecture(
    df=df, base_config=base_config, search_config=search_config
)

assert "n_layers" in best_architecture
assert "hidden_units" in best_architecture
```

Searching costs one training run per trial — budget `max_trials` accordingly.

---

## Saving and reusing a model

An embedding is only reproducible if the *preprocessing* is reproducible too. A
saved Row2Vec model carries its encoders, imputers, and scalers with it, so
embedding new rows later stays consistent with training.

```python
import tempfile
from pathlib import Path

import row2vec

df = row2vec.generate_synthetic_data(100)
base = Path(tempfile.mkdtemp()) / "model"

embeddings, script_path, _binary_path = row2vec.train_and_save_model(
    df, base_path=str(base), mode="pca", embedding_dim=2
)

model = row2vec.load_model(script_path)
new_rows = row2vec.generate_synthetic_data(20, seed=99)

assert embeddings.shape == (100, 2)
assert model.predict(new_rows).shape == (20, 2)
```

Saving produces two files: a readable Python loader script and a binary blob
holding the fitted objects. Loading executes that script, so **only load models
you trust** — see
[SECURITY.md](https://github.com/tresoldi/row2vec/blob/main/SECURITY.md).

---

## Integrations

### scikit-learn pipelines

`Row2VecTransformer` is a standard transformer: put it in a `Pipeline` and it
behaves like any other step.

```python
from sklearn.pipeline import Pipeline

import row2vec
from row2vec import EmbeddingConfig, Row2VecTransformer

df = row2vec.generate_synthetic_data(100)

pipeline = Pipeline(
    [("embed", Row2VecTransformer(config=EmbeddingConfig(mode="pca", embedding_dim=2)))]
)
transformed = pipeline.fit_transform(df)

assert transformed.shape == (100, 2)
```

### The pandas accessor

Importing `row2vec` registers a `.row2vec` accessor on `DataFrame`, which is
convenient in a notebook.

```python
import row2vec  # importing registers the accessor

df = row2vec.generate_synthetic_data(80)
embeddings = df.row2vec.pca(dim=2)

assert embeddings.shape == (80, 2)
```

### The command line

For batch work there is no need to write Python at all:

```bash
# Embed a file in one step
row2vec annotate --input data.csv --output embeddings.csv --mode pca --dim 5

# Train a reusable model, then apply it to new data
row2vec train --input data.csv --output model.py --mode unsupervised --dim 10
row2vec predict --input new.csv --model model.py --output predictions.csv
```

---

## Practical notes

### Reproducibility

Every mode takes a `seed`. The same seed and the same input give the same
embedding.

```python
import row2vec

df = row2vec.generate_synthetic_data(80)

first = row2vec.learn_embedding(df, mode="pca", embedding_dim=2, seed=1305)
second = row2vec.learn_embedding(df, mode="pca", embedding_dim=2, seed=1305)

assert first.equals(second)
```

Neural modes are seeded the same way, though exact floating-point results can
still differ across platforms and library versions.

### Configuration objects

For anything beyond a few arguments, `EmbeddingConfig` groups the settings so
they can be reused, serialised, and version-controlled.

```python
import row2vec
from row2vec import EmbeddingConfig, NeuralConfig

config = EmbeddingConfig(
    mode="unsupervised",
    embedding_dim=4,
    neural=NeuralConfig(hidden_units=[32, 16], max_epochs=2, dropout_rate=0.1),
)

df = row2vec.generate_synthetic_data(100)
embeddings = row2vec.learn_embedding_v2(df, config)

assert embeddings.shape == (100, 4)
```

### Logging

Training emits structured logs. Turn them off for quiet runs, or point them at a
file with `log_file=`.

```python
import row2vec

df = row2vec.generate_synthetic_data(50)
embeddings = row2vec.learn_embedding(df, mode="pca", embedding_dim=2, enable_logging=False)

assert embeddings.shape == (50, 2)
```

### What is not supported

Free text and datetime columns are not embedded directly. Preprocess them
yourself — sentence embeddings for text, explicit features (month, weekday,
elapsed days) for timestamps — and pass the result as ordinary columns.

---

## Next steps

- **[API Reference](reference.md)** — every public class and function, generated
  from the source.
- **[Home](index.md)** — the one-minute overview and the method table.
