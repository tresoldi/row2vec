---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Quickstart Guide

Get started with Row2Vec in 5 minutes! This guide shows the essential features through executable examples.

## Basic Usage

The core of Row2Vec is the `learn_embedding()` function:

```{code-cell} python
:tags: [remove-stderr, remove-warnings]

# Import complete suppression first
exec(open('suppress_minimal.py').read())

from row2vec import learn_embedding, generate_synthetic_data
import pandas as pd

# Generate sample data
df = generate_synthetic_data(num_records=200, seed=42)
print(f"Dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
```

## Unsupervised Embeddings

Create compressed representations of each row:

```{code-cell} python
:tags: [remove-stderr, remove-warnings]

# Learn 5-dimensional embeddings for each row
embeddings = learn_embedding(
    df,
    mode="unsupervised",
    embedding_dim=5,
    max_epochs=20,
    verbose=False
)

print(f"Embeddings shape: {embeddings.shape}")
print("\nFirst 3 embeddings:")
print(embeddings.head(3))
```

```{code-cell} python
:tags: [remove-stderr, remove-warnings]

# Verify: each row gets an embedding
print(f"Original data: {len(df)} rows")
print(f"Embeddings: {len(embeddings)} rows")
print(f"Dimensions per embedding: {embeddings.shape[1]}")
```

## Target-Based Embeddings

Learn embeddings for categorical column values:

```{code-cell} python
:tags: [remove-stderr, remove-warnings]

# Learn embeddings for each country
country_embeddings = learn_embedding(
    df,
    mode="target",
    reference_column="Country",
    embedding_dim=3,
    max_epochs=20,
    verbose=False
)

print("Country embeddings:")
print(country_embeddings)
```

## Classical Methods

Row2Vec also provides classical dimensionality reduction:

### PCA (Fast Linear Reduction)

```{code-cell} python
:tags: [remove-stderr, remove-warnings]

pca_embeddings = learn_embedding(
    df,
    mode="pca",
    embedding_dim=2,
    verbose=False
)

print("PCA embeddings (first 5):")
print(pca_embeddings.head())
```

### t-SNE (Visualization)

```{code-cell} python
:tags: [remove-stderr, remove-warnings]

tsne_embeddings = learn_embedding(
    df,
    mode="tsne",
    embedding_dim=2,
    perplexity=30,
    verbose=False
)

print("t-SNE embeddings (first 5):")
print(tsne_embeddings.head())
```

### UMAP (Balanced Approach)

```{code-cell} python
:tags: [remove-stderr, remove-warnings]

try:
    umap_embeddings = learn_embedding(
        df,
        mode="umap",
        embedding_dim=2,
        n_neighbors=15,
        verbose=False
    )
    print("UMAP embeddings (first 5):")
    print(umap_embeddings.head())
except ImportError:
    print("UMAP not installed. Install with: pip install umap-learn")
```

## Scaling Embeddings

Apply post-processing scaling to embeddings:

```{code-cell} python
:tags: [remove-stderr, remove-warnings]

# Scale embeddings to [-1, 1] range
scaled_embeddings = learn_embedding(
    df,
    mode="unsupervised",
    embedding_dim=3,
    max_epochs=10,
    scale_method="minmax",
    scale_range=(-1.0, 1.0),
    verbose=False
)

print("Scaled embeddings statistics:")
print(f"Min value: {scaled_embeddings.min().min():.3f}")
print(f"Max value: {scaled_embeddings.max().max():.3f}")
print("\nFirst 3 scaled embeddings:")
print(scaled_embeddings.head(3))
```

## Handling Missing Values

Row2Vec automatically handles missing values:

```{code-cell} python
:tags: [remove-stderr, remove-warnings]

import numpy as np

# Create data with missing values
df_missing = df.copy()
df_missing.loc[0:5, 'Sales'] = np.nan
df_missing.loc[10:15, 'Product'] = np.nan

print(f"Missing values introduced: {df_missing.isnull().sum().sum()}")

# Row2Vec handles this automatically
embeddings_missing = learn_embedding(
    df_missing,
    mode="unsupervised",
    embedding_dim=3,
    max_epochs=10,
    verbose=False
)

print(f"\nEmbeddings generated: {embeddings_missing.shape}")
print("No errors - missing values handled automatically!")
```

## Save and Load Models

Train once, use many times:

```{code-cell} python
:tags: [remove-stderr, remove-warnings]

from row2vec import train_and_save_model, load_model
import tempfile
import os

# Create temporary directory for demo
with tempfile.TemporaryDirectory() as tmpdir:
    model_path = os.path.join(tmpdir, "my_model")

    # Train and save model
    embeddings, script_path, binary_path = train_and_save_model(
        df,
        base_path=model_path,
        embedding_dim=4,
        mode="unsupervised",
        max_epochs=10,
        verbose=False
    )

    print(f"Model saved to: {script_path}")

    # Load and use model
    model = load_model(script_path)

    # Generate embeddings for new data
    new_data = generate_synthetic_data(num_records=50, seed=999)
    new_embeddings = model.predict(new_data)

    print(f"\nNew embeddings shape: {new_embeddings.shape}")
    print("Model successfully loaded and used!")
```

## Command-Line Interface

Row2Vec also provides a CLI for batch processing:

```bash
# Quick embeddings
row2vec data.csv --output embeddings.csv

# With configuration
row2vec data.csv \
  --dimensions 10 \
  --mode unsupervised \
  --output embeddings.csv

# Train and save model
row2vec-train data.csv \
  --dimensions 5 \
  --output model.pkl

# Apply saved model
row2vec-embed new_data.csv \
  --model model.py \
  --output new_embeddings.csv
```

## Method Comparison

Let's compare all methods on the same data:

```{code-cell} python
:tags: [remove-stderr, remove-warnings]

import time

methods = {
    "Neural": {"mode": "unsupervised", "max_epochs": 10},
    "PCA": {"mode": "pca"},
    "t-SNE": {"mode": "tsne", "perplexity": 30},
}

results = {}
for name, params in methods.items():
    start = time.time()
    emb = learn_embedding(df, embedding_dim=2, verbose=False, **params)
    elapsed = time.time() - start
    results[name] = {
        "time": elapsed,
        "shape": emb.shape,
        "mean": emb.mean().mean(),
        "std": emb.std().mean()
    }

print("Method Comparison:")
print("-" * 50)
for method, stats in results.items():
    print(f"{method:10} | Time: {stats['time']:.2f}s | Mean: {stats['mean']:6.3f} | Std: {stats['std']:5.3f}")
```

## Method Selection Guide

Each embedding method has different strengths. Here's when to use each:

| Method | Speed | Deterministic | Best For | Embedding Range |
|--------|-------|---------------|----------|-----------------|
| **Neural** | Medium | Yes (with seed) | Complex patterns, feature engineering | Typically [-1, 1] |
| **PCA** | Fast | Yes | Quick dimensionality reduction, linear relationships | Variable scale |
| **t-SNE** | Slow | No* | 2D/3D visualization, cluster discovery | Large range, clustered |
| **UMAP** | Fast | Yes (with seed) | General purpose, balanced local/global structure | Moderate range |

*t-SNE can be made more deterministic with proper seeding, but still has some inherent randomness.

### When to Use Each Method

**Choose Neural Networks (`mode="unsupervised"`) when:**
- You need embeddings for downstream machine learning models
- Your data has complex, non-linear relationships
- You want features that can capture intricate patterns
- You have sufficient training time and computational resources

**Choose PCA (`mode="pca"`) when:**
- You need fast, deterministic results
- Your data relationships are primarily linear
- You want interpretable principal components
- You're preprocessing data for other algorithms

**Choose t-SNE (`mode="tsne"`) when:**
- You want to visualize data in 2D or 3D
- Discovering clusters is your primary goal
- Local neighborhood preservation is most important
- You don't mind longer computation times

**Choose UMAP (`mode="umap"`) when:**
- You want general-purpose dimensionality reduction
- You need both local and global structure preserved
- You want faster performance than t-SNE
- You're working with higher-dimensional outputs (>3D)

## Next Steps

Now you know the basics! For more detailed examples:

- 📊 [Titanic Example](titanic_example.md) - Complete walkthrough with real data
- 🏠 [Housing Example](housing_example.md) - Regression features
- 🎯 [Advanced Features](advanced_features.md) - Neural architecture search, imputation
- 💻 [CLI Guide](cli_guide.md) - Command-line workflows
- 📚 [API Reference](api_reference.md) - Complete documentation
