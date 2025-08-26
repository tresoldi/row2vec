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

# Row2Vec: Learn Embeddings from Tabular Data

```{admonition} Welcome
Row2Vec is a Python library for easily generating low-dimensional vector embeddings from any tabular dataset. It uses deep learning and classical ML methods to create powerful, dense representations of your data.
```

## What is Row2Vec?

Row2Vec transforms tabular data into meaningful vector representations (embeddings) that capture the essential characteristics of your data. Instead of feeding raw data directly into models, you can create compressed, information-rich representations that models can easily process.

## Key Features

- **🎯 Five Powerful Modes**: Neural networks, PCA, t-SNE, UMAP, and target-based embeddings
- **🧠 Intelligent Preprocessing**: Automatic missing value imputation and feature encoding
- **🚀 Simple API**: One function - `learn_embedding()` - handles everything
- **💾 Model Persistence**: Save and load trained models for production use
- **🔧 Production Ready**: 92% test coverage, type safety, modern build system

## Quick Example

```{code-cell} python
:tags: [remove-stderr, remove-warnings]

# Import complete suppression first
exec(open('suppress_minimal.py').read())

import pandas as pd
from row2vec import learn_embedding, generate_synthetic_data

# Generate sample data
df = generate_synthetic_data(num_records=100, seed=42)
print(f"Data shape: {df.shape}")
print(df.head(3))
```

```{code-cell} python
:tags: [remove-stderr, remove-warnings]

# Generate unsupervised embeddings
embeddings = learn_embedding(df, mode="unsupervised", embedding_dim=2, max_epochs=10, verbose=False)
print(f"\nEmbedding shape: {embeddings.shape}")
print("\nFirst 5 embeddings:")
print(embeddings.head())
```

## Why Use Row2Vec?

### Compared to Manual Implementation

| Aspect | Row2Vec | Manual Neural Network |
|--------|---------|----------------------|
| Lines of code | ~5 | ~200+ |
| Preprocessing | Automatic | Manual pipeline |
| Missing values | Handled | Manual imputation |
| Categorical encoding | Automatic | Manual encoding |
| Scaling | Built-in | Manual setup |

### Compared to Other Methods

| Method | Use Case | Row2Vec Advantage |
|--------|----------|-------------------|
| PCA | Linear reduction | Also offers non-linear (neural) options |
| t-SNE | Visualization | Unified interface with preprocessing |
| UMAP | General reduction | Consistent API across all methods |
| Manual NN | Custom embeddings | Automatic preprocessing, simpler API |

## Installation

```bash
pip install row2vec
```

## Documentation Overview

- **[Installation](installation.md)**: Setup and requirements
- **[Quickstart](quickstart.md)**: Get started in 5 minutes
- **[Titanic Example](titanic_example.md)**: Complete walkthrough with the Titanic dataset
- **[Adult Example](adult_example.md)**: High-cardinality categorical features
- **[Housing Example](housing_example.md)**: Real estate price prediction features
- **[Advanced Features](advanced_features.md)**: Neural architecture search, imputation strategies
- **[CLI Guide](cli_guide.md)**: Command-line interface documentation
- **[API Reference](api_reference.md)**: Complete API documentation

## Next Steps

Ready to get started? Head to the [Installation](installation.md) guide or jump straight to the [Quickstart](quickstart.md) tutorial.

```{admonition} Questions or Issues?
:class: tip
- 📖 Check the [API Reference](api_reference.md) for detailed documentation
- 🐛 Report issues on [GitHub](https://github.com/evotext/row2vec/issues)
- 💬 Join discussions in the [GitHub Discussions](https://github.com/evotext/row2vec/discussions)
```
