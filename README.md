# Row2Vec

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

**Row2Vec** is a Python library for easily generating low-dimensional vector embeddings from any tabular dataset. It uses deep learning and classical methods to create powerful, dense representations of your data, suitable for visualization, feature engineering, and gaining deeper insights into your data's structure.

## Features

### 🎯 Multiple Embedding Methods
- **Neural (Autoencoder)**: Deep learning approach for complex, non-linear patterns
- **Target-based**: Learn embeddings for categorical columns and their relationships
- **PCA**: Fast, linear dimensionality reduction with interpretable components
- **t-SNE**: Excellent for 2D/3D visualization and cluster discovery
- **UMAP**: Balanced preservation of local and global structure

### 🧠 Intelligent Preprocessing
- **Adaptive Missing Value Imputation**: Automatically analyzes patterns and applies optimal strategies
- **Pattern-Aware Analysis**: Detects problematic missing patterns with configurable strategies
- **Automated Feature Engineering**: Handles scaling, encoding, and preprocessing seamlessly

### 🚀 Advanced Features
- **Neural Architecture Search (NAS)**: Automatically discovers optimal network architectures
- **Multi-layer Networks**: Support for deep architectures with dropout and regularization
- **Model Serialization**: Save and load models with full preprocessing pipelines
- **Command-Line Interface**: Complete CLI for batch processing and automation

### 🔧 Production Ready
- **Comprehensive Testing**: 163+ test functions across 17 test files
- **Type Safety**: Complete MyPy annotations
- **Modern Build System**: Uses `pyproject.toml` with hatchling backend
- **Documentation**: Interactive Jupyter Book with executable examples

## Installation

```bash
pip install row2vec
```

## Quick Start

```python
import pandas as pd
from row2vec import learn_embedding, generate_synthetic_data

# Load your data
df = generate_synthetic_data(num_records=1000)

# Generate neural embeddings for each row
embeddings = learn_embedding(
    df,
    mode="unsupervised",
    embedding_dim=5
)
print(f"Embeddings shape: {embeddings.shape}")
print(embeddings.head())

# Learn categorical embeddings
country_embeddings = learn_embedding(
    df,
    mode="target",
    reference_column="Country",
    embedding_dim=3
)
print(f"Country embeddings: {country_embeddings}")

# Compare with classical methods
pca_embeddings = learn_embedding(df, mode="pca", embedding_dim=5)
tsne_embeddings = learn_embedding(df, mode="tsne", embedding_dim=2)
```

## Command Line Interface

```bash
# Quick embeddings
row2vec annotate --input data.csv --output embeddings.csv --mode unsupervised --dim 5

# Train and save model
row2vec train --input data.csv --output model.py --mode unsupervised --dim 10 --epochs 50

# Use saved model
row2vec predict --input new_data.csv --model model.py --output predictions.csv

# Target-based embeddings
row2vec annotate --input data.csv --output categories.csv --mode target --target-col Category --dim 3
```

## Advanced Usage

### Neural Architecture Search

```python
from row2vec import ArchitectureSearchConfig, search_architecture, EmbeddingConfig, NeuralConfig

# Configure architecture search
config = ArchitectureSearchConfig(
    method='random',
    max_layers=3,
    width_options=[64, 128, 256],
    max_trials=20
)

base_config = EmbeddingConfig(
    mode="unsupervised",
    embedding_dim=8,
    neural=NeuralConfig(max_epochs=50)
)

# Find optimal architecture
best_arch, results = search_architecture(df, base_config, config)
print(f"Best architecture: {best_arch}")

# Train with optimal settings
optimal_embeddings = learn_embedding(
    df,
    mode="unsupervised",
    embedding_dim=8,
    hidden_units=best_arch.get('hidden_units', [128]),
    max_epochs=100
)
```

### Missing Value Imputation

```python
from row2vec import ImputationConfig, AdaptiveImputer, MissingPatternAnalyzer

# Analyze missing patterns
analyzer = MissingPatternAnalyzer(ImputationConfig())
analysis = analyzer.analyze(df)
print(f"Missing patterns: {analysis['recommendations']}")

# Apply adaptive imputation
imputer = AdaptiveImputer(ImputationConfig(
    numeric_strategy='knn',
    categorical_strategy='mode',
    knn_neighbors=10
))
df_clean = imputer.fit_transform(df)
```

## Documentation

### Online Documentation
- **[Installation Guide](https://evotext.github.io/row2vec/installation.html)**: Detailed setup instructions
- **[Quick Start Tutorial](https://evotext.github.io/row2vec/quickstart.html)**: Get up and running in 5 minutes
- **[API Reference](https://evotext.github.io/row2vec/api_reference.html)**: Complete function documentation
- **[Example Gallery](https://evotext.github.io/row2vec/)**: Real-world use cases and tutorials
- **[Advanced Features](https://evotext.github.io/row2vec/advanced_features.html)**: Neural architecture search, imputation strategies

### Local Documentation
- **[User Guide](docs/USER_GUIDE.md)**: Comprehensive guide with mathematical background, detailed examples, and best practices
- **[LLM Documentation](docs/LLM_DOCUMENTATION.md)**: Practical guide for LLM coding agents integrating Row2Vec
- **[API Reference](docs/API_REFERENCE.md)**: Complete function and class reference
- **[Tutorials](docs/)**: Executable Python tutorials (Nhandu format) - run `make docs` to build HTML

## Why Row2Vec?

| Method | Row2Vec Advantage | Alternative |
|--------|-------------------|-------------|
| **Manual Neural Networks** | Automated preprocessing, simple API | 200+ lines of boilerplate |
| **sklearn PCA** | Integrated preprocessing, multiple methods | Limited to linear reduction |
| **sklearn t-SNE/UMAP** | Unified interface, consistent preprocessing | Manual pipeline setup |
| **Custom Embeddings** | Production-ready with serialization | Significant development time |

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## Citation

If you use Row2Vec in your research, please cite:

```bibtex
@software{tresoldi_row2vec,
  author = {Tresoldi, Tiago},
  title = {Row2Vec: Neural and Classical Embeddings for Tabular Data},
  url = {https://github.com/evotext/row2vec},
  version = {1.0.0}
}
```

## Acknowledgments

This library was originally developed as part of the **"Cultural Evolution of Texts"** project, led by Michael Dunn at the Department of Linguistics and Philology, Uppsala University. The project investigates the application of evolutionary models to textual data and cultural transmission patterns.

## Authors

**Tiago Tresoldi**
*Affiliate Researcher, Department of Linguistics and Philology*
*Uppsala University*
*GitHub: [@tresoldi](https://github.com/tresoldi)*

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
