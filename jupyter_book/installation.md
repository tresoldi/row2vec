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

# Installation

## Requirements

Row2Vec requires Python 3.10 or higher and depends on:
- pandas >= 1.5.3
- scikit-learn >= 1.0.0
- tensorflow >= 2.8.0
- numpy >= 1.21.0
- umap-learn >= 0.5.0
- click >= 8.0.0
- rich >= 12.0.0
- pyyaml >= 6.0.0

## Install from PyPI

The simplest way to install Row2Vec:

```bash
pip install row2vec
```

## Install from Source

For the latest development version:

```bash
git clone https://github.com/evotext/row2vec.git
cd row2vec
pip install -e .
```

## Install with Development Dependencies

If you want to contribute or run tests:

```bash
git clone https://github.com/evotext/row2vec.git
cd row2vec
pip install -e ".[dev]"
```

This includes:
- pytest for testing
- ruff for linting and formatting
- mypy for type checking
- jupyter for notebook development

## Optional Dependencies

Row2Vec now includes all major dependencies by default. Optional extras:

### For development work:
```bash
pip install "row2vec[dev]"
```

### For documentation building:
```bash
pip install "row2vec[docs]"
```

### For data format support:
```bash
pip install pyarrow  # For Parquet files
pip install openpyxl  # For Excel files
```

## Verify Installation

```{code-cell} python
:tags: [remove-stderr, remove-warnings]

# Import complete suppression first
exec(open('suppress_minimal.py').read())

import row2vec
print(f"Row2Vec version: {row2vec.__version__}")

# Check available modes
from row2vec import learn_embedding
print("\nAvailable embedding modes:")
print("- unsupervised (neural network autoencoder)")
print("- target (supervised categorical embeddings)")
print("- pca (Principal Component Analysis)")
print("- tsne (t-Distributed Stochastic Neighbor Embedding)")
print("- umap (Uniform Manifold Approximation and Projection)")
```

## Platform-Specific Notes

### macOS with Apple Silicon

For optimal performance on M1/M2 Macs:

```bash
# Install TensorFlow for Apple Silicon
pip install tensorflow-macos
pip install tensorflow-metal
```

### Linux

Ensure you have Python development headers:

```bash
# Ubuntu/Debian
sudo apt-get install python3-dev

# Fedora/RHEL
sudo dnf install python3-devel
```

### Windows

We recommend using Anaconda or Miniconda:

```bash
conda create -n row2vec python=3.10
conda activate row2vec
pip install row2vec
```

## Troubleshooting

### Import Error: No module named 'tensorflow'

TensorFlow might not be installed properly:

```bash
pip uninstall tensorflow
pip install --upgrade tensorflow
```

### Memory Issues

For large datasets, ensure you have sufficient RAM or use sampling:

```python
from row2vec import learn_embedding

# Sample large datasets
embeddings = learn_embedding(
    large_df.sample(n=10000),  # Sample 10k rows
    mode="unsupervised"
)
```

### GPU Support

To use GPU acceleration:

```bash
# Check GPU availability
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# Install CUDA-enabled TensorFlow
pip install tensorflow[and-cuda]
```

## Next Steps

Installation complete! Now proceed to the [Quickstart](quickstart.md) guide to learn the basics.
