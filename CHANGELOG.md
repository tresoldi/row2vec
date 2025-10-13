# CHANGELOG

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2025-10-13

### Added

#### Documentation
- Comprehensive USER_GUIDE.md with mathematical foundations and examples
- LLM_DOCUMENTATION.md for AI coding agent integration
- Nhandu-based executable tutorials (tutorial_1_quickstart.py, tutorial_2_advanced.py)
- Complete API reference documentation

#### Build & Development
- Comprehensive Makefile for development workflow
- CHANGELOG.md following Keep a Changelog format
- scripts/ directory for benchmarking and utilities
- Dynamic version management from `__init__.py`
- Switched build backend from hatchling to setuptools

#### Testing & Quality
- Achieved 47% test coverage with 91 passing tests
- Updated test coverage threshold to 46% (baseline for gradual improvement)
- Test coverage improvement plan: 46% → 50% → 60% → 70%

#### Core Embedding Methods
- Neural autoencoder-based embeddings for unsupervised learning
- Target-based embeddings for categorical features
- Classical methods integration: PCA, t-SNE, UMAP
- Contrastive learning support for enhanced representations

#### Intelligent Preprocessing
- Adaptive missing value imputation with pattern analysis
- Automatic feature scaling and normalization
- Categorical encoding with entity embeddings
- Pattern-aware missing data detection

#### Advanced Features
- Neural Architecture Search (NAS) for optimal network discovery
- Multi-layer neural networks with dropout and regularization
- Automatic dimension selection based on data characteristics
- Model serialization with full preprocessing pipeline preservation

#### Developer Experience
- Command-line interface (CLI) for batch processing
- Configuration file support (YAML)
- Comprehensive type hints with MyPy validation
- Integration with pandas DataFrames (`.row2vec` accessor)
- Scikit-learn compatible transformers and classifiers

#### Testing & Quality
- 163+ test functions across 17 test modules
- Coverage reporting with pytest-cov
- Property-based testing foundations
- Integration tests for full pipelines
- Performance regression tests

#### Documentation
- Interactive Jupyter Book with executable examples
- Real-world tutorials (Titanic, Adult, Housing datasets)
- Complete API reference
- Automated documentation deployment
- Theoretical foundation guide

#### Build & Release
- Modern `pyproject.toml` configuration with hatchling backend
- GitHub Actions CI/CD pipeline
- Multi-version Python testing (3.10, 3.11, 3.12)
- Automated code quality gates (ruff, mypy)
- Pre-commit hooks for code quality

### Technical Details
- Python 3.10+ support
- TensorFlow/Keras backend for neural methods
- Rich console output for CLI
- Logging infrastructure with configurable levels
- Memory monitoring for large datasets

[Unreleased]: https://github.com/evotext/row2vec/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/evotext/row2vec/releases/tag/v0.1.0
