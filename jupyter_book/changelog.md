# Changelog

## Version 0.1.0 (Current Development)

### 🚀 New Features

- **Core Embedding Engine**: Complete implementation of `learn_embedding()` function
- **Five Embedding Modes**: Neural (unsupervised), target-based, PCA, t-SNE, and UMAP
- **Intelligent Preprocessing**: Automatic missing value imputation and categorical encoding
- **Neural Architecture Search**: Automated optimal architecture discovery
- **Model Serialization**: Save/load trained models with metadata
- **Comprehensive CLI**: Three-command system for different workflows
- **sklearn Integration**: Transformer and classifier interfaces
- **pandas Integration**: DataFrame accessor methods

### 🧠 Advanced Features

- **Multi-layer Networks**: Support for deep architectures with 1-4+ hidden layers
- **Adaptive Imputation**: Multiple strategies for missing value handling
- **Categorical Encoding**: OneHot, target encoding, entity embeddings
- **Auto Dimension Selection**: Find optimal embedding dimensions
- **Contrastive Learning**: Advanced embedding technique
- **Configuration Objects**: Type-safe configuration management

### 🔧 Technical Improvements

- **Modern Build System**: pyproject.toml with hatchling backend
- **Type Safety**: Complete MyPy annotations
- **Comprehensive Testing**: 111 tests with 92% coverage
- **Code Quality**: ruff linting and formatting
- **Documentation**: Jupyter Book-based executable documentation
- **Performance Monitoring**: Built-in timing and memory tracking

### 📊 Data Handling

- **Multiple Formats**: CSV, Parquet, Excel, JSON, TSV support
- **Large Dataset Support**: Sampling and batch processing
- **Schema Validation**: Input data validation for models
- **Scaling Options**: MinMax, standard, robust, L2 scaling
- **Error Handling**: Comprehensive validation and error messages

### 🏗️ Architecture

- **Modular Design**: Clear separation of concerns
- **Plugin System**: Optional dependencies (UMAP, CLI tools)
- **Pipeline Builder**: Flexible preprocessing pipeline construction
- **Extensible**: Easy to add new embedding methods

## Planned Releases

### Version 0.2.0 (Target: Q2 2025)

**Planned Features:**
- Multi-task learning mode
- Temporal embedding support
- Visualization module (`row2vec.viz`)
- Interactive documentation with Binder
- Performance benchmarks

### Version 0.3.0 (Target: Q3 2025)

**Planned Features:**
- Variational autoencoders (VAE)
- Transformer-based architectures
- Graph neural network support
- Federated learning capabilities
- Advanced visualization tools

### Version 1.0.0 (Target: Q4 2025)

**Production Release:**
- Full API stability guarantee
- Comprehensive performance optimization
- Enterprise features
- Professional support documentation

## Development History

### Pre-release Development

- **Core Implementation**: Main embedding functionality with neural networks
- **Advanced Features**: Neural Architecture Search, imputation, serialization
- **Documentation**: Comprehensive Jupyter Book documentation
- **Production Readiness**: Testing, CI/CD, packaging

## Breaking Changes

Future breaking changes will be minimized and clearly documented with advance notice.

## Acknowledgments

### Contributors

- **Tiago Tresoldi** - Creator and primary maintainer
- **Community Contributors** - Bug reports, feature requests, and feedback

### Dependencies

Row2Vec builds on excellent open-source libraries:
- **TensorFlow/Keras** - Neural network backend
- **scikit-learn** - Preprocessing and classical ML methods
- **pandas** - Data manipulation
- **NumPy** - Numerical computing

### Inspiration

- **Word2Vec** - Original embedding concept
- **Node2Vec** - Graph-based embeddings
- **FastText** - Subword embeddings
- **t-SNE/UMAP** - Dimensionality reduction techniques

## License

Row2Vec is released under the MIT License. See LICENSE file for details.

## Support

- 📖 [Documentation](intro.md)
- 🐛 [Issue Tracker](https://github.com/evotext/row2vec/issues)
- 💬 [Discussions](https://github.com/evotext/row2vec/discussions)
- 📧 Email: tiago@tresoldi.org

---

*This changelog follows [Keep a Changelog](https://keepachangelog.com/) format.*
