# Row2Vec Examples

This directory contains example scripts demonstrating various features of the Row2Vec library.

## Quick Start Examples

### Basic Usage
- **`basic_usage.py`** - Introduction to Row2Vec with synthetic data
  - Shows target-based and unsupervised embeddings
  - Demonstrates data preprocessing and embedding generation

### Configuration-Based API
- **`config_based_usage.py`** - Modern configuration-based API usage
  - Shows how to use `EmbeddingConfig` objects
  - Demonstrates different embedding modes
  - YAML configuration loading

### Configuration Files
- **`config_default.yaml`** - Default configuration template
- **`config_target.yaml`** - Target-based learning configuration
- **`config_contrastive.yaml`** - Contrastive learning configuration

## Advanced Features

### Contrastive Learning
- **`contrastive_demo.py`** - Contrastive learning setup and examples
  - Creates sample data and pair files
  - Shows CLI commands for contrastive mode

### Automatic Optimization
- **`auto_dimension_examples.py`** - Automatic dimension selection
  - Multiple methods for optimal dimension selection
  - Performance-based evaluation

- **`architecture_search_examples.py`** - Neural architecture search
  - Automatic network architecture optimization
  - Grid and random search methods

### Data Preprocessing
- **`imputation_demo.py`** - Missing value imputation
  - Intelligent missing value handling
  - Pattern analysis and adaptive strategies

- **`categorical_encoding_demo.py`** - Advanced categorical encoding
  - Entity embeddings
  - Target encoding strategies

### Classical Methods
- **`classical_methods_demo.py`** - Comparison of classical methods
  - PCA, t-SNE, UMAP implementations
  - Performance comparisons

## Integration Examples

### CLI and Automation
- **`cli_demo.py`** - Command-line interface usage
  - Example CLI commands
  - Batch processing

### Model Persistence
- **`serialization_demo.py`** - Model saving and loading
  - Train and save models
  - Load and apply to new data
  - Metadata management

### Framework Integration
- **`integration_examples.py`** - Integration with other frameworks
  - scikit-learn pipeline integration
  - pandas accessor usage

### Development Tools
- **`logging_demo.py`** - Logging configuration
- **`type_checking_demo.py`** - Type hints and validation

### Real-World Applications
- **`real_world_usage.py`** - Practical use cases
  - Customer segmentation
  - Feature engineering
  - Downstream task integration

## Running the Examples

Most examples can be run directly:

```bash
python examples/basic_usage.py
```

Some examples may require additional dependencies or take longer to run due to model training.

## Notes

- All examples use synthetic data for demonstration
- Some advanced features may require optional dependencies
- Examples are designed to be educational and may not represent production best practices
- For production use, refer to the main documentation for optimization and scaling guidelines
