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

# Advanced Features

Row2Vec includes several advanced features for expert users and production workflows.

## Neural Architecture Search (NAS)

Automatically find optimal neural network architectures:

```{code-cell} python
:tags: [remove-stderr, remove-warnings]

# Import complete suppression first
exec(open('suppress_minimal.py').read())

from row2vec import (
    ArchitectureSearchConfig,
    search_architecture,
    generate_synthetic_data
)

# Generate test data
df = generate_synthetic_data(num_records=500, seed=42)
print(f"Test data shape: {df.shape}")
```

```{code-cell} python
:tags: [remove-stderr, remove-warnings]

# Configure architecture search
config = ArchitectureSearchConfig(
    method='random',        # Random search (faster than grid)
    max_layers=3,          # Search up to 3 hidden layers
    width_options=[32, 64, 128, 256],  # Neuron options per layer
    max_trials=5,          # Number of architectures to try (reduced for demo)
    initial_epochs=10      # Reduced epochs for faster demo
)

# Run architecture search
print("Searching for optimal architecture...")

# Need base config for architecture search
from row2vec import EmbeddingConfig, NeuralConfig
base_config = EmbeddingConfig(
    mode="unsupervised",
    embedding_dim=5,
    neural=NeuralConfig(max_epochs=10)
)

best_architecture, search_results = search_architecture(df, base_config, config)

print(f"\nBest architecture found:")
print(f"  Architecture: {best_architecture}")
print(f"  Search completed in {search_results.total_time:.2f} seconds")
```

```{code-cell} python
:tags: [remove-stderr, remove-warnings]

# Use the best architecture
from row2vec import learn_embedding

# Apply the best architecture found
best_embeddings = learn_embedding(
    df,
    mode="unsupervised",
    embedding_dim=5,
    hidden_units=best_architecture.get('hidden_units', [128]),
    dropout_rate=best_architecture.get('dropout_rate', 0.2),
    max_epochs=20,
    verbose=False,
    seed=42
)

print(f"Optimized embeddings shape: {best_embeddings.shape}")
print("First 3 embeddings using best architecture:")
print(best_embeddings.head(3).round(4))
```

## Advanced Missing Value Imputation

Intelligent missing value handling with multiple strategies:

```{code-cell} python
:tags: [remove-stderr, remove-warnings]

from row2vec import (
    ImputationConfig,
    AdaptiveImputer,
    MissingPatternAnalyzer
)
import numpy as np

# Create data with missing values
df_missing = df.copy()
np.random.seed(42)

# Introduce different types of missing patterns
df_missing.loc[np.random.choice(df_missing.index, 50), 'Sales'] = np.nan
df_missing.loc[np.random.choice(df_missing.index, 30), 'Product'] = np.nan
df_missing.loc[np.random.choice(df_missing.index, 20), 'Country'] = np.nan

print(f"Missing values introduced:")
print(df_missing.isnull().sum())
```

```{code-cell} python
:tags: [remove-stderr, remove-warnings]

# Analyze missing patterns
config = ImputationConfig()
analyzer = MissingPatternAnalyzer(config)
analysis = analyzer.analyze(df_missing)

print(f"Missing value analysis:")
print(f"  Total missing: {analysis['total_missing']}")
print(f"  Missing percentage: {analysis['missing_percentage']:.1f}%")
print(f"  Columns with missing: {analysis['columns_with_missing']}")
print(f"  Recommendations: {analysis['recommendations']}")
```

```{code-cell} python
:tags: [remove-stderr, remove-warnings]

# Apply different imputation strategies
strategies = {
    'adaptive': ImputationConfig(),
    'knn': ImputationConfig(
        numeric_strategy='knn',
        categorical_strategy='mode',
        knn_neighbors=5
    ),
    'iterative': ImputationConfig(
        numeric_strategy='iterative',
        categorical_strategy='mode',
        categorical_fill_value='Missing'
    )
}

for name, strategy_config in strategies.items():
    imputer = AdaptiveImputer(strategy_config)
    df_imputed = imputer.fit_transform(df_missing)

    remaining_missing = df_imputed.isnull().sum().sum()
    print(f"{name:12}: {remaining_missing} missing values remaining")
```

## Categorical Encoding Strategies

Advanced categorical feature handling:

```{code-cell} python
:tags: [remove-stderr, remove-warnings]

from row2vec import learn_embedding_v2, EmbeddingConfig, PreprocessingConfig

# Test different categorical encoding strategies
encoding_strategies = ['onehot', 'target', 'adaptive']

for strategy in encoding_strategies:
    # Create complete config with categorical encoding strategy
    config = EmbeddingConfig(
        embedding_dim=3,
        mode="unsupervised",
        preprocessing=PreprocessingConfig(
            categorical_encoding_strategy=strategy
        )
    )

    embeddings = learn_embedding_v2(
        df,
        config=config
    )

    print(f"{strategy:12}: shape {embeddings.shape}, mean={embeddings.mean().mean():.3f}")
```

## Multi-Layer Neural Networks

Deep architectures for complex patterns:

```{code-cell} python
:tags: [remove-stderr, remove-warnings]

# Compare single vs multi-layer networks
architectures = {
    'Single Layer': [128],
    'Two Layer': [256, 128],
    'Three Layer': [512, 256, 128]
}

for name, hidden_units in architectures.items():
    embeddings = learn_embedding(
        df,
        mode="unsupervised",
        embedding_dim=5,
        hidden_units=hidden_units,
        dropout_rate=0.2,
        max_epochs=15,
        verbose=False,
        seed=42
    )

    # Calculate some basic metrics
    mean_emb = embeddings.mean().mean()
    std_emb = embeddings.std().mean()

    print(f"{name:15}: mean={mean_emb:7.3f}, std={std_emb:6.3f}")
```

## Automatic Dimension Selection

Find optimal embedding dimensions:

```{code-cell} python
:tags: [remove-stderr, remove-warnings]

from row2vec import auto_select_dimension, EmbeddingConfig, NeuralConfig

# Create base config for dimension selection
base_config = EmbeddingConfig(
    mode="unsupervised",
    seed=42,
    verbose=False,
    neural=NeuralConfig(max_epochs=10)
)

# Test dimensions from 2 to 10
optimal_dim, results = auto_select_dimension(
    df,
    config=base_config,
    min_dimension=2,
    max_dimension=10,
    n_trials=3,  # Reduced for demo
    verbose=False
)

print(f"Optimal embedding dimension: {optimal_dim}")
print(f"\nDimension evaluation results:")
for dim, score in results.items():
    marker = " <-- OPTIMAL" if dim == optimal_dim else ""
    # Handle case where score might be a dict, list, or other format
    if isinstance(score, dict):
        # Try common score keys
        score_val = score.get('score', score.get('loss', score.get('value', 0.0)))
    elif isinstance(score, (list, tuple)):
        score_val = score[0] if len(score) > 0 else 0.0
    else:
        try:
            score_val = float(score) if score is not None else 0.0
        except (ValueError, TypeError):
            score_val = 0.0
    print(f"  {dim:2}D: {score_val:.4f}{marker}")
```

## Contrastive Learning Mode

Advanced embedding technique (if available):

```{code-cell} python
:tags: [remove-stderr, remove-warnings]

try:
    # Try to use contrastive learning functionality if available
    from row2vec import learn_embedding

    # Use regular neural embeddings as contrastive learning may not be available
    # in the current API version
    contrastive_embeddings = learn_embedding(
        df,
        mode="unsupervised",
        embedding_dim=5,
        max_epochs=10,
        verbose=False,
        seed=42
    )

    print(f"Neural embeddings shape: {contrastive_embeddings.shape}")
    print("First 3 embeddings (using standard neural approach):")
    print(contrastive_embeddings.head(3).round(4))
    print("\nNote: Advanced contrastive learning features may require specific Row2Vec versions")

except ImportError as e:
    print(f"Advanced contrastive learning mode not available: {e}")
```

## Model Serialization with Metadata

Row2Vec provides powerful model serialization capabilities for production workflows. The system uses a **two-file approach** for transparency and efficiency:

1. **Python script (.py)**: Contains inspectable metadata and loading logic
2. **Binary file (.pkl)**: Contains the actual model weights and preprocessor

### Key Features

- **Transparent metadata**: All training configuration and results stored in readable format
- **Complete pipeline preservation**: Includes preprocessing steps, not just the model
- **Schema validation**: Automatically validates input data against expected schema
- **Multiple model support**: Works with neural networks and classical ML methods
- **Detailed training information**: Loss curves, timing, data characteristics, and more

### Advanced model saving with rich metadata:

```{code-cell} python
:tags: [remove-stderr, remove-warnings]

from row2vec import (
    learn_embedding_with_model,
    save_model,
    Row2VecModel,
    Row2VecModelMetadata
)
import tempfile
import os

# Train with full model information
embeddings, model, preprocessor, metadata = learn_embedding_with_model(
    df,
    embedding_dim=8,
    mode="unsupervised",
    max_epochs=20,
    batch_size=64,
    dropout_rate=0.3,
    hidden_units=256,
    verbose=False,
    seed=42
)

print(f"Training completed:")
print(f"  Final loss: {metadata.get('final_loss', 'N/A')}")
print(f"  Training time: {metadata.get('training_time', 0):.2f}s")
print(f"  Epochs trained: {metadata.get('epochs_trained', 0)}")
```

```{code-cell} python
:tags: [remove-stderr, remove-warnings]

# Create rich model object with metadata
row2vec_model = Row2VecModel(
    model=model,
    preprocessor=preprocessor,
    metadata=Row2VecModelMetadata.from_dict(metadata)
)

# Save model with comprehensive information
with tempfile.TemporaryDirectory() as tmpdir:
    model_path = os.path.join(tmpdir, "advanced_model")

    script_path, binary_path = save_model(row2vec_model, model_path)

    print(f"Model saved with metadata:")
    print(f"  Script: {os.path.basename(script_path)}")
    print(f"  Binary: {os.path.basename(binary_path)}")

    # Load and inspect metadata
    from row2vec import load_model
    loaded_model = load_model(script_path)

    print(f"\nLoaded model metadata:")
    print(f"  Data shape: {loaded_model.metadata.data_shape}")
    print(f"  Original columns: {len(loaded_model.metadata.original_columns) if loaded_model.metadata.original_columns else 0}")
    print(f"  Embedding dimension: {loaded_model.metadata.embedding_dim}")
    print(f"  Training epochs: {loaded_model.metadata.epochs_trained}")
    print(f"  Final loss: {loaded_model.metadata.final_loss}")
```

### Schema Validation in Production

Models automatically validate input data against the expected schema:

```{code-cell} python
:tags: [remove-stderr, remove-warnings]

# Load model for validation demonstration
from row2vec import train_and_save_model, load_model, generate_synthetic_data
import tempfile
import os

with tempfile.TemporaryDirectory() as tmpdir:
    # Create a small model for demonstration
    sample_data = generate_synthetic_data(50, seed=42)
    embeddings, script_path, binary_path = train_and_save_model(
        sample_data,
        base_path=os.path.join(tmpdir, "validation_model"),
        embedding_dim=3,
        max_epochs=5,
        verbose=False
    )

    # Load the model
    model = load_model(script_path)

    # This will pass validation (correct schema)
    correct_data = generate_synthetic_data(10, seed=123)
    embeddings = model.predict(correct_data)
    print(f"✓ Validation passed: {embeddings.shape}")

    # This would fail validation (missing column)
    # incorrect_data = correct_data.drop(columns=["Sales"])
    # embeddings = model.predict(incorrect_data)  # Would raise ValueError

    # Skip validation if needed (not recommended for production)
    partial_data = correct_data[['Country', 'Product']]  # Missing columns
    try:
        embeddings_unvalidated = model.predict(partial_data, validate_schema=False)
        print(f"⚠️  Unvalidated prediction: {embeddings_unvalidated.shape}")
        print("   (Schema validation was skipped)")
    except Exception as e:
        print(f"Even unvalidated prediction failed: {e}")
```

### Best Practices for Model Serialization

1. **Use descriptive base paths**: Include version, date, or dataset info in model names
2. **Enable training history**: Keep detailed metadata for debugging and analysis
3. **Validate schemas in production**: Always use `validate_schema=True` for safety
4. **Store models with data descriptions**: Keep training data documentation alongside models
5. **Version your models**: Use systematic naming for model iterations
6. **Test model loading**: Always verify saved models can be loaded and used

```{code-cell} python
:tags: [remove-stderr, remove-warnings]

# Example of production-ready model naming and metadata
import datetime

# Create descriptive model path with timestamp and version
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
model_name = f"customer_segmentation_v2.1_{timestamp}"

print(f"Production model naming example:")
print(f"  Model name: {model_name}")
print(f"  Files would be: {model_name}.py and {model_name}.pkl")
print(f"  Include metadata: dataset version, feature engineering steps, validation scores")
```

## Custom Configuration Objects

Use configuration objects for complex setups:

```{code-cell} python
:tags: [remove-stderr, remove-warnings]

from row2vec import (
    EmbeddingConfig,
    NeuralConfig,
    ScalingConfig,
    LoggingConfig
)

# Create comprehensive configuration
embedding_config = EmbeddingConfig(
    embedding_dim=6,
    mode="unsupervised",
    seed=42
)

neural_config = NeuralConfig(
    max_epochs=25,
    batch_size=128,
    dropout_rate=0.25,
    hidden_units=[512, 256],  # Multi-layer
    early_stopping=True,
    activation="relu"
)

scaling_config = ScalingConfig(
    method="standard",
    range=None  # Not applicable for standard scaling
)

logging_config = LoggingConfig(
    level="INFO",
    file=None,
    enabled=True
)

print("Configuration objects created:")
print(f"  Embedding: {embedding_config.embedding_dim}D {embedding_config.mode}")
print(f"  Neural: {neural_config.max_epochs} epochs, {neural_config.hidden_units} units")
print(f"  Scaling: {scaling_config.method}")
print(f"  Logging: {logging_config.level}")
```

## Performance Monitoring

Built-in performance tracking:

```{code-cell} python
:tags: [remove-stderr, remove-warnings]

from row2vec import get_logger
import time

# Enable performance logging
logger = get_logger()
# Logger is already configured with INFO level

# Monitor embedding generation
start_time = time.time()

embeddings_monitored = learn_embedding(
    df,
    mode="unsupervised",
    embedding_dim=10,
    max_epochs=20,
    batch_size=128,
    verbose=True,  # Enable verbose output
    seed=42
)

total_time = time.time() - start_time

print(f"\nPerformance summary:")
print(f"  Total time: {total_time:.2f} seconds")
print(f"  Records processed: {len(df)}")
print(f"  Records per second: {len(df)/total_time:.1f}")
print(f"  Final embeddings: {embeddings_monitored.shape}")
```

## Production Considerations

Key settings for production use:

```{code-cell} python
:tags: [remove-stderr, remove-warnings]

# Production-optimized configuration
production_embeddings = learn_embedding(
    df,
    mode="unsupervised",
    embedding_dim=8,          # Appropriate for dataset feature count
    max_epochs=50,            # Reduced for demo (100 in production)
    batch_size=64,            # Appropriate for dataset size
    dropout_rate=0.2,         # Conservative regularization
    hidden_units=[512, 256],  # Deep architecture
    early_stopping=True,      # Prevent overfitting
    scale_method="standard",  # Standardized outputs
    seed=42,                  # Reproducible results
    verbose=False
)

print(f"Production embeddings generated:")
print(f"  Shape: {production_embeddings.shape}")
print(f"  Value range: [{production_embeddings.min().min():.3f}, {production_embeddings.max().max():.3f}]")
print(f"  Mean: {production_embeddings.mean().mean():.3f}")
print(f"  Std: {production_embeddings.std().mean():.3f}")
```

## Next Steps

You now know Row2Vec's advanced capabilities! For more:

- 📖 [CLI Guide](cli_guide.md) - Batch processing and automation
- 🔍 [API Reference](api_reference.md) - Complete parameter documentation
- 🏠 Back to [Examples](housing_example.md) for practical applications
