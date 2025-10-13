#' # Advanced Features
#'
#' This tutorial explores advanced Row2Vec features including neural architecture search,
#' adaptive imputation, and model serialization.
#'
#' ## Neural Architecture Search
#'
#' Automatically discover optimal network architectures for your data:

# | hide
import warnings

warnings.filterwarnings("ignore")
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# |

from row2vec import (
    ArchitectureSearchConfig,
    EmbeddingConfig,
    NeuralConfig,
    generate_synthetic_data,
    search_architecture,
)

# Generate sample data
df = generate_synthetic_data(num_records=500, seed=42)
print(f"Dataset: {df.shape}")

#' ### Configure Architecture Search

# Define search space
search_config = ArchitectureSearchConfig(
    method="random",
    max_layers=3,
    width_options=[64, 128, 256],
    max_trials=10,  # Reduced for demo
)

# Base embedding configuration
base_config = EmbeddingConfig(
    mode="unsupervised",
    embedding_dim=8,
    neural=NeuralConfig(max_epochs=30, verbose=False),
)

print("Search configuration:")
print(f"  Max layers: {search_config.max_layers}")
print(f"  Width options: {search_config.width_options}")
print(f"  Max trials: {search_config.max_trials}")

#' ### Run Architecture Search

# Find optimal architecture
best_arch, results = search_architecture(df, base_config, search_config)

print("\nBest architecture found:")
print(f"  Hidden units: {best_arch.get('hidden_units', [])}")
print(f"  Dropout rate: {best_arch.get('dropout_rate', 0.0)}")
print(f"  Loss: {best_arch.get('final_loss', 'N/A')}")

#' ## Adaptive Missing Value Imputation
#'
#' Row2Vec includes intelligent imputation strategies that adapt to your data patterns:

import numpy as np
import pandas as pd

from row2vec import AdaptiveImputer, ImputationConfig, MissingPatternAnalyzer

# Create data with complex missing patterns
df_missing = df.copy()
df_missing.loc[0:10, "Sales"] = np.nan  # Completely missing block
df_missing.loc[df_missing["Product"] == "A", "Sales"] = np.nan  # Conditional missing

print(f"\nMissing values: {df_missing.isnull().sum().sum()}")

#' ### Analyze Missing Patterns

analyzer = MissingPatternAnalyzer(ImputationConfig())
analysis = analyzer.analyze(df_missing)

print("\nMissing pattern analysis:")
print(f"  Problematic columns: {analysis.get('problematic_columns', [])}")
print(f"  Recommendations: {len(analysis.get('recommendations', []))} strategies")

#' ### Apply Adaptive Imputation

imputer = AdaptiveImputer(
    ImputationConfig(
        numeric_strategy="knn", categorical_strategy="mode", knn_neighbors=10
    )
)

df_imputed = imputer.fit_transform(df_missing)

print("\nAfter imputation:")
print(f"  Remaining missing values: {df_imputed.isnull().sum().sum()}")
print("  Successfully imputed!")

#' ## Model Serialization
#'
#' Train once, use many times by saving and loading models:

import os
import tempfile

from row2vec import learn_embedding, load_model, train_and_save_model

# Create temporary directory for demo
tmpdir = tempfile.mkdtemp()
model_path = os.path.join(tmpdir, "my_model")

#' ### Train and Save Model

embeddings, script_path, binary_path = train_and_save_model(
    df,
    base_path=model_path,
    embedding_dim=6,
    mode="unsupervised",
    max_epochs=20,
    verbose=False,
)

print("\nModel saved:")
print(f"  Script: {os.path.basename(script_path)}")
print(f"  Binary: {os.path.basename(binary_path)}")
print(f"  Training embeddings: {embeddings.shape}")

#' ### Load and Use Saved Model

# Load the model
model = load_model(script_path)

# Generate embeddings for new data
new_data = generate_synthetic_data(num_records=50, seed=999)
new_embeddings = model.predict(new_data)

print("\nUsing loaded model:")
print(f"  New data: {new_data.shape}")
print(f"  New embeddings: {new_embeddings.shape}")
print("  Model successfully reused!")

# Cleanup
import shutil

shutil.rmtree(tmpdir)

#' ## Automated Dimension Selection
#'
#' Let Row2Vec automatically determine the optimal embedding dimension:

from row2vec import AutoDimensionSelector, auto_select_dimension

selector = AutoDimensionSelector(min_dim=2, max_dim=10, method="reconstruction_error")

#' Run dimension selection (this analyzes reconstruction quality at different dimensions)

print("\nAutomatic dimension selection:")
print(f"  Testing dimensions from {selector.min_dim} to {selector.max_dim}")
print(f"  Method: {selector.method}")

# Note: Full auto-selection can be time-consuming, shown here for demonstration
suggested_dim = 6  # In practice, use: selector.select(df)

print(f"  Suggested dimension: {suggested_dim}")

#' ## Categorical Encoding Options
#'
#' Row2Vec provides multiple strategies for encoding categorical features:

from row2vec import CategoricalEncoder, CategoricalEncodingConfig

# Configure entity embeddings for categories
encoding_config = CategoricalEncodingConfig(
    method="entity_embedding", embedding_dim=4, handle_unknown="default"
)

encoder = CategoricalEncoder(encoding_config)
df_encoded = encoder.fit_transform(df[["Product", "Country"]])

print("\nCategorical encoding:")
print(f"  Original columns: {list(df[['Product', 'Country']].columns)}")
print(f"  Encoded shape: {df_encoded.shape}")
print(f"  Method: {encoding_config.method}")

#' ## Configuration-Based Workflow
#'
#' For complex pipelines, use configuration objects for full control:

from row2vec import EmbeddingConfig, NeuralConfig, PreprocessingConfig, ScalingConfig

# Comprehensive configuration
full_config = EmbeddingConfig(
    mode="unsupervised",
    embedding_dim=8,
    neural=NeuralConfig(
        hidden_units=[128, 64],
        max_epochs=50,
        batch_size=32,
        dropout_rate=0.2,
        learning_rate=0.001,
        verbose=False,
    ),
    preprocessing=PreprocessingConfig(
        scaling=ScalingConfig(method="standard", feature_range=(0, 1))
    ),
)

print("\nConfiguration-based training:")
print(f"  Embedding dim: {full_config.embedding_dim}")
print(f"  Hidden layers: {full_config.neural.hidden_units}")
print(f"  Epochs: {full_config.neural.max_epochs}")
print(f"  Dropout: {full_config.neural.dropout_rate}")

# Train with configuration
from row2vec import learn_embedding_v2

final_embeddings = learn_embedding_v2(df, full_config)
print(f"  Result: {final_embeddings.shape}")

#' ## Performance Optimization Tips
#'
#' For large datasets, consider these optimizations:
#'
#' 1. **Use batch processing**: Increase `batch_size` for faster training
#' 2. **Reduce epochs**: Start with fewer epochs and monitor convergence
#' 3. **Parallel testing**: Use `pytest-xdist` for faster test execution
#' 4. **Classical methods**: PCA and UMAP are faster for initial exploration
#'
#' ## Summary
#'
#' You've learned:
#' - Neural architecture search for optimal models
#' - Adaptive imputation for missing data
#' - Model serialization for reusable pipelines
#' - Automatic dimension selection
#' - Configuration-based workflows
#'
#' These advanced features enable production-ready embedding pipelines!
