# Row2Vec: LLM Agent Documentation

## Overview

Row2Vec is a Python library for learning **low-dimensional embeddings** from tabular data rows. It transforms heterogeneous table rows (numeric and categorical features) into dense vector representations that preserve meaningful structure, enabling clustering, visualization, and downstream machine learning tasks.

**Key Use Cases:**
- Customer segmentation and profiling from transaction data
- Patient similarity analysis from electronic health records
- Feature engineering for machine learning pipelines
- Dimensionality reduction and visualization of high-dimensional tabular data
- Anomaly detection in structured datasets
- Transfer learning: pre-train embeddings on one dataset, fine-tune on another

**Installation:**
```bash
pip install row2vec
```

**Core Dependencies:**
- numpy, pandas, scikit-learn: Data manipulation and preprocessing
- tensorflow: Neural network-based embedding methods
- umap-learn: UMAP dimensionality reduction (optional)

**Core Philosophy:**

Traditional methods (PCA, t-SNE) work well for numeric data but struggle with mixed-type tables. Row2Vec handles:
- Mixed numeric and categorical features
- Missing values (multiple imputation strategies)
- Automatic categorical encoding
- Multiple embedding paradigms (supervised, unsupervised, contrastive)
- Neural architecture search for optimal model design

---

## Quick Start

### Minimal Working Example

```python
import row2vec
import pandas as pd

# Create sample data
data = pd.DataFrame({
    'age': [25, 34, 29, 42, 38],
    'income': [50000, 75000, 60000, 90000, 85000],
    'city': ['NYC', 'LA', 'NYC', 'SF', 'LA'],
    'purchased': [0, 1, 0, 1, 1]
})

# Learn 2D embeddings (unsupervised)
embeddings = row2vec.learn_embedding(
    data=data,
    embedding_size=2,
    method='autoencoder'
)

# Result: numpy array of shape (5, 2)
print(embeddings.shape)  # (5, 2)
```

### Basic Workflow with Target

```python
import row2vec
import pandas as pd

# Prepare data
X = pd.DataFrame({
    'age': [25, 34, 29, 42, 38, 45, 28],
    'income': [50000, 75000, 60000, 90000, 85000, 95000, 55000],
    'city': ['NYC', 'LA', 'NYC', 'SF', 'LA', 'SF', 'NYC'],
    'education': ['BS', 'MS', 'BS', 'PhD', 'MS', 'PhD', 'BS']
})

y = pd.Series([0, 1, 0, 1, 1, 1, 0])  # Target labels

# Learn supervised embeddings
embeddings = row2vec.learn_embedding(
    data=X,
    target=y,
    embedding_size=3,
    method='autoencoder',
    epochs=50
)

# Use embeddings for downstream tasks
from sklearn.cluster import KMeans
clusters = KMeans(n_clusters=2).fit_predict(embeddings)

# Or visualize
import matplotlib.pyplot as plt
plt.scatter(embeddings[:, 0], embeddings[:, 1], c=y)
plt.xlabel('Embedding Dim 1')
plt.ylabel('Embedding Dim 2')
plt.title('Row2Vec Embeddings')
plt.show()
```

### Comparing Multiple Methods

```python
import row2vec

methods = ['pca', 'autoencoder', 'umap', 'tsne']
results = {}

for method in methods:
    try:
        embeddings = row2vec.learn_embedding(
            data=X,
            target=y,
            embedding_size=2,
            method=method,
            verbose=0  # Suppress output
        )
        results[method] = embeddings
    except Exception as e:
        print(f"{method} failed: {e}")

# Compare visually
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, len(results), figsize=(16, 4))

for ax, (method, emb) in zip(axes, results.items()):
    ax.scatter(emb[:, 0], emb[:, 1], c=y, cmap='viridis')
    ax.set_title(f'{method.upper()}')
    ax.set_xlabel('Dim 1')
    ax.set_ylabel('Dim 2')

plt.tight_layout()
plt.show()
```

---

## Core Concepts

### Type System

Row2Vec uses standard data science types:

```python
import pandas as pd
import numpy as np
from typing import Optional, Union, Literal

# Input data types
DataInput = Union[pd.DataFrame, np.ndarray]
TargetInput = Optional[Union[pd.Series, np.ndarray]]

# Method selection
Method = Literal['autoencoder', 'variational', 'pca', 'tsne', 'umap', 'contrastive']

# Categorical encoding strategies
CategoricalStrategy = Literal['onehot', 'ordinal', 'target', 'binary']

# Missing value handling
ImputationStrategy = Literal['mean', 'median', 'most_frequent', 'constant', 'knn']

# Output
Embeddings = np.ndarray  # Shape: (n_samples, embedding_size)
```

### Common Parameters

All embedding methods share these core parameters:

```python
embeddings = row2vec.learn_embedding(
    data,                      # Required: DataFrame or array
    embedding_size=10,         # Required: output dimensionality
    method='autoencoder',      # Embedding method
    target=None,               # Optional: target for supervised learning
    categorical_features=None, # Optional: list of categorical column names
    normalize=True,            # Normalize numeric features
    handle_missing='mean',     # Missing value strategy
    random_state=42            # Reproducibility
)
```

**Parameter Guidelines:**

- `data`: pandas DataFrame or numpy array (n_samples, n_features)
- `embedding_size`: Target dimensionality (typically 2-100)
  - 2-3: Visualization
  - 10-50: Feature engineering
  - 50-100: Transfer learning
- `method`: Embedding algorithm (see Method Selection Guide below)
- `target`: For supervised methods, provide labels or continuous targets
- `categorical_features`: List of column names (auto-detected if None)
- `normalize`: Recommended for neural methods, optional for classical
- `handle_missing`: Strategy for missing values (applied before encoding)
- `random_state`: Set for reproducible results

### Method-Specific Parameters

**Neural Methods (autoencoder, variational, contrastive):**

```python
embeddings = row2vec.learn_embedding(
    data=X,
    embedding_size=16,
    method='autoencoder',

    # Architecture
    hidden_layers=[128, 64],   # Encoder layer sizes
    activation='relu',         # Activation function
    dropout_rate=0.2,          # Dropout regularization

    # Training
    epochs=100,                # Training iterations
    batch_size=32,             # Mini-batch size
    learning_rate=0.001,       # Optimizer learning rate
    verbose=1                  # Training progress (0, 1, 2)
)
```

**Classical Methods (PCA, t-SNE, UMAP):**

```python
# PCA
embeddings = row2vec.learn_embedding(
    data=X,
    embedding_size=10,
    method='pca',
    whiten=True                # Whitening transformation
)

# t-SNE
embeddings = row2vec.learn_embedding(
    data=X,
    embedding_size=2,
    method='tsne',
    perplexity=30,             # Neighborhood size
    n_iter=1000                # Optimization iterations
)

# UMAP
embeddings = row2vec.learn_embedding(
    data=X,
    embedding_size=2,
    method='umap',
    n_neighbors=15,            # Local neighborhood size
    min_dist=0.1               # Minimum distance in embedding
)
```

---

## Embedding Methods

### Method Selection Guide

**Decision Tree:**

1. **What's your data type?**

   **→ Purely numeric, linear relationships:**
   - Use: `method='pca'`
   - Fast, interpretable, works well for linear structure

   **→ Mixed numeric/categorical, complex relationships:**
   - Use: `method='autoencoder'` or `method='variational'`
   - Handles non-linearity, automatic feature interaction

   **→ Need visualization (2D/3D):**
   - Use: `method='umap'` or `method='tsne'`
   - Optimized for local structure preservation

2. **Do you have labels?**

   **→ Yes, supervised learning:**
   - Use: `method='autoencoder'` with `target=y`
   - Or: `method='contrastive'` for metric learning

   **→ No, unsupervised:**
   - Use: `method='autoencoder'`, `method='variational'`, or classical methods

3. **What's your priority?**

   **→ Speed:**
   - Use: `method='pca'` (fastest)
   - Avoid: `method='tsne'` (slowest)

   **→ Downstream ML performance:**
   - Use: `method='autoencoder'` or `method='variational'`
   - With: `target=y` for supervised guidance

   **→ Interpretability:**
   - Use: `method='pca'` (interpretable components)
   - Or: Classical methods with visualization

### Quick Reference Table

| Method | Type | Speed | Handles Categorical | Preserves | Best For |
|--------|------|-------|---------------------|-----------|----------|
| PCA | Linear | Fast | No* | Global variance | Numeric data, baseline |
| t-SNE | Non-linear | Slow | No* | Local structure | Visualization only |
| UMAP | Non-linear | Medium | No* | Local + global | Visualization, clustering |
| Autoencoder | Deep learning | Medium | Yes | Task-dependent | General purpose, mixed data |
| Variational | Deep learning | Medium | Yes | Probabilistic | Uncertainty quantification |
| Contrastive | Deep learning | Slow | Yes | Similarity | Metric learning, retrieval |

*Requires preprocessing with categorical encoding

---

## Embedding Methods: Detailed Guide

### 1. PCA (Principal Component Analysis)

**Mathematical Foundation:**
- Linear projection: maximizes variance in projected space
- Orthogonal components: uncorrelated output dimensions
- Formula: `X' = X @ W` where W are principal components

**When to use:**
- Numeric features only (or pre-encoded categorical)
- Linear relationships dominate
- Need fast, interpretable baseline
- Preprocessing for other methods

```python
import row2vec
import pandas as pd

# Numeric-only data
data = pd.DataFrame({
    'height': [170, 165, 180, 175, 160],
    'weight': [70, 60, 85, 78, 55],
    'age': [25, 30, 35, 28, 22]
})

# PCA embedding
embeddings = row2vec.learn_embedding(
    data=data,
    embedding_size=2,
    method='pca',
    normalize=True,   # Recommended
    whiten=True       # Standardize variance
)

# Explain variance
from row2vec import learn_embedding_with_model
embeddings, model = learn_embedding_with_model(
    data=data,
    embedding_size=2,
    method='pca',
    return_model=True
)

# Variance explained by each component
print(model.explained_variance_ratio_)
# [0.85, 0.13] - first component explains 85% of variance
```

**Limitations:**
- Cannot handle categorical data directly
- Assumes linear relationships
- Sensitive to outliers
- Cannot learn new representations (fixed projection)

### 2. Autoencoder (Deep Neural Network)

**Mathematical Foundation:**
- Encoder: `h = f(Wx + b)` - compresses input to bottleneck
- Decoder: `x' = g(W'h + b')` - reconstructs input
- Loss: `||x - x'||²` - reconstruction error
- Bottleneck layer `h` = learned embeddings

**When to use:**
- Mixed numeric and categorical features
- Complex, non-linear relationships
- Need flexible, learnable representations
- Have sufficient data (>1000 rows)

```python
import row2vec

# Mixed data
data = pd.DataFrame({
    'age': [25, 34, 29, 42, 38],
    'income': [50000, 75000, 60000, 90000, 85000],
    'city': ['NYC', 'LA', 'NYC', 'SF', 'LA'],
    'education': ['BS', 'MS', 'BS', 'PhD', 'MS']
})

# Autoencoder embedding
embeddings = row2vec.learn_embedding(
    data=data,
    embedding_size=8,
    method='autoencoder',
    hidden_layers=[32, 16],    # Encoder: 32→16→8 (bottleneck)
    activation='relu',
    dropout_rate=0.1,
    epochs=100,
    batch_size=32,
    normalize=True,
    verbose=1
)

# Supervised variant (with target)
y = pd.Series([0, 1, 0, 1, 1])  # Labels

embeddings = row2vec.learn_embedding(
    data=data,
    target=y,
    embedding_size=8,
    method='autoencoder',
    hidden_layers=[32, 16],
    epochs=100
)
# Now embeddings are optimized to separate classes
```

**Architecture Guidelines:**
```python
# Small datasets (<1000 rows)
hidden_layers=[32, 16]
embedding_size=8

# Medium datasets (1000-10000 rows)
hidden_layers=[128, 64]
embedding_size=32

# Large datasets (>10000 rows)
hidden_layers=[256, 128, 64]
embedding_size=64

# Rule of thumb: each layer should be 2x smaller than previous
```

### 3. Variational Autoencoder (VAE)

**Mathematical Foundation:**
- Probabilistic encoder: learns distribution `q(z|x)` (mean and variance)
- Latent space: samples `z ~ N(μ, σ²)`
- Loss: reconstruction + KL divergence
- Produces smooth, continuous embedding space

**When to use:**
- Need probabilistic embeddings (uncertainty estimates)
- Generation tasks (sample new synthetic rows)
- Smooth interpolation between data points
- Regularized embedding space

```python
import row2vec

embeddings = row2vec.learn_embedding(
    data=data,
    embedding_size=16,
    method='variational',
    hidden_layers=[64, 32],
    kl_weight=1.0,         # KL divergence weight (β-VAE)
    epochs=150,
    batch_size=32
)

# Generate synthetic data from learned distribution
from row2vec import generate_synthetic_data

synthetic = generate_synthetic_data(
    original_data=data,
    n_samples=100,
    method='variational',
    embedding_size=16
)
# Returns: DataFrame with same schema as original data
```

**β-VAE tuning:**
```python
# Standard VAE
kl_weight=1.0

# β < 1: Emphasize reconstruction (more faithful to data)
kl_weight=0.5

# β > 1: Emphasize latent structure (more disentangled)
kl_weight=2.0
```

### 4. t-SNE (t-Distributed Stochastic Neighbor Embedding)

**Mathematical Foundation:**
- Preserves local pairwise distances
- Converts distances to probabilities (Student's t-distribution)
- Minimizes KL divergence between high-D and low-D distributions

**When to use:**
- Visualization only (2D or 3D)
- Explore cluster structure
- Qualitative analysis

**Important limitations:**
- Non-parametric: cannot embed new data
- No global structure preservation
- Slow for large datasets
- Different runs give different results

```python
import row2vec

# t-SNE for visualization
embeddings = row2vec.learn_embedding(
    data=data,
    embedding_size=2,        # Always 2 or 3
    method='tsne',
    perplexity=30,           # Balance local/global structure
    n_iter=1000,             # Optimization steps
    random_state=42
)

# Perplexity tuning
# Small datasets (<100): perplexity=5-15
# Medium (100-1000): perplexity=30-50
# Large (>1000): perplexity=50-100
```

**Warning:** t-SNE cannot transform new data! If you need to embed new rows, use autoencoder or UMAP instead.

### 5. UMAP (Uniform Manifold Approximation and Projection)

**Mathematical Foundation:**
- Topological data analysis + manifold learning
- Preserves both local and global structure
- Faster than t-SNE, supports new data transformation

**When to use:**
- Visualization (superior to t-SNE in most cases)
- Clustering preprocessing
- Need to embed new data (unlike t-SNE)
- Large datasets

```python
import row2vec

embeddings = row2vec.learn_embedding(
    data=data,
    embedding_size=2,
    method='umap',
    n_neighbors=15,         # Local neighborhood size
    min_dist=0.1,           # Tightness of clusters
    metric='euclidean',     # Distance metric
    random_state=42
)

# Transform new data
from row2vec import load_model

model = load_model('embeddings.pkl')
new_embeddings = model.transform(new_data)
```

**Parameter Guidelines:**
```python
# Visualization (emphasize clusters)
n_neighbors=15
min_dist=0.1

# Preserve global structure
n_neighbors=50
min_dist=0.3

# Speed optimization (large datasets)
n_neighbors=10
min_dist=0.1
```

### 6. Contrastive Learning

**Mathematical Foundation:**
- Learns embeddings by comparing similar/dissimilar pairs
- Loss: minimize distance for similar pairs, maximize for dissimilar
- Creates metric space where semantic similarity = geometric proximity

**When to use:**
- Need similarity search or retrieval
- Few-shot learning scenarios
- Metric learning applications
- When explicit similarity labels are available

```python
import row2vec

# Basic contrastive learning (requires target)
embeddings = row2vec.learn_embedding(
    data=data,
    target=y,
    embedding_size=32,
    method='contrastive',
    margin=1.0,              # Contrastive margin
    epochs=100
)

# With explicit pair labels
# pairs = [(idx1, idx2, similarity), ...]
# embeddings = row2vec.learn_embedding_from_pairs(...)
```

---

## Data Preparation

### Handling Categorical Features

Row2Vec automatically handles categorical features with multiple encoding strategies:

**Automatic Detection:**
```python
import pandas as pd
import row2vec

data = pd.DataFrame({
    'age': [25, 30, 35],
    'city': ['NYC', 'LA', 'SF'],      # Auto-detected as categorical
    'income': [50000, 60000, 75000]
})

embeddings = row2vec.learn_embedding(
    data=data,
    embedding_size=10,
    method='autoencoder'
    # Categorical features detected automatically
)
```

**Manual Specification:**
```python
embeddings = row2vec.learn_embedding(
    data=data,
    embedding_size=10,
    method='autoencoder',
    categorical_features=['city', 'education']  # Explicit list
)
```

**Encoding Strategies:**

```python
# One-hot encoding (default)
embeddings = row2vec.learn_embedding(
    data=data,
    embedding_size=10,
    method='autoencoder',
    categorical_encoding='onehot'
)

# Ordinal encoding
embeddings = row2vec.learn_embedding(
    data=data,
    embedding_size=10,
    method='autoencoder',
    categorical_encoding='ordinal'
)

# Target encoding (requires target)
embeddings = row2vec.learn_embedding(
    data=data,
    target=y,
    embedding_size=10,
    method='autoencoder',
    categorical_encoding='target'
)

# Binary encoding
embeddings = row2vec.learn_embedding(
    data=data,
    embedding_size=10,
    method='autoencoder',
    categorical_encoding='binary'
)
```

### Handling Missing Values

Multiple strategies for missing data:

```python
import numpy as np
import pandas as pd

# Data with missing values
data = pd.DataFrame({
    'age': [25, np.nan, 35, 30],
    'income': [50000, 60000, np.nan, 70000],
    'city': ['NYC', 'LA', None, 'SF']
})

# Mean imputation (numeric)
embeddings = row2vec.learn_embedding(
    data=data,
    embedding_size=10,
    method='autoencoder',
    handle_missing='mean'
)

# Median imputation (robust to outliers)
embeddings = row2vec.learn_embedding(
    data=data,
    embedding_size=10,
    method='autoencoder',
    handle_missing='median'
)

# Most frequent (categorical)
embeddings = row2vec.learn_embedding(
    data=data,
    embedding_size=10,
    method='autoencoder',
    handle_missing='most_frequent'
)

# Constant value
embeddings = row2vec.learn_embedding(
    data=data,
    embedding_size=10,
    method='autoencoder',
    handle_missing='constant',
    fill_value=-999  # Custom constant
)

# K-Nearest Neighbors imputation
embeddings = row2vec.learn_embedding(
    data=data,
    embedding_size=10,
    method='autoencoder',
    handle_missing='knn',
    n_neighbors=5
)
```

### Normalization

Control numeric feature scaling:

```python
# Standard scaling (recommended for neural methods)
embeddings = row2vec.learn_embedding(
    data=data,
    embedding_size=10,
    method='autoencoder',
    normalize=True,
    normalization='standard'  # Mean=0, Std=1
)

# Min-max scaling
embeddings = row2vec.learn_embedding(
    data=data,
    embedding_size=10,
    method='autoencoder',
    normalize=True,
    normalization='minmax'  # Range [0, 1]
)

# Robust scaling (less sensitive to outliers)
embeddings = row2vec.learn_embedding(
    data=data,
    embedding_size=10,
    method='autoencoder',
    normalize=True,
    normalization='robust'  # Uses median and IQR
)

# No normalization
embeddings = row2vec.learn_embedding(
    data=data,
    embedding_size=10,
    method='autoencoder',
    normalize=False
)
```

---

## Advanced Features

### Neural Architecture Search (NAS)

Automatically find optimal network architecture:

```python
import row2vec

# Run NAS to find best architecture
best_embeddings, best_config = row2vec.neural_architecture_search(
    data=data,
    target=y,
    embedding_size=16,
    search_space={
        'hidden_layers': [[64, 32], [128, 64], [256, 128, 64]],
        'activation': ['relu', 'tanh', 'elu'],
        'dropout_rate': [0.1, 0.2, 0.3],
        'learning_rate': [0.001, 0.0001]
    },
    n_trials=20,           # Number of configurations to try
    evaluation_metric='reconstruction_error',  # Or 'classification_accuracy'
    cv_folds=5,            # Cross-validation
    verbose=1
)

print(f"Best configuration: {best_config}")
# {'hidden_layers': [128, 64], 'activation': 'relu', 'dropout_rate': 0.2, ...}

# Use best configuration
embeddings = row2vec.learn_embedding(
    data=data,
    embedding_size=16,
    method='autoencoder',
    **best_config
)
```

### Model Serialization

Save and load trained models:

```python
import row2vec

# Train and save
embeddings, model = row2vec.learn_embedding_with_model(
    data=data,
    embedding_size=16,
    method='autoencoder',
    return_model=True
)

# Save model
row2vec.save_model(model, 'embeddings.pkl')

# Load model
loaded_model = row2vec.load_model('embeddings.pkl')

# Transform new data
new_embeddings = loaded_model.transform(new_data)
```

### Automatic Dimensionality Selection

Let Row2Vec choose optimal embedding size:

```python
import row2vec

# Auto-select embedding size
embeddings, selected_size = row2vec.learn_embedding(
    data=data,
    embedding_size='auto',  # Automatic selection
    method='autoencoder',
    auto_dim_method='variance',  # Or 'reconstruction'
    variance_threshold=0.95  # Capture 95% of variance
)

print(f"Selected embedding size: {selected_size}")
```

### Transfer Learning

Pre-train on large dataset, fine-tune on small dataset:

```python
import row2vec

# Step 1: Pre-train on large dataset
pretrained_model = row2vec.learn_embedding_with_model(
    data=large_dataset,
    embedding_size=32,
    method='autoencoder',
    epochs=100,
    return_model=True
)[1]

# Step 2: Fine-tune on small target dataset
embeddings = row2vec.fine_tune_embedding(
    data=small_dataset,
    target=small_target,
    pretrained_model=pretrained_model,
    epochs=20,              # Fewer epochs for fine-tuning
    learning_rate=0.0001    # Lower learning rate
)
```

---

## Common Patterns

### Pattern 1: Data Exploration Pipeline

```python
import row2vec
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Load data
data = pd.read_csv('customers.csv')
print(f"Data shape: {data.shape}")

# Quick 2D visualization
embeddings_2d = row2vec.learn_embedding(
    data=data,
    embedding_size=2,
    method='umap',
    n_neighbors=15,
    verbose=0
)

# Plot
plt.figure(figsize=(10, 8))
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.5)
plt.xlabel('UMAP Dimension 1')
plt.ylabel('UMAP Dimension 2')
plt.title('Customer Embeddings')
plt.show()

# Find optimal number of clusters
embeddings_high = row2vec.learn_embedding(
    data=data,
    embedding_size=32,
    method='autoencoder',
    verbose=0
)

silhouette_scores = []
for n_clusters in range(2, 11):
    clusters = KMeans(n_clusters=n_clusters, random_state=42).fit_predict(embeddings_high)
    score = silhouette_score(embeddings_high, clusters)
    silhouette_scores.append(score)

# Best k
best_k = silhouette_scores.index(max(silhouette_scores)) + 2
print(f"Optimal number of clusters: {best_k}")

# Final clustering
final_clusters = KMeans(n_clusters=best_k, random_state=42).fit_predict(embeddings_high)

# Visualize clusters
plt.figure(figsize=(10, 8))
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=final_clusters, cmap='tab10', alpha=0.6)
plt.xlabel('UMAP Dimension 1')
plt.ylabel('UMAP Dimension 2')
plt.title(f'Customer Segments (k={best_k})')
plt.colorbar(label='Cluster')
plt.show()
```

### Pattern 2: Feature Engineering for ML

```python
import row2vec
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    data, target, test_size=0.2, random_state=42
)

# Baseline: Raw features
clf_baseline = RandomForestClassifier(random_state=42)
clf_baseline.fit(X_train, y_train)
baseline_acc = accuracy_score(y_test, clf_baseline.predict(X_test))
print(f"Baseline accuracy: {baseline_acc:.3f}")

# With Row2Vec embeddings
embeddings_train = row2vec.learn_embedding(
    data=X_train,
    target=y_train,
    embedding_size=32,
    method='autoencoder',
    epochs=100,
    verbose=0
)

# Transform test set
model = row2vec.learn_embedding_with_model(
    data=X_train,
    target=y_train,
    embedding_size=32,
    method='autoencoder',
    return_model=True
)[1]

embeddings_test = model.transform(X_test)

# Train on embeddings
clf_embeddings = RandomForestClassifier(random_state=42)
clf_embeddings.fit(embeddings_train, y_train)
embeddings_acc = accuracy_score(y_test, clf_embeddings.predict(embeddings_test))
print(f"Embeddings accuracy: {embeddings_acc:.3f}")
print(f"Improvement: {embeddings_acc - baseline_acc:.3f}")

# Combined features (concatenate)
import numpy as np
X_train_combined = np.hstack([X_train.values, embeddings_train])
X_test_combined = np.hstack([X_test.values, embeddings_test])

clf_combined = RandomForestClassifier(random_state=42)
clf_combined.fit(X_train_combined, y_train)
combined_acc = accuracy_score(y_test, clf_combined.predict(X_test_combined))
print(f"Combined accuracy: {combined_acc:.3f}")
```

### Pattern 3: Comparing Multiple Methods

```python
import row2vec
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import pandas as pd

methods = ['pca', 'autoencoder', 'variational', 'umap']
embedding_size = 16
results = []

for method in methods:
    try:
        # Learn embeddings
        embeddings = row2vec.learn_embedding(
            data=data,
            embedding_size=embedding_size,
            method=method,
            verbose=0
        )

        # Cluster
        clusters = KMeans(n_clusters=3, random_state=42).fit_predict(embeddings)

        # Evaluate
        silhouette = silhouette_score(embeddings, clusters)

        results.append({
            'method': method,
            'silhouette_score': silhouette,
            'n_samples': embeddings.shape[0],
            'embedding_shape': embeddings.shape
        })

    except Exception as e:
        print(f"{method} failed: {e}")

# Compare
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('silhouette_score', ascending=False)
print(results_df)

# Best method
best_method = results_df.iloc[0]['method']
print(f"\nBest method: {best_method}")
```

### Pattern 4: Handling Large Datasets

```python
import row2vec
import pandas as pd

# Large dataset
data = pd.read_csv('large_data.csv')  # e.g., 100K rows

# Option 1: Sample for faster prototyping
sample_data = data.sample(n=10000, random_state=42)
embeddings = row2vec.learn_embedding(
    data=sample_data,
    embedding_size=32,
    method='autoencoder',
    verbose=1
)

# Option 2: Use batch processing
embeddings = row2vec.learn_embedding(
    data=data,
    embedding_size=32,
    method='autoencoder',
    batch_size=256,      # Larger batches
    epochs=50,           # Fewer epochs
    verbose=1
)

# Option 3: Use faster method
embeddings = row2vec.learn_embedding(
    data=data,
    embedding_size=32,
    method='pca',        # Much faster
    verbose=0
)
```

### Pattern 5: Missing Value Analysis

```python
import row2vec
import pandas as pd
import numpy as np

# Data with missing values
data = pd.read_csv('data_with_missing.csv')

# Check missing proportion
missing_pct = data.isnull().sum() / len(data)
print(f"Missing value percentages:\n{missing_pct}")

# Compare imputation strategies
strategies = ['mean', 'median', 'most_frequent', 'knn']
results = {}

for strategy in strategies:
    embeddings = row2vec.learn_embedding(
        data=data,
        embedding_size=16,
        method='autoencoder',
        handle_missing=strategy,
        verbose=0
    )

    # Evaluate reconstruction quality
    model = row2vec.learn_embedding_with_model(
        data=data,
        embedding_size=16,
        method='autoencoder',
        handle_missing=strategy,
        return_model=True
    )[1]

    # Reconstruction error on non-missing values
    reconstructed = model.inverse_transform(embeddings)
    # Calculate RMSE on observed values
    # ... (implementation depends on your needs)

    results[strategy] = embeddings

print(f"Best strategy: {min(results, key=lambda k: ...)}")
```

---

## Integration Examples

### With scikit-learn

**As preprocessing step:**

```python
import row2vec
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# Create custom transformer
from sklearn.base import BaseEstimator, TransformerMixin

class Row2VecTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, embedding_size=32, method='autoencoder', epochs=50):
        self.embedding_size = embedding_size
        self.method = method
        self.epochs = epochs
        self.model_ = None

    def fit(self, X, y=None):
        _, self.model_ = row2vec.learn_embedding_with_model(
            data=X,
            target=y,
            embedding_size=self.embedding_size,
            method=self.method,
            epochs=self.epochs,
            return_model=True,
            verbose=0
        )
        return self

    def transform(self, X):
        return self.model_.transform(X)

# Build pipeline
pipeline = Pipeline([
    ('row2vec', Row2VecTransformer(embedding_size=32, epochs=50)),
    ('classifier', RandomForestClassifier(n_estimators=100))
])

# Cross-validation
scores = cross_val_score(pipeline, X, y, cv=5, scoring='accuracy')
print(f"CV accuracy: {scores.mean():.3f} ± {scores.std():.3f}")

# Fit and predict
pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)
```

### With pandas

**DataFrame in, DataFrame out:**

```python
import row2vec
import pandas as pd

# Input DataFrame
df = pd.DataFrame({
    'customer_id': range(100),
    'age': np.random.randint(18, 80, 100),
    'income': np.random.randint(30000, 150000, 100),
    'city': np.random.choice(['NYC', 'LA', 'SF', 'CHI'], 100)
})

# Learn embeddings
embeddings = row2vec.learn_embedding(
    data=df.drop('customer_id', axis=1),
    embedding_size=8,
    method='autoencoder'
)

# Create DataFrame with embeddings
embedding_cols = [f'emb_{i}' for i in range(embeddings.shape[1])]
embeddings_df = pd.DataFrame(embeddings, columns=embedding_cols)

# Combine with original IDs
result = pd.concat([
    df[['customer_id']].reset_index(drop=True),
    embeddings_df
], axis=1)

print(result.head())
# customer_id  emb_0  emb_1  emb_2  ...
```

### With matplotlib/seaborn

**Publication-quality visualizations:**

```python
import row2vec
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Configure plot style
sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10

# Learn embeddings
embeddings = row2vec.learn_embedding(
    data=data,
    target=labels,
    embedding_size=2,
    method='umap',
    verbose=0
)

# Create figure
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Colored by target
scatter1 = axes[0].scatter(
    embeddings[:, 0],
    embeddings[:, 1],
    c=labels,
    cmap='viridis',
    alpha=0.6,
    s=50
)
axes[0].set_xlabel('Embedding Dimension 1', fontsize=12)
axes[0].set_ylabel('Embedding Dimension 2', fontsize=12)
axes[0].set_title('Embeddings by Target', fontsize=14, fontweight='bold')
plt.colorbar(scatter1, ax=axes[0], label='Target')

# Plot 2: Density
from scipy.stats import gaussian_kde
xy = embeddings.T
z = gaussian_kde(xy)(xy)
scatter2 = axes[1].scatter(
    embeddings[:, 0],
    embeddings[:, 1],
    c=z,
    cmap='YlOrRd',
    alpha=0.6,
    s=50
)
axes[1].set_xlabel('Embedding Dimension 1', fontsize=12)
axes[1].set_ylabel('Embedding Dimension 2', fontsize=12)
axes[1].set_title('Embedding Density', fontsize=14, fontweight='bold')
plt.colorbar(scatter2, ax=axes[1], label='Density')

plt.tight_layout()
plt.savefig('embeddings_analysis.png', dpi=300, bbox_inches='tight')
plt.show()
```

### With TensorFlow/Keras

**Custom training loops:**

```python
import row2vec
import tensorflow as tf
from tensorflow import keras

# Get model architecture
embeddings, model = row2vec.learn_embedding_with_model(
    data=data,
    embedding_size=16,
    method='autoencoder',
    return_model=True
)

# Access Keras model
keras_model = model.model_

# Custom training with callbacks
early_stop = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

lr_schedule = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5
)

# Continue training
keras_model.fit(
    X_train,
    X_train,  # Autoencoder target = input
    validation_split=0.2,
    epochs=100,
    batch_size=32,
    callbacks=[early_stop, lr_schedule],
    verbose=1
)

# Extract embeddings from trained model
encoder = keras_model.get_layer('encoder')
new_embeddings = encoder.predict(X_test)
```

---

## Troubleshooting

### Common Errors and Solutions

**Error: TensorFlow GPU not available**

```python
# Problem
embeddings = row2vec.learn_embedding(...)  # Falls back to CPU

# Solution 1: Check GPU availability
import tensorflow as tf
print("GPUs available:", tf.config.list_physical_devices('GPU'))

# Solution 2: Force CPU usage (if GPU causes issues)
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Solution 3: Configure GPU memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
```

**Error: Out of memory during training**

```python
# Problem
embeddings = row2vec.learn_embedding(
    data=large_data,
    embedding_size=128,
    method='autoencoder'
)  # OOM error

# Solution 1: Reduce batch size
embeddings = row2vec.learn_embedding(
    data=large_data,
    embedding_size=128,
    method='autoencoder',
    batch_size=16  # Smaller batches
)

# Solution 2: Reduce model size
embeddings = row2vec.learn_embedding(
    data=large_data,
    embedding_size=64,  # Smaller embedding
    method='autoencoder',
    hidden_layers=[64, 32]  # Smaller layers
)

# Solution 3: Use classical method
embeddings = row2vec.learn_embedding(
    data=large_data,
    embedding_size=64,
    method='pca'  # No GPU memory required
)
```

**Error: Categorical encoding fails**

```python
# Problem: Unknown categories in test set
model.transform(test_data)  # ValueError: Unknown category

# Solution 1: Handle unknown categories
embeddings, model = row2vec.learn_embedding_with_model(
    data=train_data,
    embedding_size=32,
    method='autoencoder',
    categorical_encoding='onehot',
    handle_unknown='ignore',  # Treat unknown as all-zeros
    return_model=True
)

# Solution 2: Use ordinal encoding
embeddings, model = row2vec.learn_embedding_with_model(
    data=train_data,
    embedding_size=32,
    method='autoencoder',
    categorical_encoding='ordinal',  # Assigns integer to unknown
    return_model=True
)
```

**Error: Embeddings all similar (collapsed)**

```python
# Problem: All embeddings are nearly identical
embeddings = row2vec.learn_embedding(...)
print(embeddings.std(axis=0))  # Very low variance

# Solution 1: Check data normalization
embeddings = row2vec.learn_embedding(
    data=data,
    embedding_size=32,
    method='autoencoder',
    normalize=True  # Enable normalization
)

# Solution 2: Reduce regularization
embeddings = row2vec.learn_embedding(
    data=data,
    embedding_size=32,
    method='autoencoder',
    dropout_rate=0.0,  # Disable dropout
    weight_decay=0.0   # Disable L2 regularization
)

# Solution 3: Use different method
embeddings = row2vec.learn_embedding(
    data=data,
    embedding_size=32,
    method='variational',  # VAE less prone to collapse
    kl_weight=0.5
)
```

### Performance Tips

**1. Method selection for speed:**

```python
# Fast (< 1 second for 10K rows)
embeddings = row2vec.learn_embedding(data, embedding_size=32, method='pca')

# Medium (seconds to minutes)
embeddings = row2vec.learn_embedding(data, embedding_size=32, method='umap')
embeddings = row2vec.learn_embedding(data, embedding_size=32, method='autoencoder', epochs=50)

# Slow (minutes to hours)
embeddings = row2vec.learn_embedding(data, embedding_size=32, method='tsne')
embeddings = row2vec.learn_embedding(data, embedding_size=32, method='autoencoder', epochs=500)
```

**2. Batch processing for large datasets:**

```python
# Inefficient: Process entire dataset at once
embeddings = row2vec.learn_embedding(
    data=huge_data,  # 1M rows
    embedding_size=32,
    method='autoencoder'
)

# Efficient: Use larger batch size
embeddings = row2vec.learn_embedding(
    data=huge_data,
    embedding_size=32,
    method='autoencoder',
    batch_size=512  # Larger batches = faster
)
```

**3. Early stopping:**

```python
# Train until convergence (automatically stops)
embeddings = row2vec.learn_embedding(
    data=data,
    embedding_size=32,
    method='autoencoder',
    epochs=1000,  # Max epochs
    early_stopping=True,
    patience=10  # Stop if no improvement for 10 epochs
)
```

### Memory Optimization

**Monitor memory usage:**

```python
import psutil
import os

process = psutil.Process(os.getpid())
mem_before = process.memory_info().rss / 1024 / 1024

embeddings = row2vec.learn_embedding(
    data=data,
    embedding_size=32,
    method='autoencoder'
)

mem_after = process.memory_info().rss / 1024 / 1024
print(f"Memory used: {mem_after - mem_before:.1f} MB")
```

**Reduce memory footprint:**

```python
# Clear TensorFlow session after training
import tensorflow as tf
from tensorflow import keras

embeddings = row2vec.learn_embedding(...)

# Clear session
keras.backend.clear_session()
tf.compat.v1.reset_default_graph()
```

---

## Best Practices

### 1. Data Preprocessing

**Always check data quality first:**

```python
import pandas as pd
import numpy as np

# Check missing values
print(data.isnull().sum())

# Check data types
print(data.dtypes)

# Check for infinite values
print(np.isinf(data.select_dtypes(include=[np.number])).sum())

# Check for constant columns
constant_cols = [col for col in data.columns if data[col].nunique() == 1]
if constant_cols:
    print(f"Constant columns: {constant_cols}")
    data = data.drop(constant_cols, axis=1)
```

### 2. Start Simple, Then Optimize

```python
# Step 1: Baseline with PCA
embeddings_pca = row2vec.learn_embedding(
    data=data,
    embedding_size=32,
    method='pca'
)

# Step 2: Try autoencoder with default settings
embeddings_ae = row2vec.learn_embedding(
    data=data,
    embedding_size=32,
    method='autoencoder'
)

# Step 3: Hyperparameter tuning (if needed)
best_embeddings, best_config = row2vec.neural_architecture_search(
    data=data,
    embedding_size=32,
    n_trials=10
)
```

### 3. Validate on Downstream Tasks

**Never optimize embeddings in isolation:**

```python
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

def evaluate_embeddings(embeddings, labels):
    """Evaluate embedding quality."""

    # Clustering quality
    clusters = KMeans(n_clusters=len(np.unique(labels)), random_state=42).fit_predict(embeddings)
    silhouette = silhouette_score(embeddings, clusters)

    # Classification quality (if labels available)
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    clf = LogisticRegression(max_iter=1000)
    cv_scores = cross_val_score(clf, embeddings, labels, cv=5)

    return {
        'silhouette': silhouette,
        'classification_accuracy': cv_scores.mean()
    }

# Compare methods
results = {}
for method in ['pca', 'autoencoder', 'umap']:
    emb = row2vec.learn_embedding(data, embedding_size=32, method=method, verbose=0)
    results[method] = evaluate_embeddings(emb, labels)

print(results)
```

### 4. Use Appropriate Embedding Size

```python
# Rule of thumb
n_samples = len(data)
n_features = len(data.columns)

if n_samples < 100:
    embedding_size = min(8, n_features // 2)
elif n_samples < 1000:
    embedding_size = min(32, n_features // 2)
else:
    embedding_size = min(64, n_features // 2)

print(f"Recommended embedding size: {embedding_size}")
```

### 5. Save Models for Production

```python
# Train once
embeddings, model = row2vec.learn_embedding_with_model(
    data=train_data,
    embedding_size=32,
    method='autoencoder',
    return_model=True
)

# Save
row2vec.save_model(model, 'production_model.pkl')

# In production: load and transform
model = row2vec.load_model('production_model.pkl')
new_embeddings = model.transform(new_data)
```

---

## Complete Example: Customer Segmentation Pipeline

```python
import row2vec
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ===== 1. Load and Inspect Data =====
data = pd.read_csv('customers.csv')
print(f"Data shape: {data.shape}")
print(f"Columns: {data.columns.tolist()}")
print(f"Missing values:\n{data.isnull().sum()}")

# ===== 2. Data Preprocessing =====
# Drop ID columns
data_clean = data.drop(['customer_id', 'registration_date'], axis=1)

# Check data types
print(f"Data types:\n{data_clean.dtypes}")

# ===== 3. Learn Embeddings =====
print("\nLearning embeddings...")
embeddings = row2vec.learn_embedding(
    data=data_clean,
    embedding_size=32,
    method='autoencoder',
    hidden_layers=[128, 64],
    epochs=100,
    batch_size=32,
    handle_missing='mean',
    normalize=True,
    verbose=1
)

# ===== 4. Find Optimal Number of Clusters =====
print("\nFinding optimal number of clusters...")
silhouette_scores = []
k_range = range(2, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(embeddings)
    score = silhouette_score(embeddings, clusters)
    silhouette_scores.append(score)
    print(f"k={k}: silhouette={score:.3f}")

# Best k
best_k = k_range[np.argmax(silhouette_scores)]
print(f"\nOptimal k: {best_k}")

# ===== 5. Final Clustering =====
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
final_clusters = kmeans.fit_predict(embeddings)

# Add clusters to original data
data['segment'] = final_clusters

# ===== 6. Segment Analysis =====
print("\nSegment statistics:")
segment_stats = data.groupby('segment').agg({
    'age': ['mean', 'std'],
    'income': ['mean', 'std'],
    'purchase_count': ['mean', 'sum']
}).round(2)
print(segment_stats)

# ===== 7. Visualization =====
# Create 2D embeddings for visualization
embeddings_2d = row2vec.learn_embedding(
    data=data_clean,
    embedding_size=2,
    method='umap',
    n_neighbors=15,
    verbose=0
)

# Plot setup
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Plot 1: Silhouette scores
axes[0, 0].plot(k_range, silhouette_scores, 'o-')
axes[0, 0].axvline(best_k, color='r', linestyle='--', label=f'Optimal k={best_k}')
axes[0, 0].set_xlabel('Number of Clusters')
axes[0, 0].set_ylabel('Silhouette Score')
axes[0, 0].set_title('Cluster Optimization')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: 2D embeddings colored by segment
scatter1 = axes[0, 1].scatter(
    embeddings_2d[:, 0],
    embeddings_2d[:, 1],
    c=final_clusters,
    cmap='tab10',
    alpha=0.6,
    s=50
)
axes[0, 1].set_xlabel('UMAP Dimension 1')
axes[0, 1].set_ylabel('UMAP Dimension 2')
axes[0, 1].set_title('Customer Segments')
plt.colorbar(scatter1, ax=axes[0, 1], label='Segment')

# Plot 3: Segment sizes
segment_sizes = data['segment'].value_counts().sort_index()
axes[1, 0].bar(segment_sizes.index, segment_sizes.values, color='steelblue')
axes[1, 0].set_xlabel('Segment')
axes[1, 0].set_ylabel('Number of Customers')
axes[1, 0].set_title('Segment Sizes')
axes[1, 0].grid(True, alpha=0.3, axis='y')

# Plot 4: Segment characteristics (age vs income)
for segment in range(best_k):
    segment_data = data[data['segment'] == segment]
    axes[1, 1].scatter(
        segment_data['age'],
        segment_data['income'],
        label=f'Segment {segment}',
        alpha=0.6
    )
axes[1, 1].set_xlabel('Age')
axes[1, 1].set_ylabel('Income')
axes[1, 1].set_title('Segment Characteristics')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('customer_segmentation.png', dpi=300, bbox_inches='tight')
plt.show()

# ===== 8. Export Results =====
# Save model
_, model = row2vec.learn_embedding_with_model(
    data=data_clean,
    embedding_size=32,
    method='autoencoder',
    return_model=True
)
row2vec.save_model(model, 'customer_embedding_model.pkl')

# Save embeddings
embeddings_df = pd.DataFrame(
    embeddings,
    columns=[f'emb_{i}' for i in range(embeddings.shape[1])]
)
result = pd.concat([data[['customer_id']], embeddings_df, data[['segment']]], axis=1)
result.to_csv('customer_embeddings.csv', index=False)

print("\n✓ Pipeline complete!")
print(f"✓ Model saved: customer_embedding_model.pkl")
print(f"✓ Embeddings saved: customer_embeddings.csv")
print(f"✓ Visualization saved: customer_segmentation.png")
```

---

## Package Information

**Version:** 0.1.0
**Python:** Requires 3.9+
**Core dependencies:** numpy, pandas, scikit-learn, tensorflow
**Optional dependencies:** umap-learn (for UMAP method)
**License:** MIT
**Repository:** https://github.com/tresoldi/row2vec

### Import Structure

```python
# Main functionality
import row2vec
from row2vec import (
    learn_embedding,
    learn_embedding_with_model,
    generate_synthetic_data,
)

# Model management
from row2vec import (
    save_model,
    load_model,
    fine_tune_embedding,
)

# Advanced features
from row2vec import (
    neural_architecture_search,
)

# Utilities
from row2vec.utils import (
    encode_categorical,
    handle_missing_values,
)
```

### Dependencies Summary

| Package | Version | Purpose |
|---------|---------|---------|
| numpy | ≥1.20 | Array operations |
| pandas | ≥1.3 | DataFrame handling |
| scikit-learn | ≥1.0 | Classical methods, preprocessing |
| tensorflow | ≥2.10 | Neural network methods |
| umap-learn | ≥0.5 | UMAP algorithm (optional) |

---

This documentation provides comprehensive guidance for LLM agents to effectively use Row2Vec in their projects. For the latest updates, examples, and tutorials, visit the [GitHub repository](https://github.com/tresoldi/row2vec).
