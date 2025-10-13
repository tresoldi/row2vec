# Row2Vec User Guide

A comprehensive guide to generating vector embeddings from tabular data.

## Table of Contents

1. [Introduction](#introduction)
2. [Mathematical Background](#mathematical-background)
3. [Embedding Methods Overview](#embedding-methods-overview)
4. [Data Preparation](#data-preparation)
5. [Neural Network Embeddings](#neural-network-embeddings)
6. [Classical Methods](#classical-methods)
7. [Advanced Features](#advanced-features)
8. [Model Serialization](#model-serialization)
9. [Best Practices](#best-practices)
10. [Common Use Cases](#common-use-cases)

## Introduction

### What is Row2Vec?

Row2Vec is a comprehensive library for generating low-dimensional vector embeddings from tabular datasets. It transforms mixed-type tabular data (numeric and categorical columns) into dense vector representations suitable for machine learning, visualization, and analysis.

**Key Concept:** Each **row** in your dataset becomes a fixed-length **vector** in a lower-dimensional embedding space, while preserving the important relationships and patterns in your data.

### Why Vector Embeddings?

Tabular data embeddings solve several common problems:

**Dimensionality Reduction:**
- Raw data: 50+ columns → Embeddings: 5-20 dimensions
- Reduces computational cost
- Prevents overfitting
- Enables visualization

**Feature Engineering:**
- Automatically discovers complex patterns
- Captures non-linear relationships
- Creates universal representations usable across models

**Mixed-Type Data Handling:**
- Seamlessly combines numeric and categorical features
- No manual one-hot encoding needed
- Handles missing values automatically

**Visualization:**
- 2D/3D embeddings reveal clusters and patterns
- t-SNE and UMAP for exploration
- Identify outliers and structure

### When to Use Row2Vec

**Use Row2Vec when:**
- You have tabular data with mixed types (numeric + categorical)
- You need to reduce dimensionality while preserving information
- You want to visualize high-dimensional data
- You need features for downstream machine learning
- You want to discover latent structure in your data

**Don't use Row2Vec when:**
- You have pure text data (use word embeddings instead)
- You have image data (use CNNs instead)
- You have < 3 features (no dimensionality to reduce)
- You need fully interpretable features (use feature selection instead)

### Library Features

- **Multiple Methods**: Neural (autoencoder), PCA, t-SNE, UMAP, contrastive
- **Intelligent Preprocessing**: Automatic scaling, encoding, imputation
- **Neural Architecture Search**: Find optimal network structures automatically
- **Model Persistence**: Save and reuse trained models
- **Production Ready**: Full type hints, comprehensive tests, CI/CD
- **Flexible Configuration**: Simple API or detailed configuration objects

### Installation

```bash
# Core installation
pip install row2vec

# With development tools
pip install row2vec[dev]
```

## Mathematical Background

### The Embedding Problem

Given a dataset $D$ with $n$ rows and $p$ features:

$$D \in \mathbb{R}^{n \times p}$$

We want to learn a mapping $f: \mathbb{R}^p \rightarrow \mathbb{R}^d$ where $d \ll p$:

$$\mathbf{z}_i = f(\mathbf{x}_i) \quad \text{for } i = 1, \ldots, n$$

such that the embeddings $\mathbf{z}_i$ preserve important properties of the original data.

### Autoencoder Framework

Row2Vec's neural method uses autoencoders: neural networks trained to reconstruct their input.

**Architecture:**

$$
\begin{align}
\text{Encoder:} \quad & \mathbf{h} = \sigma(W_e \mathbf{x} + \mathbf{b}_e) \\
\text{Bottleneck:} \quad & \mathbf{z} = \sigma(W_b \mathbf{h} + \mathbf{b}_b) \\
\text{Decoder:} \quad & \mathbf{\hat{x}} = \sigma(W_d \mathbf{z} + \mathbf{b}_d)
\end{align}
$$

**Objective:** Minimize reconstruction error:

$$\mathcal{L} = \frac{1}{n} \sum_{i=1}^{n} \|\mathbf{x}_i - \mathbf{\hat{x}}_i\|^2$$

The bottleneck layer $\mathbf{z}$ forces the network to learn a compressed representation.

### Principal Component Analysis (PCA)

PCA finds orthogonal directions of maximum variance:

$$\mathbf{Z} = \mathbf{X} \mathbf{W}$$

where $\mathbf{W}$ are the top $d$ eigenvectors of the covariance matrix.

**Properties:**
- Linear transformation
- Optimal for Gaussian data
- Deterministic (no random initialization)
- Interpretable components

### t-SNE (t-Distributed Stochastic Neighbor Embedding)

t-SNE preserves local neighborhood structure by minimizing:

$$\text{KL}(P \| Q) = \sum_{i \neq j} p_{ij} \log \frac{p_{ij}}{q_{ij}}$$

where $P$ represents similarities in high-dimensional space and $Q$ in low-dimensional space.

**Properties:**
- Non-linear, preserves local structure
- Excellent for visualization
- Stochastic (different runs give different results)
- Slow for large datasets

### UMAP (Uniform Manifold Approximation and Projection)

UMAP constructs a fuzzy topological representation and optimizes:

$$\sum_{i,j} \left[ w_{ij} \log \frac{w_{ij}}{q_{ij}} + (1-w_{ij}) \log \frac{1-w_{ij}}{1-q_{ij}} \right]$$

**Properties:**
- Non-linear, balances local and global structure
- Faster than t-SNE
- Better preserves global structure
- Supports higher-dimensional outputs

## Embedding Methods Overview

### Method Comparison

| Method | Speed | Type | Best For | Dimensions |
|--------|-------|------|----------|------------|
| **PCA** | Fast | Linear | Quick exploration, linear relationships | 2-50 |
| **Neural** | Medium | Non-linear | Complex patterns, feature engineering | 5-100 |
| **t-SNE** | Slow | Non-linear | 2D/3D visualization, cluster discovery | 2-3 |
| **UMAP** | Fast | Non-linear | General purpose, balanced structure | 2-50 |
| **Contrastive** | Medium | Non-linear | When you have positive/negative pairs | 5-100 |

### Method Selection Guide

**Choose PCA when:**
- You need fast, deterministic results
- Your relationships are primarily linear
- You want interpretable components
- You're doing initial exploration

**Choose Neural (Autoencoder) when:**
- You have complex, non-linear patterns
- You need embeddings for downstream ML
- You have sufficient data (>1000 rows)
- You want to fine-tune architecture

**Choose t-SNE when:**
- You want 2D/3D visualization
- Discovering clusters is primary goal
- Local neighborhood is most important
- You have time for computation

**Choose UMAP when:**
- You want general-purpose embeddings
- You need both local and global structure
- You want faster computation than t-SNE
- You might need >3 dimensions

**Choose Contrastive when:**
- You have labeled similar/dissimilar pairs
- You want to enforce specific relationships
- You need metric learning
- You have domain knowledge about similarity

## Data Preparation

### Input Data Format

Row2Vec accepts pandas DataFrames with mixed types:

```python
import pandas as pd

# Example dataset
data = pd.DataFrame({
    'age': [25, 35, 45, 30],
    'income': [50000, 75000, 90000, 60000],
    'city': ['NYC', 'LA', 'Chicago', 'NYC'],
    'education': ['BS', 'MS', 'PhD', 'BS']
})
```

**Requirements:**
- DataFrame format (pandas)
- At least 3 rows (preferably 100+)
- At least 2 features
- Mixed numeric and categorical OK
- Missing values OK (automatically handled)

### Handling Missing Values

Row2Vec automatically handles missing data with intelligent imputation:

```python
from row2vec import learn_embedding

# Data with missing values
data_missing = data.copy()
data_missing.loc[0, 'income'] = np.nan
data_missing.loc[2, 'city'] = np.nan

# Automatically imputed during embedding
embeddings = learn_embedding(data_missing, mode="unsupervised", embedding_dim=5)
```

**Imputation Strategies:**
- **Numeric columns**: KNN imputation (default) or mean/median
- **Categorical columns**: Mode or most frequent
- **Pattern analysis**: Detects systematic missingness

**Manual Configuration:**

```python
from row2vec import EmbeddingConfig, PreprocessingConfig, ImputationConfig

config = EmbeddingConfig(
    mode="unsupervised",
    embedding_dim=5,
    preprocessing=PreprocessingConfig(
        imputation=ImputationConfig(
            numeric_strategy='knn',
            categorical_strategy='mode',
            knn_neighbors=5
        )
    )
)
```

### Data Scaling

Different features have different scales. Row2Vec handles this automatically:

```python
# Features with different scales
data = pd.DataFrame({
    'age': [25, 35, 45],          # Scale: 20-50
    'income': [50000, 75000, 90000],  # Scale: 50k-100k
    'score': [0.8, 0.9, 0.7]      # Scale: 0-1
})

# Automatically scaled during preprocessing
embeddings = learn_embedding(data, mode="pca", embedding_dim=2)
```

**Scaling Methods:**
- **Standard scaling** (default): Zero mean, unit variance
- **MinMax scaling**: Scale to [0, 1] range
- **Robust scaling**: Uses median and IQR (robust to outliers)

### Categorical Encoding

Row2Vec handles categorical features automatically:

```python
# Mixed data types
data = pd.DataFrame({
    'category': ['A', 'B', 'C', 'A', 'B'],
    'numeric': [1.5, 2.3, 1.8, 2.1, 1.9],
    'ordinal': ['low', 'medium', 'high', 'medium', 'low']
})

# Automatically encoded
embeddings = learn_embedding(data, mode="unsupervised", embedding_dim=3)
```

**Encoding Strategies:**
- **Entity embeddings**: Learn representations (default for neural)
- **One-hot encoding**: Binary indicators (default for PCA)
- **Target encoding**: Use target variable correlation (supervised)
- **Frequency encoding**: Use category frequencies

## Neural Network Embeddings

### Basic Usage

The simplest way to generate neural embeddings:

```python
from row2vec import learn_embedding
import pandas as pd

# Your data
df = pd.read_csv('data.csv')

# Generate embeddings
embeddings = learn_embedding(
    df,
    mode="unsupervised",
    embedding_dim=10,
    max_epochs=100,
    verbose=True
)

print(f"Embeddings shape: {embeddings.shape}")  # (n_rows, 10)
```

### Network Architecture

**Default Architecture:**
```
Input (p features) → Dense(128) → ReLU →
  Dense(64) → ReLU →
  Dense(embedding_dim) → Bottleneck →
  Dense(64) → ReLU →
  Dense(128) → ReLU →
  Dense(p) → Output
```

**Custom Architecture:**

```python
from row2vec import EmbeddingConfig, NeuralConfig

config = EmbeddingConfig(
    mode="unsupervised",
    embedding_dim=8,
    neural=NeuralConfig(
        hidden_units=[256, 128, 64],  # Encoder layers
        dropout_rate=0.2,
        learning_rate=0.001,
        batch_size=32,
        max_epochs=100
    )
)

embeddings = learn_embedding_v2(df, config)
```

### Training Parameters

**Key Hyperparameters:**

- `embedding_dim`: Output dimension (5-100)
  - Smaller: More compression, may lose information
  - Larger: Preserves more information, may overfit

- `hidden_units`: Encoder layer sizes
  - Gradually decreasing (e.g., [128, 64, 32])
  - Symmetric decoder (mirrored)

- `max_epochs`: Training iterations (50-500)
  - Too few: Underfitting
  - Too many: Overfitting
  - Use early stopping

- `batch_size`: Training batch size (16-128)
  - Smaller: More noise, better generalization
  - Larger: Faster training, smoother convergence

- `learning_rate`: Optimizer step size (0.0001-0.01)
  - Too small: Slow convergence
  - Too large: Unstable training

- `dropout_rate`: Regularization (0.0-0.5)
  - 0.0: No regularization
  - 0.2-0.3: Good default
  - >0.5: May underfit

### Neural Architecture Search

Automatically discover optimal architectures:

```python
from row2vec import (
    search_architecture,
    ArchitectureSearchConfig,
    EmbeddingConfig,
    NeuralConfig
)

# Define search space
search_config = ArchitectureSearchConfig(
    method='random',  # or 'grid'
    max_layers=4,
    width_options=[64, 128, 256, 512],
    dropout_options=[0.0, 0.1, 0.2, 0.3],
    max_trials=50
)

# Base configuration
base_config = EmbeddingConfig(
    mode="unsupervised",
    embedding_dim=10,
    neural=NeuralConfig(max_epochs=30)
)

# Search
best_arch, all_results = search_architecture(df, base_config, search_config)

print("Best architecture found:")
print(f"  Hidden units: {best_arch['hidden_units']}")
print(f"  Dropout: {best_arch['dropout_rate']}")
print(f"  Final loss: {best_arch['final_loss']:.4f}")
```

**Search Strategies:**

- **Random search**: Sample architectures randomly (faster)
- **Grid search**: Try all combinations (exhaustive)

**Search Space:**

- `max_layers`: Maximum encoder depth
- `width_options`: Possible layer sizes
- `dropout_options`: Possible dropout rates
- `max_trials`: Number of architectures to try

### Automatic Dimension Selection

Let Row2Vec choose the embedding dimension:

```python
from row2vec import auto_select_dimension, AutoDimensionSelector

selector = AutoDimensionSelector(
    min_dim=2,
    max_dim=20,
    method='reconstruction_error'  # or 'explained_variance'
)

# Analyze optimal dimension
suggested_dim = selector.select(df)
print(f"Suggested dimension: {suggested_dim}")

# Train with suggested dimension
embeddings = learn_embedding(df, mode="unsupervised", embedding_dim=suggested_dim)
```

**Selection Methods:**

- `reconstruction_error`: Minimize reconstruction loss
- `explained_variance`: PCA-based variance threshold
- `elbow`: Elbow method on reconstruction curve

## Classical Methods

### PCA (Principal Component Analysis)

Fast, linear dimensionality reduction:

```python
from row2vec import learn_embedding

# PCA embeddings
pca_embeddings = learn_embedding(
    df,
    mode="pca",
    embedding_dim=10,
    verbose=False
)

# Variance explained
from row2vec import learn_embedding_with_model

embeddings, model, preprocessor, metadata = learn_embedding_with_model(
    df,
    mode="pca",
    embedding_dim=10
)

print(f"Variance explained: {metadata.get('explained_variance_ratio', [])}")
```

**Advantages:**
- Very fast computation
- Deterministic results
- Interpretable components
- Works with any dimension

**Limitations:**
- Only captures linear relationships
- Sensitive to outliers
- Assumes Gaussian distribution

**When to Use:**
- Initial data exploration
- Linear relationships dominant
- Need interpretability
- Baseline for comparison

### t-SNE (t-Distributed Stochastic Neighbor Embedding)

Excellent for 2D/3D visualization:

```python
# t-SNE for visualization
tsne_2d = learn_embedding(
    df,
    mode="tsne",
    embedding_dim=2,
    perplexity=30,
    verbose=False
)

# Visualize
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 8))
plt.scatter(tsne_2d.iloc[:, 0], tsne_2d.iloc[:, 1], alpha=0.6)
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.title('t-SNE Visualization')
plt.show()
```

**Key Parameter: Perplexity**

- `perplexity`: Balances local vs global structure (5-50)
  - Low (5-15): Emphasizes local structure
  - Medium (15-30): Balanced (default: 30)
  - High (30-50): Emphasizes global structure

**Advantages:**
- Excellent cluster visualization
- Reveals hidden structure
- Preserves local neighborhoods

**Limitations:**
- Slow for large datasets (>10,000 rows)
- Stochastic (different runs differ)
- Not suitable for >3 dimensions
- Doesn't preserve distances well

**When to Use:**
- Visualizing clusters
- Exploring data structure
- Presentations and reports
- Validating clustering

### UMAP (Uniform Manifold Approximation and Projection)

General-purpose non-linear embedding:

```python
# UMAP embeddings
umap_embeddings = learn_embedding(
    df,
    mode="umap",
    embedding_dim=10,
    n_neighbors=15,
    min_dist=0.1,
    verbose=False
)
```

**Key Parameters:**

- `n_neighbors` (5-50): Local neighborhood size
  - Low: Local structure
  - High: Global structure
  - Default: 15

- `min_dist` (0.0-1.0): Minimum distance between points
  - Low (0.0-0.1): Tight clusters
  - High (0.5-1.0): Spread out
  - Default: 0.1

**Advantages:**
- Faster than t-SNE
- Better global structure
- Supports any dimension
- More consistent results

**Limitations:**
- Less established than PCA/t-SNE
- Some parameters hard to tune
- Requires umap-learn package

**When to Use:**
- General-purpose embeddings
- Need both local and global structure
- Larger datasets (>10,000 rows)
- Higher dimensions (>3)

## Advanced Features

### Contrastive Learning

Learn embeddings that bring similar samples closer and push dissimilar samples apart:

```python
from row2vec import learn_embedding_contrastive

# Contrastive learning with labels
embeddings = learn_embedding_contrastive(
    df,
    embedding_dim=10,
    temperature=0.5,  # Contrastive temperature
    max_epochs=100
)
```

**Use Cases:**
- Metric learning
- Similarity search
- Few-shot learning
- When you have similarity annotations

### Target-Based Embeddings

Learn embeddings for categorical features based on their relationship with other columns:

```python
# Embed each unique country based on associated features
country_embeddings = learn_embedding(
    df,
    mode="target",
    reference_column="Country",
    embedding_dim=5,
    max_epochs=50
)

print(f"Embeddings per country: {country_embeddings.shape}")
```

**Applications:**
- Entity embeddings
- Categorical feature engineering
- Recommendation systems
- Transfer learning

### Configuration-Based Workflow

For complex pipelines, use configuration objects:

```python
from row2vec import (
    EmbeddingConfig,
    NeuralConfig,
    PreprocessingConfig,
    ScalingConfig,
    ImputationConfig
)

# Comprehensive configuration
config = EmbeddingConfig(
    mode="unsupervised",
    embedding_dim=8,
    neural=NeuralConfig(
        hidden_units=[256, 128, 64],
        max_epochs=100,
        batch_size=32,
        dropout_rate=0.2,
        learning_rate=0.001,
        verbose=True
    ),
    preprocessing=PreprocessingConfig(
        scaling=ScalingConfig(
            method='standard',
            feature_range=(0, 1)
        ),
        imputation=ImputationConfig(
            numeric_strategy='knn',
            categorical_strategy='mode',
            knn_neighbors=5
        )
    )
)

# Train with full configuration
from row2vec import learn_embedding_v2
embeddings = learn_embedding_v2(df, config)
```

## Model Serialization

### Save and Load Models

Train once, use many times:

```python
from row2vec import train_and_save_model, load_model

# Train and save
embeddings, script_path, binary_path = train_and_save_model(
    df,
    base_path="my_model",
    embedding_dim=10,
    mode="unsupervised",
    max_epochs=100
)

print(f"Model saved to: {script_path}")

# Load and reuse
model = load_model(script_path)

# Apply to new data
new_embeddings = model.predict(new_df)
```

**Saved Components:**
- Trained neural network weights
- Preprocessing pipeline (scaling, encoding)
- Imputation strategies
- Model metadata

**Use Cases:**
- Production deployment
- Consistent preprocessing
- Transfer learning
- Batch processing

### Model Versioning

```python
import datetime

# Version with timestamp
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
model_path = f"models/row2vec_{timestamp}"

train_and_save_model(df, base_path=model_path, embedding_dim=10)
```

## Best Practices

### Data Quality

**Minimum Requirements:**
- At least 100 rows (preferably 1000+)
- At least 3 features (preferably 5+)
- Representative sampling
- Reasonable class balance (for target mode)

**Data Cleaning:**
```python
# Check data quality
print(f"Shape: {df.shape}")
print(f"Missing values:\n{df.isnull().sum()}")
print(f"Duplicates: {df.duplicated().sum()}")

# Clean data
df_clean = df.drop_duplicates()
df_clean = df_clean.dropna(thresh=len(df_clean.columns) * 0.5)  # Keep rows with <50% missing
```

### Choosing Embedding Dimension

**Rules of Thumb:**

- **Visualization**: 2-3 dimensions
- **Feature engineering**: 5-20 dimensions
- **Complex patterns**: 10-100 dimensions
- **Rule of thumb**: $d \approx \sqrt{p}$ where $p$ is input features

**Empirical Selection:**

```python
# Try multiple dimensions
dimensions = [2, 5, 10, 20, 50]
results = {}

for dim in dimensions:
    embeddings = learn_embedding(df, mode="unsupervised", embedding_dim=dim)
    # Evaluate downstream task performance
    score = evaluate_embeddings(embeddings, labels)
    results[dim] = score

best_dim = max(results, key=results.get)
print(f"Best dimension: {best_dim}")
```

### Hyperparameter Tuning

**Grid Search for Neural Networks:**

```python
from itertools import product

# Define grid
hidden_units_options = [[128, 64], [256, 128, 64], [512, 256, 128]]
dropout_options = [0.0, 0.2, 0.3]
learning_rate_options = [0.001, 0.0001]

best_loss = float('inf')
best_params = None

for hidden, dropout, lr in product(hidden_units_options, dropout_options, learning_rate_options):
    config = EmbeddingConfig(
        mode="unsupervised",
        embedding_dim=10,
        neural=NeuralConfig(
            hidden_units=hidden,
            dropout_rate=dropout,
            learning_rate=lr,
            max_epochs=30
        )
    )

    embeddings, model, _, metadata = learn_embedding_with_model_v2(df, config)
    loss = metadata.get('final_loss', float('inf'))

    if loss < best_loss:
        best_loss = loss
        best_params = (hidden, dropout, lr)

print(f"Best parameters: {best_params}")
print(f"Best loss: {best_loss:.4f}")
```

### Validation Strategy

**Hold-Out Validation:**

```python
from sklearn.model_selection import train_test_split

# Split data
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# Train on training set
embeddings_train, model, _, _ = learn_embedding_with_model(
    train_df,
    mode="unsupervised",
    embedding_dim=10
)

# Validate on validation set
embeddings_val = model.predict(val_df)

# Compute reconstruction error or downstream task performance
```

### Performance Optimization

**For Large Datasets (>100,000 rows):**

```python
# Use batch processing
config = EmbeddingConfig(
    mode="unsupervised",
    embedding_dim=10,
    neural=NeuralConfig(
        batch_size=128,  # Increase batch size
        max_epochs=50    # May need fewer epochs
    )
)
```

**For Many Features (>100):**

```python
# Use PCA for initial reduction
from sklearn.decomposition import PCA

pca = PCA(n_components=50)
df_reduced = pca.fit_transform(df)

# Then apply Row2Vec
embeddings = learn_embedding(
    pd.DataFrame(df_reduced),
    mode="unsupervised",
    embedding_dim=10
)
```

## Common Use Cases

### Clustering

Generate embeddings, then cluster:

```python
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Generate 2D embeddings for visualization
embeddings = learn_embedding(df, mode="tsne", embedding_dim=2)

# Cluster
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(embeddings)

# Visualize
plt.figure(figsize=(10, 8))
scatter = plt.scatter(
    embeddings.iloc[:, 0],
    embeddings.iloc[:, 1],
    c=clusters,
    cmap='viridis',
    alpha=0.6
)
plt.colorbar(scatter, label='Cluster')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.title('Clustering on Row2Vec Embeddings')
plt.show()
```

### Classification

Use embeddings as features:

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Generate embeddings
embeddings = learn_embedding(df, mode="unsupervised", embedding_dim=10)

# Add target
X = embeddings
y = labels  # Your classification target

# Split and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)

score = clf.score(X_test, y_test)
print(f"Accuracy: {score:.3f}")
```

### Anomaly Detection

Identify outliers using embeddings:

```python
from sklearn.covariance import EllipticEnvelope

# Generate embeddings
embeddings = learn_embedding(df, mode="unsupervised", embedding_dim=10)

# Detect anomalies
detector = EllipticEnvelope(contamination=0.1)  # 10% anomalies
predictions = detector.fit_predict(embeddings)

# Anomalies are marked as -1
anomalies = df[predictions == -1]
print(f"Detected {len(anomalies)} anomalies")
```

### Data Visualization

Create interactive visualizations:

```python
import plotly.express as px

# 2D embeddings
embeddings_2d = learn_embedding(df, mode="umap", embedding_dim=2)

# Interactive plot
fig = px.scatter(
    x=embeddings_2d.iloc[:, 0],
    y=embeddings_2d.iloc[:, 1],
    hover_data=df.columns.tolist(),
    title='Interactive UMAP Projection'
)
fig.show()
```

### Similarity Search

Find similar rows:

```python
from sklearn.metrics.pairwise import cosine_similarity

# Generate embeddings
embeddings = learn_embedding(df, mode="unsupervised", embedding_dim=10)

# Find similar to first row
query_embedding = embeddings.iloc[0:1]
similarities = cosine_similarity(query_embedding, embeddings)[0]

# Top 5 most similar
top_indices = similarities.argsort()[-5:][::-1]
print("Most similar rows:")
print(df.iloc[top_indices])
```

### Transfer Learning

Train on one dataset, apply to related dataset:

```python
# Train on large source dataset
embeddings_source, model, _, _ = learn_embedding_with_model(
    source_df,
    mode="unsupervised",
    embedding_dim=10,
    max_epochs=100
)

# Apply to target dataset (smaller, related domain)
embeddings_target = model.predict(target_df)
```

---

**For executable examples with code, see the tutorials:**
- tutorial_1_quickstart.html - Basic usage and method comparison
- tutorial_2_advanced.html - Advanced features and architecture search

**For API reference, see API_REFERENCE.md**

**For LLM integration, see LLM_DOCUMENTATION.md**
