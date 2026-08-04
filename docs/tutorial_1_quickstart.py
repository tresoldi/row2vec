#' # Quickstart Guide
#'
#' Get started with Row2Vec in 5 minutes! This guide shows the essential features
#' through executable examples.
#'
#' ## Installation
#'
#' ```bash
#' pip install row2vec
#' ```
#'
#' ## Basic Usage
#'
#' The core of Row2Vec is the `learn_embedding()` function:

# | hide
import warnings

warnings.filterwarnings("ignore")
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# |


from row2vec import generate_synthetic_data, learn_embedding

# Generate sample data
df = generate_synthetic_data(num_records=200, seed=42)
print(f"Dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

#' ## Unsupervised Embeddings
#'
#' Create compressed representations of each row:

# Learn 5-dimensional embeddings for each row
embeddings = learn_embedding(df, mode="unsupervised", embedding_dim=5, max_epochs=20, verbose=False)

print(f"Embeddings shape: {embeddings.shape}")
print("\nFirst 3 embeddings:")
print(embeddings.head(3))

#' Verify: each row gets an embedding

print(f"Original data: {len(df)} rows")
print(f"Embeddings: {len(embeddings)} rows")
print(f"Dimensions per embedding: {embeddings.shape[1]}")

#' ## Target-Based Embeddings
#'
#' Learn embeddings for categorical column values:

# Learn embeddings for each country
country_embeddings = learn_embedding(
    df,
    mode="target",
    reference_column="Country",
    embedding_dim=3,
    max_epochs=20,
    verbose=False,
)

print("Country embeddings:")
print(country_embeddings)

#' ## Classical Methods
#'
#' Row2Vec also provides classical dimensionality reduction methods.
#'
#' ### PCA (Fast Linear Reduction)

pca_embeddings = learn_embedding(df, mode="pca", embedding_dim=2, verbose=False)

print("PCA embeddings (first 5):")
print(pca_embeddings.head())

#' ### t-SNE (Visualization)

tsne_embeddings = learn_embedding(df, mode="tsne", embedding_dim=2, perplexity=30, verbose=False)

print("t-SNE embeddings (first 5):")
print(tsne_embeddings.head())

#' ### UMAP (Balanced Approach)

try:
    umap_embeddings = learn_embedding(
        df, mode="umap", embedding_dim=2, n_neighbors=15, verbose=False
    )
    print("UMAP embeddings (first 5):")
    print(umap_embeddings.head())
except ImportError:
    print("UMAP not installed. Install with: pip install umap-learn")

#' ## Handling Missing Values
#'
#' Row2Vec automatically handles missing values:

import numpy as np

# Create data with missing values
df_missing = df.copy()
df_missing.loc[0:5, "Sales"] = np.nan
df_missing.loc[10:15, "Product"] = np.nan

print(f"Missing values introduced: {df_missing.isnull().sum().sum()}")

# Row2Vec handles this automatically
embeddings_missing = learn_embedding(
    df_missing, mode="unsupervised", embedding_dim=3, max_epochs=10, verbose=False
)

print(f"\nEmbeddings generated: {embeddings_missing.shape}")
print("No errors - missing values handled automatically!")

#' ## Method Comparison
#'
#' Let's compare different methods on the same data:

import time

methods = {
    "Neural": {"mode": "unsupervised", "max_epochs": 10},
    "PCA": {"mode": "pca"},
    "t-SNE": {"mode": "tsne", "perplexity": 30},
}

results = {}
for name, params in methods.items():
    start = time.time()
    emb = learn_embedding(df, embedding_dim=2, verbose=False, **params)
    elapsed = time.time() - start
    results[name] = {
        "time": elapsed,
        "shape": emb.shape,
        "mean": emb.mean().mean(),
        "std": emb.std().mean(),
    }

print("Method Comparison:")
print("-" * 50)
for method, stats in results.items():
    print(
        f"{method:10} | Time: {stats['time']:.2f}s | Mean: {stats['mean']:6.3f} | Std: {stats['std']:5.3f}"
    )

#' ## Method Selection Guide
#'
#' Each embedding method has different strengths:
#'
#' | Method | Speed | Best For |
#' |--------|-------|----------|
#' | **Neural** | Medium | Complex patterns, feature engineering |
#' | **PCA** | Fast | Quick dimensionality reduction, linear relationships |
#' | **t-SNE** | Slow | 2D/3D visualization, cluster discovery |
#' | **UMAP** | Fast | General purpose, balanced local/global structure |
#'
#' ### When to Use Each Method
#'
#' **Choose Neural Networks (`mode="unsupervised"`) when:**
#' - You need embeddings for downstream machine learning models
#' - Your data has complex, non-linear relationships
#' - You want features that can capture intricate patterns
#'
#' **Choose PCA (`mode="pca"`) when:**
#' - You need fast, deterministic results
#' - Your data relationships are primarily linear
#' - You want interpretable principal components
#'
#' **Choose t-SNE (`mode="tsne"`) when:**
#' - You want to visualize data in 2D or 3D
#' - Discovering clusters is your primary goal
#' - Local neighborhood preservation is most important
#'
#' **Choose UMAP (`mode="umap"`) when:**
#' - You want general-purpose dimensionality reduction
#' - You need both local and global structure preserved
#' - You're working with higher-dimensional outputs (>3D)
#'
#' ## Next Steps
#'
#' - Try the Advanced Features tutorial for neural architecture search
#' - Explore the API Reference for complete documentation
#' - Check out real-world examples with actual datasets
