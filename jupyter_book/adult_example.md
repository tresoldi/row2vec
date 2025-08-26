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

# Adult Dataset Example

The Adult dataset demonstrates Row2Vec's capabilities with high-cardinality categorical features and larger datasets.

## Load and Explore Data

```{code-cell} python
:tags: [remove-stderr, remove-warnings]

# Import complete suppression first
exec(open('suppress_minimal.py').read())

import pandas as pd
import numpy as np
from row2vec import learn_embedding
import os

# Load Adult dataset
data_path = os.path.join('..', 'data', 'adult.csv')
df = pd.read_csv(data_path)

print(f"Dataset shape: {df.shape}")
print(f"\nColumns: {df.columns.tolist()}")
```

```{code-cell} python
:tags: [remove-stderr, remove-warnings]
# Preview the data
print("First 5 records:")
print(df.head())
```

```{code-cell} python
:tags: [remove-stderr, remove-warnings]
# Check for high-cardinality categoricals
print("Categorical column cardinalities:")
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    unique_count = df[col].nunique()
    print(f"  {col}: {unique_count} unique values")
    if unique_count <= 10:  # Show values for low-cardinality columns
        print(f"    Values: {df[col].unique()[:10].tolist()}")
```

## Data Preprocessing

```{code-cell} python
:tags: [remove-stderr, remove-warnings]
# Remove columns not useful for general embeddings
cols_to_drop = ['fnlwgt', 'education-num', 'income']  # fnlwgt is just a weight, education-num duplicates education
df_features = df.drop(columns=cols_to_drop)

print(f"Features for embedding: {df_features.columns.tolist()}")
print(f"Shape: {df_features.shape}")

# Check missing values
missing = df_features.isnull().sum()
if missing.any():
    print(f"\nMissing values:")
    print(missing[missing > 0])
else:
    print("\nNo missing values detected")
```

## Unsupervised Embeddings

Generate embeddings for the entire dataset:

```{code-cell} python
:tags: [remove-stderr, remove-warnings]

# Sample for faster demo (remove sampling for full dataset)
df_sample = df_features.sample(n=5000, random_state=42)
print(f"Working with sample of {len(df_sample)} records")

# Generate 10D embeddings
embeddings = learn_embedding(
    df_sample,
    mode="unsupervised",
    embedding_dim=10,
    max_epochs=30,
    batch_size=128,
    dropout_rate=0.3,
    hidden_units=256,
    verbose=False,
    seed=42
)

print(f"\nEmbeddings shape: {embeddings.shape}")
print("\nEmbedding statistics:")
print(embeddings.describe().round(3))
```

## High-Cardinality Categorical Embeddings

The Adult dataset's 'occupation' column has many categories - perfect for target-based embeddings:

```{code-cell} python
:tags: [remove-stderr, remove-warnings]

# Get occupation counts
occupation_counts = df['occupation'].value_counts()
print(f"Occupation column: {len(occupation_counts)} unique values")
print("\nTop 10 occupations:")
print(occupation_counts.head(10))
```

```{code-cell} python
:tags: [remove-stderr, remove-warnings]

# Learn embeddings for occupations
# df_sample already contains occupation column, so we can use it directly
occupation_embeddings = learn_embedding(
    df_sample,
    mode="target",
    reference_column="occupation",
    embedding_dim=3,
    max_epochs=40,
    batch_size=128,
    verbose=False,
    seed=42
)

print(f"Occupation embeddings shape: {occupation_embeddings.shape}")
print("\nOccupation embeddings (3D):")
print(occupation_embeddings.round(3))
```

## Analyze Occupation Relationships

```{code-cell} python
:tags: [remove-stderr, remove-warnings]
# Verify occupation_embeddings exists
if 'occupation_embeddings' not in locals():
    print("ERROR: occupation_embeddings not found, recreating...")
    occupation_embeddings = learn_embedding(
        df_sample,
        mode="target",
        reference_column="occupation",
        embedding_dim=3,
        max_epochs=40,
        batch_size=128,
        verbose=False,
        seed=42
    )

import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

# Calculate similarity matrix
similarity_matrix = cosine_similarity(occupation_embeddings.values)
occupations = occupation_embeddings.index.tolist()

print("Most similar occupation pairs:")
print("-" * 40)

# Find most similar pairs
for i in range(len(occupations)):
    for j in range(i+1, len(occupations)):
        similarity = similarity_matrix[i][j]
        if similarity > 0.8:  # High similarity threshold
            print(f"{occupations[i]} <-> {occupations[j]}: {similarity:.3f}")
```

```{code-cell} python
:tags: [remove-stderr, remove-warnings]
# Verify occupation_embeddings exists
if 'occupation_embeddings' not in locals():
    print("ERROR: occupation_embeddings not found, recreating...")
    occupation_embeddings = learn_embedding(
        df_sample,
        mode="target",
        reference_column="occupation",
        embedding_dim=3,
        max_epochs=40,
        batch_size=128,
        verbose=False,
        seed=42
    )
    occupations = occupation_embeddings.index.tolist()

# Visualize occupation embeddings in 2D (project from 3D to 2D)
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
occupation_2d = pca.fit_transform(occupation_embeddings.values)

plt.figure(figsize=(12, 8))
plt.scatter(occupation_2d[:, 0], occupation_2d[:, 1], s=100, alpha=0.7)

# Label each point
for i, occupation in enumerate(occupations):
    plt.annotate(
        occupation,
        (occupation_2d[i, 0], occupation_2d[i, 1]),
        xytext=(5, 5),
        textcoords='offset points',
        fontsize=8,
        alpha=0.8
    )

plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('Occupation Embeddings Visualization (2D Projection)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("Notice how similar occupations cluster together!")
```

## Work Class Embeddings

```{code-cell} python
:tags: [remove-stderr, remove-warnings]

# Learn embeddings for work class
# df_sample already contains workclass column, so we can use it directly
workclass_embeddings = learn_embedding(
    df_sample,
    mode="target",
    reference_column="workclass",
    embedding_dim=2,
    max_epochs=30,
    verbose=False,
    seed=42
)

print(f"Work class embeddings shape: {workclass_embeddings.shape}")
print("\nWork class embeddings:")
print(workclass_embeddings.round(3))
```

```{code-cell} python
:tags: [remove-stderr, remove-warnings]
# Verify workclass_embeddings exists
if 'workclass_embeddings' not in locals():
    print("ERROR: workclass_embeddings not found, recreating...")
    workclass_embeddings = learn_embedding(
        df_sample,
        mode="target",
        reference_column="workclass",
        embedding_dim=2,
        max_epochs=30,
        verbose=False,
        seed=42
    )

# Visualize work class embeddings
plt.figure(figsize=(10, 6))
plt.scatter(workclass_embeddings.iloc[:, 0], workclass_embeddings.iloc[:, 1], s=150, alpha=0.7)

for i, workclass in enumerate(workclass_embeddings.index):
    plt.annotate(
        workclass,
        (workclass_embeddings.iloc[i, 0], workclass_embeddings.iloc[i, 1]),
        xytext=(5, 5),
        textcoords='offset points',
        fontsize=10
    )

plt.xlabel('Embedding Dimension 0')
plt.ylabel('Embedding Dimension 1')
plt.title('Work Class Embeddings')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

## Compare Methods on Adult Data

```{code-cell} python
:tags: [remove-stderr, remove-warnings]

import time

# Compare different methods
methods = {
    "Neural": {"mode": "unsupervised", "max_epochs": 20},
    "PCA": {"mode": "pca"},
    "t-SNE": {"mode": "tsne", "perplexity": 50}
}

# Use smaller sample for t-SNE (it's slow)
small_sample = df_features.sample(n=1000, random_state=42)

results = {}
for name, params in methods.items():
    print(f"Running {name}...")
    start = time.time()

    emb = learn_embedding(
        small_sample,
        embedding_dim=2,
        verbose=False,
        seed=42,
        **params
    )

    elapsed = time.time() - start
    results[name] = {
        "time": elapsed,
        "shape": emb.shape,
        "mean": emb.mean().mean(),
        "std": emb.std().mean()
    }

print("\nMethod comparison results:")
print("-" * 60)
for method, stats in results.items():
    print(f"{method:8} | Time: {stats['time']:6.2f}s | Mean: {stats['mean']:7.3f} | Std: {stats['std']:6.3f}")
```

## Feature Engineering with Embeddings

Use embeddings as features for income prediction:

```{code-cell} python
:tags: [remove-stderr, remove-warnings]

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Prepare target variable
income_binary = (df.loc[df_sample.index, 'income'] == '>50K').astype(int)

print(f"Income distribution in sample:")
print(f"<=50K: {(income_binary == 0).sum()} ({(income_binary == 0).mean():.1%})")
print(f">50K:  {(income_binary == 1).sum()} ({(income_binary == 1).mean():.1%})")

# Use embeddings as features
X = embeddings
y = income_binary

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nIncome prediction using embeddings:")
print(f"Accuracy: {accuracy:.3f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['<=50K', '>50K']))
```

## Scaling for Different Ranges

```{code-cell} python
:tags: [remove-stderr, remove-warnings]

# Generate scaled embeddings for different use cases
scaled_embeddings = learn_embedding(
    df_sample,
    mode="unsupervised",
    embedding_dim=5,
    max_epochs=20,
    scale_method="minmax",
    scale_range=(-1.0, 1.0),
    verbose=False,
    seed=42
)

print("Scaled embeddings statistics:")
print(f"Min: {scaled_embeddings.min().min():.3f}")
print(f"Max: {scaled_embeddings.max().max():.3f}")
print(f"Mean: {scaled_embeddings.mean().mean():.3f}")
print(f"Std: {scaled_embeddings.std().mean():.3f}")

print("\nFirst 5 scaled embeddings:")
print(scaled_embeddings.head().round(3))
```

## Production Model

Create a model ready for deployment:

```{code-cell} python
:tags: [remove-stderr, remove-warnings]

from row2vec import train_and_save_model
import tempfile
import os

with tempfile.TemporaryDirectory() as tmpdir:
    model_path = os.path.join(tmpdir, "adult_model")

    # Train production model on larger dataset
    production_sample = df_features.sample(n=10000, random_state=42)

    embeddings_prod, script_path, binary_path = train_and_save_model(
        production_sample,
        base_path=model_path,
        embedding_dim=15,
        mode="unsupervised",
        max_epochs=50,
        batch_size=256,
        dropout_rate=0.25,
        hidden_units=512,
        scale_method="standard",
        verbose=False,
        seed=42
    )

    print(f"Production model saved: {os.path.basename(script_path)}")
    print(f"Final embeddings shape: {embeddings_prod.shape}")

    # Load and test
    from row2vec import load_model
    model = load_model(script_path)

    # Test on new data
    test_data = df_features.sample(n=100, random_state=999)
    test_embeddings = model.predict(test_data)

    print(f"\nModel successfully applied to new data:")
    print(f"Test embeddings shape: {test_embeddings.shape}")
    print(f"Model training time: {model.metadata.training_time:.2f} seconds")
```

## Key Insights

1. **High-Cardinality Handling**: Row2Vec excels with many categorical values
2. **Occupation Relationships**: Similar jobs cluster in embedding space
3. **Scalability**: Handles 45K+ records efficiently with sampling
4. **Feature Quality**: Embeddings achieve good accuracy on income prediction
5. **Production Ready**: Easy to save, load, and deploy models

## Next Steps

- Explore [Housing Example](housing_example.md) for regression features
- Learn about [Advanced Features](advanced_features.md) like neural architecture search
- Check out the [CLI Guide](cli_guide.md) for batch processing large datasets
