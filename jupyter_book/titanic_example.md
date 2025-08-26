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

# Titanic Dataset Example

A complete walkthrough using the famous Titanic dataset to demonstrate Row2Vec's capabilities.

## Load the Data

```{code-cell} python
:tags: [remove-stderr, remove-warnings]

# Import complete suppression first
exec(open('suppress_minimal.py').read())

import pandas as pd
import numpy as np
from row2vec import learn_embedding
import os

# Load Titanic dataset
data_path = os.path.join('..', 'data', 'titanic.csv')
df = pd.read_csv(data_path)

print(f"Dataset shape: {df.shape}")
print(f"\nColumns: {df.columns.tolist()}")
print(f"\nMissing values per column:")
print(df.isnull().sum())
```

## Data Preview

```{code-cell} python
:tags: [remove-stderr, remove-warnings]
print("First 5 passengers:")
print(df.head())
```

```{code-cell} python
:tags: [remove-stderr, remove-warnings]
# Basic statistics
print("Survival rate:", df['Survived'].mean())
print("Average age:", df['Age'].mean())
print("Average fare:", df['Fare'].mean())
print("\nPassenger classes:")
print(df['Pclass'].value_counts().sort_index())
```

## Data Preparation

```{code-cell} python
:tags: [remove-stderr, remove-warnings]
# For unsupervised learning, we'll drop the target and ID columns
df_features = df.drop(columns=['Survived', 'Name'])

print(f"Feature columns: {df_features.columns.tolist()}")
print(f"Shape for embedding: {df_features.shape}")

# Row2Vec handles missing values automatically, but let's see what we're dealing with
missing_counts = df_features.isnull().sum()
print(f"\nColumns with missing values:")
for col, count in missing_counts[missing_counts > 0].items():
    print(f"  {col}: {count} missing ({count/len(df_features)*100:.1f}%)")
```

## Unsupervised Row Embeddings

Create a 2D embedding for each passenger:

```{code-cell} python
:tags: [remove-stderr, remove-warnings]

# Generate 2D embeddings for visualization
embeddings_2d = learn_embedding(
    df_features,
    mode="unsupervised",
    embedding_dim=2,
    max_epochs=50,
    batch_size=32,
    dropout_rate=0.2,
    hidden_units=128,
    verbose=False,
    seed=42
)

print(f"Embeddings shape: {embeddings_2d.shape}")
print("\nFirst 5 passenger embeddings:")
print(embeddings_2d.head())
```

## Visualize the Embeddings

```{code-cell} python
:tags: [remove-stderr, remove-warnings]
import matplotlib.pyplot as plt

# Add survival information for coloring
embeddings_with_survival = embeddings_2d.copy()
embeddings_with_survival['Survived'] = df['Survived'].values

# Create visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Colored by survival
scatter1 = axes[0].scatter(
    embeddings_with_survival['embedding_0'],
    embeddings_with_survival['embedding_1'],
    c=embeddings_with_survival['Survived'],
    cmap='coolwarm',
    alpha=0.6,
    s=20
)
axes[0].set_xlabel('Embedding Dimension 0')
axes[0].set_ylabel('Embedding Dimension 1')
axes[0].set_title('Passenger Embeddings Colored by Survival')
plt.colorbar(scatter1, ax=axes[0], label='Survived')

# Plot 2: Colored by passenger class
embeddings_with_class = embeddings_2d.copy()
embeddings_with_class['Pclass'] = df['Pclass'].values

scatter2 = axes[1].scatter(
    embeddings_with_class['embedding_0'],
    embeddings_with_class['embedding_1'],
    c=embeddings_with_class['Pclass'],
    cmap='viridis',
    alpha=0.6,
    s=20
)
axes[1].set_xlabel('Embedding Dimension 0')
axes[1].set_ylabel('Embedding Dimension 1')
axes[1].set_title('Passenger Embeddings Colored by Class')
plt.colorbar(scatter2, ax=axes[1], label='Passenger Class')

plt.tight_layout()
plt.show()

print("Notice how passengers with similar survival outcomes and classes cluster together!")
```

## Higher-Dimensional Embeddings

For machine learning, we typically want more dimensions:

```{code-cell} python
:tags: [remove-stderr, remove-warnings]

# Generate 5D embeddings for ML features (reduced from 10D due to feature count)
embeddings_5d = learn_embedding(
    df_features,
    mode="unsupervised",
    embedding_dim=5,
    max_epochs=50,
    batch_size=64,
    dropout_rate=0.25,
    hidden_units=256,
    verbose=False,
    seed=42
)

print(f"5D Embeddings shape: {embeddings_5d.shape}")
print("\nStatistics per dimension:")
print(embeddings_5d.describe().round(3))
```

## Target-Based Embeddings

Learn embeddings for categorical columns:

### Sex Embeddings

```{code-cell} python
:tags: [remove-stderr, remove-warnings]

# Learn embeddings for Sex categories
sex_embeddings = learn_embedding(
    df,
    mode="target",
    reference_column="Sex",
    embedding_dim=2,
    max_epochs=30,
    verbose=False,
    seed=42
)

print("Sex category embeddings:")
print(sex_embeddings)

# Calculate distance between the two sex categories
# Use iloc to access by position since index might be numerical
if len(sex_embeddings) >= 2:
    emb_1 = sex_embeddings.iloc[0].values
    emb_2 = sex_embeddings.iloc[1].values
    distance = np.linalg.norm(emb_1 - emb_2)
    print(f"\nEuclidean distance between sex categories: {distance:.3f}")
else:
    print("\nInsufficient categories for distance calculation")
```

### Passenger Class Embeddings

```{code-cell} python
:tags: [remove-stderr, remove-warnings]

# Learn embeddings for passenger classes
pclass_embeddings = learn_embedding(
    df,
    mode="target",
    reference_column="Pclass",
    embedding_dim=3,
    max_epochs=30,
    verbose=False,
    seed=42
)

print("Passenger class embeddings:")
print(pclass_embeddings)

# Analyze relationships between classes
import itertools

print("\nPairwise distances between classes:")
# Use iloc to access by position since indices might be different
if len(pclass_embeddings) >= 2:
    for i, j in itertools.combinations(range(len(pclass_embeddings)), 2):
        dist = np.linalg.norm(
            pclass_embeddings.iloc[i].values -
            pclass_embeddings.iloc[j].values
        )
        print(f"  Category {i} <-> Category {j}: {dist:.3f}")
else:
    print("  Insufficient categories for distance calculation")
```

## Compare with Classical Methods

Let's see how neural embeddings compare to classical methods:

```{code-cell} python
:tags: [remove-stderr, remove-warnings]

# PCA
pca_embeddings = learn_embedding(
    df_features,
    mode="pca",
    embedding_dim=2,
    verbose=False
)

# t-SNE
tsne_embeddings = learn_embedding(
    df_features,
    mode="tsne",
    embedding_dim=2,
    perplexity=30,
    verbose=False,
    seed=42
)

print("Method comparison (2D embeddings):")
print("-" * 50)
print(f"Neural: mean={embeddings_2d.mean().mean():.3f}, std={embeddings_2d.std().mean():.3f}")
print(f"PCA:    mean={pca_embeddings.mean().mean():.3f}, std={pca_embeddings.std().mean():.3f}")
print(f"t-SNE:  mean={tsne_embeddings.mean().mean():.3f}, std={tsne_embeddings.std().mean():.3f}")
```

## Visualize Method Comparison

```{code-cell} python
:tags: [remove-stderr, remove-warnings]
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

methods = [
    ("Neural (Autoencoder)", embeddings_2d),
    ("PCA", pca_embeddings),
    ("t-SNE", tsne_embeddings)
]

for ax, (name, emb) in zip(axes, methods):
    scatter = ax.scatter(
        emb.iloc[:, 0],
        emb.iloc[:, 1],
        c=df['Survived'].values,
        cmap='coolwarm',
        alpha=0.6,
        s=20
    )
    ax.set_xlabel('Dimension 0')
    ax.set_ylabel('Dimension 1')
    ax.set_title(f'{name} Embeddings')

plt.colorbar(scatter, ax=axes, label='Survived', fraction=0.02)
plt.tight_layout()
plt.show()

print("Each method reveals different aspects of the data structure!")
```

## Use Embeddings as Features

Demonstrate using embeddings for downstream ML:

```{code-cell} python
:tags: [remove-stderr, remove-warnings]

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Prepare data
X = embeddings_5d
y = df['Survived']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train classifier on embeddings
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Survival prediction using 5D embeddings:")
print(f"Accuracy: {accuracy:.3f}")
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Died', 'Survived']))
```

## Save Model for Production

```{code-cell} python
:tags: [remove-stderr, remove-warnings]

from row2vec import train_and_save_model
import tempfile
import os

# Create a production-ready model
with tempfile.TemporaryDirectory() as tmpdir:
    model_path = os.path.join(tmpdir, "titanic_model")

    embeddings_final, script_path, binary_path = train_and_save_model(
        df_features,
        base_path=model_path,
        embedding_dim=5,
        mode="unsupervised",
        max_epochs=50,
        batch_size=64,
        dropout_rate=0.25,
        hidden_units=256,
        verbose=False,
        seed=42
    )

    print(f"Model saved to: {os.path.basename(script_path)}")

    # Show model can be loaded and used
    from row2vec import load_model

    model = load_model(script_path)
    print(f"\nModel metadata:")
    print(f"  Mode: {model.metadata.mode}")
    print(f"  Embedding dimensions: {model.metadata.embedding_dim}")
    print(f"  Training epochs: {model.metadata.epochs_trained}")

    # Handle case where final_loss might be None
    if model.metadata.final_loss is not None:
        print(f"  Final loss: {model.metadata.final_loss:.4f}")
    else:
        print(f"  Final loss: Not recorded")
```

## Key Takeaways

1. **Automatic Preprocessing**: Row2Vec handled missing Age values automatically
2. **Multiple Modes**: Neural, PCA, and t-SNE each reveal different patterns
3. **Visualization**: 2D embeddings are great for understanding data structure
4. **ML Features**: Higher-dimensional embeddings work well as ML features
5. **Target Embeddings**: Learn meaningful representations for categorical values
6. **Production Ready**: Models can be saved and deployed easily

## Next Steps

- Try the [Adult Dataset Example](adult_example.md) for high-cardinality categoricals
- Explore [Advanced Features](advanced_features.md) like neural architecture search
- Learn about the [CLI](cli_guide.md) for batch processing
