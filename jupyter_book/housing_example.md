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

# Boston Housing Dataset Example

The Boston Housing dataset demonstrates Row2Vec with continuous features and regression targets.

## Load and Explore Data

```{code-cell} python
:tags: [remove-stderr, remove-warnings]

# Import complete suppression first
exec(open('suppress_minimal.py').read())

import pandas as pd
import numpy as np
from row2vec import learn_embedding
import os

# Load Boston Housing dataset (originally mislabeled as Ames)
data_path = os.path.join('..', 'data', 'ames_housing.csv')

# Boston housing dataset column names
column_names = [
    'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS',
    'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV'
]

df = pd.read_csv(data_path, header=None, names=column_names)

print(f"Dataset shape: {df.shape}")
print(f"\nColumn names: {df.columns.tolist()}")
print(f"Total columns: {len(df.columns)}")
```

## Data Overview

```{code-cell} python
:tags: [remove-stderr, remove-warnings]
# Focus on a manageable subset of important features
important_cols = [
    'CRIM', 'ZN', 'INDUS', 'NOX', 'RM', 'AGE', 'DIS',
    'RAD', 'TAX', 'PTRATIO', 'LSTAT', 'MEDV'
]

df_subset = df[important_cols].copy()
print(f"Working with {len(important_cols)} key features:")
print(df_subset.columns.tolist())

print(f"\nDataset shape: {df_subset.shape}")
print(f"\nMissing values:")
missing = df_subset.isnull().sum()
print(missing[missing > 0] if missing.any() else "No missing values")
```

```{code-cell} python
:tags: [remove-stderr, remove-warnings]
# Basic statistics
print("Target variable (MEDV - Median home value) statistics:")
print(f"Mean: ${df_subset['MEDV'].mean():,.0f}k")
print(f"Median: ${df_subset['MEDV'].median():,.0f}k")
print(f"Min: ${df_subset['MEDV'].min():,.0f}k")
print(f"Max: ${df_subset['MEDV'].max():,.0f}k")

print("\nSample of the data:")
print(df_subset.head())
```

## Prepare Features for Embedding

```{code-cell} python
:tags: [remove-stderr, remove-warnings]
# Separate features from target
df_features = df_subset.drop(columns=['MEDV'])
print(f"Features for embedding: {df_features.columns.tolist()}")

# Check data types
print(f"\nData types:")
print(df_features.dtypes)

# Check for any categorical columns (should be none for Boston housing)
cat_cols = df_features.select_dtypes(include=['object']).columns
if len(cat_cols) > 0:
    print(f"\nCategorical column cardinalities:")
    for col in cat_cols:
        print(f"  {col}: {df_features[col].nunique()} unique values")
else:
    print(f"\nAll features are numeric (no categorical columns)")
```

## Unsupervised House Embeddings

Generate embeddings for each house:

```{code-cell} python
:tags: [remove-stderr, remove-warnings]

# Generate 8D embeddings for houses
house_embeddings = learn_embedding(
    df_features,
    mode="unsupervised",
    embedding_dim=8,
    max_epochs=40,
    batch_size=64,
    dropout_rate=0.2,
    hidden_units=256,
    verbose=False,
    seed=42
)

print(f"House embeddings shape: {house_embeddings.shape}")
print("\nEmbedding statistics:")
print(house_embeddings.describe().round(3))
```

## Visualize House Embeddings

```{code-cell} python
:tags: [remove-stderr, remove-warnings]
import matplotlib.pyplot as plt

# Create 2D embeddings for visualization
house_embeddings_2d = learn_embedding(
    df_features,
    mode="unsupervised",
    embedding_dim=2,
    max_epochs=30,
    batch_size=64,
    verbose=False,
    seed=42
)

# Create price categories for coloring
price_quartiles = df_subset['MEDV'].quantile([0.25, 0.5, 0.75])
price_categories = pd.cut(
    df_subset['MEDV'],
    bins=[0, price_quartiles[0.25], price_quartiles[0.5], price_quartiles[0.75], float('inf')],
    labels=['Low', 'Medium-Low', 'Medium-High', 'High']
)

# Plot
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Plot 1: Colored by price
scatter1 = axes[0].scatter(
    house_embeddings_2d.iloc[:, 0],
    house_embeddings_2d.iloc[:, 1],
    c=df_subset['MEDV'],
    cmap='viridis',
    alpha=0.6,
    s=20
)
axes[0].set_xlabel('Embedding Dimension 0')
axes[0].set_ylabel('Embedding Dimension 1')
axes[0].set_title('House Embeddings Colored by Median Value')
plt.colorbar(scatter1, ax=axes[0], label='Median Value ($k)')

# Plot 2: Colored by crime rate
scatter2 = axes[1].scatter(
    house_embeddings_2d.iloc[:, 0],
    house_embeddings_2d.iloc[:, 1],
    c=df_subset['CRIM'],
    cmap='coolwarm',
    alpha=0.6,
    s=20
)
axes[1].set_xlabel('Embedding Dimension 0')
axes[1].set_ylabel('Embedding Dimension 1')
axes[1].set_title('House Embeddings Colored by Crime Rate')
plt.colorbar(scatter2, ax=axes[1], label='Crime Rate')

plt.tight_layout()
plt.show()

print("Notice how expensive/high-quality houses cluster together!")
```

## Categorical Zone Embeddings

Create zone categories based on accessibility to radial highways:

```{code-cell} python
:tags: [remove-stderr, remove-warnings]

# Create accessibility zones based on RAD (radial highway access)
df_subset['AccessZone'] = pd.cut(
    df_subset['RAD'],
    bins=[0, 5, 10, 25],
    labels=['Low', 'Medium', 'High']
)

# Learn zone embeddings
zone_embeddings = learn_embedding(
    df_subset,
    mode="target",
    reference_column="AccessZone",
    embedding_dim=2,
    max_epochs=40,
    batch_size=128,
    verbose=False,
    seed=42
)

# Set proper index with category names
zone_embeddings.index = ['Low', 'Medium', 'High']

print(f"Number of access zones: {len(zone_embeddings)}")
print("\nAccess zone embeddings (2D):")
print(zone_embeddings.round(3))
```

## Analyze Zone Relationships

```{code-cell} python
:tags: [remove-stderr, remove-warnings]
# Calculate average median value by access zone for comparison
zone_prices = df_subset.groupby('AccessZone')['MEDV'].agg(['mean', 'count']).round(1)
zone_prices.columns = ['Avg_Value', 'House_Count']
zone_prices = zone_prices.sort_values('Avg_Value', ascending=False)

print("Access zones by average median value:")
print(zone_prices)
```

```{code-cell} python
:tags: [remove-stderr, remove-warnings]
# Visualize zone embeddings with price information
# Since we only have 2D embeddings, we can plot them directly
zones = zone_embeddings.index.tolist()
zone_avg_values = [zone_prices.loc[z, 'Avg_Value'] for z in zones]

plt.figure(figsize=(10, 8))
scatter = plt.scatter(
    zone_embeddings.iloc[:, 0],
    zone_embeddings.iloc[:, 1],
    c=zone_avg_values,
    cmap='viridis',
    s=200,
    alpha=0.8
)

# Label all zones
for i, zone in enumerate(zones):
    plt.annotate(
        f'{zone}\n(${zone_avg_values[i]:.1f}k)',
        (zone_embeddings.iloc[i, 0], zone_embeddings.iloc[i, 1]),
        xytext=(10, 10),
        textcoords='offset points',
        fontsize=12,
        fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7)
    )

plt.xlabel('Embedding Dimension 0')
plt.ylabel('Embedding Dimension 1')
plt.title('Access Zone Embeddings (All zones labeled with avg values)')
plt.colorbar(scatter, label='Average Median Value ($k)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("Zones with similar accessibility patterns cluster in embedding space!")
```

## Age Category Embeddings

Create age categories based on property age:

```{code-cell} python
:tags: [remove-stderr, remove-warnings]

# Create age categories based on AGE (proportion of owner-occupied units built prior to 1940)
df_subset['AgeCategory'] = pd.cut(
    df_subset['AGE'],
    bins=[0, 30, 70, 100],
    labels=['New', 'Medium', 'Old']
)

# Learn age category embeddings
age_embeddings = learn_embedding(
    df_subset,
    mode="target",
    reference_column="AgeCategory",
    embedding_dim=2,
    max_epochs=30,
    verbose=False,
    seed=42
)

# Set proper index with category names
age_embeddings.index = ['New', 'Medium', 'Old']

print("Age category embeddings:")
print(age_embeddings.round(3))

# Compare with actual values
age_values = df_subset.groupby('AgeCategory')['MEDV'].agg(['mean', 'count'])
age_values.columns = ['Avg_Value', 'Count']
print(f"\nAge categories by average median value:")
print(age_values.sort_values('Avg_Value', ascending=False).round(1))
```

## Use Embeddings for Price Prediction

```{code-cell} python
:tags: [remove-stderr, remove-warnings]

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Prepare data
X = house_embeddings  # 8D embeddings as features
y = df_subset['MEDV']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train regressor
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Predict
y_pred = rf.predict(X_test)

# Evaluate
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mse)

print(f"House median value prediction using 8D embeddings:")
print(f"R² Score: {r2:.3f}")
print(f"RMSE: ${rmse:,.1f}k")
print(f"Mean Absolute Error: ${np.mean(np.abs(y_test - y_pred)):,.1f}k")

# Feature importance (though these are embedding dimensions, not original features)
print(f"\nEmbedding dimension importance:")
for i, importance in enumerate(rf.feature_importances_):
    print(f"  Dimension {i}: {importance:.3f}")
```

## Compare Prediction Performance

```{code-cell} python
:tags: [remove-stderr, remove-warnings]

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer

# Compare embeddings vs raw features
# Prepare raw features manually
df_raw = df_features.copy()

# Label encode categorical columns (if any)
le_dict = {}
if len(cat_cols) > 0:
    for col in cat_cols:
        le = LabelEncoder()
        df_raw[col] = le.fit_transform(df_raw[col])
        le_dict[col] = le

# Scale raw features
scaler = StandardScaler()
X_raw_scaled = scaler.fit_transform(df_raw)

# Split raw features
X_raw_train, X_raw_test, y_train_raw, y_test_raw = train_test_split(
    X_raw_scaled, y, test_size=0.2, random_state=42
)

# Train on raw features
rf_raw = RandomForestRegressor(n_estimators=100, random_state=42)
rf_raw.fit(X_raw_train, y_train_raw)
y_pred_raw = rf_raw.predict(X_raw_test)

# Compare performance
r2_raw = r2_score(y_test_raw, y_pred_raw)
rmse_raw = np.sqrt(mean_squared_error(y_test_raw, y_pred_raw))

print("Performance Comparison:")
print("-" * 40)
print(f"Embeddings (8D):     R² = {r2:.3f}, RMSE = ${rmse:,.1f}k")
print(f"Raw Features ({X_raw_scaled.shape[1]}D):  R² = {r2_raw:.3f}, RMSE = ${rmse_raw:,.1f}k")
print(f"\nDimensionality reduction: {X_raw_scaled.shape[1]} → {X.shape[1]} ({(1-X.shape[1]/X_raw_scaled.shape[1]):.1%} reduction)")
```

## Classical Methods Comparison

```{code-cell} python
:tags: [remove-stderr, remove-warnings]

# Compare neural embeddings with classical methods
methods = {
    "Neural": {"mode": "unsupervised", "max_epochs": 20},
    "PCA": {"mode": "pca"}
}

# Use smaller sample for faster execution (Boston housing has 506 rows)
sample_size = 400
df_sample = df_features.sample(n=sample_size, random_state=42)
y_sample = y.loc[df_sample.index]

comparison_results = {}
for name, params in methods.items():
    print(f"Training {name}...")

    # Generate embeddings
    emb = learn_embedding(
        df_sample,
        embedding_dim=8,
        verbose=False,
        seed=42,
        **params
    )

    # Train and evaluate
    X_train_comp, X_test_comp, y_train_comp, y_test_comp = train_test_split(
        emb, y_sample, test_size=0.2, random_state=42
    )

    rf_comp = RandomForestRegressor(n_estimators=50, random_state=42)
    rf_comp.fit(X_train_comp, y_train_comp)
    y_pred_comp = rf_comp.predict(X_test_comp)

    r2_comp = r2_score(y_test_comp, y_pred_comp)
    rmse_comp = np.sqrt(mean_squared_error(y_test_comp, y_pred_comp))

    comparison_results[name] = {"r2": r2_comp, "rmse": rmse_comp}

print(f"\nMethod comparison (sample of {sample_size} houses):")
print("-" * 50)
for method, results in comparison_results.items():
    print(f"{method:8}: R² = {results['r2']:.3f}, RMSE = ${results['rmse']:,.1f}k")
```

## Production Pipeline

```{code-cell} python
:tags: [remove-stderr, remove-warnings]

from row2vec import train_and_save_model
import tempfile
import os

# Create production model
with tempfile.TemporaryDirectory() as tmpdir:
    model_path = os.path.join(tmpdir, "housing_model")

    embeddings_final, script_path, binary_path = train_and_save_model(
        df_features,
        base_path=model_path,
        embedding_dim=10,
        mode="unsupervised",
        max_epochs=50,
        batch_size=128,
        dropout_rate=0.2,
        hidden_units=512,
        scale_method="standard",
        verbose=False,
        seed=42
    )

    print(f"Housing model saved: {os.path.basename(script_path)}")

    # Demonstrate model loading and usage
    from row2vec import load_model
    model = load_model(script_path)

    # Test on new data
    test_houses = df_features.sample(n=50, random_state=999)
    test_embeddings = model.predict(test_houses)

    print(f"\nModel applied to {len(test_houses)} test houses")
    print(f"Generated embeddings shape: {test_embeddings.shape}")
    print(f"Training metadata:")
    print(f"  Epochs trained: {model.metadata.epochs_trained}")
    print(f"  Final loss: {model.metadata.final_loss if model.metadata.final_loss is not None else 'N/A'}")
    print(f"  Training time: {model.metadata.training_time:.2f}s")
```

## Key Insights

1. **Continuous Features**: Row2Vec effectively captures patterns in continuous housing data
2. **Urban Patterns**: Properties with similar accessibility and socioeconomic factors cluster together
3. **Dimensionality Reduction**: 11 features → 8 embeddings with minimal performance loss
4. **Predictive Power**: Embeddings achieve good R² for median value prediction
5. **Feature Relationships**: Crime rates, accessibility, and property age show meaningful embedding patterns

## Next Steps

- Learn about [Advanced Features](advanced_features.md) like architecture search
- Explore the [CLI Guide](cli_guide.md) for processing large real estate datasets
- Check the [API Reference](api_reference.md) for complete parameter details
