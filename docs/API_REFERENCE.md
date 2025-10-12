# API Reference

Complete reference for all Row2Vec functions and classes.

## Core Functions

### `learn_embedding()`

Primary function for generating embeddings from tabular data.

```python
learn_embedding(
    data: pd.DataFrame,
    mode: str = "unsupervised",
    embedding_dim: int = 10,
    reference_column: Optional[str] = None,
    **kwargs
) -> pd.DataFrame
```

**Parameters:**
- `data`: Input DataFrame with mixed-type columns
- `mode`: Embedding method - `"unsupervised"`, `"target"`, `"pca"`, `"tsne"`, or `"umap"`
- `embedding_dim`: Output dimensionality
- `reference_column`: Target column for `mode="target"`
- `**kwargs`: Method-specific parameters

**Returns:** DataFrame with embeddings (shape: `[n_rows, embedding_dim]`)

**Examples:**
```python
# Neural embeddings
emb = learn_embedding(df, mode="unsupervised", embedding_dim=5)

# Classical methods
pca_emb = learn_embedding(df, mode="pca", embedding_dim=3)
tsne_emb = learn_embedding(df, mode="tsne", embedding_dim=2)
```

---

### `learn_embedding_v2()`

Configuration-based embedding function with full parameter control.

```python
learn_embedding_v2(
    data: pd.DataFrame,
    config: EmbeddingConfig
) -> pd.DataFrame
```

**Parameters:**
- `data`: Input DataFrame
- `config`: Complete configuration object

**Returns:** DataFrame with embeddings

---

## Model Serialization

### `train_and_save_model()`

Train embedding model and save for reuse.

```python
train_and_save_model(
    data: pd.DataFrame,
    base_path: str,
    embedding_dim: int,
    mode: str = "unsupervised",
    **kwargs
) -> Tuple[pd.DataFrame, str, str]
```

**Parameters:**
- `base_path`: Base path for saved files
- Other parameters same as `learn_embedding()`

**Returns:** `(embeddings, script_path, binary_path)`

---

### `load_model()`

Load previously saved model.

```python
load_model(script_path: str) -> Row2VecModel
```

**Returns:** Model object with `.predict()` method

---

## Architecture Search

### `search_architecture()`

Automatically discover optimal neural network architectures.

```python
search_architecture(
    data: pd.DataFrame,
    base_config: EmbeddingConfig,
    search_config: ArchitectureSearchConfig
) -> Tuple[Dict, List[Dict]]
```

**Parameters:**
- `data`: Training data
- `base_config`: Base embedding configuration
- `search_config`: Search space definition

**Returns:** `(best_architecture, all_results)`

---

### `ArchitectureSearchConfig`

Configuration for architecture search.

```python
ArchitectureSearchConfig(
    method: str = "random",           # "random" or "grid"
    max_layers: int = 3,
    width_options: List[int] = [64, 128, 256],
    dropout_options: List[float] = [0.0, 0.2, 0.5],
    max_trials: int = 50
)
```

---

## Missing Value Handling

### `AdaptiveImputer`

Intelligent missing value imputation.

```python
imputer = AdaptiveImputer(config: ImputationConfig)
imputed_data = imputer.fit_transform(data)
```

---

### `MissingPatternAnalyzer`

Analyze missing data patterns.

```python
analyzer = MissingPatternAnalyzer(config: ImputationConfig)
analysis = analyzer.analyze(data)
```

**Returns:** Dictionary with pattern analysis and recommendations

---

## Categorical Encoding

### `CategoricalEncoder`

Encode categorical features with various strategies.

```python
encoder = CategoricalEncoder(config: CategoricalEncodingConfig)
encoded_data = encoder.fit_transform(data)
```

---

### `EntityEmbeddingTrainer`

Learn entity embeddings for categorical variables.

```python
trainer = EntityEmbeddingTrainer(
    embedding_dim: int = 10,
    epochs: int = 50
)
embeddings = trainer.fit_transform(data, target_column)
```

---

## Configuration Classes

### `EmbeddingConfig`

Main configuration for embedding generation.

```python
EmbeddingConfig(
    mode: str = "unsupervised",
    embedding_dim: int = 10,
    neural: Optional[NeuralConfig] = None,
    preprocessing: Optional[PreprocessingConfig] = None
)
```

---

### `NeuralConfig`

Neural network training parameters.

```python
NeuralConfig(
    hidden_units: List[int] = [128, 64],
    max_epochs: int = 100,
    batch_size: int = 32,
    dropout_rate: float = 0.2,
    learning_rate: float = 0.001,
    verbose: bool = True
)
```

---

### `PreprocessingConfig`

Data preprocessing settings.

```python
PreprocessingConfig(
    scaling: Optional[ScalingConfig] = None,
    imputation: Optional[ImputationConfig] = None,
    encoding: Optional[CategoricalEncodingConfig] = None
)
```

---

## Utility Functions

### `generate_synthetic_data()`

Create synthetic tabular data for testing.

```python
generate_synthetic_data(
    num_records: int = 1000,
    seed: Optional[int] = None
) -> pd.DataFrame
```

**Returns:** DataFrame with mixed numeric and categorical columns

---

### `create_dataframe_schema()`

Generate schema from DataFrame for validation.

```python
create_dataframe_schema(df: pd.DataFrame) -> Dict
```

---

### `validate_dataframe_schema()`

Validate DataFrame against schema.

```python
validate_dataframe_schema(
    df: pd.DataFrame,
    schema: Dict
) -> Tuple[bool, List[str]]
```

**Returns:** `(is_valid, error_messages)`

---

## Scikit-learn Integration

### `Row2VecTransformer`

Scikit-learn compatible transformer.

```python
from row2vec import Row2VecTransformer

transformer = Row2VecTransformer(
    embedding_dim=10,
    mode="unsupervised"
)

# Use in sklearn pipelines
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

pipeline = Pipeline([
    ('embeddings', Row2VecTransformer(embedding_dim=5)),
    ('classifier', RandomForestClassifier())
])
```

---

### `Row2VecClassifier`

Combined embedding + classification.

```python
from row2vec import Row2VecClassifier

classifier = Row2VecClassifier(
    embedding_dim=8,
    classifier_type='random_forest'
)

classifier.fit(X_train, y_train)
predictions = classifier.predict(X_test)
```

---

## Pandas Integration

### DataFrame Accessor

Use `.row2vec` accessor on DataFrames.

```python
import row2vec.pandas  # Registers accessor

# Generate embeddings
embeddings = df.row2vec.embed(mode="unsupervised", embedding_dim=5)

# Architecture search
best_arch = df.row2vec.search_architecture(max_trials=20)
```

---

## Logging

### `Row2VecLogger`

Configurable logging system.

```python
from row2vec import Row2VecLogger, LoggingConfig

logger = Row2VecLogger(LoggingConfig(
    level="INFO",
    file_path="row2vec.log"
))
```

---

### `get_logger()`

Get configured logger instance.

```python
from row2vec import get_logger

logger = get_logger(__name__)
logger.info("Processing started")
```

---

## Command-Line Interface

See [CLI Guide](cli_guide.md) for complete command-line documentation.

Basic commands:
```bash
# Generate embeddings
row2vec annotate --input data.csv --output embeddings.csv

# Train model
row2vec train --input data.csv --output model.py

# Use saved model
row2vec predict --input new_data.csv --model model.py
```

---

## Type Hints

Row2Vec is fully typed. Import types for static checking:

```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from row2vec import (
        EmbeddingConfig,
        NeuralConfig,
        Row2VecModel,
    )
```

---

For working examples, see the [Quickstart Guide](tutorial_1_quickstart.html) and [Advanced Features](tutorial_2_advanced.html).
