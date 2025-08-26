# API Reference

Complete documentation of Row2Vec's Python API.

## Core Functions

### learn_embedding()

The main function for generating embeddings from tabular data.

```{eval-rst}
.. currentmodule:: row2vec

.. autofunction:: learn_embedding
```

### learn_embedding_with_model()

Extended function that returns model components for advanced use cases.

```{eval-rst}
.. autofunction:: learn_embedding_with_model
```

## V2 API Functions

The newer, more flexible API with configuration objects.

```{eval-rst}
.. autofunction:: learn_embedding_v2
.. autofunction:: learn_embedding_with_model_v2
.. autofunction:: learn_embedding_unsupervised
.. autofunction:: learn_embedding_target
.. autofunction:: learn_embedding_classical
```

## Configuration Classes

### EmbeddingConfig

```{eval-rst}
.. autoclass:: EmbeddingConfig
   :members:
   :show-inheritance:
```

### NeuralConfig

```{eval-rst}
.. autoclass:: NeuralConfig
   :members:
   :show-inheritance:
```

### ClassicalConfig

```{eval-rst}
.. autoclass:: ClassicalConfig
   :members:
   :show-inheritance:
```

### ScalingConfig

```{eval-rst}
.. autoclass:: ScalingConfig
   :members:
   :show-inheritance:
```

### LoggingConfig

```{eval-rst}
.. autoclass:: LoggingConfig
   :members:
   :show-inheritance:
```

### PreprocessingConfig

```{eval-rst}
.. autoclass:: PreprocessingConfig
   :members:
   :show-inheritance:
```

## Architecture Search

### ArchitectureSearchConfig

```{eval-rst}
.. autoclass:: ArchitectureSearchConfig
   :members:
   :show-inheritance:
```

### search_architecture()

```{eval-rst}
.. autofunction:: search_architecture
```

### ArchitectureSearcher

```{eval-rst}
.. autoclass:: ArchitectureSearcher
   :members:
   :show-inheritance:
```

## Auto Dimension Selection

### auto_select_dimension()

```{eval-rst}
.. autofunction:: auto_select_dimension
```

### AutoDimensionSelector

```{eval-rst}
.. autoclass:: AutoDimensionSelector
   :members:
   :show-inheritance:
```

## Imputation

### ImputationConfig

```{eval-rst}
.. autoclass:: ImputationConfig
   :members:
   :show-inheritance:
```

### AdaptiveImputer

```{eval-rst}
.. autoclass:: AdaptiveImputer
   :members:
   :show-inheritance:
```

### MissingPatternAnalyzer

```{eval-rst}
.. autoclass:: MissingPatternAnalyzer
   :members:
   :show-inheritance:
```

## Categorical Encoding

### CategoricalEncodingConfig

```{eval-rst}
.. autoclass:: CategoricalEncodingConfig
   :members:
   :show-inheritance:
```

### CategoricalEncoder

```{eval-rst}
.. autoclass:: CategoricalEncoder
   :members:
   :show-inheritance:
```

### CategoricalAnalyzer

```{eval-rst}
.. autoclass:: CategoricalAnalyzer
   :members:
   :show-inheritance:
```

## Model Serialization

### Row2VecModel

```{eval-rst}
.. autoclass:: Row2VecModel
   :members:
   :show-inheritance:
```

### Row2VecModelMetadata

```{eval-rst}
.. autoclass:: Row2VecModelMetadata
   :members:
   :show-inheritance:
```

### save_model()

```{eval-rst}
.. autofunction:: save_model
```

### load_model()

```{eval-rst}
.. autofunction:: load_model
```

### train_and_save_model()

```{eval-rst}
.. autofunction:: train_and_save_model
```

## Utilities

### generate_synthetic_data()

```{eval-rst}
.. autofunction:: generate_synthetic_data
```

### create_dataframe_schema()

```{eval-rst}
.. autofunction:: create_dataframe_schema
```

### validate_dataframe_schema()

```{eval-rst}
.. autofunction:: validate_dataframe_schema
```

## Logging

### get_logger()

```{eval-rst}
.. autofunction:: get_logger
```

### Row2VecLogger

```{eval-rst}
.. autoclass:: Row2VecLogger
   :members:
   :show-inheritance:
```

## Pipeline Building

### PipelineBuilder

```{eval-rst}
.. autoclass:: PipelineBuilder
   :members:
   :show-inheritance:
```

### build_adaptive_pipeline()

```{eval-rst}
.. autofunction:: build_adaptive_pipeline
```

## sklearn Integration

When scikit-learn integration is available:

### Row2VecTransformer

```{eval-rst}
.. autoclass:: Row2VecTransformer
   :members:
   :show-inheritance:
```

### Row2VecClassifier

```{eval-rst}
.. autoclass:: Row2VecClassifier
   :members:
   :show-inheritance:
```

## pandas Integration

When pandas integration is available:

### DataFrame.row2vec Accessor

The `.row2vec` accessor provides direct embedding methods on pandas DataFrames:

```python
import pandas as pd
from row2vec import *

df = pd.read_csv('data.csv')

# Generate embeddings directly from DataFrame
embeddings = df.row2vec.embed(mode='unsupervised', embedding_dim=5)

# Target-based embeddings
category_embeddings = df.row2vec.embed_target('category_column', embedding_dim=3)

# Quick visualization embeddings
viz_embeddings = df.row2vec.embed_2d()
```

## Type Hints

Row2Vec is fully type-annotated. Key type aliases:

```python
from typing import Union, List, Tuple, Optional, Dict, Any
import pandas as pd
import numpy as np

# Common type aliases used in Row2Vec
DataFrame = pd.DataFrame
NDArray = np.ndarray
ModelType = Union['keras.Model', 'sklearn.base.BaseEstimator']
EmbeddingDimensions = Union[int, List[int]]
ScalingRange = Tuple[float, float]
ConfigDict = Dict[str, Any]
```

## Error Handling

Row2Vec defines custom exceptions for better error handling:

```python
# Common exceptions you might encounter
from row2vec.exceptions import (
    Row2VecError,           # Base exception
    ConfigurationError,     # Invalid configuration
    DataValidationError,    # Data validation failed
    ModelError,             # Model-related errors
    SerializationError      # Save/load errors
)
```

## Performance Considerations

### Memory Usage

For large datasets:
- Use `sample_size` parameter to limit memory usage
- Consider `batch_size` parameter for neural networks
- Use appropriate data types (float32 vs float64)

### Speed Optimization

- Use PCA mode for fastest results
- Reduce `max_epochs` for quick prototyping
- Use larger `batch_size` with sufficient memory
- Enable `early_stopping` to avoid overtraining

### GPU Support

Row2Vec automatically uses GPU when available through TensorFlow:

```python
import tensorflow as tf
print("GPU Available: ", tf.config.list_physical_devices('GPU'))

# Force CPU usage if needed
with tf.device('/CPU:0'):
    embeddings = learn_embedding(df, mode='unsupervised')
```

## Version Information

Check Row2Vec version and dependencies:

```python
import row2vec
print(f"Row2Vec version: {row2vec.__version__}")

# Check feature availability
print(f"Pandas integration: {row2vec._PANDAS_AVAILABLE}")
print(f"sklearn integration: {row2vec._SKLEARN_AVAILABLE}")
```
