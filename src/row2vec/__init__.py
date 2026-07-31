"""
Row2Vec: A library for learning embeddings from tabular data.

This library provides both neural network and classical machine learning
approaches for creating vector embeddings from tabular datasets.
"""

__version__ = "0.1.0"
__author__ = "Tiago Tresoldi"
__email__ = "tiago@tresoldi.org"

from .api import (
    learn_embedding_classical,
    learn_embedding_contrastive,
    learn_embedding_target,
    learn_embedding_unsupervised,
    learn_embedding_v2,
    learn_embedding_with_model_v2,
)
from .architecture_search import (
    ArchitectureSearchConfig,
    ArchitectureSearcher,
    ArchitectureSearchResult,
    search_architecture,
)
from .auto_dimension import AutoDimensionSelector, auto_select_dimension
from .categorical_encoding import (
    CategoricalAnalyzer,
    CategoricalEncoder,
    CategoricalEncodingConfig,
    EntityEmbeddingTrainer,
    TargetEncoder,
)
from .config import (
    ClassicalConfig,
    ContrastiveConfig,
    EmbeddingConfig,
    LoggingConfig,
    NeuralConfig,
    PreprocessingConfig,
    ScalingConfig,
)
from .core import learn_embedding, learn_embedding_with_model
from .imputation import AdaptiveImputer, ImputationConfig, MissingPatternAnalyzer
from .logging import Row2VecLogger, get_logger
from .pipeline_builder import (
    PipelineBuilder,
    build_adaptive_pipeline,
)
from .serialization import (
    Row2VecModel,
    Row2VecModelMetadata,
    load_model,
    save_model,
    train_and_save_model,
)
from .utils import (
    create_dataframe_schema,
    generate_synthetic_data,
    validate_dataframe_schema,
)

# Import pandas accessor to register it
try:
    from . import pandas  # This registers the .row2vec accessor

    _PANDAS_AVAILABLE = True
except ImportError:
    _PANDAS_AVAILABLE = False

# Sklearn integration (optional import)
try:
    from .sklearn import Row2VecClassifier, Row2VecTransformer

    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False

__all__ = [
    "AdaptiveImputer",
    "ArchitectureSearchConfig",
    "ArchitectureSearchResult",
    "ArchitectureSearcher",
    # Auto-optimization
    "AutoDimensionSelector",
    # Categorical encoding
    "CategoricalAnalyzer",
    "CategoricalEncoder",
    "CategoricalEncodingConfig",
    "ClassicalConfig",
    "ContrastiveConfig",
    # Configuration
    "EmbeddingConfig",
    "EntityEmbeddingTrainer",
    # Missing value imputation
    "ImputationConfig",
    "LoggingConfig",
    "MissingPatternAnalyzer",
    "NeuralConfig",
    # Pipeline building
    "PipelineBuilder",
    "PreprocessingConfig",
    "Row2VecLogger",
    # Serialization
    "Row2VecModel",
    "Row2VecModelMetadata",
    "ScalingConfig",
    "TargetEncoder",
    "auto_select_dimension",
    "build_adaptive_pipeline",
    "create_dataframe_schema",
    "generate_synthetic_data",
    # Utilities
    "get_logger",
    "learn_embedding",
    "learn_embedding_classical",
    "learn_embedding_contrastive",
    "learn_embedding_target",
    "learn_embedding_unsupervised",
    # Core API
    "learn_embedding_v2",
    "learn_embedding_with_model",
    "learn_embedding_with_model_v2",
    "load_model",
    "save_model",
    "search_architecture",
    "train_and_save_model",
    "validate_dataframe_schema",
]

# Add sklearn integrations if available
if _SKLEARN_AVAILABLE:
    __all__.extend(
        [
            "Row2VecClassifier",
            "Row2VecTransformer",
        ]
    )
