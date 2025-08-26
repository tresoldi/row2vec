"""
Row2Vec: A library for learning embeddings from tabular data.

This library provides both neural network and classical machine learning
approaches for creating vector embeddings from tabular datasets.
"""

__version__ = "0.1.0"
__author__ = "Tiago Tresoldi"
__email__ = "tiago@tresoldi.org"

from .core import learn_embedding, learn_embedding_with_model
from .api import (
    learn_embedding_v2, 
    learn_embedding_with_model_v2,
    learn_embedding_unsupervised,
    learn_embedding_target,
    learn_embedding_contrastive,
    learn_embedding_classical,
)
from .auto_dimension import AutoDimensionSelector, auto_select_dimension
from .architecture_search import (
    ArchitectureSearchConfig, 
    ArchitectureSearchResult, 
    ArchitectureSearcher, 
    search_architecture
)
from .imputation import (
    ImputationConfig,
    MissingPatternAnalyzer, 
    AdaptiveImputer
)
from .config import (
    ClassicalConfig,
    ContrastiveConfig,
    EmbeddingConfig,
    LoggingConfig,
    NeuralConfig,
    ScalingConfig,
    PreprocessingConfig,
    CategoricalEncodingConfig,
)
from .categorical_encoding import (
    CategoricalAnalyzer,
    CategoricalEncoder,
    EntityEmbeddingTrainer,
    TargetEncoder,
)
from .pipeline_builder import (
    PipelineBuilder,
    build_adaptive_pipeline,
)
from .logging import Row2VecLogger, get_logger
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
    from .sklearn import Row2VecTransformer, Row2VecClassifier
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False

__all__ = [
    # Core API
    "learn_embedding_v2",
    
    # Configuration
    "EmbeddingConfig", "NeuralConfig", "ClassicalConfig", "ContrastiveConfig", 
    "ScalingConfig", "LoggingConfig",
    
    # Auto-optimization
    "AutoDimensionSelector", "auto_select_dimension",
    "ArchitectureSearchConfig", "ArchitectureSearchResult", "ArchitectureSearcher", "search_architecture",
    
    # Missing value imputation
    "ImputationConfig", "MissingPatternAnalyzer", "AdaptiveImputer",
    
    # Utilities
    "get_logger",
]# Add sklearn integrations if available
if _SKLEARN_AVAILABLE:
    __all__.extend([
        "Row2VecTransformer",
        "Row2VecClassifier"
    ])
