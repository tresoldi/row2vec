"""
Modernized Row2Vec API with config-based parameters.

This module provides the new config-based API for Row2Vec embedding learning.
The old parameter-based API is maintained for backward compatibility but will
be deprecated in future versions.
"""

from typing import TYPE_CHECKING, Any, Optional

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer

if TYPE_CHECKING:
    from .architecture_search import ArchitectureSearchConfig

from .config import EmbeddingConfig, create_config_for_mode
from .core import learn_embedding as _legacy_learn_embedding
from .core import learn_embedding_with_model as _legacy_learn_embedding_with_model


def learn_embedding_v2(
    df: pd.DataFrame,
    config: EmbeddingConfig | None = None,
    auto_architecture: bool = False,
    architecture_search_config: Optional["ArchitectureSearchConfig"] = None,
    **config_overrides,
) -> pd.DataFrame:
    """
    Modern config-based API for learning embeddings from tabular data.

    This is the new recommended API that uses configuration objects instead
    of long parameter lists. It provides better organization, type safety,
    and extensibility.

    Args:
        df: Input DataFrame containing the data to embed
        config: Complete embedding configuration. If None, default config is used.
        auto_architecture: Enable automatic neural architecture search for neural modes
        architecture_search_config: Custom architecture search configuration
        **config_overrides: Override specific config values (supports nested keys with dots)

    Returns:
        DataFrame containing the learned embeddings

    Examples:
        # Basic usage with defaults
        embeddings = learn_embedding_v2(df)

        # Using a custom config
        config = EmbeddingConfig(
            mode="contrastive",
            embedding_dim=50,
            contrastive=ContrastiveConfig(loss_type="triplet", margin=2.0)
        )
        embeddings = learn_embedding_v2(df, config)

        # With automatic architecture search
        embeddings = learn_embedding_v2(df, config, auto_architecture=True)

        # Quick overrides without config object
        embeddings = learn_embedding_v2(df, embedding_dim=20, mode="target", reference_column="category")

        # Loading from YAML
        config = EmbeddingConfig.from_yaml("my_config.yaml")
        embeddings = learn_embedding_v2(df, config)
    """
    if config is None:
        config = EmbeddingConfig()

    # Apply any config overrides
    if config_overrides:
        config = _apply_config_overrides(config, config_overrides)

    # Handle automatic architecture search for neural modes
    if auto_architecture and config.mode in [
        "unsupervised",
        "contrastive",
    ]:  # Neural-based modes
        from .architecture_search import ArchitectureSearchConfig, search_architecture

        if architecture_search_config is None:
            architecture_search_config = ArchitectureSearchConfig()

        # Perform architecture search
        best_architecture, search_result = search_architecture(
            df, config, architecture_search_config
        )

        # Update config with best architecture
        config.neural.hidden_units = best_architecture["hidden_units"]
        config.neural.dropout_rate = best_architecture["dropout_rate"]
        config.neural.activation = best_architecture["activation"]

        if architecture_search_config and architecture_search_config.verbose:
            search_result.summary()

    # Convert config to legacy parameters and call the existing function
    legacy_params = _config_to_legacy_params(config)

    return _legacy_learn_embedding(df, **legacy_params)


def learn_embedding_with_model_v2(
    df: pd.DataFrame,
    config: EmbeddingConfig | None = None,
    **config_overrides,
) -> tuple[pd.DataFrame, Any | BaseEstimator, ColumnTransformer, dict[str, Any]]:
    """
    Modern config-based API for learning embeddings with model artifacts.

    This function returns the embeddings along with the trained model,
    preprocessor, and metadata for serialization purposes.

    Args:
        df: Input DataFrame containing the data to embed
        config: Complete embedding configuration. If None, default config is used.
        **config_overrides: Override specific config values

    Returns:
        Tuple of (embeddings, model, preprocessor, metadata)
    """
    if config is None:
        config = EmbeddingConfig()

    # Apply any config overrides
    if config_overrides:
        config = _apply_config_overrides(config, config_overrides)

    # Convert config to legacy parameters and call the existing function
    legacy_params = _config_to_legacy_params(config)

    return _legacy_learn_embedding_with_model(df, **legacy_params)


def _apply_config_overrides(
    config: EmbeddingConfig, overrides: dict[str, Any]
) -> EmbeddingConfig:
    """Apply override values to a config object."""
    # Convert config to dict, apply overrides, and convert back
    config_dict = config.to_dict()

    # Neural method parameters that need special handling
    neural_params = {
        "max_epochs",
        "batch_size",
        "dropout_rate",
        "hidden_units",
        "early_stopping",
    }

    # Classical method parameters that need special handling
    classical_params = {
        "n_neighbors",
        "perplexity",
        "min_dist",
        "n_iter",
    }

    # Contrastive method parameters that need special handling
    contrastive_params = {
        "similar_pairs",
        "dissimilar_pairs",
        "auto_pairs",
        "contrastive_loss",
        "margin",
        "negative_samples",
    }

    for key, value in overrides.items():
        if "." in key:
            # Handle nested keys like "neural.max_epochs"
            parts = key.split(".")
            current = config_dict
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            current[parts[-1]] = value
        elif key in neural_params:
            # Handle neural parameters
            if "neural" not in config_dict:
                config_dict["neural"] = {}
            config_dict["neural"][key] = value
        elif key in classical_params:
            # Handle classical parameters
            if "classical" not in config_dict:
                config_dict["classical"] = {}
            config_dict["classical"][key] = value
        elif key in contrastive_params:
            # Handle contrastive parameters
            if "contrastive" not in config_dict:
                config_dict["contrastive"] = {}
            config_dict["contrastive"][key] = value
        else:
            # Handle top-level keys
            config_dict[key] = value

    return EmbeddingConfig.from_dict(config_dict)


def _config_to_legacy_params(config: EmbeddingConfig) -> dict[str, Any]:
    """Convert an EmbeddingConfig to legacy function parameters."""
    return {
        # Core parameters
        "embedding_dim": config.embedding_dim,
        "mode": config.mode,
        "reference_column": config.reference_column,
        "seed": config.seed,
        "verbose": config.verbose,
        # Neural parameters
        "max_epochs": config.neural.max_epochs,
        "batch_size": config.neural.batch_size,
        "dropout_rate": config.neural.dropout_rate,
        "hidden_units": config.neural.hidden_units,
        "early_stopping": config.neural.early_stopping,
        # Classical parameters
        "n_neighbors": config.classical.n_neighbors,
        "perplexity": config.classical.perplexity,
        "min_dist": config.classical.min_dist,
        "n_iter": config.classical.n_iter,
        # Contrastive parameters
        "similar_pairs": config.contrastive.similar_pairs,
        "dissimilar_pairs": config.contrastive.dissimilar_pairs,
        "auto_pairs": config.contrastive.auto_pairs,
        "contrastive_loss": config.contrastive.loss_type,
        "margin": config.contrastive.margin,
        "negative_samples": config.contrastive.negative_samples,
        # Scaling parameters
        "scale_method": config.scaling.method,
        "scale_range": config.scaling.range,
        # Logging parameters
        "log_level": config.logging.level,
        "log_file": config.logging.file,
        "enable_logging": config.logging.enabled,
    }


# Convenience functions for common configurations
def learn_embedding_unsupervised(
    df: pd.DataFrame, embedding_dim: int = 10, **overrides
) -> pd.DataFrame:
    """Learn unsupervised embeddings with optimized defaults."""
    config = EmbeddingConfig(mode="unsupervised", embedding_dim=embedding_dim)
    return learn_embedding_v2(df, config, **overrides)


def learn_embedding_target(
    df: pd.DataFrame, reference_column: str, embedding_dim: int = 10, **overrides
) -> pd.DataFrame:
    """Learn target-based embeddings with optimized defaults."""
    config = EmbeddingConfig(
        mode="target", reference_column=reference_column, embedding_dim=embedding_dim
    )
    return learn_embedding_v2(df, config, **overrides)


def learn_embedding_contrastive(df: pd.DataFrame, **overrides) -> pd.DataFrame:
    """Convenience function for contrastive learning with optimized defaults."""
    config = create_config_for_mode("contrastive")

    # Handle contrastive-specific parameters that should go under contrastive config
    contrastive_params = {}
    for key in ["loss_type", "margin", "negative_samples", "auto_pairs"]:
        if key in overrides:
            contrastive_params[key] = overrides.pop(key)

    # Apply contrastive overrides
    if contrastive_params:
        for key, value in contrastive_params.items():
            setattr(config.contrastive, key, value)

    return learn_embedding_v2(df, config, **overrides)


def learn_embedding_classical(
    df: pd.DataFrame, method: str = "pca", embedding_dim: int = 10, **overrides
) -> pd.DataFrame:
    """Learn classical ML embeddings (PCA, t-SNE, UMAP) with optimized defaults."""
    if method not in ["pca", "tsne", "umap"]:
        raise ValueError("method must be one of: pca, tsne, umap")

    config = EmbeddingConfig(mode=method, embedding_dim=embedding_dim)
    return learn_embedding_v2(df, config, **overrides)
