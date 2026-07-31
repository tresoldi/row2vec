"""
Configuration classes for Row2Vec embedding methods.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# Import categorical encoding configuration


@dataclass
class NeuralConfig:
    """Configuration for neural network-based embedding methods."""

    max_epochs: int = 50
    batch_size: int = 64
    dropout_rate: float = 0.2
    hidden_units: int | list[int] = 128
    activation: str = "relu"
    early_stopping: bool = True

    def __post_init__(self) -> None:
        """Minimal validation for neural config."""
        if self.max_epochs <= 0:
            raise ValueError("max_epochs must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not 0 <= self.dropout_rate <= 1:
            raise ValueError("dropout_rate must be between 0 and 1")

        # Validate hidden_units
        if isinstance(self.hidden_units, int):
            if self.hidden_units <= 0:
                raise ValueError("hidden_units must be positive")
        elif isinstance(self.hidden_units, list):
            if not self.hidden_units or any(h <= 0 for h in self.hidden_units):
                raise ValueError("All hidden_units must be positive")
        else:
            raise ValueError("hidden_units must be int or List[int]")

        # Validate activation
        valid_activations = ["relu", "elu", "swish", "tanh", "sigmoid", "leaky_relu"]
        if self.activation not in valid_activations:
            raise ValueError(f"activation must be one of: {valid_activations}")


@dataclass
class ClassicalConfig:
    """Configuration for classical ML dimensionality reduction methods."""

    # UMAP parameters
    n_neighbors: int = 15
    min_dist: float = 0.1

    # t-SNE parameters
    perplexity: float = 30.0
    n_iter: int = 1000

    def __post_init__(self) -> None:
        """Minimal validation for classical config."""
        if self.n_neighbors <= 0:
            raise ValueError("n_neighbors must be positive")
        if self.min_dist < 0:
            raise ValueError("min_dist must be non-negative")
        if self.perplexity <= 0:
            raise ValueError("perplexity must be positive")
        if self.n_iter <= 0:
            raise ValueError("n_iter must be positive")


@dataclass
class ContrastiveConfig:
    """Configuration for contrastive learning."""

    loss_type: str = "triplet"
    similar_pairs: list[tuple[int, int]] | None = None
    dissimilar_pairs: list[tuple[int, int]] | None = None
    auto_pairs: str | None = None  # "cluster", "neighbors", "categorical", "random"
    margin: float = 1.0
    negative_samples: int = 5

    def __post_init__(self) -> None:
        """Minimal validation for contrastive config."""
        if self.loss_type not in ["triplet", "contrastive"]:
            raise ValueError("loss_type must be 'triplet' or 'contrastive'")
        if self.auto_pairs is not None and self.auto_pairs not in [
            "cluster",
            "neighbors",
            "categorical",
            "random",
        ]:
            raise ValueError(
                "auto_pairs must be one of: cluster, neighbors, categorical, random"
            )
        if self.margin < 0:
            raise ValueError("margin must be non-negative")
        if self.negative_samples <= 0:
            raise ValueError("negative_samples must be positive")


@dataclass
class ScalingConfig:
    """Configuration for embedding scaling/normalization."""

    method: str | None = None  # "none", "minmax", "standard", "l2", "tanh"
    range: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        """Minimal validation for scaling config."""
        if self.method is not None and self.method not in [
            "none",
            "minmax",
            "standard",
            "l2",
            "tanh",
        ]:
            raise ValueError("method must be one of: none, minmax, standard, l2, tanh")
        if self.range is not None:
            if len(self.range) != 2:
                raise ValueError("range must be a tuple of two values")
            if self.range[0] >= self.range[1]:
                raise ValueError("range[0] must be less than range[1]")


@dataclass
class LoggingConfig:
    """Configuration for logging and output."""

    level: str = "INFO"
    file: str | None = None
    enabled: bool = True

    def __post_init__(self) -> None:
        """Minimal validation for logging config."""
        if self.level not in ["DEBUG", "INFO", "WARNING", "ERROR"]:
            raise ValueError("level must be one of: DEBUG, INFO, WARNING, ERROR")


@dataclass
class PreprocessingConfig:
    """Configuration for data preprocessing including categorical encoding."""

    # Missing value handling
    handle_missing: str = "auto"  # "auto", "drop", "impute", "custom"

    # Scaling configuration
    numeric_scaling: str = "standard"  # "standard", "minmax", "robust", "none"

    # Categorical encoding (will be imported from categorical_encoding module)
    categorical_encoding_strategy: str = "adaptive"
    categorical_onehot_threshold: int = 20
    categorical_target_threshold: int = 100
    categorical_entity_threshold: int = 1000

    def __post_init__(self) -> None:
        """Minimal validation for preprocessing config."""
        valid_missing = ["auto", "drop", "impute", "custom"]
        if self.handle_missing not in valid_missing:
            raise ValueError(f"handle_missing must be one of: {valid_missing}")

        valid_scaling = ["standard", "minmax", "robust", "none"]
        if self.numeric_scaling not in valid_scaling:
            raise ValueError(f"numeric_scaling must be one of: {valid_scaling}")


@dataclass
class EmbeddingConfig:
    """Complete configuration for embedding learning."""

    # Core parameters
    embedding_dim: int = 10
    mode: str = "unsupervised"
    reference_column: str | None = None
    seed: int = 1305
    verbose: bool = False

    # Sub-configurations
    neural: NeuralConfig = field(default_factory=NeuralConfig)
    classical: ClassicalConfig = field(default_factory=ClassicalConfig)
    contrastive: ContrastiveConfig = field(default_factory=ContrastiveConfig)
    scaling: ScalingConfig = field(default_factory=ScalingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)

    def __post_init__(self) -> None:
        """Minimal validation for main config."""
        if self.embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")

        valid_modes = ["unsupervised", "target", "pca", "tsne", "umap", "contrastive"]
        if self.mode not in valid_modes:
            raise ValueError(f"mode must be one of: {valid_modes}")

        if self.mode == "target" and self.reference_column is None:
            raise ValueError("reference_column is required when mode='target'")

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "EmbeddingConfig":
        """Create config from dictionary (e.g., from YAML)."""
        # Extract sub-configs
        neural_dict = config_dict.pop("neural", {})
        classical_dict = config_dict.pop("classical", {})
        contrastive_dict = config_dict.pop("contrastive", {})
        scaling_dict = config_dict.pop("scaling", {})
        logging_dict = config_dict.pop("logging", {})
        preprocessing_dict = config_dict.pop("preprocessing", {})

        return cls(
            neural=NeuralConfig(**neural_dict),
            classical=ClassicalConfig(**classical_dict),
            contrastive=ContrastiveConfig(**contrastive_dict),
            scaling=ScalingConfig(**scaling_dict),
            logging=LoggingConfig(**logging_dict),
            preprocessing=PreprocessingConfig(**preprocessing_dict),
            **config_dict,
        )

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> "EmbeddingConfig":
        """Create config from YAML file."""
        with open(yaml_path) as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary."""
        result = {
            "embedding_dim": self.embedding_dim,
            "mode": self.mode,
            "reference_column": self.reference_column,
            "seed": self.seed,
            "verbose": self.verbose,
        }

        # Add sub-configs as nested dictionaries
        result["neural"] = {
            "max_epochs": self.neural.max_epochs,
            "batch_size": self.neural.batch_size,
            "dropout_rate": self.neural.dropout_rate,
            "hidden_units": self.neural.hidden_units,
            "early_stopping": self.neural.early_stopping,
        }

        result["classical"] = {
            "n_neighbors": self.classical.n_neighbors,
            "min_dist": self.classical.min_dist,
            "perplexity": self.classical.perplexity,
            "n_iter": self.classical.n_iter,
        }

        result["contrastive"] = {
            "loss_type": self.contrastive.loss_type,
            "similar_pairs": self.contrastive.similar_pairs,
            "dissimilar_pairs": self.contrastive.dissimilar_pairs,
            "auto_pairs": self.contrastive.auto_pairs,
            "margin": self.contrastive.margin,
            "negative_samples": self.contrastive.negative_samples,
        }

        result["scaling"] = {
            "method": self.scaling.method,
            "range": self.scaling.range,
        }

        result["logging"] = {
            "level": self.logging.level,
            "file": self.logging.file,
            "enabled": self.logging.enabled,
        }

        result["preprocessing"] = {
            "handle_missing": self.preprocessing.handle_missing,
            "numeric_scaling": self.preprocessing.numeric_scaling,
            "categorical_encoding_strategy": self.preprocessing.categorical_encoding_strategy,
            "categorical_onehot_threshold": self.preprocessing.categorical_onehot_threshold,
            "categorical_target_threshold": self.preprocessing.categorical_target_threshold,
            "categorical_entity_threshold": self.preprocessing.categorical_entity_threshold,
        }

        return result

    def to_yaml(self, yaml_path: str | Path) -> None:
        """Save config to YAML file."""
        with open(yaml_path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)


def create_config_for_mode(mode: str, **overrides: Any) -> EmbeddingConfig:
    """Create a pre-configured EmbeddingConfig for common use cases."""
    # Handle target mode specially since it requires reference_column
    if mode == "target":
        # Create with dummy reference_column to pass validation
        base_config = EmbeddingConfig(mode=mode, reference_column="__placeholder__")
        base_config.neural.max_epochs = 75  # Supervised learning often converges faster
        # Clear the placeholder - user will need to set proper reference_column
        base_config.reference_column = None
    else:
        base_config = EmbeddingConfig(mode=mode)

        # Mode-specific optimizations
        if mode == "contrastive":
            base_config.neural.max_epochs = (
                100  # Contrastive learning often needs more epochs
            )
            base_config.neural.batch_size = 32  # Smaller batches for triplet learning
            base_config.contrastive.auto_pairs = "cluster"  # Set default auto_pairs
        elif mode in ["pca", "tsne", "umap"]:
            # Classical methods don't use neural config, but we keep it for consistency
            pass

    # Apply any overrides
    if overrides:
        config_dict = base_config.to_dict()
        # Simple override mechanism - only supports top-level keys for now
        for key, value in overrides.items():
            if key in config_dict:
                config_dict[key] = value
        # Recreate with overrides, handling target mode validation
        if mode == "target" and "reference_column" not in overrides:
            config_dict["reference_column"] = "__placeholder__"
            base_config = EmbeddingConfig.from_dict(config_dict)
            base_config.reference_column = None
        else:
            base_config = EmbeddingConfig.from_dict(config_dict)

    return base_config
