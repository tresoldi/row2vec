"""
Automatic Neural Architecture Search for Row2Vec embeddings.

This module provides intelligent automatic architecture search to find optimal
neural network configurations for embedding generation tasks.
"""

import random
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from .api import learn_embedding_v2
from .config import EmbeddingConfig, NeuralConfig


@dataclass
class ArchitectureSearchConfig:
    """
    Configuration for automatic neural architecture search.

    This class defines the search space, evaluation criteria, and stopping
    conditions for finding optimal neural network architectures.
    """

    # Search strategy
    method: str = "random"  # "random", "grid" (future: "bayesian", "evolutionary")

    # Search budget and stopping criteria
    max_trials: int = 30
    max_time: float | None = 1800  # 30 minutes default
    patience: int = 10  # Stop if no improvement for N trials
    min_improvement: float = 0.01  # Minimum improvement threshold

    # Search space definition
    layer_range: tuple[int, int] = (1, 4)  # Min/max number of hidden layers
    max_layers: int = 4  # Maximum number of hidden layers
    width_options: list[int] = field(default_factory=lambda: [32, 64, 128, 256, 512])
    dropout_options: list[float] = field(
        default_factory=lambda: [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    )
    activation_options: list[str] = field(
        default_factory=lambda: ["relu", "elu", "swish"]
    )

    def __post_init__(self):
        """Initialize derived parameters."""
        self.max_layers = max(self.layer_range[1], 1)  # Ensure at least 1 layer

    # Progressive training settings
    initial_epochs: int = 10  # Quick screening epochs
    intermediate_epochs: int = 25  # Focused evaluation epochs
    final_epochs: int = 50  # Best candidates full training
    top_k_intermediate: int = 10  # How many to evaluate with intermediate epochs
    top_k_final: int = 3  # How many to evaluate with full epochs

    # Evaluation weights
    reconstruction_weight: float = 0.4
    clustering_weight: float = 0.3
    efficiency_weight: float = 0.2
    stability_weight: float = 0.1

    # Other settings
    verbose: bool = True
    random_seed: int | None = None
    return_full_history: bool = False


class ArchitectureSearchResult:
    """Container for architecture search results."""

    def __init__(
        self,
        best_architecture: dict[str, Any],
        best_score: float,
        search_history: list[dict[str, Any]],
        total_time: float,
        trials_completed: int,
    ):
        self.best_architecture = best_architecture
        self.best_score = best_score
        self.search_history = search_history
        self.total_time = total_time
        self.trials_completed = trials_completed

    def summary(self) -> dict[str, Any]:
        """Get a summary of the search results."""
        return {
            "best_architecture": self.best_architecture,
            "best_score": self.best_score,
            "total_time": self.total_time,
            "trials_completed": self.trials_completed,
            "improvement_over_baseline": self._compute_improvement(),
            "search_efficiency": self.best_score / self.total_time
            if self.total_time > 0
            else 0,
        }

    def _compute_improvement(self) -> float:
        """Compute improvement over baseline architecture."""
        if len(self.search_history) < 2:
            return 0.0

        baseline_score = self.search_history[0]["score"]
        return (
            (self.best_score - baseline_score) / baseline_score
            if baseline_score > 0
            else 0.0
        )


class ArchitectureSearcher:
    """
    Main class for performing neural architecture search.

    Implements multiple search strategies to find optimal neural network
    architectures for embedding generation tasks.
    """

    def __init__(self, config: ArchitectureSearchConfig):
        self.config = config
        self.search_history: list[dict[str, Any]] = []
        self.best_score = float("-inf")
        self.best_architecture = None
        self.trials_without_improvement = 0

        if config.random_seed is not None:
            random.seed(config.random_seed)
            np.random.seed(config.random_seed)

    def search(
        self,
        df: pd.DataFrame,
        base_config: EmbeddingConfig,
        target_column: str | None = None,
    ) -> ArchitectureSearchResult:
        """
        Perform architecture search on the given dataset.

        Args:
            df: Input dataframe for embedding generation
            base_config: Base embedding configuration
            target_column: Optional target column for supervised evaluation

        Returns:
            ArchitectureSearchResult containing the best architecture and metadata
        """
        if self.config.verbose:
            pass

        start_time = time.time()

        try:
            if self.config.method == "random":
                self._random_search(df, base_config, target_column)
            elif self.config.method == "grid":
                self._grid_search(df, base_config, target_column)
            else:
                raise ValueError(f"Unknown search method: {self.config.method}")

            total_time = time.time() - start_time

            if self.config.verbose:
                pass

            return ArchitectureSearchResult(
                best_architecture=self.best_architecture or {},
                best_score=self.best_score,
                search_history=self.search_history
                if self.config.return_full_history
                else [],
                total_time=total_time,
                trials_completed=len(self.search_history),
            )

        except Exception:
            if self.config.verbose:
                pass
            raise

    def _random_search(
        self,
        df: pd.DataFrame,
        base_config: EmbeddingConfig,
        target_column: str | None,
    ) -> ArchitectureSearchResult:
        """Perform random search over the architecture space."""

        for trial in range(self.config.max_trials):
            # Check stopping criteria
            if self._should_stop(time.time()):
                break

            # Sample random architecture
            architecture = self._sample_random_architecture()

            # Evaluate architecture
            score = self._evaluate_architecture(
                df, base_config, architecture, target_column, trial
            )

            # Update best
            if score > self.best_score:
                self.best_score = score
                self.best_architecture = architecture
                self.trials_without_improvement = 0
                if self.config.verbose:
                    pass
            else:
                self.trials_without_improvement += 1

        return ArchitectureSearchResult(
            best_architecture=self.best_architecture or {},
            best_score=self.best_score,
            search_history=self.search_history,
            total_time=0,  # Will be set by caller
            trials_completed=len(self.search_history),
        )

    def _grid_search(
        self,
        df: pd.DataFrame,
        base_config: EmbeddingConfig,
        target_column: str | None,
    ) -> ArchitectureSearchResult:
        """Perform grid search over the architecture space."""

        # Generate all combinations
        architectures = self._generate_grid_architectures()

        if self.config.verbose:
            pass

        for trial, architecture in enumerate(architectures):
            # Check stopping criteria
            if self._should_stop(time.time()) or trial >= self.config.max_trials:
                break

            # Evaluate architecture
            score = self._evaluate_architecture(
                df, base_config, architecture, target_column, trial
            )

            # Update best
            if score > self.best_score:
                self.best_score = score
                self.best_architecture = architecture
                if self.config.verbose:
                    pass

        return ArchitectureSearchResult(
            best_architecture=self.best_architecture or {},
            best_score=self.best_score,
            search_history=self.search_history,
            total_time=0,  # Will be set by caller
            trials_completed=len(self.search_history),
        )

    def _sample_random_architecture(self) -> dict[str, Any]:
        """Sample a random architecture from the search space."""
        # Now supports multi-layer architectures
        n_layers = random.randint(1, self.config.max_layers)

        if n_layers == 1:
            # Single layer: keep as int for backward compatibility
            width = random.choice(self.config.width_options)
            hidden_units = width
        else:
            # Multi-layer: use list of layer widths
            layer_widths = []
            for _ in range(n_layers):
                width = random.choice(self.config.width_options)
                layer_widths.append(width)
            hidden_units = layer_widths

        return {
            "n_layers": n_layers,
            "hidden_units": hidden_units,  # Now supports both int and List[int]
            "dropout_rate": random.choice(self.config.dropout_options),
            "activation": random.choice(self.config.activation_options),
        }

    def _generate_grid_architectures(self) -> list[dict[str, Any]]:
        """Generate all architectures for grid search."""
        architectures = []

        # Grid over number of layers, widths, dropout, and activation
        for n_layers in range(
            1, min(self.config.max_layers + 1, 3)
        ):  # Limit to 2 layers for grid efficiency
            for width in self.config.width_options[
                ::2
            ]:  # Use every other width for efficiency
                for dropout in self.config.dropout_options[
                    ::2
                ]:  # Use every other dropout
                    for activation in self.config.activation_options:
                        if n_layers == 1:
                            hidden_units = width
                        else:
                            # For multi-layer, create decreasing layer sizes
                            hidden_units = [width // (i + 1) for i in range(n_layers)]
                            hidden_units = [
                                max(h, 16) for h in hidden_units
                            ]  # Ensure minimum size

                        architectures.append(
                            {
                                "n_layers": n_layers,
                                "hidden_units": hidden_units,
                                "dropout_rate": dropout,
                                "activation": activation,
                            }
                        )

        return architectures[: self.config.max_trials]  # Limit grid size

    def _generate_width_configurations(self, n_layers: int) -> list[list[int]]:
        """Generate sensible width configurations for a given number of layers."""
        configs = []

        # Common patterns
        if n_layers == 1:
            configs.extend([[w] for w in [64, 128, 256]])
        elif n_layers == 2:
            configs.extend(
                [
                    [128, 64],
                    [256, 128],
                    [512, 256],
                    [128, 128],
                ]
            )
        elif n_layers == 3:
            configs.extend(
                [
                    [256, 128, 64],
                    [512, 256, 128],
                    [128, 128, 64],
                ]
            )
        elif n_layers == 4:
            configs.extend(
                [
                    [512, 256, 128, 64],
                    [256, 256, 128, 64],
                ]
            )

        return configs

    def _evaluate_architecture(
        self,
        df: pd.DataFrame,
        base_config: EmbeddingConfig,
        architecture: dict[str, Any],
        target_column: str | None,
        trial_num: int,
    ) -> float:
        """Evaluate a single architecture configuration."""

        try:
            # Determine training epochs based on progressive training strategy
            epochs = self._get_epochs_for_trial(trial_num)

            # Adjust batch size based on dataset size
            batch_size = min(64, max(8, len(df) // 4))  # Reasonable batch size

            # Create neural config from architecture (simplified for current Row2Vec)
            neural_config = NeuralConfig(
                hidden_units=architecture["hidden_units"],  # Single integer value
                dropout_rate=architecture["dropout_rate"],
                activation=architecture["activation"],
                max_epochs=epochs,
                batch_size=batch_size,
                early_stopping=True,
            )

            # Create embedding config with the same mode as base_config
            config = EmbeddingConfig(
                mode=base_config.mode,
                embedding_dim=base_config.embedding_dim,
                neural=neural_config,
            )

            # Generate embeddings
            start_time = time.time()
            embeddings = learn_embedding_v2(df, config)
            training_time = time.time() - start_time

            # Compute evaluation metrics
            metrics = self._compute_evaluation_metrics(
                df, embeddings, training_time, target_column
            )

            # Compute weighted score
            score = (
                self.config.reconstruction_weight * metrics["reconstruction_score"]
                + self.config.clustering_weight * metrics["clustering_score"]
                + self.config.efficiency_weight * metrics["efficiency_score"]
                + self.config.stability_weight * metrics["stability_score"]
            )

            # Record trial
            trial_record = {
                "trial": trial_num,
                "architecture": architecture.copy(),
                "score": score,
                "metrics": metrics,
                "training_time": training_time,
                "epochs_used": epochs,
            }
            self.search_history.append(trial_record)

            if self.config.verbose and trial_num % 5 == 0:
                pass

            return score

        except Exception as e:
            if self.config.verbose:
                pass

            # Record failed trial
            trial_record = {
                "trial": trial_num,
                "architecture": architecture.copy(),
                "score": float("-inf"),
                "metrics": {},
                "training_time": 0,
                "epochs_used": 0,
                "error": str(e),
            }
            self.search_history.append(trial_record)

            return float("-inf")

    def _get_epochs_for_trial(self, trial_num: int) -> int:
        """Determine number of epochs based on progressive training strategy."""
        if trial_num < self.config.max_trials - self.config.top_k_final:
            if trial_num < self.config.max_trials - self.config.top_k_intermediate:
                return self.config.initial_epochs
            return self.config.intermediate_epochs
        return self.config.final_epochs

    def _compute_evaluation_metrics(
        self,
        df: pd.DataFrame,
        embeddings: pd.DataFrame,
        training_time: float,
        target_column: str | None,
    ) -> dict[str, float]:
        """Compute comprehensive evaluation metrics for embeddings."""

        metrics = {}

        # 1. Reconstruction score (inverse of reconstruction error)
        try:
            # Use variance preservation as a proxy for reconstruction quality
            original_var = df.select_dtypes(include=[np.number]).var().mean()
            embedding_var = embeddings.var().mean()
            reconstruction_score = (
                min(1.0, embedding_var / original_var) if original_var > 0 else 0.5
            )
            metrics["reconstruction_score"] = reconstruction_score
        except:
            metrics["reconstruction_score"] = 0.5

        # 2. Clustering quality score
        try:
            if len(embeddings) > 10:  # Need enough samples for clustering
                n_clusters = min(8, max(2, len(embeddings) // 20))
                kmeans = KMeans(n_clusters=n_clusters, random_state=1305, n_init=10)
                cluster_labels = kmeans.fit_predict(embeddings)

                if len(np.unique(cluster_labels)) > 1:
                    silhouette = silhouette_score(embeddings, cluster_labels)
                    clustering_score = (silhouette + 1) / 2  # Normalize to [0, 1]
                else:
                    clustering_score = 0.5
            else:
                clustering_score = 0.5

            metrics["clustering_score"] = clustering_score
        except:
            metrics["clustering_score"] = 0.5

        # 3. Efficiency score (inverse of training time, normalized)
        try:
            # Normalize by dataset size and dimensionality
            time_per_sample = training_time / len(df) if len(df) > 0 else training_time
            efficiency_score = 1.0 / (1.0 + time_per_sample * 1000)  # Scale factor
            metrics["efficiency_score"] = efficiency_score
        except:
            metrics["efficiency_score"] = 0.5

        # 4. Stability score (based on embedding variance and outliers)
        try:
            # Check for numerical stability
            has_nan = embeddings.isna().any().any()
            has_inf = np.isinf(embeddings.values).any()

            if has_nan or has_inf:
                stability_score = 0.0
            else:
                # Check for reasonable value ranges
                max_abs_value = np.abs(embeddings.values).max()
                if max_abs_value > 1000:  # Very large values indicate instability
                    stability_score = 0.1
                elif max_abs_value > 100:
                    stability_score = 0.5
                else:
                    stability_score = 1.0

            metrics["stability_score"] = stability_score
        except:
            metrics["stability_score"] = 0.5

        return metrics

    def _should_stop(self, current_time: float) -> bool:
        """Check if search should be stopped based on stopping criteria."""

        # Check patience
        if self.trials_without_improvement >= self.config.patience:
            if self.config.verbose:
                pass
            return True

        # Check time limit
        if self.config.max_time and (current_time - time.time()) > self.config.max_time:
            if self.config.verbose:
                pass
            return True

        return False

    def _describe_search_space(self) -> str:
        """Generate a human-readable description of the search space."""
        layer_count = self.config.layer_range[1] - self.config.layer_range[0] + 1
        width_count = len(self.config.width_options)
        dropout_count = len(self.config.dropout_options)
        activation_count = len(self.config.activation_options)

        total_combinations = (
            layer_count * (width_count**3) * dropout_count * activation_count
        )

        return (
            f"{layer_count} layer configs × {width_count} width options × "
            f"{dropout_count} dropout rates × {activation_count} activations "
            f"≈ {total_combinations:,} total combinations"
        )


def search_architecture(
    df: pd.DataFrame,
    base_config: EmbeddingConfig,
    search_config: ArchitectureSearchConfig | None = None,
    target_column: str | None = None,
) -> tuple[dict[str, Any], ArchitectureSearchResult]:
    """
    Perform automatic neural architecture search.

    This is the main entry point for architecture search functionality.

    Args:
        df: Input dataframe for embedding generation
        base_config: Base embedding configuration
        search_config: Architecture search configuration (uses defaults if None)
        target_column: Optional target column for supervised evaluation

    Returns:
        Tuple of (best_architecture_dict, full_search_result)
    """
    if search_config is None:
        search_config = ArchitectureSearchConfig()

    searcher = ArchitectureSearcher(search_config)
    result = searcher.search(df, base_config, target_column)

    return result.best_architecture, result
