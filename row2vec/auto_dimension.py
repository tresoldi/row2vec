"""
Automatic Embedding Dimension Selection for Row2Vec

This module provides intelligent dimension selection capabilities that automatically
determine optimal embedding dimensions based on data characteristics and performance metrics.
"""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.metrics import silhouette_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder

from .api import learn_embedding_v2
from .config import EmbeddingConfig, NeuralConfig
from .logging import get_logger


class AutoDimensionSelector:
    """
    Automatically selects optimal embedding dimensions using multiple strategies.

    Combines data-driven analysis, performance optimization, and heuristic rules
    to determine the best embedding dimension for a given dataset.
    """

    def __init__(
        self,
        methods: list[str] | None = None,
        performance_weight: float = 0.4,
        efficiency_weight: float = 0.3,
        intrinsic_weight: float = 0.3,
        max_dimension: int | None = None,
        min_dimension: int = 2,
        n_trials: int = 5,
        verbose: bool = True,
    ):
        """
        Initialize automatic dimension selector.

        Args:
            methods: List of selection methods to use
            performance_weight: Weight for performance-based selection
            efficiency_weight: Weight for efficiency considerations
            intrinsic_weight: Weight for intrinsic dimensionality estimation
            max_dimension: Maximum dimension to consider (auto if None)
            min_dimension: Minimum dimension to consider
            n_trials: Number of trials for performance evaluation
            verbose: Whether to show selection progress
        """
        self.methods = methods or [
            "pca_variance",
            "intrinsic_dim",
            "performance_based",
            "clustering_quality",
            "heuristic_rules",
        ]
        self.performance_weight = performance_weight
        self.efficiency_weight = efficiency_weight
        self.intrinsic_weight = intrinsic_weight
        self.max_dimension = max_dimension
        self.min_dimension = min_dimension
        self.n_trials = n_trials
        self.verbose = verbose
        self.logger = get_logger(__name__)

        # Results storage
        self.selection_results_: dict[str, Any] = {}
        self.dimension_scores_: dict[int, float] = {}

    def select_dimension(
        self,
        df: pd.DataFrame,
        config: EmbeddingConfig,
        target_column: str | None = None,
        candidate_dims: list[int] | None = None,
    ) -> tuple[int, dict[str, Any]]:
        """
        Select optimal embedding dimension for the given data.

        Args:
            df: Input dataframe
            config: Base embedding configuration (dimension will be overridden)
            target_column: Optional target for supervised evaluation
            candidate_dims: Specific dimensions to evaluate (auto-generated if None)

        Returns:
            Tuple of (optimal_dimension, selection_metadata)
        """
        if self.verbose:
            pass

        # Generate candidate dimensions if not provided
        if candidate_dims is None:
            candidate_dims = self._generate_candidate_dimensions(df)

        if self.verbose:
            pass

        # Apply each selection method
        method_results = {}

        for method in self.methods:
            try:
                if self.verbose:
                    pass

                result = self._apply_method(
                    method, df, config, candidate_dims, target_column
                )
                method_results[method] = result

                if self.verbose:
                    pass

            except Exception:
                if self.verbose:
                    pass
                method_results[method] = {"recommended_dim": None, "score": 0.0}

        # Combine results using weighted voting
        optimal_dim = self._combine_recommendations(method_results, candidate_dims)

        # Store results
        self.selection_results_ = method_results
        self.dimension_scores_ = self._calculate_dimension_scores(
            method_results, candidate_dims
        )

        metadata = {
            "candidate_dimensions": candidate_dims,
            "method_results": method_results,
            "dimension_scores": self.dimension_scores_,
            "selection_strategy": "weighted_voting",
            "weights": {
                "performance": self.performance_weight,
                "efficiency": self.efficiency_weight,
                "intrinsic": self.intrinsic_weight,
            },
        }

        if self.verbose:
            pass

        return optimal_dim, metadata

    def _generate_candidate_dimensions(self, df: pd.DataFrame) -> list[int]:
        """Generate reasonable candidate dimensions based on data characteristics."""
        n_samples, n_features = df.shape

        # Calculate various heuristic bounds
        sqrt_features = int(np.sqrt(n_features))
        log_samples = int(np.log2(max(n_samples, 2)))

        # Set reasonable bounds
        min_dim = max(self.min_dimension, 2)

        if self.max_dimension is not None:
            max_dim = self.max_dimension
        else:
            # Auto-determine max dimension
            max_dim = min(
                n_features // 2, 50, n_samples // 10, max(sqrt_features * 2, 10)
            )

        max_dim = max(max_dim, min_dim)

        # Generate candidate list
        candidates = set()

        # Add heuristic-based candidates
        candidates.update([sqrt_features, log_samples])

        # Add evenly spaced candidates
        step = max(1, (max_dim - min_dim) // 8)
        candidates.update(range(min_dim, max_dim + 1, step))

        # Add boundary points
        candidates.update([min_dim, max_dim])

        # Filter and sort
        candidates = [d for d in candidates if min_dim <= d <= max_dim]
        return sorted(set(candidates))

    def _apply_method(
        self,
        method: str,
        df: pd.DataFrame,
        config: EmbeddingConfig,
        candidate_dims: list[int],
        target_column: str | None = None,
    ) -> dict[str, Any]:
        """Apply a specific dimension selection method."""

        if method == "pca_variance":
            return self._pca_variance_method(df, candidate_dims)
        if method == "intrinsic_dim":
            return self._intrinsic_dimensionality_method(df, candidate_dims)
        if method == "performance_based":
            return self._performance_based_method(
                df, config, candidate_dims, target_column
            )
        if method == "clustering_quality":
            return self._clustering_quality_method(df, config, candidate_dims)
        if method == "heuristic_rules":
            return self._heuristic_rules_method(df, candidate_dims)
        raise ValueError(f"Unknown method: {method}")

    def _pca_variance_method(
        self, df: pd.DataFrame, candidate_dims: list[int]
    ) -> dict[str, Any]:
        """Select dimension based on PCA explained variance analysis."""
        # Prepare numeric data
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.empty:
            return {
                "recommended_dim": candidate_dims[len(candidate_dims) // 2],
                "score": 0.0,
            }

        # Fit PCA
        max_components = min(
            len(candidate_dims), numeric_df.shape[1], numeric_df.shape[0]
        )
        pca = PCA(n_components=max_components)
        pca.fit(numeric_df.fillna(0))

        # Find elbow point in explained variance
        explained_var = np.cumsum(pca.explained_variance_ratio_)

        # Look for elbow using second derivative
        if len(explained_var) >= 3:
            second_deriv = np.diff(explained_var, 2)
            elbow_idx = np.argmax(second_deriv) + 2
        else:
            elbow_idx = len(explained_var) // 2

        # Find closest candidate dimension
        target_dim = min(int(elbow_idx + 1), max(candidate_dims))
        recommended_dim = min(candidate_dims, key=lambda x: abs(x - target_dim))

        # Score based on variance explained at recommended dimension
        score = explained_var[min(recommended_dim - 1, len(explained_var) - 1)]

        return {
            "recommended_dim": recommended_dim,
            "score": score,
            "explained_variance": explained_var.tolist(),
            "target_dimension": target_dim,
        }

    def _intrinsic_dimensionality_method(
        self, df: pd.DataFrame, candidate_dims: list[int]
    ) -> dict[str, Any]:
        """Estimate intrinsic dimensionality using manifold learning."""
        # Prepare numeric data
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.empty or numeric_df.shape[0] < 20:
            return {
                "recommended_dim": candidate_dims[len(candidate_dims) // 2],
                "score": 0.5,
            }

        try:
            # Use subset if data is large
            sample_size = min(1000, numeric_df.shape[0])
            if sample_size < numeric_df.shape[0]:
                sample_data = numeric_df.sample(n=sample_size, random_state=1305)
            else:
                sample_data = numeric_df

            sample_data = sample_data.fillna(0)

            # Estimate using locally linear embedding reconstruction error
            errors = []
            test_dims = [d for d in candidate_dims if d < sample_data.shape[1]]

            for dim in test_dims:
                try:
                    lle = LocallyLinearEmbedding(
                        n_components=dim,
                        n_neighbors=min(10, sample_data.shape[0] - 1),
                        random_state=1305,
                    )
                    lle.fit(sample_data)
                    errors.append(lle.reconstruction_error_)
                except:
                    errors.append(np.inf)

            if not errors or all(e == np.inf for e in errors):
                return {
                    "recommended_dim": candidate_dims[len(candidate_dims) // 2],
                    "score": 0.5,
                }

            # Find dimension where error stabilizes
            errors = np.array(errors)
            valid_errors = errors[errors != np.inf]

            if len(valid_errors) < 2:
                recommended_dim = candidate_dims[len(candidate_dims) // 2]
            else:
                # Normalize errors and find stabilization point
                norm_errors = (valid_errors - valid_errors.min()) / (
                    valid_errors.max() - valid_errors.min() + 1e-8
                )
                diff_errors = np.diff(norm_errors)

                # Find where improvement becomes marginal
                stabilization_idx = 0
                for i, diff in enumerate(diff_errors):
                    if abs(diff) < 0.1:  # Less than 10% relative improvement
                        stabilization_idx = i
                        break

                recommended_dim = test_dims[min(stabilization_idx, len(test_dims) - 1)]

            # Score based on error reduction
            score = 1.0 / (1.0 + errors[test_dims.index(recommended_dim)])

            return {
                "recommended_dim": recommended_dim,
                "score": score,
                "reconstruction_errors": errors.tolist(),
                "test_dimensions": test_dims,
            }

        except Exception:
            return {
                "recommended_dim": candidate_dims[len(candidate_dims) // 2],
                "score": 0.5,
            }

    def _performance_based_method(
        self,
        df: pd.DataFrame,
        config: EmbeddingConfig,
        candidate_dims: list[int],
        target_column: str | None = None,
    ) -> dict[str, Any]:
        """Select dimension based on downstream task performance."""
        if target_column is None or target_column not in df.columns:
            # Use unsupervised clustering quality
            return self._clustering_quality_method(df, config, candidate_dims)

        # Supervised evaluation
        try:
            X = df.drop(columns=[target_column])
            y = df[target_column]

            # Encode target if categorical
            if y.dtype == "object":
                le = LabelEncoder()
                y = le.fit_transform(y)

            scores = []
            for dim in candidate_dims:
                try:
                    # Generate embeddings
                    test_config = EmbeddingConfig(
                        mode=config.mode,
                        embedding_dim=dim,
                        neural=NeuralConfig(
                            max_epochs=min(10, config.neural.max_epochs)
                        ),  # Faster evaluation
                        scaling=config.scaling,
                    )

                    embeddings = learn_embedding_v2(X, test_config)

                    # Evaluate with simple classifier
                    clf = LogisticRegression(random_state=1305, max_iter=100)
                    cv_scores = cross_val_score(
                        clf, embeddings, y, cv=3, scoring="accuracy"
                    )
                    scores.append(cv_scores.mean())

                except Exception:
                    scores.append(0.0)

            if not scores or max(scores) == 0:
                return {
                    "recommended_dim": candidate_dims[len(candidate_dims) // 2],
                    "score": 0.5,
                }

            best_idx = np.argmax(scores)
            recommended_dim = candidate_dims[best_idx]

            return {
                "recommended_dim": recommended_dim,
                "score": scores[best_idx],
                "all_scores": scores,
                "evaluation_type": "supervised_classification",
            }

        except Exception:
            return {
                "recommended_dim": candidate_dims[len(candidate_dims) // 2],
                "score": 0.5,
            }

    def _clustering_quality_method(
        self,
        df: pd.DataFrame,
        config: EmbeddingConfig,
        candidate_dims: list[int],
    ) -> dict[str, Any]:
        """Select dimension based on clustering quality metrics."""
        try:
            silhouette_scores = []

            for dim in candidate_dims:
                try:
                    # Generate embeddings with fast configuration
                    test_config = EmbeddingConfig(
                        mode="pca"
                        if config.mode in ["unsupervised", "contrastive"]
                        else config.mode,
                        embedding_dim=dim,
                        scaling=config.scaling,
                    )

                    embeddings = learn_embedding_v2(df, test_config)

                    # Perform clustering
                    n_clusters = min(max(2, int(np.sqrt(len(embeddings)))), 10)
                    kmeans = KMeans(n_clusters=n_clusters, random_state=1305, n_init=3)
                    cluster_labels = kmeans.fit_predict(embeddings)

                    # Calculate silhouette score
                    if len(set(cluster_labels)) > 1:
                        sil_score = silhouette_score(embeddings, cluster_labels)
                    else:
                        sil_score = 0.0

                    silhouette_scores.append(sil_score)

                except Exception:
                    silhouette_scores.append(0.0)

            if not silhouette_scores or max(silhouette_scores) <= 0:
                return {
                    "recommended_dim": candidate_dims[len(candidate_dims) // 2],
                    "score": 0.5,
                }

            best_idx = np.argmax(silhouette_scores)
            recommended_dim = candidate_dims[best_idx]

            return {
                "recommended_dim": recommended_dim,
                "score": silhouette_scores[best_idx],
                "silhouette_scores": silhouette_scores,
                "evaluation_type": "clustering_quality",
            }

        except Exception:
            return {
                "recommended_dim": candidate_dims[len(candidate_dims) // 2],
                "score": 0.5,
            }

    def _heuristic_rules_method(
        self, df: pd.DataFrame, candidate_dims: list[int]
    ) -> dict[str, Any]:
        """Apply rule-of-thumb heuristics for dimension selection."""
        n_samples, n_features = df.shape

        # Calculate various heuristic recommendations
        heuristics = {
            "sqrt_features": int(np.sqrt(n_features)),
            "log_samples": int(np.log2(max(n_samples, 2))),
            "features_ratio": max(2, n_features // 4),
            "sample_ratio": max(2, int(np.sqrt(n_samples / 10))),
        }

        # Weight heuristics based on data characteristics
        weights = {
            "sqrt_features": 0.3,
            "log_samples": 0.2,
            "features_ratio": 0.3,
            "sample_ratio": 0.2,
        }

        # Adjust weights based on data size
        if n_samples < 100:
            weights["sample_ratio"] *= 0.5
        if n_features > 50:
            weights["features_ratio"] *= 1.5

        # Calculate weighted recommendation
        weighted_sum = sum(heuristics[h] * weights[h] for h in heuristics)
        target_dim = int(weighted_sum)

        # Find closest candidate
        recommended_dim = min(candidate_dims, key=lambda x: abs(x - target_dim))

        # Score based on how well it matches multiple heuristics
        agreements = sum(
            1 for h_dim in heuristics.values() if abs(h_dim - recommended_dim) <= 2
        )
        score = agreements / len(heuristics)

        return {
            "recommended_dim": recommended_dim,
            "score": score,
            "heuristics": heuristics,
            "target_dimension": target_dim,
            "agreements": agreements,
        }

    def _combine_recommendations(
        self, method_results: dict, candidate_dims: list[int]
    ) -> int:
        """Combine recommendations from different methods using weighted voting."""
        # Create vote matrix
        votes = dict.fromkeys(candidate_dims, 0.0)

        # Weight mapping for methods
        method_weights = {
            "pca_variance": self.intrinsic_weight * 0.6,
            "intrinsic_dim": self.intrinsic_weight * 0.4,
            "performance_based": self.performance_weight,
            "clustering_quality": self.performance_weight * 0.8,
            "heuristic_rules": self.efficiency_weight,
        }

        # Accumulate weighted votes
        for method, result in method_results.items():
            if result["recommended_dim"] is not None:
                weight = method_weights.get(method, 0.1) * result["score"]
                votes[result["recommended_dim"]] += weight

        # Find dimension with highest vote
        if not any(votes.values()):
            return candidate_dims[len(candidate_dims) // 2]

        return max(votes.keys(), key=lambda k: votes[k])

    def _calculate_dimension_scores(
        self, method_results: dict, candidate_dims: list[int]
    ) -> dict[int, float]:
        """Calculate overall scores for each candidate dimension."""
        scores = dict.fromkeys(candidate_dims, 0.0)

        method_weights = {
            "pca_variance": self.intrinsic_weight * 0.6,
            "intrinsic_dim": self.intrinsic_weight * 0.4,
            "performance_based": self.performance_weight,
            "clustering_quality": self.performance_weight * 0.8,
            "heuristic_rules": self.efficiency_weight,
        }

        for method, result in method_results.items():
            if result["recommended_dim"] is not None:
                weight = method_weights.get(method, 0.1)
                recommended_dim = result["recommended_dim"]
                method_score = result["score"]

                # Distribute score to nearby dimensions with falloff
                for dim in candidate_dims:
                    distance = abs(dim - recommended_dim)
                    if distance == 0:
                        scores[dim] += weight * method_score
                    elif distance <= 2:
                        scores[dim] += weight * method_score * (0.5**distance)

        return scores


def auto_select_dimension(
    df: pd.DataFrame,
    config: EmbeddingConfig | None = None,
    target_column: str | None = None,
    methods: list[str] | None = None,
    **selector_kwargs,
) -> tuple[int, dict[str, Any]]:
    """
    Convenience function for automatic dimension selection.

    Args:
        df: Input dataframe
        config: Base embedding configuration (uses defaults if None)
        target_column: Optional target column for supervised evaluation
        methods: List of selection methods to use
        **selector_kwargs: Additional arguments for AutoDimensionSelector

    Returns:
        Tuple of (optimal_dimension, selection_metadata)
    """
    if config is None:
        config = EmbeddingConfig(
            mode="pca", embedding_dim=5
        )  # Temporary, will be overridden

    selector = AutoDimensionSelector(methods=methods, **selector_kwargs)
    return selector.select_dimension(df, config, target_column)
