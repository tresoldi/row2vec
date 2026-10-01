"""Automatic selection of the embedding dimension.

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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from .api import learn_embedding_v2
from .config import EmbeddingConfig, NeuralConfig
from .logging import get_logger
from .utils import is_categorical_series


class AutoDimensionSelector:
    """Automatically selects optimal embedding dimensions using multiple strategies.

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
        random_state: int = 1305,
    ):
        """Initialize automatic dimension selector.

        Args:
            methods: List of selection methods to use
            performance_weight: Weight for performance-based selection
            efficiency_weight: Weight for efficiency considerations
            intrinsic_weight: Weight for intrinsic dimensionality estimation
            max_dimension: Maximum dimension to consider (auto if None)
            min_dimension: Minimum dimension to consider
            n_trials: Number of trials for performance evaluation
            random_state: Seed for every estimator this selector fits
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
        self.random_state = random_state
        self.logger = get_logger(__name__)

        # Results storage
        self.selection_results_: dict[str, Any] = {}
        self.dimension_scores_: dict[int, float] = {}

    def _log(self, message: str) -> None:
        """Emit a progress message when asked to be verbose.

        ``verbose`` used to guard six empty ``pass`` statements, so the flag
        documented as "Whether to show selection progress" did nothing at all.

        Args:
            message (str): The message to emit.
        """
        if self.verbose:
            self.logger.log_debug_info(f"[auto-dimension] {message}")

    def _failed_result(self, method: str, reason: str) -> dict[str, Any]:
        """Record that a method could not produce a recommendation.

        Every selection method used to answer a failure with the midpoint of
        the candidate list and a score of 0.5, which is indistinguishable from
        a real recommendation. A method that failed now abstains: it casts no
        vote, and says why.

        Args:
            method (str): The method that failed.
            reason (str): Why it could not recommend a dimension.

        Returns:
            dict[str, Any]: A result that scores nothing and votes for nothing.
        """
        self._log(f"{method} abstained: {reason}")
        return {"recommended_dim": None, "score": 0.0, "failed_reason": reason}

    def select_dimension(
        self,
        df: pd.DataFrame,
        config: EmbeddingConfig,
        target_column: str | None = None,
        candidate_dims: list[int] | None = None,
    ) -> tuple[int, dict[str, Any]]:
        """Select optimal embedding dimension for the given data.

        Args:
            df: Input dataframe
            config: Base embedding configuration (dimension will be overridden)
            target_column: Optional target for supervised evaluation
            candidate_dims: Specific dimensions to evaluate (auto-generated if None)

        Returns:
            Tuple of (optimal_dimension, selection_metadata)
        """
        self._log(f"Selecting an embedding dimension for a {df.shape} frame")

        # Generate candidate dimensions if not provided
        if candidate_dims is None:
            candidate_dims = self._generate_candidate_dimensions(df)

        self._log(f"Candidate dimensions: {candidate_dims}")

        # Apply each selection method
        method_results = {}

        for method in self.methods:
            try:
                self._log(f"Running {method}")
                result = self._apply_method(method, df, config, candidate_dims, target_column)
                method_results[method] = result
                self._log(f"{method} recommends {result['recommended_dim']}")
            except Exception as exc:
                method_results[method] = self._failed_result(method, str(exc))

        # Combine results using weighted voting
        optimal_dim = self._combine_recommendations(method_results, candidate_dims)

        # Store results
        self.selection_results_ = method_results
        self.dimension_scores_ = self._calculate_dimension_scores(method_results, candidate_dims)

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

        self._log(f"Selected embedding dimension {optimal_dim}")

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
            max_dim = min(n_features // 2, 50, n_samples // 10, max(sqrt_features * 2, 10))

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
        return sorted(d for d in candidates if min_dim <= d <= max_dim)

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
            return self._performance_based_method(df, config, candidate_dims, target_column)
        if method == "clustering_quality":
            return self._clustering_quality_method(df, config, candidate_dims)
        if method == "heuristic_rules":
            return self._heuristic_rules_method(df, candidate_dims)
        raise ValueError(f"Unknown method: {method}")

    @staticmethod
    def _variance_knee(explained_var: "np.ndarray[Any, Any]") -> int:
        """Number of components at the knee of a cumulative-variance curve.

        The previous implementation took ``argmax`` of the second difference of
        the cumulative curve. Because the increments of that curve are
        non-increasing, its second difference is almost always negative, so
        argmax picked the least-negative entry - in practice index 0, giving a
        constant answer of 3 regardless of the data.

        This instead takes the point furthest from the chord joining the first
        and last points of the curve, the standard knee construction, and
        refuses to recommend more components than it takes to reach 95% of the
        variance.

        Args:
            explained_var: Cumulative explained variance ratio, ascending.

        Returns:
            int: The recommended number of components, at least 1.
        """
        n = len(explained_var)
        if n == 0:
            return 1
        if n < 3:
            return n

        # Distance from each point to the straight line between the endpoints.
        x = np.arange(n, dtype=float)
        y = np.asarray(explained_var, dtype=float)
        x0, y0, x1, y1 = x[0], y[0], x[-1], y[-1]
        denominator = np.hypot(y1 - y0, x1 - x0)
        if denominator == 0:
            return n
        distances = np.abs((y1 - y0) * x - (x1 - x0) * y + x1 * y0 - y1 * x0) / denominator
        knee = int(np.argmax(distances)) + 1

        # Never ask for more components than reaching 95% of the variance needs.
        enough = int(np.searchsorted(y, 0.95) + 1)
        return max(1, min(knee, enough, n))

    def _pca_variance_method(self, df: pd.DataFrame, candidate_dims: list[int]) -> dict[str, Any]:
        """Select dimension based on PCA explained variance analysis."""
        # Prepare numeric data
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.empty:
            return self._failed_result("pca_variance", "no numeric columns to analyse")

        # Fit PCA over as many components as the data allows. This used to be
        # capped at len(candidate_dims) - the *count* of candidates, not any
        # dimension - so a candidate list of [2, 4, 8, 16, 32] fitted five
        # components and could never recommend more than five.
        max_components = min(
            max(candidate_dims),
            numeric_df.shape[1],
            numeric_df.shape[0],
        )
        pca = PCA(n_components=max_components, random_state=self.random_state)
        pca.fit(numeric_df.fillna(0))

        explained_var = np.cumsum(pca.explained_variance_ratio_)
        target_dim = self._variance_knee(explained_var)

        # Find closest candidate dimension
        recommended_dim = min(candidate_dims, key=lambda x: abs(x - target_dim))

        # Score based on variance explained at recommended dimension
        score = float(explained_var[min(recommended_dim - 1, len(explained_var) - 1)])

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
            return self._failed_result(
                "intrinsic_dim",
                "needs at least 20 rows of numeric data",
            )

        try:
            # Use subset if data is large
            sample_size = min(1000, numeric_df.shape[0])
            if sample_size < numeric_df.shape[0]:
                sample_data = numeric_df.sample(n=sample_size, random_state=self.random_state)
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
                        random_state=self.random_state,
                    )
                    lle.fit(sample_data)
                    errors.append(lle.reconstruction_error_)
                except Exception:
                    errors.append(np.inf)

            if not errors or all(e == np.inf for e in errors):
                return self._failed_result(
                    "intrinsic_dim",
                    "locally linear embedding failed at every candidate dimension",
                )

            # Find dimension where error stabilizes
            error_array = np.array(errors)
            valid_errors = error_array[error_array != np.inf]

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

            # Score based on error reduction. recommended_dim may have come
            # from the fallback above and not be in test_dims at all.
            if recommended_dim in test_dims:
                score = 1.0 / (1.0 + error_array[test_dims.index(recommended_dim)])
            else:
                score = 1.0 / (1.0 + float(valid_errors.min()))

            return {
                "recommended_dim": recommended_dim,
                "score": score,
                "reconstruction_errors": error_array.tolist(),
                "test_dimensions": test_dims,
            }

        except Exception as exc:
            return self._failed_result("intrinsic_dim", str(exc))

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
            if is_categorical_series(y):
                le = LabelEncoder()
                y = le.fit_transform(y)

            from .sklearn import Row2VecTransformer

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

                    # Embed inside the cross-validation, not before it. Fitting
                    # the embedding on all of X and then cross-validating the
                    # classifier let every test fold contribute to the
                    # representation it was scored on, which inflated the score
                    # and, since that score picks the dimension, the choice too.
                    pipeline = Pipeline(
                        [
                            (
                                "embed",
                                Row2VecTransformer(
                                    embedding_dim=dim,
                                    mode=test_config.mode,
                                    seed=test_config.seed,
                                    config=test_config,
                                ),
                            ),
                            (
                                "clf",
                                LogisticRegression(random_state=self.random_state, max_iter=100),
                            ),
                        ]
                    )
                    cv_scores = cross_val_score(pipeline, X, y, cv=3, scoring="accuracy")
                    scores.append(cv_scores.mean())

                except Exception:
                    scores.append(0.0)

            if not scores or max(scores) == 0:
                return self._failed_result(
                    "performance_based",
                    "no candidate dimension scored above zero",
                )

            best_idx = np.argmax(scores)
            recommended_dim = candidate_dims[best_idx]

            return {
                "recommended_dim": recommended_dim,
                "score": scores[best_idx],
                "all_scores": scores,
                "evaluation_type": "supervised_classification",
            }

        except Exception as exc:
            return self._failed_result("performance_based", str(exc))

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
                    kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init=3)
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
                return self._failed_result(
                    "clustering_quality",
                    "no candidate dimension produced a positive silhouette score",
                )

            best_idx = np.argmax(silhouette_scores)
            recommended_dim = candidate_dims[best_idx]

            return {
                "recommended_dim": recommended_dim,
                "score": silhouette_scores[best_idx],
                "silhouette_scores": silhouette_scores,
                "evaluation_type": "clustering_quality",
            }

        except Exception as exc:
            return self._failed_result("clustering_quality", str(exc))

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
        agreements = sum(1 for h_dim in heuristics.values() if abs(h_dim - recommended_dim) <= 2)
        score = agreements / len(heuristics)

        return {
            "recommended_dim": recommended_dim,
            "score": score,
            "heuristics": heuristics,
            "target_dimension": target_dim,
            "agreements": agreements,
        }

    def _combine_recommendations(self, method_results: dict, candidate_dims: list[int]) -> int:
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
    **selector_kwargs: Any,
) -> tuple[int, dict[str, Any]]:
    """Convenience function for automatic dimension selection.

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
        config = EmbeddingConfig(mode="pca", embedding_dim=5)  # Temporary, will be overridden

    selector = AutoDimensionSelector(methods=methods, **selector_kwargs)
    return selector.select_dimension(df, config, target_column)
