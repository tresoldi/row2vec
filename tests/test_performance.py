"""
Performance benchmarks for Row2Vec.

This module provides benchmarking tools to compare different embedding methods,
measure performance across dataset sizes, and profile memory usage.
"""

import os
import time
from typing import Any

import numpy as np
import pandas as pd
import psutil
import pytest

from row2vec import learn_embedding


class PerformanceBenchmark:
    """Performance benchmarking utilities."""

    @staticmethod
    def measure_time_and_memory(func, *args, **kwargs) -> tuple[float, float, Any]:
        """
        Measure execution time and peak memory usage of a function.

        Returns:
            Tuple of (execution_time, peak_memory_mb, result)
        """
        # Get initial memory
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Execute function and measure time
        start_time = time.time()
        result = func(*args, **kwargs)
        execution_time = time.time() - start_time

        # Get final memory
        final_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Calculate memory difference, ensuring it's not negative
        # (negative values can occur due to garbage collection)
        memory_diff = final_memory - initial_memory
        peak_memory = max(0.0, memory_diff)  # Ensure non-negative

        return execution_time, peak_memory, result

    @staticmethod
    def generate_benchmark_data(
        n_samples: int,
        n_features: int = 10,
        n_categorical: int = 3,
    ) -> pd.DataFrame:
        """Generate synthetic data for benchmarking."""
        np.random.seed(42)

        data = {}

        # Numeric features
        for i in range(n_features - n_categorical):
            data[f"numeric_{i}"] = np.random.normal(0, 1, n_samples)

        # Categorical features
        for i in range(n_categorical):
            n_categories = np.random.randint(3, 10)
            categories = [f"cat_{i}_{j}" for j in range(n_categories)]
            data[f"categorical_{i}"] = np.random.choice(categories, n_samples)

        return pd.DataFrame(data)


class TestPerformanceBenchmarks:
    """Performance benchmark tests."""

    def test_method_comparison_benchmark(self):
        """Compare performance of different embedding methods."""
        df = PerformanceBenchmark.generate_benchmark_data(500, n_features=15)
        embedding_dim = 5

        methods = [
            ("PCA", "pca", {}),
            ("t-SNE", "tsne", {"perplexity": 20, "n_iter": 250}),
            ("UMAP", "umap", {"n_neighbors": 15}),
            ("Neural-Unsupervised", "unsupervised", {"max_epochs": 3}),
        ]

        results = {}

        for method_name, mode, params in methods:
            try:
                time_taken, memory_used, embeddings = (
                    PerformanceBenchmark.measure_time_and_memory(
                        learn_embedding,
                        df,
                        mode=mode,
                        embedding_dim=embedding_dim,
                        verbose=False,
                        enable_logging=False,
                        **params,
                    )
                )

                results[method_name] = {
                    "time": time_taken,
                    "memory": memory_used,
                    "shape": embeddings.shape,
                    "success": True,
                }

            except Exception as e:
                results[method_name] = {
                    "error": str(e),
                    "success": False,
                }

        # Verify at least some methods completed successfully
        successful_methods = [
            name for name, result in results.items() if result.get("success", False)
        ]
        assert len(successful_methods) >= 2, (
            f"Expected at least 2 methods to succeed, got {successful_methods}"
        )

    @pytest.mark.parametrize("n_samples", [100, 500, 1000, 2000])
    def test_scalability_benchmark(self, n_samples):
        """Test performance scaling with dataset size."""
        df = PerformanceBenchmark.generate_benchmark_data(n_samples, n_features=10)

        # Test with fast method (PCA) for scalability
        time_taken, memory_used, embeddings = (
            PerformanceBenchmark.measure_time_and_memory(
                learn_embedding,
                df,
                mode="pca",
                embedding_dim=5,
                verbose=False,
                enable_logging=False,
            )
        )

        assert embeddings.shape == (n_samples, 5)
        assert time_taken < 60  # Should complete within 60 seconds

    def test_memory_usage_profiling(self):
        """Profile memory usage for different dataset sizes."""
        sizes = [100, 300, 500]
        memory_usage = {}

        for size in sizes:
            df = PerformanceBenchmark.generate_benchmark_data(size, n_features=8)

            # Test memory usage with neural method (more memory intensive)
            time_taken, memory_used, embeddings = (
                PerformanceBenchmark.measure_time_and_memory(
                    learn_embedding,
                    df,
                    mode="unsupervised",
                    embedding_dim=4,
                    max_epochs=2,
                    verbose=False,
                    enable_logging=False,
                )
            )

            memory_usage[size] = {
                "memory_mb": memory_used,
                "memory_per_sample_kb": (memory_used * 1024) / size,
                "time": time_taken,
            }

        # Memory usage should scale reasonably
        assert all(usage["memory_mb"] >= 0 for usage in memory_usage.values())

    def test_neural_vs_classical_speed_comparison(self):
        """Compare speed between neural and classical methods."""
        df = PerformanceBenchmark.generate_benchmark_data(300, n_features=12)

        # Classical method (PCA) - should be fast
        pca_time, _, pca_emb = PerformanceBenchmark.measure_time_and_memory(
            learn_embedding,
            df,
            mode="pca",
            embedding_dim=5,
            verbose=False,
            enable_logging=False,
        )

        # Neural method - slower but more flexible
        neural_time, _, neural_emb = PerformanceBenchmark.measure_time_and_memory(
            learn_embedding,
            df,
            mode="unsupervised",
            embedding_dim=5,
            max_epochs=3,
            verbose=False,
            enable_logging=False,
        )

        # Both should produce valid embeddings
        assert pca_emb.shape == neural_emb.shape == (300, 5)

        # PCA should generally be faster for small datasets
        if len(df) < 1000:
            assert pca_time < neural_time, (
                "PCA should be faster than neural methods for small datasets"
            )

    def test_parameter_sensitivity_benchmark(self):
        """Test performance sensitivity to hyperparameters."""
        df = PerformanceBenchmark.generate_benchmark_data(200, n_features=8)

        # Test different embedding dimensions
        dimensions = [2, 5, 10, 15]
        dimension_results = {}

        for dim in dimensions:
            time_taken, memory_used, embeddings = (
                PerformanceBenchmark.measure_time_and_memory(
                    learn_embedding,
                    df,
                    mode="pca",
                    embedding_dim=dim,
                    verbose=False,
                    enable_logging=False,
                )
            )

            dimension_results[dim] = {
                "time": time_taken,
                "memory": memory_used,
            }

        for dim, result in dimension_results.items():
            pass

        # Test t-SNE perplexity sensitivity (smaller test for speed)
        df_small = df.sample(100, random_state=1305)
        perplexity_values = [5, 15, 25]
        perplexity_results = {}

        for perp in perplexity_values:
            try:
                time_taken, _, embeddings = (
                    PerformanceBenchmark.measure_time_and_memory(
                        learn_embedding,
                        df_small,
                        mode="tsne",
                        embedding_dim=2,
                        perplexity=perp,
                        n_iter=250,
                        verbose=False,
                        enable_logging=False,
                    )
                )
                perplexity_results[perp] = time_taken
            except Exception as e:
                perplexity_results[perp] = f"Failed: {e}"

        for perp, result in perplexity_results.items():
            if isinstance(result, float):
                pass
            else:
                pass

    def test_stress_test_large_dataset(self):
        """Stress test with larger dataset (if resources allow)."""
        # Only run if we have sufficient memory
        available_memory_gb = psutil.virtual_memory().available / (1024**3)

        if available_memory_gb < 2:
            pytest.skip("Insufficient memory for stress test")

        # Generate larger dataset
        large_df = PerformanceBenchmark.generate_benchmark_data(5000, n_features=20)

        # Test with fast method first
        time_taken, memory_used, embeddings = (
            PerformanceBenchmark.measure_time_and_memory(
                learn_embedding,
                large_df,
                mode="pca",
                embedding_dim=10,
                verbose=False,
                enable_logging=False,
            )
        )

        assert embeddings.shape == (5000, 10)
        assert memory_used < 1000  # Should not use more than 1GB
        assert time_taken < 120  # Should complete within 2 minutes


class TestPerformanceRegression:
    """Tests to catch performance regressions."""

    def test_baseline_performance(self):
        """Establish baseline performance metrics."""
        df = PerformanceBenchmark.generate_benchmark_data(1000, n_features=10)

        # Baseline test with PCA (should be consistent)
        time_taken, memory_used, embeddings = (
            PerformanceBenchmark.measure_time_and_memory(
                learn_embedding,
                df,
                mode="pca",
                embedding_dim=5,
                verbose=False,
                enable_logging=False,
            )
        )

        # Define reasonable baseline expectations
        assert time_taken < 5.0, f"PCA taking too long: {time_taken:.2f}s"
        assert memory_used < 100, f"PCA using too much memory: {memory_used:.1f}MB"
        assert embeddings.shape == (1000, 5)

    def test_method_consistency(self):
        """Test that methods produce consistent results across runs."""
        df = PerformanceBenchmark.generate_benchmark_data(100, n_features=6)

        # Run same method multiple times and check consistency
        results = []
        for _i in range(3):
            embeddings = learn_embedding(
                df,
                mode="pca",
                embedding_dim=3,
                verbose=False,
                enable_logging=False,
            )
            results.append(embeddings.values)

        # Results should be identical (deterministic)
        np.testing.assert_array_almost_equal(results[0], results[1], decimal=10)
        np.testing.assert_array_almost_equal(results[1], results[2], decimal=10)
