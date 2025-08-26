"""
Example: Automatic Embedding Dimension Selection

This example demonstrates how to use Row2Vec's automatic dimension selection
to find optimal embedding dimensions for different types of datasets.
"""

import traceback

import numpy as np
import pandas as pd

from row2vec import (
    AutoDimensionSelector,
    EmbeddingConfig,
    NeuralConfig,
    auto_select_dimension,
    learn_embedding_v2,
)
from row2vec.utils import generate_synthetic_data


def basic_auto_selection_example():
    """Basic example of automatic dimension selection."""
    print("🎯 Basic Automatic Dimension Selection Example")
    print("=" * 60)

    # Generate synthetic data
    df = generate_synthetic_data(500, seed=1305)
    print(f"Dataset shape: {df.shape}")

    # Automatic dimension selection with default settings
    optimal_dim, metadata = auto_select_dimension(
        df=df,
        config=EmbeddingConfig(mode="pca"),
        verbose=True,
    )

    print(f"\n✅ Recommended embedding dimension: {optimal_dim}")
    print("📊 Method scores:")
    for method, result in metadata["method_results"].items():
        print(
            f"   {method}: dim={result['recommended_dim']}, "
            f"score={result['score']:.3f}",
        )

    return optimal_dim, metadata


def advanced_auto_selection_example():
    """Advanced example with custom selection methods and parameters."""
    print("\n🔬 Advanced Automatic Dimension Selection Example")
    print("=" * 60)

    # Generate larger, more complex dataset
    np.random.seed(42)
    df = pd.DataFrame({
        "feature1": np.random.normal(0, 1, 1000),
        "feature2": np.random.exponential(1, 1000),
        "feature3": np.random.uniform(-1, 1, 1000),
        "feature4": np.random.gamma(2, 2, 1000),
        "feature5": np.random.beta(2, 5, 1000),
        "target": np.random.choice(["A", "B", "C"], 1000),
    })

    print(f"Dataset shape: {df.shape}")

    # Custom dimension selector with specific methods
    selector = AutoDimensionSelector(
        methods=["pca_variance", "performance_based", "clustering_quality"],
        performance_weight=0.5,  # Emphasize performance
        efficiency_weight=0.2,
        intrinsic_weight=0.3,
        max_dimension=15,
        verbose=True,
    )

    # Run selection with target column for supervised evaluation
    optimal_dim, metadata = selector.select_dimension(
        df=df.drop(columns=["target"]),
        config=EmbeddingConfig(mode="unsupervised"),
        target_column="target",
        candidate_dims=[2, 3, 4, 5, 6, 8, 10, 12, 15],
    )

    print(f"\n✅ Recommended embedding dimension: {optimal_dim}")
    print("📊 Dimension scores:")
    for dim, score in metadata["dimension_scores"].items():
        print(f"   Dimension {dim}: {score:.3f}")

    return optimal_dim, metadata


def comparison_example():
    """Compare automatic vs manual dimension selection."""
    print("\n⚖️ Automatic vs Manual Dimension Selection Comparison")
    print("=" * 60)

    # Generate test dataset
    df = generate_synthetic_data(300, seed=1305)

    # Test different manual dimensions
    manual_dims = [2, 4, 8, 16]

    print("Testing manual dimensions...")
    for dim in manual_dims:
        try:
            config = EmbeddingConfig(mode="pca", embedding_dim=dim)
            result = learn_embedding_v2(df, config)
            print(f"   Dimension {dim}: SUCCESS - shape {result.shape}")
        except Exception as e:
            print(f"   Dimension {dim}: FAILED - {e}")

    # Automatic selection
    print("\nAutomatic dimension selection...")
    optimal_dim, metadata = auto_select_dimension(
        df=df,
        config=EmbeddingConfig(mode="pca"),
        methods=["pca_variance", "heuristic_rules"],
        verbose=False,
    )

    print(f"✅ Automatic selection: {optimal_dim}")
    print("📈 Confidence scores:")
    for method, result in metadata["method_results"].items():
        print(f"   {method}: {result['score']:.3f}")

    return optimal_dim


def method_comparison_example():
    """Compare different selection methods."""
    print("\n🔍 Selection Method Comparison")
    print("=" * 60)

    # Generate dataset with known structure
    np.random.seed(42)
    # Create data with clear intrinsic dimensionality
    base_dims = 4
    n_samples = 800

    # Generate base features
    base_features = np.random.randn(n_samples, base_dims)

    # Add noise dimensions
    noise_features = np.random.randn(n_samples, 6) * 0.1

    # Combine features
    all_features = np.column_stack([base_features, noise_features])
    df = pd.DataFrame(all_features, columns=[f"feature_{i}" for i in range(10)])

    print(f"Dataset: {df.shape} (expected intrinsic dim ≈ {base_dims})")

    # Test each method individually
    methods = ["pca_variance", "intrinsic_dim", "clustering_quality", "heuristic_rules"]

    for method in methods:
        try:
            selector = AutoDimensionSelector(
                methods=[method],
                verbose=False,
            )

            optimal_dim, metadata = selector.select_dimension(
                df=df,
                config=EmbeddingConfig(mode="pca"),
            )

            result = metadata["method_results"][method]
            print(f"{method:20}: dim={optimal_dim:2d}, score={result['score']:.3f}")

        except Exception as e:
            print(f"{method:20}: FAILED - {e}")

    # Combined approach
    print("\nCombined approach:")
    optimal_dim, metadata = auto_select_dimension(
        df=df,
        config=EmbeddingConfig(mode="pca"),
        verbose=False,
    )
    print(f"{'Combined':20}: dim={optimal_dim:2d}")


def integration_example():
    """Example of integrating auto-selection with existing workflows."""
    print("\n🔗 Integration with Existing Workflows")
    print("=" * 60)

    # Generate dataset
    df = generate_synthetic_data(400, seed=1305)

    # Step 1: Auto-select dimension
    print("Step 1: Automatic dimension selection...")
    optimal_dim, metadata = auto_select_dimension(
        df=df,
        config=EmbeddingConfig(mode="unsupervised"),
        methods=["pca_variance", "clustering_quality"],
        verbose=False,
    )

    print(f"✅ Selected dimension: {optimal_dim}")

    # Step 2: Use selected dimension in production config
    print("\nStep 2: Creating production configuration...")
    production_config = EmbeddingConfig(
        mode="unsupervised",
        embedding_dim=optimal_dim,
        neural=NeuralConfig(
            max_epochs=50,
            batch_size=32,
            early_stopping=True,
        ),
    )

    # Step 3: Generate final embeddings
    print("Step 3: Generating optimized embeddings...")
    final_embeddings = learn_embedding_v2(df, production_config)

    print(f"✅ Final embeddings: {final_embeddings.shape}")
    print("📊 Embedding statistics:")
    print(f"   Mean: {final_embeddings.mean().mean():.6f}")
    print(f"   Std:  {final_embeddings.std().mean():.6f}")

    return final_embeddings, metadata


def main():
    """Run all automatic dimension selection examples."""
    print("🚀 Row2Vec Automatic Dimension Selection Examples")
    print("=" * 70)

    try:
        # Basic example
        basic_optimal_dim, basic_metadata = basic_auto_selection_example()

        # Advanced example
        advanced_optimal_dim, advanced_metadata = advanced_auto_selection_example()

        # Comparison example
        comparison_optimal_dim = comparison_example()

        # Method comparison
        method_comparison_example()

        # Integration example
        final_embeddings, integration_metadata = integration_example()

        print("\n🎉 All examples completed successfully!")
        print("📊 Summary of recommendations:")
        print(f"   Basic dataset: {basic_optimal_dim}")
        print(f"   Advanced dataset: {advanced_optimal_dim}")
        print(f"   Comparison dataset: {comparison_optimal_dim}")

    except Exception as e:
        print(f"\n❌ Error in examples: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
