"""
Demo of Classical ML Methods in Row2Vec

This script demonstrates the new PCA, t-SNE, and UMAP embedding methods
that have been added as alternatives to the neural network approaches.
"""

from row2vec import generate_synthetic_data, learn_embedding


def demo_classical_methods():
    """Demonstrate all classical ML methods with a synthetic dataset."""

    print("🚀 Row2Vec Classical ML Methods Demo\n")
    print("=" * 50)

    # Generate synthetic data
    print("📊 Generating synthetic dataset...")
    df = generate_synthetic_data(num_records=150)
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print("\nSample data:")
    print(df.head())
    print()

    embedding_dim = 3

    # Test each classical method
    methods = [
        ("PCA", "pca", {}),
        ("t-SNE", "tsne", {"perplexity": 10, "n_iter": 300}),
        ("UMAP", "umap", {"n_neighbors": 10, "min_dist": 0.1}),
    ]

    results = {}

    for method_name, mode, params in methods:
        print(f"🔬 Testing {method_name}...")
        try:
            embeddings = learn_embedding(
                df,
                mode=mode,
                embedding_dim=embedding_dim,
                verbose=False,
                enable_logging=True,
                log_level="INFO",
                **params,
            )

            results[method_name] = embeddings

            print(f"✅ {method_name} completed successfully!")
            print(f"   Output shape: {embeddings.shape}")
            print(
                f"   Embedding range: [{embeddings.values.min():.3f}, {embeddings.values.max():.3f}]",
            )
            print(f"   Column names: {list(embeddings.columns)}")
            print()

        except Exception as e:
            print(f"❌ {method_name} failed: {e}")
            print()

    # Compare embeddings
    print("📈 Embedding Comparison:")
    print("-" * 30)
    for method_name, embeddings in results.items():
        variance = embeddings.var().sum()
        print(f"{method_name:8} | Variance: {variance:.3f} | Shape: {embeddings.shape}")

    print()

    # Test with different scaling methods
    print("🎛️  Testing scaling options with PCA...")
    scaling_methods = ["none", "minmax", "standard", "l2"]

    for scale_method in scaling_methods:
        try:
            scaled_emb = learn_embedding(
                df,
                mode="pca",
                embedding_dim=2,
                scale_method=scale_method,
                scale_range=(0, 1) if scale_method == "minmax" else None,
                verbose=False,
                enable_logging=False,
            )

            print(
                f"   {scale_method:8} scaling: range [{scaled_emb.values.min():.3f}, {scaled_emb.values.max():.3f}]",
            )

        except Exception as e:
            print(f"   {scale_method:8} scaling: failed - {e}")

    print()

    # Test parameter validation
    print("⚠️  Testing parameter validation...")
    validation_tests = [
        ("t-SNE with invalid perplexity", "tsne", {"perplexity": -1}),
        ("t-SNE with too large perplexity", "tsne", {"perplexity": 100}),
        ("UMAP with invalid n_neighbors", "umap", {"n_neighbors": 0}),
        ("UMAP with too large n_neighbors", "umap", {"n_neighbors": 200}),
    ]

    for test_name, mode, params in validation_tests:
        try:
            learn_embedding(
                df,
                mode=mode,
                embedding_dim=2,
                verbose=False,
                enable_logging=False,
                **params,
            )
            print(f"   {test_name}: ❌ Should have failed!")
        except ValueError:
            print(f"   {test_name}: ✅ Correctly rejected")
        except Exception as e:
            print(f"   {test_name}: ⚠️  Unexpected error: {e}")

    print()
    print("🎉 Demo completed successfully!")
    print("\nKey Features Demonstrated:")
    print("✅ PCA, t-SNE, and UMAP embedding methods")
    print("✅ Consistent interface with neural network methods")
    print("✅ Comprehensive parameter validation")
    print("✅ Multiple scaling options")
    print("✅ Structured logging integration")
    print("✅ Error handling and user-friendly messages")


if __name__ == "__main__":
    demo_classical_methods()
