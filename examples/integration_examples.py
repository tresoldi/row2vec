"""
Examples demonstrating scikit-learn and pandas integrations.
"""

import numpy as np
import pandas as pd

from row2vec.utils import generate_synthetic_data


def pandas_integration_examples():
    """Examples of using the pandas integration."""
    print("🐼 Pandas Integration Examples")
    print("=" * 50)

    # Generate sample data
    df = generate_synthetic_data(200, seed=1305)
    print(f"📊 Generated data shape: {df.shape}")

    # Example 1: Basic embedding with PCA
    print("\n1️⃣ Basic PCA embedding:")
    embeddings_pca = df.row2vec.pca(dim=5)
    print(f"   Embeddings shape: {embeddings_pca.shape}")
    print(f"   Columns: {list(embeddings_pca.columns)}")

    # Example 2: t-SNE for visualization
    print("\n2️⃣ t-SNE for 2D visualization:")
    embeddings_tsne = df.row2vec.tsne(dim=2, perplexity=30, n_iter=250)
    print(f"   Embeddings shape: {embeddings_tsne.shape}")

    # Example 3: UMAP embedding
    print("\n3️⃣ UMAP embedding:")
    embeddings_umap = df.row2vec.umap(dim=3, n_neighbors=15)
    print(f"   Embeddings shape: {embeddings_umap.shape}")

    # Example 4: Neural embedding with appropriate batch size
    print("\n4️⃣ Neural embedding (unsupervised):")
    embeddings_neural = df.row2vec.unsupervised(
        dim=8,
        max_epochs=20,
        batch_size=32,  # Appropriate for dataset size
    )
    print(f"   Embeddings shape: {embeddings_neural.shape}")

    # Example 5: Compare multiple methods
    print("\n5️⃣ Comparing multiple methods:")
    comparison = df.row2vec.compare_methods(
        dim=3,
        methods=["pca", "umap"],  # Fast methods for comparison
    )
    for method, embeddings in comparison.items():
        print(f"   {method}: {embeddings.shape}")

    return embeddings_pca


def sklearn_integration_examples():
    """Examples of using the scikit-learn integration."""
    print("\n🧬 Scikit-learn Integration Examples")
    print("=" * 50)

    # Skip if sklearn integration not available
    try:
        from row2vec import Row2VecClassifier, Row2VecTransformer
    except ImportError:
        print("❌ Scikit-learn integration not available")
        return None

    # Generate sample data
    df = generate_synthetic_data(200, seed=1305)
    print(f"📊 Generated data shape: {df.shape}")

    # Example 1: Basic transformer usage
    print("\n1️⃣ Basic Row2VecTransformer:")
    transformer = Row2VecTransformer(
        embedding_dim=5,
        mode="pca",
    )

    X_embedded = transformer.fit_transform(df)
    print(f"   Embeddings shape: {X_embedded.shape}")
    print(f"   Embeddings type: {type(X_embedded)}")

    # Example 2: Neural transformer with proper batch size
    print("\n2️⃣ Neural embedding transformer:")
    neural_transformer = Row2VecTransformer(
        embedding_dim=6,
        mode="unsupervised",
        max_epochs=15,
        batch_size=32,
    )

    X_neural = neural_transformer.fit_transform(df)
    print(f"   Neural embeddings shape: {X_neural.shape}")

    # Example 3: Using in a pipeline (with proper numeric data)
    print("\n3️⃣ Pipeline with clustering:")
    try:
        from sklearn.cluster import KMeans
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        # Create numeric-only data for sklearn pipeline
        numeric_df = pd.DataFrame(
            {
                "feature1": np.random.normal(0, 1, 200),
                "feature2": np.random.normal(5, 2, 200),
                "feature3": np.random.exponential(1, 200),
                "feature4": np.random.uniform(-1, 1, 200),
            }
        )

        pipeline = Pipeline(
            [
                ("scale", StandardScaler()),
                ("embed", Row2VecTransformer(embedding_dim=4, mode="pca")),
                ("cluster", KMeans(n_clusters=3, random_state=1305)),
            ]
        )

        cluster_labels = pipeline.fit_predict(numeric_df.values)
        print(f"   Cluster labels shape: {cluster_labels.shape}")
        print(f"   Unique clusters: {len(np.unique(cluster_labels))}")

    except ImportError:
        print("   ⚠️ sklearn.cluster not available")

    # Example 4: Classification pipeline (with proper data handling)
    print("\n4️⃣ Classification with Row2VecClassifier:")

    # Use the Sales column as the only numeric feature and create a binary target
    numeric_features = df[["Sales"]].copy()

    # Add some derived numeric features
    numeric_features["sales_squared"] = numeric_features["Sales"] ** 2
    numeric_features["sales_log"] = np.log1p(numeric_features["Sales"])
    numeric_features["sales_normalized"] = (
        numeric_features["Sales"] - numeric_features["Sales"].mean()
    ) / numeric_features["Sales"].std()

    # Create a meaningful binary target based on sales
    target = (df["Sales"] > df["Sales"].median()).astype(int)

    X = numeric_features
    y = target

    try:
        from sklearn.model_selection import train_test_split

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.3,
            random_state=1305,
        )

        classifier = Row2VecClassifier(
            embedding_dim=3,  # Smaller than input features (4)
            max_epochs=10,
            batch_size=min(32, len(X_train) // 2),  # Adaptive batch size
        )

        classifier.fit(X_train, y_train)
        predictions = classifier.predict(X_test)
        probabilities = classifier.predict_proba(X_test)

        print(f"   Test predictions shape: {predictions.shape}")
        print(f"   Probabilities shape: {probabilities.shape}")
        print(f"   Unique predicted classes: {np.unique(predictions)}")
        print(f"   Actual class distribution: {np.bincount(y_test)}")

    except ImportError:
        print("   ⚠️ sklearn.model_selection not available")

    return X_embedded


def advanced_integration_examples():
    """Advanced examples combining both integrations."""
    print("\n🚀 Advanced Integration Examples")
    print("=" * 50)

    # Generate data
    df = generate_synthetic_data(150, seed=1305)

    # Example 1: Consistency check
    print("\n1️⃣ Consistency between pandas and sklearn interfaces:")

    # Pandas approach
    pandas_result = df.row2vec.pca(dim=4)

    # Sklearn approach (if available)
    try:
        from row2vec import Row2VecTransformer

        sklearn_transformer = Row2VecTransformer(embedding_dim=4, mode="pca")
        sklearn_result = sklearn_transformer.fit_transform(df)

        # Check consistency
        pandas_values = pandas_result.values
        diff = np.abs(pandas_values - sklearn_result)
        max_diff = np.max(diff)

        print(f"   Pandas result shape: {pandas_result.shape}")
        print(f"   Sklearn result shape: {sklearn_result.shape}")
        print(f"   Maximum difference: {max_diff:.2e}")
        print(f"   Results are {'✅ consistent' if max_diff < 1e-10 else '❌ inconsistent'}")

    except ImportError:
        print("   ⚠️ Sklearn integration not available for comparison")

    # Example 2: Workflow with pandas preprocessing and sklearn pipeline
    print("\n2️⃣ Hybrid workflow (pandas + sklearn):")

    # Step 1: Data preprocessing with pandas (convert to numeric features)
    df_processed = pd.DataFrame(
        {
            "sales": df["Sales"],
            "country_usa": (df["Country"] == "USA").astype(int),
            "country_canada": (df["Country"] == "Canada").astype(int),
            "product_a": (df["Product"] == "A").astype(int),
            "product_b": (df["Product"] == "B").astype(int),
            "sales_log": np.log1p(df["Sales"]),
            "sales_squared": df["Sales"] ** 2,
        }
    )

    print(f"   After preprocessing: {df_processed.shape}")

    # Step 2: Quick exploration with pandas accessor
    quick_embed = df_processed.row2vec.pca(dim=2)
    print(f"   Quick PCA for exploration: {quick_embed.shape}")

    # Step 3: Production pipeline with sklearn (if available)
    try:
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        from row2vec import Row2VecTransformer

        production_pipeline = Pipeline(
            [
                ("standardize", StandardScaler()),
                (
                    "embed",
                    Row2VecTransformer(
                        embedding_dim=5,  # Smaller than input features (7)
                        mode="unsupervised",
                        max_epochs=10,
                        batch_size=min(32, len(df_processed) // 2),
                    ),
                ),
            ]
        )

        final_embeddings = production_pipeline.fit_transform(df_processed.values)
        print(f"   Final production embeddings: {final_embeddings.shape}")

    except ImportError:
        print("   ⚠️ Sklearn pipeline not available")


def main():
    """Run all integration examples."""
    print("🎯 Row2Vec Integration Examples")
    print("=" * 70)

    try:
        # Pandas examples
        pandas_integration_examples()

        # Sklearn examples
        sklearn_integration_examples()

        # Advanced examples
        advanced_integration_examples()

        print("\n🎉 All examples completed successfully!")

    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
