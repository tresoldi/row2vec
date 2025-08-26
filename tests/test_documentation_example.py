#!/usr/bin/env python3
"""
Test script to verify the contrastive learning examples from documentation work correctly.
"""

import pandas as pd

from row2vec import learn_embedding


def test_documentation_examples():
    """Test the examples provided in the documentation."""

    print("🧪 Testing Documentation Examples")
    print("=" * 50)

    # Create sample data similar to documentation (larger dataset)
    df = pd.DataFrame(
        {
            "feature1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "feature2": [10, 20, 15, 25, 12, 22, 30, 35, 18, 28],
            "category": ["A", "A", "B", "B", "A", "B", "A", "B", "A", "B"],
        }
    )

    print(f"Sample data shape: {df.shape}")
    print("Sample data:")
    print(df)
    print()

    # Test 1: Manual similarity pairs
    print("=== Test 1: Manual similarity pairs ===")
    similar_pairs = [(0, 1), (2, 3), (4, 6), (5, 7)]
    dissimilar_pairs = [(0, 2), (1, 3), (4, 5), (6, 9)]

    try:
        embeddings = learn_embedding(
            df,
            mode="contrastive",
            embedding_dim=3,
            similar_pairs=similar_pairs,
            dissimilar_pairs=dissimilar_pairs,
            contrastive_loss="triplet",
            max_epochs=3,
            batch_size=4,  # Smaller batch size for small dataset
            verbose=False,
        )
        print(f"✅ Manual pairs: Embeddings shape {embeddings.shape}")
    except Exception as e:
        print(f"❌ Manual pairs failed: {e}")
        return False

    # Test 2: Automatic categorical pairs
    print("\n=== Test 2: Automatic categorical pairs ===")
    try:
        embeddings = learn_embedding(
            df,
            mode="contrastive",
            embedding_dim=3,
            auto_pairs="categorical",
            reference_column="category",
            contrastive_loss="triplet",
            max_epochs=3,
            batch_size=4,
            verbose=False,
        )
        print(f"✅ Categorical pairs: Embeddings shape {embeddings.shape}")
    except Exception as e:
        print(f"❌ Categorical pairs failed: {e}")
        return False

    # Test 3: Automatic clustering pairs
    print("\n=== Test 3: Automatic clustering pairs ===")
    try:
        embeddings = learn_embedding(
            df,
            mode="contrastive",
            embedding_dim=3,
            auto_pairs="cluster",
            contrastive_loss="contrastive",
            margin=1.0,
            max_epochs=3,
            batch_size=4,
            verbose=False,
        )
        print(f"✅ Clustering pairs: Embeddings shape {embeddings.shape}")
    except Exception as e:
        print(f"❌ Clustering pairs failed: {e}")
        return False

    # Test 4: Neighbors pairs
    print("\n=== Test 4: Automatic neighbors pairs ===")
    try:
        embeddings = learn_embedding(
            df,
            mode="contrastive",
            embedding_dim=3,
            auto_pairs="neighbors",
            contrastive_loss="triplet",
            negative_samples=3,
            max_epochs=3,
            batch_size=4,
            verbose=False,
        )
        print(f"✅ Neighbors pairs: Embeddings shape {embeddings.shape}")
    except Exception as e:
        print(f"❌ Neighbors pairs failed: {e}")
        return False

    return True


if __name__ == "__main__":
    success = test_documentation_examples()
    if success:
        print("\n🎉 All documentation examples work correctly!")
    else:
        print("\n💥 Some documentation examples failed!")
