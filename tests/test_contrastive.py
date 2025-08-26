#!/usr/bin/env python3
"""
Quick test script for contrastive learning mode
"""

import numpy as np
import pandas as pd

from row2vec import learn_embedding


def test_contrastive_basic():
    """Test basic contrastive learning functionality"""
    print("Testing contrastive learning mode...")

    # Create synthetic data
    np.random.seed(42)
    n_samples = 100

    # Create data with clear patterns
    data = {
        "feature1": np.random.randn(n_samples),
        "feature2": np.random.randn(n_samples),
        "category": np.random.choice(["A", "B", "C"], n_samples),
    }

    # Add some structure: make category A have higher feature1 values
    mask_a = data["category"] == "A"
    data["feature1"] = np.where(mask_a, data["feature1"] + 2, data["feature1"])

    df = pd.DataFrame(data)
    print(f"Created dataset with shape: {df.shape}")
    print(f"Categories: {df['category'].value_counts().to_dict()}")

    # Test 1: Manual pairs
    print("\n=== Test 1: Manual similarity pairs ===")
    similar_pairs = [(0, 1), (2, 3), (4, 5)]
    dissimilar_pairs = [(0, 10), (1, 15), (2, 20)]

    try:
        embeddings = learn_embedding(
            df,
            mode="contrastive",
            similar_pairs=similar_pairs,
            dissimilar_pairs=dissimilar_pairs,
            contrastive_loss="contrastive",  # Use simpler contrastive loss first
            embedding_dim=3,
            max_epochs=5,
            verbose=True,
        )
        print(f"✅ Manual pairs test passed! Embeddings shape: {embeddings.shape}")
        print(f"Sample embeddings:\n{embeddings.head()}")
    except Exception as e:
        print(f"❌ Manual pairs test failed: {e}")
        return False

    # Test 2: Auto pairs - cluster
    print("\n=== Test 2: Auto pairs (cluster) ===")
    try:
        embeddings = learn_embedding(
            df,
            mode="contrastive",
            auto_pairs="cluster",
            embedding_dim=2,
            max_epochs=3,
            verbose=True,
        )
        print(
            f"✅ Cluster auto-pairs test passed! Embeddings shape: {embeddings.shape}"
        )
    except Exception as e:
        print(f"❌ Cluster auto-pairs test failed: {e}")
        return False

    # Test 3: Auto pairs - categorical
    print("\n=== Test 3: Auto pairs (categorical) ===")
    try:
        embeddings = learn_embedding(
            df,
            mode="contrastive",
            auto_pairs="categorical",
            embedding_dim=2,
            max_epochs=3,
            verbose=True,
        )
        print(
            f"✅ Categorical auto-pairs test passed! Embeddings shape: {embeddings.shape}"
        )
    except Exception as e:
        print(f"❌ Categorical auto-pairs test failed: {e}")
        return False

    # Test 4: Different loss functions
    print("\n=== Test 4: Different loss functions ===")
    for loss_type in ["triplet", "contrastive"]:
        try:
            embeddings = learn_embedding(
                df,
                mode="contrastive",
                auto_pairs="random",
                contrastive_loss=loss_type,
                embedding_dim=2,
                max_epochs=2,
                verbose=False,
            )
            print(
                f"✅ {loss_type} loss test passed! Embeddings shape: {embeddings.shape}"
            )
        except Exception as e:
            print(f"❌ {loss_type} loss test failed: {e}")
            return False

    print("\n🎉 All contrastive learning tests passed!")
    return True


def test_validation():
    """Test validation of contrastive parameters"""
    print("\n=== Testing parameter validation ===")

    df = pd.DataFrame(
        {
            "x": [1, 2, 3, 4],
            "y": [1, 2, 3, 4],
        }
    )

    # Test invalid contrastive_loss
    try:
        learn_embedding(
            df, mode="contrastive", auto_pairs="random", contrastive_loss="invalid"
        )
        print("❌ Should have failed with invalid contrastive_loss")
        return False
    except ValueError as e:
        print(f"✅ Correctly caught invalid contrastive_loss: {e}")

    # Test invalid auto_pairs
    try:
        learn_embedding(df, mode="contrastive", auto_pairs="invalid")
        print("❌ Should have failed with invalid auto_pairs")
        return False
    except ValueError as e:
        print(f"✅ Correctly caught invalid auto_pairs: {e}")

    # Test missing pairs/auto_pairs
    try:
        learn_embedding(df, mode="contrastive")
        print("❌ Should have failed with no pairs specified")
        return False
    except ValueError as e:
        print(f"✅ Correctly caught missing pairs: {e}")

    print("✅ All validation tests passed!")
    return True


if __name__ == "__main__":
    print("🔥 Testing Row2Vec Contrastive Learning Implementation")
    print("=" * 60)

    success = True
    success &= test_contrastive_basic()
    success &= test_validation()

    if success:
        print("\n🎊 ALL TESTS PASSED! Contrastive learning is working! 🎊")
    else:
        print("\n💥 Some tests failed. Check the implementation.")
