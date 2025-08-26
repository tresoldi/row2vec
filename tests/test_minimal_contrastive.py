#!/usr/bin/env python3
"""
Minimal test for contrastive learning debugging
"""

import pandas as pd

from row2vec import learn_embedding


def test_minimal():
    """Test minimal contrastive learning"""
    print("Testing minimal contrastive learning...")

    # Create very simple data
    df = pd.DataFrame(
        {
            "x": [1, 2, 3, 4, 5, 6],
            "y": [1, 2, 3, 4, 5, 6],
        }
    )

    # Manual pairs only
    similar_pairs = [(0, 1), (2, 3)]
    dissimilar_pairs = [(0, 4), (1, 5)]

    try:
        embeddings = learn_embedding(
            df,
            mode="contrastive",
            similar_pairs=similar_pairs,
            dissimilar_pairs=dissimilar_pairs,
            contrastive_loss="contrastive",
            embedding_dim=2,
            max_epochs=2,
            batch_size=4,  # Small batch size
            verbose=True,
        )
        print(f"✅ Success! Embeddings shape: {embeddings.shape}")
        print(f"Embeddings:\n{embeddings}")
        return True
    except Exception as e:
        print(f"❌ Failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_minimal()
