"""
Minimal smoke test for contrastive learning.

This is the smallest input that exercises the contrastive path end to end: two
perfectly correlated columns, explicit similar/dissimilar pairs, and a batch
size smaller than the dataset.
"""

import pandas as pd

from row2vec import learn_embedding


def test_minimal_contrastive_with_manual_pairs() -> None:
    """Contrastive mode runs on a tiny frame with explicit pairs."""
    df = pd.DataFrame(
        {
            "x": [1, 2, 3, 4, 5, 6],
            "y": [1, 2, 3, 4, 5, 6],
        }
    )

    embeddings = learn_embedding(
        df,
        mode="contrastive",
        similar_pairs=[(0, 1), (2, 3)],
        dissimilar_pairs=[(0, 4), (1, 5)],
        contrastive_loss="contrastive",
        embedding_dim=2,
        max_epochs=2,
        batch_size=4,
        verbose=False,
    )

    assert embeddings.shape == (len(df), 2)
    assert not embeddings.isna().to_numpy().any()
