#!/usr/bin/env python3
"""
Type checking demonstration for Row2Vec
"""

from typing import TYPE_CHECKING

from row2vec import generate_synthetic_data, learn_embedding

if TYPE_CHECKING:
    import pandas as pd


def main() -> None:
    """Demonstrate type checking works properly."""
    # Generate data with proper type hints
    df: pd.DataFrame = generate_synthetic_data(100)

    # This should work fine
    embeddings: pd.DataFrame = learn_embedding(
        df,
        embedding_dim=5,
        mode="unsupervised",
    )

    # This should also work fine
    target_embeddings: pd.DataFrame = learn_embedding(
        df,
        embedding_dim=3,
        mode="target",
        reference_column="Country",
    )

    print("Type checking passed! All functions work with proper type hints.")
    print(f"Unsupervised embeddings shape: {embeddings.shape}")
    print(f"Target embeddings shape: {target_embeddings.shape}")


if __name__ == "__main__":
    main()
