"""
Example usage of the Row2Vec library.
"""

import pandas as pd

from row2vec import generate_synthetic_data, learn_embedding

# ------------------------
# Example usage
# ------------------------

if __name__ == "__main__":
    # 1. Generate synthetic data
    df = generate_synthetic_data(1000)

    # 2. Learn target-based embeddings for the 'Country' column
    print("Target-based embedding (Country):")
    target_emb = learn_embedding(
        df,
        mode="target",
        reference_column="Country",
        embedding_dim=2,
        verbose=True,
    )

    # Map category codes back to country names for inspection
    code_to_country = dict(enumerate(df["Country"].astype("category").cat.categories))
    target_emb = target_emb.reset_index().rename(columns={"category": "CountryCode"})
    target_emb["Country"] = target_emb["CountryCode"].map(code_to_country)
    target_emb = target_emb.drop(columns=["CountryCode"])
    print(target_emb.to_string(index=False))
    print("-" * 50)

    # 3. Learn unsupervised embeddings for each row
    print("\nUnsupervised row embedding:")
    unsupervised_emb = learn_embedding(
        df,
        mode="unsupervised",
        embedding_dim=5,
        verbose=True,
    )
    print(unsupervised_emb.head().to_string(index=False))
    print("-" * 50)

    # 4. Merge and display for comparison
    print("\nMerged embeddings (averaged by country):")
    # Add country back for grouping the unsupervised embeddings
    unsupervised_emb["Country"] = df["Country"].values
    unsup_avg = unsupervised_emb.groupby("Country").mean().reset_index()

    # Merge both embeddings on country
    merged = pd.merge(
        target_emb,
        unsup_avg,
        on="Country",
        suffixes=("_target", "_unsup"),
    )

    # Reorder columns for clarity
    merged = merged[["Country"] + [col for col in merged.columns if col != "Country"]]
    print(merged.to_string(index=False))
