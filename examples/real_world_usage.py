"""
Real-world usage example for the Row2Vec library.

This script demonstrates how to generate embeddings from one of three
real-world datasets: Adult, Titanic, or Ames Housing.

Usage:
    python examples/real_world_usage.py --dataset adult
    python examples/real_world_usage.py --dataset titanic
    python examples/real_world_usage.py --dataset ames
"""

import argparse
import os

import pandas as pd

from row2vec import learn_embedding


def main(args):
    """
    Main function to load data, run embedding models, and print results.
    """
    # --- 1. Load the selected dataset ---
    dataset_name = args.dataset
    data_path = os.path.join(
        os.path.dirname(__file__), "..", "data", f"{dataset_name}.csv"
    )

    if not os.path.exists(data_path):
        print(f"Error: Dataset file not found at {data_path}")
        return

    df = pd.read_csv(data_path).dropna()

    # --- 2. Define dataset-specific configurations ---
    if dataset_name == "adult":
        target_column = "income"
        # Drop columns that are not features
        df_features = df.drop(columns=["fnlwgt", "education-num"])
    elif dataset_name == "titanic":
        target_column = "Survived"
        # Drop columns that are not features
        df_features = df.drop(columns=["Name"])
    elif dataset_name == "ames":
        # The last column is the target
        target_column = df.columns[-1]
        df_features = df.copy()
    else:
        print(f"Error: Unknown dataset '{dataset_name}'")
        return

    print(f"--- Running Row2Vec on the '{dataset_name}' dataset ---")
    print(f"Dataset shape: {df.shape}")
    print(f"Target column for supervised mode: '{target_column}'\n")

    # --- 3. Generate unsupervised embeddings for each row ---
    print("1. Unsupervised (Autoencoder) Embeddings:")
    try:
        unsupervised_emb = learn_embedding(
            df_features,
            mode="unsupervised",
            embedding_dim=10,
            max_epochs=10,  # Keep it fast for the example
            verbose=False,
        )
        print("Generated unsupervised embeddings with shape:", unsupervised_emb.shape)
        print(unsupervised_emb.head().to_string(index=False))
    except Exception as e:
        print(f"Could not generate unsupervised embeddings. Error: {e}")

    print("\n" + "-" * 50 + "\n")

    # --- 4. Generate target-based embeddings for the reference column ---
    print(f"2. Target-Based ('{target_column}') Embeddings:")
    try:
        target_emb = learn_embedding(
            df_features,
            mode="target",
            reference_column=target_column,
            embedding_dim=5,
            max_epochs=10,  # Keep it fast for the example
            verbose=False,
        )
        print(f"Generated embeddings for each unique category in '{target_column}':")
        print(target_emb.to_string())
    except Exception as e:
        print(f"Could not generate target-based embeddings. Error: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Row2Vec on real-world datasets.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["adult", "titanic", "ames"],
        required=True,
        help="The dataset to use.",
    )

    args = parser.parse_args()
    main(args)
