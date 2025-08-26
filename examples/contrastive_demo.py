#!/usr/bin/env python3
"""Demo script showing contrastive learning CLI usage."""

import pandas as pd
import numpy as np
import os
import tempfile
from pathlib import Path

# Create a temporary directory for output files
output_dir = Path(tempfile.mkdtemp(prefix="row2vec_contrastive_"))
print(f"Creating sample files in: {output_dir}")

# Create sample data
np.random.seed(1305)
sample_data = pd.DataFrame({
    'feature1': np.random.rand(100),
    'feature2': np.random.rand(100),
    'feature3': np.random.choice(['A', 'B', 'C'], 100),
    'feature4': np.random.randint(1, 10, 100)
})

# Save sample data
sample_data_path = output_dir / 'sample_data.csv'
sample_data.to_csv(sample_data_path, index=False)

# Create sample similarity pairs (first 10 rows are similar to each other)
similar_pairs = [(i, j) for i in range(5) for j in range(5, 10)]
similar_df = pd.DataFrame(similar_pairs, columns=['row1', 'row2'])
similar_pairs_path = output_dir / 'similar_pairs.csv'
similar_df.to_csv(similar_pairs_path, index=False)

# Create sample dissimilar pairs (first 5 rows are dissimilar to last 5 rows)
dissimilar_pairs = [(i, j) for i in range(5) for j in range(95, 100)]
dissimilar_df = pd.DataFrame(dissimilar_pairs, columns=['row1', 'row2'])
dissimilar_pairs_path = output_dir / 'dissimilar_pairs.csv'
dissimilar_df.to_csv(dissimilar_pairs_path, index=False)

print("\n✅ Sample data and pair files created!")
print("\nExample CLI commands for contrastive learning:")
print("\n1. Using automatic pair generation:")
print(f"   python -m row2vec train --input {sample_data_path} --output {output_dir}/embeddings.csv \\")
print("       --mode contrastive --auto-pairs cluster --contrastive-loss triplet \\")
print("       --negative-samples 5 --margin 1.0")

print("\n2. Using predefined similarity/dissimilarity pairs:")
print(f"   python -m row2vec train --input {sample_data_path} --output {output_dir}/embeddings.csv \\")
print(f"       --mode contrastive --similar-pairs-file {similar_pairs_path} \\")
print(f"       --dissimilar-pairs-file {dissimilar_pairs_path} --contrastive-loss contrastive")

print("\n3. Using both auto-generation and manual pairs:")
print(f"   python -m row2vec train --input {sample_data_path} --output {output_dir}/embeddings.csv \\")
print("       --mode contrastive --auto-pairs neighbors \\")
print(f"       --similar-pairs-file {similar_pairs_path} --negative-samples 3")

print("\n📁 Files created in temporary directory:")
print(f"- {sample_data_path} (100 rows with mixed features)")
print(f"- {similar_pairs_path} (25 similarity pairs)")
print(f"- {dissimilar_pairs_path} (25 dissimilarity pairs)")
print(f"\n💡 Tip: The temporary directory will be preserved for your use.")
print(f"   Clean it up manually when done: rm -rf {output_dir}")
