"""
Row2Vec CLI Demo

This script demonstrates the Row2Vec command-line interface capabilities
with practical examples using real datasets.
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

from row2vec import generate_synthetic_data


def run_command(cmd: list[str], description: str) -> None:
    """Run a CLI command and display results."""
    print(f"\n{'=' * 60}")
    print(f"📋 {description}")
    print(f"{'=' * 60}")
    print(f"Command: {' '.join(cmd)}")
    print("-" * 60)
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        
        if result.stdout:
            print("Output:")
            print(result.stdout)
        
        if result.stderr:
            print("Errors/Warnings:")
            print(result.stderr)
            
        if result.returncode != 0:
            print(f"❌ Command failed with exit code {result.returncode}")
        else:
            print("✅ Command completed successfully")
            
    except subprocess.TimeoutExpired:
        print("⏰ Command timed out")
    except Exception as e:
        print(f"❌ Error running command: {e}")


def main():
    """Run CLI demo with various examples."""
    print("🚀 Row2Vec CLI Demo")
    print("This demo shows the Row2Vec command-line interface capabilities")
    
    # Create temporary directory for demo files
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Generate sample data
        print(f"\n📊 Generating sample data...")
        df = generate_synthetic_data(500, seed=42)
        
        # Save as different formats
        csv_file = temp_path / "sample_data.csv"
        tsv_file = temp_path / "sample_data.tsv"
        df.to_csv(csv_file, index=False)
        df.to_csv(tsv_file, sep='\t', index=False)
        
        print(f"✓ Created sample data: {len(df)} rows, {len(df.columns)} columns")
        print(f"✓ Columns: {list(df.columns)}")
        
        # Demo 1: CLI Help
        run_command(
            ["row2vec", "--help"],
            "Display CLI Help"
        )
        
        # Demo 2: Version
        run_command(
            ["row2vec", "--version"],
            "Show Version Information"
        )
        
        # Demo 3: Data Validation
        run_command(
            ["row2vec", "annotate", 
             "--input", str(csv_file),
             "--output", str(temp_path / "dummy.csv"),
             "--validate-only"],
            "Validate Data Schema Only"
        )
        
        # Demo 4: Quick PCA Embeddings
        run_command(
            ["row2vec", "annotate",
             "--input", str(csv_file),
             "--output", str(temp_path / "pca_embeddings.csv"),
             "--mode", "pca",
             "--dim", "3"],
            "Generate PCA Embeddings (Quick)"
        )
        
        # Demo 5: t-SNE for Visualization
        run_command(
            ["row2vec", "annotate",
             "--input", str(tsv_file),
             "--output", str(temp_path / "tsne_embeddings.csv"),
             "--mode", "tsne",
             "--dim", "2",
             "--n-iter", "250",  # Fast for demo
             "--verbose"],
            "Generate t-SNE Embeddings (2D Visualization) from TSV"
        )
        
        # Demo 6: Target-based Embeddings
        run_command(
            ["row2vec", "annotate",
             "--input", str(csv_file),
             "--output", str(temp_path / "target_embeddings.csv"),
             "--mode", "target",
             "--target-col", "Country",
             "--dim", "4",
             "--epochs", "10"],
            "Generate Target-based Embeddings for Countries"
        )
        
        # Demo 7: Train and Save Model
        model_file = temp_path / "unsupervised_model.py"
        run_command(
            ["row2vec", "train",
             "--input", str(csv_file),
             "--output", str(model_file),
             "--mode", "unsupervised",
             "--dim", "5",
             "--epochs", "15",
             "--batch-size", "32",
             "--dropout", "0.1"],
            "Train and Save Unsupervised Model"
        )
        
        # Demo 8: Use Saved Model for Predictions
        if model_file.exists():
            run_command(
                ["row2vec", "predict",
                 "--input", str(csv_file),
                 "--model", str(model_file),
                 "--output", str(temp_path / "predictions.csv")],
                "Use Saved Model for Predictions"
            )
        
        # Demo 9: UMAP with Custom Parameters
        run_command(
            ["row2vec", "annotate",
             "--input", str(csv_file),
             "--output", str(temp_path / "umap_embeddings.csv"),
             "--mode", "umap",
             "--dim", "3",
             "--n-neighbors", "10",
             "--min-dist", "0.2"],
            "Generate UMAP Embeddings with Custom Parameters"
        )
        
        # Demo 10: Neural Network with Advanced Options
        run_command(
            ["row2vec", "annotate",
             "--input", str(csv_file),
             "--output", str(temp_path / "neural_embeddings.csv"),
             "--mode", "unsupervised",
             "--dim", "8",
             "--epochs", "20",
             "--hidden-units", "64",
             "--dropout", "0.3",
             "--scale-method", "minmax",
             "--scale-range", "0", "1",
             "--seed", "12345"],
            "Neural Network Embeddings with Advanced Options"
        )
        
        # Demo 11: Error Handling - Missing Target Column
        run_command(
            ["row2vec", "annotate",
             "--input", str(csv_file),
             "--output", str(temp_path / "error_test.csv"),
             "--mode", "target"],
            "Error Handling: Missing Target Column"
        )
        
        # Demo 12: Show Generated Files
        print(f"\n📁 Generated Files in Demo:")
        print(f"{'=' * 60}")
        for file_path in sorted(temp_path.glob("*")):
            if file_path.is_file():
                size = file_path.stat().st_size
                print(f"  {file_path.name:<30} ({size:,} bytes)")
        
        print(f"\n🎯 Demo Summary:")
        print(f"{'=' * 60}")
        print(f"✓ Demonstrated all CLI commands: train, predict, annotate")
        print(f"✓ Showed all embedding modes: unsupervised, target, pca, tsne, umap")
        print(f"✓ Tested various file formats: CSV, TSV")
        print(f"✓ Demonstrated parameter customization")
        print(f"✓ Showed error handling and validation")
        print(f"✓ Full workflow: data → model → predictions")
        
        print(f"\n💡 Try these commands yourself:")
        print(f"  row2vec --help")
        print(f"  row2vec train --help")
        print(f"  row2vec annotate --input data.csv --output embeddings.csv --mode pca")


if __name__ == "__main__":
    main()
