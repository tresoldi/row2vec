"""
Examples demonstrating the modernized Row2Vec config-based API.

This example shows how to use the new configuration system for more
organized and extensible embedding learning.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Import new config-based API
from row2vec import (
    # New config-based API (recommended)
    learn_embedding_v2,
    EmbeddingConfig,
    NeuralConfig,
    ClassicalConfig,
    ContrastiveConfig,
    ScalingConfig,
    LoggingConfig,
    # Convenience functions
    learn_embedding_unsupervised,
    learn_embedding_target,
    learn_embedding_contrastive,
    learn_embedding_classical,
    # Utilities
    generate_synthetic_data
)


def main():
    print("🚀 Row2Vec Config-Based API Examples")
    print("=" * 50)
    
    # Generate sample data
    df = generate_synthetic_data(200, seed=42)
    print(f"📊 Generated dataset: {df.shape}")
    print(f"   Columns: {list(df.columns)}")
    print()
    
    # Example 1: Basic usage with defaults
    print("📌 Example 1: Basic usage with defaults")
    embeddings = learn_embedding_v2(df)
    print(f"   Embeddings shape: {embeddings.shape}")
    print(f"   Method: Default (unsupervised autoencoder)")
    print()
    
    # Example 2: Using custom config object
    print("📌 Example 2: Custom configuration object")
    config = EmbeddingConfig(
        embedding_dim=8,  # Reduced to be smaller than features
        mode="unsupervised",
        seed=42,
        neural=NeuralConfig(
            max_epochs=30,
            batch_size=32,
            dropout_rate=0.3,
            hidden_units=256
        ),
        scaling=ScalingConfig(
            method="standard"
        ),
        logging=LoggingConfig(
            level="INFO",
            enabled=True
        )
    )
    embeddings = learn_embedding_v2(df, config)
    print(f"   Embeddings shape: {embeddings.shape}")
    print(f"   Custom neural config applied")
    print()
    
    # Example 3: Quick parameter overrides
    print("📌 Example 3: Quick parameter overrides")
    embeddings = learn_embedding_v2(
        df, 
        embedding_dim=8, 
        mode="pca",
        seed=42
    )
    print(f"   Embeddings shape: {embeddings.shape}")
    print(f"   Method: PCA with quick overrides")
    print()
    
    # Example 4: Nested parameter overrides
    print("📌 Example 4: Nested parameter overrides")
    embeddings = learn_embedding_v2(
        df,
        embedding_dim=6,
        **{
            "neural.max_epochs": 20,
            "neural.batch_size": 64,
            "scaling.method": "minmax"
        }
    )
    print(f"   Embeddings shape: {embeddings.shape}")
    print(f"   Method: Unsupervised with nested config overrides")
    print()
    
    # Example 5: Target-based embeddings (supervised)
    print("📌 Example 5: Target-based embeddings")
    df_with_target = df.copy()
    df_with_target['category'] = np.random.choice(['A', 'B', 'C', 'D'], len(df))
    
    target_embeddings = learn_embedding_target(
        df_with_target, 
        reference_column='category',
        embedding_dim=10
    )
    print(f"   Target embeddings shape: {target_embeddings.shape}")
    print(f"   Method: Supervised learning on 'category' column")
    print()
    
    # Example 6: Contrastive learning
    print("📌 Example 6: Contrastive learning")
    contrastive_embeddings = learn_embedding_contrastive(
        df,
        embedding_dim=6,  # Reduced size
        loss_type="triplet",
        **{
            "contrastive.auto_pairs": "cluster",
            "contrastive.margin": 2.0,
            "contrastive.negative_samples": 8,
            "neural.max_epochs": 25  # Reduced epochs for demo
        }
    )
    print(f"   Contrastive embeddings shape: {contrastive_embeddings.shape}")
    print(f"   Method: Triplet loss with automatic cluster-based pairs")
    print()
    
    # Example 7: Configuration from YAML file
    print("📌 Example 7: Configuration from YAML file")
    
    # Create a sample config file
    sample_config = EmbeddingConfig(
        embedding_dim=8,  # Reduced size
        mode="contrastive",
        neural=NeuralConfig(max_epochs=25, batch_size=32, dropout_rate=0.25),  # Reduced epochs
        contrastive=ContrastiveConfig(
            loss_type="contrastive",
            auto_pairs="neighbors",
            margin=1.5,
            negative_samples=6
        ),
        scaling=ScalingConfig(method="l2")
    )
    
    # Save to file
    config_path = "temp_config.yaml"
    sample_config.to_yaml(config_path)
    print(f"   Created config file: {config_path}")
    
    # Load and use config
    loaded_config = EmbeddingConfig.from_yaml(config_path)
    yaml_embeddings = learn_embedding_v2(df, loaded_config)
    print(f"   YAML config embeddings shape: {yaml_embeddings.shape}")
    print(f"   Method: Loaded from YAML configuration")
    
    # Clean up
    Path(config_path).unlink()
    print()
    
    # Example 8: Classical ML methods
    print("📌 Example 8: Classical ML methods")
    
    # PCA
    pca_embeddings = learn_embedding_classical(df, method="pca", embedding_dim=5)
    print(f"   PCA embeddings shape: {pca_embeddings.shape}")
    
    # t-SNE (with custom parameters)
    tsne_embeddings = learn_embedding_v2(
        df,
        mode="tsne",
        embedding_dim=2,  # Good for visualization
        **{
            "classical.perplexity": 20.0,
            "classical.n_iter": 500
        }
    )
    print(f"   t-SNE embeddings shape: {tsne_embeddings.shape}")
    
    # UMAP
    umap_embeddings = learn_embedding_classical(
        df, 
        method="umap", 
        embedding_dim=3,
        **{
            "classical.n_neighbors": 20,
            "classical.min_dist": 0.05
        }
    )
    print(f"   UMAP embeddings shape: {umap_embeddings.shape}")
    print()
    
    # Example 9: Convenience functions comparison
    print("📌 Example 9: Convenience functions comparison")
    
    print("   Comparing different methods on same data:")
    methods = [
        ("Unsupervised", lambda: learn_embedding_unsupervised(df, embedding_dim=5)),
        ("PCA", lambda: learn_embedding_classical(df, "pca", embedding_dim=5)),
        ("t-SNE", lambda: learn_embedding_classical(df, "tsne", embedding_dim=5)),
        ("UMAP", lambda: learn_embedding_classical(df, "umap", embedding_dim=5)),
    ]
    
    for name, func in methods:
        embeddings = func()
        variance = np.var(embeddings.values, axis=0).mean()
        print(f"   {name:12}: shape={embeddings.shape}, avg_variance={variance:.4f}")
    
    print()
    print("✅ All examples completed successfully!")
    print("\n💡 Key Benefits of Config-Based API:")
    print("   • Better organization of parameters")
    print("   • Type safety and validation")
    print("   • YAML configuration file support")
    print("   • Easier extension for new features")
    print("   • Clear separation of concerns")
    print("   • Backward compatibility maintained")


if __name__ == "__main__":
    main()
