"""
Example: Neural Architecture Search with Row2Vec

This example demonstrates how to use Row2Vec's automatic neural architecture search
to find optimal network configurations for embedding generation.
"""

import pandas as pd
import numpy as np
from row2vec import (
    learn_embedding_v2, 
    EmbeddingConfig, 
    NeuralConfig,
    ArchitectureSearchConfig,
    search_architecture
)
from row2vec.utils import generate_synthetic_data


def basic_architecture_search_example():
    """Basic example of automatic architecture search."""
    print("🔍 Basic Neural Architecture Search Example")
    print("=" * 60)
    
    # Generate synthetic data
    df = generate_synthetic_data(400, seed=42)
    # Add more features for architecture search testing
    np.random.seed(42)
    additional_features = pd.DataFrame({
        'feature4': np.random.normal(0, 1, len(df)),
        'feature5': np.random.uniform(-1, 1, len(df)),
        'feature6': np.random.exponential(1, len(df)),
        'feature7': np.random.gamma(2, 2, len(df)),
        'feature8': np.random.beta(2, 5, len(df))
    })
    df = pd.concat([df, additional_features], axis=1)
    print(f"Dataset shape: {df.shape}")
    
    # Basic configuration
    config = EmbeddingConfig(mode="unsupervised", embedding_dim=6)
    
    # Automatic architecture search (simple)
    print("\n🎯 Running automatic architecture search...")
    embeddings = learn_embedding_v2(
        df, 
        config, 
        auto_architecture=True
    )
    
    print(f"✅ Generated embeddings: {embeddings.shape}")
    print(f"📊 Embedding statistics:")
    print(f"   Mean: {embeddings.mean().mean():.6f}")
    print(f"   Std:  {embeddings.std().mean():.6f}")
    print(f"   Range: [{embeddings.min().min():.3f}, {embeddings.max().max():.3f}]")
    
    return embeddings


def advanced_architecture_search_example():
    """Advanced example with custom search configuration."""
    print("\n🔬 Advanced Architecture Search Example")
    print("=" * 60)
    
    # Generate larger, more complex dataset
    np.random.seed(42)
    df = pd.DataFrame({
        'feature1': np.random.normal(0, 1, 600),
        'feature2': np.random.exponential(1, 600),
        'feature3': np.random.uniform(-2, 2, 600),
        'feature4': np.random.gamma(2, 2, 600),
        'feature5': np.random.beta(2, 5, 600),
        'feature6': np.random.lognormal(0, 1, 600),
        'category': np.random.choice(['A', 'B', 'C', 'D'], 600)
    })
    
    # Remove category for embedding (keep for evaluation)
    df_features = df.drop(columns=['category'])
    print(f"Dataset shape: {df_features.shape}")
    
    # Custom search configuration
    search_config = ArchitectureSearchConfig(
        method="random",
        max_trials=25,
        max_time=900,  # 15 minutes
        patience=8,
        
        # Search space customization
        layer_range=(1, 4),
        width_options=[32, 64, 128, 256],
        dropout_options=[0.0, 0.1, 0.2, 0.3],
        activation_options=["relu", "elu", "swish"],
        
        # Evaluation weights
        reconstruction_weight=0.4,
        clustering_weight=0.4,
        efficiency_weight=0.15,
        stability_weight=0.05,
        
        verbose=True
    )
    
    # Base configuration
    base_config = EmbeddingConfig(mode="unsupervised", embedding_dim=8)
    
    # Run architecture search
    print("\n🎯 Running advanced architecture search...")
    embeddings = learn_embedding_v2(
        df_features,
        base_config,
        auto_architecture=True,
        architecture_search_config=search_config
    )
    
    print(f"✅ Generated embeddings: {embeddings.shape}")
    
    return embeddings, df


def direct_search_example():
    """Example using the direct search API for more control."""
    print("\n⚙️ Direct Architecture Search API Example")
    print("=" * 60)
    
    # Generate dataset
    df = generate_synthetic_data(300, seed=42)
    # Add more features for testing
    np.random.seed(42)
    additional_features = pd.DataFrame({
        'feature4': np.random.normal(0, 1, len(df)),
        'feature5': np.random.uniform(-1, 1, len(df))
    })
    df = pd.concat([df, additional_features], axis=1)
    print(f"Dataset shape: {df.shape}")
    
    # Search configuration
    search_config = ArchitectureSearchConfig(
        method="random",
        max_trials=15,
        max_time=600,  # 10 minutes
        patience=5,
        verbose=True,
        return_full_history=True  # Get detailed search history
    )
    
    # Base configuration
    base_config = EmbeddingConfig(mode="unsupervised", embedding_dim=5)
    
    # Direct architecture search
    print("\n🔍 Running direct architecture search...")
    best_architecture, search_result = search_architecture(
        df=df,
        base_config=base_config,
        search_config=search_config
    )
    
    # Display detailed results
    print(f"\n🏆 Search Results")
    print(f"=" * 40)
    
    summary = search_result.summary()
    print(f"📊 Search Summary:")
    print(f"   Trials completed: {summary['trials_completed']}")
    print(f"   Total time: {summary['total_time']:.1f}s")
    print(f"   Best score: {summary['best_score']:.4f}")
    print(f"   Search efficiency: {summary['search_efficiency']:.6f}")
    
    print(f"\n🎯 Best Architecture:")
    print(f"   Layers: {best_architecture['n_layers']}")
    print(f"   Layer widths: {best_architecture['layer_widths']}")
    print(f"   Dropout rate: {best_architecture['dropout_rate']:.3f}")
    print(f"   Activation: {best_architecture['activation']}")
    
    if summary['improvement_over_baseline'] > 0:
        print(f"\n📈 Improvement over baseline: {summary['improvement_over_baseline']:.2%}")
    
    # Show search history
    if len(search_result.search_history) > 0:
        print(f"\n📋 Search History (top 5 trials):")
        sorted_history = sorted(
            search_result.search_history, 
            key=lambda x: x['score'], 
            reverse=True
        )[:5]
        
        for i, trial in enumerate(sorted_history):
            print(f"   {i+1}. Score: {trial['score']:.4f}, "
                  f"Layers: {trial['architecture']['n_layers']}, "
                  f"Widths: {trial['architecture']['layer_widths']}, "
                  f"Dropout: {trial['architecture']['dropout_rate']:.2f}")
    
    # Use best architecture for final embedding
    print(f"\n🚀 Generating final embeddings with best architecture...")
    final_config = EmbeddingConfig(
        mode="unsupervised",
        embedding_dim=5,
        neural=NeuralConfig(
            hidden_units=best_architecture['layer_widths'],
            dropout_rate=best_architecture['dropout_rate'],
            activation=best_architecture['activation'],
            max_epochs=50,
            early_stopping=True
        )
    )
    
    final_embeddings = learn_embedding_v2(df, final_config)
    print(f"✅ Final embeddings: {final_embeddings.shape}")
    
    return best_architecture, search_result, final_embeddings


def grid_search_example():
    """Example using grid search for systematic exploration."""
    print("\n📋 Grid Search Example")
    print("=" * 60)
    
    # Smaller dataset for grid search (it's more expensive)
    df = generate_synthetic_data(200, seed=42)
    # Add one more feature for testing
    np.random.seed(42)
    df['feature4'] = np.random.normal(0, 1, len(df))
    print(f"Dataset shape: {df.shape}")
    
    # Grid search configuration
    search_config = ArchitectureSearchConfig(
        method="grid",
        max_trials=20,  # Limit grid size
        max_time=600,   # 10 minutes
        verbose=True
    )
    
    base_config = EmbeddingConfig(mode="unsupervised", embedding_dim=4)
    
    # Run grid search
    print("\n📊 Running systematic grid search...")
    best_architecture, search_result = search_architecture(
        df=df,
        base_config=base_config,
        search_config=search_config
    )
    
    print(f"\n🎯 Grid Search Results:")
    print(f"   Best architecture: {best_architecture}")
    print(f"   Trials completed: {search_result.trials_completed}")
    print(f"   Best score: {search_result.best_score:.4f}")
    
    return best_architecture


def comparison_example():
    """Compare manual vs automatic architecture selection."""
    print("\n⚖️ Manual vs Automatic Architecture Comparison")
    print("=" * 60)
    
    # Generate test dataset
    df = generate_synthetic_data(350, seed=42)
    # Add more features for comparison testing
    np.random.seed(42)
    additional_features = pd.DataFrame({
        'feature4': np.random.normal(0, 1, len(df)),
        'feature5': np.random.uniform(-1, 1, len(df)),
        'feature6': np.random.exponential(1, len(df))
    })
    df = pd.concat([df, additional_features], axis=1)
    print(f"Dataset shape: {df.shape}")
    
    # Test manual architectures
    manual_configs = [
        {"layers": [64], "dropout": 0.2, "activation": "relu"},
        {"layers": [128, 64], "dropout": 0.3, "activation": "elu"},
        {"layers": [256, 128, 64], "dropout": 0.1, "activation": "swish"},
    ]
    
    print("\n🔧 Testing manual architectures...")
    manual_results = []
    
    for i, arch_config in enumerate(manual_configs):
        try:
            config = EmbeddingConfig(
                mode="unsupervised",
                embedding_dim=6,
                neural=NeuralConfig(
                    hidden_units=arch_config["layers"],
                    dropout_rate=arch_config["dropout"],
                    activation=arch_config["activation"],
                    max_epochs=20  # Quick test
                )
            )
            
            embeddings = learn_embedding_v2(df, config)
            print(f"   Manual {i+1}: {arch_config['layers']} - SUCCESS")
            manual_results.append((arch_config, embeddings.shape))
            
        except Exception as e:
            print(f"   Manual {i+1}: {arch_config['layers']} - FAILED: {e}")
    
    # Automatic architecture search
    print("\n🤖 Running automatic architecture search...")
    search_config = ArchitectureSearchConfig(
        max_trials=15,
        patience=5,
        verbose=False
    )
    
    auto_embeddings = learn_embedding_v2(
        df,
        EmbeddingConfig(mode="unsupervised", embedding_dim=6),
        auto_architecture=True,
        architecture_search_config=search_config
    )
    
    print(f"✅ Automatic search completed: {auto_embeddings.shape}")
    
    print(f"\n📊 Comparison Summary:")
    print(f"   Manual architectures tested: {len(manual_results)}")
    print(f"   Manual success rate: {len(manual_results)/len(manual_configs)*100:.1f}%")
    print(f"   Automatic search: Always finds working architecture")
    print(f"   Recommendation: Use automatic search for optimal results")


def main():
    """Run all architecture search examples."""
    print("🚀 Row2Vec Neural Architecture Search Examples")
    print("=" * 70)
    
    try:
        # Import required classes
        from row2vec import NeuralConfig
        
        # Basic example
        basic_embeddings = basic_architecture_search_example()
        
        # Advanced example
        advanced_embeddings, df_with_category = advanced_architecture_search_example()
        
        # Direct search API
        best_arch, search_result, final_embeddings = direct_search_example()
        
        # Grid search
        grid_best_arch = grid_search_example()
        
        # Comparison
        comparison_example()
        
        print(f"\n🎉 All architecture search examples completed successfully!")
        print(f"📊 Summary:")
        print(f"   Basic search: {basic_embeddings.shape}")
        print(f"   Advanced search: {advanced_embeddings.shape}")
        print(f"   Direct search: {final_embeddings.shape}")
        print(f"   Best architecture found: {best_arch['layer_widths']}")
        
    except Exception as e:
        print(f"\n❌ Error in examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
