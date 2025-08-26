"""
Example: Testing Categorical Encoding Strategies

This example demonstrates the new intelligent categorical encoding
capabilities in Row2Vec with different strategies and configurations.
"""

import pandas as pd
import numpy as np
from row2vec.core import learn_embedding
from row2vec.config import EmbeddingConfig, PreprocessingConfig, CategoricalEncodingConfig
from row2vec.pipeline_builder import build_adaptive_pipeline


def create_sample_data():
    """Create sample data with various categorical features for testing."""
    np.random.seed(42)
    
    n_samples = 1000
    
    # Low cardinality categorical (good for one-hot)
    colors = np.random.choice(['red', 'blue', 'green', 'yellow'], n_samples)
    
    # Medium cardinality categorical (good for target encoding)
    cities = np.random.choice([f'city_{i}' for i in range(20)], n_samples)
    
    # High cardinality categorical (good for entity embeddings)
    product_ids = np.random.choice([f'product_{i}' for i in range(200)], n_samples)
    
    # Numeric features
    age = np.random.normal(35, 10, n_samples)
    income = np.random.normal(50000, 15000, n_samples)
    
    # Target variable (for supervised scenarios)
    # Make it somewhat correlated with features
    target = (
        (colors == 'red').astype(int) * 0.3 +
        (np.array([int(c.split('_')[1]) for c in cities]) > 10).astype(int) * 0.4 +
        age / 100 * 0.2 +
        income / 100000 * 0.1 +
        np.random.normal(0, 0.2, n_samples)
    )
    target = (target > 0.5).astype(int)
    
    df = pd.DataFrame({
        'color': colors,
        'city': cities,
        'product_id': product_ids,
        'age': age,
        'income': income,
        'target': target
    })
    
    return df


def example_adaptive_encoding():
    """Example: Adaptive encoding strategy selection."""
    print("=== Adaptive Encoding Strategy ===")
    
    df = create_sample_data()
    print(f"Dataset shape: {df.shape}")
    print(f"Categorical columns: {df.select_dtypes(include=['object']).columns.tolist()}")
    
    # Use adaptive strategy (default)
    config = EmbeddingConfig()
    
    # Build preprocessing pipeline to see what strategies are selected
    preprocessor, analysis = build_adaptive_pipeline(df, target=df['target'], config=config)
    
    print(f"\nDataset analysis:")
    print(f"- Missing data: {analysis['missing_percentage']:.1f}%")
    print(f"- Memory usage: {analysis['memory_usage_mb']:.1f} MB")
    print(f"- Numeric features: {analysis['numeric_columns']}")
    print(f"- Categorical features: {analysis['categorical_columns']}")
    
    # Generate embeddings
    embeddings = learn_embedding(
        df.drop(columns=['target']), 
        embedding_dim=8, 
        mode="unsupervised",
        config=config,
        verbose=True
    )
    
    print(f"\nEmbeddings shape: {embeddings.shape}")
    print(f"Sample embeddings:\n{embeddings.head()}")


def example_target_encoding():
    """Example: Target encoding for supervised learning."""
    print("\n=== Target Encoding Strategy ===")
    
    df = create_sample_data()
    
    # Configure for target encoding
    config = EmbeddingConfig()
    config.preprocessing.categorical_encoding_strategy = "target"
    config.preprocessing.categorical_target_threshold = 50  # Use target encoding for <50 categories
    
    embeddings = learn_embedding(
        df, 
        embedding_dim=10, 
        mode="target",
        reference_column="target",
        config=config,
        verbose=True
    )
    
    print(f"Target-aware embeddings shape: {embeddings.shape}")
    print(f"Sample embeddings:\n{embeddings.head()}")


def example_entity_embeddings():
    """Example: Entity embeddings for high-cardinality features."""
    print("\n=== Entity Embeddings Strategy ===")
    
    df = create_sample_data()
    
    # Configure for entity embeddings
    config = EmbeddingConfig()
    config.preprocessing.categorical_encoding_strategy = "entity"
    config.preprocessing.categorical_entity_threshold = 500  # Use entity embeddings for <500 categories
    
    # Build pipeline to show entity embedding decisions
    preprocessor, analysis = build_adaptive_pipeline(df, config=config)
    
    embeddings = learn_embedding(
        df.drop(columns=['target']), 
        embedding_dim=12, 
        mode="unsupervised",
        config=config,
        max_epochs=20,  # Fewer epochs for demo
        verbose=True
    )
    
    print(f"Entity embeddings shape: {embeddings.shape}")
    print(f"Sample embeddings:\n{embeddings.head()}")


def example_mixed_strategies():
    """Example: Mixed encoding strategies for different columns."""
    print("\n=== Mixed Encoding Strategies ===")
    
    df = create_sample_data()
    
    # Configure mixed strategies
    config = EmbeddingConfig()
    config.preprocessing.categorical_encoding_strategy = "adaptive"
    config.preprocessing.categorical_onehot_threshold = 5    # Very low for demo
    config.preprocessing.categorical_target_threshold = 15   # Medium threshold
    config.preprocessing.categorical_entity_threshold = 100  # Higher threshold
    
    # This will use:
    # - OneHot for 'color' (4 categories <= 5)
    # - Target encoding for 'city' (20 categories, 5 < 20 <= 15 would be target if supervised)
    # - Entity embeddings for 'product_id' (200 categories > 100)
    
    embeddings = learn_embedding(
        df, 
        embedding_dim=15, 
        mode="target",
        reference_column="target",
        config=config,
        max_epochs=15,
        verbose=True
    )
    
    print(f"Mixed strategy embeddings shape: {embeddings.shape}")
    print(f"Sample embeddings:\n{embeddings.head()}")


def example_custom_configuration():
    """Example: Custom categorical encoding configuration."""
    print("\n=== Custom Configuration ===")
    
    df = create_sample_data()
    
    # Create custom configuration
    categorical_config = CategoricalEncodingConfig(
        encoding_strategy="adaptive",
        onehot_threshold=10,
        target_threshold=30,
        entity_threshold=150,
        correlation_threshold=0.1,
        prefer_speed=True,
        handle_unknown="ignore",
        target_smoothing=1.0,
        target_cv_folds=3
    )
    
    preprocessing_config = PreprocessingConfig(
        categorical_encoding_strategy="adaptive",
        categorical_onehot_threshold=10,
        categorical_target_threshold=30,
        categorical_entity_threshold=150,
        numeric_scaling="robust"
    )
    
    config = EmbeddingConfig(preprocessing=preprocessing_config)
    
    embeddings = learn_embedding(
        df.drop(columns=['target']), 
        embedding_dim=8, 
        mode="unsupervised",
        config=config,
        verbose=True
    )
    
    print(f"Custom config embeddings shape: {embeddings.shape}")
    print(f"Sample embeddings:\n{embeddings.head()}")


if __name__ == "__main__":
    print("Row2Vec Categorical Encoding Examples")
    print("=====================================")
    
    # Run examples
    example_adaptive_encoding()
    example_target_encoding()
    example_entity_embeddings()
    example_mixed_strategies()
    example_custom_configuration()
    
    print("\n✅ All examples completed successfully!")
    print("\nKey features demonstrated:")
    print("- Adaptive strategy selection based on cardinality and target correlation")
    print("- Target encoding with cross-validation and regularization")
    print("- Entity embeddings for high-cardinality features")
    print("- Mixed strategies within the same dataset")
    print("- Custom configuration for expert control")
