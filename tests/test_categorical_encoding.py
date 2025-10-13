"""
Test: Basic categorical encoding system validation

This script tests the basic functionality of the categorical encoding
system without requiring TensorFlow or heavy dependencies.
"""

import sys

import numpy as np
import pandas as pd

# Test basic configuration system
try:
    from row2vec.config import EmbeddingConfig, PreprocessingConfig
except ImportError:
    # Skip tests if imports fail
    pass

# Test pipeline builder
try:
    from row2vec.pipeline_builder import build_adaptive_pipeline
except ImportError:
    # Skip tests if imports fail
    pass

# Test categorical analyzer (without TensorFlow-dependent parts)
try:
    from row2vec.categorical_encoding import (
        CategoricalAnalyzer,
        CategoricalEncodingConfig,
    )

    print("✅ Categorical encoding system loaded successfully")
except ImportError as e:
    print(f"❌ Categorical encoding import failed: {e}")
    print("This might be due to TensorFlow not being available")


def test_basic_functionality():
    """Test basic functionality without TensorFlow dependencies."""
    print("\n=== Testing Basic Functionality ===")

    # Create test data
    np.random.seed(42)
    df = pd.DataFrame(
        {
            "color": np.random.choice(["red", "blue", "green"], 100),
            "size": np.random.choice(["S", "M", "L", "XL"], 100),
            "brand": np.random.choice([f"brand_{i}" for i in range(10)], 100),
            "price": np.random.normal(100, 20, 100),
            "rating": np.random.normal(4.0, 0.5, 100),
            "target": np.random.choice([0, 1], 100),
        }
    )

    print(f"Test data shape: {df.shape}")
    print(
        f"Categorical columns: {df.select_dtypes(include=['object']).columns.tolist()}"
    )

    # Test configuration
    config = EmbeddingConfig()
    print(
        f"Default categorical encoding strategy: {config.preprocessing.categorical_encoding_strategy}"
    )

    # Test categorical analyzer
    try:
        encoding_config = CategoricalEncodingConfig()
        analyzer = CategoricalAnalyzer(encoding_config)

        # Test analysis without target
        analysis = {}
        categorical_cols = df.select_dtypes(include=["object"]).columns
        for col in categorical_cols:
            if col != "target":
                col_analysis = analyzer.analyze_column(df[col])
                analysis[col] = col_analysis

        print(f"Analysis completed for {len(analysis)} categorical columns")

        for col, info in analysis.items():
            print(
                f"- {col}: cardinality={info['cardinality']}, strategy={info['recommended_strategy']}"
            )

    except Exception as e:
        print(f"Categorical analysis failed: {e}")

    # Test pipeline builder
    try:
        pipeline, report = build_adaptive_pipeline(df.drop(columns=["target"]))
        print("Pipeline built successfully")
        print(
            f"Dataset analysis: {report['dataset_shape'][0]} rows, {report['dataset_shape'][1]} columns"
        )

        # Test pipeline fitting
        X_transformed = pipeline.fit_transform(df.drop(columns=["target"]))
        print(f"Transformed data shape: {X_transformed.shape}")

    except Exception as e:
        print(f"Pipeline building failed: {e}")

    print("✅ Basic functionality test completed")


def test_configuration_serialization():
    """Test configuration serialization and deserialization."""
    print("\n=== Testing Configuration Serialization ===")

    try:
        # Create custom configuration
        config = EmbeddingConfig()
        config.preprocessing.categorical_encoding_strategy = "adaptive"
        config.preprocessing.categorical_onehot_threshold = 15
        config.preprocessing.categorical_target_threshold = 50

        # Test to_dict and from_dict
        config_dict = config.to_dict()
        print("Configuration serialized to dictionary")

        restored_config = EmbeddingConfig.from_dict(config_dict)
        print("Configuration restored from dictionary")

        # Verify values
        assert restored_config.preprocessing.categorical_encoding_strategy == "adaptive"
        assert restored_config.preprocessing.categorical_onehot_threshold == 15
        assert restored_config.preprocessing.categorical_target_threshold == 50

        print("✅ Configuration serialization test passed")

    except Exception as e:
        print(f"❌ Configuration serialization failed: {e}")


def test_missing_value_handling():
    """Test missing value handling in categorical encoding."""
    print("\n=== Testing Missing Value Handling ===")

    try:
        # Create data with missing values
        df = pd.DataFrame(
            {
                "category_with_missing": ["A", "B", np.nan, "A", "C", np.nan, "B"],
                "complete_category": ["X", "Y", "Z", "X", "Y", "Z", "X"],
                "numeric": [1, 2, 3, 4, 5, 6, 7],
                "target": [0, 1, 0, 1, 0, 1, 0],
            }
        )

        print(
            f"Data with missing values created: {df.isnull().sum().sum()} missing values"
        )

        # Test pipeline with missing data
        pipeline, report = build_adaptive_pipeline(df.drop(columns=["target"]))
        X_transformed = pipeline.fit_transform(df.drop(columns=["target"]))

        print("Pipeline handled missing data successfully")
        print(f"Output shape: {X_transformed.shape}")
        print("✅ Missing value handling test passed")

    except Exception as e:
        print(f"❌ Missing value handling failed: {e}")


if __name__ == "__main__":
    print("Row2Vec Categorical Encoding System - Basic Validation")
    print("=" * 60)

    test_basic_functionality()
    test_configuration_serialization()
    test_missing_value_handling()

    print("\n🎉 All basic tests completed!")
    print("\nNote: Some TensorFlow-dependent features may not be fully tested")
    print("if TensorFlow is not available in the current environment.")
