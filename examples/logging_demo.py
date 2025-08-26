"""
Example demonstrating the comprehensive logging features of Row2Vec.

This script shows how to use the logging system to track training progress,
debug information, and performance metrics.
"""

import os
import tempfile

import numpy as np
import pandas as pd

from row2vec import generate_synthetic_data, get_logger, learn_embedding


def demo_basic_logging():
    """Demonstrate basic logging functionality."""
    print("=" * 60)
    print("DEMO 1: Basic Logging")
    print("=" * 60)

    # Generate synthetic data
    df = generate_synthetic_data(100)

    # Use logging with INFO level
    result = learn_embedding(
        df,
        embedding_dim=5,
        max_epochs=3,
        batch_size=50,
        enable_logging=True,
        log_level="INFO",
    )

    print(f"✅ Generated embeddings with shape: {result.shape}")
    return result


def demo_debug_logging():
    """Demonstrate debug-level logging."""
    print("\n" + "=" * 60)
    print("DEMO 2: Debug Logging")
    print("=" * 60)

    # Create a dataset with categorical data
    df = pd.DataFrame(
        {
            "age": np.random.randint(18, 80, 100),
            "income": np.random.normal(50000, 15000, 100),
            "category": np.random.choice(["A", "B", "C", "D"], 100),
            "region": np.random.choice(["North", "South", "East", "West"], 100),
        }
    )

    # Use target mode with debug logging
    result = learn_embedding(
        df,
        mode="target",
        reference_column="category",
        embedding_dim=3,
        max_epochs=2,
        batch_size=25,
        enable_logging=True,
        log_level="DEBUG",
    )

    print(f"✅ Generated category embeddings with shape: {result.shape}")
    return result


def demo_file_logging():
    """Demonstrate logging to a file."""
    print("\n" + "=" * 60)
    print("DEMO 3: File Logging")
    print("=" * 60)

    # Create temporary log file
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".log") as f:
        log_file = f.name

    try:
        df = generate_synthetic_data(75)

        print(f"📝 Logging to file: {log_file}")

        result = learn_embedding(
            df,
            embedding_dim=4,
            max_epochs=2,
            batch_size=25,
            enable_logging=True,
            log_level="INFO",
            log_file=log_file,
        )

        # Read and display log file contents
        print("\n📄 Log file contents:")
        print("-" * 40)
        with open(log_file) as f:
            for line_num, line in enumerate(f, 1):
                print(f"{line_num:2d}: {line.rstrip()}")
        print("-" * 40)

        print(f"✅ Generated embeddings with shape: {result.shape}")

    finally:
        # Clean up log file
        if os.path.exists(log_file):
            os.unlink(log_file)

    return result


def demo_performance_warnings():
    """Demonstrate performance warnings."""
    print("\n" + "=" * 60)
    print("DEMO 4: Performance Warnings")
    print("=" * 60)

    # Create a large dataset to trigger warnings
    large_df = generate_synthetic_data(500)

    # Add many columns to trigger high-dimensional warning
    for i in range(50):
        large_df[f"extra_feature_{i}"] = np.random.randn(500)

    print(f"📊 Dataset shape: {large_df.shape}")
    print("🔍 Watch for performance warnings in the logs...")

    result = learn_embedding(
        large_df,
        embedding_dim=30,  # High embedding dimension
        max_epochs=2,
        batch_size=100,
        enable_logging=True,
        log_level="WARNING",
    )

    print(f"✅ Generated embeddings with shape: {result.shape}")
    return result


def demo_custom_logger():
    """Demonstrate using a custom logger."""
    print("\n" + "=" * 60)
    print("DEMO 5: Custom Logger")
    print("=" * 60)

    # Create a custom logger
    custom_logger = get_logger(
        name="my_custom_logger",
        level="INFO",
        include_performance=True,
        include_memory=True,
    )

    # Log some custom messages
    custom_logger.logger.info("🔧 Starting custom analysis")

    df = generate_synthetic_data(60)

    # Manual logging of data analysis
    custom_logger.log_debug_info(
        "Custom data analysis",
        {
            "rows": df.shape[0],
            "columns": df.shape[1],
            "memory_usage_mb": df.memory_usage().sum() / 1024 / 1024,
        },
    )

    result = learn_embedding(
        df,
        embedding_dim=2,
        max_epochs=2,
        batch_size=30,
        enable_logging=True,
        log_level="INFO",
    )

    custom_logger.logger.info("🎯 Custom analysis completed")

    print(f"✅ Generated embeddings with shape: {result.shape}")
    return result


def main():
    """Run all logging demonstrations."""
    print("🚀 Row2Vec Logging System Demonstration")
    print("This demo shows comprehensive logging capabilities:")
    print("• Training progress logging")
    print("• Debug information for troubleshooting")
    print("• Performance metrics logging")
    print("• Memory usage tracking")
    print("• Performance warnings")
    print("• File-based logging")

    try:
        # Run all demos
        demo_basic_logging()
        demo_debug_logging()
        demo_file_logging()
        demo_performance_warnings()
        demo_custom_logger()

        print("\n" + "=" * 60)
        print("🎉 ALL LOGGING DEMOS COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("\n📖 Key Features Demonstrated:")
        print("✅ Structured training progress logging")
        print("✅ Detailed preprocessing information")
        print("✅ Model architecture logging")
        print("✅ Performance metrics and timing")
        print("✅ Memory usage tracking")
        print("✅ Performance warnings")
        print("✅ File-based logging")
        print("✅ Debug information")
        print("✅ Custom logger configuration")

    except Exception as e:
        print(f"❌ Demo failed with error: {e}")
        raise


if __name__ == "__main__":
    main()
