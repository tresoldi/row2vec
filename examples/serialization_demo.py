"""
Example: Row2Vec Model Serialization and Persistence

This script demonstrates how to train, save, and load Row2Vec models
using the new serialization functionality.
"""

import tempfile
from pathlib import Path

from row2vec import (
    Row2VecModel,
    Row2VecModelMetadata,
    generate_synthetic_data,
    learn_embedding_with_model,
    load_model,
    save_model,
    train_and_save_model,
)


def main():
    print("🚀 Row2Vec Model Serialization Demo")
    print("=" * 50)

    # Generate sample data
    print("\n1. Generating sample data...")
    data = generate_synthetic_data(200, seed=1305)
    print(f"   Dataset shape: {data.shape}")
    print(f"   Columns: {list(data.columns)}")
    print(f"   Unique countries: {data['Country'].nunique()}")

    # Example 1: Train and save using convenience function
    print("\n2. Training and saving models using convenience function...")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Train unsupervised model
        print("   Training unsupervised autoencoder...")
        unsup_embeddings, unsup_script, unsup_binary = train_and_save_model(
            data,
            tmpdir / "unsupervised_model",
            embedding_dim=5,
            mode="unsupervised",
            max_epochs=10,
            batch_size=32,
            verbose=False,
            enable_logging=False,
        )
        print(f"   ✅ Saved to: {unsup_script}")
        print(f"   📁 Binary file: {unsup_binary}")
        print(f"   🎯 Embeddings shape: {unsup_embeddings.shape}")

        # Train supervised model
        print("   Training supervised (target) model...")
        sup_embeddings, sup_script, sup_binary = train_and_save_model(
            data,
            tmpdir / "supervised_model",
            embedding_dim=3,
            mode="target",
            reference_column="Country",
            max_epochs=8,
            batch_size=32,
            verbose=False,
            enable_logging=False,
        )
        print(f"   ✅ Saved to: {sup_script}")
        print(f"   🎯 Embeddings shape: {sup_embeddings.shape}")

        # Train PCA model
        print("   Training PCA model...")
        pca_embeddings, pca_script, pca_binary = train_and_save_model(
            data,
            tmpdir / "pca_model",
            embedding_dim=4,
            mode="pca",
            verbose=False,
            enable_logging=False,
        )
        print(f"   ✅ Saved to: {pca_script}")
        print(f"   🎯 Embeddings shape: {pca_embeddings.shape}")

        # Example 2: Manual training and saving
        print("\n3. Manual training and saving workflow...")
        print("   Training with learn_embedding_with_model...")

        embeddings, model, preprocessor, metadata = learn_embedding_with_model(
            data,
            embedding_dim=2,
            mode="umap",
            verbose=False,
            enable_logging=False,
        )

        # Create Row2Vec model object
        row2vec_model = Row2VecModel(
            model=model,
            preprocessor=preprocessor,
            metadata=Row2VecModelMetadata.from_dict(metadata),
        )

        # Save manually
        manual_script, manual_binary = save_model(
            row2vec_model,
            tmpdir / "manual_model",
        )
        print(f"   ✅ Manually saved to: {manual_script}")

        # Example 3: Loading and using models
        print("\n4. Loading and using saved models...")

        # Load each model and test prediction
        test_data = generate_synthetic_data(20, seed=999)
        print(f"   Test data shape: {test_data.shape}")

        models_to_test = [
            ("Unsupervised", unsup_script),
            ("Supervised", sup_script),
            ("PCA", pca_script),
            ("UMAP (manual)", manual_script),
        ]

        for model_name, script_path in models_to_test:
            print(f"\n   Testing {model_name} model:")

            # Load model
            loaded_model = load_model(script_path)
            print("     ✅ Model loaded successfully")

            # Show metadata
            metadata = loaded_model.metadata
            print(f"     📊 Mode: {metadata.mode}")
            print(f"     📐 Embedding dim: {metadata.embedding_dim}")
            print(f"     ⏱️  Training time: {metadata.training_time:.2f}s")
            if metadata.final_loss:
                print(f"     📉 Final loss: {metadata.final_loss:.4f}")

            # Predict on test data
            predictions = loaded_model.predict(test_data, validate_schema=False)
            print(f"     🎯 Predictions shape: {predictions.shape}")

        # Example 4: Inspecting model metadata
        print("\n5. Inspecting model metadata...")

        # Load a model and show its metadata
        loaded = load_model(sup_script)
        metadata_dict = loaded.metadata.to_dict()

        print("   📋 Training configuration:")
        for key in [
            "mode",
            "embedding_dim",
            "reference_column",
            "max_epochs",
            "batch_size",
        ]:
            print(f"     {key}: {metadata_dict.get(key)}")

        print("   📈 Training results:")
        for key in ["epochs_trained", "final_loss", "training_time"]:
            print(f"     {key}: {metadata_dict.get(key)}")

        print("   📊 Data information:")
        print(f"     original_columns: {metadata_dict.get('original_columns')}")
        print(f"     data_shape: {metadata_dict.get('data_shape')}")

        # Example 5: Schema validation
        print("\n6. Schema validation...")

        # Test with correct schema
        correct_test = generate_synthetic_data(10, seed=777)
        try:
            loaded.validate_input_schema(correct_test, strict=True)
            print("   ✅ Schema validation passed for correct data")
        except ValueError as e:
            print(f"   ❌ Unexpected schema validation error: {e}")

        # Test with incorrect schema (missing column)
        incorrect_test = correct_test.drop(columns=["Sales"])
        try:
            loaded.validate_input_schema(incorrect_test, strict=True)
            print("   ❌ Schema validation should have failed")
        except ValueError:
            print("   ✅ Schema validation correctly failed for incorrect data")

        # Example 6: Examining generated script
        print("\n7. Generated script transparency...")
        print("   📄 First few lines of generated script:")

        with open(sup_script) as f:
            lines = f.readlines()[:25]  # First 25 lines
            for i, line in enumerate(lines, 1):
                print(f"     {i:2d}: {line.rstrip()}")

        print("   ... (script contains full metadata and loading logic)")

    print("\n" + "=" * 50)
    print("🎉 Demo completed successfully!")
    print("\nKey features demonstrated:")
    print("• Two-file approach (script + binary)")
    print("• Transparent, inspectable metadata")
    print("• Support for all embedding modes")
    print("• Schema validation")
    print("• Easy loading and prediction")


if __name__ == "__main__":
    main()
