"""
Example: Row2Vec Model Serialization and Persistence

This script demonstrates how to train, save, and load Row2Vec models
using the new serialization functionality.
"""

import tempfile
from pathlib import Path

from row2vec import (
    generate_synthetic_data,
    inspect_model,
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
        unsup_embeddings, unsup_path = train_and_save_model(
            data,
            tmpdir / "unsupervised_model",
            embedding_dim=5,
            mode="unsupervised",
            max_epochs=10,
            batch_size=32,
            verbose=False,
            enable_logging=False,
        )
        print(f"   ✅ Saved to: {unsup_path}")
        print(f"   🎯 Embeddings shape: {unsup_embeddings.shape}")

        # Train supervised model
        print("   Training supervised (target) model...")
        sup_embeddings, sup_path = train_and_save_model(
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
        print(f"   ✅ Saved to: {sup_path}")
        print(f"   🎯 Embeddings shape: {sup_embeddings.shape}")

        # Train PCA model
        print("   Training PCA model...")
        pca_embeddings, pca_path = train_and_save_model(
            data,
            tmpdir / "pca_model",
            embedding_dim=4,
            mode="pca",
            verbose=False,
            enable_logging=False,
        )
        print(f"   ✅ Saved to: {pca_path}")
        print(f"   🎯 Embeddings shape: {pca_embeddings.shape}")

        # Example 2: Manual training and saving
        print("\n3. Manual training and saving workflow...")
        print("   Training with learn_embedding_with_model...")

        # One training pass returns the embeddings and the fitted model that
        # produced them; the model carries its own preprocessor and scaler.
        _embeddings, model = learn_embedding_with_model(
            data,
            embedding_dim=2,
            mode="umap",
            verbose=False,
            enable_logging=False,
        )

        # save_model derives the metadata from the fitted model.
        manual_path = save_model(
            model,
            tmpdir / "manual_model",
        )
        print(f"   ✅ Manually saved to: {manual_path}")

        # Example 3: Loading and using models
        print("\n4. Loading and using saved models...")

        # Load each model and test prediction
        test_data = generate_synthetic_data(20, seed=999)
        print(f"   Test data shape: {test_data.shape}")

        models_to_test = [
            ("Unsupervised", unsup_path),
            ("Supervised", sup_path),
            ("PCA", pca_path),
            ("UMAP (manual)", manual_path),
        ]

        for model_name, model_file in models_to_test:
            print(f"\n   Testing {model_name} model:")

            # Load model
            loaded_model = load_model(model_file)
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
        loaded = load_model(sup_path)
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

        # Example 6: Reading a model's manifest without loading it
        print("\n7. Inspecting the file without loading it...")
        manifest = inspect_model(sup_path)
        print(f"   📄 format: {manifest['format']} v{manifest['format_version']}")
        print(f"   🏷️  written by row2vec {manifest['row2vec_version']}")
        print(f"   📚 libraries: {manifest['libraries']}")
        print(f"   📦 members: {sorted(manifest['members'])}")

    print("\n" + "=" * 50)
    print("🎉 Demo completed successfully!")
    print("\nKey features demonstrated:")
    print("• One .r2v file; loading runs no code from it")
    print("• Inspectable metadata (inspect_model)")
    print("• Support for all embedding modes")
    print("• Schema validation")
    print("• Easy loading and prediction")


if __name__ == "__main__":
    main()
