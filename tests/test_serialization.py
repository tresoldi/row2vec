"""
Tests for Row2Vec model serialization and persistence functionality.
"""

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from row2vec import (
    Row2VecModel,
    generate_synthetic_data,
    learn_embedding_with_model,
    load_model,
    save_model,
    train_and_save_model,
)
from row2vec.serialization import Row2VecModelMetadata


class TestModelSerialization:
    """Test model serialization and persistence functionality."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample data for testing."""
        return generate_synthetic_data(100, seed=1305)

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_learn_embedding_with_model_unsupervised(self, sample_data):
        """Test learn_embedding_with_model with unsupervised mode."""
        embeddings, model, preprocessor, metadata = learn_embedding_with_model(
            sample_data,
            embedding_dim=5,
            mode="unsupervised",
            max_epochs=10,
            verbose=False,
            enable_logging=False,
        )

        # Check return types and shapes
        assert isinstance(embeddings, pd.DataFrame)
        assert embeddings.shape[1] == 5
        assert embeddings.shape[0] == 100
        assert model is not None
        assert preprocessor is not None
        assert isinstance(metadata, dict)

        # Check metadata contents
        assert metadata["embedding_dim"] == 5
        assert metadata["mode"] == "unsupervised"
        assert metadata["data_shape"] == (100, 3)
        assert "training_time" in metadata
        assert "original_columns" in metadata

    def test_learn_embedding_with_model_target(self, sample_data):
        """Test learn_embedding_with_model with target mode."""
        embeddings, model, preprocessor, metadata = learn_embedding_with_model(
            sample_data,
            embedding_dim=3,
            mode="target",
            reference_column="Country",
            max_epochs=10,
            verbose=False,
            enable_logging=False,
        )

        # Check return types and shapes
        assert isinstance(embeddings, pd.DataFrame)
        # In target mode, we get grouped embeddings (one per category), not per row
        # The shape should be (n_categories, embedding_dim) - no extra category column
        # since the categories become the index after grouping
        assert embeddings.shape[1] == 3  # Just the embeddings
        assert embeddings.shape[0] <= 5  # Number of unique countries
        assert model is not None
        assert preprocessor is not None
        assert metadata["mode"] == "target"
        assert metadata["reference_column"] == "Country"

    def test_learn_embedding_with_model_pca(self, sample_data):
        """Test learn_embedding_with_model with PCA mode."""
        embeddings, model, preprocessor, metadata = learn_embedding_with_model(
            sample_data,
            embedding_dim=2,
            mode="pca",
            verbose=False,
            enable_logging=False,
        )

        # Check return types and shapes
        assert isinstance(embeddings, pd.DataFrame)
        assert embeddings.shape[1] == 2
        assert model is not None  # Should be a PCA estimator
        assert preprocessor is not None
        assert metadata["mode"] == "pca"

    def test_save_and_load_model_unsupervised(self, sample_data, temp_dir):
        """Test saving and loading an unsupervised model."""
        # Train a model
        embeddings, model, preprocessor, metadata = learn_embedding_with_model(
            sample_data,
            embedding_dim=3,
            mode="unsupervised",
            max_epochs=5,
            verbose=False,
            enable_logging=False,
        )

        # Create Row2Vec model
        row2vec_model = Row2VecModel(
            model=model,
            preprocessor=preprocessor,
            metadata=Row2VecModelMetadata.from_dict(metadata),
        )

        # Save the model
        base_path = temp_dir / "test_model"
        script_path, binary_path = save_model(row2vec_model, base_path)

        # Check files exist
        assert Path(script_path).exists()
        assert Path(binary_path).exists()
        assert script_path.endswith(".py")
        assert binary_path.endswith(".pkl")

        # Load the model
        loaded_model = load_model(script_path)

        # Check loaded model
        assert isinstance(loaded_model, Row2VecModel)
        assert loaded_model.model is not None
        assert loaded_model.preprocessor is not None
        assert loaded_model.metadata is not None

        # Test prediction with loaded model
        test_data = generate_synthetic_data(20, seed=123)
        predictions = loaded_model.predict(test_data, validate_schema=False)

        assert isinstance(predictions, pd.DataFrame)
        assert predictions.shape[1] == 3
        assert predictions.shape[0] == 20

    def test_save_and_load_model_pca(self, sample_data, temp_dir):
        """Test saving and loading a PCA model."""
        # Train a PCA model
        embeddings, model, preprocessor, metadata = learn_embedding_with_model(
            sample_data,
            embedding_dim=2,
            mode="pca",
            verbose=False,
            enable_logging=False,
        )

        # Create Row2Vec model
        row2vec_model = Row2VecModel(
            model=model,
            preprocessor=preprocessor,
            metadata=Row2VecModelMetadata.from_dict(metadata),
        )

        # Save the model
        base_path = temp_dir / "test_pca_model"
        script_path, binary_path = save_model(row2vec_model, base_path)

        # Load the model
        loaded_model = load_model(script_path)

        # Test prediction with loaded model
        test_data = generate_synthetic_data(20, seed=123)
        predictions = loaded_model.predict(test_data, validate_schema=False)

        assert isinstance(predictions, pd.DataFrame)
        assert predictions.shape[1] == 2
        assert predictions.shape[0] == 20

    def test_train_and_save_model(self, sample_data, temp_dir):
        """Test the convenience function train_and_save_model."""
        base_path = temp_dir / "convenience_model"

        embeddings, script_path, binary_path = train_and_save_model(
            sample_data,
            base_path,
            embedding_dim=4,
            mode="unsupervised",
            max_epochs=5,
            verbose=False,
            enable_logging=False,
        )

        # Check return values
        assert isinstance(embeddings, pd.DataFrame)
        assert embeddings.shape[1] == 4
        assert Path(script_path).exists()
        assert Path(binary_path).exists()

        # Load and test the saved model
        loaded_model = load_model(script_path)
        test_data = generate_synthetic_data(15, seed=456)
        predictions = loaded_model.predict(test_data, validate_schema=False)

        assert predictions.shape[1] == 4
        assert predictions.shape[0] == 15

    def test_model_metadata_inspection(self, sample_data, temp_dir):
        """Test that model metadata is properly accessible and inspectable."""
        base_path = temp_dir / "metadata_test_model"

        # Train and save a model
        embeddings, script_path, binary_path = train_and_save_model(
            sample_data,
            base_path,
            embedding_dim=3,
            mode="target",
            reference_column="Country",
            max_epochs=5,
            verbose=False,
            enable_logging=False,
        )

        # Read the generated script as text to verify metadata is readable
        script_content = Path(script_path).read_text()

        # Check that key metadata is present in the script
        assert "Mode: target" in script_content
        assert "Embedding Dimensions: 3" in script_content
        assert "Reference Column: Country" in script_content
        assert "METADATA = {" in script_content

        # Load the model and check metadata access
        loaded_model = load_model(script_path)
        metadata_dict = loaded_model.metadata.to_dict()

        assert metadata_dict["mode"] == "target"
        assert metadata_dict["embedding_dim"] == 3
        assert metadata_dict["reference_column"] == "Country"
        assert metadata_dict["data_shape"] == (100, 3)
        assert "original_columns" in metadata_dict
        assert "expected_schema" in metadata_dict

    def test_schema_validation(self, sample_data, temp_dir):
        """Test schema validation functionality."""
        # Train and save a model
        embeddings, model, preprocessor, metadata = learn_embedding_with_model(
            sample_data,
            embedding_dim=2,
            mode="unsupervised",
            max_epochs=5,
            verbose=False,
            enable_logging=False,
        )

        row2vec_model = Row2VecModel(
            model=model,
            preprocessor=preprocessor,
            metadata=Row2VecModelMetadata.from_dict(metadata),
        )

        base_path = temp_dir / "schema_test_model"
        script_path, binary_path = save_model(row2vec_model, base_path)
        loaded_model = load_model(script_path)

        # Test with correct schema
        correct_data = generate_synthetic_data(10, seed=789)
        assert loaded_model.validate_input_schema(correct_data, strict=False)

        # Test with incorrect schema (missing column)
        incorrect_data = correct_data.drop(columns=["Sales"])
        with pytest.raises(ValueError, match="Schema validation failed"):
            loaded_model.validate_input_schema(incorrect_data, strict=True)

        # Test with incorrect schema (wrong type)
        wrong_type_data = correct_data.copy()
        wrong_type_data["Sales"] = wrong_type_data["Sales"].astype(str)
        # Note: This might not fail due to type compatibility rules

    def test_overwrite_protection(self, sample_data, temp_dir):
        """Test that overwrite protection works correctly."""
        base_path = temp_dir / "overwrite_test"

        # Create first model
        train_and_save_model(
            sample_data,
            base_path,
            embedding_dim=2,
            max_epochs=2,
            verbose=False,
            enable_logging=False,
        )

        # Try to save again without overwrite - should fail
        with pytest.raises(FileExistsError):
            train_and_save_model(
                sample_data,
                base_path,
                embedding_dim=3,
                max_epochs=2,
                verbose=False,
                enable_logging=False,
                overwrite=False,
            )

        # Try to save again with overwrite - should succeed
        embeddings, script_path, binary_path = train_and_save_model(
            sample_data,
            base_path,
            embedding_dim=3,
            max_epochs=2,
            verbose=False,
            enable_logging=False,
            overwrite=True,
        )

        # Load and verify the new model has 3 dimensions
        loaded_model = load_model(script_path)
        test_predictions = loaded_model.predict(
            generate_synthetic_data(5, seed=999),
            validate_schema=False,
        )
        assert test_predictions.shape[1] == 3
