"""
Test suite for the modernized config-based Row2Vec API.

This module tests the new configuration system and ensures compatibility
between the config-based API and the legacy parameter-based API.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from row2vec.api import (
    learn_embedding_classical,
    learn_embedding_contrastive,
    learn_embedding_target,
    learn_embedding_unsupervised,
    learn_embedding_v2,
    learn_embedding_with_model_v2,
)
from row2vec.config import (
    ClassicalConfig,
    ContrastiveConfig,
    EmbeddingConfig,
    LoggingConfig,
    NeuralConfig,
    ScalingConfig,
    create_config_for_mode,
)
from row2vec.core import learn_embedding  # Legacy API for comparison
from row2vec.utils import generate_synthetic_data


class TestConfigObjects:
    """Test the configuration object system."""

    def test_embedding_config_defaults(self):
        """Test that EmbeddingConfig has sensible defaults."""
        config = EmbeddingConfig()

        assert config.embedding_dim == 10
        assert config.mode == "unsupervised"
        assert config.reference_column is None
        assert config.seed == 1305
        assert config.verbose is False

        # Test sub-configs are created
        assert isinstance(config.neural, NeuralConfig)
        assert isinstance(config.classical, ClassicalConfig)
        assert isinstance(config.contrastive, ContrastiveConfig)
        assert isinstance(config.scaling, ScalingConfig)
        assert isinstance(config.logging, LoggingConfig)

    def test_neural_config_validation(self):
        """Test neural config validation."""
        # Valid config should not raise
        NeuralConfig(max_epochs=50, batch_size=32, dropout_rate=0.2)

        # Invalid configs should raise
        with pytest.raises(ValueError, match="max_epochs must be positive"):
            NeuralConfig(max_epochs=0)

        with pytest.raises(ValueError, match="batch_size must be positive"):
            NeuralConfig(batch_size=-1)

        with pytest.raises(ValueError, match="dropout_rate must be between 0 and 1"):
            NeuralConfig(dropout_rate=1.5)

    def test_contrastive_config_validation(self):
        """Test contrastive config validation."""
        # Valid config
        ContrastiveConfig(loss_type="triplet", auto_pairs="cluster")

        # Invalid loss type
        with pytest.raises(ValueError, match="loss_type must be"):
            ContrastiveConfig(loss_type="invalid")

        # Invalid auto_pairs
        with pytest.raises(ValueError, match="auto_pairs must be one of"):
            ContrastiveConfig(auto_pairs="invalid")

    def test_config_to_dict_and_from_dict(self):
        """Test config serialization and deserialization."""
        original_config = EmbeddingConfig(
            embedding_dim=20,
            mode="contrastive",
            neural=NeuralConfig(max_epochs=100, batch_size=32),
            contrastive=ContrastiveConfig(loss_type="contrastive", margin=2.0),
        )

        # Convert to dict and back
        config_dict = original_config.to_dict()
        restored_config = EmbeddingConfig.from_dict(config_dict)

        # Check main config
        assert restored_config.embedding_dim == 20
        assert restored_config.mode == "contrastive"

        # Check sub-configs
        assert restored_config.neural.max_epochs == 100
        assert restored_config.neural.batch_size == 32
        assert restored_config.contrastive.loss_type == "contrastive"
        assert restored_config.contrastive.margin == 2.0

    def test_config_yaml_serialization(self):
        """Test YAML serialization and loading."""
        config = EmbeddingConfig(
            embedding_dim=15,
            mode="target",
            reference_column="category",
            neural=NeuralConfig(max_epochs=75, dropout_rate=0.1),
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            config.to_yaml(f.name)
            yaml_path = f.name

        try:
            # Load config from YAML
            loaded_config = EmbeddingConfig.from_yaml(yaml_path)

            assert loaded_config.embedding_dim == 15
            assert loaded_config.mode == "target"
            assert loaded_config.reference_column == "category"
            assert loaded_config.neural.max_epochs == 75
            assert loaded_config.neural.dropout_rate == 0.1
        finally:
            Path(yaml_path).unlink()  # Clean up


class TestConfigBasedAPI:
    """Test the new config-based API functions."""

    def test_learn_embedding_v2_basic(self):
        """Test basic functionality of learn_embedding_v2."""
        df = generate_synthetic_data(100)

        # Test with default config
        embeddings = learn_embedding_v2(df)
        assert embeddings.shape == (100, 10)  # Default embedding_dim=10

        # Test with custom config
        config = EmbeddingConfig(embedding_dim=5, mode="unsupervised")
        embeddings = learn_embedding_v2(df, config)
        assert embeddings.shape == (100, 5)

    def test_learn_embedding_v2_with_overrides(self):
        """Test config overrides in learn_embedding_v2."""
        df = generate_synthetic_data(100)

        # Test direct overrides
        embeddings = learn_embedding_v2(df, embedding_dim=8, mode="pca")
        assert embeddings.shape == (100, 8)

        # Test nested overrides
        embeddings = learn_embedding_v2(
            df, **{"neural.batch_size": 32, "embedding_dim": 3}
        )
        assert embeddings.shape == (100, 3)

    def test_learn_embedding_with_model_v2(self):
        """Test learn_embedding_with_model_v2 returns all artifacts."""
        df = generate_synthetic_data(80)
        config = EmbeddingConfig(embedding_dim=4)

        embeddings, model, preprocessor, metadata = learn_embedding_with_model_v2(
            df, config
        )

        assert embeddings.shape == (80, 4)
        assert model is not None
        assert preprocessor is not None
        assert isinstance(metadata, dict)
        assert "embedding_dim" in metadata

    def test_convenience_functions(self):
        """Test the convenience functions for common use cases."""
        df = generate_synthetic_data(80)

        # Test unsupervised convenience function
        embeddings = learn_embedding_unsupervised(df, embedding_dim=6)
        assert embeddings.shape == (80, 6)

        # Test target convenience function (need a categorical column)
        df_with_cat = df.copy()
        df_with_cat["category"] = np.random.choice(["A", "B", "C"], 80)
        embeddings = learn_embedding_target(df_with_cat, "category", embedding_dim=4)
        assert (
            embeddings.shape[1] == 4
        )  # 3 categories, but embedding_dim controls output

        # Test classical convenience function
        embeddings = learn_embedding_classical(df, method="pca", embedding_dim=3)
        assert embeddings.shape == (80, 3)

        # Test contrastive convenience function
        embeddings = learn_embedding_contrastive(
            df, embedding_dim=5, loss_type="triplet", auto_pairs="cluster"
        )
        assert embeddings.shape == (80, 5)


class TestAPICompatibility:
    """Test compatibility between new and legacy APIs."""

    def test_config_vs_legacy_equivalence(self):
        """Test that config-based and legacy APIs produce equivalent results."""
        df = generate_synthetic_data(100, seed=1305)  # Fixed seed for reproducibility

        # Legacy API call
        legacy_embeddings = learn_embedding(
            df,
            embedding_dim=8,
            mode="unsupervised",
            max_epochs=10,
            batch_size=32,
            seed=1305,
        )

        # Config API call with equivalent settings
        config = EmbeddingConfig(
            embedding_dim=8,
            mode="unsupervised",
            seed=1305,
            neural=NeuralConfig(max_epochs=10, batch_size=32),
        )
        config_embeddings = learn_embedding_v2(df, config)

        # Results should be very similar (allowing for small numerical differences)
        assert legacy_embeddings.shape == config_embeddings.shape
        np.testing.assert_allclose(
            legacy_embeddings.values,
            config_embeddings.values,
            rtol=1e-10,
            atol=1e-10,
        )

    def test_multiple_mode_compatibility(self):
        """Test compatibility across different embedding modes."""
        df = generate_synthetic_data(80, seed=123)

        # Test PCA mode
        legacy_pca = learn_embedding(df, mode="pca", embedding_dim=5, seed=1305)
        config_pca = learn_embedding_v2(df, mode="pca", embedding_dim=5, seed=1305)
        np.testing.assert_allclose(legacy_pca.values, config_pca.values, rtol=1e-10)

        # Test t-SNE mode
        legacy_tsne = learn_embedding(
            df, mode="tsne", embedding_dim=3, seed=1305, n_iter=500, perplexity=15.0
        )
        config_tsne = learn_embedding_v2(
            df, mode="tsne", embedding_dim=3, seed=1305, n_iter=500, perplexity=15.0
        )
        np.testing.assert_allclose(
            legacy_tsne.values, config_tsne.values, rtol=1e-5
        )  # t-SNE may have more variation


class TestConfigFactories:
    """Test configuration factory functions."""

    def test_create_config_for_mode(self):
        """Test mode-specific config creation."""
        # Test contrastive mode gets optimized settings
        contrastive_config = create_config_for_mode("contrastive")
        assert contrastive_config.mode == "contrastive"
        assert (
            contrastive_config.neural.max_epochs == 100
        )  # More epochs for contrastive
        assert contrastive_config.neural.batch_size == 32  # Smaller batches
        assert (
            contrastive_config.contrastive.auto_pairs == "cluster"
        )  # Default auto_pairs

        # Test target mode gets optimized settings
        target_config = create_config_for_mode("target")
        assert target_config.mode == "target"
        assert target_config.reference_column is None  # Starts as None, user must set
        assert target_config.neural.max_epochs == 75  # More epochs for target
        assert target_config.neural.batch_size == 64  # Standard batch size

        # Test classical modes
        pca_config = create_config_for_mode("pca")
        assert pca_config.mode == "pca"


class TestConfigValidation:
    """Test configuration validation."""

    def test_mode_specific_validation(self):
        """Test that configs validate mode-specific requirements."""
        # Target mode requires reference_column
        with pytest.raises(ValueError, match="reference_column is required"):
            EmbeddingConfig(mode="target", reference_column=None)

        # Valid target config should not raise
        config = EmbeddingConfig(mode="target", reference_column="test_col")
        assert config.mode == "target"

    def test_invalid_mode_validation(self):
        """Test validation of invalid modes."""
        with pytest.raises(ValueError, match="mode must be one of"):
            EmbeddingConfig(mode="invalid_mode")

    def test_embedding_dim_validation(self):
        """Test embedding dimension validation."""
        with pytest.raises(ValueError, match="embedding_dim must be positive"):
            EmbeddingConfig(embedding_dim=0)

        with pytest.raises(ValueError, match="embedding_dim must be positive"):
            EmbeddingConfig(embedding_dim=-1)


if __name__ == "__main__":
    pytest.main([__file__])
