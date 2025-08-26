"""
Tests for neural architecture search functionality.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from row2vec.architecture_search import (
    ArchitectureSearchConfig,
    ArchitectureSearcher,
    ArchitectureSearchResult,
    search_architecture,
)
from row2vec.config import EmbeddingConfig


class TestArchitectureSearchConfig:
    """Test ArchitectureSearchConfig class."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ArchitectureSearchConfig()

        assert config.method == "random"
        assert config.max_trials == 30
        assert config.max_time == 1800
        assert config.patience == 10
        assert config.min_improvement == 0.01
        assert config.layer_range == (1, 4)
        assert config.width_options == [32, 64, 128, 256, 512]
        assert config.dropout_options == [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        assert config.activation_options == ["relu", "elu", "swish"]
        assert config.verbose is True

    def test_custom_config(self):
        """Test custom configuration values."""
        config = ArchitectureSearchConfig(
            method="grid",
            max_trials=50,
            layer_range=(2, 5),
            width_options=[64, 128, 256],
            verbose=False,
        )

        assert config.method == "grid"
        assert config.max_trials == 50
        assert config.layer_range == (2, 5)
        assert config.width_options == [64, 128, 256]
        assert config.verbose is False


class TestArchitectureSearchResult:
    """Test ArchitectureSearchResult class."""

    def test_result_creation(self):
        """Test creating search result."""
        best_arch = {"n_layers": 2, "hidden_units": [128, 64]}
        history = [
            {"trial": 0, "score": 0.8},
            {"trial": 1, "score": 0.9},
        ]

        result = ArchitectureSearchResult(
            best_architecture=best_arch,
            best_score=0.9,
            search_history=history,
            total_time=100.0,
            trials_completed=2,
        )

        assert result.best_architecture == best_arch
        assert result.best_score == 0.9
        assert result.total_time == 100.0
        assert result.trials_completed == 2

    def test_summary(self):
        """Test result summary generation."""
        best_arch = {"n_layers": 2, "hidden_units": [128, 64]}
        history = [
            {"trial": 0, "score": 0.8},
            {"trial": 1, "score": 0.9},
        ]

        result = ArchitectureSearchResult(
            best_architecture=best_arch,
            best_score=0.9,
            search_history=history,
            total_time=100.0,
            trials_completed=2,
        )

        summary = result.summary()

        assert summary["best_architecture"] == best_arch
        assert summary["best_score"] == 0.9
        assert summary["total_time"] == 100.0
        assert summary["trials_completed"] == 2
        assert "improvement_over_baseline" in summary
        assert "search_efficiency" in summary


class TestArchitectureSearcher:
    """Test ArchitectureSearcher class."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        return pd.DataFrame(
            {
                "feature1": np.random.normal(0, 1, 100),
                "feature2": np.random.uniform(-1, 1, 100),
                "feature3": np.random.exponential(1, 100),
            }
        )

    @pytest.fixture
    def base_config(self):
        """Create base embedding config."""
        return EmbeddingConfig(mode="unsupervised", embedding_dim=5)

    @pytest.fixture
    def search_config(self):
        """Create search config for testing."""
        return ArchitectureSearchConfig(
            max_trials=3,  # Small number for testing
            max_time=60,  # Short time for testing
            patience=2,
            verbose=False,
        )

    def test_searcher_initialization(self, search_config):
        """Test searcher initialization."""
        searcher = ArchitectureSearcher(search_config)

        assert searcher.config == search_config
        assert searcher.search_history == []
        assert searcher.best_score == float("-inf")
        assert searcher.best_architecture is None
        assert searcher.trials_without_improvement == 0

    def test_sample_random_architecture(self, search_config):
        """Test random architecture sampling."""
        searcher = ArchitectureSearcher(search_config)

        architecture = searcher._sample_random_architecture()

        assert "n_layers" in architecture
        assert "hidden_units" in architecture
        assert "dropout_rate" in architecture
        assert "activation" in architecture

        assert (
            search_config.layer_range[0]
            <= architecture["n_layers"]
            <= search_config.layer_range[1]
        )
        assert architecture["dropout_rate"] in search_config.dropout_options
        assert architecture["activation"] in search_config.activation_options
        # Check that hidden_units structure matches n_layers
        if architecture["n_layers"] == 1:
            assert isinstance(architecture["hidden_units"], int)
        else:
            assert isinstance(architecture["hidden_units"], list)
            assert len(architecture["hidden_units"]) == architecture["n_layers"]

    def test_generate_grid_architectures(self, search_config):
        """Test grid architecture generation."""
        searcher = ArchitectureSearcher(search_config)

        architectures = searcher._generate_grid_architectures()

        assert len(architectures) > 0
        assert len(architectures) <= search_config.max_trials

        for arch in architectures:
            assert "n_layers" in arch
            assert "hidden_units" in arch
            assert "dropout_rate" in arch
            assert "activation" in arch

    @patch("row2vec.architecture_search.learn_embedding_v2")
    def test_evaluate_architecture(
        self, mock_learn_embedding, sample_data, base_config, search_config
    ):
        """Test architecture evaluation."""
        # Mock embedding generation
        mock_embeddings = pd.DataFrame(np.random.randn(100, 5))
        mock_learn_embedding.return_value = mock_embeddings

        searcher = ArchitectureSearcher(search_config)
        architecture = {
            "n_layers": 2,
            "hidden_units": [128, 64],
            "dropout_rate": 0.2,
            "activation": "relu",
        }

        score = searcher._evaluate_architecture(
            sample_data, base_config, architecture, None, 0
        )

        assert isinstance(score, float)
        assert score != float("-inf")  # Should not fail
        assert len(searcher.search_history) == 1

        trial_record = searcher.search_history[0]
        assert trial_record["trial"] == 0
        assert trial_record["architecture"] == architecture
        assert "score" in trial_record
        assert "metrics" in trial_record

    @patch("row2vec.architecture_search.learn_embedding_v2")
    def test_search_random(
        self, mock_learn_embedding, sample_data, base_config, search_config
    ):
        """Test random search execution."""
        # Mock embedding generation
        mock_embeddings = pd.DataFrame(np.random.randn(100, 5))
        mock_learn_embedding.return_value = mock_embeddings

        searcher = ArchitectureSearcher(search_config)
        result = searcher.search(sample_data, base_config)

        assert isinstance(result, ArchitectureSearchResult)
        assert result.best_architecture is not None
        assert result.best_score > float("-inf")
        assert result.trials_completed <= search_config.max_trials
        assert len(searcher.search_history) == result.trials_completed

    @patch("row2vec.architecture_search.learn_embedding_v2")
    def test_search_grid(self, mock_learn_embedding, sample_data, base_config):
        """Test grid search execution."""
        # Mock embedding generation
        mock_embeddings = pd.DataFrame(np.random.randn(100, 5))
        mock_learn_embedding.return_value = mock_embeddings

        search_config = ArchitectureSearchConfig(
            method="grid",
            max_trials=5,
            verbose=False,
        )

        searcher = ArchitectureSearcher(search_config)
        result = searcher.search(sample_data, base_config)

        assert isinstance(result, ArchitectureSearchResult)
        assert result.best_architecture is not None
        assert result.best_score > float("-inf")
        assert result.trials_completed <= search_config.max_trials

    def test_should_stop_patience(self, search_config):
        """Test stopping based on patience."""
        searcher = ArchitectureSearcher(search_config)
        searcher.trials_without_improvement = search_config.patience

        assert searcher._should_stop(0) is True

    def test_should_not_stop_early(self, search_config):
        """Test not stopping early."""
        searcher = ArchitectureSearcher(search_config)
        searcher.trials_without_improvement = search_config.patience - 1

        assert searcher._should_stop(0) is False

    def test_describe_search_space(self, search_config):
        """Test search space description."""
        searcher = ArchitectureSearcher(search_config)
        description = searcher._describe_search_space()

        assert isinstance(description, str)
        assert "combinations" in description.lower()


class TestSearchArchitectureFunction:
    """Test the main search_architecture function."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        return pd.DataFrame(
            {
                "feature1": np.random.normal(0, 1, 50),
                "feature2": np.random.uniform(-1, 1, 50),
            }
        )

    @pytest.fixture
    def base_config(self):
        """Create base embedding config."""
        return EmbeddingConfig(mode="unsupervised", embedding_dim=3)

    @patch("row2vec.architecture_search.learn_embedding_v2")
    def test_search_architecture_default_config(
        self, mock_learn_embedding, sample_data, base_config
    ):
        """Test search_architecture with default config."""
        # Mock embedding generation
        mock_embeddings = pd.DataFrame(np.random.randn(50, 3))
        mock_learn_embedding.return_value = mock_embeddings

        best_arch, result = search_architecture(sample_data, base_config)

        assert isinstance(best_arch, dict)
        assert isinstance(result, ArchitectureSearchResult)
        assert "n_layers" in best_arch
        assert "hidden_units" in best_arch
        assert "dropout_rate" in best_arch
        assert "activation" in best_arch

    @patch("row2vec.architecture_search.learn_embedding_v2")
    def test_search_architecture_custom_config(
        self, mock_learn_embedding, sample_data, base_config
    ):
        """Test search_architecture with custom config."""
        # Mock embedding generation
        mock_embeddings = pd.DataFrame(np.random.randn(50, 3))
        mock_learn_embedding.return_value = mock_embeddings

        search_config = ArchitectureSearchConfig(
            max_trials=5,
            method="random",
            verbose=False,
        )

        best_arch, result = search_architecture(sample_data, base_config, search_config)

        assert isinstance(best_arch, dict)
        assert isinstance(result, ArchitectureSearchResult)
        assert result.trials_completed <= 5


class TestIntegrationWithAPI:
    """Test integration with main API."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        return pd.DataFrame(
            {
                "feature1": np.random.normal(0, 1, 30),
                "feature2": np.random.uniform(-1, 1, 30),
            }
        )

    @patch("row2vec.architecture_search.search_architecture")
    @patch("row2vec.api._legacy_learn_embedding")
    def test_learn_embedding_v2_with_auto_architecture(
        self, mock_legacy, mock_search, sample_data
    ):
        """Test learn_embedding_v2 with auto_architecture=True."""
        from row2vec.api import learn_embedding_v2
        from row2vec.config import EmbeddingConfig

        # Mock search results
        mock_best_arch = {
            "hidden_units": [128, 64],
            "dropout_rate": 0.2,
            "activation": "relu",
        }
        mock_result = MagicMock()
        mock_result.summary.return_value = {
            "trials_completed": 10,
            "total_time": 60.0,
            "best_score": 0.85,
        }
        mock_search.return_value = (mock_best_arch, mock_result)

        # Mock legacy embedding
        mock_legacy.return_value = pd.DataFrame(np.random.randn(30, 5))

        config = EmbeddingConfig(mode="unsupervised", embedding_dim=5)
        result = learn_embedding_v2(sample_data, config, auto_architecture=True)

        assert mock_search.called
        assert mock_legacy.called
        assert isinstance(result, pd.DataFrame)


class TestErrorHandling:
    """Test error handling in architecture search."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        return pd.DataFrame({"feature1": [1, 2, 3]})  # Very small dataset

    @pytest.fixture
    def base_config(self):
        """Create base embedding config."""
        return EmbeddingConfig(mode="unsupervised", embedding_dim=2)

    def test_invalid_search_method(self, sample_data, base_config):
        """Test error handling for invalid search method."""
        search_config = ArchitectureSearchConfig(method="invalid_method")
        searcher = ArchitectureSearcher(search_config)

        with pytest.raises(ValueError, match="Unknown search method"):
            searcher.search(sample_data, base_config)

    @patch("row2vec.architecture_search.learn_embedding_v2")
    def test_embedding_generation_failure(
        self, mock_learn_embedding, sample_data, base_config
    ):
        """Test handling of embedding generation failures."""
        # Mock embedding generation to fail
        mock_learn_embedding.side_effect = Exception("Embedding failed")

        search_config = ArchitectureSearchConfig(max_trials=2, verbose=False)
        searcher = ArchitectureSearcher(search_config)

        # Should handle the error gracefully
        result = searcher.search(sample_data, base_config)

        # Should still return a result, even if all trials failed
        assert isinstance(result, ArchitectureSearchResult)
        assert len(searcher.search_history) == search_config.max_trials

        # All trials should have failed
        for trial in searcher.search_history:
            assert trial["score"] == float("-inf")
            assert "error" in trial
