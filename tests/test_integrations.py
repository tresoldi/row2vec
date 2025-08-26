"""
Tests for scikit-learn and pandas integrations.
"""

import numpy as np
import pandas as pd
import pytest

import row2vec
from row2vec.config import EmbeddingConfig
from row2vec.utils import generate_synthetic_data


@pytest.fixture
def sample_data():
    """Generate sample data for testing."""
    return generate_synthetic_data(100, seed=1305)


@pytest.fixture
def sample_data_with_target():
    """Generate sample data with target column."""
    df = generate_synthetic_data(100, seed=1305)
    df["category"] = np.random.choice(["A", "B", "C"], 100)
    return df


class TestSklearnIntegration:
    """Test the scikit-learn integration."""

    @pytest.mark.skipif(
        not hasattr(row2vec, "Row2VecTransformer"),
        reason="sklearn integration not available",
    )
    def test_row2vec_transformer_basic(self, sample_data):
        """Test basic Row2VecTransformer functionality."""
        from row2vec import Row2VecTransformer

        transformer = Row2VecTransformer(embedding_dim=5, mode="unsupervised")

        # Test fit
        transformer.fit(sample_data)
        assert hasattr(transformer, "config_")
        assert transformer.config_.embedding_dim == 5
        assert transformer.n_features_in_ == sample_data.shape[1]

        # Test transform
        X_embedded = transformer.transform(sample_data)
        assert X_embedded.shape == (100, 5)
        assert isinstance(X_embedded, np.ndarray)

    @pytest.mark.skipif(
        not hasattr(row2vec, "Row2VecTransformer"),
        reason="sklearn integration not available",
    )
    def test_row2vec_transformer_fit_transform(self, sample_data):
        """Test fit_transform method."""
        from row2vec import Row2VecTransformer

        transformer = Row2VecTransformer(embedding_dim=8, mode="pca")
        X_embedded = transformer.fit_transform(sample_data)

        assert X_embedded.shape == (100, 8)
        assert hasattr(transformer, "config_")
        assert transformer.config_.mode == "pca"

    @pytest.mark.skipif(
        not hasattr(row2vec, "Row2VecTransformer"),
        reason="sklearn integration not available",
    )
    def test_row2vec_transformer_with_config(self, sample_data):
        """Test transformer with pre-configured config object."""
        from row2vec import EmbeddingConfig, NeuralConfig, Row2VecTransformer

        config = EmbeddingConfig(
            mode="unsupervised",
            embedding_dim=6,
            neural=NeuralConfig(max_epochs=25, batch_size=32),
        )

        transformer = Row2VecTransformer(config=config)
        X_embedded = transformer.fit_transform(sample_data)

        assert X_embedded.shape == (100, 6)
        assert transformer.config_.neural.max_epochs == 25
        assert transformer.config_.neural.batch_size == 32

    @pytest.mark.skipif(
        not hasattr(row2vec, "Row2VecTransformer"),
        reason="sklearn integration not available",
    )
    def test_row2vec_transformer_with_numpy(self, sample_data):
        """Test transformer with numpy arrays."""
        from row2vec import Row2VecTransformer

        transformer = Row2VecTransformer(embedding_dim=4, mode="pca")

        # Convert to numpy array
        X_numpy = sample_data.values

        X_embedded = transformer.fit_transform(X_numpy)
        assert X_embedded.shape == (100, 4)

        # Test transform with same data
        X_embedded2 = transformer.transform(X_numpy)
        assert X_embedded2.shape == (100, 4)

    @pytest.mark.skipif(
        not hasattr(row2vec, "Row2VecTransformer"),
        reason="sklearn integration not available",
    )
    def test_row2vec_transformer_nested_params(self, sample_data):
        """Test transformer with nested parameter overrides."""
        from row2vec import Row2VecTransformer

        transformer = Row2VecTransformer(
            embedding_dim=7,
            mode="unsupervised",
            neural__max_epochs=15,
            neural__batch_size=16,
        )

        X_embedded = transformer.fit_transform(sample_data)
        assert X_embedded.shape == (100, 7)
        assert transformer.config_.neural.max_epochs == 15
        assert transformer.config_.neural.batch_size == 16

    @pytest.mark.skipif(
        not hasattr(row2vec, "Row2VecTransformer"),
        reason="sklearn integration not available",
    )
    def test_row2vec_transformer_sklearn_pipeline(self, sample_data):
        """Test transformer in sklearn pipeline."""
        from sklearn.cluster import KMeans
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        from row2vec import Row2VecTransformer

        pipeline = Pipeline(
            [
                ("embed", Row2VecTransformer(embedding_dim=5, mode="pca")),
                ("scale", StandardScaler()),
                ("cluster", KMeans(n_clusters=3, random_state=1305)),
            ]
        )

        # Fit pipeline
        labels = pipeline.fit_predict(sample_data)
        assert len(labels) == 100
        assert len(np.unique(labels)) <= 3  # Should have at most 3 clusters

    @pytest.mark.skipif(
        not hasattr(row2vec, "Row2VecClassifier"),
        reason="sklearn integration not available",
    )
    def test_row2vec_classifier(self, sample_data_with_target):
        """Test Row2VecClassifier."""
        from sklearn.model_selection import train_test_split

        from row2vec import Row2VecClassifier

        # Prepare data
        X = sample_data_with_target.drop("category", axis=1)
        y = sample_data_with_target["category"]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.3,
            random_state=1305,
        )

        # Test classifier (use PCA to avoid batch size issues)
        clf = Row2VecClassifier(
            embedding_dim=8,
            embedding_config=EmbeddingConfig(mode="pca"),
        )
        clf.fit(X_train, y_train)

        # Test predictions
        predictions = clf.predict(X_test)
        assert len(predictions) == len(y_test)
        assert set(predictions).issubset(set(y_train.unique()))

        # Test probabilities
        probabilities = clf.predict_proba(X_test)
        assert probabilities.shape == (len(y_test), len(y_train.unique()))
        assert np.allclose(probabilities.sum(axis=1), 1.0)


class TestPandasIntegration:
    """Test the pandas integration."""

    def test_pandas_accessor_registration(self, sample_data):
        """Test that the pandas accessor is properly registered."""
        assert hasattr(sample_data, "row2vec")

    def test_pandas_embed_basic(self, sample_data):
        """Test basic embed functionality."""
        embeddings = sample_data.row2vec.embed(dim=5, mode="pca")

        assert isinstance(embeddings, pd.DataFrame)
        assert embeddings.shape == (100, 5)
        assert all(
            col.startswith("embedding_") or col.isdigit() for col in embeddings.columns
        )

    def test_pandas_unsupervised(self, sample_data):
        """Test unsupervised method."""
        embeddings = sample_data.row2vec.unsupervised(dim=6, max_epochs=10)

        assert isinstance(embeddings, pd.DataFrame)
        assert embeddings.shape == (100, 6)

    def test_pandas_supervised(self, sample_data_with_target):
        """Test supervised method."""
        embeddings = sample_data_with_target.row2vec.supervised("category", dim=4)

        assert isinstance(embeddings, pd.DataFrame)
        # Should return category embeddings (one per unique category)
        n_categories = sample_data_with_target["category"].nunique()
        assert embeddings.shape[0] == n_categories
        assert embeddings.shape[1] == 4

    def test_pandas_contrastive(self, sample_data):
        """Test contrastive method."""
        embeddings = sample_data.row2vec.contrastive(
            dim=8,
            loss_type="triplet",
            auto_pairs="cluster",
            max_epochs=5,  # Reduce for faster testing
        )

        assert isinstance(embeddings, pd.DataFrame)
        assert embeddings.shape == (100, 8)

    def test_pandas_classical_methods(self, sample_data):
        """Test classical embedding methods."""
        # Test PCA
        pca_embeddings = sample_data.row2vec.pca(dim=3)
        assert pca_embeddings.shape == (100, 3)

        # Test t-SNE
        tsne_embeddings = sample_data.row2vec.tsne(dim=2, perplexity=15, n_iter=250)
        assert tsne_embeddings.shape == (100, 2)

        # Test UMAP
        umap_embeddings = sample_data.row2vec.umap(dim=3, n_neighbors=10)
        assert umap_embeddings.shape == (100, 3)

    def test_pandas_classical_generic(self, sample_data):
        """Test generic classical method."""
        embeddings = sample_data.row2vec.classical("pca", dim=4)
        assert embeddings.shape == (100, 4)

        embeddings = sample_data.row2vec.classical(
            "tsne", dim=2, perplexity=20, n_iter=250
        )
        assert embeddings.shape == (100, 2)

    def test_pandas_compare_methods(self, sample_data):
        """Test method comparison functionality."""
        results = sample_data.row2vec.compare_methods(
            dim=3,
            methods=["pca", "tsne"],  # Use faster methods for testing
        )

        assert isinstance(results, dict)
        assert "pca" in results
        assert "tsne" in results

        for embeddings in results.values():
            assert isinstance(embeddings, pd.DataFrame)
            assert embeddings.shape == (100, 3)

    def test_pandas_with_config_overrides(self, sample_data):
        """Test pandas accessor with configuration overrides."""
        embeddings = sample_data.row2vec.embed(
            dim=5,
            mode="unsupervised",
            max_epochs=15,
            batch_size=32,
            dropout_rate=0.1,
        )

        assert embeddings.shape == (100, 5)


class TestIntegrationCompatibility:
    """Test compatibility between different integration approaches."""

    def test_sklearn_pandas_consistency(self, sample_data):
        """Test that sklearn and pandas integrations produce similar results."""
        # Skip if sklearn integration not available
        if not hasattr(row2vec, "Row2VecTransformer"):
            pytest.skip("sklearn integration not available")

        from row2vec import Row2VecTransformer

        # Generate embeddings using sklearn interface
        transformer = Row2VecTransformer(embedding_dim=5, mode="pca")
        sklearn_embeddings = transformer.fit_transform(sample_data)

        # Generate embeddings using pandas interface
        pandas_embeddings = sample_data.row2vec.pca(dim=5)

        # Should have same shape
        assert sklearn_embeddings.shape == pandas_embeddings.values.shape
        assert sklearn_embeddings.shape == (100, 5)

        # Results should be very similar (allowing for minor numerical differences)
        np.testing.assert_allclose(
            sklearn_embeddings,
            pandas_embeddings.values,
            rtol=1e-10,
        )

    def test_api_consistency_with_integrations(self, sample_data):
        """Test that integrations are consistent with the main API."""
        from row2vec import EmbeddingConfig, learn_embedding_v2

        # Create config
        config = EmbeddingConfig(mode="pca", embedding_dim=4)

        # Direct API call
        direct_embeddings = learn_embedding_v2(sample_data, config)

        # Pandas interface
        pandas_embeddings = sample_data.row2vec.embed(dim=4, mode="pca")

        # Should be identical
        assert direct_embeddings.shape == pandas_embeddings.shape
        np.testing.assert_allclose(
            direct_embeddings.values,
            pandas_embeddings.values,
            rtol=1e-10,
        )


def test_integrations_manually():
    """Manual test function for integration testing."""
    sample_data = generate_synthetic_data(50, seed=1305)

    print("Testing pandas integration...")
    try:
        embeddings = sample_data.row2vec.embed(dim=3, mode="pca")
        print(f"✅ Pandas integration works! Embeddings shape: {embeddings.shape}")
    except Exception as e:
        print(f"❌ Pandas integration failed: {e}")

    print("\nTesting sklearn integration...")
    try:
        from row2vec import Row2VecTransformer

        transformer = Row2VecTransformer(embedding_dim=3, mode="pca")
        X_embedded = transformer.fit_transform(sample_data)
        print(f"✅ Sklearn integration works! Embeddings shape: {X_embedded.shape}")
    except ImportError:
        print("⚠️  Sklearn integration not available")
    except Exception as e:
        print(f"❌ Sklearn integration failed: {e}")

    print("\nAll tests completed!")


if __name__ == "__main__":
    test_integrations_manually()
