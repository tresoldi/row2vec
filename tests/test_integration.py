"""
Integration tests using real datasets.

These tests validate Row2Vec functionality with actual datasets,
including Titanic, Housing, and Adult datasets.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from row2vec import learn_embedding


class TestRealDatasetIntegration:
    """Integration tests with real-world datasets."""

    @pytest.fixture(scope="class")
    def data_dir(self):
        """Get the data directory path."""
        return Path(__file__).parent.parent / "data"

    @pytest.fixture(scope="class")
    def titanic_data(self, data_dir):
        """Load Titanic dataset."""
        titanic_path = data_dir / "titanic.csv"
        if not titanic_path.exists():
            pytest.skip("Titanic dataset not available")
        return pd.read_csv(titanic_path)

    @pytest.fixture(scope="class")
    def housing_data(self, data_dir):
        """Load Ames housing dataset."""
        housing_path = data_dir / "ames_housing.csv"
        if not housing_path.exists():
            pytest.skip("Housing dataset not available")
        return pd.read_csv(housing_path)

    @pytest.fixture(scope="class")
    def adult_data(self, data_dir):
        """Load Adult dataset."""
        adult_path = data_dir / "adult.csv"
        if not adult_path.exists():
            pytest.skip("Adult dataset not available")
        return pd.read_csv(adult_path)

    def test_titanic_unsupervised_embedding(self, titanic_data):
        """Test unsupervised embeddings on Titanic dataset."""
        # Clean the data for testing - use correct column names
        df = titanic_data.dropna(subset=["Age", "Fare"]).copy()
        df = df[
            [
                "Pclass",
                "Sex",
                "Age",
                "Siblings/Spouses Aboard",
                "Parents/Children Aboard",
                "Fare",
            ]
        ].dropna()

        # Use appropriate batch size
        batch_size = min(32, len(df))

        # Test PCA
        embeddings_pca = learn_embedding(
            df,
            mode="pca",
            embedding_dim=3,
            verbose=False,
            enable_logging=False,
            batch_size=batch_size,
        )

        # Validate results
        assert embeddings_pca.shape[0] == len(df)
        assert embeddings_pca.shape[1] == 3
        assert not embeddings_pca.isnull().any().any()

        # Test neural network method
        embeddings_neural = learn_embedding(
            df,
            mode="unsupervised",
            embedding_dim=3,
            max_epochs=2,
            verbose=False,
            enable_logging=False,
            batch_size=batch_size,
        )

        # Validate neural results
        assert embeddings_neural.shape[0] == len(df)
        assert embeddings_neural.shape[1] == 3
        assert not embeddings_neural.isnull().any().any()

    def test_titanic_target_embedding(self, titanic_data):
        """Test target embeddings on Titanic dataset."""
        # Clean the data - use correct column names
        df = titanic_data.dropna(subset=["Age", "Fare", "Survived"]).copy()
        df = df[
            [
                "Pclass",
                "Sex",
                "Age",
                "Siblings/Spouses Aboard",
                "Parents/Children Aboard",
                "Fare",
                "Survived",
            ]
        ].dropna()

        # Use appropriate batch size
        batch_size = min(32, len(df))

        # Convert Survived to categorical for embedding
        df["Survived"] = df["Survived"].astype(str)
        df_sample = df.sample(n=min(200, len(df)), random_state=1305)

        embeddings = learn_embedding(
            df_sample,
            mode="target",
            reference_column="Survived",
            embedding_dim=3,
            max_epochs=3,
            verbose=False,
            enable_logging=False,
            batch_size=batch_size,
        )

        # Should have embeddings for each unique value of Survived
        unique_survived = df_sample["Survived"].nunique()
        assert embeddings.shape[0] == unique_survived
        assert embeddings.shape[1] == 3

    def test_housing_classical_methods(self, housing_data):
        """Test classical methods on housing dataset."""
        # Select numeric and a few categorical columns
        numeric_cols = housing_data.select_dtypes(include=[np.number]).columns[:5]
        categorical_cols = (
            ["MSZoning", "Street"] if "MSZoning" in housing_data.columns else []
        )

        selected_cols = list(numeric_cols) + categorical_cols
        df = housing_data[selected_cols].dropna()
        df_sample = df.sample(n=min(150, len(df)), random_state=1305)

        # Test PCA
        pca_emb = learn_embedding(
            df_sample,
            mode="pca",
            embedding_dim=4,
            verbose=False,
            enable_logging=False,
        )
        assert pca_emb.shape == (len(df_sample), 4)

        # Test t-SNE (2D for visualization)
        tsne_emb = learn_embedding(
            df_sample,
            mode="tsne",
            embedding_dim=2,
            perplexity=min(10, len(df_sample) // 4),
            n_iter=250,
            verbose=False,
            enable_logging=False,
        )
        assert tsne_emb.shape == (len(df_sample), 2)

        # Test UMAP
        umap_emb = learn_embedding(
            df_sample,
            mode="umap",
            embedding_dim=3,
            n_neighbors=min(10, len(df_sample) - 1),
            verbose=False,
            enable_logging=False,
        )
        assert umap_emb.shape == (len(df_sample), 3)

    def test_adult_large_categorical_handling(self, adult_data):
        """Test handling of datasets with many categorical variables."""
        # Adult dataset has many categorical columns
        categorical_cols = ["workclass", "education", "marital-status", "occupation"]
        numeric_cols = ["age", "fnlwgt", "education-num", "hours-per-week"]

        # Select available columns
        available_cols = []
        for col in categorical_cols + numeric_cols:
            if col in adult_data.columns:
                available_cols.append(col)

        if len(available_cols) < 3:
            pytest.skip("Insufficient columns in adult dataset")

        df = adult_data[available_cols].dropna()
        df_sample = df.sample(n=min(200, len(df)), random_state=1305)

        embeddings = learn_embedding(
            df_sample,
            mode="unsupervised",
            embedding_dim=6,
            max_epochs=2,
            verbose=False,
            enable_logging=False,
        )

        assert embeddings.shape[0] == len(df_sample)
        assert embeddings.shape[1] == 6
        assert not embeddings.isnull().any().any()


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_column_dataframe(self):
        """Test with DataFrame containing only one column."""
        df = pd.DataFrame({"single_col": range(50)})

        # Use appropriate batch size
        batch_size = min(32, len(df))

        embeddings = learn_embedding(
            df,
            mode="pca",
            embedding_dim=1,
            verbose=False,
            enable_logging=False,
            batch_size=batch_size,
        )

        assert embeddings.shape == (50, 1)

    def test_constant_values_dataframe(self):
        """Test with DataFrame containing constant values."""
        df = pd.DataFrame(
            {
                "constant_num": [5.0] * 100,
                "constant_cat": ["A"] * 100,
                "variable": range(100),
            },
        )

        # This should work but might give warnings
        embeddings = learn_embedding(
            df,
            mode="pca",
            embedding_dim=2,
            verbose=False,
            enable_logging=False,
        )

        assert embeddings.shape == (100, 2)

    def test_mixed_data_types_extreme(self):
        """Test with extreme mix of data types."""
        df = pd.DataFrame(
            {
                "tiny_int": np.random.randint(0, 2, 100),
                "large_float": np.random.uniform(1e6, 1e9, 100),
                "many_categories": np.random.choice(
                    [f"cat_{i}" for i in range(20)],
                    100,
                ),
                "binary_str": np.random.choice(["yes", "no"], 100),
                "normal_float": np.random.normal(0, 1, 100),
            },
        )

        embeddings = learn_embedding(
            df,
            mode="unsupervised",
            embedding_dim=4,
            max_epochs=2,
            verbose=False,
            enable_logging=False,
        )

        assert embeddings.shape == (100, 4)
        assert not embeddings.isnull().any().any()

    def test_missing_values_handling(self):
        """Test handling of missing values in data."""
        df = pd.DataFrame(
            {
                "col1": [1, 2, np.nan, 4, 5] * 20,
                "col2": ["A", "B", "C", np.nan, "D"] * 20,
                "col3": np.random.normal(0, 1, 100),
            },
        )

        # The preprocessing should handle NaNs
        embeddings = learn_embedding(
            df,
            mode="pca",
            embedding_dim=3,
            verbose=False,
            enable_logging=False,
        )

        assert embeddings.shape[0] == 100
        assert embeddings.shape[1] == 3
        assert not embeddings.isnull().any().any()

    def test_minimal_dataset_size(self):
        """Test with very small datasets."""
        # Test with minimum viable dataset size
        df = pd.DataFrame(
            {
                "col1": [1, 2, 3],
                "col2": ["A", "B", "C"],
            },
        )

        # Use appropriate batch size
        batch_size = min(2, len(df))

        embeddings = learn_embedding(
            df,
            mode="pca",
            embedding_dim=1,
            verbose=False,
            enable_logging=False,
            batch_size=batch_size,
        )

        assert embeddings.shape == (3, 1)

    def test_high_dimensional_input(self):
        """Test with high-dimensional input (many columns)."""
        # Create dataset with many columns
        n_cols = 50
        df_dict = {}
        for i in range(n_cols):
            df_dict[f"col_{i}"] = np.random.normal(0, 1, 100)
        df = pd.DataFrame(df_dict)

        embeddings = learn_embedding(
            df,
            mode="pca",
            embedding_dim=10,
            verbose=False,
            enable_logging=False,
        )

        assert embeddings.shape == (100, 10)
        assert not embeddings.isnull().any().any()


class TestCompatibility:
    """Test compatibility across different configurations."""

    def test_different_embedding_dimensions(self):
        """Test various embedding dimensions."""
        df = pd.DataFrame(
            {
                "x": np.random.normal(0, 1, 100),
                "y": np.random.normal(0, 1, 100),
                "category": np.random.choice(["A", "B", "C"], 100),
            },
        )

        # After one-hot encoding, we'll have 2 numeric + 3 categorical = 5 features max
        # So test dimensions that are reasonable for this dataset
        dimensions = [1, 2, 3, 4]  # Stay within feasible range

        for dim in dimensions:
            embeddings = learn_embedding(
                df,
                mode="pca",
                embedding_dim=dim,
                verbose=False,
                enable_logging=False,
            )
            assert embeddings.shape == (100, dim)

    def test_different_batch_sizes(self):
        """Test various batch sizes for neural methods."""
        df = pd.DataFrame(
            {
                "x": np.random.normal(0, 1, 200),
                "y": np.random.normal(0, 1, 200),
                "category": np.random.choice(["A", "B", "C"], 200),
            },
        )

        batch_sizes = [16, 32, 64, 128]

        for batch_size in batch_sizes:
            embeddings = learn_embedding(
                df,
                mode="unsupervised",
                embedding_dim=3,
                batch_size=batch_size,
                max_epochs=2,
                verbose=False,
                enable_logging=False,
            )
            assert embeddings.shape == (200, 3)

    def test_scaling_methods_integration(self):
        """Test all scaling methods work with real data processing."""
        df = pd.DataFrame(
            {
                "numeric1": np.random.exponential(2, 100),  # Skewed distribution
                "numeric2": np.random.uniform(-10, 10, 100),
                "category": np.random.choice(["low", "medium", "high"], 100),
            },
        )

        scaling_methods = ["none", "minmax", "standard", "l2", "tanh"]

        for scale_method in scaling_methods:
            embeddings = learn_embedding(
                df,
                mode="pca",
                embedding_dim=3,
                scale_method=scale_method,
                scale_range=(0, 1) if scale_method == "minmax" else None,
                verbose=False,
                enable_logging=False,
            )

            assert embeddings.shape == (100, 3)
            assert not embeddings.isnull().any().any()

            # Verify scaling worked
            if scale_method == "minmax":
                assert embeddings.values.min() >= -1e-6
                assert embeddings.values.max() <= 1 + 1e-6
            elif scale_method == "tanh":
                assert embeddings.values.min() >= -1 - 1e-6
                assert embeddings.values.max() <= 1 + 1e-6
