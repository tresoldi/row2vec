"""
Test data fixtures for Row2Vec testing.

Provides standardized datasets and fixtures for consistent testing
across different test modules and performance regression testing.
"""

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from row2vec import generate_synthetic_data
from row2vec.core import learn_embedding


class TestDataManager:
    """Manages test datasets and fixtures."""

    @staticmethod
    def get_fixture_path() -> Path:
        """Get the path to test fixtures directory."""
        test_dir = Path(__file__).parent
        fixture_dir = test_dir / "fixtures"
        fixture_dir.mkdir(exist_ok=True)
        return fixture_dir

    @staticmethod
    def save_fixture(data: Any, name: str) -> None:
        """Save a test fixture to disk."""
        fixture_path = TestDataManager.get_fixture_path() / f"{name}.pkl"
        with open(fixture_path, "wb") as f:
            pickle.dump(data, f)

    @staticmethod
    def load_fixture(name: str) -> Any:
        """Load a test fixture from disk."""
        fixture_path = TestDataManager.get_fixture_path() / f"{name}.pkl"
        if not fixture_path.exists():
            return None
        with open(fixture_path, "rb") as f:
            return pickle.load(f)


@pytest.fixture(scope="session")
def small_synthetic_data():
    """Small synthetic dataset for quick tests."""
    return generate_synthetic_data(num_records=50, seed=1305)


@pytest.fixture(scope="session")
def medium_synthetic_data():
    """Medium synthetic dataset for thorough tests."""
    return generate_synthetic_data(num_records=200, seed=1305)


@pytest.fixture(scope="session")
def large_synthetic_data():
    """Large synthetic dataset for performance tests."""
    return generate_synthetic_data(num_records=1000, seed=1305)


@pytest.fixture(scope="session")
def numeric_only_data():
    """Dataset with only numeric columns."""
    np.random.seed(42)
    return pd.DataFrame(
        {
            "feature_1": np.random.normal(0, 1, 100),
            "feature_2": np.random.exponential(1, 100),
            "feature_3": np.random.uniform(-5, 5, 100),
            "feature_4": np.random.gamma(2, 2, 100),
        },
    )


@pytest.fixture(scope="session")
def categorical_only_data():
    """Dataset with only categorical columns."""
    np.random.seed(42)
    return pd.DataFrame(
        {
            "category_a": np.random.choice(["X", "Y", "Z"], 100),
            "category_b": np.random.choice(["low", "medium", "high"], 100),
            "category_c": np.random.choice([f"group_{i}" for i in range(5)], 100),
            "binary": np.random.choice(["yes", "no"], 100),
        },
    )


@pytest.fixture(scope="session")
def mixed_complex_data():
    """Complex mixed dataset with various data types and challenges."""
    np.random.seed(42)

    n_samples = 300
    data = {
        # Numeric features with different distributions
        "normal_dist": np.random.normal(0, 1, n_samples),
        "skewed_dist": np.random.exponential(2, n_samples),
        "uniform_dist": np.random.uniform(-10, 10, n_samples),
        "binary_numeric": np.random.choice([0, 1], n_samples),
        # Categorical features with different cardinalities
        "low_cardinality": np.random.choice(["A", "B", "C"], n_samples),
        "medium_cardinality": np.random.choice(
            [f"cat_{i}" for i in range(10)],
            n_samples,
        ),
        "high_cardinality": np.random.choice(
            [f"item_{i}" for i in range(50)],
            n_samples,
        ),
        # Features with missing values
        "numeric_with_nan": np.random.normal(0, 1, n_samples),
        "categorical_with_nan": np.random.choice(["P", "Q", "R"], n_samples),
    }

    # Introduce some NaN values manually
    nan_indices = np.random.choice(n_samples, size=int(0.1 * n_samples), replace=False)
    data["numeric_with_nan"][nan_indices] = np.nan

    # Create DataFrame
    df = pd.DataFrame(data)

    # Add NaN to categorical column using a simpler approach
    cat_nan_indices = np.random.choice(
        n_samples,
        size=int(0.05 * n_samples),
        replace=False,
    )
    for idx in cat_nan_indices:
        df.at[idx, "categorical_with_nan"] = None

    return df


@pytest.fixture(scope="session")
def real_data_samples():
    """Samples from real datasets for integration testing."""
    data_dir = Path(__file__).parent.parent / "data"
    samples = {}

    # Load Titanic sample
    titanic_path = data_dir / "titanic.csv"
    if titanic_path.exists():
        titanic_full = pd.read_csv(titanic_path)
        # Clean and sample
        titanic_clean = titanic_full.dropna(subset=["Age", "Fare"]).copy()
        titanic_features = [
            "Pclass",
            "Sex",
            "Age",
            "SibSp",
            "Parch",
            "Fare",
            "Embarked",
        ]
        available_features = [f for f in titanic_features if f in titanic_clean.columns]
        if len(available_features) >= 4:
            titanic_sample = (
                titanic_clean[available_features]
                .dropna()
                .sample(
                    n=min(150, len(titanic_clean)),
                    random_state=1305,
                )
            )
            samples["titanic"] = titanic_sample

    # Load Housing sample
    housing_path = data_dir / "ames_housing.csv"
    if housing_path.exists():
        housing_full = pd.read_csv(housing_path)
        # Select numeric features and a couple categorical
        numeric_cols = housing_full.select_dtypes(include=[np.number]).columns[:8]
        categorical_cols = (
            ["MSZoning", "Street"] if "MSZoning" in housing_full.columns else []
        )
        selected_cols = list(numeric_cols) + categorical_cols
        housing_sample = (
            housing_full[selected_cols]
            .dropna()
            .sample(
                n=min(150, len(housing_full)),
                random_state=1305,
            )
        )
        samples["housing"] = housing_sample

    return samples


@pytest.fixture(scope="session")
def performance_baseline_data():
    """Standardized dataset for performance baseline testing."""
    np.random.seed(42)

    # Create a dataset that represents typical use case
    n_samples = 1000
    data = {
        "numeric_1": np.random.normal(0, 1, n_samples),
        "numeric_2": np.random.exponential(1, n_samples),
        "numeric_3": np.random.uniform(-5, 5, n_samples),
        "categorical_1": np.random.choice(["A", "B", "C", "D"], n_samples),
        "categorical_2": np.random.choice(["low", "medium", "high"], n_samples),
        "binary": np.random.choice([0, 1], n_samples),
    }

    return pd.DataFrame(data)


class TestFixtureGeneration:
    """Test that fixtures are generated correctly."""

    def test_small_synthetic_data_fixture(self, small_synthetic_data):
        """Test small synthetic data fixture."""
        assert isinstance(small_synthetic_data, pd.DataFrame)
        assert len(small_synthetic_data) == 50
        assert "Country" in small_synthetic_data.columns
        assert "Product" in small_synthetic_data.columns
        assert "Sales" in small_synthetic_data.columns

    def test_numeric_only_fixture(self, numeric_only_data):
        """Test numeric-only data fixture."""
        assert isinstance(numeric_only_data, pd.DataFrame)
        assert len(numeric_only_data) == 100
        assert all(numeric_only_data.dtypes == "float64")
        assert numeric_only_data.shape[1] == 4

    def test_categorical_only_fixture(self, categorical_only_data):
        """Test categorical-only data fixture."""
        assert isinstance(categorical_only_data, pd.DataFrame)
        assert len(categorical_only_data) == 100
        assert all(dtype == "object" for dtype in categorical_only_data.dtypes)
        assert categorical_only_data.shape[1] == 4

    def test_mixed_complex_fixture(self, mixed_complex_data):
        """Test complex mixed data fixture."""
        assert isinstance(mixed_complex_data, pd.DataFrame)
        assert len(mixed_complex_data) == 300
        assert mixed_complex_data.shape[1] == 9

        # Check for numeric columns
        numeric_cols = mixed_complex_data.select_dtypes(include=[np.number]).columns
        assert len(numeric_cols) >= 4

        # Check for categorical columns
        categorical_cols = mixed_complex_data.select_dtypes(include=["object"]).columns
        assert len(categorical_cols) >= 3

        # Check for missing values
        assert mixed_complex_data.isnull().any().any()

    def test_real_data_samples_fixture(self, real_data_samples):
        """Test real data samples fixture."""
        assert isinstance(real_data_samples, dict)

        # If Titanic data is available, test it
        if "titanic" in real_data_samples:
            titanic = real_data_samples["titanic"]
            assert isinstance(titanic, pd.DataFrame)
            assert len(titanic) <= 150
            assert titanic.shape[1] >= 4

        # If housing data is available, test it
        if "housing" in real_data_samples:
            housing = real_data_samples["housing"]
            assert isinstance(housing, pd.DataFrame)
            assert len(housing) <= 150
            assert housing.shape[1] >= 4

    def test_performance_baseline_fixture(self, performance_baseline_data):
        """Test performance baseline data fixture."""
        assert isinstance(performance_baseline_data, pd.DataFrame)
        assert len(performance_baseline_data) == 1000
        assert performance_baseline_data.shape[1] == 6

        # Check data types
        numeric_cols = performance_baseline_data.select_dtypes(
            include=[np.number],
        ).columns
        categorical_cols = performance_baseline_data.select_dtypes(
            include=["object"],
        ).columns

        assert len(numeric_cols) >= 3
        assert len(categorical_cols) >= 2


class TestRegressionFixtures:
    """Test fixtures for regression testing."""

    def test_save_and_load_baseline_results(self, performance_baseline_data):
        """Test saving and loading baseline results for regression testing."""
        from row2vec import learn_embedding

        # Generate baseline embeddings
        baseline_embeddings = learn_embedding(
            performance_baseline_data,
            mode="pca",
            embedding_dim=5,
            verbose=False,
            enable_logging=False,
        )

        # Save baseline
        baseline_data = {
            "embeddings": baseline_embeddings.values,
            "shape": baseline_embeddings.shape,
            "columns": list(baseline_embeddings.columns),
            "method": "pca",
            "embedding_dim": 5,
        }

        TestDataManager.save_fixture(baseline_data, "pca_baseline")

        # Load and verify
        loaded_baseline = TestDataManager.load_fixture("pca_baseline")

        assert loaded_baseline is not None
        assert loaded_baseline["shape"] == (1000, 5)
        assert loaded_baseline["method"] == "pca"
        assert len(loaded_baseline["columns"]) == 5

        # Test that we can reproduce the same results
        new_embeddings = learn_embedding(
            performance_baseline_data,
            mode="pca",
            embedding_dim=5,
            verbose=False,
            enable_logging=False,
        )

        np.testing.assert_array_almost_equal(
            baseline_embeddings.values,
            new_embeddings.values,
            decimal=10,
        )

    def test_cross_method_consistency_fixtures(self, small_synthetic_data):
        """Test consistency across different methods using fixtures."""
        embedding_dim = 3

        # Generate embeddings with different methods
        methods_results = {}

        # Use appropriate batch size for small dataset
        batch_size = min(32, len(small_synthetic_data))

        # PCA (deterministic)
        pca_emb = learn_embedding(
            small_synthetic_data,
            mode="pca",
            embedding_dim=embedding_dim,
            verbose=False,
            enable_logging=False,
            batch_size=batch_size,
        )
        methods_results["pca"] = pca_emb

        # Neural (deterministic with seed)
        neural_emb = learn_embedding(
            small_synthetic_data,
            mode="unsupervised",
            embedding_dim=embedding_dim,
            max_epochs=2,
            seed=1305,
            verbose=False,
            enable_logging=False,
            batch_size=batch_size,
        )
        methods_results["neural"] = neural_emb

        # Verify all methods produce valid embeddings
        for embeddings in methods_results.values():
            assert embeddings.shape == (len(small_synthetic_data), embedding_dim)
            assert not embeddings.isnull().any().any()
            assert all(col.startswith("embedding_") for col in embeddings.columns)

        # Save for regression testing
        consistency_data = {
            method: {
                "embeddings": emb.values,
                "shape": emb.shape,
                "columns": list(emb.columns),
            }
            for method, emb in methods_results.items()
        }

        TestDataManager.save_fixture(consistency_data, "method_consistency_baseline")
