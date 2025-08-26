"""Tests for utility functions."""

import pandas as pd
import pytest

from row2vec.utils import (
    create_dataframe_schema,
    generate_synthetic_data,
    validate_dataframe_schema,
)


class TestGenerateSyntheticData:
    """Test synthetic data generation."""

    def test_generate_synthetic_data_basic(self):
        """Test basic synthetic data generation."""
        df = generate_synthetic_data(100)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 100
        assert set(df.columns) == {"Country", "Product", "Sales"}

    def test_generate_synthetic_data_custom_seed(self):
        """Test synthetic data generation with custom seed."""
        df1 = generate_synthetic_data(50, seed=1305)
        df2 = generate_synthetic_data(50, seed=1305)
        df3 = generate_synthetic_data(50, seed=123)

        # Same seed should produce same data
        pd.testing.assert_frame_equal(df1, df2)

        # Different seed should produce different data
        assert not df1.equals(df3)

    def test_generate_synthetic_data_small_dataset(self):
        """Test with very small dataset."""
        df = generate_synthetic_data(5)
        assert len(df) == 5
        assert all(df["Sales"] >= 0)  # Sales should be non-negative

    def test_generate_synthetic_data_countries_and_products(self):
        """Test that generated data contains expected countries and products."""
        df = generate_synthetic_data(1000, seed=1305)

        expected_countries = ["USA", "Canada", "Mexico", "Brazil", "Italy"]
        expected_products = ["A", "B", "C", "D"]

        assert set(df["Country"].unique()).issubset(set(expected_countries))
        assert set(df["Product"].unique()).issubset(set(expected_products))


class TestCreateDataFrameSchema:
    """Test DataFrame schema creation."""

    def test_create_schema_basic(self):
        """Test basic schema creation."""
        df = pd.DataFrame(
            {
                "int_col": [1, 2, 3],
                "float_col": [1.1, 2.2, 3.3],
                "str_col": ["a", "b", "c"],
            }
        )

        schema = create_dataframe_schema(df)

        assert "columns" in schema
        assert "dtypes" in schema
        assert "shape" in schema
        assert "nullable_columns" in schema

        assert schema["columns"] == ["int_col", "float_col", "str_col"]
        assert schema["shape"] == (3, 3)

    def test_create_schema_with_nulls(self):
        """Test schema creation with null values."""
        df = pd.DataFrame(
            {
                "col1": [1, 2, None],
                "col2": ["a", None, "c"],
                "col3": [1.1, 2.2, 3.3],
            }
        )

        schema = create_dataframe_schema(df)

        assert schema["nullable_columns"]["col1"] is True
        assert schema["nullable_columns"]["col2"] is True
        assert schema["nullable_columns"]["col3"] is False

    def test_create_schema_with_categories(self):
        """Test schema creation with categorical columns."""
        df = pd.DataFrame(
            {
                "category_col": ["X", "Y", "Z", "X", "Y"],
                "numeric_col": [1, 2, 3, 4, 5],
            }
        )

        schema = create_dataframe_schema(df)

        assert "categorical_info" in schema
        assert set(schema["categorical_info"]["category_col"]) == {"X", "Y", "Z"}

    def test_create_schema_too_many_categories(self):
        """Test schema creation with too many categories."""
        # Create a column with more than 50 unique values
        df = pd.DataFrame(
            {
                "many_cats": [f"cat_{i}" for i in range(60)],
                "numeric": list(range(60)),
            }
        )

        schema = create_dataframe_schema(df)

        assert "categorical_info" in schema
        assert len(schema["categorical_info"]["many_cats"]) == 1
        assert "Too many categories" in schema["categorical_info"]["many_cats"][0]


class TestValidateDataFrameSchema:
    """Test DataFrame schema validation."""

    def test_validate_schema_success(self):
        """Test successful schema validation."""
        df = pd.DataFrame(
            {
                "col1": [1, 2, 3],
                "col2": ["a", "b", "c"],
            }
        )

        schema = create_dataframe_schema(df)

        # Should not raise any exception
        validate_dataframe_schema(df, schema)

    def test_validate_schema_not_dataframe(self):
        """Test validation with non-DataFrame input."""
        schema = {"columns": ["col1"]}

        with pytest.raises(ValueError, match="Expected pandas DataFrame"):
            validate_dataframe_schema([1, 2, 3], schema)

    def test_validate_schema_empty_dataframe(self):
        """Test validation with empty DataFrame."""
        df = pd.DataFrame()
        schema = {"columns": ["col1"]}

        with pytest.raises(ValueError, match="DataFrame cannot be empty"):
            validate_dataframe_schema(df, schema)

    def test_validate_schema_missing_columns(self):
        """Test validation with missing columns."""
        df = pd.DataFrame({"col1": [1, 2, 3]})
        schema = {"columns": ["col1", "col2"]}

        with pytest.raises(ValueError, match="Missing required columns"):
            validate_dataframe_schema(df, schema)

    def test_validate_schema_allow_missing_columns(self):
        """Test validation allowing missing columns."""
        df = pd.DataFrame({"col1": [1, 2, 3]})
        schema = {"columns": ["col1", "col2"]}

        # Should not raise when allowing missing columns
        validate_dataframe_schema(df, schema, allow_missing_columns=True)

    def test_validate_schema_extra_columns(self):
        """Test validation with extra columns."""
        df = pd.DataFrame(
            {
                "col1": [1, 2, 3],
                "col2": ["a", "b", "c"],
                "extra": [4, 5, 6],
            }
        )
        schema = {"columns": ["col1", "col2"]}

        with pytest.raises(ValueError, match="Unexpected columns found"):
            validate_dataframe_schema(df, schema)

    def test_validate_schema_allow_extra_columns(self):
        """Test validation allowing extra columns."""
        df = pd.DataFrame(
            {
                "col1": [1, 2, 3],
                "col2": ["a", "b", "c"],
                "extra": [4, 5, 6],
            }
        )
        schema = {"columns": ["col1", "col2"]}

        # Should not raise when allowing extra columns
        validate_dataframe_schema(df, schema, allow_extra_columns=True)

    def test_validate_schema_wrong_dtype(self):
        """Test validation with wrong data types."""
        df = pd.DataFrame(
            {
                "col1": ["a", "b", "c"],  # Should be numeric
            }
        )
        schema = {
            "columns": ["col1"],
            "dtypes": {"col1": "int64"},
        }

        with pytest.raises(ValueError, match="incorrect type"):
            validate_dataframe_schema(df, schema)

    def test_validate_schema_compatible_dtypes(self):
        """Test validation with compatible data types."""
        df = pd.DataFrame(
            {
                "int_col": [1, 2, 3],  # int64
                "float_col": [1.0, 2.0, 3.0],  # float64
                "str_col": ["a", "b", "c"],  # object
            }
        )

        # Create schema with slightly different but compatible types
        schema = {
            "columns": ["int_col", "float_col", "str_col"],
            "dtypes": {
                "int_col": "int32",  # Compatible with int64
                "float_col": "float32",  # Compatible with float64
                "str_col": "string",  # Compatible with object
            },
        }

        # Should not raise - types should be considered compatible
        validate_dataframe_schema(df, schema)
