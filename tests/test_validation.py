"""
Tests for input validation and error handling in Row2Vec
"""

import numpy as np
import pandas as pd
import pytest

from row2vec import generate_synthetic_data, learn_embedding


class TestInputValidation:
    """Test suite for input validation."""

    def test_empty_dataframe_raises_error(self):
        """Test that empty DataFrame raises appropriate error."""
        empty_df = pd.DataFrame()
        with pytest.raises(ValueError, match="DataFrame cannot be empty"):
            learn_embedding(empty_df)

    def test_dataframe_with_no_columns_raises_error(self):
        """Test that DataFrame with no columns raises error."""
        df = pd.DataFrame(index=[0, 1, 2])  # No columns
        with pytest.raises(ValueError, match="DataFrame cannot be empty"):
            learn_embedding(df)

    def test_dataframe_with_insufficient_rows_raises_error(self):
        """Test that DataFrame with too few rows raises error."""
        df = pd.DataFrame({"A": [1], "B": ["x"]})  # Only 1 row
        with pytest.raises(ValueError, match="DataFrame must have at least 2 rows"):
            learn_embedding(df)

    def test_invalid_embedding_dimension_type_raises_error(self):
        """Test that non-integer embedding_dim raises TypeError."""
        df = generate_synthetic_data(10)
        with pytest.raises(TypeError, match="embedding_dim must be an integer"):
            learn_embedding(df, embedding_dim=5.5)  # type: ignore[arg-type]

    def test_negative_embedding_dimension_raises_error(self):
        """Test that negative embedding_dim raises ValueError."""
        df = generate_synthetic_data(10)
        with pytest.raises(ValueError, match="embedding_dim must be positive"):
            learn_embedding(df, embedding_dim=-1)

    def test_zero_embedding_dimension_raises_error(self):
        """Test that zero embedding_dim raises ValueError."""
        df = generate_synthetic_data(10)
        with pytest.raises(ValueError, match="embedding_dim must be positive"):
            learn_embedding(df, embedding_dim=0)

    def test_invalid_mode_raises_error(self):
        """Test that invalid mode raises ValueError."""
        df = generate_synthetic_data(10)
        with pytest.raises(ValueError, match="mode must be one of"):
            learn_embedding(df, embedding_dim=2, mode="invalid_mode")

    def test_target_mode_without_reference_column_raises_error(self):
        """Test that target mode without reference_column raises ValueError."""
        df = generate_synthetic_data(10)
        with pytest.raises(
            ValueError,
            match="reference_column is required when mode='target'",
        ):
            learn_embedding(df, embedding_dim=2, mode="target")

    def test_target_mode_with_invalid_reference_column_type_raises_error(self):
        """Test that non-string reference_column raises TypeError."""
        df = generate_synthetic_data(10)
        with pytest.raises(TypeError, match="reference_column must be a string"):
            learn_embedding(df, embedding_dim=2, mode="target", reference_column=123)  # type: ignore[arg-type]

    def test_target_mode_with_nonexistent_reference_column_raises_error(self):
        """Test that non-existent reference_column raises ValueError."""
        df = generate_synthetic_data(10)
        with pytest.raises(
            ValueError,
            match="reference_column 'NonExistent' not found",
        ):
            learn_embedding(
                df,
                embedding_dim=2,
                mode="target",
                reference_column="NonExistent",
            )

    def test_target_mode_with_single_unique_value_raises_error(self):
        """Test that reference column with only one unique value raises error."""
        df = pd.DataFrame(
            {
                "feature1": [1, 2, 3, 4, 5],
                "feature2": [6, 7, 8, 9, 10],
                "constant_target": ["A", "A", "A", "A", "A"],  # Only one unique value
            },
        )
        with pytest.raises(ValueError, match="must have at least 2 unique values"):
            learn_embedding(
                df,
                embedding_dim=1,
                mode="target",
                reference_column="constant_target",
            )

    def test_target_mode_with_too_many_unique_values_raises_error(self):
        """Test that reference column with too many unique values raises error."""
        # Create DataFrame with high cardinality target
        df = pd.DataFrame(
            {
                "feature1": range(1001),
                "target": [f"category_{i}" for i in range(1001)],  # 1001 unique values
            },
        )
        with pytest.raises(ValueError, match="has too many unique values"):
            learn_embedding(
                df,
                embedding_dim=1,
                mode="target",
                reference_column="target",
            )

    def test_invalid_max_epochs_raises_error(self):
        """Test that invalid max_epochs raises ValueError."""
        df = generate_synthetic_data(10)
        with pytest.raises(ValueError, match="max_epochs must be a positive integer"):
            learn_embedding(df, embedding_dim=2, max_epochs=-1)

    def test_invalid_batch_size_raises_error(self):
        """Test that invalid batch_size raises ValueError."""
        df = generate_synthetic_data(10)
        with pytest.raises(ValueError, match="batch_size must be a positive integer"):
            learn_embedding(df, embedding_dim=2, batch_size=0)

    def test_batch_size_larger_than_dataset_raises_error(self):
        """Test that batch_size larger than dataset raises ValueError."""
        df = generate_synthetic_data(5)
        with pytest.raises(
            ValueError,
            match="batch_size .* cannot be larger than dataset size",
        ):
            learn_embedding(df, embedding_dim=1, batch_size=10)

    def test_invalid_dropout_rate_raises_error(self):
        """Test that invalid dropout_rate raises ValueError."""
        df = generate_synthetic_data(10)
        with pytest.raises(
            ValueError,
            match="dropout_rate must be a number between 0 and 1",
        ):
            learn_embedding(
                df,
                embedding_dim=2,
                dropout_rate=1.5,
                max_epochs=1,
                batch_size=10,
            )

    def test_invalid_hidden_units_raises_error(self):
        """Test that invalid hidden_units raises ValueError."""
        df = generate_synthetic_data(10)
        with pytest.raises(ValueError, match="hidden_units must be a positive integer"):
            learn_embedding(
                df,
                embedding_dim=2,
                hidden_units=-10,
                max_epochs=1,
                batch_size=10,
            )

    def test_invalid_scale_method_raises_error(self):
        """Test that invalid scale_method raises ValueError."""
        df = generate_synthetic_data(10)
        with pytest.raises(ValueError, match="scale_method must be one of"):
            learn_embedding(
                df,
                embedding_dim=2,
                scale_method="invalid_scaling",
                max_epochs=1,
                batch_size=10,
            )

    def test_invalid_scale_range_type_raises_error(self):
        """Test that invalid scale_range type raises TypeError."""
        df = generate_synthetic_data(10)
        with pytest.raises(TypeError, match="scale_range must be a tuple or list"):
            learn_embedding(
                df,
                embedding_dim=2,
                scale_range="invalid",
                max_epochs=1,
                batch_size=10,
            )  # type: ignore[arg-type]

    def test_invalid_scale_range_length_raises_error(self):
        """Test that scale_range with wrong length raises TypeError."""
        df = generate_synthetic_data(10)
        with pytest.raises(
            TypeError,
            match="scale_range must be a tuple or list of two numbers",
        ):
            learn_embedding(
                df,
                embedding_dim=2,
                scale_range=(0, 1, 2),
                max_epochs=1,
                batch_size=10,
            )  # type: ignore[arg-type]

    def test_invalid_scale_range_values_raises_error(self):
        """Test that scale_range with low >= high raises ValueError."""
        df = generate_synthetic_data(10)
        with pytest.raises(
            ValueError,
            match="scale_range low value .* must be less than high value",
        ):
            learn_embedding(
                df,
                embedding_dim=2,
                scale_range=(1, 0),
                max_epochs=1,
                batch_size=10,
            )

    def test_all_nan_columns_raises_error(self):
        """Test that DataFrame with all-NaN columns raises ValueError."""
        df = pd.DataFrame(
            {
                "valid_col": [1, 2, 3, 4, 5],
                "nan_col": [np.nan, np.nan, np.nan, np.nan, np.nan],
            },
        )
        with pytest.raises(
            ValueError,
            match="DataFrame contains columns with all NaN values",
        ):
            learn_embedding(df, embedding_dim=1, max_epochs=1, batch_size=5)

    def test_no_usable_columns_raises_error(self):
        """Test that DataFrame with no usable columns raises ValueError."""
        # In target mode, if reference column is the only column
        df = pd.DataFrame({"target_only": ["A", "B", "A", "B", "A"]})
        with pytest.raises(
            ValueError,
            match="DataFrame must contain at least one numeric or categorical column",
        ):
            learn_embedding(
                df,
                mode="target",
                reference_column="target_only",
                embedding_dim=1,
                max_epochs=1,
                batch_size=5,
            )

    def test_wrong_dataframe_type_raises_error(self):
        """Test that non-DataFrame input raises TypeError."""
        with pytest.raises(TypeError, match="Expected pandas DataFrame"):
            learn_embedding([1, 2, 3, 4, 5], max_epochs=1, batch_size=5)  # type: ignore[arg-type]


class TestScalingValidation:
    """Test suite for scaling validation."""

    def test_constant_column_minmax_scaling_error(self):
        """Test that constant columns are handled appropriately."""
        df = pd.DataFrame(
            {
                "varying": [1, 2, 3, 4, 5],
                "constant": [5, 5, 5, 5, 5],  # Constant column
            },
        )
        # The preprocessing should handle constant columns appropriately
        result = learn_embedding(
            df,
            embedding_dim=1,
            scale_method="minmax",
            max_epochs=1,
            batch_size=5,
        )
        # Should work since preprocessing handles constant columns
        assert result.shape == (5, 1)


class TestValidInputs:
    """Test that valid inputs work correctly."""

    def test_minimum_valid_dataframe_unsupervised(self):
        """Test that minimum valid DataFrame works in unsupervised mode."""
        df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
        result = learn_embedding(df, embedding_dim=1, max_epochs=1, batch_size=2)
        assert result.shape == (2, 1)

    def test_minimum_valid_dataframe_target(self):
        """Test that minimum valid DataFrame works in target mode."""
        df = pd.DataFrame(
            {
                "feature": [1, 2, 3, 4],
                "target": ["A", "B", "A", "B"],
            },
        )
        result = learn_embedding(
            df,
            mode="target",
            reference_column="target",
            embedding_dim=1,
            max_epochs=1,
            batch_size=4,
        )
        assert result.shape == (2, 1)  # 2 unique categories

    def test_edge_case_parameters(self):
        """Test edge case parameters that should work."""
        df = generate_synthetic_data(10)

        # Test minimum parameters
        result = learn_embedding(
            df,
            embedding_dim=1,
            max_epochs=1,
            batch_size=10,  # Match dataset size
            dropout_rate=0.0,
            hidden_units=1,
        )
        assert result.shape == (10, 1)

    def test_large_embedding_dimension_warning(self):
        """Test that very large embedding dimensions are handled."""
        df = pd.DataFrame({"A": range(10), "B": range(10, 20)})
        # Should work but might not be optimal - needs appropriate batch_size
        with pytest.raises(
            ValueError,
            match="embedding_dim .* cannot be larger than the number of features",
        ):
            learn_embedding(df, embedding_dim=100, batch_size=10)
