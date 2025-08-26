"""
Row2Vec: Utility functions
"""

import random
from typing import Any

import numpy as np
import pandas as pd


def generate_synthetic_data(num_records: int, seed: int = 1305) -> pd.DataFrame:
    """
    Generates a synthetic DataFrame for demonstration purposes.

    Args:
        num_records (int): The number of records to generate.
        seed (int): A random seed for reproducibility.

    Returns:
        pd.DataFrame: A synthetic DataFrame with mixed data types.
    """
    random.seed(seed)
    rng = np.random.default_rng(seed)

    countries: list[str] = ["USA", "Canada", "Mexico", "Brazil", "Italy"]
    products: list[str] = ["A", "B", "C", "D"]

    data: list[dict[str, Any]] = []
    for _ in range(num_records):
        country: str = random.choice(countries)
        product: str = random.choice(products)

        sales: float
        if country in ["USA", "Canada"]:
            sales = rng.normal(100, 10)
        else:
            sales = rng.choice(
                [rng.normal(500, 50), rng.normal(20, 5)],
            )

        data.append({"Country": country, "Product": product, "Sales": max(0, sales)})

    return pd.DataFrame(data)


def create_dataframe_schema(df: pd.DataFrame) -> dict[str, Any]:
    """
    Create a schema dictionary from a DataFrame for validation purposes.

    Args:
        df: DataFrame to analyze

    Returns:
        Dictionary containing schema information
    """
    schema = {
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "shape": df.shape,
        "nullable_columns": df.isnull().any().to_dict(),
    }

    # Add categorical information for object columns
    categorical_info = {}
    for col in df.select_dtypes(include=["object", "category"]).columns:
        unique_values = df[col].unique()
        if len(unique_values) <= 50:  # Only store if reasonable number of categories
            categorical_info[col] = list(unique_values)
        else:
            categorical_info[col] = [f"Too many categories: {len(unique_values)}"]

    if categorical_info:
        schema["categorical_info"] = categorical_info

    return schema


def validate_dataframe_schema(
    df: pd.DataFrame,
    expected_schema: dict[str, Any],
    allow_extra_columns: bool = False,
    allow_missing_columns: bool = False,
) -> None:
    """
    Validate DataFrame schema against expected schema.

    Args:
        df: DataFrame to validate
        expected_schema: Expected schema dictionary
        allow_extra_columns: Whether to allow extra columns in df
        allow_missing_columns: Whether to allow missing columns in df

    Raises:
        ValueError: If schema validation fails
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError(f"Expected pandas DataFrame, got {type(df).__name__}")

    if df.empty:
        raise ValueError("DataFrame cannot be empty")

    expected_columns = set(expected_schema.get("columns", []))
    actual_columns = set(df.columns)

    # Check for missing columns
    missing_columns = expected_columns - actual_columns
    if missing_columns and not allow_missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    # Check for extra columns
    extra_columns = actual_columns - expected_columns
    if extra_columns and not allow_extra_columns:
        raise ValueError(f"Unexpected columns found: {sorted(extra_columns)}")

    # Check data types for common columns
    common_columns = expected_columns & actual_columns
    expected_dtypes = expected_schema.get("dtypes", {})

    for col in common_columns:
        if col in expected_dtypes:
            expected_dtype = expected_dtypes[col]
            actual_dtype = str(df[col].dtype)

            # Allow some flexibility in numeric types
            if _are_compatible_dtypes(expected_dtype, actual_dtype):
                continue

            if expected_dtype != actual_dtype:
                raise ValueError(
                    f"Column '{col}' has incorrect type. "
                    f"Expected: {expected_dtype}, got: {actual_dtype}",
                )


def _are_compatible_dtypes(expected: str, actual: str) -> bool:
    """
    Check if two data types are compatible for schema validation.

    Args:
        expected: Expected data type string
        actual: Actual data type string

    Returns:
        True if types are compatible
    """
    # Numeric type compatibility
    numeric_types = {
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "float16",
        "float32",
        "float64",
    }

    # If both are numeric, they're compatible
    if expected in numeric_types and actual in numeric_types:
        return True

    # Object and string types are compatible
    string_types = {"object", "string"}
    if expected in string_types and actual in string_types:
        return True

    # Exact match
    return expected == actual
