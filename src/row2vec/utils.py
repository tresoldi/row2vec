"""Utility helpers for synthetic data, dtype classification, and DataFrame schemas."""

import random
from collections.abc import Hashable
from typing import Any

import numpy as np
import pandas as pd


def is_categorical_series(series: pd.Series) -> bool:
    """Report whether a column should be treated as categorical.

    Row2Vec splits every column into four buckets — numeric (scaled),
    categorical (encoded), boolean (0/1) and datetime (cyclical features plus
    a trend). This is the single place that decides whether a column is
    categorical; see also :func:`numeric_columns`, :func:`boolean_columns` and
    :func:`datetime_columns`.

    A plain ``series.dtype in ("object", "category")`` test is not sufficient:
    since pandas 3.0 a column of text is inferred as ``str`` (a ``StringDtype``)
    rather than ``object``, so such a test silently routes text into the numeric
    branch and the first reduction on it raises. This predicate recognises
    object, categorical, and string dtypes under both pandas 2 and 3.

    Args:
        series (pd.Series): The column to classify.

    Returns:
        bool: True if the column is categorical rather than numeric.

    Examples:
        >>> import pandas as pd
        >>> from row2vec.utils import is_categorical_series
        >>> is_categorical_series(pd.Series(["a", "b"]))
        True
        >>> is_categorical_series(pd.Series([1.0, 2.0]))
        False
    """
    dtype = series.dtype
    return bool(
        isinstance(dtype, pd.CategoricalDtype)
        or dtype == np.dtype("O")
        or pd.api.types.is_string_dtype(dtype)
    )


def categorical_columns(df: pd.DataFrame) -> list[Hashable]:
    """Return the names of the categorical columns of a DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to inspect.

    Returns:
        list[Hashable]: Column labels classified as categorical, in column order.

    Examples:
        >>> import pandas as pd
        >>> from row2vec.utils import categorical_columns
        >>> categorical_columns(pd.DataFrame({"n": [1], "c": ["x"]}))
        ['c']
    """
    return [col for col in df.columns if is_categorical_series(df[col])]


def numeric_columns(df: pd.DataFrame) -> list[Hashable]:
    """Return the names of the numeric columns of a DataFrame.

    Booleans and datetimes are deliberately excluded: they are neither scaled
    like numbers nor encoded like categories. See :func:`boolean_columns` and
    :func:`datetime_columns`.

    Args:
        df (pd.DataFrame): The DataFrame to inspect.

    Returns:
        list[Hashable]: Column labels classified as numeric, in column order.

    Examples:
        >>> import pandas as pd
        >>> from row2vec.utils import numeric_columns
        >>> numeric_columns(pd.DataFrame({"n": [1], "c": ["x"]}))
        ['n']
    """
    # The labels themselves, not str() of them: a DataFrame may legitimately
    # have integer column names, and stringifying them made every later
    # df[cols] lookup fail with a KeyError.
    return list(df.select_dtypes(include=[np.number]).columns)


def boolean_columns(df: pd.DataFrame) -> list[Hashable]:
    """Return the names of the boolean columns (``bool`` and nullable ``boolean``).

    Args:
        df (pd.DataFrame): The DataFrame to inspect.

    Returns:
        list[Hashable]: Column labels classified as boolean, in column order.

    Examples:
        >>> import pandas as pd
        >>> from row2vec.utils import boolean_columns
        >>> boolean_columns(pd.DataFrame({"n": [1], "b": [True]}))
        ['b']
    """
    return [col for col in df.columns if pd.api.types.is_bool_dtype(df[col].dtype)]


def datetime_columns(df: pd.DataFrame) -> list[Hashable]:
    """Return the names of the datetime columns (naive or timezone-aware).

    Args:
        df (pd.DataFrame): The DataFrame to inspect.

    Returns:
        list[Hashable]: Column labels classified as datetime, in column order.

    Examples:
        >>> import pandas as pd
        >>> from row2vec.utils import datetime_columns
        >>> datetime_columns(pd.DataFrame({"n": [1], "t": pd.to_datetime(["2024-01-01"])}))
        ['t']
    """
    return [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col].dtype)]


def generate_synthetic_data(num_records: int, seed: int = 1305) -> pd.DataFrame:
    """Generates a synthetic DataFrame for demonstration purposes.

    Args:
        num_records (int): The number of records to generate.
        seed (int): A random seed for reproducibility.

    Returns:
        pd.DataFrame: A synthetic DataFrame with mixed data types.
    """
    # A private Random instance: seeding the global `random` module mutated
    # the caller's process state as a side effect of asking for sample data.
    py_random = random.Random(seed)
    rng = np.random.default_rng(seed)

    countries: list[str] = ["USA", "Canada", "Mexico", "Brazil", "Italy"]
    products: list[str] = ["A", "B", "C", "D"]

    data: list[dict[str, Any]] = []
    for _ in range(num_records):
        country: str = py_random.choice(countries)
        product: str = py_random.choice(products)

        sales: float
        if country in ["USA", "Canada"]:
            sales = rng.normal(100, 10)
        else:
            sales = float(
                rng.choice(
                    [rng.normal(500, 50), rng.normal(20, 5)],
                )
            )

        data.append({"Country": country, "Product": product, "Sales": max(0, sales)})

    return pd.DataFrame(data)


def create_dataframe_schema(df: pd.DataFrame) -> dict[str, Any]:
    """Create a schema dictionary from a DataFrame for validation purposes.

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
    for col in categorical_columns(df):
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
    """Validate DataFrame schema against expected schema.

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
    """Check if two data types are compatible for schema validation.

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

    # Object and string types are compatible. "str" is what pandas 3.0 reports
    # for an inferred text column, where pandas 2 reported "object".
    string_types = {"object", "string", "str"}
    if expected in string_types and actual in string_types:
        return True

    # Exact match
    return expected == actual
