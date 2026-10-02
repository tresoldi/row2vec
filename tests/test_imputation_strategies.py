"""Every explicit imputation strategy, and the config that rejects bad ones."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from row2vec import AdaptiveImputer, ImputationConfig


@pytest.fixture
def gappy() -> pd.DataFrame:
    rng = np.random.default_rng(1305)
    df = pd.DataFrame(
        {
            "x": rng.normal(size=80),
            "y": rng.normal(size=80),
            "c": rng.choice(["a", "b", "c"], size=80),
        }
    )
    df.loc[rng.choice(80, 12, replace=False), "x"] = np.nan
    df.loc[rng.choice(80, 12, replace=False), "c"] = None
    return df


@pytest.mark.parametrize("numeric", ["mean", "median", "knn", "iterative"])
@pytest.mark.parametrize("categorical", ["mode", "constant", "missing_category"])
def test_explicit_strategies_fill_every_gap(
    gappy: pd.DataFrame, numeric: str, categorical: str
) -> None:
    config = ImputationConfig(numeric_strategy=numeric, categorical_strategy=categorical)

    filled = AdaptiveImputer(config).fit_transform(gappy)

    assert not filled.isna().any().any()
    assert len(filled) == len(gappy)


def test_report_names_the_imputer_applied_to_each_column(gappy: pd.DataFrame) -> None:
    imputer = AdaptiveImputer(ImputationConfig(numeric_strategy="median"))
    imputer.fit(gappy)

    report = imputer.get_imputation_report()

    assert report["applied_strategies"]["x"] == "SimpleImputer"
    assert report["missing_indicators_added"] == []


def test_report_before_fit_says_so() -> None:
    assert "error" in AdaptiveImputer(ImputationConfig()).get_imputation_report()


def test_all_missing_column_warns_and_is_left_alone() -> None:
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "empty": [np.nan] * 4})

    with pytest.warns(UserWarning, match="all missing"):
        AdaptiveImputer(ImputationConfig()).fit(df)


def test_high_missingness_warns() -> None:
    df = pd.DataFrame({"x": [1.0] * 10 + [np.nan] * 90, "y": range(100)})

    with pytest.warns(UserWarning, match="High missingness"):
        AdaptiveImputer(ImputationConfig()).fit(df)


@pytest.mark.parametrize(
    "kwargs", [{"numeric_strategy": "bogus"}, {"categorical_strategy": "bogus"}]
)
def test_unknown_strategies_are_rejected(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="must be one of"):
        ImputationConfig(**kwargs)
