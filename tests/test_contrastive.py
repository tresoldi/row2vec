"""
Tests for contrastive embedding mode.

Covers the two ways of supplying pairs (explicit lists and the ``auto_pairs``
strategies), both loss functions, and the parameter validation that guards them.
"""

import numpy as np
import pandas as pd
import pytest

from row2vec import learn_embedding


@pytest.fixture
def structured_data() -> pd.DataFrame:
    """A frame with a category whose members have a shifted feature1."""
    np.random.seed(42)
    n_samples = 100
    category = np.random.choice(["A", "B", "C"], n_samples)
    feature1 = np.random.randn(n_samples)
    return pd.DataFrame(
        {
            # Category A sits higher on feature1, giving the pairs something to find.
            "feature1": np.where(category == "A", feature1 + 2, feature1),
            "feature2": np.random.randn(n_samples),
            "category": category,
        }
    )


def test_manual_pairs(structured_data: pd.DataFrame) -> None:
    """Explicit similar/dissimilar pairs produce an embedding of the right shape."""
    embeddings = learn_embedding(
        structured_data,
        mode="contrastive",
        similar_pairs=[(0, 1), (2, 3), (4, 5)],
        dissimilar_pairs=[(0, 10), (1, 15), (2, 20)],
        contrastive_loss="contrastive",
        embedding_dim=3,
        max_epochs=5,
        verbose=False,
    )

    assert embeddings.shape == (len(structured_data), 3)
    assert not embeddings.isna().to_numpy().any()


@pytest.mark.parametrize("strategy", ["cluster", "categorical", "neighbors", "random"])
def test_auto_pairs_strategies(structured_data: pd.DataFrame, strategy: str) -> None:
    """Every automatic pair-generation strategy produces a usable embedding."""
    embeddings = learn_embedding(
        structured_data,
        mode="contrastive",
        auto_pairs=strategy,
        embedding_dim=2,
        max_epochs=3,
        verbose=False,
    )

    assert embeddings.shape == (len(structured_data), 2)


@pytest.mark.parametrize("loss_type", ["triplet", "contrastive"])
def test_loss_functions(structured_data: pd.DataFrame, loss_type: str) -> None:
    """Both contrastive objectives train without error."""
    embeddings = learn_embedding(
        structured_data,
        mode="contrastive",
        auto_pairs="random",
        contrastive_loss=loss_type,
        embedding_dim=2,
        max_epochs=2,
        verbose=False,
    )

    assert embeddings.shape == (len(structured_data), 2)


@pytest.fixture
def tiny_data() -> pd.DataFrame:
    """A small frame, but large enough to reach contrastive parameter validation.

    A frame smaller than the default `batch_size` trips the batch-size check
    first, which is not what these tests are about.
    """
    return pd.DataFrame({"x": list(range(80)), "y": list(range(80))})


def test_invalid_contrastive_loss_raises(tiny_data: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="contrastive_loss"):
        learn_embedding(
            tiny_data,
            mode="contrastive",
            auto_pairs="random",
            contrastive_loss="invalid",
        )


def test_invalid_auto_pairs_raises(tiny_data: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="auto_pairs"):
        learn_embedding(tiny_data, mode="contrastive", auto_pairs="invalid")


def test_missing_pairs_raises(tiny_data: pd.DataFrame) -> None:
    """Contrastive mode needs either explicit pairs or a strategy to derive them."""
    with pytest.raises(ValueError, match="must provide either"):
        learn_embedding(tiny_data, mode="contrastive")
