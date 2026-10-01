"""Lazy access to the optional neural backend (TensorFlow / Keras).

``pip install row2vec`` gives PCA, t-SNE and UMAP. The autoencoder, target,
and contrastive modes and entity-embedding categorical encoding train Keras
models, so they need ``pip install "row2vec[neural]"``. Nothing in the package
imports TensorFlow at module load; code that needs it calls
:func:`require_tensorflow` at the point of use, which either returns the
module or raises one error that names the extra to install.
"""

from __future__ import annotations

import importlib
import importlib.util
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import ModuleType

__all__ = ["NeuralBackendMissing", "neural_available", "require_tensorflow"]


class NeuralBackendMissing(ImportError):
    """A neural feature was used but TensorFlow is not installed."""


_INSTALL_HINT = 'pip install "row2vec[neural]"'


def neural_available() -> bool:
    """Whether TensorFlow is installed, without importing it."""
    return importlib.util.find_spec("tensorflow") is not None


def require_tensorflow(feature: str) -> ModuleType:
    """Import and return ``tensorflow``, or explain how to get it.

    Args:
        feature (str): What needs it, for the error message
            (e.g. ``"mode='unsupervised'"``).

    Raises:
        ImportError: If TensorFlow is not installed.
    """
    try:
        return importlib.import_module("tensorflow")
    except ImportError as exc:
        raise NeuralBackendMissing(
            f"{feature} needs TensorFlow, which is not installed. Install it with: {_INSTALL_HINT}"
        ) from exc
