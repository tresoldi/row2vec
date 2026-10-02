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
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import ModuleType

__all__ = ["NeuralBackendMissing", "neural_available", "require_tensorflow"]


class NeuralBackendMissing(ImportError):
    """A neural feature was used but TensorFlow is not installed."""


_INSTALL_HINT = 'pip install "row2vec[neural]"'


def install_hint() -> str:
    """How to get the neural backend on this interpreter."""
    if sys.version_info >= (3, 14):
        return (
            f"{_INSTALL_HINT} (TensorFlow publishes no wheels for Python "
            f"{sys.version_info.major}.{sys.version_info.minor} yet; use Python 3.10-3.13 "
            "for the neural modes)"
        )
    return _INSTALL_HINT


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
            f"{feature} needs TensorFlow, which is not installed. Install it with: {install_hint()}"
        ) from exc
