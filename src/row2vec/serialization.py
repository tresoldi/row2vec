"""Saving and loading trained models.

A saved model is a single ``.r2v`` file: a zip archive holding

* ``manifest.json`` - format version, versions of the libraries that wrote it, the
  training metadata, and a SHA-256 for every other member. Plain JSON: read it
  with :func:`inspect_model` or any zip tool, without loading the model.
* ``config.json`` and ``state.json`` - the configuration and the fitted
  bookkeeping (columns, dtypes, schema, training history), as tagged JSON.
* ``preprocessor.skops`` and ``scaler.skops`` (and ``projector.skops`` for PCA,
  UMAP and t-SNE) - scikit-learn objects written with
  `skops <https://skops.readthedocs.io>`_.
* ``encoder.keras`` - the trained encoder of a neural mode, in Keras's own format.

**Loading never executes code from the file.** There is no loader script and no
pickle. skops reconstructs only types on an allow-list kept in this module and
refuses the file if it names anything else; Keras loads with ``safe_mode`` and
only layers on a short allow-list. This protects against a *malicious* model
file. It does not authenticate one: the SHA-256 values catch corruption, not an
attacker who rewrites both the member and its hash. See ``SECURITY.md``.
"""

from __future__ import annotations

import copy
import hashlib
import io
import json
import os
import tempfile
import zipfile
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ._backend import require_tensorflow
from .config import EmbeddingConfig
from .core import learn_embedding_with_model
from .model import Row2VecModel, get_feature_names


def _package_version() -> str:
    """The running package version.

    Imported lazily: ``row2vec/__init__.py`` is the single source of truth for
    the version (pyproject reads it from there), and it imports this module, so
    a module-level import would be circular. This used to be the string
    "0.1.0", hardcoded, which meant every saved model misreported the version
    that produced it.
    """
    from row2vec import __version__

    return __version__


class Row2VecModelMetadata:
    """Container for Row2Vec model training metadata."""

    def __init__(
        self,
        # Training parameters
        embedding_dim: int,
        mode: str,
        reference_column: str | None = None,
        max_epochs: int = 50,
        batch_size: int = 64,
        dropout_rate: float = 0.2,
        hidden_units: int | list[int] = 128,
        early_stopping: bool = True,
        seed: int = 1305,
        scale_method: str | None = None,
        scale_range: tuple[float, float] | None = None,
        # Classical ML parameters
        n_neighbors: int = 15,
        perplexity: float = 30.0,
        min_dist: float = 0.1,
        n_iter: int = 1000,
        # Training results
        training_history: dict[str, Any] | None = None,
        final_loss: float | None = None,
        epochs_trained: int | None = None,
        training_time: float | None = None,
        # Data information
        original_columns: list[str] | None = None,
        preprocessed_feature_names: list[str] | None = None,
        data_shape: tuple[int, int] | None = None,
        data_types: dict[str, str] | None = None,
        # Schema validation
        expected_schema: dict[str, Any] | None = None,
    ):
        # Training configuration
        self.embedding_dim = embedding_dim
        self.mode = mode
        self.reference_column = reference_column
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.dropout_rate = dropout_rate
        self.hidden_units = hidden_units
        self.early_stopping = early_stopping
        self.seed = seed
        self.scale_method = scale_method
        self.scale_range = scale_range

        # Classical ML parameters
        self.n_neighbors = n_neighbors
        self.perplexity = perplexity
        self.min_dist = min_dist
        self.n_iter = n_iter

        # Training results
        self.training_history = training_history or {}
        self.final_loss = final_loss
        self.epochs_trained = epochs_trained
        self.training_time = training_time

        # Data information
        self.original_columns = original_columns or []
        self.preprocessed_feature_names = preprocessed_feature_names or []
        self.data_shape = data_shape
        self.data_types = data_types or {}

        # Schema validation
        self.expected_schema = expected_schema or {}

        # Metadata
        self.created_at = datetime.now().isoformat()
        self.row2vec_version = _package_version()

    def to_dict(self) -> dict[str, Any]:
        """Convert metadata to dictionary for serialization."""
        return {
            # Training configuration
            "embedding_dim": self.embedding_dim,
            "mode": self.mode,
            "reference_column": self.reference_column,
            "max_epochs": self.max_epochs,
            "batch_size": self.batch_size,
            "dropout_rate": self.dropout_rate,
            "hidden_units": self.hidden_units,
            "early_stopping": self.early_stopping,
            "seed": self.seed,
            "scale_method": self.scale_method,
            "scale_range": self.scale_range,
            # Classical ML parameters
            "n_neighbors": self.n_neighbors,
            "perplexity": self.perplexity,
            "min_dist": self.min_dist,
            "n_iter": self.n_iter,
            # Training results
            "training_history": self.training_history,
            "final_loss": self.final_loss,
            "epochs_trained": self.epochs_trained,
            "training_time": self.training_time,
            # Data information
            "original_columns": self.original_columns,
            "preprocessed_feature_names": self.preprocessed_feature_names,
            "data_shape": self.data_shape,
            "data_types": self.data_types,
            # Schema validation
            "expected_schema": self.expected_schema,
            # Metadata
            "created_at": self.created_at,
            "row2vec_version": self.row2vec_version,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Row2VecModelMetadata:
        """Create metadata from dictionary."""
        # Extract only the parameters that the constructor accepts
        constructor_params = {
            "embedding_dim",
            "mode",
            "reference_column",
            "max_epochs",
            "batch_size",
            "dropout_rate",
            "hidden_units",
            "early_stopping",
            "seed",
            "scale_method",
            "scale_range",
            "n_neighbors",
            "perplexity",
            "min_dist",
            "n_iter",
            "training_history",
            "final_loss",
            "epochs_trained",
            "training_time",
            "original_columns",
            "preprocessed_feature_names",
            "data_shape",
            "data_types",
            "expected_schema",
        }

        # Filter the data to only include constructor parameters
        filtered_data = {k: v for k, v in data.items() if k in constructor_params}

        # Create the instance
        instance = cls(**filtered_data)

        # Set additional fields that are not constructor parameters
        if "created_at" in data:
            instance.created_at = data["created_at"]
        if "row2vec_version" in data:
            instance.row2vec_version = data["row2vec_version"]

        return instance


def describe_model(
    model: Row2VecModel,
    *,
    include_training_history: bool = True,
) -> Row2VecModelMetadata:
    """Describe a fitted model for the saved loader script.

    Everything needed is recorded on the model during ``fit``, so the training
    frame does not have to be kept around or passed in again.

    Args:
        model (Row2VecModel): The fitted model.
        include_training_history (bool): Whether to keep the per-epoch history.
            Dropping it makes the saved script considerably smaller.

    Returns:
        Row2VecModelMetadata: Metadata describing the training run.
    """
    config = model.config
    if not model.is_fitted:
        raise ValueError("Cannot describe an unfitted Row2VecModel.")
    assert model.preprocessor_ is not None

    return Row2VecModelMetadata(
        embedding_dim=config.embedding_dim,
        mode=config.mode,
        reference_column=config.reference_column,
        max_epochs=config.neural.max_epochs,
        batch_size=config.neural.batch_size,
        dropout_rate=config.neural.dropout_rate,
        hidden_units=config.neural.hidden_units,
        early_stopping=config.neural.early_stopping,
        seed=config.seed,
        scale_method=config.scaling.method,
        scale_range=config.scaling.range,
        n_neighbors=config.classical.n_neighbors,
        perplexity=config.classical.perplexity,
        min_dist=config.classical.min_dist,
        n_iter=config.classical.n_iter,
        training_history=model.training_history_ if include_training_history else {},
        final_loss=model.final_loss_,
        epochs_trained=model.epochs_trained_,
        training_time=model.training_time_,
        original_columns=[str(c) for c in model.training_columns_],
        preprocessed_feature_names=get_feature_names(model.preprocessor_),
        data_shape=model.training_shape_,
        data_types=model.training_dtypes_,
        expected_schema=model.training_schema_,
    )


# --------------------------------------------------------------------------- #
# The file format
# --------------------------------------------------------------------------- #

FORMAT_NAME = "row2vec-model"
FORMAT_VERSION = 1
MODEL_SUFFIX = ".r2v"

_MANIFEST = "manifest.json"
_STATE = "state.json"
_CONFIG = "config.json"
_PREPROCESSOR = "preprocessor.skops"
_SCALER = "scaler.skops"
_PROJECTOR = "projector.skops"
_ENCODER = "encoder.keras"
_REQUIRED = frozenset({_MANIFEST, _STATE, _CONFIG, _PREPROCESSOR, _SCALER})
_ALLOWED_MEMBERS = _REQUIRED | {_PROJECTOR, _ENCODER}
_LEGACY_SUFFIXES = (".py", ".pkl")

#: Largest member we will read, as a guard against a decompression bomb.
_MAX_MEMBER_BYTES = 2 * 1024**3

#: Types skops may reconstruct beyond the scikit-learn / NumPy ones it trusts by
#: default. Exact names rather than a prefix: this list is the whole of what a
#: model file can make the loader instantiate.
TRUSTED_TYPES = frozenset(
    {
        "row2vec.categorical_encoding.CategoricalAnalyzer",
        "row2vec.categorical_encoding.CategoricalEncoder",
        "row2vec.categorical_encoding.CategoricalEncodingConfig",
        "row2vec.categorical_encoding.TargetEncoder",
        "row2vec.imputation.AdaptiveImputer",
        "row2vec.imputation.ImputationConfig",
        "row2vec.imputation.MissingPatternAnalyzer",
        "row2vec.model.EmbeddingScaler",
        "row2vec.pipeline_builder._TolerantMinMaxScaler",
        "numpy.dtype",
        "scipy.sparse._csr.csr_matrix",
        "umap.umap_.UMAP",
    }
)

#: Layers an encoder may contain. Row2Vec builds Dense/Dropout stacks only.
_ALLOWED_KERAS_LAYERS = frozenset({"InputLayer", "Dense", "Dropout"})


class ModelFormatError(ValueError):
    """A saved model is not in a form that can be loaded safely."""


# -- tagged JSON ------------------------------------------------------------ #
#
# Plain JSON loses tuples, non-string dict keys and NaN, all of which turn up in
# schemas and training histories. These tags put them back. Decoding only builds
# builtin containers, numbers, strings and pandas Timestamps: it never calls
# anything named in the file.


def _encode(obj: Any) -> Any:
    if obj is None or isinstance(obj, bool | int | str):
        return obj
    if isinstance(obj, float):
        if np.isnan(obj):
            return {"__float__": "nan"}
        if np.isinf(obj):
            return {"__float__": "inf" if obj > 0 else "-inf"}
        return obj
    if isinstance(obj, np.generic):
        return _encode(obj.item())
    if isinstance(obj, np.ndarray):
        return _encode(obj.tolist())
    if isinstance(obj, tuple):
        return {"__tuple__": [_encode(v) for v in obj]}
    if isinstance(obj, list | set | frozenset):
        return [_encode(v) for v in obj]
    if isinstance(obj, dict):
        if all(isinstance(k, str) and not k.startswith("__") for k in obj):
            return {k: _encode(v) for k, v in obj.items()}
        return {"__dict__": [[_encode(k), _encode(v)] for k, v in obj.items()]}
    if isinstance(obj, pd.Timestamp | datetime | date):
        return {"__datetime__": obj.isoformat()}
    if isinstance(obj, Path):
        return str(obj)
    # Anything else (a stray object in a schema) is recorded as its text rather
    # than blocking the save; none of it is read back as anything but text.
    return str(obj)


def _decode(obj: Any) -> Any:
    if isinstance(obj, list):
        return [_decode(v) for v in obj]
    if isinstance(obj, dict):
        if set(obj) == {"__float__"}:
            return float(obj["__float__"])
        if set(obj) == {"__tuple__"}:
            return tuple(_decode(v) for v in obj["__tuple__"])
        if set(obj) == {"__dict__"}:
            return {_hashable(_decode(k)): _decode(v) for k, v in obj["__dict__"]}
        if set(obj) == {"__datetime__"}:
            return pd.Timestamp(obj["__datetime__"])
        return {k: _decode(v) for k, v in obj.items()}
    return obj


def _hashable(key: Any) -> Any:
    return tuple(key) if isinstance(key, list) else key


def _dumps_json(obj: Any) -> bytes:
    return json.dumps(_encode(obj), indent=2, sort_keys=False, allow_nan=False).encode("utf-8")


# -- skops and Keras components --------------------------------------------- #

_UMAP_DISTANCE_ATTRS = ("_input_distance_func", "_inverse_distance_func", "_output_distance_func")


def _skops():  # type: ignore[no-untyped-def]
    try:
        import skops.io as sio
    except ImportError as exc:  # pragma: no cover - skops is a core dependency
        raise ImportError("Saving and loading models needs skops: pip install skops") from exc
    return sio


def _dump_sklearn(obj: Any) -> bytes:
    """Serialise a scikit-learn style object with skops."""
    if type(obj).__module__.startswith("umap"):
        obj = _strip_umap(obj)
    return bytes(_skops().dumps(obj))


def _load_sklearn(blob: bytes, what: str) -> Any:
    """Load a skops blob, refusing any type that is not on the allow-list."""
    sio = _skops()
    try:
        named = set(sio.get_untrusted_types(data=blob))
    except Exception as exc:
        raise ModelFormatError(f"{what} is not a valid skops file: {exc}") from exc
    unexpected = named - TRUSTED_TYPES
    if unexpected:
        raise ModelFormatError(
            f"{what} refers to types this version of row2vec does not trust, so it "
            f"was not loaded: {sorted(unexpected)}. A file written by row2vec only "
            "names the types in row2vec.serialization.TRUSTED_TYPES; this one was "
            "either written by a different version or has been tampered with."
        )
    obj = sio.loads(blob, trusted=sorted(named))
    if type(obj).__module__.startswith("umap"):
        _restore_umap(obj)
    return obj


def _strip_umap(model: Any) -> Any:
    """A shallow copy of a UMAP model without its compiled distance functions.

    Those three attributes are numba functions, which cannot be serialised; they
    are derived from the metric names, so :func:`_restore_umap` rebuilds them.
    Only named metrics round-trip; a callable metric cannot be stored safely.
    """
    for attr in ("metric", "output_metric"):
        if not isinstance(getattr(model, attr, None), str):
            raise ModelFormatError(
                f"UMAP was fitted with a callable {attr}, which cannot be saved safely. "
                "Use a named metric."
            )
    stripped = copy.copy(model)
    for attr in _UMAP_DISTANCE_ATTRS:
        stripped.__dict__.pop(attr, None)
    return stripped


def _restore_umap(model: Any) -> None:
    try:
        import umap.distances as dist

        model._input_distance_func = dist.named_distances[model.metric]
        model._inverse_distance_func = dist.named_distances_with_gradients.get(model.metric)
        model._output_distance_func = dist.named_distances_with_gradients[model.output_metric]
    except (ImportError, KeyError, AttributeError) as exc:
        import umap

        raise ModelFormatError(
            "This UMAP model could not be rebuilt with the installed umap-learn "
            f"({getattr(umap, '__version__', 'unknown')}); it was saved with a "
            "different version whose internals differ. Retrain the model, or "
            "install the umap-learn version recorded in the file's manifest."
        ) from exc


def _dump_keras(encoder: Any) -> bytes:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "encoder.keras"
        encoder.save(path)
        return path.read_bytes()


def _load_keras(blob: bytes) -> Any:
    _check_keras_layers(blob)
    require_tensorflow("Loading a neural model")
    from tensorflow.keras.models import load_model as keras_load

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "encoder.keras"
        path.write_bytes(blob)
        return keras_load(path, compile=False, safe_mode=True)


def _check_keras_layers(blob: bytes) -> None:
    """Refuse an encoder that contains anything but plain Dense/Dropout layers."""
    try:
        with zipfile.ZipFile(io.BytesIO(blob)) as archive:
            config = json.loads(archive.read("config.json"))
        layers = config["config"]["layers"]
        kinds = {layer["class_name"] for layer in layers}
        modules = {layer.get("module", "") for layer in layers}
    except (KeyError, ValueError, zipfile.BadZipFile) as exc:
        raise ModelFormatError(f"The encoder is not a valid Keras file: {exc}") from exc
    if config.get("class_name") != "Functional" or not kinds <= _ALLOWED_KERAS_LAYERS:
        raise ModelFormatError(
            "The encoder contains layers row2vec does not write "
            f"({sorted(kinds - _ALLOWED_KERAS_LAYERS)}); it was not loaded."
        )
    if any(not m.startswith("keras") for m in modules):
        raise ModelFormatError(f"The encoder names layers outside Keras: {sorted(modules)}")


# -- paths ------------------------------------------------------------------ #


def _model_path(path: str | Path) -> Path:
    """``model`` and ``model.r2v`` both mean the same file."""
    path = Path(path)
    if path.suffix in _LEGACY_SUFFIXES:
        raise ModelFormatError(
            f"{path.name} looks like the script-and-pickle format of earlier "
            "development versions. That format ran code from the file when loaded and is no "
            f"longer supported (nothing was executed). Retrain the model and save it as a "
            f"{MODEL_SUFFIX} file."
        )
    return path if path.suffix == MODEL_SUFFIX else path.with_name(path.name + MODEL_SUFFIX)


def _libraries() -> dict[str, str]:
    import sklearn

    found = {"numpy": np.__version__, "pandas": pd.__version__, "scikit-learn": sklearn.__version__}
    for name in ("skops", "umap", "tensorflow", "keras"):
        try:
            module = __import__(name)
            found[name] = str(getattr(module, "__version__", "unknown"))
        except ImportError:
            continue
    return found


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


def save_model(
    model: Row2VecModel,
    base_path: str | Path,
    overwrite: bool = False,
) -> str:
    """Save a fitted model as a single ``.r2v`` file.

    Args:
        model (Row2VecModel): The fitted model.
        base_path (str | Path): Where to write it. ``.r2v`` is appended unless
            already present.
        overwrite (bool): Whether to replace an existing file.

    Returns:
        str: The path of the file written.

    Raises:
        FileExistsError: If the file exists and ``overwrite`` is false.
        ValueError: If the model is not fitted.
        ModelFormatError: If the model contains something that cannot be saved
            safely, such as a UMAP model fitted with a custom metric.

    Examples:
        >>> import tempfile, row2vec
        >>> df = row2vec.generate_synthetic_data(60)
        >>> _, model = row2vec.learn_embedding_with_model(
        ...     df, mode="pca", embedding_dim=2, enable_logging=False
        ... )
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     path = row2vec.save_model(model, tmp + "/demo")
        ...     restored = row2vec.load_model(path)
        ...     path.endswith(".r2v"), restored.predict(df).shape
        (True, (60, 2))
    """
    path = _model_path(base_path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Model file already exists: {path}")
    if not model.is_fitted:
        raise ValueError("Model must be fitted before saving")
    if model.metadata is None:
        model.metadata = describe_model(model)

    state = model.to_state()
    members: dict[str, bytes] = {
        _CONFIG: _dumps_json(state["config"].to_dict()),
        _STATE: _dumps_json(
            {
                key: state[key]
                for key in (
                    "aggregate_by_reference",
                    "feature_names_in_",
                    "n_features_in_",
                    "training_columns_",
                    "training_shape_",
                    "training_dtypes_",
                    "training_schema_",
                    "training_history_",
                    "final_loss_",
                    "epochs_trained_",
                    "training_time_",
                )
            }
        ),
        _PREPROCESSOR: _dump_sklearn(state["preprocessor_"]),
        _SCALER: _dump_sklearn(state["embedding_scaler_"]),
    }
    if state["encoder_"] is not None:
        # A neural model needs only its encoder to embed rows; the full training
        # network (decoder, classifier head, siamese wrapper) is not saved.
        members[_ENCODER] = _dump_keras(state["encoder_"])
    elif state["projector_"] is not None:
        members[_PROJECTOR] = _dump_sklearn(state["projector_"])

    manifest = {
        "format": FORMAT_NAME,
        "format_version": FORMAT_VERSION,
        "row2vec_version": _package_version(),
        "created_at": datetime.now().isoformat(),
        "mode": model.config.mode,
        "embedding_dim": model.config.embedding_dim,
        "libraries": _libraries(),
        "members": {name: hashlib.sha256(blob).hexdigest() for name, blob in members.items()},
        "metadata": model.metadata.to_dict(),
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    # Write beside the target and move into place, so an interrupted save never
    # leaves a half-written model where a good one used to be.
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with (
            os.fdopen(fd, "wb") as handle,
            zipfile.ZipFile(handle, "w", zipfile.ZIP_DEFLATED) as zf,
        ):
            zf.writestr(_MANIFEST, _dumps_json(manifest))
            for name, blob in members.items():
                zf.writestr(name, blob)
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise
    return str(path)


def _read_archive(path: Path) -> tuple[dict[str, Any], dict[str, bytes]]:
    """Read and verify every member of a model file. Executes nothing."""
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    try:
        archive = zipfile.ZipFile(path)
    except zipfile.BadZipFile as exc:
        raise ModelFormatError(f"{path.name} is not a row2vec model file.") from exc
    with archive:
        names = set(archive.namelist())
        unknown = names - _ALLOWED_MEMBERS
        if unknown:
            raise ModelFormatError(f"{path.name} contains unexpected members: {sorted(unknown)}")
        missing = _REQUIRED - names
        if missing:
            raise ModelFormatError(f"{path.name} is missing required members: {sorted(missing)}")
        for info in archive.infolist():
            if info.file_size > _MAX_MEMBER_BYTES:
                raise ModelFormatError(
                    f"{info.filename} is implausibly large; refusing to read it."
                )
        raw = {name: archive.read(name) for name in names}

    try:
        manifest = json.loads(raw[_MANIFEST])
    except ValueError as exc:
        raise ModelFormatError("manifest.json is not valid JSON.") from exc
    if manifest.get("format") != FORMAT_NAME:
        raise ModelFormatError(f"{path.name} is not a {FORMAT_NAME} file.")
    version = manifest.get("format_version")
    if version != FORMAT_VERSION:
        raise ModelFormatError(
            f"{path.name} uses model format version {version}; this row2vec reads version "
            f"{FORMAT_VERSION}. Upgrade row2vec to read newer files."
        )
    for name, expected in manifest.get("members", {}).items():
        if name not in raw or hashlib.sha256(raw[name]).hexdigest() != expected:
            raise ModelFormatError(
                f"{name} in {path.name} does not match its recorded checksum: the file is "
                "corrupt or has been modified."
            )
    if set(manifest.get("members", {})) != names - {_MANIFEST}:
        raise ModelFormatError(f"{path.name} has members that its manifest does not list.")
    return _decode(manifest), raw


def inspect_model(path: str | Path) -> dict[str, Any]:
    """Read a saved model's manifest without loading the model.

    Nothing from the file is deserialised beyond JSON, so this is safe to call
    on a model you have not decided to trust yet.

    Args:
        path (str | Path): A ``.r2v`` file.

    Returns:
        dict[str, Any]: Format and library versions, mode, embedding dimension,
        and the training metadata (columns, dtypes, history, ...).
    """
    manifest, _ = _read_archive(_model_path(path))
    return manifest


def load_model(path: str | Path) -> Row2VecModel:
    """Load a model written by :func:`save_model`.

    Loading does not execute code from the file; see the module docstring for
    exactly what is and is not guaranteed.

    Args:
        path (str | Path): The ``.r2v`` file (the suffix may be omitted).

    Returns:
        Row2VecModel: The restored model, ready to ``transform``/``predict``.

    Raises:
        FileNotFoundError: If the file does not exist.
        ModelFormatError: If the file is corrupt, from another format version,
            names a type that is not on the allow-list, or is the old
            script-and-pickle format.
    """
    manifest, raw = _read_archive(_model_path(path))
    state = _decode(json.loads(raw[_STATE]))
    state["config"] = EmbeddingConfig.from_dict(_decode(json.loads(raw[_CONFIG])))
    state["preprocessor_"] = _load_sklearn(raw[_PREPROCESSOR], "preprocessor.skops")
    state["embedding_scaler_"] = _load_sklearn(raw[_SCALER], "scaler.skops")
    state["projector_"] = (
        _load_sklearn(raw[_PROJECTOR], "projector.skops") if _PROJECTOR in raw else None
    )
    state["encoder_"] = _load_keras(raw[_ENCODER]) if _ENCODER in raw else None

    model = Row2VecModel.from_state(state)
    model.metadata = Row2VecModelMetadata.from_dict(manifest["metadata"])
    return model


def train_and_save_model(
    df: pd.DataFrame,
    base_path: str | Path,
    # All learn_embedding parameters
    embedding_dim: int = 10,
    mode: str = "unsupervised",
    reference_column: str | None = None,
    max_epochs: int = 50,
    batch_size: int = 64,
    dropout_rate: float = 0.2,
    hidden_units: int = 128,
    early_stopping: bool = True,
    seed: int = 1305,
    verbose: bool = False,
    scale_method: str | None = None,
    scale_range: tuple[float, float] | None = None,
    log_level: str = "INFO",
    log_file: str | None = None,
    enable_logging: bool = True,
    n_neighbors: int = 15,
    perplexity: float = 30.0,
    min_dist: float = 0.1,
    n_iter: int = 1000,
    # Contrastive learning parameters
    similar_pairs: list[tuple[int, int]] | None = None,
    dissimilar_pairs: list[tuple[int, int]] | None = None,
    auto_pairs: str | None = None,
    negative_samples: int = 5,
    contrastive_loss: str = "triplet",
    margin: float = 1.0,
    # Serialization parameters
    overwrite: bool = False,
    include_training_history: bool = True,
) -> tuple[pd.DataFrame, str]:
    """Train a Row2Vec model and save it as a single ``.r2v`` file.

    This is a convenience function that combines training and saving.

    Args:
        df (pd.DataFrame): The input DataFrame containing numeric and categorical features.
        base_path (str | Path): Where to save the model; ``.r2v`` is appended
            unless present.
        embedding_dim (int): The dimensionality of the embedding space.
        mode (str): Embedding method - 'unsupervised' (autoencoder), 'target' (supervised),
                   'pca' (Principal Component Analysis), 'tsne' (t-SNE), 'umap' (UMAP),
                   or 'contrastive' (contrastive learning).
        reference_column (str): The target column for 'target' mode.
        max_epochs (int): The maximum number of training epochs (neural methods only).
        batch_size (int): The batch size for training (neural methods only).
        dropout_rate (float): The dropout rate for regularization (neural methods only).
        hidden_units (Union[int, list[int]]): Hidden layer configuration - single int for one layer
                     or list of ints for multiple layers (neural methods only).
        early_stopping (bool): Whether to use early stopping (neural methods only).
        seed (int): A random seed for reproducibility.
        verbose (bool): Whether to print training progress.
        scale_method (str, optional): Scaling method for embeddings. Options:
                     'none', 'minmax', 'standard', 'l2', 'tanh'.
        scale_range (tuple, optional): Range for minmax scaling. Default: (0, 1).
        log_level (str): Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR').
        log_file (str, optional): File path for logging output.
        enable_logging (bool): Whether to enable structured logging.
        n_neighbors (int): Number of neighbors for UMAP (default: 15).
        perplexity (float): Perplexity parameter for t-SNE (default: 30.0).
        min_dist (float): Minimum distance for UMAP (default: 0.1).
        n_iter (int): Number of iterations for t-SNE (default: 1000).
        similar_pairs (list[tuple[int, int]], optional): List of (row_idx1, row_idx2) pairs
                     that should have similar embeddings (for contrastive mode).
        dissimilar_pairs (list[tuple[int, int]], optional): List of (row_idx1, row_idx2) pairs
                     that should have dissimilar embeddings (for contrastive mode).
        auto_pairs (str, optional): Strategy for automatic pair generation. Options:
                     'cluster' (cluster-based), 'neighbors' (k-NN based),
                     'categorical' (same category values), 'random' (random sampling).
        contrastive_loss (str): Contrastive loss function. Options: 'triplet', 'contrastive'.
        margin (float): Margin parameter for contrastive loss functions (default: 1.0).
        negative_samples (int): Number of negative samples per positive pair (default: 5).
        overwrite (bool): Whether to overwrite existing model files.
        include_training_history (bool): Whether to include the full training history in the metadata.

    Returns:
        Tuple of (embeddings, path), where ``path`` is the ``.r2v`` file written.
    """
    # One training pass produces both the embeddings and the fitted model.
    embeddings, model = learn_embedding_with_model(
        df=df,
        embedding_dim=embedding_dim,
        mode=mode,
        reference_column=reference_column,
        max_epochs=max_epochs,
        batch_size=batch_size,
        dropout_rate=dropout_rate,
        hidden_units=hidden_units,
        early_stopping=early_stopping,
        seed=seed,
        verbose=verbose,
        scale_method=scale_method,
        scale_range=scale_range,
        log_level=log_level,
        log_file=log_file,
        enable_logging=enable_logging,
        n_neighbors=n_neighbors,
        perplexity=perplexity,
        min_dist=min_dist,
        n_iter=n_iter,
        similar_pairs=similar_pairs,
        dissimilar_pairs=dissimilar_pairs,
        auto_pairs=auto_pairs,
        negative_samples=negative_samples,
        contrastive_loss=contrastive_loss,
        margin=margin,
    )

    model.metadata = describe_model(model, include_training_history=include_training_history)

    # Save the model
    path = save_model(model, base_path, overwrite=overwrite)

    return embeddings, path
