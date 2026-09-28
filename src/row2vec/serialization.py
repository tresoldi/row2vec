"""Saving and loading trained models.

This module provides functionality to save and load trained Row2Vec models
with their preprocessing pipelines and training metadata using a transparent
two-file approach (Python script + binary blob).
"""

import pickle
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

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
    def from_dict(cls, data: dict[str, Any]) -> "Row2VecModelMetadata":
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


def save_model(
    model: Row2VecModel,
    base_path: str | Path,
    overwrite: bool = False,
) -> tuple[str, str]:
    """Save a Row2Vec model using the two-file approach.

    Args:
        model: The Row2Vec model to save
        base_path: Base path for saving (without extension)
        overwrite: Whether to overwrite existing files

    Returns:
        Tuple of (script_path, binary_path)

    Raises:
        FileExistsError: If files exist and overwrite=False
        ValueError: If model is incomplete
    """
    base_path = Path(base_path)
    script_path = base_path.with_suffix(".py")
    binary_path = base_path.with_suffix(".pkl")

    # Check for existing files
    if not overwrite:
        if script_path.exists():
            raise FileExistsError(f"Script file already exists: {script_path}")
        if binary_path.exists():
            raise FileExistsError(f"Binary file already exists: {binary_path}")

    # Validate model completeness
    if not model.is_fitted:
        raise ValueError("Model must be fitted before saving")
    if model.metadata is None:
        # Everything metadata needs was recorded during fit.
        model.metadata = describe_model(model)

    # Persist the entire fitted state. Saving only the projector and the
    # preprocessor is what left the embedding scaler behind, so a reloaded
    # model returned differently scaled values than training had.
    binary_data = {"state": model.to_state()}

    with open(binary_path, "wb") as f:
        pickle.dump(binary_data, f)

    # Generate Python script with metadata and loading logic
    script_content = _generate_model_script(model.metadata, binary_path.name)

    with open(script_path, "w", encoding="utf-8") as f:
        f.write(script_content)

    return str(script_path), str(binary_path)


def load_model(script_path: str | Path) -> Row2VecModel:
    """Load a Row2Vec model from the script file.

    Args:
        script_path: Path to the Python script file

    Returns:
        Loaded Row2Vec model

    Raises:
        FileNotFoundError: If script or binary file not found
        ValueError: If loading fails
    """
    script_path = Path(script_path)

    if not script_path.exists():
        raise FileNotFoundError(f"Script file not found: {script_path}")

    # Execute the script in a controlled namespace
    # Add the script directory to help find the binary file
    namespace: dict[str, Any] = {
        "__script_dir__": str(script_path.parent),
        "__script_path__": str(script_path),
    }
    # Row2Vec models are saved as a Python loader script plus a binary blob, so
    # loading one runs its script. Only load models from a trusted source; see
    # SECURITY.md.
    exec(script_path.read_text(encoding="utf-8"), namespace)  # nosec B102

    # Get the load function from the script
    if "load_model" not in namespace:
        raise ValueError("Script does not contain load_model function")

    # Load the model
    loaded = namespace["load_model"]()
    if not isinstance(loaded, Row2VecModel):
        raise TypeError(f"Loader script returned {type(loaded).__name__}, expected Row2VecModel")
    return loaded


def _generate_model_script(metadata: Row2VecModelMetadata, binary_filename: str) -> str:
    """Generate the Python script for model loading.

    Args:
        metadata: Model metadata
        binary_filename: Name of the binary file

    Returns:
        Python script content as string
    """

    # Convert metadata to JSON for inclusion in script, handling None values
    def json_serializer(obj: Any) -> Any:
        """Custom JSON serializer to handle None and other Python objects."""
        import numpy as np

        if obj is None:
            return None
        if isinstance(obj, tuple | list):
            return list(obj)
        if isinstance(obj, np.floating | np.integer):
            if np.isnan(obj):
                return None
            return float(obj)
        return str(obj)

    metadata_dict = metadata.to_dict()
    # Convert None values and handle NaN values
    for key, value in metadata_dict.items():
        if value is None:
            metadata_dict[key] = None  # Keep as Python None
        elif isinstance(value, np.floating | np.integer):
            if np.isnan(value):
                metadata_dict[key] = None
            else:
                metadata_dict[key] = float(value)

    # Use Python repr instead of JSON to handle None values properly
    metadata_repr = repr(metadata_dict)

    # Handle potential NaN values in metadata for f-string formatting
    def safe_format(value: Any) -> Any:
        if value is None:
            return "Not recorded"
        if isinstance(value, np.floating | np.integer) and np.isnan(value):
            return "Not recorded"
        return value

    return f'''"""
Row2Vec Model: {metadata.mode} mode, {metadata.embedding_dim}D embeddings
Created: {metadata.created_at}
Row2Vec Version: {metadata.row2vec_version}

This script contains the metadata and loading logic for a trained Row2Vec model.
The actual model weights and preprocessor are stored in the accompanying binary file.

Training Configuration:
- Mode: {metadata.mode}
- Embedding Dimensions: {metadata.embedding_dim}
- Reference Column: {safe_format(metadata.reference_column)}
- Epochs Trained: {safe_format(metadata.epochs_trained)}
- Final Loss: {safe_format(metadata.final_loss)}
- Training Time: {safe_format(metadata.training_time)}s

Data Information:
- Original Shape: {metadata.data_shape}
- Original Columns: {len(metadata.original_columns)} columns
- Preprocessed Features: {len(metadata.preprocessed_feature_names)} features

Usage:
    from pathlib import Path
    model = load_model()  # This function is defined below
    embeddings = model.predict(your_dataframe)
"""

import pickle
import numpy as np
from pathlib import Path
from typing import Any, Dict

# Import Row2Vec components (assumes row2vec is installed)
try:
    from row2vec.serialization import Row2VecModelMetadata
    from row2vec.model import Row2VecModel
except ImportError:
    raise ImportError(
        "row2vec package not found. Please install it first: pip install row2vec"
    )


# Model metadata (inspectable dictionary)
METADATA = {metadata_repr}


def get_metadata() -> Dict[str, Any]:
    """Get the model metadata as an inspectable dictionary."""
    return METADATA.copy()


def load_model() -> Row2VecModel:
    """
    Load the complete Row2Vec model with preprocessor and metadata.

    Returns:
        Row2VecModel: Loaded model ready for inference

    Raises:
        FileNotFoundError: If binary file not found
        Exception: If loading fails
    """
    # Get the path to the binary file using the script directory
    binary_filename = "{binary_filename}"

    # Try to use the script directory if available (passed from load_model function)
    if "__script_dir__" in globals():
        script_dir = Path(globals()["__script_dir__"])
        binary_path = script_dir / binary_filename
    else:
        # Fallback to current working directory
        binary_path = Path(binary_filename)

    if not binary_path.exists():
        raise FileNotFoundError(
            f"Binary model file not found: {{binary_path}}\\n"
            f"Expected filename: {{binary_filename}}\\n"
            f"Searched in: {{binary_path.parent if binary_path.parent != binary_path else Path.cwd()}}"
        )

    try:
        # Load binary components
        with open(binary_path, "rb") as f:
            binary_data = pickle.load(f)

        # Rebuild the fitted model and reattach its metadata
        model = Row2VecModel.from_state(binary_data["state"])
        model.metadata = Row2VecModelMetadata.from_dict(METADATA)

        return model

    except Exception as e:
        raise Exception(f"Failed to load model: {{str(e)}}")


if __name__ == "__main__":
    # Demo usage
    print("Row2Vec Model Information:")
    print("=" * 50)

    metadata = get_metadata()
    print(f"Mode: {{metadata['mode']}}")
    print(f"Embedding Dimensions: {{metadata['embedding_dim']}}")
    print(f"Created: {{metadata['created_at']}}")
    print(f"Training Time: {{metadata.get('training_time', 'N/A')}}s")
    print(f"Final Loss: {{metadata.get('final_loss', 'N/A')}}")
    print(f"Epochs Trained: {{metadata.get('epochs_trained', 'N/A')}}")

    print("\\nOriginal Columns:")
    for col in metadata.get('original_columns', []):
        print(f"  - {{col}}")

    print("\\nTo use this model:")
    print("  model = load_model()")
    print("  embeddings = model.predict(your_dataframe)")
'''


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
) -> tuple[pd.DataFrame, str, str]:
    """Train a Row2Vec model and save it using the two-file approach.

    This is a convenience function that combines training and saving.

    Args:
        df (pd.DataFrame): The input DataFrame containing numeric and categorical features.
        base_path (str | Path): Base path for the saved model, without a suffix.
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
        Tuple of (embeddings, script_path, binary_path)
    """
    # Import here to avoid circular imports

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
    script_path, binary_path = save_model(model, base_path, overwrite=overwrite)

    return embeddings, script_path, binary_path
