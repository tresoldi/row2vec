"""
Row2Vec Command Line Interface

A comprehensive CLI for training embeddings, making predictions, and annotating datasets
with Row2Vec embeddings using various methods (neural, PCA, t-SNE, UMAP).
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

from . import __version__
from .core import learn_embedding
from .serialization import load_model, train_and_save_model
from .utils import create_dataframe_schema, validate_dataframe_schema


def _detect_input_format(file_path: Path) -> str:
    """Detect input file format based on extension."""
    suffix = file_path.suffix.lower()
    if suffix in [".csv"]:
        return "csv"
    if suffix in [".tsv"]:
        return "tsv"
    if suffix in [".parquet", ".pq"]:
        return "parquet"
    raise ValueError(
        f"Unsupported file format: {suffix}. Supported formats: .csv, .tsv, .parquet",
    )


def _load_dataframe(file_path: Path, validate_only: bool = False) -> pd.DataFrame:
    """Load DataFrame from file with auto-format detection."""
    if not file_path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")

    format_type = _detect_input_format(file_path)

    try:
        if format_type == "csv":
            df = pd.read_csv(file_path, encoding="utf-8")
        elif format_type == "tsv":
            df = pd.read_csv(file_path, sep="\t", encoding="utf-8")
        elif format_type == "parquet":
            df = pd.read_parquet(file_path)
        else:
            raise ValueError(f"Unsupported format: {format_type}")

        if validate_only:
            pass

        return df

    except Exception as e:
        raise ValueError(f"Failed to load {file_path}: {e!s}")


def _save_dataframe(df: pd.DataFrame, file_path: Path) -> None:
    """Save DataFrame to file with format detection."""
    file_path.parent.mkdir(parents=True, exist_ok=True)

    format_type = _detect_input_format(file_path)

    try:
        if format_type in ["csv", "tsv"]:
            separator = "," if format_type == "csv" else "\t"
            df.to_csv(file_path, sep=separator, index=False, encoding="utf-8")
        elif format_type == "parquet":
            df.to_parquet(file_path, index=False)

    except Exception as e:
        raise ValueError(f"Failed to save to {file_path}: {e!s}")


def _validate_schema_friendly(
    df: pd.DataFrame, reference_column: str | None = None
) -> bool:
    """Validate DataFrame schema with user-friendly error messages."""
    try:
        # Create and validate schema
        schema = create_dataframe_schema(df)
        validate_dataframe_schema(df, schema)

        # Additional validations
        if df.empty:
            raise ValueError("Dataset is empty (0 rows)")

        if len(df.columns) == 0:
            raise ValueError("Dataset has no columns")

        if len(df) < 10:
            pass

        # Mode-specific validation
        if reference_column:
            if reference_column not in df.columns:
                raise ValueError(
                    f"Target column '{reference_column}' not found in dataset. Available columns: {list(df.columns)}"
                )

            unique_values = df[reference_column].nunique()
            if unique_values < 2:
                raise ValueError(
                    f"Target column '{reference_column}' must have at least 2 unique values, found {unique_values}"
                )
            if unique_values > 1000:
                pass

        # Check for missing data
        missing_cols = df.columns[df.isnull().any()].tolist()
        if missing_cols:
            {col: df[col].isnull().sum() for col in missing_cols}

        # Check data types
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        categorical_cols = df.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()

        if reference_column and reference_column in categorical_cols:
            categorical_cols.remove(reference_column)

        if len(numeric_cols) == 0 and len(categorical_cols) == 0:
            raise ValueError(
                "Dataset must contain at least one numeric or categorical column for embedding"
            )

        return True

    except Exception:
        return False


def _generate_model_name(mode: str, timestamp: str | None = None) -> str:
    """Generate automatic model filename."""
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"row2vec_model_{mode}_{timestamp}"


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add common arguments to subcommand parsers."""
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output and training progress",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress all non-error output",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate input data schema without processing",
    )


def _add_embedding_args(parser: argparse.ArgumentParser) -> None:
    """Add embedding-specific arguments."""
    # Core parameters
    parser.add_argument(
        "--mode",
        "-m",
        choices=["unsupervised", "target", "pca", "tsne", "umap", "contrastive"],
        default="unsupervised",
        help="Embedding method (default: unsupervised)",
    )
    parser.add_argument(
        "--dim",
        "-d",
        type=int,
        default=10,
        help="Embedding dimensions (default: 10)",
    )
    parser.add_argument(
        "--target-col",
        type=str,
        help="Target column name for target mode (required for target mode)",
    )

    # Neural network parameters
    neural_group = parser.add_argument_group("Neural Network Parameters")
    neural_group.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Maximum training epochs for neural methods (default: 50)",
    )
    neural_group.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for neural methods (default: 64)",
    )
    neural_group.add_argument(
        "--dropout",
        type=float,
        default=0.2,
        help="Dropout rate for neural methods (default: 0.2)",
    )
    neural_group.add_argument(
        "--hidden-units",
        type=int,
        default=128,
        help="Hidden layer units for neural methods (default: 128)",
    )
    neural_group.add_argument(
        "--no-early-stopping",
        action="store_true",
        help="Disable early stopping for neural methods",
    )

    # Classical ML parameters
    classical_group = parser.add_argument_group("Classical ML Parameters")
    classical_group.add_argument(
        "--n-neighbors",
        type=int,
        default=15,
        help="Number of neighbors for UMAP (default: 15)",
    )
    classical_group.add_argument(
        "--perplexity",
        type=float,
        default=30.0,
        help="Perplexity parameter for t-SNE (default: 30.0)",
    )
    classical_group.add_argument(
        "--min-dist",
        type=float,
        default=0.1,
        help="Minimum distance for UMAP (default: 0.1)",
    )
    classical_group.add_argument(
        "--n-iter",
        type=int,
        default=1000,
        help="Number of iterations for t-SNE (default: 1000)",
    )

    # Contrastive learning parameters
    contrastive_group = parser.add_argument_group("Contrastive Learning Parameters")
    contrastive_group.add_argument(
        "--contrastive-loss",
        choices=["triplet", "contrastive"],
        default="triplet",
        help="Loss function for contrastive learning (default: triplet)",
    )
    contrastive_group.add_argument(
        "--auto-pairs",
        choices=["cluster", "neighbors", "categorical", "random"],
        help="Automatic pair generation strategy for contrastive learning",
    )
    contrastive_group.add_argument(
        "--similar-pairs-file",
        type=str,
        help="CSV file with similar pairs (format: idx1,idx2 per line)",
    )
    contrastive_group.add_argument(
        "--dissimilar-pairs-file",
        type=str,
        help="CSV file with dissimilar pairs (format: idx1,idx2 per line)",
    )
    contrastive_group.add_argument(
        "--negative-samples",
        type=int,
        default=5,
        help="Number of negative samples per positive sample (default: 5)",
    )
    contrastive_group.add_argument(
        "--margin",
        type=float,
        default=1.0,
        help="Margin parameter for loss functions (default: 1.0)",
    )

    # Scaling and reproducibility
    scaling_group = parser.add_argument_group("Scaling and Reproducibility")
    scaling_group.add_argument(
        "--scale-method",
        choices=["none", "minmax", "standard", "l2", "tanh"],
        help="Scaling method for embeddings (default: none)",
    )
    scaling_group.add_argument(
        "--scale-range",
        nargs=2,
        type=float,
        metavar=("MIN", "MAX"),
        help="Range for minmax scaling (e.g., --scale-range 0 1)",
    )
    scaling_group.add_argument(
        "--seed",
        type=int,
        default=1305,
        help="Random seed for reproducibility (default: 1305)",
    )

    # Logging
    logging_group = parser.add_argument_group("Logging Parameters")
    logging_group.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)",
    )
    logging_group.add_argument(
        "--log-file",
        type=str,
        help="Log file path (default: console only)",
    )
    logging_group.add_argument(
        "--no-logging",
        action="store_true",
        help="Disable structured logging",
    )


def cmd_train(args: argparse.Namespace) -> int:
    """Train and save a Row2Vec model."""
    try:
        if not args.quiet:
            pass

        # Load and validate data
        df = _load_dataframe(Path(args.input), args.validate_only)

        if args.validate_only:
            success = _validate_schema_friendly(df, args.target_col)
            return 0 if success else 1

        if not _validate_schema_friendly(df, args.target_col):
            return 1

        # Check mode-specific requirements
        if args.mode == "target" and not args.target_col:
            return 1

        # Load contrastive learning pairs if specified
        similar_pairs = None
        dissimilar_pairs = None

        if args.mode == "contrastive":
            if args.similar_pairs_file:
                try:
                    pairs_df = pd.read_csv(args.similar_pairs_file, header=None)
                    if pairs_df.shape[1] != 2:
                        return 1
                    similar_pairs = [
                        (int(row[0]), int(row[1])) for _, row in pairs_df.iterrows()
                    ]
                    if not args.quiet:
                        pass
                except Exception:
                    return 1

            if args.dissimilar_pairs_file:
                try:
                    pairs_df = pd.read_csv(args.dissimilar_pairs_file, header=None)
                    if pairs_df.shape[1] != 2:
                        return 1
                    dissimilar_pairs = [
                        (int(row[0]), int(row[1])) for _, row in pairs_df.iterrows()
                    ]
                    if not args.quiet:
                        pass
                except Exception:
                    return 1

            # Validate contrastive learning requirements
            if not similar_pairs and not dissimilar_pairs and not args.auto_pairs:
                return 1

            if args.auto_pairs == "categorical" and not args.target_col:
                return 1

        # Convert scale range to tuple if provided
        scale_range = tuple(args.scale_range) if args.scale_range else None

        # Generate model path if not provided
        if not args.output:
            model_name = _generate_model_name(args.mode)
            model_path = Path.cwd() / f"{model_name}.py"
        else:
            model_path = Path(args.output)
            if model_path.suffix != ".py":
                model_path = model_path.with_suffix(".py")

        if not args.quiet:
            pass

        # Train and save model
        start_time = time.time()
        embeddings, script_path, binary_path = train_and_save_model(
            df=df,
            base_path=str(model_path.with_suffix("")),
            embedding_dim=args.dim,
            mode=args.mode,
            reference_column=args.target_col,
            max_epochs=args.epochs,
            batch_size=args.batch_size,
            dropout_rate=args.dropout,
            hidden_units=args.hidden_units,
            early_stopping=not args.no_early_stopping,
            seed=args.seed,
            verbose=args.verbose and not args.quiet,
            scale_method=args.scale_method,
            scale_range=scale_range,
            log_level=args.log_level,
            log_file=args.log_file,
            enable_logging=not args.no_logging,
            n_neighbors=args.n_neighbors,
            perplexity=args.perplexity,
            min_dist=args.min_dist,
            n_iter=args.n_iter,
            # Contrastive learning parameters
            similar_pairs=similar_pairs,
            dissimilar_pairs=dissimilar_pairs,
            auto_pairs=args.auto_pairs,
            negative_samples=args.negative_samples,
            contrastive_loss=args.contrastive_loss,
            margin=args.margin,
            overwrite=True,  # CLI should overwrite existing files
        )

        time.time() - start_time

        if not args.quiet:
            pass

        return 0

    except Exception:
        return 1


def cmd_predict(args: argparse.Namespace) -> int:
    """Make predictions using a saved Row2Vec model."""
    try:
        if not args.quiet:
            pass

        # Load and validate data
        df = _load_dataframe(Path(args.input), args.validate_only)

        if args.validate_only:
            success = _validate_schema_friendly(df)
            return 0 if success else 1

        if not _validate_schema_friendly(df):
            return 1

        # Load model
        model_path = Path(args.model)
        if not model_path.exists():
            return 1

        if not args.quiet:
            pass

        model = load_model(str(model_path))

        if not args.quiet:
            model.metadata.to_dict()

        # Make predictions
        if not args.quiet:
            pass

        start_time = time.time()
        embeddings = model.predict(df)
        time.time() - start_time

        # Save results
        output_path = Path(args.output)
        _save_dataframe(embeddings, output_path)

        if not args.quiet:
            pass

        return 0

    except Exception:
        return 1


def cmd_annotate(args: argparse.Namespace) -> int:
    """Generate embeddings directly without saving a model."""
    try:
        if not args.quiet:
            pass

        # Load and validate data
        df = _load_dataframe(Path(args.input), args.validate_only)

        if args.validate_only:
            success = _validate_schema_friendly(df, args.target_col)
            return 0 if success else 1

        if not _validate_schema_friendly(df, args.target_col):
            return 1

        # Check mode-specific requirements
        if args.mode == "target" and not args.target_col:
            return 1

        # Load contrastive learning pairs if specified
        similar_pairs = None
        dissimilar_pairs = None

        if args.mode == "contrastive":
            if args.similar_pairs_file:
                try:
                    pairs_df = pd.read_csv(args.similar_pairs_file, header=None)
                    if pairs_df.shape[1] != 2:
                        return 1
                    similar_pairs = [
                        (int(row[0]), int(row[1])) for _, row in pairs_df.iterrows()
                    ]
                    if not args.quiet:
                        pass
                except Exception:
                    return 1

            if args.dissimilar_pairs_file:
                try:
                    pairs_df = pd.read_csv(args.dissimilar_pairs_file, header=None)
                    if pairs_df.shape[1] != 2:
                        return 1
                    dissimilar_pairs = [
                        (int(row[0]), int(row[1])) for _, row in pairs_df.iterrows()
                    ]
                    if not args.quiet:
                        pass
                except Exception:
                    return 1

            # Validate contrastive learning requirements
            if not similar_pairs and not dissimilar_pairs and not args.auto_pairs:
                return 1

            if args.auto_pairs == "categorical" and not args.target_col:
                return 1

        # Convert scale range to tuple if provided
        scale_range = tuple(args.scale_range) if args.scale_range else None

        if not args.quiet:
            pass

        # Generate embeddings
        start_time = time.time()
        embeddings = learn_embedding(
            df=df,
            embedding_dim=args.dim,
            mode=args.mode,
            reference_column=args.target_col,
            max_epochs=args.epochs,
            batch_size=args.batch_size,
            dropout_rate=args.dropout,
            hidden_units=args.hidden_units,
            early_stopping=not args.no_early_stopping,
            seed=args.seed,
            verbose=args.verbose and not args.quiet,
            scale_method=args.scale_method,
            scale_range=scale_range,
            log_level=args.log_level,
            log_file=args.log_file,
            enable_logging=not args.no_logging,
            n_neighbors=args.n_neighbors,
            perplexity=args.perplexity,
            min_dist=args.min_dist,
            n_iter=args.n_iter,
            # Contrastive learning parameters
            similar_pairs=similar_pairs,
            dissimilar_pairs=dissimilar_pairs,
            auto_pairs=args.auto_pairs,
            negative_samples=args.negative_samples,
            contrastive_loss=args.contrastive_loss,
            margin=args.margin,
        )

        time.time() - start_time

        # Save results
        output_path = Path(args.output)
        _save_dataframe(embeddings, output_path)

        if not args.quiet:
            pass

        return 0

    except Exception:
        return 1


def cmd_search_architecture(args: argparse.Namespace) -> int:
    """Architecture search command: Find optimal neural architecture."""
    if not args.quiet:
        pass

    # Load and validate input data
    input_path = Path(args.input)
    try:
        df = _load_dataframe(input_path)
        if not args.quiet:
            pass

    except Exception:
        return 1

    # Import architecture search components
    try:
        from .architecture_search import ArchitectureSearchConfig, search_architecture
        from .config import EmbeddingConfig
    except ImportError:
        return 1

    # Create base embedding config
    try:
        base_config = EmbeddingConfig(
            mode=args.mode or "neural",
            embedding_dim=args.dim or 10,  # Default dimension
        )

        # Create architecture search config
        search_config = ArchitectureSearchConfig(
            method=args.search_method,
            max_trials=args.max_trials,
            max_time=args.max_time,
            patience=args.patience,
            verbose=not args.quiet,
        )

        if not args.quiet:
            pass

    except Exception:
        return 1

    # Perform architecture search
    try:
        time.time()
        best_architecture, search_result = search_architecture(
            df=df,
            base_config=base_config,
            search_config=search_config,
            target_column=getattr(args, "target_col", None),
        )
        time.time()

        # Display results
        summary = search_result.summary()

        if not args.quiet:
            # Handle both single layer (int) and multi-layer (list) formats
            hidden_units = best_architecture["hidden_units"]
            if isinstance(hidden_units, list):
                pass
            else:
                pass

            if summary["improvement_over_baseline"] > 0:
                pass

    except Exception:
        return 1

    # Save results if output specified
    if hasattr(args, "output") and args.output:
        try:
            output_path = Path(args.output)

            # Save detailed results
            results_dict = {
                "best_architecture": best_architecture,
                "search_summary": summary,
                "search_config": {
                    "method": search_config.method,
                    "max_trials": search_config.max_trials,
                    "max_time": search_config.max_time,
                    "patience": search_config.patience,
                },
            }

            if output_path.suffix == ".json":
                import json

                with open(output_path, "w") as f:
                    json.dump(results_dict, f, indent=2)
            else:
                # Save as YAML
                import yaml

                with open(output_path, "w") as f:
                    yaml.dump(results_dict, f, default_flow_style=False)

            if not args.quiet:
                pass

        except Exception:
            return 1

    return 0


def create_parser() -> argparse.ArgumentParser:
    """Create the main argument parser."""
    parser = argparse.ArgumentParser(
        prog="row2vec",
        description="Row2Vec: Generate embeddings from tabular data using neural networks and classical ML methods",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train and save a model
  row2vec train --input data.csv --output model.py --mode unsupervised --dim 10

  # Make predictions with saved model
  row2vec predict --input new_data.csv --model model.py --output embeddings.csv

  # Generate embeddings directly (no model saving)
  row2vec annotate --input data.csv --output embeddings.csv --mode pca --dim 5

  # Target-based embeddings
  row2vec train --input data.csv --output model.py --mode target --target-col Country --dim 3

  # Validate data only
  row2vec train --input data.csv --validate-only

Supported formats: CSV, TSV, Parquet (auto-detected by file extension)
        """,
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"Row2Vec {__version__}",
    )

    # Subcommands
    subparsers = parser.add_subparsers(
        dest="command",
        help="Available commands",
        metavar="command",
    )

    # Train subcommand
    train_parser = subparsers.add_parser(
        "train",
        help="Train and save a Row2Vec model",
        description="Train a Row2Vec embedding model and save it for later use",
    )
    train_parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Input data file (CSV, TSV, or Parquet)",
    )
    train_parser.add_argument(
        "--output",
        "-o",
        help="Output model file path (default: auto-generated with timestamp)",
    )
    _add_common_args(train_parser)
    _add_embedding_args(train_parser)
    train_parser.set_defaults(func=cmd_train)

    # Predict subcommand
    predict_parser = subparsers.add_parser(
        "predict",
        help="Generate embeddings using a saved model",
        description="Use a previously trained Row2Vec model to generate embeddings for new data",
    )
    predict_parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Input data file (CSV, TSV, or Parquet)",
    )
    predict_parser.add_argument(
        "--model",
        "-m",
        required=True,
        help="Saved Row2Vec model file (.py)",
    )
    predict_parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="Output embeddings file (CSV, TSV, or Parquet)",
    )
    _add_common_args(predict_parser)
    predict_parser.set_defaults(func=cmd_predict)

    # Annotate subcommand
    annotate_parser = subparsers.add_parser(
        "annotate",
        help="Generate embeddings directly without saving model",
        description="Generate embeddings for data without saving a model (for one-time use)",
    )
    annotate_parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Input data file (CSV, TSV, or Parquet)",
    )
    annotate_parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="Output embeddings file (CSV, TSV, or Parquet)",
    )
    _add_common_args(annotate_parser)
    _add_embedding_args(annotate_parser)
    annotate_parser.set_defaults(func=cmd_annotate)

    # Architecture search subcommand
    arch_search_parser = subparsers.add_parser(
        "search-architecture",
        help="Find optimal neural network architecture",
        description="Automatically search for the best neural architecture for your dataset",
    )
    arch_search_parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Input data file (CSV, TSV, or Parquet)",
    )
    arch_search_parser.add_argument(
        "--output",
        "-o",
        help="Output results file (JSON or YAML)",
    )
    arch_search_parser.add_argument(
        "--search-method",
        choices=["random", "grid"],
        default="random",
        help="Architecture search method (default: random)",
    )
    arch_search_parser.add_argument(
        "--max-trials",
        type=int,
        default=30,
        help="Maximum number of architecture trials (default: 30)",
    )
    arch_search_parser.add_argument(
        "--max-time",
        type=int,
        default=1800,
        help="Maximum search time in seconds (default: 1800)",
    )
    arch_search_parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Stop after N trials without improvement (default: 10)",
    )
    arch_search_parser.add_argument(
        "--mode",
        choices=["unsupervised", "target", "contrastive"],
        default="unsupervised",
        help="Embedding mode for search (default: unsupervised)",
    )
    arch_search_parser.add_argument(
        "--dim",
        type=int,
        default=10,
        help="Embedding dimension (default: 10)",
    )
    arch_search_parser.add_argument(
        "--target-col",
        help="Target column for supervised evaluation",
    )
    _add_common_args(arch_search_parser)
    arch_search_parser.set_defaults(func=cmd_search_architecture)

    return parser


def main() -> int:
    """Main CLI entry point."""
    parser = create_parser()

    # Handle case where no arguments are provided
    if len(sys.argv) == 1:
        parser.print_help()
        return 1

    args = parser.parse_args()

    # Handle case where no subcommand is provided
    if not hasattr(args, "func"):
        parser.print_help()
        return 1

    # Set up quiet mode
    if args.quiet and args.verbose:
        return 1

    # Execute the subcommand
    try:
        return args.func(args)
    except KeyboardInterrupt:
        if not args.quiet:
            pass
        return 1
    except Exception:
        return 1


if __name__ == "__main__":
    sys.exit(main())
