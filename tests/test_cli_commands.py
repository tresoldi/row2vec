"""End-to-end tests for the CLI subcommands.

These drive `main()` the way a user would — real files in, real files out — so
they cover argument handling, I/O, and the error paths that a parser-only test
cannot reach.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import pandas as pd
import pytest

from row2vec import generate_synthetic_data
from row2vec.cli import main

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """A temporary directory holding an input CSV."""
    generate_synthetic_data(120, seed=1305).to_csv(tmp_path / "data.csv", index=False)
    return tmp_path


def run_cli(*argv: str) -> int:
    """Invoke `main()` with the given argv, returning its exit code."""
    original = sys.argv
    sys.argv = ["row2vec", *argv]
    try:
        return main()
    finally:
        sys.argv = original


class TestAnnotate:
    def test_writes_embeddings_csv(self, workspace: Path) -> None:
        out = workspace / "embeddings.csv"

        code = run_cli(
            "annotate",
            "--input",
            str(workspace / "data.csv"),
            "--output",
            str(out),
            "--mode",
            "pca",
            "--dim",
            "3",
            "--quiet",
        )

        assert code == 0
        assert out.exists()
        # Since 0.4.0 the input frame's index is written alongside the
        # embedding columns, so the output can be joined back to the input
        # rather than matched positionally. --no-index opts out.
        result = pd.read_csv(out, index_col=0)
        assert len(result) == 120
        assert result.shape[1] == 3
        assert list(result.columns) == ["embedding_0", "embedding_1", "embedding_2"]

    def test_tsne_mode(self, workspace: Path) -> None:
        out = workspace / "tsne.csv"

        code = run_cli(
            "annotate",
            "--input",
            str(workspace / "data.csv"),
            "--output",
            str(out),
            "--mode",
            "tsne",
            "--dim",
            "2",
            "--perplexity",
            "10",
            "--quiet",
        )

        assert code == 0
        assert pd.read_csv(out, index_col=0).shape == (120, 2)

    def test_output_index_matches_the_input_rows(self, workspace: Path) -> None:
        """The written index is what lets the output be joined back on."""
        source = pd.read_csv(workspace / "data.csv")
        out = workspace / "indexed.csv"

        code = run_cli(
            "annotate",
            "--input",
            str(workspace / "data.csv"),
            "--output",
            str(out),
            "--mode",
            "pca",
            "--dim",
            "2",
            "--quiet",
        )

        assert code == 0
        result = pd.read_csv(out, index_col=0)
        assert list(result.index) == list(source.index)
        # The whole point: a positional join is no longer required.
        joined = source.join(result)
        assert len(joined) == len(source)
        assert joined.isnull().sum().sum() == 0

    def test_no_index_omits_the_index_column(self, workspace: Path) -> None:
        """--no-index restores the pre-0.4.0 output shape."""
        out = workspace / "no_index.csv"

        code = run_cli(
            "annotate",
            "--input",
            str(workspace / "data.csv"),
            "--output",
            str(out),
            "--mode",
            "pca",
            "--dim",
            "2",
            "--no-index",
            "--quiet",
        )

        assert code == 0
        result = pd.read_csv(out)
        assert result.shape == (120, 2)
        assert list(result.columns) == ["embedding_0", "embedding_1"]

    def test_target_mode_without_target_column_fails(self, workspace: Path) -> None:
        code = run_cli(
            "annotate",
            "--input",
            str(workspace / "data.csv"),
            "--output",
            str(workspace / "out.csv"),
            "--mode",
            "target",
            "--quiet",
        )

        assert code == 1

    def test_missing_input_file_fails(self, workspace: Path) -> None:
        code = run_cli(
            "annotate",
            "--input",
            str(workspace / "nonexistent.csv"),
            "--output",
            str(workspace / "out.csv"),
            "--mode",
            "pca",
            "--quiet",
        )

        assert code == 1

    def test_validate_only_does_not_write_output(self, workspace: Path) -> None:
        out = workspace / "should-not-exist.csv"

        code = run_cli(
            "annotate",
            "--input",
            str(workspace / "data.csv"),
            "--output",
            str(out),
            "--validate-only",
            "--quiet",
        )

        assert code == 0
        assert not out.exists()


class TestTrainAndPredict:
    def test_round_trip(self, workspace: Path) -> None:
        """Train a model, then use it to embed a second file."""
        model = workspace / "model.py"

        train_code = run_cli(
            "train",
            "--input",
            str(workspace / "data.csv"),
            "--output",
            str(model),
            "--mode",
            "pca",
            "--dim",
            "2",
            "--quiet",
        )

        assert train_code == 0
        assert model.exists()

        # A second file with the same structure.
        new_data = workspace / "new.csv"
        generate_synthetic_data(40, seed=99).to_csv(new_data, index=False)
        predictions = workspace / "predictions.csv"

        predict_code = run_cli(
            "predict",
            "--input",
            str(new_data),
            "--model",
            str(model),
            "--output",
            str(predictions),
            "--quiet",
        )

        assert predict_code == 0
        assert len(pd.read_csv(predictions)) == 40

    def test_predict_with_missing_model_fails(self, workspace: Path) -> None:
        code = run_cli(
            "predict",
            "--input",
            str(workspace / "data.csv"),
            "--model",
            str(workspace / "nonexistent.py"),
            "--output",
            str(workspace / "out.csv"),
            "--quiet",
        )

        assert code == 1


class TestTopLevel:
    def test_no_arguments_prints_help(self) -> None:
        assert run_cli() == 1

    def test_version_exits_cleanly(self) -> None:
        with pytest.raises(SystemExit) as excinfo:
            run_cli("--version")
        assert excinfo.value.code == 0

    def test_quiet_and_verbose_together_is_rejected(self, workspace: Path) -> None:
        code = run_cli(
            "annotate",
            "--input",
            str(workspace / "data.csv"),
            "--output",
            str(workspace / "out.csv"),
            "--mode",
            "pca",
            "--quiet",
            "--verbose",
        )

        assert code == 1


class TestSearchArchitecture:
    @pytest.mark.slow
    def test_writes_json_results(self, workspace: Path) -> None:
        """The search command runs and serialises its results."""
        pytest.importorskip("tensorflow", reason="architecture search needs the [neural] extra")
        out = workspace / "search.json"

        code = run_cli(
            "search-architecture",
            "--input",
            str(workspace / "data.csv"),
            "--output",
            str(out),
            "--search-method",
            "random",
            "--max-trials",
            "2",
            "--max-time",
            "120",
            "--dim",
            "2",
            "--quiet",
        )

        assert code == 0
        assert out.exists()

        import json

        results = json.loads(out.read_text())
        assert "best_architecture" in results
        assert "search_summary" in results

    def test_missing_input_file_fails(self, workspace: Path) -> None:
        code = run_cli(
            "search-architecture",
            "--input",
            str(workspace / "nonexistent.csv"),
            "--max-trials",
            "1",
            "--quiet",
        )

        assert code == 1


class TestFileFormats:
    @pytest.mark.parametrize("suffix", [".csv", ".tsv"])
    def test_round_trips_delimited_formats(self, workspace: Path, suffix: str) -> None:
        """Input and output format are detected from the file extension."""
        source = pd.read_csv(workspace / "data.csv")
        input_file = workspace / f"data{suffix}"
        source.to_csv(input_file, sep="," if suffix == ".csv" else "\t", index=False)
        out = workspace / f"embeddings{suffix}"

        code = run_cli(
            "annotate",
            "--input",
            str(input_file),
            "--output",
            str(out),
            "--mode",
            "pca",
            "--dim",
            "2",
            "--quiet",
        )

        assert code == 0
        result = pd.read_csv(out, sep="," if suffix == ".csv" else "\t", index_col=0)
        assert result.shape == (120, 2)

    def test_unsupported_extension_fails(self, workspace: Path) -> None:
        unsupported = workspace / "data.xyz"
        unsupported.write_text("not really data")

        code = run_cli(
            "annotate",
            "--input",
            str(unsupported),
            "--output",
            str(workspace / "out.csv"),
            "--mode",
            "pca",
            "--quiet",
        )

        assert code == 1
