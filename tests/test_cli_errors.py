"""CLI paths the happy-path tests do not reach: pair files, bad input, validation."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import pandas as pd
import pytest

from row2vec import generate_synthetic_data
from row2vec.cli import _detect_input_format, _validate_schema_friendly, main

if TYPE_CHECKING:
    from pathlib import Path


def run_cli(*argv: str) -> int:
    original = sys.argv
    sys.argv = ["row2vec", *argv]
    try:
        return main()
    finally:
        sys.argv = original


@pytest.fixture
def data(tmp_path: Path) -> Path:
    generate_synthetic_data(60, seed=1305).to_csv(tmp_path / "data.csv", index=False)
    return tmp_path / "data.csv"


@pytest.mark.parametrize("command", ["train", "annotate"])
class TestContrastivePairFiles:
    def _args(self, command: str, data: Path, out: Path, *extra: str) -> list[str]:
        flag = ["--output", str(out)]
        return [command, "--input", str(data), *flag, "--mode", "contrastive", "--quiet", *extra]

    def test_pair_file_with_wrong_width_fails(
        self, command: str, data: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        pairs = tmp_path / "pairs.csv"
        pairs.write_text("1,2,3\n4,5,6\n")
        out = tmp_path / ("m.r2v" if command == "train" else "e.csv")

        code = run_cli(*self._args(command, data, out, "--similar-pairs-file", str(pairs)))

        assert code == 1
        assert "exactly 2 columns" in capsys.readouterr().err

    def test_dissimilar_pair_file_with_wrong_width_fails(
        self, command: str, data: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        pairs = tmp_path / "pairs.csv"
        pairs.write_text("1,2,3\n")
        out = tmp_path / ("m.r2v" if command == "train" else "e.csv")

        code = run_cli(*self._args(command, data, out, "--dissimilar-pairs-file", str(pairs)))

        assert code == 1
        assert "exactly 2 columns" in capsys.readouterr().err

    def test_unreadable_pair_file_fails(
        self, command: str, data: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        out = tmp_path / ("m.r2v" if command == "train" else "e.csv")

        code = run_cli(
            *self._args(command, data, out, "--similar-pairs-file", str(tmp_path / "nope.csv"))
        )

        assert code == 1
        assert "could not read" in capsys.readouterr().err

    def test_contrastive_without_any_pairs_fails(
        self, command: str, data: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        out = tmp_path / ("m.r2v" if command == "train" else "e.csv")

        code = run_cli(*self._args(command, data, out))

        assert code == 1
        assert "needs --similar-pairs-file" in capsys.readouterr().err

    def test_categorical_pairs_need_a_target_column(
        self, command: str, data: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        out = tmp_path / ("m.r2v" if command == "train" else "e.csv")

        code = run_cli(*self._args(command, data, out, "--auto-pairs", "categorical"))

        assert code == 1
        assert "requires --target-col" in capsys.readouterr().err


class TestValidation:
    def test_unsupported_extension_is_named(self) -> None:
        from pathlib import Path

        with pytest.raises(ValueError, match="Unsupported file format"):
            _detect_input_format(Path("data.xlsx"))

    def test_empty_frame_is_unusable(self) -> None:
        ok, messages = _validate_schema_friendly(pd.DataFrame({"a": []}))

        assert not ok
        assert messages

    def test_missing_target_column_lists_the_available_ones(self) -> None:
        df = generate_synthetic_data(30, seed=1)

        ok, messages = _validate_schema_friendly(df, "nope")

        assert not ok
        assert "not found" in messages[0]

    def test_constant_target_is_rejected(self) -> None:
        df = generate_synthetic_data(30, seed=1).assign(flag="same")

        ok, messages = _validate_schema_friendly(df, "flag")

        assert not ok
        assert "at least 2 unique values" in messages[0]

    def test_small_and_incomplete_frames_warn_but_pass(self) -> None:
        df = generate_synthetic_data(8, seed=1)
        df.loc[0, "Sales"] = None

        ok, messages = _validate_schema_friendly(df)

        assert ok
        assert any("Only 8 rows" in m for m in messages)
        assert any("missing values" in m for m in messages)

    def test_validate_only_reports_without_training(
        self, data: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        out = tmp_path / "m.r2v"

        code = run_cli("train", "--input", str(data), "--output", str(out), "--validate-only")

        assert code == 0
        assert not out.exists()
        assert "valid" in capsys.readouterr().out.lower()


def test_parquet_round_trip(data: Path, tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    out = tmp_path / "emb.parquet"

    code = run_cli(
        "annotate", "--input", str(data), "--output", str(out), "--mode", "pca", "--dim", "2",
        "--quiet",
    )  # fmt: skip

    assert code == 0
    assert pd.read_parquet(out).shape[0] == 60
