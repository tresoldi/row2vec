"""Tests for the Row2Vec command-line parser.

These assert against the parser's observable behaviour (what it accepts and what
it puts on the resulting namespace) rather than argparse internals, so they stay
valid if the parser is reorganised.
"""

import argparse
import inspect

import pytest

from row2vec.cli import cmd_train, create_parser

EXPECTED_MODES = ["unsupervised", "target", "pca", "tsne", "umap", "contrastive"]
EXPECTED_SUBCOMMANDS = ["train", "predict", "annotate"]


def test_create_parser_returns_argument_parser() -> None:
    assert isinstance(create_parser(), argparse.ArgumentParser)


@pytest.mark.parametrize("subcommand", EXPECTED_SUBCOMMANDS)
def test_subcommand_is_available(subcommand: str) -> None:
    """Each documented subcommand is registered and dispatches to a handler."""
    args = create_parser().parse_args(
        [subcommand, "--input", "in.csv", "--output", "out.csv"]
        + (["--model", "model.py"] if subcommand == "predict" else [])
    )
    assert args.command == subcommand
    assert callable(args.func)


@pytest.mark.parametrize("mode", EXPECTED_MODES)
def test_train_accepts_mode(mode: str) -> None:
    """Every embedding mode is a valid `--mode` choice for `train`."""
    args = create_parser().parse_args(
        ["train", "--input", "in.csv", "--output", "out.csv", "--mode", mode]
    )
    assert args.mode == mode


def test_train_rejects_unknown_mode() -> None:
    with pytest.raises(SystemExit):
        create_parser().parse_args(
            ["train", "--input", "in.csv", "--output", "out.csv", "--mode", "nonsense"]
        )


@pytest.mark.parametrize(
    "dest",
    [
        "similar_pairs_file",
        "dissimilar_pairs_file",
        "auto_pairs",
        "negative_samples",
        "contrastive_loss",
        "margin",
    ],
)
def test_contrastive_arguments_reach_the_namespace(dest: str) -> None:
    """The contrastive argument group populates every documented destination."""
    args = create_parser().parse_args(
        ["train", "--input", "in.csv", "--output", "out.csv", "--mode", "contrastive"]
    )
    assert hasattr(args, dest)


def test_cmd_train_takes_a_namespace() -> None:
    """Subcommand handlers share one signature: they consume the parsed namespace."""
    params = list(inspect.signature(cmd_train).parameters)
    assert params == ["args"]
