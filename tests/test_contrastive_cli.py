"""Tests for the contrastive-learning arguments of the CLI parser."""

from row2vec.cli import create_parser


def test_contrastive_mode_arguments_parse() -> None:
    """The contrastive mode and its tuning parameters are accepted and stored."""
    args = create_parser().parse_args(
        [
            "train",
            "--input",
            "dummy.csv",
            "--output",
            "dummy_output.csv",
            "--mode",
            "contrastive",
            "--contrastive-loss",
            "triplet",
            "--auto-pairs",
            "cluster",
            "--negative-samples",
            "3",
            "--margin",
            "0.5",
        ]
    )

    assert args.mode == "contrastive"
    assert args.contrastive_loss == "triplet"
    assert args.auto_pairs == "cluster"
    assert args.negative_samples == 3
    assert args.margin == 0.5


def test_contrastive_pair_file_arguments_parse() -> None:
    """Pairs can also be supplied as files rather than derived automatically."""
    args = create_parser().parse_args(
        [
            "train",
            "--input",
            "dummy.csv",
            "--output",
            "dummy_output.csv",
            "--mode",
            "contrastive",
            "--similar-pairs-file",
            "similar.csv",
            "--dissimilar-pairs-file",
            "dissimilar.csv",
        ]
    )

    assert args.similar_pairs_file == "similar.csv"
    assert args.dissimilar_pairs_file == "dissimilar.csv"
