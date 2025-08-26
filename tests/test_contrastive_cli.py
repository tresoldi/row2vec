#!/usr/bin/env python3
"""Test to verify contrastive learning CLI integration with actual argument parsing."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from row2vec.cli import create_parser


def test_contrastive_integration():
    """Test that contrastive mode and parameters work in CLI."""
    parser = create_parser()

    # Test 1: Verify that 'contrastive' is a valid mode choice
    test_args = [
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

    try:
        args = parser.parse_args(test_args)
        print("✓ Successfully parsed contrastive mode arguments")
        print(f"  Mode: {args.mode}")
        print(f"  Contrastive loss: {args.contrastive_loss}")
        print(f"  Auto pairs: {args.auto_pairs}")
        print(f"  Negative samples: {args.negative_samples}")
        print(f"  Margin: {args.margin}")

        # Check that contrastive mode validation would work
        if args.mode == "contrastive":
            print("✓ Contrastive mode properly recognized")
        else:
            print("✗ Contrastive mode not properly recognized")

    except SystemExit as e:
        print(f"✗ Failed to parse contrastive arguments: {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False

    # Test 2: Verify that similar/dissimilar pairs file arguments are available
    test_args_with_files = [
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

    try:
        args = parser.parse_args(test_args_with_files)
        print("✓ Successfully parsed contrastive file arguments")
        print(f"  Similar pairs file: {args.similar_pairs_file}")
        print(f"  Dissimilar pairs file: {args.dissimilar_pairs_file}")

    except SystemExit as e:
        print(f"✗ Failed to parse contrastive file arguments: {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error with file arguments: {e}")
        return False

    print("\n✓ All contrastive learning CLI tests passed!")
    return True


if __name__ == "__main__":
    success = test_contrastive_integration()
    sys.exit(0 if success else 1)
