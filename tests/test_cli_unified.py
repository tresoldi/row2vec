#!/usr/bin/env python3
"""Unified CLI tests for Row2Vec."""

import inspect
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from row2vec.cli import cmd_train, create_parser


def test_cli_parser():
    """Test that the CLI parser is created successfully."""
    try:
        create_parser()
        print("✓ CLI parser created successfully")
        return True
    except Exception as e:
        print(f"✗ Failed to create CLI parser: {e}")
        return False


def test_cli_modes():
    """Test that all expected modes are available."""
    parser = create_parser()

    # Find the train subcommand
    for action in parser._subparsers._actions:
        if hasattr(action, "choices") and action.choices:
            if "train" in action.choices:
                train_parser = action.choices["train"]

                # Check available modes
                for action in train_parser._actions:
                    if hasattr(action, "dest") and action.dest == "mode":
                        expected_modes = [
                            "unsupervised",
                            "target",
                            "pca",
                            "tsne",
                            "umap",
                            "contrastive",
                        ]
                        actual_modes = action.choices

                        print(f"Expected modes: {expected_modes}")
                        print(f"Actual modes: {actual_modes}")

                        for mode in expected_modes:
                            if mode in actual_modes:
                                print(f"  ✓ {mode} mode found")
                            else:
                                print(f"  ✗ {mode} mode missing")
                                return False
                        return True

    print("✗ Could not find mode choices in train command")
    return False


def test_cli_contrastive_args():
    """Test that contrastive learning arguments are available."""
    parser = create_parser()

    # Find the train subcommand
    for action in parser._subparsers._actions:
        if hasattr(action, "choices") and action.choices:
            if "train" in action.choices:
                train_parser = action.choices["train"]

                # Look for contrastive argument group
                found_group = False
                for group in train_parser._action_groups:
                    if "contrastive" in group.title.lower():
                        found_group = True
                        print(f"✓ Found contrastive argument group: {group.title}")

                        # List available arguments
                        contrastive_args = []
                        for action in group._group_actions:
                            if hasattr(action, "dest"):
                                contrastive_args.append(action.dest)

                        expected_args = [
                            "similar_pairs_file",
                            "dissimilar_pairs_file",
                            "auto_pairs",
                            "negative_samples",
                            "contrastive_loss",
                            "margin",
                        ]

                        print(f"  Available contrastive arguments: {contrastive_args}")

                        for arg in expected_args:
                            if arg in contrastive_args:
                                print(f"    ✓ {arg} found")
                            else:
                                print(f"    ✗ {arg} missing")

                        return found_group

                if not found_group:
                    print("✗ Contrastive argument group not found")
                    return False

    return False


def test_cmd_train_signature():
    """Test that cmd_train function has expected parameters."""
    sig = inspect.signature(cmd_train)

    # Check for key parameters
    expected_params = [
        "input",
        "output",
        "mode",
        "embedding_dim",
        "max_epochs",
        "batch_size",
    ]

    print("Checking cmd_train function signature:")
    for param in expected_params:
        if param in sig.parameters:
            print(f"  ✓ {param} parameter found")
        else:
            print(f"  ✗ {param} parameter missing")
            return False

    return True


def main():
    """Run all CLI tests."""
    print("=" * 60)
    print("Row2Vec Unified CLI Tests")
    print("=" * 60)

    all_passed = True

    tests = [
        ("CLI Parser Creation", test_cli_parser),
        ("CLI Modes", test_cli_modes),
        ("Contrastive Arguments", test_cli_contrastive_args),
        ("cmd_train Signature", test_cmd_train_signature),
    ]

    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * 40)
        if not test_func():
            all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("✅ All CLI tests passed!")
        return 0
    print("❌ Some CLI tests failed")
    return 1


if __name__ == "__main__":
    sys.exit(main())
