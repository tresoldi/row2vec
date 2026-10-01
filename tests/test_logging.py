"""
Tests for the logging functionality in Row2Vec.
"""

import logging
import os
import tempfile
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from row2vec import Row2VecLogger, get_logger, learn_embedding
from row2vec.utils import generate_synthetic_data


def release_log_files(name: str = "row2vec") -> None:
    """Close every file handler under `name` so the file can be deleted.

    Windows refuses to unlink a file that is still open (WinError 32), so a
    test that points a logger at a temp file must detach the handler before
    cleaning up. `learn_embedding` creates its own logger internally, which the
    caller never sees, hence reaching for it by name here.
    """
    names = [n for n in logging.root.manager.loggerDict if n == name or n.startswith(f"{name}.")]
    for logger_name in [name, *names]:
        logger = logging.getLogger(logger_name)
        for handler in list(logger.handlers):
            if isinstance(handler, logging.FileHandler):
                handler.close()
                logger.removeHandler(handler)


class TestRow2VecLogger:
    """Test suite for Row2VecLogger class."""

    def test_logger_initialization(self) -> None:
        """Test basic logger initialization."""
        logger = Row2VecLogger()
        assert logger.logger.name == "row2vec"
        assert logger.include_performance is True
        assert logger.include_memory is True

    def test_logger_with_custom_settings(self) -> None:
        """Test logger with custom settings."""
        logger = Row2VecLogger(
            name="test_logger",
            level="DEBUG",
            include_performance=False,
            include_memory=False,
        )
        assert logger.logger.name == "test_logger"
        assert logger.include_performance is False
        assert logger.include_memory is False

    def test_logger_with_file_output(self) -> None:
        """Test logger with file output."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            log_file = f.name

        try:
            logger = Row2VecLogger(log_file=log_file)
            logger.logger.info("Test message")

            # Check that file was created and contains content
            assert os.path.exists(log_file)
            with open(log_file) as f:
                content = f.read()
                assert "Test message" in content
        finally:
            logger.close()
            if os.path.exists(log_file):
                os.unlink(log_file)

    def test_log_file_is_utf8_and_keeps_emoji_lines(self) -> None:
        """Emoji-bearing messages must survive into the log file.

        FileHandler defaults to the locale encoding, which is cp1252 on
        Windows. Most Row2Vec messages begin with an emoji, which cp1252
        cannot encode; logging swallows the UnicodeEncodeError and drops the
        record, so those lines vanished from the file with nothing failing.

        The encoding assertion is what makes this test meaningful off Windows:
        on a UTF-8 locale the round-trip would pass either way.
        """
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".log") as f:
            log_file = f.name

        logger = Row2VecLogger(name="row2vec.utf8_check", log_file=log_file)
        try:
            handlers = [
                h
                for h in logging.getLogger("row2vec.utf8_check").handlers
                if isinstance(h, logging.FileHandler)
            ]
            assert handlers, "expected a file handler"
            assert all((h.encoding or "").lower().replace("-", "") == "utf8" for h in handlers)

            logger.logger.info("\U0001f680 Starting training with configuration: mode=pca")

            with open(log_file, encoding="utf-8") as f:
                content = f.read()
            assert "Starting training" in content
            assert "\U0001f680" in content
        finally:
            logger.close()
            if os.path.exists(log_file):
                os.unlink(log_file)

    def test_training_lifecycle_logging(self) -> None:
        """Test the complete training lifecycle logging."""
        logger = Row2VecLogger()

        # Mock the memory tracking to avoid system dependencies in tests
        with patch.object(logger, "_get_memory_usage", return_value=100.0):
            # Start training
            logger.start_training(mode="unsupervised", embedding_dim=5)
            assert logger.training_start_time is not None

            # Epoch logging
            logger.start_epoch(0, 10)
            assert logger.epoch_start_time is not None

            logger.log_epoch_metrics(0, 0.5, 0.6, {"accuracy": 0.85})

            # End training
            logger.end_training(0.1, 10)

    def test_error_logging(self) -> None:
        """Test error logging functionality."""
        logger = Row2VecLogger()

        try:
            raise ValueError("Test error")
        except ValueError as e:
            logger.log_error(e, "Test context")
            # Should not raise any exceptions

    def test_get_logger_convenience_function(self) -> None:
        """Test the get_logger convenience function."""
        logger = get_logger("test")
        assert isinstance(logger, Row2VecLogger)
        assert logger.logger.name == "test"


class TestLoggingIntegration:
    """Test suite for logging integration with learn_embedding."""

    def test_learn_embedding_with_logging_enabled(self) -> None:
        """Test learn_embedding with logging enabled."""
        df = generate_synthetic_data(100)

        # Test with logging enabled (default)
        result = learn_embedding(
            df,
            embedding_dim=3,
            max_epochs=2,
            batch_size=50,
            enable_logging=True,
            log_level="INFO",
        )

        assert result.shape == (100, 3)

    def test_learn_embedding_with_logging_disabled(self) -> None:
        """Test learn_embedding with logging disabled."""
        df = generate_synthetic_data(50)

        result = learn_embedding(
            df,
            embedding_dim=2,
            max_epochs=2,
            batch_size=25,
            enable_logging=False,
        )

        assert result.shape == (50, 2)

    def test_learn_embedding_with_file_logging(self) -> None:
        """Test learn_embedding with file logging."""
        df = generate_synthetic_data(30)

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".log") as f:
            log_file = f.name

        try:
            result = learn_embedding(
                df,
                embedding_dim=2,
                max_epochs=2,
                batch_size=15,
                enable_logging=True,
                log_file=log_file,
                log_level="DEBUG",
            )

            assert result.shape == (30, 2)

            # Check that log file contains expected content
            with open(log_file) as f:
                content = f.read()
                assert "Starting training" in content
                assert "Preprocessing data" in content
                assert "Training completed" in content

        finally:
            release_log_files()
            if os.path.exists(log_file):
                os.unlink(log_file)

    def test_target_mode_logging(self) -> None:
        """Test logging in target mode."""
        df = pd.DataFrame(
            {
                "feature1": np.random.randn(50),
                "feature2": np.random.randn(50),
                "category": np.random.choice(["A", "B", "C"], 50),
            },
        )

        result = learn_embedding(
            df,
            mode="target",
            reference_column="category",
            embedding_dim=2,
            max_epochs=2,
            batch_size=25,
            enable_logging=True,
            log_level="DEBUG",
        )

        # One row per input row since 0.3.0, not one per category.
        assert result.shape == (len(df), 2)

    def test_performance_warnings(self) -> None:
        """Test that performance warnings are logged appropriately."""
        # Create a dataset that will trigger performance warnings
        large_df = generate_synthetic_data(1000)  # Large dataset warning

        # Create many columns to trigger high-dimensional warning
        for i in range(100):
            large_df[f"extra_col_{i}"] = np.random.randn(1000)

        result = learn_embedding(
            large_df,
            embedding_dim=50,  # High embedding dimension
            max_epochs=1,
            batch_size=100,
            enable_logging=True,
            log_level="WARNING",
        )

        # Should complete without errors
        assert result.shape[0] == 1000
        assert result.shape[1] == 50

    def test_validation_error_logging(self) -> None:
        """Test that validation errors are properly logged."""
        df = generate_synthetic_data(10)

        with pytest.raises(ValueError, match="embedding_dim"):
            learn_embedding(
                df,
                embedding_dim=1000,  # Too large, will trigger validation error
                max_epochs=1,
                batch_size=5,
                enable_logging=True,
                log_level="DEBUG",
            )


class TestTrainingProgressCallback:
    """Test suite for TrainingProgressCallback."""

    def test_callback_integration(self) -> None:
        """Test that the callback integrates properly with Keras training."""
        df = generate_synthetic_data(50)

        # This test ensures the callback doesn't break training
        result = learn_embedding(
            df,
            embedding_dim=3,
            max_epochs=3,
            batch_size=25,
            enable_logging=True,
            verbose=False,  # Disable Keras verbose to test our logging
        )

        assert result.shape == (50, 3)


if __name__ == "__main__":
    pytest.main([__file__])


def test_per_epoch_metrics_are_logged() -> None:
    """Each epoch must report its loss, not a warning.

    The training callback implemented on_epoch_end but never on_epoch_begin,
    so the logger never learned when an epoch started and log_epoch_metrics
    returned early every time. Per-epoch metrics were therefore never logged
    at all, and every training run emitted one WARNING per epoch instead.
    """
    import logging as _logging

    df = generate_synthetic_data(120)

    logger = _logging.getLogger("row2vec")
    records: list[_logging.LogRecord] = []

    class _Collect(_logging.Handler):
        def emit(self, record: _logging.LogRecord) -> None:
            records.append(record)

    handler = _Collect()
    logger.addHandler(handler)
    try:
        learn_embedding(
            df,
            mode="unsupervised",
            embedding_dim=2,
            max_epochs=3,
            enable_logging=True,
            log_level="INFO",
        )
    finally:
        logger.removeHandler(handler)

    messages = [r.getMessage() for r in records]

    assert not any("Epoch start time not recorded" in m for m in messages)
    assert sum("Epoch" in m and "completed in" in m for m in messages) >= 1
    assert any("loss=" in m for m in messages)
