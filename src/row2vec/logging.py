"""
Logging utilities for Row2Vec library.

This module provides structured logging capabilities for training progress,
debug information, and performance metrics.
"""

import logging
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
import psutil


class Row2VecLogger:
    """
    Centralized logging system for Row2Vec operations.

    Provides structured logging for training progress, debug information,
    and performance metrics with configurable output formats and levels.
    """

    def __init__(
        self,
        name: str = "row2vec",
        level: str = "INFO",
        log_file: str | Path | None = None,
        include_performance: bool = True,
        include_memory: bool = True,
    ) -> None:
        """
        Initialize Row2Vec logger.

        Args:
            name: Logger name
            level: Logging level (DEBUG, INFO, WARNING, ERROR)
            log_file: Optional file path for logging output
            include_performance: Whether to include performance metrics
            include_memory: Whether to include memory usage tracking
        """
        self.logger = logging.getLogger(name)

        # Respect the current logging configuration level if it's already set
        # Only set level if it's not already configured or if it's more permissive
        current_level = self.logger.getEffectiveLevel()
        desired_level = getattr(logging, level.upper())

        if current_level > desired_level:
            self.logger.setLevel(desired_level)

        # Only add handlers if logger doesn't already have them AND the parent loggers don't have handlers
        # This allows external logging configuration to take precedence
        should_add_handler = not self.logger.handlers

        # Also check if parent loggers have handlers (indicating external configuration)
        parent = self.logger.parent
        while parent and should_add_handler:
            if parent.handlers:
                should_add_handler = False
                break
            parent = parent.parent

        # Only add our own handler if no existing handlers are found in the hierarchy
        if should_add_handler:
            # Configure formatter
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )

            # Console handler - only if no handlers exist in hierarchy
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

        # File handler if specifically requested
        if log_file:
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

        # Performance tracking flags
        self.include_performance = include_performance
        self.include_memory = include_memory

        # Training state tracking
        self.training_start_time: float | None = None
        self.epoch_start_time: float | None = None
        self.initial_memory: float | None = None

    def _should_log(self, level: int) -> bool:
        """
        Check if logging should occur at the given level.

        Respects both the logger's level and any parent logger levels.
        This allows external logging configuration to fully suppress output.
        """
        return self.logger.isEnabledFor(level)

    def start_training(self, **kwargs: Any) -> None:
        """Log training start with configuration details."""
        self.training_start_time = time.time()

        if self.include_memory:
            self.initial_memory = self._get_memory_usage()

        if self._should_log(logging.INFO):
            config_str = ", ".join([f"{k}={v}" for k, v in kwargs.items()])
            self.logger.info(f"🚀 Starting training with configuration: {config_str}")

        if self.include_memory and self._should_log(logging.DEBUG):
            self.logger.debug(f"Initial memory usage: {self.initial_memory:.2f} MB")

    def start_epoch(self, epoch: int, total_epochs: int) -> None:
        """Log epoch start."""
        self.epoch_start_time = time.time()
        if self._should_log(logging.INFO):
            self.logger.info(f"📊 Epoch {epoch + 1}/{total_epochs} started")

    def log_epoch_metrics(
        self,
        epoch: int,
        loss: float,
        val_loss: float | None = None,
        additional_metrics: dict[str, float] | None = None,
    ) -> None:
        """Log epoch completion with metrics."""
        if self.epoch_start_time is None:
            if self._should_log(logging.WARNING):
                self.logger.warning("Epoch start time not recorded")
            return

        epoch_duration = time.time() - self.epoch_start_time

        if self._should_log(logging.INFO):
            # Build metrics message
            metrics_parts = [f"loss={loss:.6f}"]
            if val_loss is not None:
                metrics_parts.append(f"val_loss={val_loss:.6f}")

            if additional_metrics:
                for metric, value in additional_metrics.items():
                    metrics_parts.append(f"{metric}={value:.6f}")

            metrics_str = ", ".join(metrics_parts)

            if self.include_performance:
                self.logger.info(
                    f"✅ Epoch {epoch + 1} completed in {epoch_duration:.2f}s - {metrics_str}",
                )
            else:
                self.logger.info(f"✅ Epoch {epoch + 1} completed - {metrics_str}")

        # Memory tracking
        if self.include_memory and self._should_log(logging.DEBUG):
            current_memory = self._get_memory_usage()
            self.logger.debug(f"Memory usage: {current_memory:.2f} MB")

    def log_early_stopping(self, epoch: int, reason: str) -> None:
        """Log early stopping event."""
        if self._should_log(logging.INFO):
            self.logger.info(f"⏹️  Early stopping at epoch {epoch + 1}: {reason}")

    def end_training(self, final_loss: float, total_epochs: int) -> None:
        """Log training completion with summary."""
        if self.training_start_time is None:
            if self._should_log(logging.WARNING):
                self.logger.warning("Training start time not recorded")
            return

        total_duration = time.time() - self.training_start_time

        if self._should_log(logging.INFO):
            self.logger.info(
                f"🎉 Training completed! Final loss: {final_loss:.6f}, "
                f"Total time: {total_duration:.2f}s, Epochs: {total_epochs}",
            )

            if self.include_performance:
                avg_epoch_time = total_duration / max(total_epochs, 1)
                self.logger.info(f"📈 Average epoch time: {avg_epoch_time:.2f}s")

            if self.include_memory and self.initial_memory:
                final_memory = self._get_memory_usage()
                memory_delta = final_memory - self.initial_memory
                self.logger.info(
                    f"💾 Memory usage: {final_memory:.2f} MB "
                    f"(+{memory_delta:.2f} MB from start)",
                )

    def log_data_preprocessing(
        self, df_shape: tuple[int, int], processing_steps: list[str]
    ) -> None:
        """Log data preprocessing information."""
        if self._should_log(logging.INFO):
            self.logger.info(f"🔄 Preprocessing data: shape {df_shape}")
        if self._should_log(logging.DEBUG):
            for step in processing_steps:
                self.logger.debug(f"  - {step}")

    def log_preprocessing_result(
        self,
        original_shape: tuple[int, int],
        processed_shape: tuple[int, int],
        processing_time: float,
    ) -> None:
        """Log preprocessing completion."""
        if self._should_log(logging.INFO):
            self.logger.info(
                f"✅ Preprocessing completed in {processing_time:.2f}s: "
                f"{original_shape} → {processed_shape}",
            )

    def log_model_architecture(self, model_summary: str) -> None:
        """Log model architecture details."""
        if self._should_log(logging.DEBUG):
            self.logger.debug("🏗️  Model architecture:")
            for line in model_summary.split("\n"):
                if line.strip():
                    self.logger.debug(f"  {line}")

    def log_embedding_stats(self, embeddings: pd.DataFrame) -> None:
        """Log embedding statistics."""
        if self._should_log(logging.INFO):
            stats = {
                "shape": embeddings.shape,
                "mean": float(embeddings.mean().mean()),
                "std": float(embeddings.std().mean()),
                "min": float(embeddings.min().min()),
                "max": float(embeddings.max().max()),
            }

            self.logger.info(
                f"📊 Embedding statistics: shape={stats['shape']}, "
                f"mean={stats['mean']:.4f}, std={stats['std']:.4f}, "
                f"range=[{stats['min']:.4f}, {stats['max']:.4f}]",
            )

    def log_performance_warning(self, message: str) -> None:
        """Log performance-related warnings."""
        if self._should_log(logging.WARNING):
            self.logger.warning(f"⚠️ Performance Warning: {message}")

    def log_validation_issue(self, message: str) -> None:
        """Log validation or data quality issues."""
        if self._should_log(logging.WARNING):
            self.logger.warning(f"⚠️ Validation Issue: {message}")

    def log_debug_info(self, message: str, data: dict[str, Any] | None = None) -> None:
        """Log debug information with optional data context."""
        if self._should_log(logging.DEBUG):
            debug_msg = f"🔍 {message}"
            if data:
                debug_msg += f" | Data: {data}"
            self.logger.debug(debug_msg)

    def log_error(self, error: Exception, context: str | None = None) -> None:
        """Log error with context information."""
        if self._should_log(logging.ERROR):
            error_msg = f"❌ Error: {error!s}"
            if context:
                error_msg += f" | Context: {context}"
            self.logger.error(error_msg, exc_info=True)

    def log_completion(
        self, message: str = "Embedding generation completed successfully!"
    ) -> None:
        """Log completion of embedding generation."""
        if self._should_log(logging.INFO):
            self.logger.info(f"🎯 {message}")

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            process = psutil.Process()
            return float(process.memory_info().rss / 1024 / 1024)  # Convert to MB
        except Exception:
            return 0.0


class TrainingProgressCallback:
    """
    Keras callback for logging training progress.

    Integrates with Row2VecLogger to provide structured logging
    during model training.
    """

    def __init__(self, logger: Row2VecLogger) -> None:
        """Initialize callback with logger instance."""
        self.logger = logger
        self.epoch_start_time: float | None = None

    def set_model(self, model: Any) -> None:
        """Set the model (required by Keras)."""
        self.model = model

    def set_params(self, params: Any) -> None:
        """Set training parameters (required by Keras)."""
        self.params = params

    def on_train_begin(self, logs: Any = None) -> None:
        """Called at the beginning of training."""
        self.logger.logger.debug("🔄 Keras training started")

    def on_epoch_begin(self, epoch: int, logs: Any = None) -> None:
        """Called at the beginning of each epoch."""
        self.epoch_start_time = time.time()

    def on_epoch_end(self, epoch: int, logs: Any = None) -> None:
        """Called at the end of each epoch."""
        if logs is None:
            logs = {}

        # Extract metrics
        loss = logs.get("loss", 0.0)
        val_loss = logs.get("val_loss")

        # Remove loss from additional metrics to avoid duplication
        additional_metrics = {
            k: v for k, v in logs.items() if k not in ["loss", "val_loss"]
        }

        self.logger.log_epoch_metrics(
            epoch=epoch,
            loss=loss,
            val_loss=val_loss,
            additional_metrics=additional_metrics if additional_metrics else None,
        )

    def on_train_end(self, logs: Any = None) -> None:
        """Called at the end of training."""
        self.logger.logger.debug("✅ Keras training completed")


# Convenience function for creating loggers
def get_logger(
    name: str = "row2vec",
    level: str = "INFO",
    log_file: str | Path | None = None,
    **kwargs: Any,
) -> Row2VecLogger:
    """
    Create a Row2Vec logger with standard configuration.

    Args:
        name: Logger name
        level: Logging level
        log_file: Optional log file path
        **kwargs: Additional arguments for Row2VecLogger

    Returns:
        Configured Row2VecLogger instance
    """
    return Row2VecLogger(name=name, level=level, log_file=log_file, **kwargs)
