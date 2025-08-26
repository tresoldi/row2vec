"""
Minimal suppression system that doesn't crash the kernel.
"""
import warnings
import logging
import os

# Suppress all warnings
warnings.filterwarnings('ignore')

# Basic TensorFlow suppression
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['PYTHONWARNINGS'] = 'ignore'

# Configure root logger to be completely silent
root_logger = logging.getLogger()
root_logger.setLevel(logging.CRITICAL)

# Clear any existing handlers first
for handler in root_logger.handlers[:]:
    root_logger.removeHandler(handler)

# Add null handler to prevent any output
root_logger.addHandler(logging.NullHandler())

# Suppress common library loggers more aggressively
loggers_to_suppress = [
    'tensorflow', 'keras', 'row2vec', 'row2vec.learn_embedding', 
    'row2vec.core', 'sklearn', 'absl'
]

for logger_name in loggers_to_suppress:
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.CRITICAL)
    # Clear handlers and prevent propagation
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    logger.addHandler(logging.NullHandler())
    logger.propagate = False

print("✓ Enhanced minimal suppression active")