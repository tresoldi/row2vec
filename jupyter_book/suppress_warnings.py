"""
Configuration to suppress warnings and logging during notebook execution.
This module should be imported at the beginning of notebooks for clean output.
"""

import os
import warnings
import logging

# Suppress all warnings
warnings.filterwarnings('ignore')

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN warnings

# Suppress other common warnings
os.environ['PYTHONWARNINGS'] = 'ignore'

# Configure logging to only show critical errors
logging.getLogger().setLevel(logging.CRITICAL)
logging.getLogger('tensorflow').setLevel(logging.CRITICAL)
logging.getLogger('keras').setLevel(logging.CRITICAL)
logging.getLogger('absl').setLevel(logging.CRITICAL)

# Suppress specific TensorFlow warnings
import tensorflow as tf
if hasattr(tf, 'get_logger'):
    tf.get_logger().setLevel('ERROR')

# Suppress sklearn warnings
from sklearn.utils import all_estimators
import sklearn
sklearn.set_config(assume_finite=True)

print("✓ Warnings and logging suppressed for clean output")