"""
Complete suppression of all warnings, logging, and verbose output.
Import this module FIRST before any other imports.
"""
import warnings
import logging
import os
import sys
from io import StringIO

# Suppress all warnings
warnings.filterwarnings('ignore')

# Set environment variables before importing TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' 
os.environ['PYTHONWARNINGS'] = 'ignore'
# Additional TensorFlow suppression
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Disable CUDA to avoid GPU warnings
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['AUTOGRAPH_VERBOSITY'] = '0'

# Configure logging to suppress everything except critical errors
root_logger = logging.getLogger()
root_logger.setLevel(logging.CRITICAL)

# Add a null handler to indicate external logging configuration
# This prevents row2vec from creating its own console handler
null_handler = logging.NullHandler()
root_logger.addHandler(null_handler)

# Suppress specific common loggers
loggers_to_suppress = [
    'tensorflow',
    'keras', 
    'absl',
    'row2vec',
    'row2vec.learn_embedding',
    'row2vec.core',
    'sklearn',
    'matplotlib',
    'pandas'
]

for logger_name in loggers_to_suppress:
    logging.getLogger(logger_name).setLevel(logging.CRITICAL)

# Additional suppression for TensorFlow
try:
    import tensorflow as tf
    if hasattr(tf, 'get_logger'):
        tf.get_logger().setLevel('ERROR')
    if hasattr(tf, 'logging'):
        tf.logging.set_verbosity(tf.logging.ERROR)
    
    # Disable Keras progress bars and verbose output
    os.environ['TF_KERAS_UTILS_DISABLE_INTERACTIVE_LOGGING'] = '1'
    os.environ['KERAS_BACKEND'] = 'tensorflow'
    
    # Suppress TensorFlow model.summary() output
    try:
        from tensorflow.keras.models import Model as KerasModel
        original_summary = KerasModel.summary
        def silent_summary(self, *args, **kwargs):
            # Completely suppress summary output
            return None
        KerasModel.summary = silent_summary
    except ImportError:
        pass
    
    # Monkey-patch keras.utils.Progbar to be silent
    try:
        from tensorflow.keras.utils import Progbar
        original_progbar_init = Progbar.__init__
        def silent_progbar(self, *args, **kwargs):
            kwargs['verbose'] = 0
            return original_progbar_init(self, *args, **kwargs)
        Progbar.__init__ = silent_progbar
        
        # Also override update method to be silent
        original_progbar_update = Progbar.update
        def silent_progbar_update(self, *args, **kwargs):
            # Capture and discard any output
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            sys.stdout = StringIO()
            sys.stderr = StringIO()
            try:
                result = original_progbar_update(self, *args, **kwargs)
            finally:
                sys.stdout = old_stdout
                sys.stderr = old_stderr
            return result
        Progbar.update = silent_progbar_update
    except ImportError:
        pass
    
    # Force all Keras model.fit() calls to be non-verbose
    from tensorflow.keras import Model
    original_fit = Model.fit
    def silent_fit(self, *args, **kwargs):
        kwargs['verbose'] = 0  # Force verbose=0 for all fit calls
        # Capture any remaining output
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = StringIO()
        sys.stderr = StringIO()
        try:
            result = original_fit(self, *args, **kwargs)
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
        return result
    Model.fit = silent_fit
    
except ImportError:
    pass

# Suppress tqdm and other progress bars
try:
    import tqdm
    # Disable all tqdm progress bars by setting disable=True globally
    original_tqdm_init = tqdm.tqdm.__init__
    def silent_tqdm_init(self, *args, **kwargs):
        kwargs['disable'] = True
        kwargs['file'] = open(os.devnull, 'w')
        return original_tqdm_init(self, *args, **kwargs)
    tqdm.tqdm.__init__ = silent_tqdm_init
    
    # Also override auto tqdm
    if hasattr(tqdm, 'auto'):
        tqdm.auto.tqdm = lambda *args, **kwargs: tqdm.tqdm(*args, **{**kwargs, 'disable': True})
except ImportError:
    pass

# Suppress any other common progress bar libraries
try:
    import progressbar
    progressbar.streams.wrap_stderr = lambda: None
except ImportError:
    pass

# Global context manager for complete output suppression during training
class SuppressAllOutput:
    def __init__(self):
        self.devnull = open(os.devnull, 'w')
        self.old_stdout = None
        self.old_stderr = None
    
    def __enter__(self):
        self.old_stdout = sys.stdout
        self.old_stderr = sys.stderr
        sys.stdout = self.devnull
        sys.stderr = self.devnull
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr
        self.devnull.close()

# Make the suppressor available globally
suppress_output = SuppressAllOutput

# Add context manager for suppressing all output during specific operations
import contextlib

@contextlib.contextmanager
def suppress_all_output():
    """Context manager that suppresses all stdout and stderr output."""
    with open(os.devnull, 'w') as devnull:
        old_stdout, old_stderr = sys.stdout, sys.stderr
        try:
            sys.stdout = devnull
            sys.stderr = devnull
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

# Make available for import
__all__ = ['suppress_all_output', 'SuppressAllOutput']

print("✓ Complete output suppression system active (including TensorFlow/Keras progress bars and model summaries)")