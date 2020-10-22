import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import logging

logging.disable(logging.WARNING)
import warnings

warnings.filterwarnings("ignore")