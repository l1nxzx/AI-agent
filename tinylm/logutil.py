"""Logging helpers: human-readable log to console + file, and machine-readable metrics as JSON lines."""
import json
import logging
import os
import sys
import time

LOGGER_NAME = "tinylm"


def setup_logging(log_file=None, level=logging.INFO):
    """Log to stdout (flushed per line so Colab shows it live) and, if given, to a file (append mode,
    so a resumed run continues the same log)."""
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level)
    logger.handlers.clear()
    logger.propagate = False
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(fmt)
    logger.addHandler(stream)
    if log_file:
        os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
        fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


def get_logger():
    return logging.getLogger(LOGGER_NAME)


class MetricsWriter:
    """Appends one JSON object per line. Easy to load with pandas or json for plotting later."""

    def __init__(self, path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.path = path

    def write(self, **record):
        record["time"] = time.time()
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
