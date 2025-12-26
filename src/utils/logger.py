import logging
import sys

def setup_logging(level: str = "INFO", logger_name: str = "FashionEngine") -> logging.Logger:
    """
    Configures and returns the application logger.
    Idempotent: will not add duplicate handlers.
    """
    logger = logging.getLogger(logger_name)
    
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(numeric_level)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        # Cleaner, aligned format
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-7s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False  # Avoid duplicates from root logger

    return logger

def get_logger(name: str = "FashionEngine") -> logging.Logger:
    return logging.getLogger(name)
