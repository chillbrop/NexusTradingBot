"""NexusTrade — Logging Configuration"""

import logging
import os
import sys
from logging.handlers import RotatingFileHandler


def setup_logging(level: str = "INFO", log_file: str = "logs/nexustrade.log",
                  console: bool = True):
    """Configure root logger with file rotation and optional console output."""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)-22s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Rotating file handler (10 MB × 5 files)
    fh = RotatingFileHandler(log_file, maxBytes=10 * 1024 * 1024, backupCount=5)
    fh.setFormatter(fmt)
    root.addHandler(fh)

    if console:
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(fmt)
        root.addHandler(ch)

    # Silence noisy third-party loggers
    for lib in ("websockets", "aiohttp", "urllib3", "asyncio"):
        logging.getLogger(lib).setLevel(logging.WARNING)
