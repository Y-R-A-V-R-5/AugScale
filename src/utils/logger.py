"""
src/utils/logger.py
====================
Centralized logging utility for Phase-2 YOLOv8 robustness framework.

This module provides a consistent logger configuration across all
components including:

- training pipeline
- evaluation engine
- ranking and visualization modules

Design Goals
-------------
- consistent log format across modules
- UTF-8 safe output (Windows + Linux compatible)
- single handler per logger (avoid duplicate logs)
- lightweight setup without external dependencies
"""

from __future__ import annotations

import io
import logging
import sys


# ---------------------------------------------------------------------
# UTF-8 Safe Stdout Wrapper
# ---------------------------------------------------------------------

def _utf8_stdout() -> io.TextIOWrapper:
    """
    Return UTF-8 wrapped stdout for cross-platform compatibility.

    Ensures logs display correctly on:
    - Windows (cp1252 default terminal)
    - Linux / macOS terminals
    """

    try:
        return io.TextIOWrapper(
            sys.stdout.buffer,
            encoding="utf-8",
            errors="replace",
            line_buffering=True,
        )

    except AttributeError:
        # Fallback for environments without buffer (e.g., IDLE, notebooks)
        return sys.stdout


# ---------------------------------------------------------------------
# Logger Factory
# ---------------------------------------------------------------------

def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Create or retrieve a configured logger instance.

    Parameters
    ----------
    name : str
        Logger namespace (usually __name__)

    level : int
        Logging level (default: INFO)

    Returns
    -------
    logging.Logger
        Configured logger with consistent formatting
    """

    logger = logging.getLogger(name)

    # Avoid duplicate handlers if logger already initialized
    if not logger.handlers:

        # UTF-8 safe output stream
        stream = _utf8_stdout()

        handler = logging.StreamHandler(stream)

        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s  %(levelname)-8s  %(name)s | %(message)s",
                datefmt="%H:%M:%S",
            )
        )

        logger.addHandler(handler)

    logger.setLevel(level)

    return logger