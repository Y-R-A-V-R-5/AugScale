"""
src/utils/io.py
================
Utility functions for file system and YAML operations.

This module provides:
- safe YAML loading/saving
- directory creation helpers
- UTF-8 consistent file handling

Used across:
- training pipeline (configs, datasets)
- evaluation pipeline (metrics, conditions)
- experiment outputs (logs, rankings, plots)
"""

from __future__ import annotations

from pathlib import Path
import yaml


# ---------------------------------------------------------------------
# YAML Loading
# ---------------------------------------------------------------------

def load_yaml(path: str | Path) -> dict:
    """
    Load a YAML file into a Python dictionary.

    Parameters
    ----------
    path : str | Path
        Path to YAML file

    Returns
    -------
    dict
        Parsed YAML content
    """

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------
# YAML Saving
# ---------------------------------------------------------------------

def save_yaml(data: dict, path: str | Path) -> None:
    """
    Save a Python dictionary to a YAML file.

    Parameters
    ----------
    data : dict
        Data to serialize

    path : str | Path
        Output file path

    Notes
    -----
    - Ensures parent directory exists
    - Uses UTF-8 encoding for cross-platform compatibility
    - Preserves readable YAML formatting
    """

    ensure_dir(Path(path).parent)

    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(
            data,
            f,
            default_flow_style=False,
            allow_unicode=True,
        )


# ---------------------------------------------------------------------
# Directory Utility
# ---------------------------------------------------------------------

def ensure_dir(path: str | Path) -> Path:
    """
    Ensure a directory exists.

    Parameters
    ----------
    path : str | Path
        Directory path to create

    Returns
    -------
    Path
        Resolved Path object of created/existing directory
    """

    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p