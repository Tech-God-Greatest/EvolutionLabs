"""Ecosystem State Persistence: Save Engine.

Serializes the complete simulation state into versioned JSON or gzip-compressed
JSON files (`digital_evolution_save_v1.json`).
"""
from __future__ import annotations

import gzip
import json
import os
from typing import Any
import numpy as np

from ..core.errors import SaveError

SAVE_FORMAT = "digital_evolution_save"
SAVE_VERSION = 1


def _to_serializable(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(x) for x in obj]
    return obj


def save_ecosystem(eco, path: str, compress: bool = False) -> None:
    """Save the full ecosystem state to disk."""
    try:
        data = {
            "format": SAVE_FORMAT,
            "version": SAVE_VERSION,
            "ecosystem": eco.state_dict(),
        }
        serializable_data = _to_serializable(data)

        # Temporary file write to prevent corruption on crash
        tmp_path = path + ".tmp"
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

        if compress or path.endswith(".gz"):
            with gzip.open(tmp_path, "wt", encoding="utf-8") as f:
                json.dump(serializable_data, f)
        else:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(serializable_data, f, indent=2)

        if os.path.exists(path):
            os.remove(path)
        os.rename(tmp_path, path)

    except Exception as e:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        raise SaveError(f"Failed to save simulation state to '{path}': {e}") from e
