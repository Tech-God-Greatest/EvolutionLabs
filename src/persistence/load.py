"""Ecosystem State Persistence: Load Engine.

Validates save format and version, checks integrity, and restores the full
ecosystem simulation state from JSON or gzip-compressed files.
"""
from __future__ import annotations

import gzip
import json
import os
from typing import Optional

from ..core.errors import SaveError
from ..core.config import Config
from ..simulation.ecosystem import Ecosystem
from .save import SAVE_FORMAT, SAVE_VERSION


def load_ecosystem(path: str, eco: Optional[Ecosystem] = None) -> Ecosystem:
    """Load ecosystem from file.

    If an existing Ecosystem instance is provided, it is updated in-place;
    otherwise, a new Ecosystem instance is constructed.
    """
    if not os.path.exists(path):
        raise SaveError(f"Save file not found: {path}")

    try:
        if path.endswith(".gz"):
            with gzip.open(path, "rt", encoding="utf-8") as f:
                data = json.load(f)
        else:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
    except json.JSONDecodeError as e:
        raise SaveError(f"Corrupted or invalid JSON in save file '{path}': {e}") from e
    except Exception as e:
        raise SaveError(f"Cannot read save file '{path}': {e}") from e

    if not isinstance(data, dict):
        raise SaveError(f"Invalid save format in '{path}': root must be a JSON object")

    fmt = data.get("format")
    if fmt != SAVE_FORMAT:
        raise SaveError(f"Unrecognized file format in '{path}': expected '{SAVE_FORMAT}', got '{fmt}'")

    ver = data.get("version", 0)
    if ver > SAVE_VERSION:
        raise SaveError(f"Save version {ver} is newer than supported version {SAVE_VERSION}")

    eco_dict = data.get("ecosystem")
    if not isinstance(eco_dict, dict):
        raise SaveError(f"Save file '{path}' is missing 'ecosystem' payload")

    try:
        cfg_dict = eco_dict.get("config", {})
        cfg = Config(cfg_dict)

        if eco is None:
            # Construct uninitialized ecosystem and load state
            eco = Ecosystem(config=cfg, seed=cfg.seed, init=False)

        eco.load_state_dict(eco_dict)
        return eco

    except Exception as e:
        raise SaveError(f"Failed to restore simulation state from '{path}': {e}") from e
