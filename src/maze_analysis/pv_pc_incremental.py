"""Manuscript definitions collected from pv_pc_incremental.py.
See provenance/source_files.csv for original source-file hashes.
"""

from __future__ import annotations

import hashlib

import json

from pathlib import Path

import numpy as np

CACHE_SCHEMA_VERSION = 1

def analysis_cache_path(cache_root, analysis, config, mouse):
    """Return a deterministic subject-shard path for one analysis config."""
    normalized = _normalize_value(config)
    encoded = json.dumps(normalized, sort_keys=True, separators=(",", ":"), default=str)
    config_tag = hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:12]
    return Path(cache_root) / str(analysis) / config_tag / f"{mouse}.npy"

def _normalize_value(value):
    if isinstance(value, dict):
        return {str(key): _normalize_value(item) for key, item in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return tuple(_normalize_value(item) for item in value)
    if isinstance(value, np.ndarray):
        return tuple(_normalize_value(item) for item in value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value.resolve())
    return value
