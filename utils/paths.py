"""Portable locations for downloaded Wan weights."""
import os
from pathlib import Path


def wan_model_path(relative_path: str) -> str:
    root = Path(os.environ.get("DUOMATCHING_WAN_ROOT", "wan_models")).expanduser()
    return str(root / relative_path)
