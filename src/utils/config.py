# src/utils/config.py
"""
config.py
---------
Utility functions for loading and accessing YAML configuration files.
All processing nodes (elements, chunks, embed, weaviate) import from here.
"""

import yaml
from pathlib import Path
from typing import Any, Dict, Optional


def load_config(config_path: str = "configs/default.yaml") -> Dict[str, Any]:
    """
    Load the global configuration YAML file.

    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        Dict[str, Any]: Parsed configuration dictionary.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def get_section(config: Dict[str, Any], section: str, default: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Retrieve a specific section from the configuration.

    Args:
        config (Dict[str, Any]): Full configuration dictionary.
        section (str): Key of the desired section (e.g., 'embedding', 'weaviate').
        default (Optional[Dict[str, Any]]): Default value if section is missing.

    Returns:
        Dict[str, Any]: Section dictionary.
    """
    return config.get(section, default or {})


def resolve_path(config: Dict[str, Any], *keys) -> Path:
    """
    Safely resolve nested path keys from the configuration.

    Args:
        config (Dict[str, Any]): Full configuration dictionary.
        *keys: One or more nested keys (e.g., 'paths', 'elements_dir').

    Returns:
        Path: Resolved absolute Path.
    """
    sub = config
    for k in keys:
        sub = sub.get(k, {})
    if not isinstance(sub, str):
        raise ValueError(f"Invalid path key: {'/'.join(keys)}")
    return Path(sub).expanduser().resolve()
