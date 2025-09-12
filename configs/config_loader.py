import os
from typing import Any, Dict
import yaml

_DEFAULT_PATH = "configs/default.yaml"

def load_config(config_path: str = _DEFAULT_PATH) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.

    Args:
        config_path: Path to the YAML configuration file.
    Returns:
        A dictionary containing the configuration.
    """
    
    if not os.path.exists(config_path):
        print(f"Configuration file {config_path} does not exist.")
        return {}
    
    with open(config_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file) or {}
    
    return config

def get(config: Dict[str, Any], *keys, default = None):
    """
    Retrieve a value from the configuration dictionary with a default fallback.

    Args:
        config: The configuration dictionary.
        keys: keys to look for.
        default: The default value to return if the key is not found.
    Returns:
        The value associated with the key, or the default value.
    """
    
    cur = config
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur