"""
Configuration Management for REMEMVR

Centralized loading and access to YAML configuration files:
- config/paths.yaml - File paths and directory structures
- config/plotting.yaml - Plotting aesthetics and parameters
- config/irt.yaml - IRT model settings and purification
- config/lmm.yaml - LMM formulas and fitting parameters

Usage:
    from tools.config import (
        load_config_from_yaml,
        resolve_path_from_config,
        load_plot_config_from_yaml,
        load_irt_config_from_yaml,
        load_lmm_config_from_yaml
    )

    # Get full config
    config = load_config_from_yaml('paths')

    # Get nested value with dot notation
    master_path = resolve_path_from_config('data.master')
    dpi = load_plot_config_from_yaml('global.dpi')

    # Get section
    colors = load_plot_config_from_yaml('colors.factors')

Design Principles:
- Single source of truth for all configuration
- Type-safe access with clear error messages
- Support for RQ-specific overrides
- Validation on load (paths exist, values in range)
- Cache configs after first load

Status: TEMPLATE - To be implemented in Phase 1

Implementation Steps:
1. Install PyYAML: poetry add pyyaml
2. Implement config loading with caching
3. Add path validation and creation
4. Support environment variable substitution
5. Implement RQ-specific override mechanism
6. Add unit tests (tests/test_config.py)

TDD Approach:
- Write tests first (RED phase)
- Implement functions to pass tests (GREEN phase)
- Refactor for clarity (REFACTOR phase)
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional
import yaml


# ─── CONFIGURATION CACHE ─────────────────────────────────────────────────────

_CONFIG_CACHE: Dict[str, Dict] = {}


# ─── CORE LOADING FUNCTION ───────────────────────────────────────────────────

def load_config_from_file(config_name: str) -> Dict[str, Any]:
    """
    Load YAML configuration file.

    Args:
        config_name: Name of config (e.g., 'paths', 'plotting', 'irt', 'lmm')

    Returns:
        Dictionary containing configuration

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid

    Example:
        config = load_config_from_file('paths')
        master_path = config['data']['master']
    """
    # 1. Check cache first
    if config_name in _CONFIG_CACHE:
        return _CONFIG_CACHE[config_name]

    # 2. Construct path to config/{config_name}.yaml
    # Get project root (go up from tools/ to project root)
    project_root = Path(__file__).parent.parent
    config_path = project_root / "config" / f"{config_name}.yaml"

    # 3. Load YAML
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Invalid YAML in {config_path}: {e}")

    # 4. Validate structure (basic check)
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a dictionary, got {type(config)}")

    # 5. Cache and return
    _CONFIG_CACHE[config_name] = config
    return config


def load_config_from_yaml(config_name: str, key_path: Optional[str] = None) -> Any:
    """
    Get configuration value by key path.

    Args:
        config_name: Name of config ('paths', 'plotting', 'irt', 'lmm')
        key_path: Dot-separated path to value (e.g., 'data.master')
                  If None, returns entire config

    Returns:
        Configuration value

    Raises:
        KeyError: If key_path doesn't exist

    Example:
        master = load_config_from_yaml('paths', 'data.master')
        colors = load_config_from_yaml('plotting', 'colors.factors')
    """
    # 1. Load config (uses cache)
    config = load_config_from_file(config_name)

    # 2. If key_path is None, return full config
    if key_path is None:
        return config

    # 3. Split key_path by '.'
    keys = key_path.split('.')

    # 4. Navigate nested dict
    value = config
    for key in keys:
        if not isinstance(value, dict):
            raise KeyError(f"Cannot navigate '{key}' in non-dict value at '{'.'.join(keys[:keys.index(key)])}'")
        if key not in value:
            raise KeyError(f"Key '{key}' not found in config '{config_name}' at path '{key_path}'")
        value = value[key]

    # 5. Return value
    return value


# ─── CONVENIENCE FUNCTIONS ───────────────────────────────────────────────────

def resolve_path_from_config(key_path: str, **kwargs) -> Path:
    """
    Get path from paths.yaml.

    Args:
        key_path: Dot-separated path (e.g., 'data.master')
        **kwargs: Template variables for path formatting (e.g., n=1 for rq{n})

    Returns:
        Path object (absolute)

    Example:
        master = resolve_path_from_config('data.master')
        rq_path = resolve_path_from_config('results.ch5.rq_template', n=1)  # results/ch5/rq1/
    """
    # 1. Get path string from config
    path_str = load_config_from_yaml('paths', key_path)

    if not isinstance(path_str, str):
        raise ValueError(f"Path value at '{key_path}' must be a string, got {type(path_str)}")

    # 2. Format with kwargs if template
    if kwargs:
        path_str = path_str.format(**kwargs)

    # 3. Convert to absolute Path
    # Get project root (go up from tools/ to project root)
    project_root = Path(__file__).parent.parent
    path = Path(path_str)

    # Make absolute if relative
    if not path.is_absolute():
        path = project_root / path

    # 4. Return path (don't validate existence - let caller decide)
    return path


def load_plot_config_from_yaml(key_path: Optional[str] = None) -> Any:
    """Get value from plotting.yaml."""
    return load_config_from_yaml('plotting', key_path)


def load_irt_config_from_yaml(key_path: Optional[str] = None) -> Any:
    """Get value from irt.yaml."""
    return load_config_from_yaml('irt', key_path)


def load_lmm_config_from_yaml(key_path: Optional[str] = None) -> Any:
    """Get value from lmm.yaml."""
    return load_config_from_yaml('lmm', key_path)


# ─── VALIDATION FUNCTIONS ────────────────────────────────────────────────────

def validate_paths_exist(config: Dict[str, Any]) -> None:
    """
    Validate that critical paths exist.

    Args:
        config: Paths configuration dictionary

    Raises:
        FileNotFoundError: If critical path doesn't exist
    """
    # TODO: Implement
    # 1. Check data.master exists
    # 2. Create output directories if needed
    # 3. Log warnings for missing optional paths
    raise NotImplementedError("To be implemented in Phase 1")


def validate_irt_params(config: Dict[str, Any]) -> None:
    """
    Validate IRT configuration.

    Args:
        config: IRT configuration dictionary

    Raises:
        ValueError: If config values are invalid
    """
    # TODO: Implement
    # 1. Check batch_size > 0
    # 2. Check thresholds are reasonable
    # 3. Validate factor definitions
    raise NotImplementedError("To be implemented in Phase 1")


# ─── RQ-SPECIFIC OVERRIDES ───────────────────────────────────────────────────

def merge_config_dicts(base: Dict, override: Dict) -> Dict:
    """
    Deep merge two dictionaries (override takes precedence).

    Args:
        base: Base dictionary
        override: Override dictionary (takes precedence)

    Returns:
        Merged dictionary (new dict, doesn't modify inputs)

    Example:
        >>> base = {'a': 1, 'b': {'c': 2, 'd': 3}}
        >>> override = {'b': {'d': 4, 'e': 5}}
        >>> merge_config_dicts(base, override)
        {'a': 1, 'b': {'c': 2, 'd': 4, 'e': 5}}
    """
    result = base.copy()

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            # Recursively merge nested dicts
            result[key] = merge_config_dicts(result[key], value)
        else:
            # Override takes precedence
            result[key] = value

    return result


def load_rq_config_merged(chapter: int, rq: int) -> Dict[str, Any]:
    """
    Load RQ-specific configuration with overrides.

    Merges:
    1. Global configs (config/*.yaml)
    2. Chapter-level overrides (results/ch{chapter}/shared/config.yaml)
    3. RQ-level overrides (results/ch{chapter}/rq{rq}/config.yaml)

    Args:
        chapter: Chapter number (5, 6, 7)
        rq: RQ number (1-15 for ch5/ch6, 1-20 for ch7)

    Returns:
        Merged configuration dictionary

    Example:
        config = load_rq_config_merged(5, 1)  # Loads config for RQ 5.1
    """
    # 1. Load global configs
    global_config = {}

    # Load each global config file and add to merged dict
    config_names = ['paths', 'plotting', 'irt', 'lmm', 'logging', 'models']
    for config_name in config_names:
        try:
            cfg = load_config_from_file(config_name)
            global_config[config_name] = cfg
        except FileNotFoundError:
            # Config file doesn't exist, skip
            pass

    # 2. Load chapter-level config if exists
    project_root = Path(__file__).parent.parent
    chapter_config_path = project_root / "results" / f"ch{chapter}" / "shared" / "config.yaml"

    chapter_config = {}
    if chapter_config_path.exists():
        with open(chapter_config_path, 'r') as f:
            chapter_config = yaml.safe_load(f) or {}

    # 3. Load RQ-level config
    rq_config_path = project_root / "results" / f"ch{chapter}" / f"rq{rq}" / "config.yaml"

    if not rq_config_path.exists():
        raise FileNotFoundError(f"RQ config not found: {rq_config_path}")

    with open(rq_config_path, 'r') as f:
        rq_config = yaml.safe_load(f)

    if not isinstance(rq_config, dict):
        raise ValueError(f"RQ config must be a dictionary, got {type(rq_config)}")

    # 4. Deep merge: global < chapter < RQ
    merged = global_config

    if chapter_config:
        merged = merge_config_dicts(merged, chapter_config)

    merged = merge_config_dicts(merged, rq_config)

    # 5. Return merged config
    return merged


# ─── ENVIRONMENT VARIABLE SUBSTITUTION ───────────────────────────────────────

def expand_env_vars_in_path(path_str: str) -> str:
    """
    Expand environment variables in path string.

    Args:
        path_str: Path with ${VAR} placeholders

    Returns:
        Path with variables expanded

    Example:
        expand_env_vars_in_path("${REMEMVR_ROOT}/data/master.xlsx")
    """
    # TODO: Implement
    # Use os.path.expandvars or custom implementation
    raise NotImplementedError("To be implemented in Phase 1")


# ─── TESTING HELPERS ─────────────────────────────────────────────────────────

def reset_config_cache() -> None:
    """
    Reset configuration cache.

    Useful for testing to force reload of configs.
    """
    global _CONFIG_CACHE
    _CONFIG_CACHE = {}


# ─── EXAMPLE USAGE ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Example usage (will work after implementation).
    """
    print("Configuration system examples:")
    print()

    # Load full config
    # paths_config = load_config_from_yaml('paths')
    # print(f"Paths config: {paths_config}")

    # Get specific values
    # master_path = resolve_path_from_config('data.master')
    # print(f"Master data: {master_path}")

    # Get plotting settings
    # colors = load_plot_config_from_yaml('colors.factors')
    # print(f"Factor colors: {colors}")

    # Get IRT settings
    # batch_size = load_irt_config_from_yaml('calibration.model_fit.batch_size')
    # print(f"Batch size: {batch_size}")

    # Load RQ-specific config
    # rq_config = load_rq_config_merged(5, 1)
    # print(f"RQ 5.1 config: {rq_config}")

    print("(Not yet implemented - Phase 1)")
