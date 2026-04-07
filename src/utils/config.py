"""Configuration loader for YAML config files.

Responsibilities
----------------
- Read a YAML file from disk.
- Optionally validate that a set of required keys is present.
- Return the parsed data as a plain dict.

Design notes
------------
- Pure function: no global state, no mutation of the caller's namespace.
- All validation errors are raised as ValueError so callers need only catch
  one exception type (KeyError is also re-raised as ValueError for consistency).
- yaml.safe_load is used — never yaml.load — to prevent arbitrary code execution.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(
    path: str,
    required_keys: list[str] | None = None,
) -> dict[str, Any]:
    """Load a YAML configuration file and validate required keys.

    Parameters
    ----------
    path:
        Filesystem path to a ``.yaml`` / ``.yml`` file.
    required_keys:
        Optional list of keys that *must* be present in the loaded dict.
        Raises ``ValueError`` if any are missing.

    Returns
    -------
    dict[str, Any]
        Parsed YAML content.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist on disk.
    ValueError
        If the file cannot be parsed as YAML, if the top-level value is not a
        mapping, or if any *required_keys* are absent.
    """
    config_path = Path(path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    raw_text = config_path.read_text(encoding="utf-8")

    try:
        data = yaml.safe_load(raw_text)
    except yaml.YAMLError as exc:
        raise ValueError(f"Failed to parse YAML from '{path}': {exc}") from exc

    # An empty file produces None from safe_load; normalise to empty dict.
    if data is None:
        data = {}

    if not isinstance(data, dict):
        raise ValueError(
            f"Expected a YAML mapping at the top level of '{path}', "
            f"got {type(data).__name__}."
        )

    if required_keys:
        missing = [k for k in required_keys if k not in data]
        if missing:
            raise ValueError(
                f"Config file '{path}' is missing required key(s): {missing}"
            )

    return data
