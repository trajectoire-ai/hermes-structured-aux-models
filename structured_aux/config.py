"""Settings and credential resolution for the ``structured-aux`` provider.

A model-provider plugin is loaded by ``providers/__init__.py``, not by the general
``PluginManager``, so there is no ``ctx`` object to read configuration from. Settings
are therefore read lazily from the live Hermes config at
``plugins.entries.<PLUGIN_ID>.settings``, with an environment fallback for the
credential only.

Nothing here raises on a missing or malformed setting: an unconfigured plugin must
degrade to "cannot serve this request", which makes Hermes fall back to the
operator's real auxiliary provider.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

PLUGIN_ID = "hermes-structured-structure-aux-models"
PROVIDER_NAME = "structured-aux"

DEFAULT_BASE_URL = "https://openrouter.ai"
DEFAULT_PATH = "/api/alpha/decisions"
DEFAULT_MODEL = "typesafe/jev-1.13"
CREDENTIAL_ENV_VAR = "OPENROUTER_API_KEY"

# Task -> the model id suffix an operator writes into auxiliary.<task>.model.
# The shim reads this back off the request to know which contract to apply.
MODEL_PREFIX = "structured-aux"

_DEFAULTS: dict[str, Any] = {
    "tasks": ("approval", "mcp", "compression"),
    "decision_model": DEFAULT_MODEL,
    "decision_base_url": DEFAULT_BASE_URL,
    "decision_path": DEFAULT_PATH,
    "timeout_seconds": 15.0,
    "compression_blocks_per_call": 16,
    "compression_max_blocks": 48,
    "compression_output_budget_chars": 6000,
    "compression_min_block_chars": 120,
}

_cache: dict[str, Any] | None = None


def _raw_settings() -> dict[str, Any]:
    """The plugin's ``settings`` mapping from the live Hermes config, or ``{}``."""
    global _cache
    if _cache is not None:
        return _cache
    raw: dict[str, Any] = {}
    try:
        from hermes_cli.config import load_config

        config = load_config() or {}
        entries = config.get("plugins", {}).get("entries", {})
        if isinstance(entries, dict):
            entry = entries.get(PLUGIN_ID) or entries.get(PROVIDER_NAME) or {}
            settings = entry.get("settings") if isinstance(entry, dict) else None
            if isinstance(settings, dict):
                raw = settings
    except Exception:
        raw = {}
    _cache = raw
    return raw


def reset_cache() -> None:
    """Drop the cached settings (tests, and after an operator config change)."""
    global _cache
    _cache = None


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def tasks() -> tuple[str, ...]:
    """Auxiliary task keys this plugin is allowed to serve."""
    value = _raw_settings().get("tasks", _DEFAULTS["tasks"])
    if isinstance(value, str):
        value = [part.strip() for part in value.split(",")]
    if not isinstance(value, (list, tuple)):
        return tuple(_DEFAULTS["tasks"])
    return tuple(str(item).strip() for item in value if str(item).strip())


def decision_model() -> str:
    return str(_raw_settings().get("decision_model") or _DEFAULTS["decision_model"]).strip()


def decision_base_url() -> str:
    return str(_raw_settings().get("decision_base_url") or _DEFAULTS["decision_base_url"]).strip().rstrip("/")


def decision_path() -> str:
    path = str(_raw_settings().get("decision_path") or _DEFAULTS["decision_path"]).strip()
    return path if path.startswith("/") else "/" + path


def timeout_seconds() -> float:
    return min(120.0, max(1.0, _as_float(_raw_settings().get("timeout_seconds"), _DEFAULTS["timeout_seconds"])))


def compression_blocks_per_call() -> int:
    return min(16, max(2, _as_int(_raw_settings().get("compression_blocks_per_call"), _DEFAULTS["compression_blocks_per_call"])))


def compression_max_blocks() -> int:
    return min(96, max(2, _as_int(_raw_settings().get("compression_max_blocks"), _DEFAULTS["compression_max_blocks"])))


def compression_output_budget_chars() -> int:
    return max(400, _as_int(_raw_settings().get("compression_output_budget_chars"), _DEFAULTS["compression_output_budget_chars"]))


def compression_min_block_chars() -> int:
    return max(1, _as_int(_raw_settings().get("compression_min_block_chars"), _DEFAULTS["compression_min_block_chars"]))


def _dotenv_value(name: str) -> str:
    """Read ``name`` from the active profile's ``.env`` without importing Hermes internals."""
    try:
        from hermes_constants import get_hermes_home

        candidates = [get_hermes_home() / ".env"]
    except Exception:
        candidates = []
    candidates.append(Path.home() / ".hermes" / ".env")
    for path in candidates:
        try:
            if not path.is_file():
                continue
            for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
                stripped = line.strip()
                if not stripped or stripped.startswith("#") or "=" not in stripped:
                    continue
                key, _, value = stripped.partition("=")
                if key.strip() == name:
                    return value.strip().strip("'\"")
        except Exception:
            continue
    return ""


def api_key() -> str:
    """The OpenRouter credential, from the process environment then the profile ``.env``."""
    return (os.getenv(CREDENTIAL_ENV_VAR) or "").strip() or _dotenv_value(CREDENTIAL_ENV_VAR)
