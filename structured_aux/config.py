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

PLUGIN_ID = "hermes-structured-aux-models"
PROVIDER_NAME = "structured-aux"

DEFAULT_BASE_URL = "https://openrouter.ai"
DEFAULT_PATH = "/api/alpha/decisions"
# OpenRouter app attribution (https://openrouter.ai/docs/app-attribution): HTTP-Referer
# is the primary app identifier (without it the app shows as "Unknown" and its usage is
# not attributed), X-OpenRouter-Title sets the display name. Both are plain, non-secret
# strings, so they are defaults rather than required configuration.
DEFAULT_APP_REFERER = "https://trajectoire.ai"
DEFAULT_APP_TITLE = "hermes-structured-aux"
# The `~` prefix marks a moving alias, so the default tracks the newest Jev release
# instead of pinning a dated version that has to be bumped by hand. Operators can pin
# an exact id with plugins.entries.<PLUGIN_ID>.settings.decision_model.
DEFAULT_MODEL = "~typesafe/jev-latest"
CREDENTIAL_ENV_VAR = "OPENROUTER_API_KEY"
# Hermes' own credential store key for OpenRouter, used as the last-resort credential
# source (see ``_pool_api_key``). A default, not a fixture: an install whose OpenRouter key
# is stored under a different provider name sets ``credential_pool_provider``.
DEFAULT_POOL_PROVIDER = "openrouter"

# Task -> the model id suffix an operator writes into auxiliary.<task>.model.
# The shim reads this back off the request to know which contract to apply.
MODEL_PREFIX = "structured-aux"

_DEFAULTS: dict[str, Any] = {
    "tasks": ("approval", "mcp", "compression"),
    "decision_model": DEFAULT_MODEL,
    "decision_base_url": DEFAULT_BASE_URL,
    "decision_path": DEFAULT_PATH,
    "credential_pool_provider": DEFAULT_POOL_PROVIDER,
    "app_referer": DEFAULT_APP_REFERER,
    "app_title": DEFAULT_APP_TITLE,
    "timeout_seconds": 15.0,
    "decision_max_attempts": 3,
    "decision_retry_backoff_seconds": 0.5,
    "compression_blocks_per_call": 16,
    "compression_max_blocks": 48,
    "compression_output_budget_chars": 18000,
    "compression_min_block_chars": 120,
    "compression_call_budget_tokens": 8000,
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


def app_referer() -> str:
    """The OpenRouter ``HTTP-Referer`` value: the app's primary identifier."""
    return str(_raw_settings().get("app_referer") or _DEFAULTS["app_referer"]).strip()


def app_title() -> str:
    """The OpenRouter ``X-OpenRouter-Title`` value: the app's display name."""
    return str(_raw_settings().get("app_title") or _DEFAULTS["app_title"]).strip()


def timeout_seconds() -> float:
    return min(120.0, max(1.0, _as_float(_raw_settings().get("timeout_seconds"), _DEFAULTS["timeout_seconds"])))


def decision_max_attempts() -> int:
    """Total attempts for one decision request, first try included.

    A transient failure gets another try because Hermes does not retry a critical-path
    auxiliary call itself — it hands compression to the main model instead.
    """
    return min(10, max(1, _as_int(_raw_settings().get("decision_max_attempts"), _DEFAULTS["decision_max_attempts"])))


def decision_retry_backoff_seconds() -> float:
    """Base delay before a retry; doubles per attempt (0.5 s, 1.0 s, ... for 3 attempts)."""
    return min(30.0, max(0.0, _as_float(
        _raw_settings().get("decision_retry_backoff_seconds"), _DEFAULTS["decision_retry_backoff_seconds"],
    )))


def compression_blocks_per_call() -> int:
    return min(16, max(2, _as_int(_raw_settings().get("compression_blocks_per_call"), _DEFAULTS["compression_blocks_per_call"])))


def compression_max_blocks() -> int:
    return min(96, max(2, _as_int(_raw_settings().get("compression_max_blocks"), _DEFAULTS["compression_max_blocks"])))


def compression_output_budget_chars() -> int:
    """Character ceiling for the assembled digest.

    The digest is the concatenation of the retained blocks, so its size is not bounded by
    one decision call's window — every call stays inside Jev's 32,000-token window and the
    digest sums however many chunks survived. The ceiling therefore only has to stay small
    enough to be cheaper than the turns it replaces, not small enough to fit a call.
    """
    return max(400, _as_int(_raw_settings().get("compression_output_budget_chars"), _DEFAULTS["compression_output_budget_chars"]))


def compression_min_block_chars() -> int:
    return max(1, _as_int(_raw_settings().get("compression_min_block_chars"), _DEFAULTS["compression_min_block_chars"]))


def compression_call_budget_tokens() -> int:
    """Estimated-token ceiling for one decision call.

    Jev's window is 32,000 tokens, so this is capped below it: a call that exceeded the
    window could not be answered at all, and the whole point of the budget is that a
    compression prompt of any size is served in however many calls fit.
    """
    value = _as_int(_raw_settings().get("compression_call_budget_tokens"), _DEFAULTS["compression_call_budget_tokens"])
    return min(28000, max(1000, value))


def credential_pool_provider() -> str:
    """Provider key whose Hermes credential-pool entry supplies the decision credential.

    Defaults to ``openrouter``, because that is where ``hermes auth add openrouter`` lands.
    Set ``plugins.entries.hermes-structured-aux-models.settings.credential_pool_provider``
    when the OpenRouter key is stored under a different provider name — the pool lookup is
    keyed on the provider, so a hardcoded name silently finds nothing and the plugin
    degrades to "no credential" on an install that legitimately holds one.

    A blank value is treated as unset rather than as a provider named "": an empty key
    would look like a deliberate misconfiguration and fail the lookup the same way.
    """
    value = str(_raw_settings().get("credential_pool_provider") or "").strip()
    return value or _DEFAULTS["credential_pool_provider"]


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


def _pool_api_key() -> str:
    """The OpenRouter key Hermes itself uses, from the active profile's credential pool.

    An operator who already ran ``hermes auth add openrouter`` has the credential in Hermes'
    store, and that is the same key the rest of Hermes bills; reading it here keeps one source
    of truth instead of duplicating the secret into ``.env``. The provider key is
    ``credential_pool_provider``. Wrapped so an unresolvable pool degrades to "no credential"
    — the decision client then raises and Hermes falls back to the operator's real auxiliary
    provider.
    """
    try:
        from agent.credential_pool import load_pool

        pool = load_pool(credential_pool_provider())
        if pool is None or not pool.has_credentials():
            return ""
        return str(getattr(pool.select(), "runtime_api_key", "") or "").strip()
    except Exception:
        return ""


def api_key() -> str:
    """The OpenRouter credential, from the process environment, the profile ``.env``, then
    Hermes' own credential pool for the provider."""
    return (
        (os.getenv(CREDENTIAL_ENV_VAR) or "").strip()
        or _dotenv_value(CREDENTIAL_ENV_VAR)
        or _pool_api_key()
    )
