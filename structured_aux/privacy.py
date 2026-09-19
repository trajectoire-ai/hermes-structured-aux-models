"""Redaction applied to every payload before it leaves the process.

Decision state is built from prompts that may carry tool output, file contents, or
operator policy text. Hermes ships a redactor; reuse it when available and fall back
to a conservative local pattern set so this plugin never depends on a private import
to avoid leaking a credential.
"""

from __future__ import annotations

import re

# Conservative shapes only: an obvious credential assignment or a known token prefix.
# Deliberately narrow — over-redacting would strip the command text the approval
# contract exists to assess.
_ASSIGNMENT_RE = re.compile(
    r"(?i)\b([A-Z0-9_]*(?:API_?KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL)[A-Z0-9_]*)"
    r"\s*[:=]\s*(\"[^\"]*\"|'[^']*'|\S+)"
)
_BEARER_RE = re.compile(r"(?i)\b(bearer)\s+([A-Za-z0-9._\-]{12,})")
_KNOWN_PREFIX_RE = re.compile(r"\b(sk-[A-Za-z0-9\-_]{12,}|gh[pousr]_[A-Za-z0-9]{20,}|ghs_[A-Za-z0-9]{20,})\b")

_REDACTED = "[REDACTED]"


def _local_redact(text: str) -> str:
    text = _ASSIGNMENT_RE.sub(lambda m: f"{m.group(1)}={_REDACTED}", text)
    text = _BEARER_RE.sub(lambda m: f"{m.group(1)} {_REDACTED}", text)
    text = _KNOWN_PREFIX_RE.sub(_REDACTED, text)
    return text


def redact_text(value: object) -> str:
    """Redact ``value`` rendered as text. Never raises."""
    text = value if isinstance(value, str) else str(value)
    try:
        from agent.redact import redact_sensitive_text

        return redact_sensitive_text(text, force=True)
    except Exception:
        return _local_redact(text)


def redact_state(state: object) -> object:
    """Recursively redact string leaves of a JSON-compatible structure. Never raises."""
    if isinstance(state, str):
        return redact_text(state)
    if isinstance(state, dict):
        return {str(key): redact_state(value) for key, value in state.items()}
    if isinstance(state, (list, tuple)):
        return [redact_state(item) for item in state]
    return state
