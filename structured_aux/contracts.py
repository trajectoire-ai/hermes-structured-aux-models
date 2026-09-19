"""Decision contracts: which auxiliary task a request belongs to, and what to ask.

Each contract bounds the outcome set *before* the provider is called. That bound is
the whole point — a decision model is only trustworthy when every possible answer is
enumerated by the caller, so the plugin never lets the provider choose outside it.
"""

from __future__ import annotations

from typing import Any

from . import config

TASK_APPROVAL = "approval"
TASK_MCP = "mcp"
TASK_COMPRESSION = "compression"

# ── approval ────────────────────────────────────────────────────────────────────
#
# Hermes' smart-approval guardian asks for exactly one word back and maps anything it
# does not recognise to "escalate". Three labels, not two: ESCALATE is how Hermes
# expresses "a human must decide", and collapsing it into DENY would silently turn an
# unknown-risk command into a refusal.

APPROVAL_LABELS: tuple[str, ...] = ("APPROVE", "DENY", "ESCALATE")
APPROVAL_CRITERIA: dict[str, str] = {
    "APPROVE": (
        "The operations are safe to run unattended: read-only, idempotent, or their "
        "effects are confined and clearly intended."
    ),
    "DENY": (
        "The operations are genuinely destructive, exfiltrating, or otherwise "
        "harmful, and must not run."
    ),
    "ESCALATE": (
        "The risk cannot be determined from the supplied command and description; a "
        "human must decide."
    ),
}
APPROVAL_INSTRUCTIONS = (
    "Assess the ACTUAL risk of the shell operations in the request. Many flagged "
    "commands are false positives. Choose APPROVE only when the operations are clearly "
    "safe, DENY only when they are clearly harmful, and ESCALATE whenever the supplied "
    "evidence is insufficient to be confident."
)


def approval_questions() -> dict[str, dict[str, Any]]:
    """The single bounded question the approval contract asks."""
    return {
        "verdict": {
            "type": "choice",
            "instructions": APPROVAL_INSTRUCTIONS,
            "criteria": dict(APPROVAL_CRITERIA),
        }
    }


# ── mcp ─────────────────────────────────────────────────────────────────────────


def mcp_tool_questions(candidates: dict[str, str]) -> dict[str, dict[str, Any]]:
    """A bounded choice over MCP tool names. Requires at least two candidates."""
    return {
        "tool": {
            "type": "choice",
            "instructions": (
                "Select the single tool that best satisfies the request. Choose only "
                "from the supplied candidates."
            ),
            "criteria": dict(candidates),
        }
    }


# ── compression ─────────────────────────────────────────────────────────────────

COMPRESSION_LABELS: tuple[str, ...] = ("KEEP", "DROP")
COMPRESSION_CRITERIA: dict[str, str] = {
    "KEEP": (
        "This block carries constraints, decisions, exact failure evidence, unresolved "
        "state, or identifiers that later work still depends on."
    ),
    "DROP": (
        "This block is stale, superseded, routine, or recoverable by rerunning the work, "
        "and nothing later depends on its exact text."
    ),
}
COMPRESSION_INSTRUCTIONS = (
    "Decide whether this block of working context must survive into the retained set. "
    "Bias toward KEEP when the block holds an exact value, an open question, or evidence "
    "of a failure; bias toward DROP only for clearly redundant or recoverable material."
)


def compression_questions(block_names: list[str]) -> dict[str, dict[str, Any]]:
    """One bounded KEEP/DROP question per candidate block."""
    return {
        name: {
            "type": "choice",
            "instructions": COMPRESSION_INSTRUCTIONS,
            "criteria": dict(COMPRESSION_CRITERIA),
        }
        for name in block_names
    }


# ── task detection ──────────────────────────────────────────────────────────────
#
# Primary signal is the model id the operator wrote into auxiliary.<task>.model, e.g.
# "structured-aux/approval". Hermes passes that string through untouched for a provider
# it does not recognise, so it is a stable channel. Content sniffing is a narrow
# fallback only for the one prompt whose wording is unmistakable.

_APPROVAL_MARKER = "approve, deny, or escalate"
_SNUFFABLE = (TASK_APPROVAL,)


def _model_task(model_id: str, allowed: tuple[str, ...]) -> str | None:
    name = (model_id or "").strip().lower()
    if not name:
        return None
    for separator in ("/", ":", "."):
        if separator in name:
            tail = name.rsplit(separator, 1)[-1].strip()
            if tail in allowed:
                return tail
    for separator in ("-", "_"):
        for task in allowed:
            if name.endswith(separator + task):
                return task
    return name if name in allowed else None


def _messages_text(messages: Any) -> str:
    parts: list[str] = []
    if isinstance(messages, (list, tuple)):
        for message in messages:
            if isinstance(message, dict):
                content = message.get("content")
                if isinstance(content, str):
                    parts.append(content)
    elif isinstance(messages, str):
        parts.append(messages)
    return "\n".join(parts).lower()


def _sniff_task(messages: Any, allowed: tuple[str, ...]) -> str | None:
    text = _messages_text(messages)
    if not text:
        return None
    for task in _SNUFFABLE:
        if task not in allowed:
            continue
        if task == TASK_APPROVAL and _APPROVAL_MARKER in text:
            return task
    return None


def detect_task(model_id: str, messages: Any) -> str | None:
    """Which contract applies, or ``None`` when this request must not be served.

    Returning ``None`` is the safe outcome: the caller raises and Hermes falls back to
    the operator's real auxiliary provider.
    """
    allowed = config.tasks()
    return _model_task(model_id, allowed) or _sniff_task(messages, allowed)
