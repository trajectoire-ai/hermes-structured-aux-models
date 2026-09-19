"""OpenAI-shaped client shim.

Hermes' auxiliary client resolves a provider to a client object and calls
``client.chat.completions.create(**kwargs)``. This shim implements exactly that surface
and nothing else. It inspects each request, decides whether the request can be expressed
as a bounded decision, and either answers it from the decision provider or raises
``UnsupportedRequest`` so Hermes falls back to the operator's real auxiliary provider.

``HERMES_SKIP_TRANSPORT_WRAP`` and ``HERMES_SKIP_ASYNC_WRAP`` are read by
``agent.auxiliary_client._client_declares`` and keep this client out of the Anthropic /
Codex / Bedrock wire adapters, which would otherwise re-dispatch it over HTTP.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from . import compression, config, contracts
from .decisions import DecisionClient, UnsupportedRequest

__all__ = ["StructuredAuxClient", "UnsupportedRequest"]


# ── Minimal OpenAI-shaped response objects ──────────────────────────────────────
#
# Only the attributes Hermes' auxiliary call sites actually read. Building these by
# hand keeps the plugin free of any OpenAI SDK dependency.


@dataclass
class _FunctionCall:
    name: str
    arguments: str = "{}"


@dataclass
class _ToolCall:
    id: str
    type: str = "function"
    function: _FunctionCall = field(default_factory=lambda: _FunctionCall(name=""))


@dataclass
class _Message:
    role: str = "assistant"
    content: Any = None
    tool_calls: list = field(default_factory=list)


@dataclass
class _Choice:
    message: _Message
    finish_reason: str = "stop"
    index: int = 0


@dataclass
class _Response:
    choices: list
    model: str = ""
    usage: dict = field(default_factory=dict)
    id: str = ""
    object: str = "chat.completion"
    created: int = 0


def _usage(raw: Any) -> dict:
    """Normalise decision-provider usage into the OpenAI token-count keys."""
    if not isinstance(raw, dict) or not raw:
        return {}
    prompt = raw.get("prompt_tokens", raw.get("input_tokens"))
    completion = raw.get("completion_tokens", raw.get("output_tokens"))
    total = raw.get("total_tokens")
    if total is None and isinstance(prompt, int) and isinstance(completion, int):
        total = prompt + completion
    result: dict[str, Any] = {}
    if prompt is not None:
        result["prompt_tokens"] = prompt
    if completion is not None:
        result["completion_tokens"] = completion
    if total is not None:
        result["total_tokens"] = total
    return result


def _split_messages(messages: Any) -> tuple[str, str]:
    """Return ``(system_text, user_text)`` joined from string-content messages."""
    system: list[str] = []
    user: list[str] = []
    if isinstance(messages, (list, tuple)):
        for message in messages:
            if not isinstance(message, dict):
                continue
            content = message.get("content")
            if not isinstance(content, str) or not content:
                continue
            role = str(message.get("role") or "")
            if role == "system":
                system.append(content)
            elif role == "user":
                user.append(content)
    return "\n\n".join(system), "\n\n".join(user)


def _tool_candidates(tools: Any) -> dict[str, str]:
    """Argument-free MCP tool candidates as ``{name: description}``.

    A decision returns a label, never synthesised arguments, so a tool that requires
    parameters is not answerable by this provider and is excluded rather than guessed.
    """
    candidates: dict[str, str] = {}
    if not isinstance(tools, (list, tuple)):
        return candidates
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        inner = tool.get("function")
        function: dict = inner if isinstance(inner, dict) else tool
        name = str(function.get("name") or "").strip()
        if not name:
            continue
        parameters = function.get("parameters")
        if isinstance(parameters, dict) and parameters.get("required"):
            continue
        candidates[name] = str(function.get("description") or "").strip() or f"Tool {name}"
    return candidates


class _Completions:
    """``client.chat.completions`` — the only call surface Hermes' aux path uses."""

    def __init__(self, owner: "StructuredAuxClient") -> None:
        self._owner = owner

    def create(self, **kwargs: Any) -> _Response:
        return self._owner.create(**kwargs)


class _Chat:
    def __init__(self, owner: "StructuredAuxClient") -> None:
        self.completions = _Completions(owner)


class StructuredAuxClient:
    """Decision-backed stand-in for an OpenAI chat client."""

    HERMES_SKIP_TRANSPORT_WRAP = True
    HERMES_SKIP_ASYNC_WRAP = True

    def __init__(
        self,
        *,
        api_key: str = "",
        base_url: str = "",
        transport: Any = None,
        decision_model: str | None = None,
    ) -> None:
        self.api_key = api_key or config.api_key()
        self.base_url = base_url or config.decision_base_url()
        self.chat = _Chat(self)
        self._transport = transport
        self._decision_model = decision_model

    # -- internals ---------------------------------------------------------------

    def _decision_client(self) -> DecisionClient:
        return DecisionClient(
            api_key=self.api_key,
            base_url=self.base_url,
            model=self._decision_model,
            transport=self._transport,
        )

    def _response(self, *, content: Any, model: str, usage: dict, request_id: str = "", tool_calls: list | None = None) -> _Response:
        return _Response(
            choices=[
                _Choice(
                    message=_Message(content=content, tool_calls=tool_calls or []),
                    finish_reason="tool_calls" if tool_calls else "stop",
                )
            ],
            model=model,
            usage=_usage(usage),
            id=request_id,
            created=int(time.time()),
        )

    # -- task handlers -----------------------------------------------------------

    def _approval(self, messages: Any, model: str) -> _Response:
        system_text, user_text = _split_messages(messages)
        if not user_text.strip():
            raise UnsupportedRequest("approval request carried no command text")
        result = self._decision_client().ask(
            state={"request": user_text, "guardian_policy": system_text},
            questions=contracts.approval_questions(),
            model=model or None,
        )
        verdict = result.label("verdict", contracts.APPROVAL_LABELS)
        return self._response(
            content=verdict, model=result.model, usage=result.usage, request_id=result.request_id
        )

    def _mcp(self, messages: Any, tools: Any, model: str) -> _Response:
        candidates = _tool_candidates(tools)
        if len(candidates) < 2:
            raise UnsupportedRequest(
                "MCP sampling request has fewer than two argument-free tools; "
                "a bounded choice needs at least two candidates"
            )
        _, user_text = _split_messages(messages)
        result = self._decision_client().ask(
            state={"request": user_text, "available_tools": candidates},
            questions=contracts.mcp_tool_questions(candidates),
            model=model or None,
        )
        chosen = result.label("tool", tuple(candidates))
        return self._response(
            content=None,
            model=result.model,
            usage=result.usage,
            request_id=result.request_id,
            tool_calls=[
                _ToolCall(id=f"call_{result.request_id or 'structured_aux'}", function=_FunctionCall(name=chosen))
            ],
        )

    def _compression(self, messages: Any, model: str) -> _Response:
        _, prompt = _split_messages(messages)
        if not prompt.strip():
            raise UnsupportedRequest("compression request carried no text")
        digest = compression.compress(self._decision_client(), prompt, model=model)
        return self._response(content=digest, model=model, usage={})

    # -- entry point -------------------------------------------------------------

    def create(self, **kwargs: Any) -> _Response:
        """Answer one auxiliary call, or raise to hand it back to Hermes."""
        if kwargs.get("stream"):
            raise UnsupportedRequest("structured-aux does not serve streaming requests")

        model = str(kwargs.get("model") or "")
        messages = kwargs.get("messages")
        task = contracts.detect_task(model, messages)
        if task is None:
            raise UnsupportedRequest(
                f"model {model!r} does not name a configured structured-aux task"
            )

        if task == contracts.TASK_APPROVAL:
            return self._approval(messages, model)
        if task == contracts.TASK_MCP:
            return self._mcp(messages, kwargs.get("tools"), model)
        if task == contracts.TASK_COMPRESSION:
            return self._compression(messages, model)
        raise UnsupportedRequest(f"no handler registered for task {task!r}")
