"""Bounded decision transport for the OpenRouter ``/api/alpha/decisions`` endpoint.

The endpoint takes one shared ``state`` and a map of typed questions and returns a
typed answer per question. It is not a chat endpoint: it cannot generate prose, and it
cannot answer a question whose outcome set is not bounded in advance. Everything in
this plugin is built around that constraint.

Dependency-free by design — ``urllib`` only — so the plugin installs without pulling
anything into the Hermes runtime. Tests inject a fake transport.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable

from . import config, privacy

USER_AGENT = "hermes-structured-structure-aux-models/0.1.0"

# (url, headers, body, timeout) -> (status, body_bytes, response_headers)
Transport = Callable[[str, dict[str, str], bytes, float], tuple]


class DecisionError(RuntimeError):
    """The decision provider could not produce a usable answer."""


class UnsupportedRequest(RuntimeError):
    """The request cannot be expressed as a bounded decision.

    Raised deliberately so Hermes' auxiliary fallback takes over. The shim must
    never substitute a plausible-looking answer for a decision it cannot express.
    """


@dataclass(frozen=True)
class DecisionResult:
    """One decision-provider response, already validated to contain every answer."""

    answers: dict[str, dict[str, Any]]
    model: str
    usage: dict[str, Any]
    request_id: str
    latency_ms: float
    transport: str = "openrouter-decisions"

    def answer(self, question: str) -> dict[str, Any]:
        value = self.answers.get(question)
        if not isinstance(value, dict):
            raise DecisionError(f"decision provider returned no answer for {question!r}")
        return value

    def label(self, question: str, allowed: tuple[str, ...]) -> str:
        """The chosen label for a ``choice`` question, validated against ``allowed``."""
        value = str(self.answer(question).get("choice") or "").strip()
        if value not in allowed:
            raise DecisionError(
                f"decision provider returned out-of-contract choice {value!r} for {question!r}"
            )
        return value

    def probabilities(self, question: str) -> dict[str, float]:
        raw = self.answer(question).get("probabilities") or {}
        result: dict[str, float] = {}
        if isinstance(raw, dict):
            for key, value in raw.items():
                try:
                    result[str(key)] = float(value)
                except (TypeError, ValueError):
                    continue
        return result

    def confidence(self, question: str) -> float:
        answer = self.answer(question)
        raw = answer.get("confidence")
        if raw is None:
            chosen = str(answer.get("choice") or "")
            return self.probabilities(question).get(chosen, 0.0)
        try:
            return float(raw)
        except (TypeError, ValueError):
            return 0.0


def _urllib_transport(url: str, headers: dict[str, str], body: bytes, timeout: float) -> tuple:
    """POST ``body`` to ``url``. HTTPS is enforced by the caller and re-checked here."""
    if not url.startswith("https://"):
        raise DecisionError("decision transport refuses a non-HTTPS URL")
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - https enforced above
            return int(response.status), response.read(), dict(response.headers.items())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:500]
        raise DecisionError(f"decision provider returned HTTP {exc.code}: {detail}") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise DecisionError(f"decision provider connection failed: {exc}") from exc


class DecisionClient:
    """One configured decision endpoint."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str | None = None,
        path: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
        transport: Transport | None = None,
    ) -> None:
        self.api_key = (api_key or "").strip()
        self.base_url = (base_url or config.decision_base_url()).rstrip("/")
        self.path = path or config.decision_path()
        self.model = model or config.decision_model()
        self.timeout = float(timeout if timeout is not None else config.timeout_seconds())
        self._transport: Transport = transport or _urllib_transport

    @property
    def url(self) -> str:
        return self.base_url + self.path

    def ask(
        self,
        *,
        state: Any,
        questions: dict[str, dict[str, Any]],
        model: str | None = None,
    ) -> DecisionResult:
        """Send one bounded decision request and return its validated answers."""
        if not self.api_key:
            raise DecisionError("no OpenRouter credential is configured")
        if not self.base_url.startswith("https://"):
            raise DecisionError("decision base_url must use https://")
        if not questions:
            raise DecisionError("at least one question is required")

        payload = {
            "state": privacy.redact_state(state),
            "model": model or self.model,
            "questions": privacy.redact_state(questions),
        }
        body = json.dumps(payload, separators=(",", ":"), ensure_ascii=False, default=str).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": USER_AGENT,
        }

        started = time.monotonic()
        raw_result = self._transport(self.url, headers, body, self.timeout)
        latency_ms = (time.monotonic() - started) * 1000.0

        if not isinstance(raw_result, tuple) or len(raw_result) < 2:
            raise DecisionError("decision transport returned an invalid response tuple")
        status, raw = int(raw_result[0]), raw_result[1]
        if status < 200 or status >= 300:
            detail = bytes(raw).decode("utf-8", "replace")[:500] if raw else ""
            raise DecisionError(f"decision provider returned HTTP {status}: {detail}")

        try:
            data = json.loads(raw)
        except (TypeError, ValueError) as exc:
            raise DecisionError("decision provider returned invalid JSON") from exc
        if not isinstance(data, dict):
            raise DecisionError("decision provider returned a non-object response")

        answers = data.get("answers")
        if not isinstance(answers, dict):
            raise DecisionError("decision provider response is missing 'answers'")
        missing = [name for name in questions if name not in answers]
        if missing:
            raise DecisionError(
                "decision provider response is missing answers for: " + ", ".join(sorted(missing))
            )

        usage = data.get("usage")
        return DecisionResult(
            answers={str(k): v for k, v in answers.items() if isinstance(v, dict)},
            model=str(data.get("model") or model or self.model),
            usage=usage if isinstance(usage, dict) else {},
            request_id=str(data.get("id") or ""),
            latency_ms=latency_ms,
        )
