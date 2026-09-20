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
import logging
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable

from . import config, privacy

logger = logging.getLogger("structured_aux.decisions")

USER_AGENT = "hermes-structured-aux-models/0.1.0"

# (url, headers, body, timeout) -> (status, body_bytes, response_headers)
Transport = Callable[[str, dict[str, str], bytes, float], tuple]

# A decision call that failed for one of these reasons is worth trying again: the request
# never reached a verdict, and the same payload has a real chance of succeeding. Statuses
# that name a payload or credential problem are deliberately absent — retrying a 400/401/422
# only repeats a request the provider has already judged wrong.
RETRYABLE_STATUSES = frozenset({408, 425, 429, 500, 502, 503, 504, 520, 522, 524})


class DecisionError(RuntimeError):
    """The decision provider could not produce a usable answer.

    ``retryable`` marks a transient transport failure (connection error, read timeout, or a
    retryable HTTP status) that a later attempt may still turn into a verdict. ``status``
    carries the HTTP status when there was one, for the log line.
    """

    def __init__(self, message: str, *, retryable: bool = False, status: int | None = None) -> None:
        super().__init__(message)
        self.retryable = retryable
        self.status = status


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
    # urllib title-cases every header name it is handed, which would put
    # `Http-referer` / `X-openrouter-title` on the wire. Headers are case-insensitive,
    # but send OpenRouter's attribution headers with the spelling its docs use so the
    # request is recognisable in a capture or a support thread.
    request.headers = {name: value for name, value in headers.items()}
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - https enforced above
            return int(response.status), response.read(), dict(response.headers.items())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:500]
        raise DecisionError(
            f"decision provider returned HTTP {exc.code}: {detail}",
            retryable=exc.code in RETRYABLE_STATUSES,
            status=int(exc.code),
        ) from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise DecisionError(f"decision provider connection failed: {exc}", retryable=True) from exc


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
        """Send one bounded decision request and return its validated answers.

        A *transient* failure — a connection error, a read timeout, or a retryable HTTP
        status — is retried up to ``decision_max_attempts`` (3 by default) with a doubling
        backoff. That matters beyond politeness: Hermes treats an auxiliary compression call
        as critical-path work and, on a full-budget timeout, skips its own same-provider
        retry and falls back to the main model — which turns one stalled decision call into
        a prose summary and loses the extractive guarantee. Every attempt is logged, so a
        call that needed retrying (or one that never recovered) is visible in `errors.log`.
        """
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
            "HTTP-Referer": config.app_referer(),
            "X-OpenRouter-Title": config.app_title(),
        }

        resolved_model = model or self.model
        max_attempts = config.decision_max_attempts()
        attempt = 1
        while True:
            started = time.monotonic()
            try:
                raw_result = self._transport(self.url, headers, body, self.timeout)
                result = self._parse(
                    raw_result, questions, model or self.model,
                    latency_ms=(time.monotonic() - started) * 1000.0,
                )
            except DecisionError as exc:
                elapsed_ms = (time.monotonic() - started) * 1000.0
                if not exc.retryable or attempt >= max_attempts:
                    logger.error(
                        "decision call failed: attempt %d/%d after %.0f ms (model=%s, questions=%d, status=%s): %s",
                        attempt, max_attempts, elapsed_ms, resolved_model, len(questions),
                        exc.status if exc.status is not None else "-", exc,
                    )
                    raise
                delay = config.decision_retry_backoff_seconds() * (2 ** (attempt - 1))
                logger.warning(
                    "decision call failed transiently: attempt %d/%d after %.0f ms "
                    "(model=%s, questions=%d, status=%s), retrying in %.1f s: %s",
                    attempt, max_attempts, elapsed_ms, resolved_model, len(questions),
                    exc.status if exc.status is not None else "-", delay, exc,
                )
                time.sleep(delay)
                attempt += 1
                continue

            elapsed_ms = result.latency_ms
            if attempt > 1:
                logger.warning(
                    "decision call recovered on attempt %d/%d after %.0f ms (model=%s, questions=%d)",
                    attempt, max_attempts, elapsed_ms, resolved_model, len(questions),
                )
            elif elapsed_ms >= self.slow_call_ms:
                logger.warning(
                    "decision call slow: %.0f ms for attempt 1/%d (model=%s, questions=%d)",
                    elapsed_ms, max_attempts, resolved_model, len(questions),
                )
            else:
                logger.debug(
                    "decision call answered in %.0f ms on attempt %d/%d (model=%s, questions=%d)",
                    elapsed_ms, attempt, max_attempts, resolved_model, len(questions),
                )
            return result

    @property
    def slow_call_ms(self) -> float:
        """A call slower than half its timeout is worth a log line: it is the shape of stall
        that, at the full timeout, costs the whole compression its extractive path."""
        return max(1000.0, self.timeout * 500.0)

    def _parse(
        self, raw_result: tuple, questions: dict[str, dict[str, Any]], model: str, *, latency_ms: float,
    ) -> DecisionResult:
        """Validate one transport response into a ``DecisionResult``."""
        if not isinstance(raw_result, tuple) or len(raw_result) < 2:
            raise DecisionError("decision transport returned an invalid response tuple")
        status, raw = int(raw_result[0]), raw_result[1]
        if status < 200 or status >= 300:
            detail = bytes(raw).decode("utf-8", "replace")[:500] if raw else ""
            raise DecisionError(
                f"decision provider returned HTTP {status}: {detail}",
                retryable=status in RETRYABLE_STATUSES,
                status=status,
            )

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
