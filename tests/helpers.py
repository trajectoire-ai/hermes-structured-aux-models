"""Shared test helpers. No network access anywhere in this suite."""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Make the plugin importable as a package without installing it.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from structured_aux.decisions import DecisionResult  # noqa: E402


def make_transport(*, answers=None, status=200, body=None, model="typesafe/jev-1.13", usage=None, request_id="gen-test"):
    """A fake decision transport returning one canned response."""
    if body is None:
        payload = {
            "answers": answers if answers is not None else {},
            "model": model,
            "usage": usage if usage is not None else {"input_tokens": 11, "output_tokens": 2, "total_tokens": 13},
            "id": request_id,
        }
        body = json.dumps(payload).encode("utf-8")

    calls: list[dict] = []

    def transport(url, headers, body_bytes, timeout):
        calls.append({"url": url, "headers": headers, "body": json.loads(body_bytes.decode("utf-8")), "timeout": timeout})
        return status, body, {"content-type": "application/json"}

    transport.calls = calls  # type: ignore[attr-defined]
    return transport


def choice(label: str, probabilities=None, confidence=None):
    """A ``choice`` answer object in the provider's wire shape."""
    return {
        "choice": label,
        "probabilities": probabilities if probabilities is not None else {label: 0.9},
        "confidence": confidence if confidence is not None else 0.9,
    }


class FakeDecisionClient:
    """Records every ``ask`` and answers from a supplied mapping.

    ``responder`` receives ``(state, questions)`` and returns ``{question_name: answer}``.
    """

    def __init__(self, responder):
        self._responder = responder
        self.calls: list[dict] = []

    def ask(self, *, state, questions, model=None):
        self.calls.append({"state": state, "questions": questions, "model": model})
        answers = self._responder(state, questions)
        return DecisionResult(
            answers=answers,
            model=model or "typesafe/jev-1.13",
            usage={},
            request_id="fake",
            latency_ms=1.0,
        )
