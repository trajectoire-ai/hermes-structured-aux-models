"""Shared fixtures for the structured-aux suite.

Fully offline: every fixture builds a fake transport or a fake client, and no test makes
a network call. A live provider run requires explicit operator approval.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Make the plugin importable as a package without installing it.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from structured_aux.decisions import DecisionResult  # noqa: E402


@pytest.fixture
def choice():
    """Build a ``choice`` answer object in the provider's wire shape."""

    def build(label, probabilities=None, confidence=None):
        return {
            "choice": label,
            "probabilities": probabilities if probabilities is not None else {label: 0.9},
            "confidence": confidence if confidence is not None else 0.9,
        }

    return build


@pytest.fixture
def make_transport():
    """Factory for a fake decision transport returning one canned response."""

    def build(*, answers=None, status=200, body=None, model="~typesafe/jev-latest", usage=None, request_id="gen-test"):
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
            calls.append(
                {
                    "url": url,
                    "headers": headers,
                    "body": json.loads(body_bytes.decode("utf-8")),
                    "timeout": timeout,
                }
            )
            return status, body, {"content-type": "application/json"}

        transport.calls = calls
        return transport

    return build


@pytest.fixture
def dynamic_transport():
    """Factory for a fake transport whose answers are computed from the request body."""

    def build(responder):
        calls: list[dict] = []

        def transport(url, headers, body_bytes, timeout):
            body = json.loads(body_bytes.decode("utf-8"))
            calls.append({"url": url, "headers": headers, "body": body, "timeout": timeout})
            payload = {"answers": responder(body), "model": "~typesafe/jev-latest", "usage": {}, "id": "gen-1"}
            return 200, json.dumps(payload).encode("utf-8"), {}

        transport.calls = calls
        return transport

    return build


@pytest.fixture
def fake_client():
    """Factory for a fake decision client that records every ``ask`` call."""

    def build(responder):
        class _FakeDecisionClient:
            def __init__(self):
                self.calls: list[dict] = []

            def ask(self, *, state, questions, model=None):
                self.calls.append({"state": state, "questions": questions, "model": model})
                return DecisionResult(
                    answers=responder(state, questions),
                    model=model or "~typesafe/jev-latest",
                    usage={},
                    request_id="fake",
                    latency_ms=1.0,
                )

        return _FakeDecisionClient()

    return build


@pytest.fixture
def shim():
    """Factory for a ``StructuredAuxClient`` wired to a fake transport."""
    from structured_aux.shim import StructuredAuxClient

    def build(transport, **kwargs):
        return StructuredAuxClient(api_key="test-key", transport=transport, **kwargs)

    return build


@pytest.fixture
def approval_messages():
    """A realistic Hermes smart-approval request."""
    return [
        {"role": "system", "content": "You are a command-risk guardian."},
        {
            "role": "user",
            "content": (
                "The following command was flagged as: script execution via -c flag\n\n"
                "<command>\npython3 -c 'print(1)'\n</command>\n\n"
                "Respond with exactly one word: APPROVE, DENY, or ESCALATE"
            ),
        },
    ]
