"""Decision transport: request shape, response parsing, and failure behaviour."""

from __future__ import annotations

import json

import pytest

from structured_aux import config, decisions
from structured_aux.decisions import DecisionClient, DecisionError

QUESTIONS = {"verdict": {"type": "choice", "instructions": "pick", "criteria": {"APPROVE": "a", "DENY": "b"}}}


@pytest.fixture
def client():
    """Factory for a DecisionClient with an injected transport."""

    def build(transport, *, api_key="test-key", base_url="https://openrouter.ai", model=None, timeout=5.0):
        return DecisionClient(api_key=api_key, base_url=base_url, model=model, timeout=timeout, transport=transport)

    return build


def test_posts_state_model_and_questions(client, make_transport, choice):
    transport = make_transport(answers={"verdict": choice("APPROVE")})
    result = client(transport).ask(state={"request": "ls -la"}, questions=QUESTIONS)

    assert result.label("verdict", ("APPROVE", "DENY")) == "APPROVE"
    call = transport.calls[0]
    assert call["url"] == "https://openrouter.ai/api/alpha/decisions"
    assert call["body"]["state"] == {"request": "ls -la"}
    assert call["body"]["questions"] == QUESTIONS
    assert call["headers"]["Authorization"] == "Bearer test-key"


def test_default_model_is_the_moving_alias(client, make_transport, choice):
    # The default must track the newest Jev release rather than pin a dated version,
    # so operators are not forced to bump a model id in config.
    transport = make_transport(answers={"verdict": choice("APPROVE")})
    client(transport).ask(state={}, questions=QUESTIONS)

    assert config.DEFAULT_MODEL == "~typesafe/jev-latest"
    assert transport.calls[0]["body"]["model"] == "~typesafe/jev-latest"


def test_explicit_model_still_overrides_the_default(client, make_transport, choice):
    transport = make_transport(answers={"verdict": choice("APPROVE")})
    client(transport, model="~typesafe/jev-1.13").ask(state={}, questions=QUESTIONS)
    assert transport.calls[0]["body"]["model"] == "~typesafe/jev-1.13"


def test_credentials_are_redacted_before_send(client, make_transport, choice):
    transport = make_transport(answers={"verdict": choice("APPROVE")})
    client(transport).ask(state={"request": "export API_KEY=supersecretvalue123"}, questions=QUESTIONS)

    sent = json.dumps(transport.calls[0]["body"])
    assert "supersecretvalue123" not in sent
    assert "API_KEY=" in sent  # the assignment survives; only the value is masked


def test_missing_credential_raises(client, make_transport):
    with pytest.raises(DecisionError):
        client(make_transport(), api_key="").ask(state={}, questions=QUESTIONS)


def test_non_https_base_url_raises(client, make_transport):
    with pytest.raises(DecisionError):
        client(make_transport(), base_url="http://openrouter.ai").ask(state={}, questions=QUESTIONS)


def test_no_questions_raises(client, make_transport):
    with pytest.raises(DecisionError):
        client(make_transport()).ask(state={}, questions={})


def test_http_error_raises(client, make_transport):
    with pytest.raises(DecisionError):
        client(make_transport(status=429, body=b"slow down")).ask(state={}, questions=QUESTIONS)


def test_invalid_json_raises(client, make_transport):
    with pytest.raises(DecisionError):
        client(make_transport(body=b"not json")).ask(state={}, questions=QUESTIONS)


def test_missing_answers_raises(client, make_transport):
    with pytest.raises(DecisionError):
        client(make_transport(body=json.dumps({"model": "m"}).encode())).ask(state={}, questions=QUESTIONS)


def test_partial_answers_raise(client, make_transport, choice):
    transport = make_transport(answers={"other": choice("APPROVE")})
    with pytest.raises(DecisionError):
        client(transport).ask(state={}, questions=QUESTIONS)


def test_out_of_contract_label_raises(client, make_transport, choice):
    transport = make_transport(answers={"verdict": choice("MAYBE")})
    result = client(transport).ask(state={}, questions=QUESTIONS)
    with pytest.raises(DecisionError):
        result.label("verdict", ("APPROVE", "DENY"))


def test_default_transport_wraps_connection_errors():
    with pytest.raises(decisions.DecisionError):
        decisions._urllib_transport("https://example.invalid/x", {}, b"{}", 1.0)


def test_default_transport_refuses_non_https():
    with pytest.raises(DecisionError):
        decisions._urllib_transport("http://example.invalid/x", {}, b"{}", 1.0)


def test_probabilities_and_confidence(client, make_transport, choice):
    transport = make_transport(answers={"verdict": choice("DENY", {"DENY": 0.7, "APPROVE": 0.3}, 0.7)})
    result = client(transport).ask(state={}, questions=QUESTIONS)
    assert result.confidence("verdict") == pytest.approx(0.7)
    assert result.probabilities("verdict") == {"DENY": 0.7, "APPROVE": 0.3}


def test_confidence_falls_back_to_chosen_probability(client, make_transport):
    transport = make_transport(answers={"verdict": {"choice": "DENY", "probabilities": {"DENY": 0.42}}})
    result = client(transport).ask(state={}, questions=QUESTIONS)
    assert result.confidence("verdict") == pytest.approx(0.42)
