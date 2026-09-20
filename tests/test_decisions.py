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


# ── retry and logging on failure ────────────────────────────────────────────────


def test_a_transient_failure_is_retried_and_the_answer_returned(client, flaky_transport, choice):
    transport = flaky_transport(failures=2, answers={"verdict": choice("APPROVE")})

    result = client(transport).ask(state={}, questions=QUESTIONS)

    assert result.label("verdict", ("APPROVE", "DENY")) == "APPROVE"
    assert len(transport.attempts) == 3


def test_a_retryable_status_is_retried(client, make_transport):
    # 503 is the provider being briefly unavailable, not a judgement on the request.
    transport = make_transport(status=503, body=b"unavailable")

    with pytest.raises(DecisionError):
        client(transport).ask(state={}, questions=QUESTIONS)

    assert len(transport.calls) == config.decision_max_attempts() == 3


def test_a_payload_or_credential_status_is_not_retried(client, make_transport):
    transport = make_transport(status=401, body=b"no key")

    with pytest.raises(DecisionError):
        client(transport).ask(state={}, questions=QUESTIONS)

    assert len(transport.calls) == 1


def test_a_contract_violation_is_not_retried(client, make_transport):
    # A 200 whose answers are missing or unparseable is not a transport blip: repeating it
    # only repeats the same contract failure.
    transport = make_transport(body=json.dumps({"model": "m"}).encode())

    with pytest.raises(DecisionError):
        client(transport).ask(state={}, questions=QUESTIONS)

    assert len(transport.calls) == 1


def test_attempts_stop_at_the_ceiling_and_the_last_error_is_raised(client, flaky_transport):
    transport = flaky_transport(failures=99)

    with pytest.raises(DecisionError):
        client(transport).ask(state={}, questions=QUESTIONS)

    assert len(transport.attempts) == config.decision_max_attempts() == 3


def test_retry_backoff_doubles_per_attempt(client, flaky_transport, monkeypatch):
    monkeypatch.setattr(config, "decision_retry_backoff_seconds", lambda: 0.5)
    slept: list[float] = []
    monkeypatch.setattr(decisions.time, "sleep", slept.append)
    transport = flaky_transport(failures=2)

    with pytest.raises(DecisionError):
        client(transport).ask(state={}, questions=QUESTIONS)  # third attempt is the last, so it raises

    assert slept == [0.5, 1.0]


def test_a_recovered_call_is_logged_at_warning(client, flaky_transport, choice, caplog):
    transport = flaky_transport(failures=1, answers={"verdict": choice("APPROVE")})

    with caplog.at_level("WARNING"):
        client(transport).ask(state={}, questions=QUESTIONS)

    messages = [record.getMessage() for record in caplog.records]
    assert any("attempt 1/3" in message and "retrying" in message for message in messages)
    assert any("recovered on attempt 2/3" in message for message in messages)


def test_the_final_failure_is_logged_at_error_with_its_attempt_context(client, flaky_transport, caplog):
    transport = flaky_transport(failures=99)

    with caplog.at_level("ERROR"), pytest.raises(DecisionError):
        client(transport).ask(state={}, questions=QUESTIONS)

    errors = [record.getMessage() for record in caplog.records if record.levelname == "ERROR"]
    assert any("attempt 3/3" in message and "questions=1" in message for message in errors)


def test_a_slow_successful_call_is_logged_at_warning(client, choice, monkeypatch, caplog):
    # The stall that costs a compression its extractive path starts as a slow call: half the
    # timeout is worth a line well before the full one.
    def slow_transport(url, headers, body_bytes, timeout):
        return 200, json.dumps({
            "answers": {"verdict": choice("APPROVE")}, "model": "m", "usage": {}, "id": "x",
        }).encode("utf-8"), {}

    ticks = {"n": 0}

    def fake_monotonic():
        ticks["n"] += 1
        return 0.0 if ticks["n"] == 1 else 8.0  # 8 s for a 5 s timeout: past the slow-call line

    monkeypatch.setattr(decisions.time, "monotonic", fake_monotonic)

    with caplog.at_level("WARNING"):
        client(slow_transport).ask(state={}, questions=QUESTIONS)

    assert any("decision call slow" in record.getMessage() for record in caplog.records)


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


def test_attribution_headers_are_sent(client, make_transport, choice):
    # OpenRouter attributes usage to an app by HTTP-Referer; without it the app shows as
    # "Unknown" and its usage is not listed. X-OpenRouter-Title names it.
    transport = make_transport(answers={"verdict": choice("APPROVE")})
    client(transport).ask(state={}, questions=QUESTIONS)

    headers = transport.calls[0]["headers"]
    assert headers["HTTP-Referer"] == "https://trajectoire.ai"
    assert headers["X-OpenRouter-Title"] == "hermes-structured-aux"


def test_attribution_headers_can_be_overridden(client, make_transport, choice, monkeypatch):
    monkeypatch.setattr(
        config,
        "_raw_settings",
        lambda: {"app_referer": "https://example.test/app", "app_title": "Example App"},
    )
    transport = make_transport(answers={"verdict": choice("APPROVE")})
    client(transport).ask(state={}, questions=QUESTIONS)

    headers = transport.calls[0]["headers"]
    assert headers["HTTP-Referer"] == "https://example.test/app"
    assert headers["X-OpenRouter-Title"] == "Example App"


def test_default_transport_keeps_documented_header_casing(monkeypatch):
    # urllib title-cases header names; the attribution headers must still go out with the
    # spelling OpenRouter documents.
    seen: dict[str, str] = {}

    class _Response:
        status = 200
        headers: dict[str, str] = {}

        def read(self) -> bytes:
            return b"{}"

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def fake_urlopen(request, timeout=None):
        seen.update(dict(request.header_items()))
        return _Response()

    monkeypatch.setattr(decisions.urllib.request, "urlopen", fake_urlopen)
    decisions._urllib_transport(
        "https://example.invalid/decisions",
        {"HTTP-Referer": "https://trajectoire.ai", "X-OpenRouter-Title": "hermes-structured-aux"},
        b"{}",
        1.0,
    )

    assert seen["HTTP-Referer"] == "https://trajectoire.ai"
    assert seen["X-OpenRouter-Title"] == "hermes-structured-aux"
