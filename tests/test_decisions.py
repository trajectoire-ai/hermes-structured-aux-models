"""Decision transport: request shape, response parsing, and failure behaviour."""

from __future__ import annotations

import json
import unittest

from helpers import choice, make_transport

from structured_aux import decisions
from structured_aux.decisions import DecisionClient, DecisionError

QUESTIONS = {"verdict": {"type": "choice", "instructions": "pick", "criteria": {"APPROVE": "a", "DENY": "b"}}}


def client(transport, **kwargs):
    return DecisionClient(
        api_key=kwargs.pop("api_key", "test-key"),
        base_url=kwargs.pop("base_url", "https://openrouter.ai"),
        model=kwargs.pop("model", "typesafe/jev-1.13"),
        timeout=kwargs.pop("timeout", 5.0),
        transport=transport,
        **kwargs,
    )


class RequestShapeTests(unittest.TestCase):
    def test_posts_state_model_and_questions(self):
        transport = make_transport(answers={"verdict": choice("APPROVE")})
        result = client(transport).ask(state={"request": "ls -la"}, questions=QUESTIONS)

        self.assertEqual(result.label("verdict", ("APPROVE", "DENY")), "APPROVE")
        call = transport.calls[0]
        self.assertEqual(call["url"], "https://openrouter.ai/api/alpha/decisions")
        self.assertEqual(call["body"]["model"], "typesafe/jev-1.13")
        self.assertEqual(call["body"]["state"], {"request": "ls -la"})
        self.assertEqual(call["body"]["questions"], QUESTIONS)
        self.assertEqual(call["headers"]["Authorization"], "Bearer test-key")

    def test_credentials_are_redacted_before_send(self):
        transport = make_transport(answers={"verdict": choice("APPROVE")})
        client(transport).ask(
            state={"request": "export API_KEY=supersecretvalue123"},
            questions=QUESTIONS,
        )
        sent = json.dumps(transport.calls[0]["body"])
        self.assertNotIn("supersecretvalue123", sent)
        self.assertIn("API_KEY=", sent)  # the assignment survives; only the value is masked

    def test_custom_model_overrides_configured_default(self):
        transport = make_transport(answers={"verdict": choice("APPROVE")})
        client(transport).ask(state={}, questions=QUESTIONS, model="typesafe/jev-1.14")
        self.assertEqual(transport.calls[0]["body"]["model"], "typesafe/jev-1.14")


class FailureBehaviourTests(unittest.TestCase):
    def test_missing_credential_raises(self):
        with self.assertRaises(DecisionError):
            client(make_transport(), api_key="").ask(state={}, questions=QUESTIONS)

    def test_non_https_base_url_raises(self):
        with self.assertRaises(DecisionError):
            client(make_transport(), base_url="http://openrouter.ai").ask(state={}, questions=QUESTIONS)

    def test_no_questions_raises(self):
        with self.assertRaises(DecisionError):
            client(make_transport()).ask(state={}, questions={})

    def test_http_error_raises(self):
        with self.assertRaises(DecisionError):
            client(make_transport(status=429, body=b"slow down")).ask(state={}, questions=QUESTIONS)

    def test_invalid_json_raises(self):
        with self.assertRaises(DecisionError):
            client(make_transport(body=b"not json")).ask(state={}, questions=QUESTIONS)

    def test_missing_answers_raises(self):
        with self.assertRaises(DecisionError):
            client(make_transport(body=json.dumps({"model": "m"}).encode())).ask(state={}, questions=QUESTIONS)

    def test_partial_answers_raise(self):
        transport = make_transport(answers={"other": choice("APPROVE")})
        with self.assertRaises(DecisionError):
            client(transport).ask(state={}, questions=QUESTIONS)

    def test_out_of_contract_label_raises(self):
        transport = make_transport(answers={"verdict": choice("MAYBE")})
        result = client(transport).ask(state={}, questions=QUESTIONS)
        with self.assertRaises(DecisionError):
            result.label("verdict", ("APPROVE", "DENY"))

    def test_transport_exception_is_wrapped(self):
        def boom(url, headers, body, timeout):
            raise OSError("connection reset")

        # The default transport wraps OSError; a custom one propagates, which Hermes'
        # auxiliary fallback then handles. Assert the default wrapping explicitly.
        with self.assertRaises(decisions.DecisionError):
            decisions._urllib_transport("https://example.invalid/x", {}, b"{}", 1.0)

    def test_non_https_url_refused_by_default_transport(self):
        with self.assertRaises(DecisionError):
            decisions._urllib_transport("http://example.invalid/x", {}, b"{}", 1.0)


class ResultAccessorTests(unittest.TestCase):
    def test_probabilities_and_confidence(self):
        transport = make_transport(answers={"verdict": choice("DENY", {"DENY": 0.7, "APPROVE": 0.3}, 0.7)})
        result = client(transport).ask(state={}, questions=QUESTIONS)
        self.assertAlmostEqual(result.confidence("verdict"), 0.7)
        self.assertEqual(result.probabilities("verdict"), {"DENY": 0.7, "APPROVE": 0.3})

    def test_confidence_falls_back_to_chosen_probability(self):
        transport = make_transport(answers={"verdict": {"choice": "DENY", "probabilities": {"DENY": 0.42}}})
        result = client(transport).ask(state={}, questions=QUESTIONS)
        self.assertAlmostEqual(result.confidence("verdict"), 0.42)


if __name__ == "__main__":
    unittest.main()
