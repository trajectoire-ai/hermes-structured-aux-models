"""Shim behaviour: which requests are answered, and how."""

from __future__ import annotations

import json
import unittest

from helpers import choice, make_transport

from structured_aux.decisions import DecisionError, UnsupportedRequest
from structured_aux.shim import StructuredAuxClient

APPROVAL_MESSAGES = [
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


def dynamic_transport(responder):
    """Transport whose answers are computed from the request body."""
    calls = []

    def transport(url, headers, body_bytes, timeout):
        body = json.loads(body_bytes.decode("utf-8"))
        calls.append(body)
        payload = {"answers": responder(body), "model": "typesafe/jev-1.13", "usage": {}, "id": "gen-1"}
        return 200, json.dumps(payload).encode("utf-8"), {}

    transport.calls = calls
    return transport


def shim(transport, **kwargs):
    return StructuredAuxClient(api_key="test-key", transport=transport, **kwargs)


class ApprovalRoutingTests(unittest.TestCase):
    def test_approve_verdict_is_returned_verbatim(self):
        transport = make_transport(answers={"verdict": choice("APPROVE")})
        response = shim(transport).create(model="structured-aux/approval", messages=APPROVAL_MESSAGES)
        self.assertEqual(response.choices[0].message.content, "APPROVE")
        self.assertEqual(response.choices[0].finish_reason, "stop")

    def test_deny_and_escalate_survive(self):
        for label in ("DENY", "ESCALATE"):
            transport = make_transport(answers={"verdict": choice(label)})
            response = shim(transport).create(model="structured-aux/approval", messages=APPROVAL_MESSAGES)
            self.assertEqual(response.choices[0].message.content, label)

    def test_out_of_contract_verdict_raises(self):
        transport = make_transport(answers={"verdict": choice("PROBABLY_FINE")})
        with self.assertRaises(DecisionError):
            shim(transport).create(model="structured-aux/approval", messages=APPROVAL_MESSAGES)

    def test_guardian_policy_is_forwarded_as_state(self):
        transport = make_transport(answers={"verdict": choice("APPROVE")})
        shim(transport).create(model="structured-aux/approval", messages=APPROVAL_MESSAGES)
        state = transport.calls[0]["body"]["state"]
        self.assertEqual(state["guardian_policy"], "You are a command-risk guardian.")
        self.assertIn("python3 -c", state["request"])

    def test_empty_command_text_is_unsupported(self):
        transport = make_transport(answers={"verdict": choice("APPROVE")})
        with self.assertRaises(UnsupportedRequest):
            shim(transport).create(model="structured-aux/approval", messages=[{"role": "system", "content": "only"}])

    def test_detected_from_content_when_model_is_generic(self):
        transport = make_transport(answers={"verdict": choice("APPROVE")})
        response = shim(transport).create(model="", messages=APPROVAL_MESSAGES)
        self.assertEqual(response.choices[0].message.content, "APPROVE")


class UnsupportedRoutingTests(unittest.TestCase):
    def test_unknown_model_is_unsupported(self):
        with self.assertRaises(UnsupportedRequest):
            shim(make_transport()).create(model="gpt-4o", messages=[{"role": "user", "content": "hi"}])

    def test_no_model_and_no_marker_is_unsupported(self):
        with self.assertRaises(UnsupportedRequest):
            shim(make_transport()).create(model="", messages=[{"role": "user", "content": "summarise this"}])

    def test_streaming_is_unsupported(self):
        with self.assertRaises(UnsupportedRequest):
            shim(make_transport()).create(
                model="structured-aux/approval", messages=APPROVAL_MESSAGES, stream=True
            )

    def test_out_of_scope_task_is_unsupported(self):
        with self.assertRaises(UnsupportedRequest):
            shim(make_transport()).create(model="structured-aux/skills_hub", messages=[])


class McpRoutingTests(unittest.TestCase):
    TOOLS = [
        {"type": "function", "function": {"name": "read_file", "description": "Read a file", "parameters": {"type": "object", "properties": {}}}},
        {"type": "function", "function": {"name": "search", "description": "Search text", "parameters": {"type": "object", "properties": {}}}},
    ]

    def test_selects_a_tool(self):
        transport = make_transport(answers={"tool": choice("search")})
        response = shim(transport).create(
            model="structured-aux/mcp", messages=[{"role": "user", "content": "find the config"}], tools=self.TOOLS
        )
        call = response.choices[0].message.tool_calls[0]
        self.assertEqual(call.function.name, "search")
        self.assertEqual(response.choices[0].finish_reason, "tool_calls")

    def test_candidates_are_sent_as_criteria(self):
        transport = make_transport(answers={"tool": choice("read_file")})
        shim(transport).create(
            model="structured-aux/mcp", messages=[{"role": "user", "content": "x"}], tools=self.TOOLS
        )
        criteria = transport.calls[0]["body"]["questions"]["tool"]["criteria"]
        self.assertEqual(set(criteria), {"read_file", "search"})

    def test_tools_requiring_arguments_are_excluded(self):
        tools = list(self.TOOLS) + [
            {"type": "function", "function": {"name": "write_file", "description": "Write", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}}
        ]
        transport = make_transport(answers={"tool": choice("read_file")})
        shim(transport).create(model="structured-aux/mcp", messages=[{"role": "user", "content": "x"}], tools=tools)
        criteria = transport.calls[0]["body"]["questions"]["tool"]["criteria"]
        self.assertNotIn("write_file", criteria)

    def test_no_tools_is_unsupported(self):
        with self.assertRaises(UnsupportedRequest):
            shim(make_transport()).create(model="structured-aux/mcp", messages=[{"role": "user", "content": "x"}])

    def test_single_candidate_is_unsupported(self):
        with self.assertRaises(UnsupportedRequest):
            shim(make_transport()).create(
                model="structured-aux/mcp",
                messages=[{"role": "user", "content": "x"}],
                tools=[self.TOOLS[0]],
            )

    def test_only_argument_bearing_tools_is_unsupported(self):
        tools = [
            {"type": "function", "function": {"name": "a", "parameters": {"required": ["x"]}}},
            {"type": "function", "function": {"name": "b", "parameters": {"required": ["y"]}}},
        ]
        with self.assertRaises(UnsupportedRequest):
            shim(make_transport()).create(model="structured-aux/mcp", messages=[{"role": "user", "content": "x"}], tools=tools)


class CompressionRoutingTests(unittest.TestCase):
    def test_returns_an_extractive_digest(self):
        def responder(body):
            return {name: choice("KEEP") for name in body["questions"]}

        transport = dynamic_transport(responder)
        response = shim(transport).create(
            model="structured-aux/compression",
            messages=[{"role": "user", "content": "alpha block\n\nbeta block\n\ngamma block"}],
        )
        content = response.choices[0].message.content
        self.assertIn("[structured-aux extractive digest]", content)
        self.assertIn("alpha block", content)

    def test_digest_never_contains_generated_text(self):
        def responder(body):
            return {name: choice("DROP") for name in body["questions"]}

        transport = dynamic_transport(responder)
        response = shim(transport).create(
            model="structured-aux/compression",
            messages=[{"role": "user", "content": "alpha block\n\nbeta block"}],
        )
        content = response.choices[0].message.content
        self.assertIn("beta block", content)  # final block retained as a floor
        self.assertIn("no text was generated", content)

    def test_empty_prompt_is_unsupported(self):
        with self.assertRaises(UnsupportedRequest):
            shim(make_transport()).create(
                model="structured-aux/compression", messages=[{"role": "system", "content": "x"}]
            )


class ResponseShapeTests(unittest.TestCase):
    def test_usage_is_normalised_to_openai_keys(self):
        transport = make_transport(
            answers={"verdict": choice("APPROVE")},
            usage={"input_tokens": 7, "output_tokens": 3, "total_tokens": 10},
        )
        response = shim(transport).create(model="structured-aux/approval", messages=APPROVAL_MESSAGES)
        self.assertEqual(response.usage["prompt_tokens"], 7)
        self.assertEqual(response.usage["completion_tokens"], 3)
        self.assertEqual(response.usage["total_tokens"], 10)

    def test_declares_wrapper_opt_outs(self):
        # Read by agent.auxiliary_client._client_declares; without these the client is
        # re-dispatched through an HTTP wire adapter.
        self.assertTrue(StructuredAuxClient.HERMES_SKIP_TRANSPORT_WRAP)
        self.assertTrue(StructuredAuxClient.HERMES_SKIP_ASYNC_WRAP)

    def test_exposes_chat_completions_surface(self):
        client = shim(make_transport(answers={"verdict": choice("APPROVE")}))
        self.assertTrue(callable(client.chat.completions.create))


if __name__ == "__main__":
    unittest.main()
