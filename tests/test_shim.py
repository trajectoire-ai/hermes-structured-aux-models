"""Shim behaviour: which requests are answered, how, and with which model."""

from __future__ import annotations

import pytest

from structured_aux import config
from structured_aux.decisions import DecisionError, UnsupportedRequest

TOOLS = [
    {
        "type": "function",
        "function": {"name": "read_file", "description": "Read a file", "parameters": {"type": "object", "properties": {}}},
    },
    {
        "type": "function",
        "function": {"name": "search", "description": "Search text", "parameters": {"type": "object", "properties": {}}},
    },
]


# ── approval ────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("label", ["APPROVE", "DENY", "ESCALATE"])
def test_verdict_is_returned_verbatim(shim, make_transport, choice, approval_messages, label):
    transport = make_transport(answers={"verdict": choice(label)})
    response = shim(transport).create(model="structured-aux/approval", messages=approval_messages)

    assert response.choices[0].message.content == label
    assert response.choices[0].finish_reason == "stop"


def test_out_of_contract_verdict_raises(shim, make_transport, choice, approval_messages):
    transport = make_transport(answers={"verdict": choice("PROBABLY_FINE")})
    with pytest.raises(DecisionError):
        shim(transport).create(model="structured-aux/approval", messages=approval_messages)


def test_guardian_policy_is_forwarded_as_state(shim, make_transport, choice, approval_messages):
    transport = make_transport(answers={"verdict": choice("APPROVE")})
    shim(transport).create(model="structured-aux/approval", messages=approval_messages)

    state = transport.calls[0]["body"]["state"]
    assert state["guardian_policy"] == "You are a command-risk guardian."
    assert "python3 -c" in state["request"]


def test_detected_from_content_when_model_is_generic(shim, make_transport, choice, approval_messages):
    transport = make_transport(answers={"verdict": choice("APPROVE")})
    response = shim(transport).create(model="", messages=approval_messages)
    assert response.choices[0].message.content == "APPROVE"


# ── the routing token is not a model id ─────────────────────────────────────────


@pytest.mark.parametrize(
    "model_id,task",
    [("structured-aux/approval", "approval"), ("structured-aux/mcp", "mcp"), ("structured-aux/compression", "compression")],
)
def test_routing_token_is_never_sent_as_the_model(shim, make_transport, dynamic_transport, choice, approval_messages, model_id, task):
    """Regression: the operator's routing token must not reach the provider as a model.

    ``auxiliary.<task>.model`` carries ``structured-aux/<task>`` so the shim can pick a
    contract. Sending that string as the Jev model would fail every decision call.
    """
    if task == "compression":
        transport = dynamic_transport(lambda body: {name: choice("KEEP") for name in body["questions"]})
        messages = [{"role": "user", "content": "alpha\n\nbeta"}]
    elif task == "mcp":
        transport = make_transport(answers={"tool": choice("search")})
        messages = [{"role": "user", "content": "find it"}]
    else:
        transport = make_transport(answers={"verdict": choice("APPROVE")})
        messages = approval_messages

    kwargs = {"tools": TOOLS} if task == "mcp" else {}
    shim(transport).create(model=model_id, messages=messages, **kwargs)

    sent_model = transport.calls[0]["body"]["model"]
    assert sent_model == config.decision_model()
    assert sent_model != model_id
    assert "structured-aux" not in sent_model


# ── unsupported requests ────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "model_id,messages",
    [
        ("gpt-4o", [{"role": "user", "content": "hi"}]),
        ("", [{"role": "user", "content": "summarise this"}]),
        ("structured-aux/skills_hub", []),
    ],
)
def test_unsupported_requests_raise(shim, make_transport, model_id, messages):
    with pytest.raises(UnsupportedRequest):
        shim(make_transport()).create(model=model_id, messages=messages)


def test_streaming_is_unsupported(shim, make_transport, approval_messages):
    with pytest.raises(UnsupportedRequest):
        shim(make_transport()).create(model="structured-aux/approval", messages=approval_messages, stream=True)


def test_approval_without_command_text_is_unsupported(shim, make_transport):
    with pytest.raises(UnsupportedRequest):
        shim(make_transport()).create(model="structured-aux/approval", messages=[{"role": "system", "content": "only"}])


# ── mcp ─────────────────────────────────────────────────────────────────────────


def test_mcp_selects_a_tool(shim, make_transport, choice):
    transport = make_transport(answers={"tool": choice("search")})
    response = shim(transport).create(
        model="structured-aux/mcp", messages=[{"role": "user", "content": "find the config"}], tools=TOOLS
    )

    assert response.choices[0].message.tool_calls[0].function.name == "search"
    assert response.choices[0].finish_reason == "tool_calls"


def test_mcp_sends_candidates_as_criteria(shim, make_transport, choice):
    transport = make_transport(answers={"tool": choice("read_file")})
    shim(transport).create(model="structured-aux/mcp", messages=[{"role": "user", "content": "x"}], tools=TOOLS)

    criteria = transport.calls[0]["body"]["questions"]["tool"]["criteria"]
    assert set(criteria) == {"read_file", "search"}


def test_mcp_excludes_tools_requiring_arguments(shim, make_transport, choice):
    tools = list(TOOLS) + [
        {
            "type": "function",
            "function": {
                "name": "write_file",
                "description": "Write",
                "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
            },
        }
    ]
    transport = make_transport(answers={"tool": choice("read_file")})
    shim(transport).create(model="structured-aux/mcp", messages=[{"role": "user", "content": "x"}], tools=tools)

    assert "write_file" not in transport.calls[0]["body"]["questions"]["tool"]["criteria"]


@pytest.mark.parametrize(
    "tools",
    [
        None,
        [TOOLS[0]],
        [
            {"type": "function", "function": {"name": "a", "parameters": {"required": ["x"]}}},
            {"type": "function", "function": {"name": "b", "parameters": {"required": ["y"]}}},
        ],
    ],
    ids=["no-tools", "single-candidate", "only-argument-bearing"],
)
def test_mcp_without_two_answerable_candidates_is_unsupported(shim, make_transport, tools):
    with pytest.raises(UnsupportedRequest):
        shim(make_transport()).create(
            model="structured-aux/mcp", messages=[{"role": "user", "content": "x"}], tools=tools
        )


# ── compression ─────────────────────────────────────────────────────────────────


def test_compression_returns_an_extractive_digest(shim, dynamic_transport, choice):
    transport = dynamic_transport(lambda body: {name: choice("KEEP") for name in body["questions"]})
    response = shim(transport).create(
        model="structured-aux/compression", messages=[{"role": "user", "content": "alpha block\n\nbeta block"}]
    )

    content = response.choices[0].message.content
    assert "[structured-aux extractive digest]" in content
    assert "alpha block" in content


def test_compression_digest_never_contains_generated_text(shim, dynamic_transport, choice):
    transport = dynamic_transport(lambda body: {name: choice("DROP") for name in body["questions"]})
    response = shim(transport).create(
        model="structured-aux/compression", messages=[{"role": "user", "content": "alpha block\n\nbeta block"}]
    )

    content = response.choices[0].message.content
    assert "beta block" in content  # final block retained as a floor
    assert "no text was generated" in content


def test_compression_without_text_is_unsupported(shim, make_transport):
    with pytest.raises(UnsupportedRequest):
        shim(make_transport()).create(model="structured-aux/compression", messages=[{"role": "system", "content": "x"}])


# ── response shape ──────────────────────────────────────────────────────────────


def test_usage_is_normalised_to_openai_keys(shim, make_transport, choice, approval_messages):
    transport = make_transport(
        answers={"verdict": choice("APPROVE")},
        usage={"input_tokens": 7, "output_tokens": 3, "total_tokens": 10},
    )
    response = shim(transport).create(model="structured-aux/approval", messages=approval_messages)

    assert response.usage == {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}


def test_client_declares_wrapper_opt_outs(shim, make_transport):
    # Read by agent.auxiliary_client._client_declares; without these the client is
    # re-dispatched through an HTTP wire adapter.
    client = shim(make_transport())
    assert client.HERMES_SKIP_TRANSPORT_WRAP is True
    assert client.HERMES_SKIP_ASYNC_WRAP is True


def test_client_exposes_the_chat_completions_surface(shim, make_transport):
    assert callable(shim(make_transport()).chat.completions.create)


def test_external_process_placeholder_is_not_used_as_a_credential(monkeypatch, make_transport):
    """The external_process credential branch passes the provider id as a placeholder.

    Sending it as a bearer token would 401 every decision call.
    """
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-real-key")
    from structured_aux.shim import StructuredAuxClient

    assert StructuredAuxClient(api_key="structured-aux").api_key == "sk-or-real-key"
    assert StructuredAuxClient(api_key="").api_key == "sk-or-real-key"
    assert StructuredAuxClient(api_key="sk-or-explicit").api_key == "sk-or-explicit"
