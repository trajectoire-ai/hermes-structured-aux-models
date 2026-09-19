"""Task detection and contract shape."""

from __future__ import annotations

import pytest

from structured_aux import contracts


@pytest.mark.parametrize(
    "model_id,expected",
    [
        ("structured-aux/approval", "approval"),
        ("structured-aux/compression", "compression"),
        ("structured-aux/mcp", "mcp"),
        ("structured-aux:approval", "approval"),
        ("structured-aux.compression", "compression"),
        ("structured-aux-mcp", "mcp"),
        ("structured_aux_approval", "approval"),
        ("approval", "approval"),
    ],
)
def test_detect_task_from_model_id(model_id, expected):
    assert contracts.detect_task(model_id, []) == expected


@pytest.mark.parametrize("model_id", ["gpt-4o", "typesafe/jev-1.13", "~typesafe/jev-latest", ""])
def test_unknown_model_is_not_served(model_id):
    assert contracts.detect_task(model_id, []) is None


def test_content_fallback_for_approval_only():
    messages = [{"role": "user", "content": "Respond with exactly one word: APPROVE, DENY, or ESCALATE"}]
    assert contracts.detect_task("", messages) == "approval"


def test_content_fallback_does_not_hijack_other_tasks():
    messages = [{"role": "user", "content": "Summarise the following conversation transcript."}]
    assert contracts.detect_task("", messages) is None


def test_out_of_scope_task_is_not_served():
    # skills_hub is deliberately out of scope: Hermes has no call site for it.
    assert contracts.detect_task("structured-aux/skills_hub", []) is None


def test_approval_question_is_a_bounded_choice():
    questions = contracts.approval_questions()
    assert list(questions) == ["verdict"]
    verdict = questions["verdict"]
    assert verdict["type"] == "choice"
    assert set(verdict["criteria"]) == set(contracts.APPROVAL_LABELS)
    assert len(verdict["criteria"]) >= 2


def test_approval_labels_cover_escalation():
    # Hermes maps any unrecognised guardian answer to "escalate"; the contract must
    # express that outcome explicitly rather than collapsing it into a refusal.
    assert "ESCALATE" in contracts.APPROVAL_LABELS


def test_mcp_question_is_a_choice_over_candidates():
    candidates = {"read_file": "Read a file", "search": "Search text"}
    questions = contracts.mcp_tool_questions(candidates)
    assert questions["tool"]["type"] == "choice"
    assert set(questions["tool"]["criteria"]) == set(candidates)


def test_compression_asks_one_question_per_block():
    questions = contracts.compression_questions(["block_0", "block_1", "block_2"])
    assert list(questions) == ["block_0", "block_1", "block_2"]
    for question in questions.values():
        assert question["type"] == "choice"
        assert set(question["criteria"]) == set(contracts.COMPRESSION_LABELS)
