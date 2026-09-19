"""Task detection and contract shape."""

from __future__ import annotations

import unittest

from helpers import make_transport  # noqa: F401  (ensures repo root is on sys.path)

from structured_aux import contracts


class DetectTaskTests(unittest.TestCase):
    def test_model_id_slash_form(self):
        self.assertEqual(contracts.detect_task("structured-aux/approval", []), "approval")
        self.assertEqual(contracts.detect_task("structured-aux/compression", []), "compression")
        self.assertEqual(contracts.detect_task("structured-aux/mcp", []), "mcp")

    def test_model_id_colon_and_dot_forms(self):
        self.assertEqual(contracts.detect_task("structured-aux:approval", []), "approval")
        self.assertEqual(contracts.detect_task("structured-aux.compression", []), "compression")

    def test_model_id_hyphen_form(self):
        self.assertEqual(contracts.detect_task("structured-aux-mcp", []), "mcp")
        self.assertEqual(contracts.detect_task("structured_aux_approval", []), "approval")

    def test_bare_task_name(self):
        self.assertEqual(contracts.detect_task("approval", []), "approval")

    def test_unknown_model_is_not_served(self):
        self.assertIsNone(contracts.detect_task("gpt-4o", []))
        self.assertIsNone(contracts.detect_task("typesafe/jev-1.13", []))
        self.assertIsNone(contracts.detect_task("", []))

    def test_content_fallback_for_approval_only(self):
        messages = [{"role": "user", "content": "Respond with exactly one word: APPROVE, DENY, or ESCALATE"}]
        self.assertEqual(contracts.detect_task("", messages), "approval")

    def test_content_fallback_does_not_hijack_other_tasks(self):
        messages = [{"role": "user", "content": "Summarise the following conversation transcript."}]
        self.assertIsNone(contracts.detect_task("", messages))

    def test_unrelated_task_name_is_not_served(self):
        # skills_hub is deliberately out of scope: Hermes has no call site for it.
        self.assertIsNone(contracts.detect_task("structured-aux/skills_hub", []))


class ApprovalContractTests(unittest.TestCase):
    def test_question_is_a_bounded_choice(self):
        questions = contracts.approval_questions()
        self.assertEqual(list(questions), ["verdict"])
        verdict = questions["verdict"]
        self.assertEqual(verdict["type"], "choice")
        self.assertEqual(set(verdict["criteria"]), set(contracts.APPROVAL_LABELS))
        self.assertGreaterEqual(len(verdict["criteria"]), 2)

    def test_labels_cover_escalation(self):
        # Hermes maps any unrecognised guardian answer to "escalate"; the contract must
        # express that outcome explicitly rather than collapsing it into a refusal.
        self.assertIn("ESCALATE", contracts.APPROVAL_LABELS)


class McpContractTests(unittest.TestCase):
    def test_choice_over_candidates(self):
        candidates = {"read_file": "Read a file", "search": "Search text"}
        questions = contracts.mcp_tool_questions(candidates)
        self.assertEqual(questions["tool"]["type"], "choice")
        self.assertEqual(set(questions["tool"]["criteria"]), set(candidates))


class CompressionContractTests(unittest.TestCase):
    def test_one_question_per_block(self):
        questions = contracts.compression_questions(["block_0", "block_1", "block_2"])
        self.assertEqual(list(questions), ["block_0", "block_1", "block_2"])
        for question in questions.values():
            self.assertEqual(question["type"], "choice")
            self.assertEqual(set(question["criteria"]), set(contracts.COMPRESSION_LABELS))


if __name__ == "__main__":
    unittest.main()
