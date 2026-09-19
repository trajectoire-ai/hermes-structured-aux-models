"""Segmentation, selection, and digest construction."""

from __future__ import annotations

import unittest

from helpers import FakeDecisionClient, choice

from structured_aux import compression


def keep_all(state, questions):
    return {name: choice("KEEP") for name in questions}


def drop_all(state, questions):
    return {name: choice("DROP") for name in questions}


class SegmentTests(unittest.TestCase):
    def test_splits_on_blank_lines(self):
        blocks = compression.segment("a\n\nb\n\nc", max_blocks=10, min_block_chars=1)
        self.assertEqual(blocks, ["a", "b", "c"])

    def test_merges_small_paragraphs_up_to_min_size(self):
        text = "aa\n\nbb\n\ncc\n\ndd"
        blocks = compression.segment(text, max_blocks=10, min_block_chars=5)
        self.assertEqual(blocks, ["aa\n\nbb\n\ncc", "dd"])

    def test_is_deterministic(self):
        text = "\n\n".join(f"paragraph {i}" for i in range(40))
        first = compression.segment(text, max_blocks=8, min_block_chars=1)
        second = compression.segment(text, max_blocks=8, min_block_chars=1)
        self.assertEqual(first, second)

    def test_cap_merges_overflow_instead_of_dropping_text(self):
        text = "\n\n".join(f"p{i}" for i in range(20))
        blocks = compression.segment(text, max_blocks=4, min_block_chars=1)
        self.assertEqual(len(blocks), 4)
        joined = "\n\n".join(blocks)
        for i in range(20):
            self.assertIn(f"p{i}", joined)

    def test_empty_text_yields_no_blocks(self):
        self.assertEqual(compression.segment(""), [])
        self.assertEqual(compression.segment("\n\n   \n\n"), [])


class SelectTests(unittest.TestCase):
    def test_returns_one_flag_per_block(self):
        client = FakeDecisionClient(lambda state, questions: {n: choice("KEEP") for n in questions})
        flags = compression.select(client, ["a", "b", "c"])
        self.assertEqual(flags, [True, True, True])
        self.assertEqual(len(client.calls), 1)

    def test_batches_larger_inputs(self):
        client = FakeDecisionClient(lambda state, questions: {n: choice("KEEP") for n in questions})
        flags = compression.select(client, [f"b{i}" for i in range(40)], blocks_per_call=16)
        self.assertEqual(len(flags), 40)
        self.assertEqual(len(client.calls), 3)
        self.assertEqual([len(c["questions"]) for c in client.calls], [16, 16, 8])

    def test_mixed_decisions_are_respected(self):
        def responder(state, questions):
            return {name: choice("KEEP" if name == "block_1" else "DROP") for name in questions}

        client = FakeDecisionClient(responder)
        flags = compression.select(client, ["a", "b", "c"])
        self.assertEqual(flags, [False, True, False])

    def test_blocks_are_sent_as_state(self):
        client = FakeDecisionClient(keep_all)
        compression.select(client, ["alpha", "beta"])
        blocks = client.calls[0]["state"]["blocks"]
        self.assertEqual([b["id"] for b in blocks], ["block_0", "block_1"])
        self.assertEqual([b["text"] for b in blocks], ["alpha", "beta"])

    def test_empty_input_makes_no_call(self):
        client = FakeDecisionClient(keep_all)
        self.assertEqual(compression.select(client, []), [])
        self.assertEqual(client.calls, [])


class DigestTests(unittest.TestCase):
    def test_header_records_provenance(self):
        digest = compression.build_digest(["a", "b"], [True, False], model="typesafe/jev-1.13")
        self.assertIn("[structured-aux extractive digest]", digest)
        self.assertIn("retained 1/2 blocks", digest)
        self.assertIn("typesafe/jev-1.13", digest)

    def test_only_kept_blocks_appear(self):
        digest = compression.build_digest(["keep me", "drop me"], [True, False])
        self.assertIn("keep me", digest)
        self.assertNotIn("drop me", digest)

    def test_retains_final_block_when_nothing_is_kept(self):
        digest = compression.build_digest(["first", "last"], [False, False])
        self.assertIn("last", digest)
        self.assertNotIn("first", digest)

    def test_respects_the_budget(self):
        blocks = [f"block-{i}-" + ("x" * 500) for i in range(20)]
        digest = compression.build_digest(blocks, [True] * 20, budget_chars=800)
        self.assertLessEqual(len(digest), 800)

    def test_never_empty(self):
        self.assertTrue(compression.build_digest([], []))


class CompressTests(unittest.TestCase):
    def test_end_to_end_selects_then_digests(self):
        client = FakeDecisionClient(keep_all)
        digest = compression.compress(
            client, "alpha\n\nbeta\n\ngamma", model="typesafe/jev-1.13", min_block_chars=1
        )
        self.assertIn("alpha", digest)
        self.assertIn("gamma", digest)

    def test_end_to_end_drops_are_applied(self):
        client = FakeDecisionClient(drop_all)
        digest = compression.compress(client, "alpha\n\nbeta\n\ngamma", min_block_chars=1)
        self.assertIn("gamma", digest)
        self.assertNotIn("alpha", digest)

    def test_default_segmentation_keeps_a_short_prompt_as_one_block(self):
        # Below the minimum block size the whole prompt is a single unit, so a DROP
        # decision cannot lose part of it — the floor retains it intact.
        client = FakeDecisionClient(drop_all)
        digest = compression.compress(client, "alpha\n\nbeta\n\ngamma")
        self.assertIn("alpha", digest)

    def test_empty_prompt_raises(self):
        with self.assertRaises(ValueError):
            compression.compress(FakeDecisionClient(keep_all), "   ")


if __name__ == "__main__":
    unittest.main()
