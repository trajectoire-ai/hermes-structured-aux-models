"""Segmentation, selection, and digest construction."""

from __future__ import annotations

import pytest

from structured_aux import compression


def keep_all(state, questions):
    return {name: {"choice": "KEEP", "probabilities": {"KEEP": 0.9}, "confidence": 0.9} for name in questions}


def drop_all(state, questions):
    return {name: {"choice": "DROP", "probabilities": {"DROP": 0.9}, "confidence": 0.9} for name in questions}


# ── segmentation ────────────────────────────────────────────────────────────────


def test_splits_on_blank_lines():
    assert compression.segment("a\n\nb\n\nc", max_blocks=10, min_block_chars=1) == ["a", "b", "c"]


def test_merges_small_paragraphs_up_to_min_size():
    blocks = compression.segment("aa\n\nbb\n\ncc\n\ndd", max_blocks=10, min_block_chars=5)
    assert blocks == ["aa\n\nbb\n\ncc", "dd"]


def test_is_deterministic():
    text = "\n\n".join(f"paragraph {i}" for i in range(40))
    assert compression.segment(text, max_blocks=8, min_block_chars=1) == compression.segment(
        text, max_blocks=8, min_block_chars=1
    )


def test_cap_merges_overflow_instead_of_dropping_text():
    text = "\n\n".join(f"p{i}" for i in range(20))
    blocks = compression.segment(text, max_blocks=4, min_block_chars=1)

    assert len(blocks) == 4
    joined = "\n\n".join(blocks)
    for i in range(20):
        assert f"p{i}" in joined


@pytest.mark.parametrize("text", ["", "\n\n   \n\n"])
def test_empty_text_yields_no_blocks(text):
    assert compression.segment(text) == []


# ── selection ───────────────────────────────────────────────────────────────────


def test_returns_one_flag_per_block(fake_client):
    client = fake_client(lambda state, questions: {n: {"choice": "KEEP"} for n in questions})
    flags = compression.select(client, ["a", "b", "c"])

    assert flags == [True, True, True]
    assert len(client.calls) == 1


def test_batches_larger_inputs(fake_client):
    client = fake_client(keep_all)
    flags = compression.select(client, [f"b{i}" for i in range(40)], blocks_per_call=16)

    assert len(flags) == 40
    assert [len(call["questions"]) for call in client.calls] == [16, 16, 8]


def test_mixed_decisions_are_respected(fake_client):
    client = fake_client(lambda state, questions: {n: {"choice": "KEEP" if n == "block_1" else "DROP"} for n in questions})
    assert compression.select(client, ["a", "b", "c"]) == [False, True, False]


def test_blocks_are_sent_as_state(fake_client):
    client = fake_client(keep_all)
    compression.select(client, ["alpha", "beta"])

    blocks = client.calls[0]["state"]["blocks"]
    assert [b["id"] for b in blocks] == ["block_0", "block_1"]
    assert [b["text"] for b in blocks] == ["alpha", "beta"]


def test_empty_input_makes_no_call(fake_client):
    client = fake_client(keep_all)
    assert compression.select(client, []) == []
    assert client.calls == []


# ── digest ──────────────────────────────────────────────────────────────────────


def test_header_records_provenance():
    digest = compression.build_digest(["a", "b"], [True, False], model="~typesafe/jev-latest")

    assert "[structured-aux extractive digest]" in digest
    assert "retained 1/2 blocks" in digest
    assert "~typesafe/jev-latest" in digest


def test_only_kept_blocks_appear():
    digest = compression.build_digest(["keep me", "drop me"], [True, False])

    assert "keep me" in digest
    assert "drop me" not in digest


def test_retains_final_block_when_nothing_is_kept():
    digest = compression.build_digest(["first", "last"], [False, False])

    assert "last" in digest
    assert "first" not in digest


def test_respects_the_budget():
    blocks = [f"block-{i}-" + ("x" * 500) for i in range(20)]
    assert len(compression.build_digest(blocks, [True] * 20, budget_chars=800)) <= 800


def test_never_empty():
    assert compression.build_digest([], [])


# ── end to end ──────────────────────────────────────────────────────────────────


def test_selects_then_digests(fake_client):
    digest = compression.compress(fake_client(keep_all), "alpha\n\nbeta\n\ngamma", min_block_chars=1)

    assert "alpha" in digest
    assert "gamma" in digest


def test_drops_are_applied(fake_client):
    digest = compression.compress(fake_client(drop_all), "alpha\n\nbeta\n\ngamma", min_block_chars=1)

    assert "gamma" in digest
    assert "alpha" not in digest


def test_short_prompt_stays_one_block(fake_client):
    # Below the minimum block size the whole prompt is a single unit, so a DROP
    # decision cannot lose part of it — the floor retains it intact.
    assert "alpha" in compression.compress(fake_client(drop_all), "alpha\n\nbeta\n\ngamma")


def test_empty_prompt_raises(fake_client):
    with pytest.raises(ValueError):
        compression.compress(fake_client(keep_all), "   ")
