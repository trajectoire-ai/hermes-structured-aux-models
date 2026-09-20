"""Segmentation, selection, and digest construction."""

from __future__ import annotations

import json

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


def test_default_budget_comes_from_the_configured_setting(monkeypatch):
    monkeypatch.setattr(compression.config, "compression_output_budget_chars", lambda: 1200)
    blocks = [f"block-{i}-" + ("x" * 500) for i in range(10)]

    assert len(compression.build_digest(blocks, [True] * 10)) <= 1200


def test_the_failing_compression_call_is_named_in_the_log(caplog):
    # A compaction prompt is served by several decision calls, so an error has to say which
    # one failed — that is the first question anyone asks of the log.
    from structured_aux.decisions import DecisionResult

    class _FailingOnSecondCall:
        def __init__(self):
            self.n = 0

        def ask(self, *, state, questions, model=None):
            self.n += 1
            if self.n == 2:
                raise RuntimeError("decision provider connection failed: The read operation timed out")
            return DecisionResult(
                answers={name: {"choice": "KEEP"} for name in questions},
                model="m", usage={}, request_id="x", latency_ms=1.0,
            )

    with caplog.at_level("ERROR"), pytest.raises(RuntimeError):
        compression.select(_FailingOnSecondCall(), ["a" * 500, "b" * 500], blocks_per_call=1, budget_tokens=1000)

    errors = [record.getMessage() for record in caplog.records if record.levelname == "ERROR"]
    assert any("compression decision call 2/2 failed" in message for message in errors)


def test_shipped_digest_budget_matches_the_documented_default():
    # The shipped ceiling is what the README's settings table advertises; pin it so the two
    # cannot drift. Raise both together when it changes.
    assert compression.config.compression_output_budget_chars() == 18000


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


# ── call budget (Jev's window is 32,000 tokens) ─────────────────────────────────


def payload_tokens(call):
    """Estimated tokens of the JSON a call would actually put on the wire."""
    return compression.estimate_tokens(
        json.dumps({"state": call["state"], "questions": call["questions"]}, separators=(",", ":"))
    )


def huge_block(paragraphs=300, size=600):
    return "\n\n".join(f"paragraph {i} " + "x" * size for i in range(paragraphs))


def test_split_by_budget_preserves_text():
    text = huge_block(paragraphs=40, size=500)
    pieces = compression.split_by_budget(text, 500)

    assert len(pieces) > 1
    assert "".join(pieces) == text
    assert all(compression.estimate_tokens(piece) <= 500 for piece in pieces)


def test_split_prefers_line_boundaries():
    text = "\n\n".join(f"para {i} " + "y" * 900 for i in range(10))
    pieces = compression.split_by_budget(text, 400)

    # Cuts land on the separators, so no piece begins mid-word or mid-paragraph.
    assert all(piece.startswith("para ") for piece in pieces)
    assert "".join(pieces) == text


def test_one_oversized_block_is_split_across_calls(fake_client):
    client = fake_client(keep_all)
    flags = compression.select(client, [huge_block()], budget_tokens=2000)

    assert flags == [True]  # one block in, one flag out
    assert len(client.calls) > 1  # it did not go out in a single call
    assert all(payload_tokens(call) <= 2000 for call in client.calls)


def test_every_call_in_a_big_compression_fits_the_budget(fake_client):
    client = fake_client(keep_all)
    prompt = huge_block(paragraphs=400, size=1500)  # ~600k chars, ~150k tokens
    compression.compress(client, prompt, max_blocks=48, min_block_chars=120)

    assert all(payload_tokens(call) <= compression.config.compression_call_budget_tokens() for call in client.calls)


def test_piece_keep_retains_the_whole_block(fake_client):
    client = fake_client(
        lambda state, questions: {
            name: {"choice": "KEEP" if name.endswith("_1") else "DROP"} for name in questions
        }
    )
    assert compression.select(client, [huge_block()], budget_tokens=2000) == [True]


def test_all_pieces_dropped_drops_the_block(fake_client):
    client = fake_client(drop_all)
    assert compression.select(client, [huge_block()], budget_tokens=2000) == [False]


def test_small_blocks_still_batch_by_count(fake_client):
    client = fake_client(keep_all)
    flags = compression.select(client, [f"b{i}" for i in range(40)], blocks_per_call=16)

    assert flags == [True] * 40
    assert [len(call["questions"]) for call in client.calls] == [16, 16, 8]
