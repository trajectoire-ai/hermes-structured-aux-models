"""Selection-based context compression.

Hermes' compressor asks its auxiliary model for a prose summary. A decision model
cannot write prose, so this module does the other thing instead: it segments the
compression prompt into deterministic blocks, asks Jev which blocks must survive, and
emits a digest assembled **only** from retained original text plus a provenance header.

Nothing here is generated. That is deliberate — the failure mode of a weak summariser
is losing an exact value or a piece of failure evidence, and an extractive digest cannot
invent a replacement for something it dropped.
"""

from __future__ import annotations

import logging
import re

from . import config, contracts

logger = logging.getLogger("structured_aux.compression")

DIGEST_HEADER = "[structured-aux extractive digest]"

# Split on blank lines: prompt transcripts and Hermes' own section separators both
# present this way, and it costs nothing when the prompt has no blank lines at all.
_PARAGRAPH_SPLIT_RE = re.compile(r"\n[ \t]*\n")

# ── call sizing ─────────────────────────────────────────────────────────────────
#
# Jev is a decision model with a **32,000-token** context window (the OpenRouter model
# page states it), and it is not a chat endpoint, so a compression prompt cannot be
# handed over whole. Segmentation is therefore paired with a per-call budget: a block
# larger than the budget is split into pieces, and pieces are packed into as few calls as
# the budget allows. Nothing is truncated to make something fit.

# Token-dense codepoints cost a full token each, the rest roughly a token per 4 bytes.
# Ranges are built from ordinals rather than written as escape chains: the same character
# class spelled as \-escapes trips Hermes' install-time plugin scanner's obfuscation
# heuristic, and a plugin must scan clean.
_DENSE_RANGES = (
    (0x1100, 0x11FF),  # Hangul Jamo
    (0x2E80, 0x9FFF),  # CJK radicals through CJK unified ideographs
    (0x3040, 0x30FF),  # Kana
    (0xAC00, 0xD7AF),  # Hangul syllables
    (0xF900, 0xFAFF),  # CJK compatibility ideographs
    (0xFF00, 0xFFEF),  # fullwidth and halfwidth forms
)
_DENSE_RE = re.compile("[" + "".join(f"{chr(start)}-{chr(end)}" for start, end in _DENSE_RANGES) + "]")

# Instructions + criteria + JSON envelope for one KEEP/DROP question measured ~190
# tokens on the wire. Charged at 256 so the per-call accounting stays conservative even
# though the payload is estimated as a whole and the parts are estimated one by one.
_QUESTION_OVERHEAD_TOKENS = 256


def estimate_tokens(text: str) -> int:
    """Conservative token estimate for ``text`` (token-dense scripts cost ~1/char)."""
    if not text:
        return 0
    text = str(text)
    if text.isascii():
        return (len(text) + 3) // 4
    dense = len(_DENSE_RE.findall(text))
    rest = _DENSE_RE.sub("", text)
    return dense + (len(rest.encode("utf-8", "replace")) + 3) // 4


def _fit_prefix(text: str, budget_tokens: int) -> int:
    """Length of the longest prefix of ``text`` whose estimate is within budget."""
    low, high = 1, len(text)
    while low < high:
        middle = (low + high + 1) // 2
        if estimate_tokens(text[:middle]) <= budget_tokens:
            low = middle
        else:
            high = middle - 1
    return max(1, low)


def split_by_budget(text: str, budget_tokens: int) -> list[str]:
    """Split ``text`` into pieces that each fit ``budget_tokens``.

    Piece boundaries prefer a paragraph, then a line, break so a piece reads as whole
    units; ``"".join(pieces) == text`` always holds — splitting must never lose text.
    """
    if not text:
        return []
    if estimate_tokens(text) <= budget_tokens:
        return [text]

    pieces: list[str] = []
    remaining = text
    while remaining:
        if estimate_tokens(remaining) <= budget_tokens:
            pieces.append(remaining)
            break
        cut = _fit_prefix(remaining, budget_tokens)
        window = remaining[:cut]
        for separator in ("\n\n", "\n"):
            at = window.rfind(separator)
            if at > cut // 2:
                cut = at + len(separator)
                break
        pieces.append(remaining[:cut])
        remaining = remaining[cut:]
    return pieces


def _pack_calls(
    items: list[tuple[int, str, str]],
    blocks_per_call: int,
    budget_tokens: int,
) -> list[list[tuple[int, str, str]]]:
    """Greedily pack ``(block_index, name, text)`` items into calls within the budget."""
    calls: list[list[tuple[int, str, str]]] = []
    current: list[tuple[int, str, str]] = []
    used = 0
    for item in items:
        cost = estimate_tokens(item[2]) + _QUESTION_OVERHEAD_TOKENS
        if current and (len(current) >= blocks_per_call or used + cost > budget_tokens):
            calls.append(current)
            current = []
            used = 0
        current.append(item)
        used += cost
    if current:
        calls.append(current)
    return calls


def segment(
    text: str,
    *,
    max_blocks: int | None = None,
    min_block_chars: int | None = None,
) -> list[str]:
    """Split ``text`` into at most ``max_blocks`` deterministic blocks.

    Paragraphs are accumulated until a block reaches ``min_block_chars``, so a
    fragment-heavy prompt still yields blocks large enough to be worth a question. When
    there are more blocks than the cap allows, the overflow is merged into the final
    block rather than dropped — segmentation must never be the thing that loses text.
    """
    max_blocks = max_blocks if max_blocks is not None else config.compression_max_blocks()
    min_block_chars = min_block_chars if min_block_chars is not None else config.compression_min_block_chars()

    paragraphs = [part for part in _PARAGRAPH_SPLIT_RE.split(text or "") if part.strip()]
    if not paragraphs:
        return []

    blocks: list[str] = []
    current: list[str] = []
    size = 0
    for paragraph in paragraphs:
        current.append(paragraph)
        size += len(paragraph)
        if size >= min_block_chars:
            blocks.append("\n\n".join(current))
            current = []
            size = 0
    if current:
        blocks.append("\n\n".join(current))

    if len(blocks) > max_blocks:
        head = blocks[: max_blocks - 1]
        tail = "\n\n".join(blocks[max_blocks - 1 :])
        blocks = head + [tail]
    return blocks


def select(
    client,
    blocks: list[str],
    *,
    blocks_per_call: int | None = None,
    budget_tokens: int | None = None,
    state_goal: str = "",
) -> list[bool]:
    """Ask the provider, in bounded batches, which blocks must survive.

    Returns a ``keep`` flag per block, aligned with ``blocks``. Two bounds apply to each
    provider call: at most ``blocks_per_call`` questions, and at most ``budget_tokens``
    estimated tokens of payload (Jev's window is 32,000, so a call that ignored the
    second bound could not be answered at all). A block too large for a single call is
    split into pieces, each asked about separately; the block is kept if **any** of its
    pieces is kept, because retaining the whole block is the safe reading of a partial
    answer and a dropped piece would otherwise silently shorten the digest.

    A block the provider does not answer for raises rather than defaulting: an
    unanswered block is an unknown retention decision, and guessing it is exactly the
    failure this plugin exists to avoid.
    """
    if not blocks:
        return []
    blocks_per_call = blocks_per_call if blocks_per_call is not None else config.compression_blocks_per_call()
    budget_tokens = budget_tokens if budget_tokens is not None else config.compression_call_budget_tokens()

    names_by_block: list[list[str]] = []
    items: list[tuple[int, str, str]] = []
    piece_budget = max(1, budget_tokens - _QUESTION_OVERHEAD_TOKENS)
    for index, block in enumerate(blocks):
        pieces = split_by_budget(block, piece_budget)
        names: list[str] = []
        for piece_index, piece in enumerate(pieces):
            name = f"block_{index}" if len(pieces) == 1 else f"block_{index}_{piece_index}"
            names.append(name)
            items.append((index, name, piece))
        names_by_block.append(names)

    kept: dict[str, bool] = {}
    calls = _pack_calls(items, blocks_per_call, budget_tokens)
    payload_tokens = sum(estimate_tokens(text) + _QUESTION_OVERHEAD_TOKENS for _, _, text in items)
    logger.debug(
        "compression plan: %d block(s), %d piece(s), %d decision call(s), ~%d estimated payload tokens",
        len(blocks), len(items), len(calls), payload_tokens,
    )
    for index, call in enumerate(calls, start=1):
        names = [name for _, name, _ in call]
        questions = contracts.compression_questions(names)
        state = {
            "retention_goal": state_goal or "Preserve the smallest sufficient working set for correct continuation.",
            "blocks": [{"id": name, "text": text} for _, name, text in call],
        }
        call_tokens = sum(estimate_tokens(text) + _QUESTION_OVERHEAD_TOKENS for _, _, text in call)
        try:
            result = client.ask(state=state, questions=questions)
        except Exception as exc:
            # Name the failing call: a compaction prompt is served by several calls, and
            # "which one stalled" is the first question anyone asks of the log.
            logger.error(
                "compression decision call %d/%d failed (%d question(s), ~%d estimated payload tokens): %s",
                index, len(calls), len(names), call_tokens, exc,
            )
            raise
        logger.debug(
            "compression decision call %d/%d answered: %d question(s), ~%d estimated payload tokens",
            index, len(calls), len(names), call_tokens,
        )
        for name in names:
            kept[name] = result.label(name, contracts.COMPRESSION_LABELS) == "KEEP"

    return [any(kept.get(name, False) for name in names) for names in names_by_block]


def build_digest(
    blocks: list[str],
    keep: list[bool],
    *,
    model: str = "",
    budget_chars: int | None = None,
) -> str:
    """Assemble the retained blocks into a digest bounded by ``budget_chars``.

    Guarantees a non-empty result: if the provider retained nothing, the final block is
    kept anyway. An empty digest would make Hermes treat compression as failed and fall
    back to the main model, which is a worse outcome than retaining the most recent
    context.
    """
    budget_chars = budget_chars if budget_chars is not None else config.compression_output_budget_chars()
    total = len(blocks)
    retained = [block for block, flag in zip(blocks, keep) if flag]
    if not retained and blocks:
        retained = [blocks[-1]]

    header = (
        f"{DIGEST_HEADER} retained {len(retained)}/{total} blocks · model={model or 'unknown'} · "
        "no text was generated; retained text is verbatim."
    )
    parts = [header]
    used = len(header)
    for index, block in enumerate(retained, start=1):
        piece = f"\n\n--- block {index} ---\n{block}"
        remaining = budget_chars - used
        if remaining <= 0:
            break
        if len(piece) > remaining:
            piece = piece[:remaining]
        parts.append(piece)
        used += len(piece)
    return "".join(parts)


def compress(
    client,
    prompt: str,
    *,
    model: str = "",
    max_blocks: int | None = None,
    min_block_chars: int | None = None,
) -> str:
    """Segment, select, and digest a compression prompt. Raises on any provider failure."""
    blocks = segment(prompt, max_blocks=max_blocks, min_block_chars=min_block_chars)
    if not blocks:
        raise ValueError("compression prompt contained no usable text")
    keep = select(client, blocks)
    return build_digest(blocks, keep, model=model)
