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

import re

from . import config, contracts

DIGEST_HEADER = "[structured-aux extractive digest]"

# Split on blank lines: prompt transcripts and Hermes' own section separators both
# present this way, and it costs nothing when the prompt has no blank lines at all.
_PARAGRAPH_SPLIT_RE = re.compile(r"\n[ \t]*\n")


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
    state_goal: str = "",
) -> list[bool]:
    """Ask the provider, in bounded batches, which blocks must survive.

    Returns a ``keep`` flag per block, aligned with ``blocks``. A block the provider
    does not answer for raises rather than defaulting: an unanswered block is an unknown
    retention decision, and guessing it is exactly the failure this plugin exists to
    avoid.
    """
    if not blocks:
        return []
    blocks_per_call = blocks_per_call if blocks_per_call is not None else config.compression_blocks_per_call()

    keep: list[bool] = []
    for start in range(0, len(blocks), blocks_per_call):
        chunk = blocks[start : start + blocks_per_call]
        names = [f"block_{start + offset}" for offset in range(len(chunk))]
        questions = contracts.compression_questions(names)
        state = {
            "retention_goal": state_goal or "Preserve the smallest sufficient working set for correct continuation.",
            "blocks": [{"id": name, "text": text} for name, text in zip(names, chunk)],
        }
        result = client.ask(state=state, questions=questions)
        for name in names:
            keep.append(result.label(name, contracts.COMPRESSION_LABELS) == "KEEP")
    return keep


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
