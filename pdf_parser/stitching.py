"""Cross-page merge — the one place we want deliberate, careful logic.

Goal: fragments of ONE logical table that were parsed on separate pages (the
incident-code Table 8 runs pages 19-22) become a single ``TableBlock`` with a
correct ``page_span`` and no rows lost.

CRITICAL GUARD: we merge ONLY on an explicit continuation signal —
  * the later fragment's ``continued_from`` flag, or
  * the earlier fragment's ``continued`` flag, or
  * a repeated real table label ("Table 8" == "Table 8"), or
  * a "(cont'd)" / "(continued)" marker in the later fragment's label/caption.
We NEVER merge on column similarity or section adjacency alone. That is exactly
what would wrongly fuse the eight Section-6 archived reaction tables (pages
24-31) — they share the Case Definition / Severity / Imputability columns and the
same section, but each is a distinct table with no continuation signal.

A secondary structural bound (same-or-next page) keeps a stray flag from running
a merge across the whole document; it only ever *restricts* merging, never causes
it.
"""

from __future__ import annotations

import re
from typing import Iterable

from .schema import ContentBlock, PageParse, Skeleton, TableBlock

_CONTD_RE = re.compile(r"\(?\bcont(?:'?d|inued)\b\)?", re.I)
_REAL_TABLE_LABEL_RE = re.compile(r"^(Table|Figure)\s+\d+", re.I)


# --------------------------------------------------------------------------- #
# Page helpers
# --------------------------------------------------------------------------- #
def _start_page(block: ContentBlock) -> int:
    return block.page_span[0] if isinstance(block, TableBlock) else block.page


def _end_page(block: ContentBlock) -> int:
    return block.page_span[1] if isinstance(block, TableBlock) else block.page


# --------------------------------------------------------------------------- #
# Continuation decision
# --------------------------------------------------------------------------- #
def _has_contd_marker(table: TableBlock) -> bool:
    return bool(_CONTD_RE.search(table.label or "") or _CONTD_RE.search(table.caption or ""))


def _same_real_label(a: TableBlock, b: TableBlock) -> bool:
    """True only for a repeated genuine table label like 'Table 8' == 'Table 8'.
    Archived reaction tables are labeled by distinct reaction names, so this is
    False across two different archived tables."""
    la, lb = (a.label or "").strip(), (b.label or "").strip()
    return bool(la) and la == lb and bool(_REAL_TABLE_LABEL_RE.match(la))


def _is_continuation(open_table: TableBlock, candidate: TableBlock) -> bool:
    """Whether ``candidate`` continues ``open_table``. Requires an explicit
    continuation signal AND same-or-next-page adjacency (the latter only bounds
    runaway merges; it can never trigger a merge on its own)."""
    signal = (
        candidate.continued_from
        or open_table.continued
        or _same_real_label(open_table, candidate)
        or _has_contd_marker(candidate)
    )
    if not signal:
        return False
    # Structural bound: the continuation must begin on the same page the open
    # table ended on, or the next page.
    return _start_page(candidate) <= _end_page(open_table) + 1


# --------------------------------------------------------------------------- #
# Merge
# --------------------------------------------------------------------------- #
def _extend_unique(target: list[str], extra: Iterable[str]) -> None:
    for item in extra:
        if item not in target:
            target.append(item)


def _merge_into(open_table: TableBlock, frag: TableBlock) -> None:
    """Fold ``frag`` into ``open_table`` in place: no rows lost, span widened."""
    open_table.rows.extend(frag.rows)
    open_table.page_span = [
        min(open_table.page_span[0], frag.page_span[0]),
        max(open_table.page_span[1], frag.page_span[1]),
    ]
    if not open_table.columns and frag.columns:
        open_table.columns = list(frag.columns)
    if not open_table.caption and frag.caption:
        open_table.caption = frag.caption
    _extend_unique(open_table.notes, frag.notes)
    _extend_unique(open_table.footnotes, frag.footnotes)
    # Propagate whether the logical table is still open after this fragment.
    open_table.continued = frag.continued


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def merge_blocks(blocks: list[ContentBlock]) -> list[ContentBlock]:
    """Merge cross-page table fragments in an ordered (reading-order) block list.

    Non-table blocks pass through untouched and in place. Input blocks are not
    mutated — fragments are deep-copied before merging."""
    result: list[ContentBlock] = []
    last_table: TableBlock | None = None

    for block in blocks:
        if isinstance(block, TableBlock):
            frag = block.model_copy(deep=True)
            if last_table is not None and _is_continuation(last_table, frag):
                _merge_into(last_table, frag)
            else:
                result.append(frag)
                last_table = frag
        else:
            result.append(block.model_copy(deep=True))
            # last_table stays referenced: a continuation may legitimately follow
            # a header line on the next page. The explicit-signal guard prevents
            # an unrelated later table from merging.
    return result


def stitch(skeleton: Skeleton, page_results: dict[int, PageParse]) -> list[ContentBlock]:
    """Flatten per-page results into one reading-order stream, then merge."""
    stream: list[ContentBlock] = []
    for page_number in sorted(page_results):
        stream.extend(page_results[page_number].blocks)
    return merge_blocks(stream)
