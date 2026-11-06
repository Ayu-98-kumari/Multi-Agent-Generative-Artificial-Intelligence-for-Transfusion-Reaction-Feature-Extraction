"""Tests for cross-page stitching.

Two anchors:
  * Table 8 (pages 19-22): four fragments -> one table, span [19,22], no row loss.
  * The eight Section-6 archived reaction tables share columns + section but have
    NO continuation signal -> they must stay eight distinct tables (the main risk).
"""

from __future__ import annotations

from pdf_parser.schema import ProseBlock, TableBlock
from pdf_parser.stitching import merge_blocks

ARCHIVED_COLUMNS = ["Case Definition", "Severity", "Imputability"]


def _table(label, page, *, rows=1, columns=None, caption="",
           continued_from=False, continued=False):
    cols = columns if columns is not None else ["Code", "Description"]
    return TableBlock(
        label=label, caption=caption, page_span=[page, page], columns=cols,
        rows=[{"cells": {c: f"{label}-r{i}-{c}" for c in cols}} for i in range(rows)],
        continued_from=continued_from, continued=continued,
    )


# --------------------------------------------------------------------------- #
# Table 8 — the merge that must happen
# --------------------------------------------------------------------------- #
def test_table8_four_fragments_merge_to_one():
    frags = [
        _table("Table 8", 19, rows=10, continued=True),
        _table("Table 8", 20, rows=12, continued_from=True, continued=True),
        _table("Table 8", 21, rows=12, continued_from=True, continued=True),
        _table("Table 8", 22, rows=6, continued_from=True),
    ]
    out = merge_blocks(frags)
    assert len(out) == 1
    merged = out[0]
    assert merged.label == "Table 8"
    assert merged.page_span == [19, 22]
    assert len(merged.rows) == 10 + 12 + 12 + 6  # no row loss
    assert merged.continued is False            # closed at the last fragment


def test_table8_merges_via_continued_from_even_without_repeated_label():
    """A continuation fragment with a blank label still merges via continued_from."""
    frags = [
        _table("Table 8", 19, rows=5, continued=True),
        _table("", 20, rows=5, continued_from=True),
    ]
    out = merge_blocks(frags)
    assert len(out) == 1
    assert out[0].page_span == [19, 20]
    assert len(out[0].rows) == 10


def test_contd_marker_triggers_merge():
    frags = [
        _table("Table 8", 19, rows=3),
        _table("Table 8 (cont'd)", 20, rows=3, continued_from=False),
    ]
    out = merge_blocks(frags)
    assert len(out) == 1
    assert out[0].page_span == [19, 20]


# --------------------------------------------------------------------------- #
# Section 6 — the merge that must NOT happen (main risk)
# --------------------------------------------------------------------------- #
def test_eight_archived_tables_stay_distinct():
    names = [
        ("Allergic reaction", 24), ("Delayed hemolytic transfusion reaction (DHTR)", 25),
        ("Delayed serologic transfusion reaction (DSTR)", 26),
        ("Febrile non-hemolytic transfusion reaction (FNHTR)", 27),
        ("Hypotensive transfusion reaction (HTR)", 28),
        ("Post transfusion purpura (PTP)", 29),
        ("Transfusion-associated dyspnea (TAD)", 30),
        ("Transfusion-associated graft vs. host disease (TAGVHD)", 31),
    ]
    # Same columns, consecutive pages, same section — but NO continuation flags.
    frags = [_table(name, page, rows=4, columns=ARCHIVED_COLUMNS)
             for name, page in names]
    out = merge_blocks(frags)
    assert len(out) == 8                         # eight in -> eight out
    assert [t.label for t in out] == [n for n, _ in names]
    assert all(t.page_span[0] == t.page_span[1] for t in out)  # none widened


def test_same_columns_adjacent_pages_do_not_merge_without_signal():
    frags = [
        _table("Table A", 5, columns=ARCHIVED_COLUMNS),
        _table("Table B", 6, columns=ARCHIVED_COLUMNS),  # identical columns, next page
    ]
    out = merge_blocks(frags)
    assert len(out) == 2


def test_prose_between_tables_is_preserved_and_ordered():
    blocks = [
        _table("Table 8", 19, rows=2, continued=True),
        ProseBlock(text="interruption", page=19),
        _table("Table 8", 20, rows=2, continued_from=True),
    ]
    out = merge_blocks(blocks)
    # Table 8 still merges across the prose; prose retained in order.
    types = [type(b).__name__ for b in out]
    assert types == ["TableBlock", "ProseBlock"]
    assert out[0].page_span == [19, 20]


def test_far_apart_pages_not_merged_even_with_flag():
    """A stray continued_from on a distant table must not fuse across the doc."""
    frags = [
        _table("Table 1", 4, rows=2),          # not continued
        _table("Table 8", 19, rows=2, continued_from=True),  # far away
    ]
    out = merge_blocks(frags)
    assert len(out) == 2                         # page-adjacency bound holds
