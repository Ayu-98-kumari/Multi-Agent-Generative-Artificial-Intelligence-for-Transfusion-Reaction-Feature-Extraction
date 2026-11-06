"""Deterministic-extractor tests. No Groq, no network, no spend.

A tiny fixture JSON mimics the parser schema: a few reaction tables on target
pages, the excluded Table 7 (page 16), and a non-reaction table on a non-target
page. Assertions lock the selection and Column-1/Column-2 rules.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from reaction_extractor import config
from reaction_extractor.extractor import (
    ExtractionError,
    case_definition,
    extract_records,
    reaction_name,
    select_target_tables,
)


def _table(label, caption, page, rows, columns=None):
    return {
        "type": "table",
        "label": label,
        "caption": caption,
        "page_span": [page, page],
        "columns": columns or ["Case Definition", "Severity", "Imputability"],
        "rows": [{"cells": c} for c in rows],
    }


def _cd(text):
    return {"Case Definition": text, "Severity": "SEV", "Imputability": "IMP"}


def _doc(tables):
    return {"sections": [{"section_number": 1, "title": "x",
                          "page_span": [1, 99], "content": tables}]}


# --------------------------------------------------------------------------- #
# Selection
# --------------------------------------------------------------------------- #
def test_selects_exactly_target_pages_and_excludes_16(monkeypatch):
    # Restrict target pages for a compact fixture.
    monkeypatch.setattr(config, "TARGET_PAGES", [11, 13, 25])
    tables = [
        _table("Table 3", "TTI", 11, [_cd("def TTI")]),
        _table("Table 4", "AHTR", 13, [_cd("def AHTR")]),
        _table("Table 7", "Other transfusion reaction", 16, [_cd("Not Applicable")]),
        _table("Archived: DHTR", "DHTR full", 25, [_cd("def DHTR")]),
        _table("Table 2", "Some other table", 8, [_cd("irrelevant")]),
    ]
    doc = _doc(tables)
    pages = [t["page_span"][0] for t in select_target_tables(doc)]
    assert pages == [11, 13, 25]
    assert 16 not in pages   # Table 7 excluded
    assert 8 not in pages    # non-target table excluded


def test_missing_target_page_is_hard_error():
    doc = _doc([_table("Table 3", "TTI", 11, [_cd("x")])])  # 11 only
    with pytest.raises(ExtractionError):
        select_target_tables(doc)  # pages 13..31 missing


def test_duplicate_table_on_target_page_is_hard_error(monkeypatch):
    monkeypatch.setattr(config, "TARGET_PAGES", [11])
    doc = _doc([
        _table("Table 3", "TTI", 11, [_cd("a")]),
        _table("Table 3b", "TTI dup", 11, [_cd("b")]),
    ])
    with pytest.raises(ExtractionError):
        select_target_tables(doc)


# --------------------------------------------------------------------------- #
# Column 1 — reaction name
# --------------------------------------------------------------------------- #
def test_reaction_name_prefers_caption():
    t = _table("Archived: DHTR", "Delayed hemolytic transfusion reaction (DHTR)", 25, [])
    assert reaction_name(t) == "Delayed hemolytic transfusion reaction (DHTR)"


def test_reaction_name_falls_back_to_label_when_caption_empty():
    t = _table("Allergic reaction", "", 24, [])
    assert reaction_name(t) == "Allergic reaction"


def test_reaction_name_strips_archived_prefix_on_fallback():
    t = _table("Archived: DHTR", "", 25, [])
    assert reaction_name(t) == "DHTR"


# --------------------------------------------------------------------------- #
# Column 2 — case definition only
# --------------------------------------------------------------------------- #
def test_case_definition_pulls_only_that_column():
    t = _table("Table 4", "AHTR", 13, [
        {"Case Definition": "Definitive: back pain",
         "Severity": "Severe: hospitalization",
         "Imputability": "Definite: rechallenge"},
    ])
    cd = case_definition(t)
    assert "Definitive: back pain" in cd
    assert "Severe" not in cd          # Severity column excluded
    assert "hospitalization" not in cd
    assert "Definite: rechallenge" not in cd  # Imputability column excluded


def test_case_definition_concatenates_multiple_rows_in_order():
    t = _table("Table 3", "TTI", 11, [
        _cd("row one"), _cd(""), _cd("row three"),  # empty middle cell skipped
    ])
    assert case_definition(t) == "row one\n\nrow three"


def test_case_definition_preserves_logic_verbatim():
    t = _table("Table 6", "TRALI", 15,
               [_cd("A AND B EITHER C OR D")])
    assert case_definition(t) == "A AND B EITHER C OR D"


# --------------------------------------------------------------------------- #
# End-to-end records
# --------------------------------------------------------------------------- #
def test_extract_records_shape(monkeypatch):
    monkeypatch.setattr(config, "TARGET_PAGES", [11, 24])
    doc = _doc([
        _table("Table 3", "TTI", 11, [_cd("def TTI")]),
        _table("Allergic reaction", "", 24, [_cd("def Allergic")]),
        _table("Table 7", "Other", 16, [_cd("Not Applicable")]),
    ])
    recs = extract_records(doc)
    assert [r.source_page for r in recs] == [11, 24]
    assert recs[1].reaction == "Allergic reaction"
    assert recs[0].definition == "def TTI"


# --------------------------------------------------------------------------- #
# Against the real parsed JSON, when present
# --------------------------------------------------------------------------- #
REAL_JSON = config.PROJECT_ROOT / "bv-hv-protocol-current.json"


@pytest.mark.skipif(not REAL_JSON.exists(), reason="sample parsed JSON missing")
def test_real_json_yields_twelve():
    doc = json.loads(Path(REAL_JSON).read_text(encoding="utf-8"))
    recs = extract_records(doc)
    assert len(recs) == 12
    assert [r.source_page for r in recs] == config.TARGET_PAGES
    assert all(r.reaction and r.definition for r in recs)
    assert 16 not in [r.source_page for r in recs]
