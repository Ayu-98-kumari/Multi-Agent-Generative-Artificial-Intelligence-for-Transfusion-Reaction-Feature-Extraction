"""Tests for the deterministic validation gate.

Uses a small synthetic skeleton + matching document (no PDF, no network) so each
check can be exercised in isolation, including the archived-table completeness
case that the Section-6 fix exists for.
"""

from __future__ import annotations

import copy

import pytest

from pdf_parser.schema import (
    Document, FigureBlock, InventoryItem, PageText, ProseBlock, Section,
    SectionInfo, Skeleton, TableBlock, TableRow,
)
from pdf_parser.validation import HARD, FLAG, check_golden_facts, validate


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture
def skeleton():
    return Skeleton(
        source_filename="synthetic.pdf",
        title="Synthetic", version="v1.0", total_pages=3,
        sections=[
            SectionInfo(section_number=1, title="S1", page_span=[1, 2]),
            SectionInfo(section_number=6, title="S6", page_span=[3, 3]),
        ],
        inventory=[
            InventoryItem(kind="table", number=1, label="Table 1",
                          caption="Hemovigilance Module Reports", page=1),
            InventoryItem(kind="figure", number=1, label="Figure 1",
                          caption="Surveillance", page=2),
            InventoryItem(kind="table", number=0,
                          label="Archived: DHTR",
                          caption="Delayed hemolytic transfusion reaction (DHTR)",
                          page=3, archived=True),
        ],
        pages=[
            PageText(page=1, section_number=1,
                     text="table 1 hemovigilance module reports code description aa bb"),
            PageText(page=2, section_number=1,
                     text="figure 1 surveillance flowchart purpose overview"),
            PageText(page=3, section_number=6,
                     text="delayed hemolytic transfusion reaction dhtr case "
                          "definition severity imputability antibody 24 hours 28 days"),
        ],
    )


@pytest.fixture
def document():
    return Document(
        title="Synthetic", version="v1.0", source_filename="synthetic.pdf",
        total_pages=3,
        sections=[
            Section(section_number=1, title="S1", page_span=[1, 2], content=[
                TableBlock(label="Table 1", caption="Hemovigilance Module Reports",
                           page_span=[1, 1], columns=["Code", "Description"],
                           rows=[{"cells": {"Code": "aa", "Description": "bb"}}]),
                FigureBlock(label="Figure 1", caption="Surveillance",
                            description="flowchart purpose overview", page=2),
            ]),
            Section(section_number=6, title="S6", page_span=[3, 3], content=[
                TableBlock(label="Delayed hemolytic transfusion reaction (DHTR)",
                           caption="Delayed hemolytic transfusion reaction (DHTR)",
                           page_span=[3, 3],
                           columns=["Case Definition", "Severity", "Imputability"],
                           rows=[{"cells": {
                               "Case Definition": "antibody 24 hours 28 days",
                               "Severity": "severe", "Imputability": "definite"}}]),
            ]),
        ],
    )


def _codes(issues):
    return {(i.check, i.severity) for i in issues}


# --------------------------------------------------------------------------- #
# Happy path
# --------------------------------------------------------------------------- #
def test_valid_document_passes_hard_checks(document, skeleton):
    report = validate(document, skeleton)
    assert report.ok, report.render()
    assert report.coverage == pytest.approx(1.0)
    # No hard failures; golden facts may flag (those clinical strings aren't in
    # this synthetic doc) but flags never block.
    assert report.hard_failures == []


# --------------------------------------------------------------------------- #
# 2. Inventory completeness — including the archived table
# --------------------------------------------------------------------------- #
def test_missing_archived_table_fails(document, skeleton):
    doc = copy.deepcopy(document)
    doc.sections[1].content = []          # drop the DHTR archived table
    report = validate(doc, skeleton)
    assert not report.ok
    assert any(i.check == "inventory" and "DHTR" in i.message
               for i in report.hard_failures)


def test_duplicate_table_fails(document, skeleton):
    doc = copy.deepcopy(document)
    dup = copy.deepcopy(doc.sections[0].content[0])   # second "Table 1" at page 1
    doc.sections[0].content.append(dup)
    report = validate(doc, skeleton)
    assert any(i.check == "inventory" and "appears 2x" in i.message
               for i in report.hard_failures)


def test_unstitched_fragment_flagged_as_unexpected(document, skeleton):
    """A Table-8-style fragment that failed to stitch (extra table on an
    unexpected page) is caught as a hard inventory failure."""
    doc = copy.deepcopy(document)
    frag = TableBlock(label="Table 8", page_span=[2, 2], columns=["x"],
                      rows=[{"cells": {"x": "y"}}])
    doc.sections[0].content.append(frag)
    report = validate(doc, skeleton)
    assert any(i.check == "inventory" and "unexpected table" in i.message
               for i in report.hard_failures)


# --------------------------------------------------------------------------- #
# 3. Coverage round-trip
# --------------------------------------------------------------------------- #
def test_missing_content_fails_coverage(document, skeleton):
    doc = copy.deepcopy(document)
    # Gut page-3 table cells -> many baseline tokens vanish from the JSON.
    doc.sections[1].content[0].rows = [TableRow(cells={
        "Case Definition": "", "Severity": "", "Imputability": ""})]
    report = validate(doc, skeleton)
    assert report.coverage < 0.95
    assert any(i.check == "coverage" for i in report.hard_failures)
    # The failing page is named in the message.
    assert any("p3" in i.message for i in report.hard_failures if i.check == "coverage")


def test_coverage_is_order_independent(document, skeleton):
    """Reordering cell text must not change coverage (token-level recall)."""
    doc = copy.deepcopy(document)
    cell = doc.sections[1].content[0].rows[0].cells
    cell["Case Definition"] = "days 28 hours 24 antibody"   # same tokens, shuffled
    report = validate(doc, skeleton)
    assert report.coverage == pytest.approx(1.0)
    assert report.ok


# --------------------------------------------------------------------------- #
# 4. Structural invariants
# --------------------------------------------------------------------------- #
def test_table_with_no_rows_fails(document, skeleton):
    doc = copy.deepcopy(document)
    doc.sections[0].content[0].rows = []
    report = validate(doc, skeleton)
    assert any(i.check == "structure" and "no rows" in i.message
               for i in report.hard_failures)


def test_empty_figure_description_fails(document, skeleton):
    doc = copy.deepcopy(document)
    doc.sections[0].content[1].description = "   "
    report = validate(doc, skeleton)
    assert any(i.check == "structure" and "empty description" in i.message
               for i in report.hard_failures)


def test_page_outside_section_span_fails(document, skeleton):
    doc = copy.deepcopy(document)
    doc.sections[0].content[1].page = 9      # figure now off-section
    report = validate(doc, skeleton)
    assert any(i.check == "structure" and "outside section span" in i.message
               for i in report.hard_failures)


def test_row_with_stray_column_fails(document, skeleton):
    doc = copy.deepcopy(document)
    doc.sections[0].content[0].rows[0].cells["Ghost"] = "x"
    report = validate(doc, skeleton)
    assert any(i.check == "structure" and "not in header" in i.message
               for i in report.hard_failures)


# --------------------------------------------------------------------------- #
# 5. Golden facts (flag only)
# --------------------------------------------------------------------------- #
def test_golden_facts_flag_but_do_not_block(document, skeleton):
    report = validate(document, skeleton)
    # The synthetic doc has no tables on the golden-fact pages -> flags, never
    # hard failures.
    assert report.flags
    assert all(i.severity == FLAG for i in report.flags)
    assert report.ok


# -- a document satisfying every (table-scoped) golden fact ----------------- #
def _tbl(label, span, text, notes=None):
    return TableBlock(label=label, page_span=span, columns=["c"],
                      rows=[TableRow(cells={"c": text})], notes=notes or [])


@pytest.fixture
def golden_doc():
    tables = [
        _tbl("Table 2", [8, 8], "transfusion-transmitted infection pathogens",
             notes=["the 14 states considered babesia endemic or contiguous"]),
        _tbl("Table 4", [13, 13], "ahtr occurs within 24 hours of cessation; "
             "positive direct antiglobulin test (dat) for anti-igg"),
        _tbl("Table 5", [14, 14], "taco within 12 hours; elevated brain "
             "natriuretic peptide (bnp)"),
        _tbl("Table 6", [15, 15], "trali within 6 hours; pao2/fio2 less than or "
             "equal to 300 mm hg"),
        _tbl("Table 8", [19, 22], "pc 03 detail; ut 06 detail; ms 99; ut 26"),
        _tbl("Delayed hemolytic transfusion reaction (DHTR)", [25, 25],
             "antibodies developed between 24 hours and 28 days after cessation"),
        _tbl("Hypotensive transfusion reaction (HTR)", [28, 28],
             "drop in systolic bp >= 30 mmhg and systolic bp <= 80 mmhg"),
        _tbl("Post transfusion purpura (PTP)", [29, 29],
             "occurs 5-12 days post-transfusion; platelets to less than 20% of baseline"),
        _tbl("Transfusion-associated dyspnea (TAD)", [30, 30],
             "acute respiratory distress occurring within 24 hours of cessation"),
    ]
    return Document(
        source_filename="g.pdf", total_pages=34,
        sections=[Section(section_number=1, title="All", page_span=[1, 34],
                          content=tables)])


def test_all_golden_facts_pass_when_scoped_values_present(golden_doc):
    issues = check_golden_facts(golden_doc)
    assert issues == [], "\n".join(str(i) for i in issues)


def test_table8_tail_dropped_is_flagged(golden_doc):
    """Simulate stitching dropping the last fragment: tail codes vanish."""
    t8 = next(b for b in golden_doc.sections[0].content if b.label == "Table 8")
    t8.rows = [TableRow(cells={"c": "pc 03 detail; ut 06 detail"})]  # no ms 99 / ut 26
    issues = check_golden_facts(golden_doc)
    assert any(i for i in issues if "table8_tail" in i.message)


def test_golden_fact_scoping_distinguishes_dhtr_from_dstr(golden_doc):
    """The DHTR fact (page 25) must NOT be satisfied by DSTR (page 26) carrying
    the same '24 hours and 28 days' phrase."""
    doc = copy.deepcopy(golden_doc)
    # Strip the phrase from DHTR (page 25)...
    dhtr = next(b for b in doc.sections[0].content if b.page_span == [25, 25])
    dhtr.rows = [TableRow(cells={"c": "delayed hemolytic, antibody present"})]
    # ...but add a DSTR table at page 26 that DOES carry it.
    doc.sections[0].content.append(_tbl(
        "Delayed serologic transfusion reaction (DSTR)", [26, 26],
        "antibodies developed between 24 hours and 28 days after cessation"))
    issues = check_golden_facts(doc)
    assert any("dhtr_antibody_window" in i.message for i in issues)
