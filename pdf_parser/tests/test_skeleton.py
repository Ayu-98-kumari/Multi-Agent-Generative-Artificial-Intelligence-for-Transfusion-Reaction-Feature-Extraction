"""Tests for Pass 1 (skeleton) against the real protocol PDF.

The sample PDF lives at the project root. If it is absent these tests skip,
so the suite stays runnable in environments without the document.
"""

from __future__ import annotations

import pytest

from pdf_parser import config
from pdf_parser.skeleton import build_skeleton, normalize_ws

PDF_PATH = config.PROJECT_ROOT / "bv-hv-protocol-current.pdf"
pytestmark = pytest.mark.skipif(
    not PDF_PATH.exists(), reason=f"sample PDF not found at {PDF_PATH}"
)


@pytest.fixture(scope="module")
def skeleton():
    return build_skeleton(PDF_PATH)


# --------------------------------------------------------------------------- #
# Document-level facts
# --------------------------------------------------------------------------- #
def test_basic_metadata(skeleton):
    assert skeleton.total_pages == 34
    assert skeleton.version == "v3.0"
    assert "Biovigilance" in (skeleton.title or "")
    assert skeleton.source_filename == "bv-hv-protocol-current.pdf"


# --------------------------------------------------------------------------- #
# Sections
# --------------------------------------------------------------------------- #
def test_sections_and_spans(skeleton):
    spans = {s.section_number: tuple(s.page_span) for s in skeleton.sections}
    # Front matter pages before Section 1, then the 7 numbered sections.
    assert spans[0] == (1, 2)
    assert spans[1] == (3, 5)
    assert spans[2] == (6, 6)
    assert spans[3] == (7, 11)
    assert spans[4] == (12, 16)
    assert spans[5] == (17, 22)
    assert spans[6] == (23, 31)
    assert spans[7] == (32, 34)


def test_sections_cover_every_page_without_overlap(skeleton):
    covered = []
    for s in skeleton.sections:
        covered.extend(range(s.page_span[0], s.page_span[1] + 1))
    assert covered == sorted(covered)              # ordered, no backtracking
    assert covered == list(range(1, skeleton.total_pages + 1))  # gap/overlap-free


# --------------------------------------------------------------------------- #
# Inventory
# --------------------------------------------------------------------------- #
def test_inventory_counts(skeleton):
    tables = skeleton.expected_tables()
    numbered = [t for t in tables if t.number > 0]
    archived = [t for t in tables if t.archived]
    version_history = [t for t in tables if t.label == "Version History"]
    # 8 numbered + 8 archived Section-6 reaction tables + 1 version-history table.
    assert len(numbered) == 8
    assert len(archived) == 8
    assert len(version_history) == 1
    assert len(tables) == 17
    assert len(skeleton.expected_figures()) == 3
    assert [t.number for t in numbered] == list(range(1, 9))
    assert [f.number for f in skeleton.expected_figures()] == [1, 2, 3]


def test_version_history_table_detected(skeleton):
    vh = next(t for t in skeleton.expected_tables() if t.label == "Version History")
    assert vh.page == 32
    assert vh.number == 0
    assert not vh.archived
    assert "Revisions" in vh.caption


def test_archived_reaction_tables(skeleton):
    """The 8 Section-6 reaction tables (no 'Table N.' prefix) are in the
    inventory, one per page 24-31, so the VLM expects them and validation
    counts them."""
    archived = sorted((t for t in skeleton.expected_tables() if t.archived),
                      key=lambda t: t.page)
    assert [t.page for t in archived] == list(range(24, 32))
    labels = {t.page: t.label for t in archived}
    assert labels[24] == "Archived: Allergic reaction"
    assert labels[25] == "Archived: DHTR"
    assert labels[31] == "Archived: TAGVHD"
    # Full reaction-name captions preserved (not the "Adverse Reactions" banner).
    caps = {t.page: t.caption for t in archived}
    assert caps[24] == "Allergic reaction"
    assert "Post transfusion purpura" in caps[29]


def test_inventory_pages_and_captions(skeleton):
    by_label = {i.label: i for i in skeleton.inventory}
    assert by_label["Table 4"].page == 13
    assert "AHTR" in by_label["Table 4"].caption
    assert by_label["Table 8"].page == 19
    assert by_label["Table 8"].caption == "Incident Codes"
    assert by_label["Figure 1"].page == 3
    # Each inventory item appears exactly once (no duplicate labels).
    labels = [i.label for i in skeleton.inventory]
    assert len(labels) == len(set(labels))


# --------------------------------------------------------------------------- #
# Page -> section map
# --------------------------------------------------------------------------- #
def test_page_to_section_map(skeleton):
    p2s = skeleton.page_to_section()
    assert p2s[1] == 0 and p2s[2] == 0      # front matter
    assert p2s[3] == 1                       # Section 1 start
    assert p2s[19] == 5                      # incident codes -> Section 5
    assert p2s[34] == 7                      # last page -> Section 7


# --------------------------------------------------------------------------- #
# Header/footer stripping + near-empty + coverage tokens
# --------------------------------------------------------------------------- #
def test_headers_footers_stripped(skeleton):
    joined = " ".join(p.text for p in skeleton.pages)
    # The "Page N of 34" footer is unambiguously footer-only and never survives.
    # (The date "January 2026" is NOT asserted: it also appears as genuine
    # content on the title page and in the version history.)
    import re
    assert not re.search(r"Page \d+ of 34", joined)
    # The running header is stripped from every content page. The phrase itself
    # legitimately appears once, inside Figure 1's caption on page 3 — so we
    # assert it is NOT repeated page after page, not that it never appears.
    header_pages = [p.page for p in skeleton.pages
                    if "NHSN Biovigilance Component" in p.text]
    assert header_pages == [3]
    # Real body content is preserved.
    assert "Hemovigilance Module" in joined


def test_near_empty_flagged(skeleton):
    near_empty = [p.page for p in skeleton.pages if p.is_near_empty]
    assert 1 in near_empty                   # title page is sparse


def test_golden_tokens_survive_extraction(skeleton):
    """Anti-omission spot check: key incident codes survive Pass 1 as tokens."""
    joined = " ".join(p.text for p in skeleton.pages)
    for code in ("PC 03", "UT 06", "Babesia"):
        assert code in joined, f"missing {code!r} from Pass-1 text"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def test_normalize_ws():
    assert normalize_ws("a b\n\n c\t d ") == "a b c d"
    assert normalize_ws("soft­hyphen") == "softhyphen"
