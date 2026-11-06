"""Pass 1 — deterministic structure extraction (no LLM).

Why PyMuPDF (``fitz``) and not pdfplumber: Pass 1 needs the TOC, normalized
per-page text, label/caption inventory via regex, a page->section map, *and*
page-image rendering for Pass 2. PyMuPDF does all of this in a single
dependency, including positioned text (``get_text("dict")``) for column-aware
reading order and header/footer stripping by y-position. pdfplumber's strength
is deterministic table extraction, which we deliberately leave to the VLM, so it
would only add a second imaging dependency for no gain here.

Output: a ``Skeleton`` (written to ``skeleton.json``) — the reproducible backbone
and the validation checklist. No network, no model calls.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF

from . import config
from .schema import InventoryItem, PageText, SectionInfo, Skeleton

# Heuristic: a page whose stripped text has fewer than this many non-space
# characters is "near-empty" (still emitted, so coverage stays honest).
NEAR_EMPTY_CHARS = 40

# Version pattern, read from the running header ("... Protocol v3.0").
_VERSION_RE = re.compile(r"\bv(\d+(?:\.\d+)*)\b", re.I)


@dataclass
class _Line:
    text: str
    x0: float
    y0: float


# --------------------------------------------------------------------------- #
# Text extraction helpers
# --------------------------------------------------------------------------- #
def normalize_ws(text: str) -> str:
    """Normalize unicode + collapse whitespace (the canonical form for coverage)."""
    text = unicodedata.normalize("NFKC", text)
    # Common PDF artifacts: non-breaking space, soft hyphen, smart quotes.
    text = text.replace(" ", " ").replace("­", "")
    return re.sub(r"\s+", " ", text).strip()


def _is_header_footer(line: _Line, page_height: float) -> bool:
    """True if a line is a running header/footer (pattern + y-band guard)."""
    txt = line.text.strip()
    matches_pattern = any(p.match(txt) for p in config.HEADER_FOOTER_PATTERNS)
    if not matches_pattern:
        return False
    in_band = (
        line.y0 <= page_height * config.HEADER_BAND_FRAC
        or line.y0 >= page_height * config.FOOTER_BAND_FRAC
    )
    return in_band


def _page_lines(page: "fitz.Page") -> list[_Line]:
    """Extract text lines with positions, dropping empty ones."""
    out: list[_Line] = []
    data = page.get_text("dict")
    for block in data.get("blocks", []):
        for line in block.get("lines", []):
            txt = "".join(span["text"] for span in line.get("spans", [])).strip()
            if not txt:
                continue
            x0, y0, _, _ = line["bbox"]
            out.append(_Line(text=txt, x0=x0, y0=y0))
    return out


def _detect_two_column(lines: list[_Line], page_width: float) -> bool:
    """Two-column when both sides of the midline are well populated and the
    right-side lines start clearly past the gutter (parallel columns)."""
    mid = page_width / 2
    left = [l for l in lines if l.x0 < mid]
    right = [l for l in lines if l.x0 >= mid]
    if (
        len(left) < config.TWO_COLUMN_MIN_LINES_PER_SIDE
        or len(right) < config.TWO_COLUMN_MIN_LINES_PER_SIDE
    ):
        return False
    gutter = page_width * config.TWO_COLUMN_RIGHT_GUTTER_FRAC
    return all(l.x0 >= gutter for l in right)


def _ordered_text(lines: list[_Line]) -> str:
    """Reading-order text for the coverage baseline: a stable ``(y, x)`` sort.

    We deliberately do NOT split into columns here. A left-then-right split
    scrambles *wide tables* (which look two-column but reflow across the page) —
    e.g. it once broke "within 12 hours of cessation" into "in 12 ... hours of
    cessation". A ``(y, x)`` sort keeps row-adjacent text together, so cell text
    and golden-fact phrases stay intact. On genuinely two-column pages (incident
    codes) each entry is a self-contained line, so ``(y, x)`` interleaves whole
    entries without ever breaking or omitting one. True column reading order for
    those pages is reconstructed by the VLM in Pass 2 from the page image."""
    ordered = sorted(lines, key=lambda l: (round(l.y0), l.x0))
    return normalize_ws(" ".join(l.text for l in ordered))


# --------------------------------------------------------------------------- #
# Structure: sections, inventory
# --------------------------------------------------------------------------- #
def _toc_section_starts(doc: "fitz.Document") -> list[tuple[int, str, int]]:
    """Section starts from the TOC: (section_number, title, start_page).

    Falls back to a body-text scan if the TOC has no Section entries (we do not
    assume a tagged structure tree — only the bookmark outline, then text)."""
    starts: list[tuple[int, str, int]] = []
    for _level, title, page in doc.get_toc():
        m = config.SECTION_RE.match(title.strip())
        if m:
            starts.append((int(m.group(1)), (m.group(2) or "").strip(), page))
    if starts:
        return starts

    # Fallback: scan each page's lines for a "Section N." heading.
    for pi in range(doc.page_count):
        for line in _page_lines(doc[pi]):
            m = config.SECTION_RE.match(line.text)
            if m:
                starts.append((int(m.group(1)), (m.group(2) or "").strip(), pi + 1))
                break
    return starts


def _build_sections(
    doc: "fitz.Document", starts: list[tuple[int, str, int]]
) -> list[SectionInfo]:
    """Turn section starts into spans; prepend front matter if pages precede S1."""
    starts = sorted(starts, key=lambda s: s[2])
    total = doc.page_count
    sections: list[SectionInfo] = []

    if not starts:
        # Degenerate doc with no detectable sections: one span over everything.
        return [SectionInfo(section_number=0, title="Document", page_span=[1, total])]

    first_start = starts[0][2]
    if first_start > 1:
        sections.append(
            SectionInfo(section_number=0, title="Front Matter",
                        page_span=[1, first_start - 1])
        )

    for idx, (num, title, start) in enumerate(starts):
        end = (starts[idx + 1][2] - 1) if idx + 1 < len(starts) else total
        sections.append(
            SectionInfo(section_number=num, title=title, page_span=[start, end])
        )
    return sections


def _body_lines(page: "fitz.Page") -> list[_Line]:
    """Page lines with running headers/footers removed."""
    return [l for l in _page_lines(page)
            if not _is_header_footer(l, page.rect.height)]


def _detect_archived_table(
    body_lines: list[_Line], page_number: int
) -> Optional[InventoryItem]:
    """Section-6 archived reaction tables have no "Table N." prefix. Fingerprint
    them by the shared Case Definition / Severity / Imputability column triplet,
    then take the topmost body line as the reaction-name caption."""
    page_text = " ".join(l.text for l in body_lines)
    if not all(token in page_text for token in config.ARCHIVED_TABLE_TRIPLET):
        return None

    # The reaction name is the heading immediately above the table. Anchor on the
    # "Note:" line if present, else the "Case Definition" column header. Taking
    # the body line just above the anchor skips any group banner (e.g. page 24
    # carries an "Adverse Reactions" banner above "Allergic reaction").
    def _first_y(predicate) -> Optional[float]:
        ys = [l.y0 for l in body_lines if predicate(l.text)]
        return min(ys) if ys else None

    anchor_y = _first_y(lambda t: t.startswith("Note")) or _first_y(
        lambda t: t.startswith("Case Definition")
    )
    above = [l for l in body_lines if anchor_y is None or l.y0 < anchor_y]
    heading_line = max(above, key=lambda l: l.y0) if above else min(
        body_lines, key=lambda l: (round(l.y0), l.x0)
    )
    heading = heading_line.text.strip()
    m = config.ARCHIVED_ACRONYM_RE.search(heading)
    acronym = m.group(1).strip() if m else heading
    return InventoryItem(
        kind="table",
        label=f"Archived: {acronym}",
        number=0,
        caption=heading,
        page=page_number,
        archived=True,
    )


def _detect_version_history(
    body_lines: list[_Line], page_number: int
) -> Optional[InventoryItem]:
    """The Section-7 version-history table has no 'Table N.' label; fingerprint it
    by its column headers (Version / Release Date / Summary of Revisions)."""
    page_text = " ".join(l.text for l in body_lines)
    if not all(token in page_text for token in config.VERSION_HISTORY_HEADER):
        return None
    return InventoryItem(
        kind="table", number=0, label="Version History",
        caption="Protocol Summary of Revisions", page=page_number, archived=False)


def _build_inventory(doc: "fitz.Document") -> list[InventoryItem]:
    """Find every expected table/figure for the validation checklist:

    * numbered ``Table N.`` / ``Figure N.`` labels (first body occurrence),
    * archived Section-6 reaction tables (detected structurally, no number), and
    * the Section-7 version-history table (detected by header fingerprint)."""
    seen: dict[str, InventoryItem] = {}
    # Keyed by label so a multi-page detected table (e.g. version history, whose
    # header repeats on its continuation pages) is inventoried once, at its start.
    detected: dict[str, InventoryItem] = {}

    for pi in range(doc.page_count):
        body = _body_lines(doc[pi])
        page_has_numbered_table = False
        for line in body:
            for kind, rx in (("table", config.TABLE_RE), ("figure", config.FIGURE_RE)):
                m = rx.match(line.text)
                if not m:
                    continue
                if kind == "table":
                    page_has_numbered_table = True
                label = f"{kind.capitalize()} {int(m.group(1))}"
                if label in seen:
                    continue
                seen[label] = InventoryItem(
                    kind=kind,
                    number=int(m.group(1)),
                    label=label,
                    caption=(m.group(2) or "").strip().rstrip("."),
                    page=pi + 1,
                )
        # A page with no "Table N." label of its own may still be an unlabeled
        # table we detect by fingerprint (archived reaction table, or the
        # version-history table).
        if not page_has_numbered_table and body:
            item = (_detect_archived_table(body, pi + 1)
                    or _detect_version_history(body, pi + 1))
            if item is not None and item.label not in detected:
                detected[item.label] = item

    # Stable order by (kind, page): numbered tables come in page == number order,
    # detected unlabeled tables follow on their later pages; figures by page.
    return sorted([*seen.values(), *detected.values()],
                  key=lambda i: (i.kind, i.page))


def _page_to_section(page: int, sections: list[SectionInfo]) -> int:
    for s in sections:
        if s.page_span[0] <= page <= s.page_span[1]:
            return s.section_number
    return sections[-1].section_number if sections else 0


def _extract_version(doc: "fitz.Document") -> Optional[str]:
    """Read the version from the running header of the first content page."""
    for pi in range(min(doc.page_count, 5)):
        for line in _page_lines(doc[pi]):
            m = _VERSION_RE.search(line.text)
            if m and "protocol" in line.text.lower():
                return f"v{m.group(1)}"
    return None


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def build_skeleton(pdf_path: str | Path) -> Skeleton:
    """Extract the deterministic skeleton from a PDF. No LLM, no network."""
    pdf_path = Path(pdf_path)
    doc = fitz.open(pdf_path)
    try:
        starts = _toc_section_starts(doc)
        sections = _build_sections(doc, starts)
        inventory = _build_inventory(doc)

        pages: list[PageText] = []
        for pi in range(doc.page_count):
            page = doc[pi]
            lines = _body_lines(page)
            two_col = _detect_two_column(lines, page.rect.width)  # advisory hint only
            text = _ordered_text(lines)
            pages.append(
                PageText(
                    page=pi + 1,
                    section_number=_page_to_section(pi + 1, sections),
                    text=text,
                    is_near_empty=len(text.replace(" ", "")) < NEAR_EMPTY_CHARS,
                    two_column=two_col,
                )
            )

        return Skeleton(
            source_filename=pdf_path.name,
            title=(doc.metadata or {}).get("title") or None,
            version=_extract_version(doc),
            total_pages=doc.page_count,
            sections=sections,
            inventory=inventory,
            pages=pages,
        )
    finally:
        doc.close()


def write_skeleton(skeleton: Skeleton, out_dir: str | Path) -> Path:
    """Write skeleton.json into out_dir; return its path."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / config.SKELETON_FILENAME
    out_path.write_text(skeleton.model_dump_json(indent=2), encoding="utf-8")
    return out_path
