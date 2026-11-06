"""Deterministic validation gate — runs after stitching, before acceptance.

Layered, ordered checks. Hard failures (1-4) block acceptance; golden-fact
failures (5) are flagged for human review. The report names the failing check
and the section/page so issues are locatable.

  1. Schema conformance        — output validates against the Pydantic models.
  2. Inventory completeness    — every skeleton table (16: 8 numbered + 8
                                 archived) and figure (3) appears exactly once
                                 after stitching. Matched by page, so it does NOT
                                 only check numbered tables.
  3. Coverage round-trip       — token-level, ORDER-INDEPENDENT recall of the
                                 Pass-1 baseline (anti-silent-omission net).
  4. Structural invariants     — table shape, parent section, page-in-span,
                                 reading order, non-empty figure descriptions.
  5. Golden facts              — specific clinical values survived (flag only).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from . import config
from .schema import Document, FigureBlock, ProseBlock, Skeleton, TableBlock

HARD = "hard"
FLAG = "flag"

_TOKEN_RE = re.compile(r"[a-z0-9]+")


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
@dataclass
class ValidationIssue:
    check: str
    severity: str            # HARD | FLAG
    message: str
    location: str = ""

    def __str__(self) -> str:
        loc = f" [{self.location}]" if self.location else ""
        return f"{self.severity.upper():4} {self.check}{loc}: {self.message}"


@dataclass
class ValidationReport:
    issues: list[ValidationIssue] = field(default_factory=list)
    coverage: float | None = None

    @property
    def hard_failures(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == HARD]

    @property
    def flags(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == FLAG]

    @property
    def ok(self) -> bool:
        return not self.hard_failures

    def render(self) -> str:
        lines = [
            "VALIDATION REPORT",
            f"  result   : {'PASS' if self.ok else 'FAIL'}",
            f"  coverage : {self.coverage:.4f}" if self.coverage is not None
            else "  coverage : n/a",
            f"  hard     : {len(self.hard_failures)}   flags: {len(self.flags)}",
        ]
        if self.issues:
            lines.append("  issues:")
            lines += [f"    - {i}" for i in self.issues]
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Text / token helpers
# --------------------------------------------------------------------------- #
def _block_text(block) -> str:
    if isinstance(block, ProseBlock):
        return " ".join(filter(None, [block.subheading, block.text]))
    if isinstance(block, TableBlock):
        parts = [block.label, block.caption, *block.columns, *block.notes,
                 *block.footnotes]
        for row in block.rows:
            parts.extend(row.cells.values())
        return " ".join(filter(None, parts))
    if isinstance(block, FigureBlock):
        return " ".join(filter(None, [block.label, block.caption,
                                      block.description, block.abbreviations]))
    return ""


def _tokens(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(text.lower()))


def _document_tokens(document: Document) -> set[str]:
    toks: set[str] = set()
    for section in document.sections:
        for block in section.content:
            toks |= _tokens(_block_text(block))
    return toks


def _block_start_page(block) -> int:
    return block.page_span[0] if isinstance(block, TableBlock) else block.page


def _block_end_page(block) -> int:
    return block.page_span[1] if isinstance(block, TableBlock) else block.page


# --------------------------------------------------------------------------- #
# 1. Schema conformance
# --------------------------------------------------------------------------- #
def check_schema(document: Document) -> list[ValidationIssue]:
    try:
        Document.model_validate(document.model_dump(mode="json"))
    except Exception as exc:  # pydantic ValidationError
        return [ValidationIssue("schema", HARD, f"document fails schema: {exc}")]
    return []


# --------------------------------------------------------------------------- #
# 2. Inventory completeness  (matched by page, covers archived tables)
# --------------------------------------------------------------------------- #
def check_inventory(document: Document, skeleton: Skeleton) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    p2s = skeleton.page_to_section()

    doc_tables = [b for s in document.sections for b in s.content
                  if isinstance(b, TableBlock)]
    doc_figures = [b for s in document.sections for b in s.content
                   if isinstance(b, FigureBlock)]

    expected_table_pages = {i.page for i in skeleton.expected_tables()}
    expected_figure_pages = {i.page for i in skeleton.expected_figures()}

    # Every expected table/figure present exactly once (matched by start page).
    for item in skeleton.expected_tables():
        n = sum(1 for t in doc_tables if t.page_span[0] == item.page)
        loc = f"section {p2s.get(item.page, '?')} / page {item.page}"
        if n == 0:
            issues.append(ValidationIssue(
                "inventory", HARD, f"missing table {item.label!r}", loc))
        elif n > 1:
            issues.append(ValidationIssue(
                "inventory", HARD,
                f"table {item.label!r} appears {n}x (expected 1)", loc))

    for item in skeleton.expected_figures():
        n = sum(1 for f in doc_figures if f.page == item.page)
        loc = f"section {p2s.get(item.page, '?')} / page {item.page}"
        if n == 0:
            issues.append(ValidationIssue(
                "inventory", HARD, f"missing figure {item.label!r}", loc))
        elif n > 1:
            issues.append(ValidationIssue(
                "inventory", HARD,
                f"figure {item.label!r} appears {n}x (expected 1)", loc))

    # Tables/figures not anchored to any expected page often signal a stitch
    # failure (an unmerged continuation fragment).
    for t in doc_tables:
        if t.page_span[0] not in expected_table_pages:
            issues.append(ValidationIssue(
                "inventory", HARD,
                f"unexpected table {t.label!r} starting at page {t.page_span[0]} "
                f"(possible unstitched fragment)",
                f"page {t.page_span[0]}"))
    for f in doc_figures:
        if f.page not in expected_figure_pages:
            issues.append(ValidationIssue(
                "inventory", HARD,
                f"unexpected figure {f.label!r} at page {f.page}",
                f"page {f.page}"))

    return issues


# --------------------------------------------------------------------------- #
# 3. Coverage round-trip  (token-level, order-independent)
# --------------------------------------------------------------------------- #
def check_coverage(
    document: Document, skeleton: Skeleton, threshold: float
) -> tuple[list[ValidationIssue], float]:
    # NOTE: the baseline is the Pass-1 extractor's text, NOT absolute ground
    # truth. We measure ORDER-INDEPENDENT token recall on purpose: dense table
    # pages interleave cells across columns, so any order-sensitive comparison
    # would raise false omission alarms. We only ask: does each baseline token
    # reappear somewhere in the final JSON?
    json_tokens = _document_tokens(document)

    baseline_tokens: set[str] = set()
    for page in skeleton.pages:
        baseline_tokens |= _tokens(page.text)

    if not baseline_tokens:
        return [], 1.0

    recall = len(baseline_tokens & json_tokens) / len(baseline_tokens)
    issues: list[ValidationIssue] = []
    if recall < threshold:
        # Localize: name the pages whose tokens are least covered.
        worst = []
        for page in skeleton.pages:
            pt = _tokens(page.text)
            if not pt:
                continue
            page_recall = len(pt & json_tokens) / len(pt)
            if page_recall < threshold:
                worst.append((page.page, page_recall))
        worst.sort(key=lambda x: x[1])
        detail = ", ".join(f"p{pg}={r:.2f}" for pg, r in worst[:8])
        issues.append(ValidationIssue(
            "coverage", HARD,
            f"token recall {recall:.4f} < {threshold} "
            f"(worst pages: {detail or 'n/a'})"))
    return issues, recall


# --------------------------------------------------------------------------- #
# 4. Structural invariants
# --------------------------------------------------------------------------- #
def check_structure(document: Document, skeleton: Skeleton) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    p2s = skeleton.page_to_section()

    for section in document.sections:
        lo, hi = section.page_span
        prev_start = 0
        for block in section.content:
            start, end = _block_start_page(block), _block_end_page(block)
            loc = f"section {section.section_number} / page {start}"

            # Page(s) inside the section span.
            if not (lo <= start <= hi and lo <= end <= hi):
                issues.append(ValidationIssue(
                    "structure", HARD,
                    f"{type(block).__name__} pages [{start},{end}] outside "
                    f"section span [{lo},{hi}]", loc))

            # Block placed under the section its (start) page maps to.
            expected_section = p2s.get(start)
            if expected_section is not None and expected_section != section.section_number:
                issues.append(ValidationIssue(
                    "structure", HARD,
                    f"block at page {start} placed in section "
                    f"{section.section_number} but maps to section "
                    f"{expected_section}", loc))

            # Reading order: non-decreasing start pages within a section.
            if start < prev_start:
                issues.append(ValidationIssue(
                    "structure", HARD,
                    f"reading order broken: page {start} after page {prev_start}",
                    loc))
            prev_start = max(prev_start, start)

            # Per-type invariants.
            if isinstance(block, TableBlock):
                issues += _check_table(block, loc)
            elif isinstance(block, FigureBlock):
                if not (block.description or "").strip():
                    issues.append(ValidationIssue(
                        "structure", HARD,
                        f"figure {block.label!r} has empty description", loc))
    return issues


def _check_table(block: TableBlock, loc: str) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    if not block.columns:
        issues.append(ValidationIssue(
            "structure", HARD, f"table {block.label!r} has no columns", loc))
    if len(block.rows) < 1:
        issues.append(ValidationIssue(
            "structure", HARD, f"table {block.label!r} has no rows", loc))
    cols = set(block.columns)
    for idx, row in enumerate(block.rows):
        stray = set(row.cells) - cols
        if stray:
            issues.append(ValidationIssue(
                "structure", HARD,
                f"table {block.label!r} row {idx} has columns not in header: "
                f"{sorted(stray)}", loc))
    return issues


# --------------------------------------------------------------------------- #
# 5. Golden facts  (flag only)
# --------------------------------------------------------------------------- #
def _scoped_table_text(document: Document, page: int) -> str | None:
    """Text of the table covering ``page`` (handles the stitched Table 8 span);
    falls back to a figure on that page. None if nothing is found there."""
    for section in document.sections:
        for block in section.content:
            if isinstance(block, TableBlock) and \
                    block.page_span[0] <= page <= block.page_span[1]:
                return _block_text(block)
    for section in document.sections:
        for block in section.content:
            if isinstance(block, FigureBlock) and block.page == page:
                return _block_text(block)
    return None


def check_golden_facts(document: Document) -> list[ValidationIssue]:
    from .tests.golden_facts import GOLDEN_FACTS, normalize_for_golden

    issues: list[ValidationIssue] = []
    for fact in GOLDEN_FACTS:
        loc = f"page {fact.page}"
        text = _scoped_table_text(document, fact.page)
        if text is None:
            issues.append(ValidationIssue(
                "golden", FLAG,
                f"{fact.name}: no table found at page {fact.page} to verify — "
                f"{fact.description}", loc))
        elif not fact.check(normalize_for_golden(text)):
            issues.append(ValidationIssue(
                "golden", FLAG, f"{fact.name}: {fact.description}", loc))
    return issues


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def validate(
    document: Document,
    skeleton: Skeleton,
    coverage_threshold: float = config.COVERAGE_THRESHOLD,
) -> ValidationReport:
    report = ValidationReport()
    report.issues += check_schema(document)
    report.issues += check_inventory(document, skeleton)
    cov_issues, recall = check_coverage(document, skeleton, coverage_threshold)
    report.issues += cov_issues
    report.coverage = recall
    report.issues += check_structure(document, skeleton)
    report.issues += check_golden_facts(document)
    return report
