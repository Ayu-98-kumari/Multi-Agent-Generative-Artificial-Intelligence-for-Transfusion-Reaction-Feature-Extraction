"""Deterministic extraction of Columns 1 & 2 from the parsed-document JSON.

No AI here. This reads the read-only ``pdf_parser`` output JSON and, for each of
the configured target pages, pulls out:

* **Column 1 (Reaction)** — the reaction name, from the table's ``caption`` when
  non-empty, else its ``label`` (with any ``"Archived: "`` prefix stripped).
* **Column 2 (Definition)** — the concatenation of every non-empty cell under the
  ``"Case Definition"`` column, in row order. Severity and Imputability columns
  are never read. Text is preserved verbatim (including AND/OR/EITHER logic).

Selection is strictly by page: a table is a target iff ``page_span[0]`` is in
``config.TARGET_PAGES``. That list already omits page 16 (Table 7).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import config


class ExtractionError(RuntimeError):
    """Raised when the source JSON does not yield exactly the expected tables."""


@dataclass
class ReactionRecord:
    """Columns 1 & 2 for one reaction (Column 3 is filled later by features.py)."""

    reaction: str
    definition: str
    source_page: int


def load_document(path: str | Path) -> dict[str, Any]:
    """Load the parser's output JSON (read-only)."""
    path = Path(path)
    if not path.exists():
        raise ExtractionError(f"Input JSON not found: {path}")
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def _iter_tables(doc: dict[str, Any]):
    """Yield every table content block across all sections."""
    for section in doc.get("sections", []):
        for block in section.get("content", []):
            if block.get("type") == "table":
                yield block


def _table_start_page(table: dict[str, Any]) -> int | None:
    span = table.get("page_span")
    if isinstance(span, list) and span:
        return span[0]
    return None


def reaction_name(table: dict[str, Any]) -> str:
    """Column 1: caption when non-empty, else label (minus the archived prefix)."""
    caption = (table.get("caption") or "").strip()
    if caption:
        return caption
    label = (table.get("label") or "").strip()
    if label.startswith(config.ARCHIVED_LABEL_PREFIX):
        label = label[len(config.ARCHIVED_LABEL_PREFIX):].strip()
    return label


def case_definition(table: dict[str, Any]) -> str:
    """Column 2: non-empty "Case Definition" cells, in row order, joined by blank lines.

    Faithful concatenation — no semantic filtering. Whatever the parser placed
    under the Case-Definition column is preserved verbatim; empty cells are
    skipped so the joined text has no dangling separators.
    """
    parts: list[str] = []
    for row in table.get("rows", []):
        cells = row.get("cells", {})
        cell = (cells.get(config.CASE_DEFINITION_COLUMN) or "").strip()
        if cell:
            parts.append(cell)
    return "\n\n".join(parts)


def select_target_tables(doc: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the target table blocks in TARGET_PAGES order, one per page.

    Raises ExtractionError if any target page is missing or has more than one
    table, so a silent omission or duplicate can never slip through.
    """
    by_page: dict[int, list[dict[str, Any]]] = {}
    for table in _iter_tables(doc):
        start = _table_start_page(table)
        if start in config.TARGET_PAGES:
            by_page.setdefault(start, []).append(table)

    ordered: list[dict[str, Any]] = []
    problems: list[str] = []
    for page in config.TARGET_PAGES:
        found = by_page.get(page, [])
        if len(found) == 0:
            problems.append(f"page {page}: no table found")
        elif len(found) > 1:
            labels = ", ".join(repr(t.get("label", "")) for t in found)
            problems.append(f"page {page}: {len(found)} tables found ({labels})")
        else:
            ordered.append(found[0])
    if problems:
        raise ExtractionError(
            "target-table selection failed:\n  " + "\n  ".join(problems)
        )
    return ordered


def extract_records(doc: dict[str, Any]) -> list[ReactionRecord]:
    """Build the 12 deterministic (Reaction, Definition, source_page) records."""
    records: list[ReactionRecord] = []
    for table in select_target_tables(doc):
        page = _table_start_page(table)
        name = reaction_name(table)
        definition = case_definition(table)
        records.append(
            ReactionRecord(reaction=name, definition=definition, source_page=page)
        )
    return records
