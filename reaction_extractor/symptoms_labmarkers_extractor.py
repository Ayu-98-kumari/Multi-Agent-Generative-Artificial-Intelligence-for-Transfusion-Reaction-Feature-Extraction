"""Symptoms & lab-markers extractor (deterministic, no Groq).

Reads the already-built ``reaction_features.json`` and produces a flat,
deduplicated single-column table of every ``signs_symptoms`` and ``lab_markers``
value across all reactions. ``time_window`` is completely ignored.

Ordering: all ``signs_symptoms`` items first (reaction p11 -> p31), then all
``lab_markers`` items (p11 -> p31) — each collected in source order and
deduplicated as encountered. Deduplication is case-insensitive but the original
casing of the first occurrence is preserved. ``null`` fields are skipped. A value
appearing in both fields survives once, in the signs_symptoms section (collected
first).

This module never modifies ``reaction_features.json``; it only reads it.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from . import config

# Below this many unique items the input was probably empty or malformed.
MIN_EXPECTED_ITEMS = 10

OUTPUT_KEY = "symptoms_labmarkers"
SYMPTOMS_LABMARKERS_FILENAME = "symptoms_labmarkers.json"


def _iter_field(reactions: list, field: str):
    """Yield feature values for one field across reactions in source order.

    Skips reactions without a dict ``features``, ``null`` fields, and any
    non-string / blank item. The list fields yield each item in turn.
    """
    for reaction in reactions:
        features = reaction.get("features")
        if not isinstance(features, dict):
            continue
        value = features.get(field)
        if value is None:
            continue
        items = value if isinstance(value, list) else [value]
        for item in items:
            if isinstance(item, str) and item.strip():
                yield item


def extract_symptoms_labmarkers(document: dict) -> dict:
    """Build the symptoms/lab-markers result dict from a features document.

    Pure (no I/O). ``document`` is the parsed ``reaction_features.json``.
    ``time_window`` is never read. Returns
    ``{"items": [...], "source_counts": {...}}`` where ``source_counts`` are the
    pre-deduplication totals per field.
    """
    reactions = document.get("reactions", [])

    items: list[str] = []
    seen: set[str] = set()  # lowercased keys -> case-insensitive dedup
    source_counts = {"signs_symptoms": 0, "lab_markers": 0}

    # One shared dedup set across both passes, so a value appearing in both
    # fields keeps only its earliest position — i.e. in the signs_symptoms
    # section, which is collected first.
    for field in ("signs_symptoms", "lab_markers"):
        for item in _iter_field(reactions, field):
            source_counts[field] += 1
            key = item.lower()
            if key in seen:
                continue
            seen.add(key)
            items.append(item)  # preserve first-seen original casing

    return {"items": items, "source_counts": source_counts}


def build_symptoms_labmarkers(
    input_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
) -> dict:
    """Read reaction_features.json, write symptoms_labmarkers.json, return report.

    The input file is read-only. Returns a report dict with the written
    ``document`` plus ``source_counts`` for the CLI to render.
    """
    input_path = Path(input_path) if input_path is not None else \
        config.OUTPUT_DIR / config.OUTPUT_FILENAME_DEFAULT
    output_path = Path(output_path) if output_path is not None else \
        config.OUTPUT_DIR / SYMPTOMS_LABMARKERS_FILENAME

    source_document = json.loads(input_path.read_text(encoding="utf-8"))
    result = extract_symptoms_labmarkers(source_document)
    items = result["items"]

    document = {
        "metadata": {
            "source_filename": input_path.name,
            "generated_on": datetime.now(timezone.utc).isoformat(),
            "total_unique_items": len(items),
        },
        OUTPUT_KEY: items,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "document": document,
        "source_counts": result["source_counts"],
        "output_path": output_path,
    }


def render_report(report: dict) -> str:
    """Render the lightweight validation report as a printable string."""
    document = report["document"]
    counts = report["source_counts"]
    total = document["metadata"]["total_unique_items"]
    items = document[OUTPUT_KEY]
    pre_dedup = sum(counts.values())

    lines = [
        f"Wrote {total} unique item(s) -> {report['output_path']}",
        f"  source (pre-dedup): {pre_dedup} total — "
        f"{counts['signs_symptoms']} signs_symptoms, "
        f"{counts['lab_markers']} lab_markers",
        "  time_window: ignored (never read).",
    ]
    if total == 0:
        lines.append("  ⚠ zero items — input file empty or malformed?")
    elif total < MIN_EXPECTED_ITEMS:
        lines.append(f"  ⚠ only {total} items (< {MIN_EXPECTED_ITEMS}) — "
                     "input file may be incomplete.")
    return "\n".join(lines)
