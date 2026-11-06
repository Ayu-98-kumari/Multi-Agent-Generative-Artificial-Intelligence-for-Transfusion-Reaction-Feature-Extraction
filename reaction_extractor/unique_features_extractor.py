"""Unique-features extractor (deterministic, no Groq).

Reads the already-built ``reaction_features.json`` and produces a flat,
deduplicated single-column table of every feature value across all reactions:
``time_window`` (one string), each ``signs_symptoms`` item, each ``lab_markers``
item.

Ordering: ``time_window`` values first, then ``signs_symptoms``, then
``lab_markers`` — each collected in source order (reaction p11 -> p31) and
deduplicated as encountered. Deduplication is case-insensitive but the original
casing of the first occurrence is preserved. ``null`` fields are skipped.

This module never modifies ``reaction_features.json``; it only reads it.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from . import config

# Below this many unique features the input was probably empty or malformed.
MIN_EXPECTED_UNIQUE = 20

OUTPUT_KEY = "Unique_features"
UNIQUE_FEATURES_FILENAME = "unique_features.json"


def _iter_field(reactions: list, field: str):
    """Yield feature values for one field across reactions in source order.

    ``time_window`` yields at most one string per reaction; the list fields
    yield each item. ``null`` / non-string / blank values are skipped.
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


def extract_unique_features(document: dict) -> dict:
    """Build the unique-features result dict from a reaction_features document.

    Pure (no I/O). ``document`` is the parsed ``reaction_features.json``.
    Returns ``{"unique": [...], "source_counts": {...}}`` where ``source_counts``
    are the pre-deduplication totals per field.
    """
    reactions = document.get("reactions", [])

    unique: list[str] = []
    seen: set[str] = set()  # lowercased keys -> case-insensitive dedup
    source_counts = {"time_window": 0, "signs_symptoms": 0, "lab_markers": 0}

    # One shared dedup set across all three passes, so a value appearing in more
    # than one field keeps only its earliest (highest-priority) position.
    for field in ("time_window", "signs_symptoms", "lab_markers"):
        for item in _iter_field(reactions, field):
            source_counts[field] += 1
            key = item.lower()
            if key in seen:
                continue
            seen.add(key)
            unique.append(item)  # preserve first-seen original casing

    return {"unique": unique, "source_counts": source_counts}


def build_unique_features(
    input_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
) -> dict:
    """Read reaction_features.json, write unique_features.json, return the report.

    The input file is read-only. Returns a report dict with the written
    ``document`` plus ``source_counts`` for the CLI to render.
    """
    input_path = Path(input_path) if input_path is not None else \
        config.OUTPUT_DIR / config.OUTPUT_FILENAME_DEFAULT
    output_path = Path(output_path) if output_path is not None else \
        config.OUTPUT_DIR / UNIQUE_FEATURES_FILENAME

    source_document = json.loads(input_path.read_text(encoding="utf-8"))
    result = extract_unique_features(source_document)
    unique = result["unique"]

    document = {
        "metadata": {
            "source_filename": input_path.name,
            "generated_on": datetime.now(timezone.utc).isoformat(),
            "total_unique_features": len(unique),
        },
        OUTPUT_KEY: unique,
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
    total = document["metadata"]["total_unique_features"]
    pre_dedup = sum(counts.values())

    lines = [
        f"Wrote {total} unique feature(s) -> {report['output_path']}",
        f"  source (pre-dedup): {pre_dedup} total — "
        f"{counts['time_window']} time_window, "
        f"{counts['signs_symptoms']} signs_symptoms, "
        f"{counts['lab_markers']} lab_markers",
    ]
    if total == 0:
        lines.append("  ⚠ zero unique features — input file empty or malformed?")
    elif total < MIN_EXPECTED_UNIQUE:
        lines.append(f"  ⚠ only {total} unique features (< {MIN_EXPECTED_UNIQUE}) "
                     "— input file may be incomplete.")
    return "\n".join(lines)
