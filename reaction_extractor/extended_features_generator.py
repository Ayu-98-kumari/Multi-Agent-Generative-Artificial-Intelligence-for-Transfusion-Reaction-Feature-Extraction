"""Extended-features generator (deterministic, no Groq).

Reads the already-built ``unique_features.json`` and produces, for each feature,
an interleaved pair: the original feature verbatim, immediately followed by a
negated version ``"Evidence for absence of " + feature.lower()``.

The output list is therefore exactly twice the length of the input, laid out as
``[orig0, neg0, orig1, neg1, ...]``.

This module never modifies ``unique_features.json``; it only reads it.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from . import config

# Exact, invariant negation prefix (trailing space intentional).
ABSENCE_PREFIX = "Evidence for absence of "

INPUT_KEY = "Unique_features"
OUTPUT_KEY = "Extended_features"
EXTENDED_FEATURES_FILENAME = "extended_features.json"
UNIQUE_FEATURES_FILENAME = "unique_features.json"


def negate(feature: str) -> str:
    """Return the negated form: prefix + the feature lowercased."""
    return f"{ABSENCE_PREFIX}{feature.lower()}"


def extend_features(unique: list[str]) -> list[str]:
    """Interleave each feature with its negated pair (pure, no I/O)."""
    extended: list[str] = []
    for feature in unique:
        extended.append(feature)       # original, verbatim
        extended.append(negate(feature))
    return extended


def build_extended_features(
    input_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
) -> dict:
    """Read unique_features.json, write extended_features.json, return the report.

    The input file is read-only. Returns a report dict with the written
    ``document`` and the ``source_count`` for the CLI to render.
    """
    input_path = Path(input_path) if input_path is not None else \
        config.OUTPUT_DIR / UNIQUE_FEATURES_FILENAME
    output_path = Path(output_path) if output_path is not None else \
        config.OUTPUT_DIR / EXTENDED_FEATURES_FILENAME

    source_document = json.loads(input_path.read_text(encoding="utf-8"))
    unique = source_document.get(INPUT_KEY, [])
    extended = extend_features(unique)

    document = {
        "metadata": {
            "source_filename": input_path.name,
            "generated_on": datetime.now(timezone.utc).isoformat(),
            "total_features": len(extended),
        },
        OUTPUT_KEY: extended,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "document": document,
        "source_count": len(unique),
        "output_path": output_path,
    }


def _interleaving_ok(unique: list[str], extended: list[str]) -> bool:
    """True iff even indices are originals and odd indices their negated pairs."""
    if len(extended) != 2 * len(unique):
        return False
    for i, feature in enumerate(unique):
        if extended[2 * i] != feature or extended[2 * i + 1] != negate(feature):
            return False
    return True


def render_report(report: dict) -> str:
    """Render the lightweight validation report as a printable string."""
    document = report["document"]
    source_count = report["source_count"]
    total = document["metadata"]["total_features"]
    extended = document[OUTPUT_KEY]

    lines = [
        f"Wrote {total} extended feature(s) -> {report['output_path']}",
        f"  source: {source_count} unique -> expected {2 * source_count}",
    ]
    if total != 2 * source_count:
        lines.append(f"  ⚠ total {total} is not exactly 2 × {source_count} "
                     f"({2 * source_count}) — source may be malformed.")
    # Reconstruct the source list from even indices to verify interleaving.
    unique = extended[0::2]
    if _interleaving_ok(unique, extended):
        lines.append("  interleaving OK — even = original, odd = negated pair.")
    else:
        lines.append("  ⚠ interleaving check FAILED.")
    return "\n".join(lines)
