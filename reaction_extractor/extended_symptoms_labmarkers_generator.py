"""Extended symptoms/lab-markers generator (deterministic, no Groq).

Reads the already-built ``symptoms_labmarkers.json`` and produces, for each
feature, an interleaved pair: the original feature verbatim, immediately followed
by a negated version ``"Evidence for absence of " + feature.lower()``.

The output list is therefore exactly twice the length of the input, laid out as
``[orig0, neg0, orig1, neg1, ...]``. This mirrors ``extended_features.json``.

This module never modifies ``symptoms_labmarkers.json``; it only reads it.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from . import config

# Exact, invariant negation prefix (trailing space intentional).
ABSENCE_PREFIX = "Evidence for absence of "

INPUT_KEY = "symptoms_labmarkers"
OUTPUT_KEY = "extended_symptoms_labmarkers"
EXTENDED_SYMPTOMS_LABMARKERS_FILENAME = "extended_symptoms_labmarkers.json"
SYMPTOMS_LABMARKERS_FILENAME = "symptoms_labmarkers.json"


def negate(feature: str) -> str:
    """Return the negated form: prefix + the feature lowercased."""
    return f"{ABSENCE_PREFIX}{feature.lower()}"


def extend(items: list[str]) -> list[str]:
    """Interleave each feature with its negated pair (pure, no I/O)."""
    extended: list[str] = []
    for feature in items:
        extended.append(feature)       # original, verbatim
        extended.append(negate(feature))
    return extended


def build_extended_symptoms_labmarkers(
    input_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
) -> dict:
    """Read symptoms_labmarkers.json, write the extended file, return the report.

    The input file is read-only. Returns a report dict with the written
    ``document`` and the ``source_count`` for the CLI to render.
    """
    input_path = Path(input_path) if input_path is not None else \
        config.OUTPUT_DIR / SYMPTOMS_LABMARKERS_FILENAME
    output_path = Path(output_path) if output_path is not None else \
        config.OUTPUT_DIR / EXTENDED_SYMPTOMS_LABMARKERS_FILENAME

    source_document = json.loads(input_path.read_text(encoding="utf-8"))
    source = source_document.get(INPUT_KEY, [])
    extended = extend(source)

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
        "source_count": len(source),
        "output_path": output_path,
    }


def _interleaving_ok(source: list[str], extended: list[str]) -> bool:
    """True iff even indices are originals and odd indices their negated pairs."""
    if len(extended) != 2 * len(source):
        return False
    for i, feature in enumerate(source):
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
        f"  source: {source_count} items -> expected {2 * source_count}",
    ]
    if total != 2 * source_count:
        lines.append(f"  ⚠ total {total} is not exactly 2 × {source_count} "
                     f"({2 * source_count}) — source may be malformed.")
    # Reconstruct the source list from even indices to verify interleaving.
    source = extended[0::2]
    if _interleaving_ok(source, extended):
        lines.append("  interleaving OK — even = original, odd = negated pair.")
    else:
        lines.append("  ⚠ interleaving check FAILED.")
    # Every negated (odd-indexed) item must carry the exact prefix.
    if all(item.startswith(ABSENCE_PREFIX) for item in extended[1::2]):
        lines.append(f'  prefix OK — every negation starts "{ABSENCE_PREFIX}".')
    else:
        lines.append("  ⚠ prefix check FAILED on one or more negations.")
    return "\n".join(lines)
