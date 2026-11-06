"""Orchestration: extract (Cols 1&2) -> features (Col 3) -> validate -> write.

The heavy parser gate is not reproduced here; validation is a lightweight,
deterministic sanity check that surfaces (not hides) anything odd.
"""

from __future__ import annotations

import datetime as _dt
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from . import config
from .extractor import ReactionRecord, extract_records, load_document
from .features import FEATURE_KEYS, FeatureExtractor


@dataclass
class Entry:
    reaction: str
    definition: str
    features: dict
    source_page: int

    def to_dict(self) -> dict:
        return {
            "reaction": self.reaction,
            "definition": self.definition,
            "features": self.features,
            "source_page": self.source_page,
        }


@dataclass
class ValidationReport:
    flags: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.flags

    def render(self) -> str:
        lines = ["Validation report", "-" * 17]
        lines.extend(f"  • {n}" for n in self.notes)
        if self.flags:
            lines.append("  FLAGS:")
            lines.extend(f"    ! {f}" for f in self.flags)
        else:
            lines.append("  no flags")
        return "\n".join(lines)


def _features_flags(entry: "Entry") -> list[str]:
    """Advisory checks on one entry's features object (flag, never hard-fail)."""
    flags: list[str] = []
    where = f"({entry.reaction!r}, page {entry.source_page})"
    feats = entry.features
    if not isinstance(feats, dict) or set(feats) != set(FEATURE_KEYS):
        flags.append(f"features object has unexpected keys {where} — review")
        return flags
    if all(feats.get(k) is None for k in FEATURE_KEYS):
        # A reaction with no time window, no symptoms and no labs is suspicious.
        flags.append(f"all feature fields null {where} — review")
    for key in ("signs_symptoms", "lab_markers"):
        value = feats.get(key)
        if value is not None and (not isinstance(value, list) or not value):
            flags.append(f"{key} is non-null but empty {where} — review")
    # A time window stated in the definition but returned as null was likely
    # missed (e.g. DSTR's "between 24 hours and 28 days").
    if feats.get("time_window") is None and config.TIME_WINDOW_HINT.search(entry.definition):
        flags.append(f"time_window null but definition mentions a time window {where} — review")
    return flags


def _match(record: ReactionRecord, selector: str) -> bool:
    """Match a --single selector against a record by page number or name substring."""
    sel = selector.strip().lower()
    if sel.isdigit():
        return record.source_page == int(sel)
    return sel in record.reaction.lower()


class Pipeline:
    def __init__(self, extractor: Optional[FeatureExtractor] = None):
        self.feature_extractor = extractor or FeatureExtractor()

    # -- assemble ------------------------------------------------------------ #
    def build_entries(self, records: list[ReactionRecord], *, use_cache: bool = True,
                      only: Optional[str] = None) -> list[Entry]:
        entries: list[Entry] = []
        for rec in records:
            if only is not None and not _match(rec, only):
                continue
            features = self.feature_extractor.features_for(
                rec.reaction, rec.definition, use_cache=use_cache)
            entries.append(Entry(reaction=rec.reaction, definition=rec.definition,
                                 features=features, source_page=rec.source_page))
        return entries

    # -- validate ------------------------------------------------------------ #
    @staticmethod
    def validate(entries: list[Entry], *, single: bool) -> ValidationReport:
        report = ValidationReport()
        pages = [e.source_page for e in entries]

        if single:
            report.notes.append(f"single mode: {len(entries)} entry/entries produced")
        else:
            report.notes.append(f"{len(entries)} entries produced (expected 12)")
            if len(entries) != 12:
                report.flags.append(f"expected 12 entries, got {len(entries)}")
            if 16 in pages:
                report.flags.append("page 16 (Table 7) unexpectedly present")
            missing = [p for p in config.TARGET_PAGES if p not in pages]
            if missing:
                report.flags.append(f"target pages missing: {missing}")
            dupes = sorted({p for p in pages if pages.count(p) > 1})
            if dupes:
                report.flags.append(f"duplicate source pages: {dupes}")

        for e in entries:
            if not e.reaction.strip():
                report.flags.append(f"empty reaction name (page {e.source_page})")
            if not e.definition.strip():
                report.flags.append(f"empty definition ({e.reaction!r}, page {e.source_page})")
            report.flags.extend(_features_flags(e))
        return report

    # -- write --------------------------------------------------------------- #
    @staticmethod
    def build_document(entries: list[Entry], source_filename: str) -> dict:
        return {
            "metadata": {
                "source_filename": source_filename,
                "generated_on": _dt.datetime.now(_dt.timezone.utc).isoformat(),
                "count": len(entries),
                "model": config.VISION_MODEL,
            },
            "reactions": [e.to_dict() for e in entries],
        }

    @staticmethod
    def write_document(document: dict, output_path: Path) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
        return output_path

    # -- run ----------------------------------------------------------------- #
    def run(self, input_path: Path, output_path: Path, *, use_cache: bool = True,
            only: Optional[str] = None) -> tuple[dict, ValidationReport]:
        doc = load_document(input_path)
        records = extract_records(doc)
        entries = self.build_entries(records, use_cache=use_cache, only=only)
        report = self.validate(entries, single=only is not None)
        document = self.build_document(entries, source_filename=Path(input_path).name)
        self.write_document(document, output_path)
        return document, report
