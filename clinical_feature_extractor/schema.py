"""Read-only NHSN hierarchical-schema loader (the Model-3 rewrite input).

The old Model 3 labelled each feature's *relatedness* to a transfusion reaction.
Doctors found that added little value, so Model 3 now instead fills in the NHSN
Hemovigilance variables for each POSITIVE feature. For every positive feature the
schema workbook defines a menu of variables in two groups — **Timing** and
**Details** (the ``Level 5 - Variable group`` column). Model 3 is run separately
per feature per group: it is shown the note, the feature, and that group's
variable menu (id, definition, response type, allowed values, units) and returns
the applicable variable(s) with the appropriate value/unit.

This module is the deterministic, read-only bridge between a positive feature
name (canonical vocabulary spelling) and its variable menu. It never calls a
model and never writes the workbook. The workbook is loaded once, read-only.

The schema's ``Level 4 - Exact source feature`` names are canonicalized through
the same :class:`~clinical_feature_extractor.vocabulary.Vocabulary` the pipeline
uses, so a positive feature emitted by Model 1 (already in canonical spelling)
looks up cleanly — including the one cosmetic mismatch (schema
``pre-transfusion value`` vs vocabulary ``pre- transfusion value``), which the
vocabulary's formatting-normalized tier resolves to the same canonical entry.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from . import config
from .vocabulary import Vocabulary, load_vocabulary

# The two variable groups, re-exported from config for convenient importing.
TIMING = config.SCHEMA_GROUP_TIMING
DETAILS = config.SCHEMA_GROUP_DETAILS
GROUPS = (TIMING, DETAILS)


class SchemaError(RuntimeError):
    """The schema workbook is missing, malformed, or internally inconsistent."""


@dataclass(frozen=True)
class SchemaVariable:
    """One NHSN variable row from a feature's Timing or Details menu.

    ``allowed_values`` and ``units`` are the ``|``-separated menus split into
    lists (empty when the cell was blank). Everything else is the cell's text as
    stored. All of this is *input context* for Model 3 — the definition, response
    type, allowed values, and units menu tell the model what the variable means
    and what it may answer; only a subset is echoed into the model's output.
    """

    variable_id: str
    order: int
    definition: str
    response_type: str
    allowed_values: tuple[str, ...]
    units: tuple[str, ...]
    notes: Optional[str]
    dependencies: Optional[str]


def _clean(value) -> Optional[str]:
    """Return a stripped string for a cell, or None for blank/None cells."""
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _split_menu(value) -> tuple[str, ...]:
    """Split a ``a | b | c`` menu cell into a tuple of trimmed options."""
    text = _clean(value)
    if not text:
        return ()
    return tuple(part.strip() for part in text.split("|") if part.strip())


class FeatureSchema:
    """Read-only view over the NHSN per-feature variable menus.

    Keyed by canonical feature name; ``variables(feature, group)`` returns that
    feature's Timing or Details variables in ``Variable order``. Construction is
    the only place the workbook is touched.
    """

    def __init__(self, by_feature: dict[str, dict[str, tuple[SchemaVariable, ...]]],
                 canonical_names: dict[str, str]):
        # canonical-lowercase -> {group -> (SchemaVariable, ...)}
        self._by_feature = by_feature
        # canonical-lowercase -> canonical display spelling (for .features)
        self._canonical_names = canonical_names

    @property
    def features(self) -> list[str]:
        """The canonical feature names that have a schema entry (sorted)."""
        return sorted(self._canonical_names.values())

    def __len__(self) -> int:
        return len(self._by_feature)

    def __contains__(self, feature: str) -> bool:
        return isinstance(feature, str) and feature.strip().lower() in self._by_feature

    def variables(self, feature: str, group: str) -> tuple[SchemaVariable, ...]:
        """Return the variable menu for ``feature`` in ``group`` (Timing/Details).

        ``feature`` is matched case-insensitively against the canonical names.
        Returns an empty tuple when the feature is unknown or has no variables in
        that group. ``group`` must be one of :data:`GROUPS`.
        """
        if group not in GROUPS:
            raise SchemaError(
                f"unknown variable group {group!r}; expected one of {GROUPS}")
        if not isinstance(feature, str):
            return ()
        return self._by_feature.get(feature.strip().lower(), {}).get(group, ())


def load_schema(path: Optional[Path] = None,
                vocabulary: Optional[Vocabulary] = None) -> FeatureSchema:
    """Load the NHSN hierarchical schema, keyed by canonical feature name.

    The workbook is opened read-only. Fully blank spacer rows are skipped; every
    real row must carry a feature, a group, and a variable id (a violation raises
    :class:`SchemaError` rather than being silently dropped). Feature names are
    canonicalized through ``vocabulary`` so pipeline features look up cleanly.
    """
    import openpyxl

    path = Path(path) if path else config.SCHEMA_XLSX
    if not path.exists():
        raise SchemaError(f"schema workbook not found: {path}")
    vocab = vocabulary or load_vocabulary()

    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    if config.SCHEMA_SHEET not in wb.sheetnames:
        raise SchemaError(
            f"sheet {config.SCHEMA_SHEET!r} not found in {path.name} "
            f"(sheets: {wb.sheetnames})")
    ws = wb[config.SCHEMA_SHEET]
    rows = list(ws.iter_rows(values_only=True))
    wb.close()
    if not rows:
        raise SchemaError(f"sheet {config.SCHEMA_SHEET!r} in {path.name} is empty")

    header = rows[0]
    try:
        col = {name: header.index(name) for name in (
            config.SCHEMA_COL_FEATURE, config.SCHEMA_COL_GROUP,
            config.SCHEMA_COL_ORDER, config.SCHEMA_COL_VARIABLE_ID,
            config.SCHEMA_COL_DEFINITION, config.SCHEMA_COL_RESPONSE_TYPE,
            config.SCHEMA_COL_ALLOWED_VALUES, config.SCHEMA_COL_UNITS,
            config.SCHEMA_COL_NOTES, config.SCHEMA_COL_DEPENDENCIES)}
    except ValueError as exc:
        raise SchemaError(
            f"schema sheet {config.SCHEMA_SHEET!r} missing an expected column: {exc}\n"
            f"header was: {header}") from exc

    by_feature: dict[str, dict[str, list[SchemaVariable]]] = {}
    canonical_names: dict[str, str] = {}

    for i, row in enumerate(rows[1:], start=2):  # 1-based worksheet row (header is 1)
        # Fully blank spacer rows separate feature blocks in the source; skip them.
        if all(v is None or (isinstance(v, str) and not v.strip()) for v in row):
            continue

        # openpyxl (read-only) trims trailing empty cells, so a row whose last
        # columns are blank comes back shorter than the header. Pad it back out
        # so every column index is valid -- a missing trailing cell reads as
        # empty (None), exactly as a present-but-blank cell already does.
        if len(row) < len(header):
            row = tuple(row) + (None,) * (len(header) - len(row))

        raw_feature = _clean(row[col[config.SCHEMA_COL_FEATURE]])
        group = _clean(row[col[config.SCHEMA_COL_GROUP]])
        variable_id = _clean(row[col[config.SCHEMA_COL_VARIABLE_ID]])
        if not raw_feature or not group or not variable_id:
            raise SchemaError(
                f"schema row {i} is non-blank but missing a feature, group, or "
                f"variable id (feature={raw_feature!r}, group={group!r}, "
                f"variable_id={variable_id!r})")
        if group not in GROUPS:
            raise SchemaError(
                f"schema row {i}: unexpected variable group {group!r} "
                f"(expected one of {GROUPS})")

        canonical = vocab.match(raw_feature)
        if canonical is None:
            raise SchemaError(
                f"schema row {i}: feature {raw_feature!r} does not match any "
                f"vocabulary entry; the schema and vocabulary are out of sync")
        key = canonical.strip().lower()
        canonical_names.setdefault(key, canonical)

        order_raw = row[col[config.SCHEMA_COL_ORDER]]
        try:
            order = int(order_raw)
        except (TypeError, ValueError):
            order = 0  # tolerate a missing order; sort is stable on load order

        variable = SchemaVariable(
            variable_id=variable_id,
            order=order,
            definition=_clean(row[col[config.SCHEMA_COL_DEFINITION]]) or "",
            response_type=_clean(row[col[config.SCHEMA_COL_RESPONSE_TYPE]]) or "",
            allowed_values=_split_menu(row[col[config.SCHEMA_COL_ALLOWED_VALUES]]),
            units=_split_menu(row[col[config.SCHEMA_COL_UNITS]]),
            notes=_clean(row[col[config.SCHEMA_COL_NOTES]]),
            dependencies=_clean(row[col[config.SCHEMA_COL_DEPENDENCIES]]),
        )
        by_feature.setdefault(key, {}).setdefault(group, []).append(variable)

    # Freeze: sort each group's variables by their declared order and tuple-ify.
    frozen: dict[str, dict[str, tuple[SchemaVariable, ...]]] = {}
    for key, groups in by_feature.items():
        frozen[key] = {
            group: tuple(sorted(vs, key=lambda v: v.order))
            for group, vs in groups.items()
        }
    if not frozen:
        raise SchemaError(f"no schema rows found in {path.name}")
    return FeatureSchema(frozen, canonical_names)
