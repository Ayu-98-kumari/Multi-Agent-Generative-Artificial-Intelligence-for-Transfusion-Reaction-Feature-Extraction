"""Format-agnostic entry point: run extraction over a ``.xlsx`` OR a ``.csv`` input.

The input format is chosen by file extension:
  * ``.xlsx`` / ``.xlsm`` -> delegates to the existing, tested
    :class:`clinical_feature_extractor.pipeline.Pipeline` unchanged.
  * ``.csv``              -> handled here via a tiny worksheet-shaped adapter
    (:class:`CsvWorksheet`) so the SAME header-detection and row-collection logic
    (``find_header_row`` / ``collect_data_rows``) runs identically for both.

Nothing in this module modifies the existing pipeline/extractor/vocabulary code;
it only imports and reuses them. Behavior for both formats is symmetric: the input
file is filled in place (Column B written back) and the result is exported to CSV.
"""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path
from typing import Optional

from . import config
from .extractor import ClinicalFeatureExtractor
from .pipeline import (
    Pipeline,
    PipelineError,
    collect_data_rows,
    find_header_row,
    render_report,  # re-exported for a single import site in the CLI
)
from .vocabulary import Vocabulary, load_vocabulary

__all__ = ["run_extraction", "render_report", "PipelineError", "CsvWorksheet"]

XLSX_SUFFIXES = (".xlsx", ".xlsm")
CSV_SUFFIXES = (".csv",)


def _log(msg: str) -> None:
    """Case-level progress to stderr when CLF_PROGRESS is on (no-op otherwise)."""
    if not config.PROGRESS:
        return
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


# --------------------------------------------------------------------------- #
# CSV adapter — duck-types the small slice of an openpyxl worksheet that
# find_header_row / collect_data_rows / the extraction loop rely on:
#   ws.max_row  and  ws.cell(row, column).value  (read AND write, 1-based).
# --------------------------------------------------------------------------- #
class _CsvCell:
    __slots__ = ("_sheet", "_row", "_col")

    def __init__(self, sheet: "CsvWorksheet", row: int, col: int):
        self._sheet = sheet
        self._row = row
        self._col = col

    @property
    def value(self):
        return self._sheet._get(self._row, self._col)

    @value.setter
    def value(self, v):
        self._sheet._set(self._row, self._col, v)


class CsvWorksheet:
    """A mutable 1-based grid over CSV rows, exposing the openpyxl ``ws`` subset."""

    def __init__(self, rows: list[list]):
        # Internal store is 0-indexed, ragged rows allowed.
        self._rows: list[list] = rows

    @property
    def rows(self) -> list[list]:
        return self._rows

    @property
    def max_row(self) -> int:
        return len(self._rows)

    def cell(self, row: int, column: int) -> _CsvCell:
        return _CsvCell(self, row, column)

    def _get(self, row: int, col: int):
        if 1 <= row <= len(self._rows):
            rowlist = self._rows[row - 1]
            if 1 <= col <= len(rowlist):
                val = rowlist[col - 1]
                # An empty CSV cell reads as None, so collect_data_rows stops on it
                # exactly as it does for a blank openpyxl cell.
                return None if val == "" else val
        return None

    def _set(self, row: int, col: int, v) -> None:
        while len(self._rows) < row:
            self._rows.append([])
        rowlist = self._rows[row - 1]
        while len(rowlist) < col:
            rowlist.append(None)
        rowlist[col - 1] = v


def _load_csv_rows(path: Path) -> list[list]:
    # utf-8-sig transparently strips a BOM if the CSV came from Excel.
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return [list(row) for row in csv.reader(fh)]


def _save_csv_rows(path: Path, rows: list[list]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        for row in rows:
            writer.writerow(["" if v is None else v for v in row])


# --------------------------------------------------------------------------- #
# Shared output writers (kept local so this module needs no private access to
# Pipeline; identical semantics to the xlsx path).
# --------------------------------------------------------------------------- #
def _write_dropped_log(out_dir: Path, input_path: Path, records: list[dict]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / config.dropped_log_name(input_path)
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return path


def _write_absences_log(out_dir: Path, input_path: Path, records: list[dict]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / config.absences_log_name(input_path)
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return path


def _write_schema_log(out_dir: Path, input_path: Path, records: list[dict]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / config.schema_log_name(input_path)
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return path


def _export_csv(ws, data_rows: list[tuple[int, str]], output_csv: Path) -> None:
    with output_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([config.CLINICAL_TEXT_HEADER,
                         config.EXTRACTED_FEATURES_HEADER,
                         config.TREATMENTS_HEADER,
                         config.PAST_MEDICAL_HISTORY_HEADER])
        for row_index, text in data_rows:
            extracted = ws.cell(row=row_index, column=2).value
            treatments = ws.cell(row=row_index, column=3).value
            pmh = ws.cell(row=row_index, column=4).value
            writer.writerow([text,
                             extracted if extracted is not None else "",
                             treatments if treatments is not None else "",
                             pmh if pmh is not None else ""])


# --------------------------------------------------------------------------- #
# CSV run (mirrors Pipeline.run's semantics, swapping load/save for CSV)
# --------------------------------------------------------------------------- #
def _run_csv(input_csv: Path, output_csv: Path, *, vocab: Vocabulary,
             extractor: ClinicalFeatureExtractor, use_cache: bool,
             single_row: Optional[int], write_back: bool) -> dict:
    rows = _load_csv_rows(input_csv)
    ws = CsvWorksheet(rows)

    header_row = find_header_row(ws)          # reused, unchanged
    data_rows = collect_data_rows(ws, header_row)  # reused, unchanged
    if not data_rows:
        raise PipelineError(
            f"no clinical-text rows found below the header (row {header_row})")

    if single_row is not None:
        valid = {idx for idx, _ in data_rows}
        if single_row not in valid:
            raise PipelineError(
                f"--single-row {single_row} is not a data row "
                f"(data rows: {min(valid)}..{max(valid)})")
        data_rows = [(idx, text) for idx, text in data_rows if idx == single_row]

    # Label Columns C and D so the written-back file is self-describing.
    ws.cell(row=header_row, column=3).value = config.TREATMENTS_HEADER
    ws.cell(row=header_row, column=4).value = config.PAST_MEDICAL_HISTORY_HEADER

    dropped_records: list[dict] = []
    absence_records: list[dict] = []
    schema_dropped_records: list[dict] = []
    zero_feature_rows: list[int] = []
    total_features = 0
    total_treatments = 0
    total_pmh = 0
    total_absences_deduped = 0

    total_cases = len(data_rows)
    for case_no, (row_index, text) in enumerate(data_rows, 1):
        _log(f"case {case_no}/{total_cases} (row {row_index}): start")
        case_started = time.time()
        result = extractor.extract(text, use_cache=use_cache)
        features = result["features"]
        treatments = result["treatments"]
        total_features += len(features)
        total_treatments += len(treatments)
        total_absences_deduped += result.get("absences_deduped", 0)
        if not features:
            zero_feature_rows.append(row_index)
        for rec in result["dropped"]:
            dropped_records.append({"row": row_index, **rec})
        for feat in result.get("absences_removed", []):
            absence_records.append({"row": row_index,
                                    "feature": feat.get("feature"),
                                    "evidence": feat.get("evidence")})
        for rec in result.get("schema_dropped", []):
            schema_dropped_records.append({"row": row_index, **rec})
        ws.cell(row=row_index, column=2).value = json.dumps(
            {"features": features}, ensure_ascii=False)
        ws.cell(row=row_index, column=3).value = json.dumps(
            {"treatments": treatments}, ensure_ascii=False)
        # Model 5 (INDEPENDENT of the four stages above): past medical history from
        # the note alone, written as a separate object in Column D.
        pmh = extractor.extract_past_medical_history(text, use_cache=use_cache)
        total_pmh += len(pmh)
        ws.cell(row=row_index, column=4).value = json.dumps(
            {"past_medical_history": pmh}, ensure_ascii=False)
        _log(f"case {case_no}/{total_cases} (row {row_index}): done in "
             f"{time.time() - case_started:.0f}s -- {len(features)} features, "
             f"{len(treatments)} treatments, {len(pmh)} pmh")
        # Incremental checkpoint: flush the output CSV after every case so a crash
        # mid-run keeps the cases already finished (paired with the per-stage cache,
        # which lets a re-run resume without re-calling the model). Gated by
        # CLF_PROGRESS so the main pipeline's single-write-at-end behaviour is intact.
        if config.PROGRESS:
            output_csv.parent.mkdir(parents=True, exist_ok=True)
            _export_csv(ws, data_rows, output_csv)

    dropped_log_path = _write_dropped_log(output_csv.parent, input_csv, dropped_records)
    absences_log_path = _write_absences_log(output_csv.parent, input_csv, absence_records)
    schema_log_path = _write_schema_log(
        output_csv.parent, input_csv, schema_dropped_records)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if write_back:
        _save_csv_rows(input_csv, ws.rows)   # fill Column B in the input, in place
    _export_csv(ws, data_rows, output_csv)   # always: the separate deliverable

    return {
        "input_xlsx": input_csv,   # keep the report key stable across formats
        "output_csv": output_csv,
        "dropped_log": dropped_log_path,
        "absences_log": absences_log_path,
        "schema_log": schema_log_path,
        "rows_processed": len(data_rows),
        "total_features": total_features,
        "total_treatments": total_treatments,
        "total_pmh": total_pmh,
        "absences_deduped": total_absences_deduped,
        "absence_records": absence_records,
        "schema_dropped": len(schema_dropped_records),
        "schema_dropped_records": schema_dropped_records,
        "dropped_count": len(dropped_records),
        "dropped_records": dropped_records,
        "zero_feature_rows": zero_feature_rows,
        "header_row": header_row,
        "single_row": single_row,
    }


# --------------------------------------------------------------------------- #
# Public dispatch
# --------------------------------------------------------------------------- #
def run_extraction(input_path, output_csv=None, *, use_cache: bool = True,
                   single_row: Optional[int] = None, write_back: bool = True,
                   vocabulary: Optional[Vocabulary] = None,
                   extractor=None) -> dict:
    """Run extraction over a ``.xlsx`` or ``.csv`` input, chosen by extension.

    ``.xlsx``/``.xlsm`` delegate to the existing :class:`Pipeline` unchanged;
    ``.csv`` runs the adapter path here. Both return the same report shape.

    ``write_back`` (CSV path only) controls whether Column B is filled in the
    *input* file in place; set it False to leave the input untouched and produce
    only the separate ``output_csv`` deliverable.
    """
    input_path = Path(input_path)
    output_csv = Path(output_csv) if output_csv else config.OUTPUT_CSV_DEFAULT
    if not input_path.exists():
        raise PipelineError(f"input file not found: {input_path}")

    suffix = input_path.suffix.lower()

    if suffix in XLSX_SUFFIXES:
        # Existing, tested path — untouched.
        pipeline = Pipeline(vocabulary=vocabulary, extractor=extractor)
        return pipeline.run(input_path, output_csv,
                            use_cache=use_cache, single_row=single_row)

    if suffix in CSV_SUFFIXES:
        vocab = vocabulary or load_vocabulary()
        ext = extractor or ClinicalFeatureExtractor(vocab)
        return _run_csv(input_path, output_csv, vocab=vocab, extractor=ext,
                        use_cache=use_cache, single_row=single_row,
                        write_back=write_back)

    raise PipelineError(
        f"unsupported input format {suffix!r}: give a .xlsx/.xlsm or .csv file")
