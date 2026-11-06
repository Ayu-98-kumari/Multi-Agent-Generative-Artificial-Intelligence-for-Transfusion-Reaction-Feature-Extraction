"""Orchestration: read xlsx -> extract per row -> write xlsx -> export CSV.

Row detection is header-driven and scalable: the header row is found by locating
the ``Clinical Text`` cell in Column A (wherever it sits), and data rows are the
contiguous non-empty Column A cells beneath it — so a trailing block of empty
formatted rows (openpyxl reports ``max_row`` = 1000 here) is ignored and no row
count is ever hardcoded.

Each row's validated features JSON is written back into Column B. Off-list /
malformed features are dropped by the deterministic post-check and recorded, with
their row number, to a per-input ``<input-stem>.dropped.log`` (JSONL) — never silently
discarded. After processing, the workbook is saved and the sheet is exported to
CSV.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Optional

from . import config
from .extractor import ClinicalFeatureExtractor
from .vocabulary import Vocabulary, load_vocabulary


class PipelineError(RuntimeError):
    """A structural problem with the input workbook (not a model failure)."""


# --------------------------------------------------------------------------- #
# Workbook reading (header-driven, scalable)
# --------------------------------------------------------------------------- #
def find_header_row(ws) -> int:
    """Return the 1-based row index whose Column A equals the Clinical Text header.

    Anchors to the header text, not a literal row number, so blank rows above the
    header (in future files) do not break detection. The comparison is
    case-insensitive and whitespace-tolerant, so ``Clinical text`` / ``Clinical
    Text`` / ``  clinical text `` all match.
    """
    target = config.CLINICAL_TEXT_HEADER.strip().casefold()
    for row in range(1, (ws.max_row or 1) + 1):
        value = ws.cell(row=row, column=1).value
        if isinstance(value, str) and value.strip().casefold() == target:
            return row
    raise PipelineError(
        f"header {config.CLINICAL_TEXT_HEADER!r} not found in column A of sheet "
        f"{config.SHEET_NAME!r}")


def collect_data_rows(ws, header_row: int) -> list[tuple[int, str]]:
    """Collect ``(row_index, clinical_text)`` for each non-empty row below the header.

    Starts at ``header_row + 1`` and stops at the first empty Column A cell, so the
    real contiguous data block is processed regardless of its size or of trailing
    formatted-but-empty rows.
    """
    rows: list[tuple[int, str]] = []
    row = header_row + 1
    while row <= (ws.max_row or 0):
        value = ws.cell(row=row, column=1).value
        text = value.strip() if isinstance(value, str) else ("" if value is None
                                                              else str(value).strip())
        if not text:
            break
        rows.append((row, text))
        row += 1
    return rows


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
class Pipeline:
    """End-to-end run over a workbook."""

    def __init__(self, vocabulary: Optional[Vocabulary] = None, extractor=None):
        self.vocab = vocabulary or load_vocabulary()
        self.extractor = extractor or ClinicalFeatureExtractor(self.vocab)

    def run(self, input_xlsx: Optional[Path] = None,
            output_csv: Optional[Path] = None, *, use_cache: bool = True,
            single_row: Optional[int] = None) -> dict:
        import openpyxl

        input_xlsx = Path(input_xlsx) if input_xlsx else config.INPUT_XLSX_DEFAULT
        output_csv = Path(output_csv) if output_csv else config.OUTPUT_CSV_DEFAULT
        if not input_xlsx.exists():
            raise PipelineError(f"input workbook not found: {input_xlsx}")

        wb = openpyxl.load_workbook(input_xlsx)
        if config.SHEET_NAME not in wb.sheetnames:
            raise PipelineError(
                f"sheet {config.SHEET_NAME!r} not found in {input_xlsx.name} "
                f"(sheets: {wb.sheetnames})")
        ws = wb[config.SHEET_NAME]

        header_row = find_header_row(ws)
        data_rows = collect_data_rows(ws, header_row)
        if not data_rows:
            raise PipelineError(
                f"no clinical-text rows found below the header (row {header_row})")

        # --single-row targets an absolute worksheet row number; it must be one of
        # the detected data rows.
        if single_row is not None:
            valid = {idx for idx, _ in data_rows}
            if single_row not in valid:
                raise PipelineError(
                    f"--single-row {single_row} is not a data row "
                    f"(data rows: {min(valid)}..{max(valid)})")
            data_rows = [(idx, text) for idx, text in data_rows if idx == single_row]

        # Label Columns C and D so the written-back workbook is self-describing.
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

        for row_index, text in data_rows:
            result = self.extractor.extract(text, use_cache=use_cache)
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
                # Model 3 returned a variable_id not in that feature's Timing/Details
                # menu; it is DROPPED (never emitted) and recorded here so the gap is
                # visible, not silent.
                schema_dropped_records.append({"row": row_index, **rec})
            # Write the validated features JSON (with each positive feature's nested
            # NHSN `schema` object) back into Column B of this row, and the Model 4
            # treatments JSON (separate object) into Column C.
            ws.cell(row=row_index, column=2).value = json.dumps(
                {"features": features}, ensure_ascii=False)
            ws.cell(row=row_index, column=3).value = json.dumps(
                {"treatments": treatments}, ensure_ascii=False)
            # Model 5 (INDEPENDENT of the four stages above): past medical history,
            # extracted from the note alone and written as a separate object in Column D.
            pmh = self.extractor.extract_past_medical_history(text, use_cache=use_cache)
            total_pmh += len(pmh)
            ws.cell(row=row_index, column=4).value = json.dumps(
                {"past_medical_history": pmh}, ensure_ascii=False)

        # Persist the dropped-feature log (fresh for this run) before saving.
        dropped_log_path = self._write_dropped_log(
            output_csv.parent, input_xlsx, dropped_records)
        # Persist the redundant-absence log (fresh for this run).
        absences_log_path = self._write_absences_log(
            output_csv.parent, input_xlsx, absence_records)
        # Persist the Model-3 schema-drop log (fresh for this run).
        schema_log_path = self._write_schema_log(
            output_csv.parent, input_xlsx, schema_dropped_records)

        # Save the workbook (Column B now filled) and export the sheet to CSV.
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        wb.save(input_xlsx)
        self._export_csv(ws, header_row, data_rows, output_csv)

        return {
            "input_xlsx": input_xlsx,
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

    # -- outputs ------------------------------------------------------------- #
    def _write_dropped_log(self, out_dir: Path, input_path: Path,
                           records: list[dict]) -> Path:
        """Write one JSON record per dropped feature (JSONL). Always written.

        Named after the input file so runs on different inputs don't clobber each
        other's logs.
        """
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / config.dropped_log_name(input_path)
        with path.open("w", encoding="utf-8") as fh:
            for rec in records:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return path

    def _write_absences_log(self, out_dir: Path, input_path: Path,
                            records: list[dict]) -> Path:
        """Write one JSON record per redundant absence removed (JSONL). Always written.

        Named after the input file (``<input-stem>.absences.log``) so runs on
        different inputs don't clobber each other's logs.
        """
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / config.absences_log_name(input_path)
        with path.open("w", encoding="utf-8") as fh:
            for rec in records:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return path

    def _write_schema_log(self, out_dir: Path, input_path: Path,
                          records: list[dict]) -> Path:
        """Write one JSON record per variable_id Model 3 returned that was NOT in the
        feature's Timing/Details menu (JSONL). Always written. Such a variable is
        DROPPED from the output (never emitted); this log records the drop so an
        off-menu hallucination is visible, never silent."""
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / config.schema_log_name(input_path)
        with path.open("w", encoding="utf-8") as fh:
            for rec in records:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return path

    def _export_csv(self, ws, header_row: int,
                    data_rows: list[tuple[int, str]], output_csv: Path) -> None:
        """Export the four sheet columns (Clinical Text, Extracted Features,
        Treatments, Past Medical History) to CSV."""
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


def render_report(report: dict) -> str:
    """Human-readable run summary."""
    lines = [
        f"Input workbook : {report['input_xlsx']}",
        f"Output CSV     : {report['output_csv']}",
        f"Dropped log    : {report['dropped_log']}",
        f"Absences log   : {report.get('absences_log', '-')}",
        f"Schema log     : {report.get('schema_log', '-')}",
        f"Header row     : {report['header_row']}",
        f"Rows processed : {report['rows_processed']}"
        + (f" (single row {report['single_row']})" if report['single_row'] else ""),
        f"Features kept  : {report['total_features']}",
        f"Treatments     : {report.get('total_treatments', 0)}",
        f"Past med. hist.: {report.get('total_pmh', 0)}",
        f"Redundant absences removed: {report.get('absences_deduped', 0)}",
        f"Schema var drops (off-menu): {report.get('schema_dropped', 0)}"
        + (" (see schema log)" if report.get('schema_dropped') else ""),
        f"Off-list dropped: {report['dropped_count']}"
        + (" (see dropped log)" if report['dropped_count'] else ""),
    ]
    if report["zero_feature_rows"]:
        rows = ", ".join(str(r) for r in report["zero_feature_rows"])
        lines.append(
            f"⚠ zero-feature rows (possible miss, review): {rows}")
    return "\n".join(lines)
