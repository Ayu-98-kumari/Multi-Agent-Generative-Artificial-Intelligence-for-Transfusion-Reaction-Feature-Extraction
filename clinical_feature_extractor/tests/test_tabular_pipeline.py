"""Tests for the format-agnostic (.csv / .xlsx) entrypoint — fake Groq, no spend.

Reuses the same fake-client pattern as test_extractor.py. Verifies the CSV path
end to end, that .xlsx still routes to the existing pipeline, that header/row
detection behaves identically on CSV, and extension dispatch / errors.
"""

from __future__ import annotations

import csv
import json

import pytest

from clinical_feature_extractor import config
from clinical_feature_extractor.tabular_pipeline import (
    CsvWorksheet,
    PipelineError,
    run_extraction,
)
from clinical_feature_extractor.pipeline import collect_data_rows, find_header_row
from clinical_feature_extractor.vocabulary import Vocabulary

# Reuse the fake client + extractor factory from the extractor test module.
from clinical_feature_extractor.tests.test_extractor import (
    VOCAB,
    _features_json,
    _resp,
    make_extractor,
    make_workbook,
)


def write_csv(path, texts, *, header_row=1, extra_cols=None):
    """Write a CSV with the two headers and given texts (header may be shifted)."""
    rows = [[] for _ in range(header_row - 1)]  # blank rows above the header
    rows.append([config.CLINICAL_TEXT_HEADER, config.EXTRACTED_FEATURES_HEADER])
    for t in texts:
        rows.append([t, ""])
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        for r in rows:
            writer.writerow(r)
    return path


# --------------------------------------------------------------------------- #
# CsvWorksheet duck-typing: header detection + row collection reuse
# --------------------------------------------------------------------------- #
def test_csv_worksheet_supports_header_and_row_detection():
    grid = [[config.CLINICAL_TEXT_HEADER, config.EXTRACTED_FEATURES_HEADER],
            ["first", ""], ["second", ""]]
    ws = CsvWorksheet(grid)
    assert find_header_row(ws) == 1
    assert [r for r, _ in collect_data_rows(ws, 1)] == [2, 3]


def test_csv_worksheet_cell_write_expands_grid():
    ws = CsvWorksheet([["only one col"]])
    ws.cell(row=1, column=2).value = "written"
    assert ws.cell(row=1, column=2).value == "written"


def test_csv_header_can_be_below_row_1():
    grid = [[], [], [config.CLINICAL_TEXT_HEADER, config.EXTRACTED_FEATURES_HEADER],
            ["t1", ""]]
    ws = CsvWorksheet(grid)
    assert find_header_row(ws) == 3
    assert [r for r, _ in collect_data_rows(ws, 3)] == [4]


def test_header_detection_is_case_insensitive():
    """A differently-cased header (e.g. 'Clinical text') still anchors detection."""
    grid = [["Clinical text", "Features"], ["t1", ""], ["t2", ""]]
    ws = CsvWorksheet(grid)
    assert find_header_row(ws) == 1
    assert [r for r, _ in collect_data_rows(ws, 1)] == [2, 3]


def test_csv_input_untouched_when_write_back_disabled(tmp_path):
    src = write_csv(tmp_path / "in.csv", ["only text"])
    before = src.read_bytes()
    out = tmp_path / "out.csv"
    ext = make_extractor(
        [_resp(_features_json([{"feature": "cough", "evidence": "x"}]))], tmp_path)
    run_extraction(src, out, vocabulary=VOCAB, extractor=ext,
                   use_cache=False, write_back=False)
    assert src.read_bytes() == before      # input untouched
    assert out.exists()                    # deliverable still produced


# --------------------------------------------------------------------------- #
# CSV end-to-end
# --------------------------------------------------------------------------- #
def _run_csv(texts, behaviors, tmp_path, **kw):
    src = write_csv(tmp_path / "in.csv", texts)
    out = tmp_path / "out" / "extracted.csv"
    ext = make_extractor(behaviors, tmp_path)
    report = run_extraction(src, out, vocabulary=VOCAB, extractor=ext,
                            use_cache=False, **kw)
    return report, src, out


def test_csv_input_extracts_and_writes_back(tmp_path):
    texts = ["clinical text one", "clinical text two"]
    behaviors = [
        _resp(_features_json([{"feature": "dyspnea", "evidence": "a"}])),
        _resp(_features_json([{"feature": "cough", "evidence": "b"}])),
    ]
    report, src, out = _run_csv(texts, behaviors, tmp_path)
    assert report["rows_processed"] == 2
    assert report["total_features"] == 2
    assert report["absences_log"].name == "in.absences.log"   # per-input, always written
    # Input CSV filled in place (Column B).
    rows = list(csv.reader(src.open(encoding="utf-8-sig")))
    assert json.loads(rows[1][1])["features"][0]["feature"] == "dyspnea"
    # Deliverable CSV: header + 2 data rows, with the Treatments and Past Medical
    # History (Model 5) columns.
    out_rows = list(csv.reader(out.open(encoding="utf-8")))
    assert out_rows[0] == [config.CLINICAL_TEXT_HEADER,
                           config.EXTRACTED_FEATURES_HEADER,
                           config.TREATMENTS_HEADER,
                           config.PAST_MEDICAL_HISTORY_HEADER]
    assert len(out_rows) == 3


@pytest.mark.parametrize("n", [3, 25])
def test_csv_scalability_any_row_count(tmp_path, n):
    texts = [f"row text {i}" for i in range(n)]
    behaviors = [_resp(_features_json([{"feature": "cough", "evidence": "x"}]))
                 for _ in range(n)]
    report, src, out = _run_csv(texts, behaviors, tmp_path)
    assert report["rows_processed"] == n
    assert len(list(csv.reader(out.open(encoding="utf-8")))) == n + 1


def test_csv_off_list_dropped_and_logged(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "VALIDATE_FEATURES", True)  # dropping needs the check
    texts = ["t1"]
    behaviors = [_resp(_features_json([{"feature": "cough", "evidence": "a"},
                                       {"feature": "off-list-z", "evidence": "b"}]))]
    report, src, out = _run_csv(texts, behaviors, tmp_path)
    assert report["dropped_count"] == 1
    assert report["dropped_log"].name == "in.dropped.log"  # named after input (in.csv)
    entries = [json.loads(l) for l in
               report["dropped_log"].read_text(encoding="utf-8").splitlines() if l.strip()]
    assert entries[0]["feature"] == "off-list-z"
    assert entries[0]["row"] == 2


def test_csv_single_row(tmp_path):
    texts = ["r2", "r3", "r4"]
    behaviors = [_resp(_features_json([{"feature": "cough", "evidence": "x"}]))]
    report, src, out = _run_csv(texts, behaviors, tmp_path, single_row=3)
    assert report["rows_processed"] == 1
    rows = list(csv.reader(src.open(encoding="utf-8-sig")))
    assert rows[2][1] != ""       # row 3 written
    assert rows[1][1] == ""       # row 2 untouched


# --------------------------------------------------------------------------- #
# Dispatch: .xlsx still routes to the existing pipeline; bad extension errors
# --------------------------------------------------------------------------- #
def test_xlsx_still_dispatches_to_existing_pipeline(tmp_path):
    texts = ["a", "b"]
    xlsx = make_workbook(tmp_path / "in.xlsx", texts)
    ext = make_extractor(
        [_resp(_features_json([{"feature": "cough", "evidence": "x"}])),
         _resp(_features_json([{"feature": "fever", "evidence": "y"}]))], tmp_path)
    report = run_extraction(xlsx, tmp_path / "out.csv",
                            vocabulary=VOCAB, extractor=ext, use_cache=False)
    assert report["rows_processed"] == 2


def test_unsupported_extension_raises(tmp_path):
    bad = tmp_path / "notes.txt"
    bad.write_text("Clinical Text\nsomething\n", encoding="utf-8")
    with pytest.raises(PipelineError):
        run_extraction(bad, tmp_path / "out.csv", vocabulary=VOCAB)


def test_missing_input_raises(tmp_path):
    with pytest.raises(PipelineError):
        run_extraction(tmp_path / "nope.csv", tmp_path / "out.csv", vocabulary=VOCAB)


def test_vocabulary_source_not_modified(tmp_path):
    before = config.VOCABULARY_JSON.read_bytes()
    _run_csv(["only text"],
             [_resp(_features_json([{"feature": "cough", "evidence": "x"}]))], tmp_path)
    assert config.VOCABULARY_JSON.read_bytes() == before
