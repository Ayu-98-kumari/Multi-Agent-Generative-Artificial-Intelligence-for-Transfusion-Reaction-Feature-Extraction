"""Convert the PMC-Patients JSON dump to a full CSV, streaming (low memory).

PMC-Patients ships as one giant JSON array -- `[ {record}, {record}, ... ]` --
250k records in a single ~837 MB file. Unlike MIMIC's gzipped CSV it is not
line-oriented, so it cannot be read in row chunks with pandas. This script
streams it with ijson: one record is materialised at a time, so peak memory
stays flat no matter how large the file or how small the machine.

Every column is preserved (nothing is dropped or slimmed). Nested fields --
`age` is a list of `[value, unit]` pairs -- are serialised to a JSON string in
their cell, so the CSV stays a flat table while losing no information; a reader
can `json.loads` the cell to get the structure back.

Two streaming passes over the file:
  1. collect the union of every key that appears, in first-seen order, so the
     header captures all columns even if the schema is not perfectly uniform;
  2. write one CSV row per record.

The CSV lands beside the JSON in the data directory (both stay gitignored).
Override locations with $PMC_DATA_DIR or --input/--output.
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import ijson

SCRIPT_DIR = Path(__file__).resolve().parent

# Data (source + derived) lives outside this tracked package, under the
# gitignored umbrella -- same convention as mimic-pipeline.
DATA_DIR = Path(os.environ.get(
    "PMC_DATA_DIR", SCRIPT_DIR.parent / "kept-gitignored" / "pmc-patients"))

DEFAULT_INPUT = "PMC-Patients-V2.json"
DEFAULT_OUTPUT = "PMC-Patients-V2.csv"
PROGRESS_EVERY_N = 25000

# csv's default field-size limit (131,072) is smaller than some patient
# narratives; raise it so no cell is silently refused on read-back.
csv.field_size_limit(min(sys.maxsize, 2**31 - 1))


def stream_records(path):
    """Yield each record dict from the top-level JSON array, one at a time.

    use_float=True makes ijson emit plain floats for JSON numbers instead of
    Decimal objects (which json.dumps cannot serialise) -- e.g. the 34.0 in an
    ``age`` entry ``[[34.0, "year"]]``.
    """
    with open(path, "rb") as fh:
        yield from ijson.items(fh, "item", use_float=True)


def collect_fieldnames(path):
    """Return every key seen across all records, in first-seen order."""
    fieldnames = []
    seen = set()
    for count, record in enumerate(stream_records(path), start=1):
        for key in record:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
        if count % PROGRESS_EVERY_N == 0:
            print(f"  scanned {count:,} records for columns...")
    return fieldnames, count


def flatten_value(value):
    """Make a value CSV-safe: scalars as-is, structures as JSON strings."""
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return value
    # lists / dicts (e.g. age) -> compact JSON so nothing is lost.
    return json.dumps(value, ensure_ascii=False)


def convert(input_path, output_path):
    print(f"Pass 1/2: scanning columns in {input_path.name} ...")
    fieldnames, total = collect_fieldnames(input_path)
    print(f"  found {len(fieldnames)} columns across {total:,} records: "
          f"{', '.join(fieldnames)}")

    print(f"Pass 2/2: writing {output_path.name} ...")
    written = 0
    with open(output_path, "w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=fieldnames)
        writer.writeheader()
        for record in stream_records(input_path):
            writer.writerow({k: flatten_value(record.get(k)) for k in fieldnames})
            written += 1
            if written % PROGRESS_EVERY_N == 0:
                print(f"  wrote {written:,} / {total:,} rows...")

    print(f"\nFinished. Wrote {written:,} rows to {output_path}")
    if written != total:
        raise SystemExit(
            f"Row count mismatch: scanned {total:,} but wrote {written:,}.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=DEFAULT_INPUT,
                        help=f"PMC JSON array (default: {DEFAULT_INPUT})")
    parser.add_argument("--output", default=DEFAULT_OUTPUT,
                        help=f"output CSV (default: {DEFAULT_OUTPUT})")
    args = parser.parse_args()

    # Relative paths resolve against the data dir, never the caller's cwd, so
    # output never lands in this tracked package by accident.
    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = DATA_DIR / input_path

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = DATA_DIR / output_path

    if not input_path.exists():
        raise SystemExit(
            f"Input not found: {input_path}\n"
            f"Set $PMC_DATA_DIR if the PMC data lives elsewhere.")

    convert(input_path, output_path)


if __name__ == "__main__":
    main()
