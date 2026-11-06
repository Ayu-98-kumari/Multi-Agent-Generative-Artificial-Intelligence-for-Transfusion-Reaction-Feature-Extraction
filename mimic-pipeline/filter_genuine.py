"""Extract the genuine transfusion-reaction cases into a single cohort file.

Stage 3 of the funnel, and a pure join -- no model calls. It takes the triage
verdicts (``mimic_transfusion_cases_triage_output.csv``) and the screened notes
(``mimic_transfusion_cases.csv``, which still carries the full ``text`` and every
original column), keeps only the rows the triager graded ``genuine``, and writes
them out with the triage rationale attached.

The result is the audit-ready cohort: for each genuine case, all of its source
columns plus the verdict's ``reason`` and ``supporting_quote``, so a reviewer can
see both the full note and why it was kept.

Deterministic (no API key needed). Re-run any time the triage output changes.

Code lives in this tracked package; MIMIC data and everything derived from it
stay in the gitignored data directory. Override with $MIMIC_DATA_DIR or the
--* flags.
"""

import argparse
import os
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent

# DUA-covered data (input CSVs and this script's output) lives outside this
# tracked package.
DATA_DIR = Path(os.environ.get(
    "MIMIC_DATA_DIR", SCRIPT_DIR.parent / "kept-gitignored" / "mimic-iv-note"))

DEFAULT_TRIAGE = "mimic_transfusion_cases_triage_output.csv"
DEFAULT_SCREENED = "mimic_transfusion_cases.csv"
DEFAULT_OUTPUT = "mimic_transfusion_genuine(424).csv"

JOIN_KEY = "note_id"
GENUINE = "genuine"
# The triage columns worth carrying onto each genuine row.
TRIAGE_COLS = ["verdict", "reason", "supporting_quote"]


def filter_genuine(triage_file, screened_file, output_file):
    triage = pd.read_csv(triage_file)
    screened = pd.read_csv(screened_file)

    if JOIN_KEY not in triage.columns:
        raise SystemExit(f"{triage_file} has no '{JOIN_KEY}' column.")
    if JOIN_KEY not in screened.columns:
        raise SystemExit(f"{screened_file} has no '{JOIN_KEY}' column.")
    if "verdict" not in triage.columns:
        raise SystemExit(f"{triage_file} has no 'verdict' column.")

    genuine = triage[triage["verdict"] == GENUINE]

    # Attach the triage rationale to the full screened row (all source columns +
    # note text). Inner-join from genuine onto screened so we keep every original
    # column and add verdict/reason/supporting_quote.
    keep = [JOIN_KEY] + [c for c in TRIAGE_COLS if c in genuine.columns]
    merged = screened.merge(genuine[keep], on=JOIN_KEY, how="inner")

    # One triage verdict per note; guard against an unexpected fan-out.
    if len(merged) != len(genuine):
        raise SystemExit(
            f"Join produced {len(merged)} rows from {len(genuine)} genuine "
            f"verdicts -- duplicate '{JOIN_KEY}' values? Aborting rather than "
            f"writing a mismatched cohort.")

    merged.to_csv(output_file, index=False)
    return len(merged)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--triage", default=DEFAULT_TRIAGE,
                        help=f"triage results CSV (default: {DEFAULT_TRIAGE})")
    parser.add_argument("--screened", default=DEFAULT_SCREENED,
                        help=f"screened notes CSV (default: {DEFAULT_SCREENED})")
    parser.add_argument("--output", default=DEFAULT_OUTPUT,
                        help=f"genuine cohort CSV (default: {DEFAULT_OUTPUT})")
    args = parser.parse_args()

    def resolve(p):
        p = Path(p)
        return p if p.is_absolute() else DATA_DIR / p

    triage_file = resolve(args.triage)
    screened_file = resolve(args.screened)
    output_file = resolve(args.output)

    for f in (triage_file, screened_file):
        if not f.exists():
            raise SystemExit(
                f"Input not found: {f}\n"
                f"Run the screen + triage steps first, or set $MIMIC_DATA_DIR.")

    n = filter_genuine(triage_file, screened_file, output_file)

    print(f"Genuine cases written: {n}")
    print(f"Saved to: {output_file}")
    if "(424)" in Path(args.output).name and n != 424:
        print(f"NOTE: the filename says 424 but this run found {n} genuine cases; "
              f"rename the file if that count is now the source of truth.")


if __name__ == "__main__":
    main()
