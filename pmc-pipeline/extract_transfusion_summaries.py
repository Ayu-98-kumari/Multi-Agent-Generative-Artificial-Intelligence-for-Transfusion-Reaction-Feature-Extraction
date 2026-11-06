"""Screen PMC-Patients case reports for transfusion-reaction mentions.

Stage 1 of the same two-stage funnel used for MIMIC: a deterministic,
recall-oriented regex screen that narrows 250k case reports down to a candidate
pool, which stage 2 (LLM triage) then judges. It starts from the same 12
full-phrase, word-boundary-anchored regexes as mimic-pipeline, but deliberately
diverges in one way: PMC case reports are formal publications that lean heavily
on uppercase acronyms (TRALI, TACO, DHTR, ...), so each named reaction also
matches its bare acronym. That is pure recall -- the triager rules out the
noise. The acronym alternatives are CASE-SENSITIVE (only the full phrase is
case-folded, via an inline (?i:...) group), so e.g. "TTI" never resurfaces
inside "geTTIng". The input column also differs: PMC's narrative lives in
`patient`, MIMIC's in `text`.

Being recall-oriented, this lifts PMC from 114 matches (full phrases only) to
174 (with acronyms). A match is still not a reaction -- triage decides.

Reads the CSV in chunks so the 731 MB file is never fully in memory, and writes
every matching case -- with all its original columns -- tagged with a
`matched_reactions` column.

A regex match is not a reaction: reports discussing a reaction that was ruled
out match too. The output is a candidate pool for triage, not a positive cohort.

Code lives here; data does not. Both the source CSV and everything derived from
it stay in the gitignored data directory. Override with $PMC_DATA_DIR or
--input/--output.
"""

import argparse
import os
import re
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent

# Data (source + derived) lives outside this tracked package.
DATA_DIR = Path(os.environ.get(
    "PMC_DATA_DIR", SCRIPT_DIR.parent / "kept-gitignored" / "pmc-patients"))

DEFAULT_INPUT = "PMC-Patients-V2.csv"
DEFAULT_OUTPUT = "pmc_transfusion_cases.csv"
TEXT_COLUMN = "patient"
CHUNK_SIZE = 5000
PROGRESS_EVERY_N_CHUNKS = 5

# Starts from mimic-pipeline's 12 full phrases, then adds a CASE-SENSITIVE bare
# acronym for each named reaction (see module docstring). The full phrase is
# case-folded with an inline (?i:...) group; the acronym outside it stays
# case-sensitive so uppercase-only letters match and lowercase look-alikes
# (e.g. "getting" for TTI) do not. GENERAL_TR, HTR and TAD get no acronym --
# "TR" is too generic, and HTR/TAD collide with unrelated abbreviations.
REACTION_PATTERNS = {
    "GENERAL_TR": re.compile(
        r"\btransfusion[- ]reaction(?:s)?\b",
        re.IGNORECASE,
    ),
    "TTI": re.compile(
        r"(?i:\bTransfusion-transmitted infection\b)|\bTTI\b",
    ),
    "AHTR": re.compile(
        r"(?i:\bAcute hemolytic transfusion reaction\b)|\bAHTR\b",
    ),
    "TACO": re.compile(
        r"(?i:\bTransfusion-associated circulatory overload\b)|\bTACO\b",
    ),
    "TRALI": re.compile(
        r"(?i:\bTransfusion-related acute lung injury\b)|\bTRALI\b",
    ),
    "DHTR": re.compile(
        r"(?i:\bDelayed hemolytic transfusion reaction\b)|\bDHTR\b",
    ),
    "DSTR": re.compile(
        r"(?i:\bDelayed serologic transfusion reaction\b)|\bDSTR\b",
    ),
    "FNHTR": re.compile(
        r"(?i:\bFebrile non-hemolytic transfusion reaction\b)|\bFNHTR\b",
    ),
    "HTR": re.compile(
        r"\bHypotensive transfusion reaction\b",
        re.IGNORECASE,
    ),
    "PTP": re.compile(
        r"(?i:\bPost transfusion purpura\b)|\bPTP\b",
    ),
    "TAD": re.compile(
        r"\bTransfusion-associated dyspnea\b",
        re.IGNORECASE,
    ),
    "TAGVHD": re.compile(
        r"(?i:\bTransfusion-associated graft\s+vs\.?\s+host disease\b)|\bTAGVHD\b",
    ),
}


def match_reactions(text):
    """Return the comma-joined names of every pattern matching ``text``."""
    return ", ".join(
        reaction
        for reaction, pattern in REACTION_PATTERNS.items()
        if pattern.search(text)
    )


def screen_notes(input_file, output_file):
    """Write case reports mentioning a transfusion reaction to ``output_file``."""
    total_processed = 0
    total_matched = 0
    first_write = True

    for chunk_number, chunk in enumerate(
        pd.read_csv(input_file, chunksize=CHUNK_SIZE),
        start=1,
    ):
        chunk["matched_reactions"] = [
            match_reactions(text)
            for text in chunk[TEXT_COLUMN].fillna("").astype(str)
        ]

        matched_chunk = chunk[chunk["matched_reactions"] != ""].copy()

        if not matched_chunk.empty:
            matched_chunk.to_csv(
                output_file,
                mode="w" if first_write else "a",
                header=first_write,
                index=False,
            )

            first_write = False
            total_matched += len(matched_chunk)

        total_processed += len(chunk)

        if chunk_number % PROGRESS_EVERY_N_CHUNKS == 0:
            print(
                f"Processed: {total_processed:,} case reports | "
                f"Matches found: {total_matched:,}"
            )

    return total_processed, total_matched


def report_counts(output_file):
    """Print how many case reports matched each reaction pattern."""
    results = pd.read_csv(output_file, usecols=["matched_reactions"])

    reaction_counts = (
        results["matched_reactions"]
        .str.split(", ")
        .explode()
        .value_counts()
    )

    print("\nMatches per reaction:")
    print(reaction_counts.to_string())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT,
        help=f"PMC case-report CSV (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"matching case reports CSV (default: {DEFAULT_OUTPUT})",
    )
    args = parser.parse_args()

    # Relative paths resolve against the data directory, not the caller's cwd,
    # so output never lands in this tracked package by accident.
    input_file = Path(args.input)
    if not input_file.is_absolute():
        input_file = DATA_DIR / input_file

    output_file = Path(args.output)
    if not output_file.is_absolute():
        output_file = DATA_DIR / output_file

    if not input_file.exists():
        raise SystemExit(
            f"Input not found: {input_file}\n"
            f"Run convert_pmc_json_to_csv.py first, or set $PMC_DATA_DIR if the "
            f"PMC data lives elsewhere.")

    total_processed, total_matched = screen_notes(input_file, output_file)

    print("\nFinished.")
    print(f"Total case reports processed: {total_processed:,}")
    print(f"Total matching case reports: {total_matched:,}")
    print(f"Saved to: {output_file}")

    if total_matched:
        report_counts(output_file)


if __name__ == "__main__":
    main()
