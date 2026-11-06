"""Screen MIMIC-IV discharge summaries for transfusion-reaction mentions.

Reads the gzipped notes in chunks (never decompressed to disk, never fully in
memory) and writes every note matching at least one reaction pattern, tagged
with the reaction names it matched.

Ported from MIMIC_Read.ipynb. The full-phrase patterns are the notebook's
original set (843 matches on their own); this version additionally matches a
CASE-SENSITIVE bare acronym (TRALI, TACO, DHTR, ...) for each named reaction --
the same detection dictionary pmc-pipeline uses, so both datasets screen
identically. That lifts the recall-oriented pool from 843 to 1,210 out of
331,793 summaries; the triager judges precision. The acronym alternatives are
case-sensitive (only the full phrase is case-folded, via an inline (?i:...)
group), so e.g. "TTI" never matches inside "geTTIng".

Code lives here; MIMIC data does not. Both the source notes and everything
derived from them are covered by the PhysioNet DUA and must stay in the
gitignored data directory -- never in this tracked one. Override the location
with $MIMIC_DATA_DIR or --input/--output.
"""

import argparse
import os
import re
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent

# DUA-covered data lives outside this tracked package.
DATA_DIR = Path(os.environ.get(
    "MIMIC_DATA_DIR", SCRIPT_DIR.parent / "kept-gitignored" / "mimic-iv-note"))

DEFAULT_INPUT = "discharge.csv.gz"
DEFAULT_OUTPUT = "mimic_transfusion_cases.csv"
CHUNK_SIZE = 5000
PROGRESS_EVERY_N_CHUNKS = 5

# The 12 full phrases, each (except the three noted) also matching a
# CASE-SENSITIVE bare acronym: the full phrase is case-folded with an inline
# (?i:...) group, while the acronym outside it stays case-sensitive so only
# uppercase standalone letters match (never "getting" for TTI). GENERAL_TR, HTR
# and TAD get no acronym -- "TR" is too generic, and HTR/TAD collide with
# unrelated abbreviations. Identical to pmc-pipeline's dictionary.
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
    """Write notes mentioning a transfusion reaction to ``output_file``."""
    total_processed = 0
    total_matched = 0
    first_write = True

    for chunk_number, chunk in enumerate(
        pd.read_csv(
            input_file,
            compression="gzip",
            chunksize=CHUNK_SIZE,
        ),
        start=1,
    ):
        chunk["matched_reactions"] = [
            match_reactions(text)
            for text in chunk["text"].fillna("").astype(str)
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
                f"Processed: {total_processed:,} summaries | "
                f"Matches found: {total_matched:,}"
            )

    return total_processed, total_matched


def report_counts(output_file):
    """Print how many summaries matched each reaction pattern."""
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
        help=f"gzipped notes CSV (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"matching summaries CSV (default: {DEFAULT_OUTPUT})",
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
            f"Set $MIMIC_DATA_DIR if the MIMIC data lives elsewhere.")

    total_processed, total_matched = screen_notes(input_file, output_file)

    print("\nFinished.")
    print(f"Total summaries processed: {total_processed:,}")
    print(f"Total matching summaries: {total_matched:,}")
    print(f"Saved to: {output_file}")

    if total_matched:
        report_counts(output_file)


if __name__ == "__main__":
    main()
