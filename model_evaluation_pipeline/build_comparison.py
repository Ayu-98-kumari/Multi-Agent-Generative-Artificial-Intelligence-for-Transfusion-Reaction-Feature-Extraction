"""Phase 2 of the model-comparison experiment: build the side-by-side CSV.

Standalone post-evaluation script -- not part of any pipeline. It reads the three
output CSVs produced by run_models.sh (the SAME input scored by three different
models) and writes ONE comparison CSV: one row per summary, with each model's
extracted-features JSON in its own column.

    id, summary, features_qwen, features_gpt120b, features_kimi

Rows are joined on a stable content ID: the first 16 hex chars of the SHA-256 of
the summary text. The hash is computed over the WHOLE summary, so the same
summary always gets the same ID across every run and every future batch -- this
scales past the current 5-row sample without needing an ID column in the data.
16 hex chars = 64 bits, so collisions between different summaries are effectively
impossible even at millions of rows.

The input CSV, the per-model outputs and this comparison CSV all live together in
model_evaluation_pipeline/input_output/. Pass a path to score a CSV elsewhere;
with no argument the script uses the single CSV sitting in that folder.
"""

import argparse
import hashlib
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
IO_DIR = SCRIPT_DIR / "input_output"

SUMMARY_COL = "Clinical Text"
FEATURES_COL = "Extracted Features"
ID_LEN = 16

# The model tags, in output column order. The per-run source CSVs and the
# comparison output are named from the input stem (see run_models.sh), so the
# same <stem>_output_<tag>.csv files this reads are the ones that script wrote.
TAGS = ("qwen", "gpt120b", "kimi")


def content_id(text):
    """Stable ID = first ID_LEN hex chars of SHA-256 over the whole summary."""
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()[:ID_LEN]


def resolve_default_input():
    """The single non-generated CSV in input_output/, or a helpful exit."""
    csvs = [p for p in sorted(IO_DIR.glob("*.csv"))
            if "_output_" not in p.name
            and not p.name.endswith("_model_comparison.csv")]
    if not csvs:
        raise SystemExit(
            f"No input CSV found.\nPut your CSV in: {IO_DIR}\n"
            f"or pass a file path as the input argument.")
    if len(csvs) > 1:
        names = "\n".join(f"  - {p.name}" for p in csvs)
        raise SystemExit(
            f"Multiple input CSVs found in {IO_DIR}:\n{names}\n"
            f"Pass the one you want as the input argument.")
    return csvs[0]


def load_run(tag, path):
    if not path.exists():
        raise SystemExit(
            f"Missing run output for '{tag}': {path}\n"
            f"Run model_evaluation_pipeline/run_models.sh first.")
    df = pd.read_csv(path)
    for col in (SUMMARY_COL, FEATURES_COL):
        if col not in df.columns:
            raise SystemExit(f"{path} has no '{col}' column.")
    df = df[[SUMMARY_COL, FEATURES_COL]].copy()
    df["id"] = df[SUMMARY_COL].map(content_id)
    if df["id"].duplicated().any():
        dups = df.loc[df["id"].duplicated(keep=False), SUMMARY_COL].tolist()
        raise SystemExit(
            f"Duplicate summary IDs within {path.name} -- the same summary text "
            f"appears more than once. Aborting.\n{dups}")
    return df


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input", nargs="?", default=None,
        help="The input CSV that was scored. Default: the single CSV in "
             "model_evaluation_pipeline/input_output/. Per-model outputs and the "
             "comparison CSV are found/written beside it, named from its stem.")
    parser.add_argument(
        "-o", "--output", default=None,
        help="Path for the comparison CSV. Defaults to "
             "<input-stem>_model_comparison.csv beside the input.")
    args = parser.parse_args()

    input_path = Path(args.input) if args.input else resolve_default_input()
    out_dir = input_path.parent
    stem = input_path.stem
    out_path = (Path(args.output) if args.output
                else out_dir / f"{stem}_model_comparison.csv")

    runs = [(tag, load_run(tag, out_dir / f"{stem}_output_{tag}.csv"))
            for tag in TAGS]

    # Every run must cover exactly the same set of summaries.
    base_tag, base_df = runs[0]
    base_ids = set(base_df["id"])
    for tag, df in runs[1:]:
        ids = set(df["id"])
        if ids != base_ids:
            missing, extra = base_ids - ids, ids - base_ids
            raise SystemExit(
                f"Run '{tag}' does not cover the same summaries as '{base_tag}': "
                f"missing {len(missing)}, extra {len(extra)}. Aborting.")

    # Start from id + summary, then merge each model's features column onto it.
    out = base_df[["id", SUMMARY_COL]].rename(columns={SUMMARY_COL: "summary"})
    for tag, df in runs:
        col = f"features_{tag}"
        out = out.merge(
            df[["id", FEATURES_COL]].rename(columns={FEATURES_COL: col}),
            on="id", how="inner")

    if len(out) != len(base_ids):
        raise SystemExit(
            f"Join produced {len(out)} rows from {len(base_ids)} summaries -- "
            f"unexpected fan-out. Aborting.")

    out.to_csv(out_path, index=False)
    print(f"Wrote {len(out)} rows x {len(out.columns)} cols to: {out_path}")
    print(f"Columns: {list(out.columns)}")


if __name__ == "__main__":
    main()
