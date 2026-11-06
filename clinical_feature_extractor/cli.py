"""Command-line entrypoint for the clinical feature extractor.

    python -m clinical_feature_extractor.cli \
        --input input/Automatic_extraction.xlsx --output output/extracted.csv

Flags:
    --no-cache        ignore/refresh the per-row cache and re-call the model
    --single-row N    process just one row, by ABSOLUTE worksheet row number
                      (e.g. 2 = the first data row) — for a cheap first real run
    -v                verbose (list per-row feature counts)

The extraction step needs GROQ_API_KEY in the project-root .env.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import config
from .extractor import ExtractionError
from .pipeline import Pipeline, PipelineError, render_report
from .vocabulary import VocabularyError


def _resolve(path_arg: str | None, default: Path) -> Path:
    if not path_arg:
        return default
    p = Path(path_arg)
    # A bare filename resolves against the module dir sensibly; a path is respected.
    return p if p.is_absolute() or p.parent != Path(".") else (config.PACKAGE_DIR / p)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="clinical_feature_extractor",
        description="Extract canonical clinical features from spreadsheet clinical "
                    "notes and export to CSV.")
    parser.add_argument("--input", "-i", default=None,
                        help="input .xlsx (default: input/Automatic_extraction.xlsx).")
    parser.add_argument("--output", "-o", default=None,
                        help="output .csv (default: output/extracted.csv).")
    parser.add_argument("--no-cache", action="store_true",
                        help="ignore cached row results and re-call the model.")
    parser.add_argument("--single-row", type=int, default=None, metavar="N",
                        help="process one row only, by absolute worksheet row "
                             "number (cheap first real run).")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    input_xlsx = _resolve(args.input, config.INPUT_XLSX_DEFAULT)
    output_csv = _resolve(args.output, config.OUTPUT_CSV_DEFAULT)

    try:
        pipeline = Pipeline()
        report = pipeline.run(
            input_xlsx, output_csv,
            use_cache=not args.no_cache, single_row=args.single_row)
    except VocabularyError as exc:
        print(f"[vocabulary] {exc}", file=sys.stderr)
        return 2
    except PipelineError as exc:
        print(f"[pipeline] {exc}", file=sys.stderr)
        return 3
    except ExtractionError as exc:
        print(f"[extractor] {exc}", file=sys.stderr)
        return 4

    print(render_report(report))
    if args.verbose:
        # Off-list drops, with the row they came from, for quick review.
        for rec in report["dropped_records"]:
            print(f"  row {rec['row']}: dropped off-list feature "
                  f"{rec['feature']!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
