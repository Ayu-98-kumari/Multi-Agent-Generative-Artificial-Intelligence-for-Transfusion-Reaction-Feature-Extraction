"""Format-agnostic CLI: extract features from a ``.xlsx`` OR a ``.csv`` input.

    python -m clinical_feature_extractor.run --input <file.xlsx|file.csv> \
        --output output/extracted.csv

The input format is auto-detected by extension. ``.xlsx``/``.xlsm`` use the
existing pipeline; ``.csv`` uses the CSV adapter. This is a separate entrypoint
from ``cli.py`` (which stays xlsx-only and unchanged).

Flags:
    --no-cache        ignore/refresh the per-row cache and re-call the model
    --single-row N    process just one row, by ABSOLUTE row number (e.g. 2 = the
                      first data row) — for a cheap first real run
    -v                verbose (list off-list drops with their row)

The extraction step needs GROQ_API_KEY in the project-root .env.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import config
from .extractor import ExtractionError
from .tabular_pipeline import PipelineError, render_report, run_extraction
from .vocabulary import VocabularyError


def _resolve(path_arg: str | None, default: Path) -> Path:
    if not path_arg:
        return default
    p = Path(path_arg)
    return p if p.is_absolute() or p.parent != Path(".") else (config.PACKAGE_DIR / p)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="clinical_feature_extractor.run",
        description="Extract canonical clinical features from a .xlsx or .csv of "
                    "clinical notes and export to CSV.")
    parser.add_argument("--input", "-i", default=None,
                        help="input .xlsx/.xlsm or .csv "
                             "(default: input/Automatic_extraction.xlsx).")
    parser.add_argument("--output", "-o", default=None,
                        help="output .csv (default: output/extracted.csv).")
    parser.add_argument("--no-cache", action="store_true",
                        help="ignore cached row results and re-call the model.")
    parser.add_argument("--single-row", type=int, default=None, metavar="N",
                        help="process one row only, by absolute row number.")
    parser.add_argument("--no-write-back", action="store_true",
                        help="leave the input file untouched; write only the "
                             "output CSV (CSV inputs only).")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    input_path = _resolve(args.input, config.INPUT_XLSX_DEFAULT)
    output_csv = _resolve(args.output, config.OUTPUT_CSV_DEFAULT)

    try:
        report = run_extraction(
            input_path, output_csv,
            use_cache=not args.no_cache, single_row=args.single_row,
            write_back=not args.no_write_back)
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
        for rec in report["dropped_records"]:
            print(f"  row {rec['row']}: dropped off-list feature {rec['feature']!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
