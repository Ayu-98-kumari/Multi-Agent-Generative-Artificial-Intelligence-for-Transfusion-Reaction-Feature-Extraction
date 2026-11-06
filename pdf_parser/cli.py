"""Command-line entrypoint.

    python -m pdf_parser.cli --input <pdf> --output <json>

Stage flags let you run pieces independently:
    --skeleton-only      run Pass 1 only; write + summarize skeleton.json (no VLM)
    --single-page N      parse only page N via the VLM (cheap, isolated first run)
    --skip-validation    parse + write JSON but don't run the validation gate
    --force-skeleton     re-extract the skeleton even if cached
    --no-cache           don't read/write the per-page VLM cache

Exit code is non-zero when the validation gate reports hard failures.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from . import config
from .pipeline import Pipeline


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pdf_parser",
        description="Parse a well-structured clinical PDF into faithful JSON.")
    p.add_argument("--input", "-i", required=True, help="input PDF path")
    p.add_argument("--output", "-o", help="output JSON path "
                   "(default: <stem>.json next to the input)")
    p.add_argument("--work-dir", help="dir for skeleton.json + per-page cache "
                   "(default: <stem>_pdfparser next to the input)")

    p.add_argument("--skeleton-only", action="store_true",
                   help="run Pass 1 only; write and summarize the skeleton")
    p.add_argument("--single-page", type=int, metavar="N",
                   help="parse only page N via the VLM; skip stitch/validate")
    p.add_argument("--skip-validation", action="store_true",
                   help="parse and write JSON without running validation")

    p.add_argument("--force-skeleton", action="store_true",
                   help="rebuild the skeleton even if a cached one exists")
    p.add_argument("--no-cache", action="store_true",
                   help="disable the per-page VLM result cache")
    p.add_argument("--coverage-threshold", type=float,
                   default=config.COVERAGE_THRESHOLD,
                   help=f"coverage recall threshold (default {config.COVERAGE_THRESHOLD})")
    p.add_argument("-v", "--verbose", action="count", default=0,
                   help="-v for INFO, -vv for DEBUG logging")
    return p


def _configure_logging(verbosity: int) -> None:
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")


def _summarize_skeleton(skeleton) -> str:
    tabs = skeleton.expected_tables()
    figs = skeleton.expected_figures()
    numbered = sum(1 for t in tabs if t.number > 0)
    unlabeled = len(tabs) - numbered          # archived reaction + version history
    lines = [
        f"  title       : {skeleton.title}",
        f"  version     : {skeleton.version}",
        f"  pages       : {skeleton.total_pages}",
        f"  sections    : {len(skeleton.sections)}",
        f"  tables      : {len(tabs)} ({numbered} numbered + {unlabeled} unlabeled)",
        f"  figures     : {len(figs)}",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    _configure_logging(args.verbose)

    pdf_path = Path(args.input)
    if not pdf_path.exists():
        print(f"error: input not found: {pdf_path}", file=sys.stderr)
        return 2

    pipeline = Pipeline(
        pdf_path,
        output_path=args.output,
        work_dir=args.work_dir,
        coverage_threshold=args.coverage_threshold,
        use_cache=not args.no_cache,
    )

    # --- skeleton only --------------------------------------------------- #
    if args.skeleton_only:
        skeleton = pipeline.load_or_build_skeleton(force=args.force_skeleton)
        print(f"skeleton -> {pipeline.skeleton_path}")
        print(_summarize_skeleton(skeleton))
        return 0

    # --- single page (isolated VLM call) -------------------------------- #
    if args.single_page is not None:
        skeleton = pipeline.load_or_build_skeleton(force=args.force_skeleton)
        n = args.single_page
        if not (1 <= n <= skeleton.total_pages):
            print(f"error: page {n} out of range 1..{skeleton.total_pages}",
                  file=sys.stderr)
            return 2
        page = pipeline.parse_single_page(skeleton, n)
        out = (Path(args.output) if args.output
               else pipeline.work_dir / f"page_{n:03d}.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(page.model_dump_json(indent=2), encoding="utf-8")
        print(f"page {n} -> {out}  ({len(page.blocks)} blocks)")
        return 0

    # --- full run -------------------------------------------------------- #
    result = pipeline.run(
        do_validate=not args.skip_validation,
        force_skeleton=args.force_skeleton,
    )
    print(f"document -> {result.output_path}")
    if result.report is not None:
        print(result.report.render())
        return 0 if result.report.ok else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
