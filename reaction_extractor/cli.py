"""Command-line entrypoint for the reaction feature extractor.

    python -m reaction_extractor.cli --input <parsed.json> --output reaction_features.json

Flags:
    --no-cache        ignore/refresh the per-reaction features cache
    --single <sel>    process just one reaction (by page number or name substring),
                      for a cheap first real Groq run
    --unique-features build output/unique_features.json from the existing
                      output/reaction_features.json (deterministic, no Groq)
    --extended-features
                      build output/extended_features.json from the existing
                      output/unique_features.json (deterministic, no Groq)
    --symptoms-labmarkers
                      build output/symptoms_labmarkers.json from the existing
                      output/reaction_features.json — signs_symptoms then
                      lab_markers only, time_window ignored (deterministic, no Groq)
    --extended-symptoms-labmarkers
                      build output/extended_symptoms_labmarkers.json from the
                      existing output/symptoms_labmarkers.json (deterministic, no Groq)
    --dedup-features  build output/deduplicated_symptoms_labmarkers.json from the
                      existing output/extended_symptoms_labmarkers.json by removing
                      plain-English paraphrase duplicates (Groq VLM; honors
                      --no-cache)
    -v                verbose
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import config
from .extractor import ExtractionError
from .features import FeatureError
from .pipeline import Pipeline
from .unique_features_extractor import build_unique_features, render_report
from .extended_features_generator import (
    build_extended_features,
    render_report as render_extended_report,
)
from .symptoms_labmarkers_extractor import (
    build_symptoms_labmarkers,
    render_report as render_symptoms_labmarkers_report,
)
from .extended_symptoms_labmarkers_generator import (
    build_extended_symptoms_labmarkers,
    render_report as render_extended_symptoms_labmarkers_report,
)
from .dedup_features import (
    build_deduplicated_symptoms_labmarkers,
    render_report as render_dedup_report,
)


def _resolve_output(output: str) -> Path:
    p = Path(output)
    # A bare filename lands in the module's output/ dir; a path is respected.
    if p.parent == Path("."):
        return config.OUTPUT_DIR / p.name
    return p


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="reaction_extractor",
        description="Build the 3-column reaction/definition/features JSON from a "
                    "pdf_parser output JSON.")
    parser.add_argument("--input", "-i", default=str(config.INPUT_JSON_DEFAULT),
                        help="parsed-document JSON (pdf_parser output).")
    parser.add_argument("--output", "-o", default=config.OUTPUT_FILENAME_DEFAULT,
                        help="output JSON (bare name -> output/ dir).")
    parser.add_argument("--no-cache", action="store_true",
                        help="ignore cached feature results and re-call the model.")
    parser.add_argument("--single", metavar="SEL", default=None,
                        help="process one reaction only, by page number or name "
                             "substring (cheap first real run).")
    parser.add_argument("--unique-features", action="store_true",
                        help="build output/unique_features.json from the existing "
                             "reaction_features.json (deterministic, no Groq).")
    parser.add_argument("--extended-features", action="store_true",
                        help="build output/extended_features.json from the existing "
                             "unique_features.json (deterministic, no Groq).")
    parser.add_argument("--symptoms-labmarkers", action="store_true",
                        help="build output/symptoms_labmarkers.json from the "
                             "existing reaction_features.json — signs_symptoms then "
                             "lab_markers, time_window ignored (no Groq).")
    parser.add_argument("--extended-symptoms-labmarkers", action="store_true",
                        help="build output/extended_symptoms_labmarkers.json from "
                             "the existing symptoms_labmarkers.json (no Groq).")
    parser.add_argument("--dedup-features", action="store_true",
                        help="build output/deduplicated_symptoms_labmarkers.json "
                             "by removing plain-English paraphrase duplicates "
                             "(Groq VLM; honors --no-cache).")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    # Deterministic post-processing step: read the existing features JSON and
    # emit the deduplicated unique-features table. Short-circuits before any
    # Groq work.
    if args.unique_features:
        report = build_unique_features()
        print(render_report(report))
        return 0

    # Deterministic: expand unique_features.json into interleaved
    # original/negated pairs. Short-circuits before any Groq work.
    if args.extended_features:
        report = build_extended_features()
        print(render_extended_report(report))
        return 0

    # Deterministic: emit the signs_symptoms + lab_markers table (time_window
    # ignored). Short-circuits before any Groq work.
    if args.symptoms_labmarkers:
        report = build_symptoms_labmarkers()
        print(render_symptoms_labmarkers_report(report))
        return 0

    # Deterministic: expand symptoms_labmarkers.json into interleaved
    # original/negated pairs. Short-circuits before any Groq work.
    if args.extended_symptoms_labmarkers:
        report = build_extended_symptoms_labmarkers()
        print(render_extended_symptoms_labmarkers_report(report))
        return 0

    # VLM-driven: remove plain-English paraphrase duplicates from the extended
    # list. Honors --no-cache; short-circuits before the features pipeline.
    if args.dedup_features:
        try:
            report = build_deduplicated_symptoms_labmarkers(
                use_cache=not args.no_cache)
        except FeatureError as exc:
            print(f"[dedup] {exc}", file=sys.stderr)
            return 3
        print(render_dedup_report(report))
        return 0

    input_path = Path(args.input)
    output_path = _resolve_output(args.output)

    pipeline = Pipeline()
    try:
        document, report = pipeline.run(
            input_path, output_path,
            use_cache=not args.no_cache, only=args.single)
    except ExtractionError as exc:
        print(f"[extractor] {exc}", file=sys.stderr)
        return 2
    except FeatureError as exc:
        print(f"[features] {exc}", file=sys.stderr)
        return 3

    count = document["metadata"]["count"]
    print(f"Wrote {count} reaction(s) -> {output_path}")
    if args.verbose:
        for r in document["reactions"]:
            f = r["features"]
            tw = "✓" if f.get("time_window") else "–"
            n_sym = len(f.get("signs_symptoms") or [])
            n_lab = len(f.get("lab_markers") or [])
            print(f"  p{r['source_page']:>2} {r['reaction']}: "
                  f"time_window {tw}, {n_sym} symptom(s), {n_lab} lab(s)")
    print(report.render())
    # Flags are advisory (surface for review); they do not fail the run.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
