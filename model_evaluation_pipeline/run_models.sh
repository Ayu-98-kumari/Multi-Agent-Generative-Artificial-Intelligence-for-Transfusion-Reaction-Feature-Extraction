#!/usr/bin/env bash
# Phase 1 of the model-comparison experiment.
#
# Runs the main pipeline (clinical_feature_extractor) THREE times on the SAME
# input CSV, once per model, each to its own output CSV. Only the model changes
# -- via the GROQ_MODEL env override in clinical_feature_extractor/config.py.
# Every other hyperparameter (temperature, reasoning_effort, prompts, vocab) is
# whatever the pipeline already uses, so the three outputs differ only by model.
#
# Each model runs with its OWN per-model cache dir (<out-dir>/.cache_<tag>/) and
# CLF_PROGRESS on, so a crash mid-run keeps the notes already finished and a
# re-run RESUMES from cache instead of restarting. Per-model dirs are required
# because the cache key does not include the model name -- a shared dir would
# serve one model's answer to another. --no-write-back leaves the input CSV
# untouched. Pass --fresh to ignore the cache and force clean model calls (e.g.
# after editing a prompt). After the three runs the side-by-side comparison CSV
# is built automatically.
#
# The input CSV is the first argument. If omitted, the script looks for a single
# CSV in model_evaluation_pipeline/input_output/ and uses that; if that folder is
# empty it tells you where to put a file. Each model's output is written next to
# the input as <input-stem>_output_<tag>.csv, so pointing this at any CSV Just
# Works.
#
#   ./run_models.sh                   # use the CSV placed in input_output/
#   ./run_models.sh path/to/notes.csv # or point at any CSV (e.g. to reproduce the paper)
#   ./run_models.sh --fresh           # ignore the cache; call every model from scratch
#
# Pass -n/--single-row N to score only row N with every model -- a cheap smoke
# test that the three models work before committing to the full file. Its
# outputs are suffixed <input-stem>_row<N>_output_<tag>.csv so they never
# overwrite a real full run (the comparison step is skipped for a smoke test):
#
#   ./run_models.sh -n 2 path/to/notes.csv
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PY="$REPO_ROOT/.venv/bin/python"
IO_DIR="$REPO_ROOT/model_evaluation_pipeline/input_output"

# Parse an optional -n/--single-row N and an optional positional input CSV.
SINGLE_ROW=""
INPUT=""
FRESH=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    -n|--single-row)
      SINGLE_ROW="${2:-}"
      if [[ -z "$SINGLE_ROW" ]]; then
        echo "$1 needs a row number" >&2
        exit 1
      fi
      shift 2
      ;;
    --fresh)
      FRESH=1
      shift
      ;;
    -h|--help)
      grep '^#' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    -*)
      echo "Unknown option: $1" >&2
      exit 1
      ;;
    *)
      if [[ -n "$INPUT" ]]; then
        echo "Unexpected extra argument: $1" >&2
        exit 1
      fi
      INPUT="$1"
      shift
      ;;
  esac
done
# No input path given: use the single CSV sitting in input_output/. Generated
# files from an earlier run (outputs and the comparison CSV) are ignored so a
# re-run always finds the original input, not its own results.
if [[ -z "$INPUT" ]]; then
  shopt -s nullglob
  candidates=()
  for f in "$IO_DIR"/*.csv; do
    case "$(basename "$f")" in
      *_output_*.csv|*_model_comparison.csv) ;;   # skip generated files
      *) candidates+=("$f") ;;
    esac
  done
  shopt -u nullglob
  if [[ ${#candidates[@]} -eq 0 ]]; then
    echo "No input CSV found." >&2
    echo "Put your CSV in: $IO_DIR" >&2
    echo "or pass a file path:  ./run_models.sh path/to/notes.csv" >&2
    exit 1
  elif [[ ${#candidates[@]} -gt 1 ]]; then
    echo "Multiple input CSVs found in $IO_DIR:" >&2
    for f in "${candidates[@]}"; do echo "  - $(basename "$f")" >&2; done
    echo "Pass the one you want:  ./run_models.sh path/to/notes.csv" >&2
    exit 1
  fi
  INPUT="${candidates[0]}"
fi

if [[ ! -f "$INPUT" ]]; then
  echo "Input not found: $INPUT" >&2
  exit 1
fi

# All outputs land beside the input, named from its stem. A single-row smoke
# test gets a _row<N> suffix so it never clobbers a real full-file run.
OUT_DIR="$(cd "$(dirname "$INPUT")" && pwd)"
STEM="$(basename "$INPUT" .csv)"
# Caching is ON by default (each model to its own dir, set in run_one) so a crash
# can resume; --fresh switches it off for a clean, cache-free run.
EXTRA_ARGS=(--no-write-back)
if [[ -n "$FRESH" ]]; then
  EXTRA_ARGS+=(--no-cache)
fi
if [[ -n "$SINGLE_ROW" ]]; then
  STEM="${STEM}_row${SINGLE_ROW}"
  EXTRA_ARGS+=(--single-row "$SINGLE_ROW")
fi

run_one () {
  local model="$1" tag="$2" effort="$3" max_tokens="$4" provider="${5:-groq}"
  local out="$OUT_DIR/${STEM}_output_${tag}.csv"
  # Each model gets its OWN cache dir (the cache key omits the model name, so a
  # shared dir would cross-contaminate). CLF_PROGRESS gives per-note logs and
  # flushes the output CSV after every note, so a crash keeps finished work.
  local cache_dir="$OUT_DIR/.cache_${tag}"
  echo ">>> [$tag] provider=$provider model=$model reasoning_effort=$effort max_completion_tokens=$max_tokens"
  echo "    -> $out"
  CLF_PROGRESS=1 CLF_CACHE_DIR="$cache_dir" \
    LLM_PROVIDER="$provider" GROQ_MODEL="$model" GROQ_REASONING_EFFORT="$effort" \
    GROQ_MAX_COMPLETION_TOKENS="$max_tokens" "$PY" \
    -m clinical_feature_extractor.run \
    --input "$INPUT" --output "$out" "${EXTRA_ARGS[@]}"
  echo ""
}

# Two per-model hyperparameters, each set to what the model requires/supports:
#   reasoning_effort -- qwen MUST be "none" (its <think> prefix breaks JSON mode);
#     gpt-oss-120b runs at "medium" (the project's gpt-oss triager value).
#   max_completion_tokens -- qwen caps at 16384 (its model max); the gpt-oss models
#     cap at 65536 and run at 50000 (well under the 250000/min account rate limit).
run_one "qwen/qwen3.6-27b"    "qwen"    "none"   16384
run_one "openai/gpt-oss-120b" "gpt120b" "medium" 50000
# Kimi K3 via OpenRouter (LLM_PROVIDER=openrouter): pinned to Moonshot with
# fallbacks off; temperature is omitted (fixed 1.0 server-side) and reasoning is
# capped to "low" (config.OPENROUTER_REASONING_EFFORT) to bound latency/cost. The
# 20000 output budget clears the longest notes' features without truncating.
run_one "moonshotai/kimi-k3"  "kimi"    "none"   20000 "openrouter"

echo "All runs complete."

# Phase 2 runs automatically: build the side-by-side comparison from the three
# per-model outputs. Skipped for a single-row smoke test (its row-suffixed
# outputs are not what the comparison reads). Soft-fail: if a model output is
# missing, the per-model CSVs are left intact and you can build it by hand later.
if [[ -n "$SINGLE_ROW" ]]; then
  echo "Single-row smoke test -- skipping comparison build."
else
  echo ""
  echo ">>> Building comparison CSV from the three model outputs..."
  if ! "$PY" -m model_evaluation_pipeline.build_comparison "$INPUT"; then
    echo "NOTE: model runs finished, but the comparison could not be built (see error above)." >&2
    echo "      Your per-model output CSVs are intact; fix the issue and re-run:" >&2
    echo "        $PY -m model_evaluation_pipeline.build_comparison \"$INPUT\"" >&2
  fi
fi
