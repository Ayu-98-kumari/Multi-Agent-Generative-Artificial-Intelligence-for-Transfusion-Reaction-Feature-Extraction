# model_evaluation_pipeline

Post-evaluation experiment (not part of any production pipeline): score the same
clinical notes with **three different models** and lay their extracted features
side by side, so they can be compared directly.

The models: `qwen/qwen3.6-27b` (the pipeline's current default) and `openai/gpt-oss-120b`,
both served through Groq, plus `moonshotai/kimi-k3`, served through OpenRouter.

## How the model is switched

`clinical_feature_extractor/config.py` reads these optional env vars:

- `GROQ_MODEL` — the model id, used for all five extraction stages.
- `GROQ_REASONING_EFFORT` — the reasoning effort.
- `GROQ_MAX_COMPLETION_TOKENS` — the per-call output-token budget.
- `LLM_PROVIDER=openrouter` — routes every stage through OpenRouter instead of Groq
  (used for the Kimi arm); the model, provider, and reasoning effort then come from
  the `OPENROUTER_*` settings (defaults: `moonshotai/kimi-k3`, Moonshot, `low`).

**Unset, all keep their old defaults** (`qwen/qwen3.6-27b`, `none`, `16384`, Groq),
so the pipeline behaves exactly as before. These backward-compatible hooks are the
only change to the main pipeline.

Both `reasoning_effort` and `max_completion_tokens` have to vary by model:

- **reasoning_effort** — qwen **requires** `none` (its `<think>` prefix breaks JSON
  mode); gpt-oss-120b **rejects** `none` (Groq 400 — must be low/medium/high) and
  runs at `medium` (the project's gpt-oss triager value); the Kimi arm runs at
  `low` (`OPENROUTER_REASONING_EFFORT`) to bound latency and cost.
- **max_completion_tokens** — qwen caps at `16384` (its model max); gpt-oss-120b
  allows up to `65536` and runs at `50000` here — well under this account's
  measured Groq rate limit of 250,000 tokens/min; the Kimi arm runs at `20000`.
  One flat budget per call, no token-escalation retry (a `length` finish is a hard
  error).

| run | model | reasoning_effort | max_completion_tokens |
|---|---|---|---|
| qwen | `qwen/qwen3.6-27b` | `none` | `16384` |
| gpt120b | `openai/gpt-oss-120b` | `medium` | `50000` |
| kimi | `moonshotai/kimi-k3` (via `LLM_PROVIDER=openrouter`) | `low` | `20000` |

## Two phases

### 1. `run_models.sh` — run all three models (and build the comparison)

Put one input CSV in [`input_output/`](input_output/) (exact format: see its
[README](input_output/README.md)) and run, from the repository root:

```bash
bash model_evaluation_pipeline/run_models.sh                     # uses the CSV in input_output/
bash model_evaluation_pipeline/run_models.sh path/to/notes.csv   # or any CSV elsewhere
bash model_evaluation_pipeline/run_models.sh --fresh             # ignore caches, call every model again
```

With no argument it uses the single CSV in `input_output/`; if the folder is
empty it stops and tells you where to put one. Each model writes
`<input-stem>_output_{qwen,gpt120b,kimi}.csv` **next to the input**, and the
comparison CSV (phase 2) is then built automatically.

- **Resumable.** Each model caches its calls in its own folder next to the input
  (`.cache_qwen/`, `.cache_gpt120b/`, `.cache_kimi/` — one per model, because the
  cache key does not include the model name). Progress is printed per note and
  the output CSV is saved after every note, so if a run stops part-way, running
  the same command again re-uses everything already done and only calls the
  models for what is left.
- `--fresh` — turns the cache off for a clean run. Use it after changing a prompt;
  otherwise the cache can return the earlier answers.
- `--no-write-back` is always passed — the input CSV is never modified.

Add `-n N` (`--single-row N`) to score only spreadsheet row `N` with all three
models — a cheap smoke test before committing to the whole file. Its outputs are
suffixed `<input-stem>_row<N>_output_<tag>.csv`, so a quick check never
overwrites a real full run, and the comparison step is skipped:

```bash
bash model_evaluation_pipeline/run_models.sh -n 2
```

Needs `GROQ_API_KEY` in the repo-root `.env` for the qwen/gpt-oss runs, and
`OPENROUTER_API_KEY` for the Kimi run (template: `.env.example`).

### 2. `build_comparison.py` — join them into one CSV

Runs automatically at the end of `run_models.sh`. To rebuild it on its own (no
model calls), from the repository root:

```bash
.venv/bin/python -m model_evaluation_pipeline.build_comparison                    # the CSV in input_output/
.venv/bin/python -m model_evaluation_pipeline.build_comparison path/to/notes.csv  # the same input you scored
```

Emits `<input-stem>_model_comparison.csv` beside the input, one row per summary:

| column | contents |
|---|---|
| `id` | first 16 hex chars of SHA-256(summary text) — stable across runs and future batches |
| `summary` | the clinical note |
| `features_qwen` | qwen's full `Extracted Features` JSON (feature, evidence, time, schema) |
| `features_gpt120b` | gpt-oss-120b's full features JSON |
| `features_kimi` | kimi-k3's full features JSON |

Rows are joined on `id`; the script aborts if the three runs don't cover exactly
the same set of summaries.

### 3. `model_comparison_viewer.html` — read it in the browser

A standalone local viewer (no server, nothing uploaded). Open the file in a
browser and drag in the comparison CSV, or click **Open comparison CSV…**. It
shows each summary once with every model's features **side by side**, one column
per model, using the same present / absence / time / transfusion-adjacent badges
as the main pipeline's `output_viewer.html`.

It auto-detects the `summary` column and every `features_*` column, so it keeps
working if the set of models changes. Per-column and per-run feature counts are
shown, there's a text filter, and **Export to Excel** writes a shareable table
(Row, id, Summary, one column per model). The CSV stays local — open the HTML
from the tracked folder and point it at the gitignored data.

## Boundaries

This folder holds **code only** and imports nothing from the pipeline it drives
beyond invoking its CLI. All data — inputs, outputs and caches — stays in the
gitignored [`input_output/`](input_output/) folder, or next to whatever input you
point it at (e.g. under the gitignored `kept-gitignored/`).
