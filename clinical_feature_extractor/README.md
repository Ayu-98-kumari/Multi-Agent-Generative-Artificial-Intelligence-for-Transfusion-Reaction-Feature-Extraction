# clinical_feature_extractor

A standalone module that reads clinical notes from a spreadsheet (**`.xlsx` or
`.csv`**), extracts clinical features against a fixed **canonical vocabulary**
and the **NHSN Hemovigilance hierarchical schema** using a Groq model, runs a
deterministic vocabulary post-check, and writes structured JSON (features with
timing and NHSN variables, treatments, and past medical history) back into the
file and out to CSV.

It does **not** modify `pdf_parser/` or `reaction_extractor/`. It only *reads*
two artifacts they produced, both read-only:

- `reaction_extractor/output/deduplicated_symptoms_labmarkers.json` — the
  canonical feature vocabulary.
- `NHSN_hierarchical_feature_schema_2.0.xlsx` — the per-feature NHSN variable
  menus (Timing / Details).

## Layout

```
clinical_feature_extractor/
  config.py            # paths, model ids, feature flags, retry/backoff, cache version
  vocabulary.py        # loads + indexes the canonical list (two-tier matcher)
  schema.py            # loads the NHSN hierarchical schema (per-feature variable menus)
  extractor.py         # FIVE stages (see below)
  pipeline.py          # xlsx path: read -> extract -> write back -> export csv
  tabular_pipeline.py  # format dispatch (.xlsx | .csv) + CSV adapter
  cli.py               # xlsx-only entrypoint (original)
  run.py               # format-agnostic entrypoint (.xlsx OR .csv)  <-- use this
  prompts/
    features_system.md      # Model 1 (features + evidence, no time)
    time.md                 # Model 2 (contextual time only)
    schema_timing.md        # Model 3, Timing group (NHSN variable fill-in)
    schema_details.md       # Model 3, Details group (NHSN variable fill-in)
    treatments.md           # Model 4 (treatments + response)
    past_medical_history.md # Model 5 (past medical history; independent stage)
  input/
    Automatic_extraction.xlsx   # place the input file here (xlsx or csv)
  output/              # extracted.csv, <input>.{dropped,absences,schema}.log, cache/
  tests/
```

## How it works

### Vocabulary matching (deterministic, no model)

The vocabulary (181 items: 93 positive + 88 negated `Evidence for absence of …`
forms) is loaded from the read-only dedup JSON. Matching is **two-tier and fully
deterministic** (no model, no fuzzy scoring):

- **Exact** case-insensitive match (highest confidence); then
- **Formatting-normalized** fallback that ignores cosmetic, document-to-document
  differences (whitespace, parentheses/brackets, edge punctuation, and hyphens —
  which are treated as optional) while preserving every *meaningful* token —
  numbers, thresholds, units, `%`, `°`, `<`/`>`, `/`. A normalized match is
  accepted only when **unambiguous** (maps to exactly one entry); ambiguous forms
  fall back to exact-only. Both tiers return the **vocabulary's spelling**, so
  output stays consistent with the list.

### Row detection (scalable, header-driven)

The header row is found by locating the `Clinical Text` cell in Column A
(wherever it sits); data rows are the contiguous non-empty Column A cells beneath
it. No row count is ever hardcoded; trailing empty formatted rows are ignored.
Behaves identically for `.xlsx` and `.csv`.

### Extraction — five stages

Stages 1–4 form one chained `extract()` call; stage 5 is independent (see below).
All stages use the same model (`qwen/qwen3.6-27b` by default, `temperature=0`,
`reasoning_effort="none"`, JSON mode). **Each stage is cached separately** under
`output/cache/`, so re-runs cost no tokens; `--no-cache` bypasses every stage.

1. **Model 1 (features)** — the canonical list + the clinical text → `{feature,
   evidence}` objects only, **no time**. A feature stated to be *absent* is
   returned in its negated `Evidence for absence of …` form, verbatim from the
   list. This raw output runs through the vocabulary post-check below.

2. **Feature post-check — toggleable, on by default** (`config.VALIDATE_FEATURES`).
   Model 1 always receives the vocabulary in its prompt, so it is guided toward the
   canonical names; this switch only controls after-the-fact enforcement:
   - **`True` (default)** → each `feature` is matched against the vocabulary; a
     **hit** is rewritten to the canonical spelling (other keys preserved), a
     **miss** is dropped **and logged** (row + feature) to a per-input
     `<input-stem>.dropped.log`.
   - **`False`** → *passthrough*: every returned feature is kept with the model's
     own spelling, nothing is dropped (favours recall at scale). Bare strings are
     still shaped into `{"feature": …}` objects so the later stages work. Both
     paths are always tested.

3. **Redundant-absence trim — toggleable, on by default**
   (`config.DEDUPE_ABSENCE_FEATURES`). Negated `Evidence for absence of …`
   features that repeat an **identical** evidence phrase are collapsed to the
   first occurrence; positive features are never touched. This curbs Model 1
   over-generating absences from one generic phrase. It removes only true repeats
   (a no-op otherwise), and each removed item is logged to a per-input
   `<input-stem>.absences.log`.

4. **Model 2 (time)** — the clinical text + the *already-validated* numbered
   features (with evidence, but **not** the feature list) → a **contextual** `time`
   per feature: its reference event **and** direction (before / after / during /
   since), not a bare duration (e.g. `"6.25 hours after completion of the
   apheresis platelet transfusion"`, not `"6.25 hours"`); `"uncertain"` when none
   is tied. Times are attached onto Model 1's features **programmatically by
   index** — Model 1's `feature`/`evidence` are authoritative; Model 2 can never
   rename, drop, or reorder a feature, so the feature count is preserved by
   construction.

5. **Model 3 (NHSN schema fill-in)** — for each feature, its NHSN variable menu is
   looked up from the hierarchical schema and filled in, run **separately per
   feature per group** (`Timing`, then `Details`). Each call sees the note, the one
   feature (name + evidence), and that group's menu (variable id, definition,
   response type, allowed values, and — for numeric variables — a unit menu); it
   returns only the variables the note supports, each with a `value` (and a `unit`
   for numeric variables, written as a bare JSON number).
   - **Positive features** get their full Timing **and** Details.
   - **Negated features** go through the **same full Timing call** (identical model
     behaviour — no new hallucination surface), after which only the first
     supported timing variable is kept; their Details variable is **stamped
     deterministically** with the feature's already-extracted `evidence` phrase
     (no LLM Details call).
   - The merge is **deterministic and schema-authoritative**: only variables in that
     feature's menu are kept, each in menu order; any `variable_id` the model
     returns that is **not** in the menu is **dropped** (never emitted) and logged
     to a per-input `<input-stem>.schema.log`; each kept variable's
     `notes`/`dependencies` are attached from the **schema**, never the model.
   - The result is nested on the feature as a `schema` object:
     `{"schema": {"timing": [...], "details": [...]}}`.

6. **Model 4 (treatments)** — the clinical text + **all** the positive features
   found above (name + evidence + time, so they are easy to locate in the note;
   negated features are not sent — you do not treat a symptom that isn't there).
   The nested `schema` object is stripped before this call, so Model 4 sees only
   name/evidence/time. It returns a raw list of `{treatment, response, timing,
   feature_list}` items (`feature_list: true` when the treatment addresses a
   feature-list sign/symptom, else `false`). This stage is deliberately
   **unvalidated**: the model's `treatments` list is stored exactly as returned (no
   vocabulary matching, no dropping, no reshaping).

7. **Model 5 (past medical history) — independent.** `extract_past_medical_history()`
   is a standalone, additive stage: it reads **only** the note, uses no vocabulary
   and no feature list, and neither feeds nor depends on stages 1–4 — so running it
   changes nothing about feature/time/schema/treatment extraction. It returns a
   raw, **unvalidated** list of `{condition, detail}` objects (stored exactly as
   returned, like the treatments stage).

### Output

One run produces everything, written back into the workbook and exported to CSV:

| Column | Header | Contents |
|---|---|---|
| A | `Clinical Text` | the input note (unchanged) |
| B | `Extracted Features` | `{"features": [...]}` — each feature with `evidence`, contextual `time`, and its nested NHSN `schema` object |
| C | `Treatments` | `{"treatments": [...]}` — Model 4's raw list (separate object) |
| D | `Past Medical History` | `{"past_medical_history": [...]}` — Model 5's raw list (separate object) |

Three per-input JSONL logs are always written next to the output CSV, each named
after the input stem: `.dropped.log` (off-list features), `.absences.log`
(redundant absences removed), and `.schema.log` (off-menu variable ids dropped).
A run summary also flags any **zero-feature rows** for review.

## Usage

Use `run` — it auto-detects `.xlsx`/`.xlsm` vs `.csv` by file extension:

```bash
# from the project root, with the venv active
python -m clinical_feature_extractor.run \
    --input input/Automatic_extraction.xlsx \
    --output output/extracted.csv

# a CSV input works the same way
python -m clinical_feature_extractor.run --input path/to/notes.csv

# cheap first real run: one row only (absolute worksheet row number)
python -m clinical_feature_extractor.run --single-row 2

# force fresh model calls (bypass every stage's cache)
python -m clinical_feature_extractor.run --no-cache
```

For a `.xlsx` input the file must contain a `Sheet1` sheet with a `Clinical Text`
header in column A; a `.csv` needs the same `Clinical Text` header row. Results
are written to columns B–D and exported to the `--output` CSV.

> The original `cli.py` entrypoint (`python -m clinical_feature_extractor.cli`)
> still works but is **xlsx-only**; `run` supersedes it for general use.

Requires `GROQ_API_KEY` in the project-root `.env`.

### Switching the model

`config.py` reads three optional env vars so a whole run can target a different
model without code changes (used by `model_evaluation_pipeline/`): `GROQ_MODEL`,
`GROQ_REASONING_EFFORT`, and `GROQ_MAX_COMPLETION_TOKENS`. Unset, all three keep
their qwen defaults and the pipeline behaves exactly as before.

## Tests (mocked Groq — no token spend)

```bash
pytest clinical_feature_extractor/tests -q
```
