# reaction_extractor

A downstream module that consumes the **read-only** output of `pdf_parser`
(a parsed-document JSON) and produces a new JSON: a single unified 3-column
table over the **12 transfusion-reaction case-definition tables**.

It never modifies `pdf_parser/` or the source JSON — they are inputs.

## The 3 columns

| Column | Source | AI? |
|--------|--------|-----|
| **1 · Reaction** | the table's `caption` (else its `label`, minus any `Archived: ` prefix) | no |
| **2 · Definition** | only the cells under the source `"Case Definition"` column, concatenated in row order (verbatim, incl. AND/OR/EITHER logic). Severity & Imputability columns are never read. | no |
| **3 · Features** | the Groq model extracts a structured object from Column 2 — `time_window` (string\|null), `signs_symptoms` (list\|null), `lab_markers` (list\|null) — from the definition text only, thresholds verbatim, `null` for anything not stated | **yes** |

## The 12 tables (excludes Table 7, p16)

Selected strictly by start page — `config.TARGET_PAGES`:

```
11 TTI    13 AHTR   14 TACO   15 TRALI
24 Allergic  25 DHTR   26 DSTR   27 FNHTR
28 HTR    29 PTP    30 TAD    31 TAGVHD
```

Table 7 "Other transfusion reaction" (p16) is omitted from the list, so its
"Not Applicable" case definition is never included.

## Install

```bash
pip install -r reaction_extractor/requirements.txt
```

`GROQ_API_KEY` is read from the **project-root `.env`** — only the features step
needs it. Extraction of Columns 1 & 2 runs with no key and no spend.

## Run

```bash
# Cheap, isolated first real Groq run — one reaction (by page or name)
python -m reaction_extractor.cli --single 13
python -m reaction_extractor.cli --single AHTR

# Full run -> writes output/reaction_features.json + a validation report
python -m reaction_extractor.cli --input bv-hv-protocol-current.json \
    --output reaction_features.json

# Ignore the per-reaction cache and re-call the model
python -m reaction_extractor.cli --no-cache
```

Flags: `--input/-i`, `--output/-o` (bare name -> `output/`), `--no-cache`,
`--single <page-or-name>`, `-v`. Each reaction's features are cached under
`output/cache/<slug>.json`, so re-runs cost no tokens. Cache files are stamped
with `config.FEATURES_SCHEMA_VERSION`; when the prompt or output shape changes
the version is bumped and stale entries are ignored automatically (recomputed on
the next run) — no manual clearing or `--no-cache` needed.

## Building the 181-feature vocabulary

The extraction pipeline (`clinical_feature_extractor/`) reads one file from this
module: `output/deduplicated_symptoms_labmarkers.json`, the canonical list of 181
features. It is built in four steps, run in this order from the repository root;
each step reads the previous step's output from `output/`:

```bash
# 1. Extract each reaction's features from the parsed protocol (Groq)
python -m reaction_extractor.cli --input bv-hv-protocol-current.json \
    --output reaction_features.json                              # -> output/reaction_features.json (12 reactions)

# 2. Flatten into one list of signs/symptoms and lab markers (no Groq)
python -m reaction_extractor.cli --symptoms-labmarkers           # -> output/symptoms_labmarkers.json (93)

# 3. Add an "Evidence for absence of ..." form after each item (no Groq)
python -m reaction_extractor.cli --extended-symptoms-labmarkers  # -> output/extended_symptoms_labmarkers.json (186)

# 4. Remove awkward double negatives (Groq)
python -m reaction_extractor.cli --dedup-features                # -> output/deduplicated_symptoms_labmarkers.json (181)
```

Step 1 needs `bv-hv-protocol-current.json`, the output of `pdf_parser/` run on the
NHSN protocol PDF (see the main [README](../README.md) for the exact version).
Steps 1 and 4 call Groq and need `GROQ_API_KEY`; both cache their model calls
under `output/cache/`, so re-runs cost no tokens.

Step 4 also writes `output/dedup_removed_items.json`, listing every feature it
removed, each with the reason `mechanically-generated double negative` (the only
kind of duplicate it removes). Check this file to see exactly what the dedup
dropped. For the published vocabulary it removed 5 of the 186 items:

- Evidence for absence of no evidence of left atrial hypertension
- Evidence for absence of asymptomatic
- Evidence for absence of absence of clinical signs of hemolysis
- Evidence for absence of no pathogen is detected in the recipient
- Evidence for absence of serologic testing is negative

**Only the final 181-feature list is included in this repository.** The
intermediate files and `dedup_removed_items.json` are recreated whenever you
rebuild the vocabulary. (`--unique-features` and `--extended-features` build an
alternative feature list that the pipeline does not use.)

## Output shape

```json
{
  "metadata": {"source_filename": "...", "generated_on": "...", "count": 12, "model": "qwen/qwen3.6-27b"},
  "reactions": [
    {"reaction": "Acute hemolytic transfusion reaction (AHTR)",
     "definition": "<case-definition text only>",
     "features": {
       "time_window": "within 24 hours of cessation of transfusion",
       "signs_symptoms": ["fever", "hypotension", "back pain"],
       "lab_markers": ["positive DAT for anti-IgG or anti-C3", "elevated LDH"]
     },
     "source_page": 13}
  ]
}
```

Each `features` field is extracted from the Column-2 text only (never general
medical knowledge), with thresholds and time windows preserved verbatim. A field
the definition does not state is `null` — e.g. DSTR (p26, serologic only) has
`signs_symptoms: null`, and TAD (p30) has `lab_markers: null`.

In `signs_symptoms` and `lab_markers`, a "/" that joins two independent clinical
terms is split into separate items (`"Chills/rigors"` → `"Chills"`, `"Rigors"`;
`"Back/flank pain"` → `"Back pain"`, `"Flank pain"`). The prompt is the primary
fix; a post-processing safety net in `features.py` re-applies it. A "/" that is a
ratio, unit equivalence, abbreviation, or logical connector stays intact
(`"PaO2/FiO2 ≤ 300 mm Hg"`, `"38°C/100.4°F"`, `"and/or"`, `"c/w"`). The
do-not-split conditions are named constants in `config.py`
(`SPLIT_PROTECTED_SUBSTRINGS`, `SPLIT_MIN_TERM_LEN`).

`signs_symptoms` also gets a generalizable compound-term split (`features.py`,
`split_signs_symptoms`): Pattern 1 turns `"<prefix> a, b and c"` into one item
per term (`"Edema of lips, tongue and uvula"` → three `"Edema of …"` items), and
Pattern 2 turns `"a and b <prep-phrase>"` into two suffix-sharing items
(`"Erythema and edema of the periorbital area"` → `"Erythema of…"`, `"Edema
of…"`). A shared guard protects thresholds (digits, `°`, `%`), `and/or`, and long
descriptive phrases; the word-length ceilings are named constants in `config.py`
(`SPLIT_PATTERN1_MAX_TERM_WORDS`, `SPLIT_PATTERN2_MAX_LEADING_WORDS`,
`SPLIT_PATTERN2_PREPOSITIONS`). Only `signs_symptoms` is touched.

### Faithfulness

`prompts/features.md` carries a **Faithfulness Rules** section that forbids the
model from over-simplifying: keep quantified thresholds in full (Rule 1), never
split a term from its parenthetical/semicolon clarification (Rule 2), give each
age-group variant its own item with its values (Rule 3), extract time windows
even mid-sentence (Rule 4), and preserve verbatim when in doubt (Rule 5). Two
lightweight safety nets back this up: `normalize_feature_list` rejoins an
orphaned `(…)`/`;…` clarification into the item it clarifies, and validation
flags (advisory) a `time_window: null` when the definition text still matches a
`within…`/`between…and…` time-window pattern (`config.TIME_WINDOW_HINT`).

## Groq settings (mirrors `pdf_parser`)

`VISION_MODEL="qwen/qwen3.6-27b"`, `temperature=0`, `reasoning_effort="none"`,
JSON mode, client `max_retries=0` with our own exponential-backoff + jitter on
429/transient errors, and a truncation detector (`finish_reason=="length"` →
one retry at a higher budget, then a hard `TruncationError` — never silently
accepted). Token budgets stay below `MODEL_OTPM_LIMIT` (32000). All in
`config.py`; the prompt is the editable `prompts/features.md`.

## Validation (lightweight, deterministic)

Full run checks: exactly 12 entries; page 16 absent; all 12 target pages present
exactly once; non-empty reaction & definition. For each entry's features: the
object must hold exactly the three keys, at least one non-null (all-null is
**flagged for review, not hard-failed**), and a non-null list field must be
non-empty. The report prints after each run; flags are advisory and do not
change the exit code.

## Tests

```bash
python -m pytest reaction_extractor/tests/ -q
```

`test_extractor.py` (deterministic selection + Column 1/2 rules, and the real
JSON when present) and `test_features.py` (fake Groq client: success, cache,
transient/truncation/json_validate_failed retry, pipeline assembly, `--single`,
validation flags). **No real Groq calls, no spend.**
