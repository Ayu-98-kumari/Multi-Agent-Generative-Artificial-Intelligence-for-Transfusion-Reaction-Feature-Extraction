# mimic-pipeline

Code that turns MIMIC-IV discharge summaries into input for the main
transfusion-reaction pipeline. Imports nothing from `pdf_parser`,
`reaction_extractor`, or `clinical_feature_extractor`, and changes nothing in
them.

## The code/data boundary

| Directory | Contents | Git |
|---|---|---|
| `mimic-pipeline/` (here) | code only | **tracked** |
| `../kept-gitignored/mimic-iv-note/` | MIMIC data + everything derived from it | **gitignored** |

The PhysioNet DUA forbids redistributing MIMIC-IV, and anything derived from it
(the screened CSVs, the notebook with its saved cell outputs) is covered too.
So scripts live here and read/write `../kept-gitignored/mimic-iv-note/`. `.gitignore` also
refuses `*.csv`, `*.json`, and `*.ipynb` inside this directory, so a stray
output cannot become committable.

`MIMIC_Read.ipynb` stays in the data directory on purpose: its saved outputs
contain real discharge-note text.

## Setup

Uses the project `.venv` at the repo root -- the same interpreter as the rest
of the pipeline. pandas is installed there; `requirements.txt` records what this
package needs.

Point at data elsewhere with `$MIMIC_DATA_DIR` (defaults to `../kept-gitignored/mimic-iv-note`).

The funnel has three stages. Stage 1 (screen) and stage 3 (filter) are
deterministic and need no API key; stage 2 (triage) calls Groq.

## 1. extract_transfusion_summaries.py — screen

Screens every discharge summary for transfusion-reaction mentions using the
detection dictionary shared with `pmc-pipeline`: 12 full-phrase,
word-boundary-anchored regexes (`GENERAL_TR` plus 11 named reaction types), each
named reaction *also* matching its bare uppercase acronym (TRALI, TACO, DHTR, …).

```bash
# from the repo root
.venv/bin/python mimic-pipeline/extract_transfusion_summaries.py
```

Reads `discharge.csv.gz` in 5,000-row chunks (the 1.1 GB gzip is never
decompressed to disk and never fully in memory); writes matching notes to
`mimic_transfusion_cases.csv`, tagged with a `matched_reactions` column.

**Result: 1,210 matches out of 331,793 summaries.** Per-reaction: GENERAL_TR
836, TRALI 196, PTP 190, TACO 87, FNHTR 41, DHTR 20, AHTR 10, TTI 1, TAD 1.

Deterministic — same input, same 1,210 rows. Any drift is a bug.

The full-phrase patterns alone match 843 (byte-identical to the original
notebook); the acronyms lift that to 1,210. That is pure recall — the triager
rules out the noise (e.g. most of the 190 `PTP` hits are not post-transfusion
purpura). The point is to screen MIMIC and PMC with the *same* dictionary so the
two cohorts are comparable.

### Why the acronyms are case-sensitive and anchored

An earlier experiment OR'd bare abbreviations into the patterns *without* care
and matched them case-insensitively. With no `\b` anchors, `TTI` matches inside
ordinary words — "ge**tti**ng", "se**tti**ng", "pu**tti**ng" — producing a
2.5 GB file of noise. The current dictionary avoids that two ways: each acronym
is `\b`-anchored, and only the full phrase is case-folded (via an inline
`(?i:…)` group) while the acronym stays **case-sensitive**, so only a standalone
uppercase token matches. `GENERAL_TR`, `HTR` and `TAD` get no acronym at all —
"TR" is too generic and HTR/TAD collide with unrelated abbreviations. Keep both
the anchors and the case-sensitivity.

### Known caveat

A regex match is not a reaction. Notes saying "work-up for transfusion reaction
was not concerning" match perfectly and are negative cases. The 1,210 is a
recall-oriented candidate pool, not a positive cohort — that is what stage 2 is
for.

## 2. triage_transfusion_cases.py — triage (needs `GROQ_API_KEY`)

Reads each screened note in full (uncut, no matter how long) and asks the model
to grade it `genuine` / `ruled_out` / `mention_only`, with a one-sentence reason
and a verbatim supporting quote for every verdict.

```bash
.venv/bin/python mimic-pipeline/triage_transfusion_cases.py
```

Writes `mimic_transfusion_cases_triage_output.csv` (and a crash-safe `.jsonl`
audit log flushed per row). Determinism knobs: `openai/gpt-oss-120b`,
`temperature=0`, `reasoning_effort="medium"`, JSON mode. Each verdict is cached
by a hash of *note text + prompt + model*, so re-running is free and only newly
added or prompt-changed notes are billed.

**Result: 1,210 triaged → 424 genuine, 233 ruled_out, 553 mention_only, 0
errors.** (The 843 full-phrase notes yield the original 370 genuine; the acronym
expansion adds 54 more.)

## 3. filter_genuine.py — cohort

A pure join (no API). Keeps the rows triaged `genuine` and merges them back onto
the screened CSV, so each case carries all of its source columns plus the
verdict's `reason` and `supporting_quote`.

```bash
.venv/bin/python mimic-pipeline/filter_genuine.py
```

Writes `mimic_transfusion_genuine(424).csv`. A row-count guard aborts rather
than write a mismatched cohort if the join fans out. Re-run any time the triage
output changes.
