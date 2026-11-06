# pmc-pipeline

Finds genuine transfusion-reaction cases in the **PMC-Patients** collection of
published patient case reports, using the same two-stage funnel and the same
detection dictionary as `mimic-pipeline`, so the two cohorts are comparable.
Imports nothing from `pdf_parser`, `reaction_extractor`, or
`clinical_feature_extractor`, and changes nothing in them.

## The code/data boundary

| Directory | Contents | Git |
|---|---|---|
| `pmc-pipeline/` (here) | code only | **tracked** |
| `../kept-gitignored/pmc-patients/` | PMC data + everything derived from it | **gitignored** |

PMC-Patients is open-access (no DUA), but the raw JSON is 837 MB and the derived
CSV is 731 MB — far past GitHub's limit — so the data stays under
`kept-gitignored/` for the same reason MIMIC does. `.gitignore` also refuses
`*.csv`, `*.json`, and `*.jsonl` inside this directory as defense-in-depth.

Point at data elsewhere with `$PMC_DATA_DIR` (defaults to
`../kept-gitignored/pmc-patients`).

## Setup

Uses the project `.venv` at the repo root. `requirements.txt` records what this
package needs: `ijson` (for streaming the giant JSON) and `pandas`. The triage
stage additionally needs `groq` + `python-dotenv` (already installed for the
main pipeline) and your own `GROQ_API_KEY` in the repo-root `.env`.

Get the source file `PMC-Patients-V2.json` from the PMC-Patients project and
place it at `../kept-gitignored/pmc-patients/PMC-Patients-V2.json`.

## 0. convert_pmc_json_to_csv.py — JSON → CSV

The download is a single 837 MB JSON array, not line-delimited, so pandas can't
chunk it. This streams it with `ijson` (low memory) in two passes — collect the
union of all keys, then write rows — and saves the CSV beside the JSON.

```bash
# from the repo root
.venv/bin/python pmc-pipeline/convert_pmc_json_to_csv.py
```

**Result: 250,294 rows** written to `PMC-Patients-V2.csv`. Nested values become
JSON strings; note text is never truncated (verified byte-identical JSON↔CSV).

## 1. extract_transfusion_summaries.py — screen

Same detection dictionary as `mimic-pipeline` (12 full-phrase anchored regexes,
each named reaction also matching its case-sensitive bare acronym). The only
differences from MIMIC are the input column (`patient` here vs `text`) and no
gzip.

```bash
.venv/bin/python pmc-pipeline/extract_transfusion_summaries.py
```

Reads the CSV in 5,000-row chunks; writes matching reports to
`pmc_transfusion_cases.csv`, tagged with `matched_reactions`.

**Result: 174 matches out of 250,294 reports** (full phrases alone find 114; the
acronyms add 60). Per-reaction: GENERAL_TR 89, TRALI 47, TACO 21, DHTR 20,
PTP 18, TTI 11, AHTR 8, TAGVHD 1, FNHTR 1, DSTR 1. A match is not a reaction —
that is what stage 2 decides.

## 2. triage_transfusion_cases.py — triage (needs `GROQ_API_KEY`)

Identical model setup and hyperparameters to the MIMIC triager
(`openai/gpt-oss-120b`, `temperature=0`, `reasoning_effort="medium"`, JSON mode,
full report sent uncut, per-note cache). Only the prompt framing ("one published
patient case report") and the id columns differ.

```bash
.venv/bin/python pmc-pipeline/triage_transfusion_cases.py
```

Writes `pmc_transfusion_cases_triage_output.csv` (+ a crash-safe `.jsonl`).

**Result: 174 triaged → 97 genuine, 37 ruled_out, 40 mention_only, 0 errors.**

## 3. filter_genuine.py — cohort

A pure join (no API): keeps the `genuine` rows and merges them back onto the
screened CSV with the verdict's `reason` and `supporting_quote` attached. A
row-count guard aborts on any fan-out.

```bash
.venv/bin/python pmc-pipeline/filter_genuine.py
```

Writes `pmc_transfusion_genuine(97).csv` — the audit-ready cohort (full report
text + all columns + why each was kept).
