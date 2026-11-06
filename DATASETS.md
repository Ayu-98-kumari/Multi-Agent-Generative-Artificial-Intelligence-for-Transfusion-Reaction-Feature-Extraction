# Datasets

How clinical datasets are organized in this project, and how to set one up on a
fresh clone.

## Layout convention

Two parallel locations, one per dataset:

| What | Where | Git |
|---|---|---|
| **Data** — everything downloaded or derived | `kept-gitignored/<dataset>/` | **gitignored** (never committed) |
| **Code** — the scripts that process it | `<dataset>-pipeline/` (repo root) | **tracked** |

`kept-gitignored/` is a single umbrella that the whole repo ignores with one
rule (`kept-gitignored/` in `.gitignore`). Nothing inside it can ever be
committed, so licensed or sensitive data (and anything derived from it) stays
local by construction. Each dataset gets its own subfolder underneath.

A fresh clone will **not** contain `kept-gitignored/` (it holds only ignored
files, and empty folders aren't tracked). You create it yourself:

```bash
mkdir -p kept-gitignored
```

Then place each dataset in its own subfolder, as described below.

## One-time environment setup

```bash
# from the repo root
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

The LLM triage stage calls Groq, so it needs **your own** API key. Create a
`.env` at the repo root from the template (it is gitignored — never commit it)
and fill in `GROQ_API_KEY`:

```bash
cp .env.example .env
```

Without a key the deterministic screening stage still runs; only the triage
stage requires it (and it spends your own tokens against your own Groq plan).

---

## MIMIC-IV discharge summaries

**Code:** `mimic-pipeline/`  **Data:** `kept-gitignored/mimic-iv-note/`

MIMIC-IV is under a PhysioNet Data Use Agreement — it cannot be redistributed,
so it is not included here. Download it yourself with your own PhysioNet
credentials.

1. Create the data folder and place the note file there:

   ```
   kept-gitignored/mimic-iv-note/discharge.csv.gz
   ```

   (If your data lives elsewhere, point at it with `MIMIC_DATA_DIR` instead of
   moving it — every script honors that environment variable and falls back to
   `kept-gitignored/mimic-iv-note/`.)

2. **Screen** for transfusion-reaction mentions (deterministic, no API key):

   ```bash
   .venv/bin/python mimic-pipeline/extract_transfusion_summaries.py
   ```

   Writes `mimic_transfusion_cases.csv` into the data folder.

3. **Triage** the screened notes into `genuine` / `ruled_out` / `mention_only`
   (LLM; needs `GROQ_API_KEY`):

   ```bash
   .venv/bin/python mimic-pipeline/triage_transfusion_cases.py
   ```

4. **Filter** to the genuine cohort (deterministic join, no API key):

   ```bash
   .venv/bin/python mimic-pipeline/filter_genuine.py
   ```

   Writes `mimic_transfusion_genuine(424).csv` — the genuine cases with their
   full text, all columns, and each verdict's reason + supporting quote.

Counts: 1,210 screened → 424 genuine / 233 ruled_out / 553 mention_only. See
`mimic-pipeline/README.md` for the full description of each stage.

---

## PMC-Patients case reports

**Code:** `pmc-pipeline/`  **Data:** `kept-gitignored/pmc-patients/`

PMC-Patients is open-access, but the raw file is 837 MB (and the derived CSV
731 MB), well past GitHub's limit, so it lives under the same gitignored
umbrella as MIMIC.

1. Place the download in the data folder:

   ```
   kept-gitignored/pmc-patients/PMC-Patients-V2.json
   ```

   (Point elsewhere with `PMC_DATA_DIR` if it lives outside the repo.)

2. **Convert** the streamed JSON array to CSV (no API key):

   ```bash
   .venv/bin/python pmc-pipeline/convert_pmc_json_to_csv.py
   ```

3. **Screen** → **triage** → **filter**, exactly as for MIMIC:

   ```bash
   .venv/bin/python pmc-pipeline/extract_transfusion_summaries.py
   .venv/bin/python pmc-pipeline/triage_transfusion_cases.py       # needs GROQ_API_KEY
   .venv/bin/python pmc-pipeline/filter_genuine.py
   ```

Counts: 250,294 reports → 174 screened → 97 genuine. See
`pmc-pipeline/README.md` for details.

---

## The notes used in the study

**PMC-Patients.** [`study_note_ids.csv`](study_note_ids.csv) lists every
PMC-Patients note the published results are based on — identifiers only, no note
text:

| column | meaning |
|---|---|
| `set` | `evaluation` — the 20 PMC-Patients test notes scored by the physicians; `development` — the 3 notes used to refine the prompts. The two sets do not overlap. |
| `dataset` | `PMC-Patients` |
| `id_type` | `patient_uid` (PMC-Patients V2) |
| `id` | the note's `patient_uid` |
| `text_sha256_16` | first 16 hex characters of the SHA-256 of the note text (UTF-8, leading/trailing whitespace removed) |

Each note was passed to the pipeline exactly as it appears in the dataset, so the
checksum computed from your own download should match. It is also the `id` column
of the `*_model_comparison.csv` output, linking every result back to its note.

To reproduce the PMC-Patients evaluation as closely as possible: look up the
`evaluation` rows in your own download, put their text in a CSV under a
`Clinical Text` header (one note per row), confirm the checksums, and run the
three-model comparison on it (see the main [README](README.md)).

**MIMIC-IV.** The study also evaluated 20 MIMIC-IV-Note discharge summaries (and
used 5 more for prompt development). Because MIMIC-IV is distributed under a
PhysioNet credentialed-access agreement, their note IDs are not published here;
they are available from the authors upon request.

The study's other 10 evaluation cases (the golden set) are not included in this
file.

---

## The shared method

Both datasets run the same funnel, so the two cohorts are directly comparable:

1. **Screen** — one deterministic regex dictionary (12 full phrases + their
   case-sensitive acronyms) narrows the corpus to a recall-oriented candidate
   pool. Identical across datasets; only the text column differs.
2. **Triage** — an LLM reads each candidate *in full* and grades it `genuine` /
   `ruled_out` / `mention_only` with a reason and a verbatim quote. Same model
   and hyperparameters for both datasets. Verdicts are cached by a hash of note
   text + prompt + model, so re-runs are free and only new or prompt-changed
   notes are billed.
3. **Filter** — a pure join keeps the `genuine` rows with their full text and
   the triage rationale attached.

Operating notes: the screen and filter stages are deterministic (same input →
same output; any drift is a bug). A regex match is not a reaction — a note can
mention a reaction that was ruled out, which is why triage exists. If triage
records any `error` rows, just re-run it — cached verdicts are reused and only
the failed notes are retried. Never put an API key in these docs or any tracked
file; each user supplies their own in the gitignored `.env`.

---

## Adding a new dataset

Follow the same convention:

1. Put the data under `kept-gitignored/<dataset>/` (it's already gitignored).
2. Put the processing code in `<dataset>-pipeline/` at the repo root.
3. Add a section here documenting where to get the data, where it goes, and how
   to run it.
