# Transfusion Reaction Detection

An end-to-end pipeline that reads free-text clinical notes and extracts
**structured transfusion-reaction features** aligned to the **CDC NHSN
Hemovigilance** protocol — each feature with its supporting evidence, contextual
timing, and the protocol's own Timing/Details variables, plus the treatments
given and the patient's past medical history.

The system is built as a chain of standalone modules. The first two turn the NHSN
protocol PDF into a machine-readable **canonical feature vocabulary**; the dataset
pipelines assemble cohorts of genuine transfusion-reaction notes; the core module
extracts structured features from those notes; and the evaluation pipeline scores
the extraction across several models.

## How the modules fit together

```
CDC NHSN Hemovigilance protocol (PDF)
   │   pdf_parser/            faithful VLM parse → structured JSON
   ▼
bv-hv-protocol-current.json
   │   reaction_extractor/    12 reaction case-definition tables →
   ▼                          signs/symptoms + lab markers → canonical vocabulary
canonical feature vocabulary ─────────────────────────┐
                                                       │
clinical datasets (MIMIC-IV, PMC-Patients)             │
   │   mimic-pipeline/ , pmc-pipeline/                 │
   │   screen → triage → filter → genuine cohort       │
   ▼                                                   ▼
genuine transfusion-reaction notes  ───►  clinical_feature_extractor/   (the core)
                                              5-stage extraction:
                                              features · timing · NHSN schema ·
                                              treatments · past medical history
                                                       │
                                                       ▼
                                          model_evaluation_pipeline/
                                          same notes, several models, side by side
```

## Modules

| Module | Role | Docs |
|---|---|---|
| **`pdf_parser/`** | Converts the NHSN Hemovigilance protocol PDF into a faithful structured JSON (deterministic skeleton → Groq VLM per page → stitch cross-page tables → validate). | [README](pdf_parser/README.md) |
| **`reaction_extractor/`** | Consumes that JSON (read-only) and derives the **canonical feature vocabulary** — the transfusion-reaction signs/symptoms and lab markers, extracted from the 12 reaction case-definition tables, then deduplicated. | [README](reaction_extractor/README.md) |
| **`clinical_feature_extractor/`** ★ | The core. A 5-stage extractor: (1) features + evidence against the vocabulary, (2) contextual timing, (3) NHSN Timing/Details variable fill-in, (4) treatments + response, (5) past medical history. Deterministic post-checks and per-stage caching throughout. | [README](clinical_feature_extractor/README.md) |
| **`mimic-pipeline/`** | Builds a genuine transfusion-reaction cohort from **MIMIC-IV** discharge summaries (screen → LLM triage → filter). | [README](mimic-pipeline/README.md) |
| **`pmc-pipeline/`** | The same funnel over **PMC-Patients** open-access case reports. | [README](pmc-pipeline/README.md) |
| **`model_evaluation_pipeline/`** | Runs the core extractor over one input with **three different models** and lays their outputs side by side for comparison. | [README](model_evaluation_pipeline/README.md) |

The two data pipelines and the shared screen→triage→filter method are documented
in [`DATASETS.md`](DATASETS.md).

## Getting started

### 1. Environment

Tested with **Python 3.13**. From the repository root:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

The root `requirements.txt` installs everything every module needs, pinned to the
exact versions the published results were produced with. (Each module also keeps
its own `requirements.txt` if you only want that one module.)

### 2. API keys

Every module that calls a model reads its keys from a **project-root `.env`**
(gitignored — never commit it). Copy the template and fill in your own keys:

```bash
cp .env.example .env
```

```
GROQ_API_KEY=your_groq_key_here              # qwen, gpt-oss-120b, every Groq-backed stage
OPENROUTER_API_KEY=your_openrouter_key_here  # only for the Kimi K3 run of the model comparison
```

Each user supplies their own keys; calls are billed to your own Groq / OpenRouter
accounts. The deterministic stages (PDF skeleton, vocabulary Columns 1–2, dataset
screening/filtering) run with no key.

### 3. Data

Clinical datasets are **not** included — MIMIC-IV is under a PhysioNet Data Use
Agreement and PMC-Patients is too large for GitHub. Both live under a single
gitignored umbrella, `kept-gitignored/`, with each dataset in its own subfolder.
See [`DATASETS.md`](DATASETS.md) for exactly where to place each download and how
to build a cohort.

**NHSN protocol PDF.** The canonical feature vocabulary was built from the CDC
*NHSN Biovigilance Component Hemovigilance Module Protocol*, **v3.0, January 2026**
(34 pages). Neither the PDF nor the JSON parsed from it is included in this
repository. The vocabulary derived from it
(`reaction_extractor/output/deduplicated_symptoms_labmarkers.json`) **is** included,
so you only need the PDF to re-run `pdf_parser/` and `reaction_extractor/` yourself.
To do so, download it from CDC —
<https://www.cdc.gov/nhsn/PDFs/Biovigilance/BV-HV-protocol-current.pdf> — and save
it at the repository root as `bv-hv-protocol-current.pdf`. That link always serves
the *current* version of the protocol; if it no longer reads v3.0 (January 2026),
the parse and the vocabulary may differ from ours. The exact file used had SHA-256
`38c5cd09c66df6022a8b92526f362562c618732db5fe8844a0afff3c7c0c416c`.

### 4. Run the core extractor

```bash
# from the project root, with the venv active
python -m clinical_feature_extractor.run --input path/to/notes.csv --output output/extracted.csv
```

The input is a spreadsheet (`.xlsx` or `.csv`) with a `Clinical Text` header in
column A; results are written to columns B–D and exported to the output CSV. See
the [module README](clinical_feature_extractor/README.md) for the full stage-by-stage
description, feature flags, and caching.

### 5. Run the three-model comparison

1. Put **one** CSV of clinical notes in
   [`model_evaluation_pipeline/input_output/`](model_evaluation_pipeline/input_output/):
   header `Clinical Text` in column A, one note per row, no blank rows. The
   folder's [README](model_evaluation_pipeline/input_output/README.md) has the
   exact format and an example.
2. From the repository root, run:

   ```bash
   bash model_evaluation_pipeline/run_models.sh
   ```

3. The three per-model outputs and `<your-file>_model_comparison.csv` appear in
   that same folder. If a run stops part-way, run the same command again — it
   picks up where it stopped instead of starting over.

See the [model_evaluation_pipeline README](model_evaluation_pipeline/README.md)
for the models, their settings, and the options.

## Tests

Every module ships mocked-model tests (**no real API calls, no token spend**):

```bash
python -m pytest clinical_feature_extractor/tests -q
python -m pytest reaction_extractor/tests -q
python -m pytest pdf_parser/tests -q
```

## Data & licensing note

This repository contains **code and the derived NHSN feature schema/vocabulary
only**. It does not redistribute MIMIC-IV, PMC-Patients, or any patient data —
those stay local under `kept-gitignored/`. Obtain each dataset yourself under its
own license/agreement.
