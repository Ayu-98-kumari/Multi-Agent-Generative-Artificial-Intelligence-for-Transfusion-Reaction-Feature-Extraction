# pdf_parser

Converts well-structured clinical surveillance PDFs (e.g. CDC NHSN Hemovigilance
protocols) into a faithful, structured JSON representation for a downstream RAG
pipeline. The design goal is **faithfulness over cleverness**: the two failures
it actively guards against are **silent omission** and **table fragmentation**.

## Pipeline

```
skeleton (Pass 1, deterministic)  ->  vision (Pass 2, Groq VLM)  ->  stitch  ->  validate  ->  JSON
```

1. **Skeleton (`skeleton.py`)** — no LLM. Uses **PyMuPDF (`fitz`)** for the TOC,
   positioned per-page text, a Table/Figure/Section inventory (regex + the
   archived-table fingerprint), a page→section map, and page-image rendering for
   Pass 2 — all in one dependency. The skeleton is the reproducible backbone and
   the validation checklist; it is written to `skeleton.json`.
2. **Vision (`vision_parser.py`)** — renders each page and sends it to the Groq
   VLM with that page's skeleton context, asking it to fill a **fixed** schema.
   `temperature=0`, JSON mode, exponential-backoff retries, and a **truncation
   detector** (a response that finishes for `length` is retried once at a higher
   token budget, then surfaced as a hard `TruncationError` — never accepted).
3. **Stitch (`stitching.py`)** — merges cross-page table fragments (the
   incident-code Table 8 runs pages 19–22) into one table with a correct
   `page_span`. Merges **only** on explicit continuation signals.
4. **Validate (`validation.py`)** — deterministic acceptance gate (below).
5. **Assemble + write (`pipeline.py`)** — groups blocks into sections, assigns
   `block_id`s, writes the final `<stem>.json`.

## Install

```bash
pip install -r pdf_parser/requirements.txt
```

`GROQ_API_KEY` is read from the **project-root `.env`** (only the vision pass
needs it). The skeleton, stitch, and validation stages run without a key.

## Run

```bash
# Pass 1 only — inspect the skeleton before spending any VLM calls
python -m pdf_parser.cli --input bv-hv-protocol-current.pdf --skeleton-only

# Cheap, isolated first real VLM call — parse a single page
python -m pdf_parser.cli --input bv-hv-protocol-current.pdf --single-page 19

# Full run -> writes <stem>.json and prints the validation report
python -m pdf_parser.cli --input bv-hv-protocol-current.pdf --output out.json
```

Flags: `--skeleton-only`, `--single-page N`, `--skip-validation`,
`--force-skeleton`, `--no-cache`, `--coverage-threshold F`, `-v/-vv`.
Exit code is non-zero when validation reports hard failures.

Artifacts default to `<stem>_pdfparser/` next to the input (`skeleton.json` and a
per-page VLM cache under `pages/`), so re-runs are cheap and the structure pass
never re-extracts.

## Configuration (`config.py`)

| Constant | Default | Notes |
|----------|---------|-------|
| `VISION_MODEL` | `qwen/qwen3.6-27b` | single source of truth; **no fallbacks** |
| `MAX_COMPLETION_TOKENS` | `32768` | raise if pages are large; medical content must never truncate |
| `MAX_COMPLETION_TOKENS_RETRY` | `65536` | budget for the one truncation retry |
| `TEMPERATURE` | `0` | determinism |
| `REASONING_EFFORT` | `"none"` | qwen3 is a reasoning model; its `<think>` prefix breaks strict JSON mode on complex pages (→ 400 `json_validate_failed`). Disabling it yields clean, fast JSON. Set `None` to omit. |
| `MODEL_OTPM_LIMIT` | `32000` | tier's output-tokens-per-minute cap; the token budgets must stay below it or every request 429s |
| `RENDER_DPI` | `200` | raise if small-font tables are hard to read |
| `COVERAGE_THRESHOLD` | `0.95` | coverage round-trip recall |
| `MAX_RETRIES` / `BACKOFF_*` | — | transient-error / 429 backoff |

Prompts are editable files in `prompts/` (`system.md`, `page_user.md`).

## JSON schema (`schema.py`, Pydantic v2)

A shallow outline of the document. **Page is metadata (a field/span), never a
structural level.**

```
Document(title, version, source_filename, total_pages, parsed_on, sections=[
  Section(section_number, title, page_span=[start,end], content=[
    ProseBlock {type:"prose", subheading?, text, page, block_id}
    TableBlock {type:"table", label, caption, page_span, columns, rows[{cells}],
                notes, footnotes, continued_from, continued, block_id}
    FigureBlock{type:"figure", label, caption, description, page, abbreviations?, block_id}
  ])
])
```

Content blocks stay **flat** under each section (one level), so the shape holds
across the whole document and across page-spanning tables. `subheading` is a
label on a prose block, not a nested container.

Added attributes (justified): `block_id` (stable `s<sec>-b<idx>` id for
validation traceability and RAG citation) and, on skeleton inventory items,
`archived` (Section-6 reaction tables have no `Table N.` prefix — see below).

### Archived Section-6 reaction tables

Section 6 contains eight reaction tables (Allergic, DHTR, DSTR, FNHTR, HTR, PTP,
TAD, TAGVHD on pages 24–31) titled by **reaction name with no `Table N.`
prefix**. They are detected structurally by the shared
`Case Definition / Severity / Imputability` column triplet, added to the
inventory as `Archived: <acronym>`, and so are both expected by the VLM and
counted by validation — exactly like numbered tables.

## Validation (`validation.py`)

Ordered checks. **Hard failures (1–4) block** acceptance; **golden facts (5)
flag** for review. The report names the failing check and section/page.

1. **Schema conformance** — output validates against the Pydantic models.
2. **Inventory completeness** — every skeleton table (17: 8 numbered + 8
   archived Section-6 reaction tables + 1 Section-7 version-history table) and
   figure (3) appears **exactly once** after stitching, matched **by page** (so
   unlabeled detected tables count too). An extra table on an unexpected page is
   flagged as a likely unstitched fragment.
3. **Coverage round-trip** — **order-independent token recall** of the Pass-1
   baseline ≥ `COVERAGE_THRESHOLD`. Order-independent on purpose: dense table
   pages interleave cells across columns, so an order-sensitive check would raise
   false alarms. *Caveat: the baseline is the extractor's text, not absolute
   ground truth.*
4. **Structural invariants** — each table ≥1 row, no stray cell columns; every
   block's page(s) within its section span and under the correct parent section;
   non-decreasing reading order within a section; non-empty figure descriptions.
5. **Golden facts** — see below.

### Golden facts (`tests/golden_facts.py`)

Hand-written assertions that specific clinical values survived parsing. Every
fact is **scoped to a specific table** (located by a page the table covers,
including the stitched Table 8 span) and checked against that table's text only —
**not** the whole document. Scoping is essential because values recur across
reactions (DHTR vs DSTR both cite "24 hours and 28 days"; HTR adult vs pediatric
BP thresholds; "within 24 hours" in both AHTR and TAD).

Seeded set includes TACO=12h, TRALI=6h, AHTR=24h, incident codes UT 06 / PC 03,
the Babesia 14-states footnote, the TRALI PaO2/FiO2 ≤300 threshold, the DHTR
24h–28d antibody window, the HTR adult systolic BP thresholds, PTP timing, TAD
onset, and a **stitch-integrity** fact asserting Table 8 still contains a
last-fragment code (MS 99 / UT 26, page 22).

**To extend:** add a `GoldenFact(name, description, page, check)` where `page` is
any page the target table covers and `check` runs over the normalized
(lowercased, whitespace-collapsed) table text. Use `_all_present(...)` /
`_any_present(...)` helpers.

## Tests

```bash
python -m pytest pdf_parser/tests/ -q
```

`test_skeleton.py` (Pass 1 vs the real PDF), `test_vision_parser.py` (fake Groq
client: success / truncation / retry), `test_stitching.py` (Table 8 merges; the
eight archived tables stay distinct), `test_validation.py` (each gate + golden
scoping), `test_pipeline.py` (assemble/stitch wiring + CLI). Tests use mocked VLM
responses only — **no real Groq calls, no spend**. Tests that need the sample PDF
skip cleanly if it is absent.
