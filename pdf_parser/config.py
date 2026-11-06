"""Central configuration: constants, model id, token limits, thresholds, patterns.

Everything tunable lives here so the rest of the module reads like prose.
The ``.env`` at the *project root* (one level above this package) is loaded
once on import; only GROQ_API_KEY is required and only for the vision pass.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from dotenv import load_dotenv

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent
PROMPTS_DIR = PACKAGE_DIR / "prompts"

# Load the root .env (does nothing if absent; never overrides real env vars).
load_dotenv(PROJECT_ROOT / ".env")

# --------------------------------------------------------------------------- #
# Groq / vision model  (Pass 2)
# --------------------------------------------------------------------------- #
# Single source of truth for the model id. Do NOT add fallbacks.
VISION_MODEL = "qwen/qwen3.6-27b"

# HARD CEILING (discovered empirically): this model on the `on_demand` tier
# rejects any single request that *reserves* more output than its output-tokens-
# per-minute limit. Exceeding it yields an instant 429 "Request too large ...
# (OTPM): Limit 32000" before the model even runs. So max_completion_tokens must
# stay strictly below this. Raise only if your tier's OTPM is higher.
MODEL_OTPM_LIMIT = 32000

# Per-page output budget. A page of structured JSON is far smaller than this, so
# 8192 avoids truncation for real pages while leaving headroom under the OTPM
# ceiling for throughput. If a page still finishes for `length`, the truncation
# detector retries once at MAX_COMPLETION_TOKENS_RETRY. Both MUST stay below
# MODEL_OTPM_LIMIT.
MAX_COMPLETION_TOKENS = 8192
MAX_COMPLETION_TOKENS_RETRY = 16384

assert MAX_COMPLETION_TOKENS_RETRY < MODEL_OTPM_LIMIT, (
    "MAX_COMPLETION_TOKENS_RETRY must stay below the model OTPM limit or every "
    "request 429s as 'Request too large'."
)

TEMPERATURE = 0  # determinism; lowest the model supports

# qwen3 is a REASONING model. With JSON mode, its <think> prefix makes the full
# output invalid on complex pages (e.g. the dense incident-code page), which Groq
# rejects as 400 json_validate_failed — often with an EMPTY generation, so it is
# not a truncation you can detect from finish_reason. Disabling reasoning yields
# clean, fast, low-token JSON. Set to None to omit this parameter entirely (e.g.
# if you swap in a non-reasoning model that rejects it).
REASONING_EFFORT = "none"

# Page render resolution for the VLM. 200 DPI keeps small-font tables legible
# without ballooning the image payload. Tune if cells are hard to read.
RENDER_DPI = 200

GROQ_API_KEY_ENV = "GROQ_API_KEY"


def get_groq_api_key() -> str:
    """Return the Groq key or raise a clear error (only the vision pass needs it)."""
    key = os.environ.get(GROQ_API_KEY_ENV)
    if not key:
        raise RuntimeError(
            f"{GROQ_API_KEY_ENV} not found. Add it to {PROJECT_ROOT / '.env'}."
        )
    return key


# --------------------------------------------------------------------------- #
# Retry / backoff  (transient errors + 429s)
# --------------------------------------------------------------------------- #
MAX_RETRIES = 5            # attempts on transient/429 errors before giving up
BACKOFF_INITIAL = 1.0      # seconds
BACKOFF_FACTOR = 2.0       # exponential multiplier
BACKOFF_MAX = 60.0         # cap per-sleep
BACKOFF_JITTER = 0.25      # +/- fraction of jitter to avoid thundering herd

# --------------------------------------------------------------------------- #
# Validation thresholds
# --------------------------------------------------------------------------- #
# Coverage round-trip: fraction of Pass-1 page tokens that must reappear in the
# final JSON. Baseline is the extractor's text, not absolute ground truth.
COVERAGE_THRESHOLD = 0.95

# --------------------------------------------------------------------------- #
# Heading / inventory regexes  (Pass 1)
# --------------------------------------------------------------------------- #
SECTION_RE = re.compile(r"^Section\s+(\d+)\.\s*(.*\S)?\s*$")
TABLE_RE = re.compile(r"^Table\s+(\d+)\.\s*(.*\S)?\s*$")
FIGURE_RE = re.compile(r"^Figure\s+(\d+)\.\s*(.*\S)?\s*$")

# Archived reaction tables (Section 6) have NO "Table N." prefix — they are
# titled by reaction name. Every one shares this column-header triplet, which we
# use as the structural fingerprint to add them to the inventory checklist.
ARCHIVED_TABLE_TRIPLET = ("Case Definition", "Severity", "Imputability")

# Pull a trailing acronym like "(DHTR)" / "(TAGVHD)" from a reaction title.
ARCHIVED_ACRONYM_RE = re.compile(r"\(([A-Za-z][A-Za-z/&.\- ]*)\)\s*$")

# The Section-7 version-history table also has no "Table N." label; detect it by
# its column-header fingerprint so it is inventoried and validated like any table.
VERSION_HISTORY_HEADER = ("Version", "Release Date", "Summary of Revisions")

# --------------------------------------------------------------------------- #
# Header / footer stripping  (Pass 1)
# --------------------------------------------------------------------------- #
# Lines matching these patterns are running headers/footers, removed from the
# per-page text. Matching is by content (robust), not by y-position alone.
HEADER_FOOTER_PATTERNS = [
    re.compile(r"^NHSN Biovigilance Component$", re.I),
    # Two running-header variants appear in this doc: with and without
    # "Surveillance". Match both (and any future minor wording) defensively.
    re.compile(r"^Hemovigilance Module(?: Surveillance)? Protocol\s+v[\d.]+$", re.I),
    re.compile(r"^Page\s+\d+\s+of\s+\d+$", re.I),
    re.compile(
        r"^(January|February|March|April|May|June|July|August|"
        r"September|October|November|December)\s+\d{4}$",
        re.I,
    ),
]

# Vertical bands (as a fraction of page height) where header/footer lines live.
# Used only as a secondary guard so a pattern can't strip body text that happens
# to look like a header in the middle of the page.
HEADER_BAND_FRAC = 0.08   # top 8%
FOOTER_BAND_FRAC = 0.90   # bottom 10%

# --------------------------------------------------------------------------- #
# Multi-column reading order  (Pass 1)
# --------------------------------------------------------------------------- #
# A page is treated as two-column when both sides of the vertical midline carry
# enough lines and the right-side lines start clearly past the gutter.
TWO_COLUMN_MIN_LINES_PER_SIDE = 4
TWO_COLUMN_RIGHT_GUTTER_FRAC = 0.50  # right-column lines must start at >= this*width

# --------------------------------------------------------------------------- #
# Artifact filenames
# --------------------------------------------------------------------------- #
SKELETON_FILENAME = "skeleton.json"
