"""Central configuration for the reaction feature extractor.

This module consumes the *output* of the (separate, read-only) ``pdf_parser``
package: a parsed-document JSON. It never imports from or modifies ``pdf_parser``.

The Groq settings deliberately MIRROR the parser's ``config.py`` so the two
modules behave identically against the same tier/model. Only Column 3 (Features)
uses the model; it is a text-only call.

The ``.env`` at the *project root* (one level above this package) is loaded once
on import; only ``GROQ_API_KEY`` is required, and only for the features step.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent
PROMPTS_DIR = PACKAGE_DIR / "prompts"
OUTPUT_DIR = PACKAGE_DIR / "output"
CACHE_DIR = OUTPUT_DIR / "cache"

# Load the root .env (does nothing if absent; never overrides real env vars).
load_dotenv(PROJECT_ROOT / ".env")

# Default source: the parser's output JSON at the project root. Override with
# --input on the CLI.
INPUT_JSON_DEFAULT = PROJECT_ROOT / "bv-hv-protocol-current.json"

# Default output filename (written under OUTPUT_DIR).
OUTPUT_FILENAME_DEFAULT = "reaction_features.json"

# Bumped whenever the features prompt or the Column-3 output shape changes.
# Cache files are stamped with this; a hit is honored only when it matches, so a
# stale cache from an older shape can never silently produce old-format output.
#   1 = flat list[str]         (retired)
#   2 = {time_window, signs_symptoms, lab_markers} structured object
#   3 = adds "/" term-splitting normalization on the two list fields
#   4 = faithfulness-tightened prompt + parenthetical-fragment rejoin
FEATURES_SCHEMA_VERSION = 4

# --------------------------------------------------------------------------- #
# "/" term-splitting (signs_symptoms / lab_markers normalization)
# --------------------------------------------------------------------------- #
# A "/" that joins two independent clinical terms (e.g. "Chills/rigors") should
# become two list items; a "/" that is a ratio, unit-equivalence, abbreviation,
# or logical connector must stay intact. These are the DO-NOT-SPLIT conditions,
# named here so they are easy to extend for future PDFs.
#
# Substrings whose presence (case-insensitive) marks the "/" as a connector or
# abbreviation, never a term separator:
SPLIT_PROTECTED_SUBSTRINGS = ("and/or", "c/w")
# Neither side of a splittable "/" may be shorter than this (guards "c/w" etc.):
SPLIT_MIN_TERM_LEN = 3

# Compound-term splitting for signs_symptoms (comma/and lists). Word-length
# ceilings distinguish a short clinical noun/phrase from a long descriptive
# phrase — tune these for future PDFs.
#   Pattern 1: "<prefix> a, b and c" — each term may be up to this many words.
SPLIT_PATTERN1_MAX_TERM_WORDS = 3
#   Pattern 2: "a and b <prep-phrase>" — each leading noun up to this many words.
SPLIT_PATTERN2_MAX_LEADING_WORDS = 2
#   Pattern 2: the shared suffix must begin with one of these prepositions.
SPLIT_PATTERN2_PREPOSITIONS = ("of", "in", "at", "on", "from")

# --------------------------------------------------------------------------- #
# Time-window presence hint (validation safety net)
# --------------------------------------------------------------------------- #
# If a definition matches this but the model returned time_window == null, the
# validator raises an advisory flag (a window was likely missed, e.g. DSTR's
# "between 24 hours and 28 days"). Advisory only — never a hard failure.
import re as _re  # noqa: E402  (local alias, keeps the pattern next to its use)

TIME_WINDOW_HINT = _re.compile(
    r"\b(within|between)\b.{0,60}?\b(minute|hour|day|week|month|year)s?\b",
    _re.IGNORECASE,
)

# --------------------------------------------------------------------------- #
# Target selection
# --------------------------------------------------------------------------- #
# The 12 reaction tables to include, keyed by the page their table starts on
# (page_span[0]). This list is the single source of truth for selection: it
# already OMITS page 16 (Table 7 "Other transfusion reaction", whose case
# definition is "Not Applicable"), so that table is structurally excluded.
#
#   11 TTI        13 AHTR       14 TACO       15 TRALI
#   24 Allergic   25 DHTR       26 DSTR       27 FNHTR
#   28 HTR        29 PTP        30 TAD        31 TAGVHD
TARGET_PAGES = [11, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31]

# The source column whose cells become Column 2 (Definition). Severity and
# Imputability columns are never read.
CASE_DEFINITION_COLUMN = "Case Definition"

# Label prefix stripped when falling back to a table's label for the reaction
# name (archived Section-6 tables are labelled e.g. "Archived: DHTR").
ARCHIVED_LABEL_PREFIX = "Archived: "

# --------------------------------------------------------------------------- #
# Groq / model  (Features step only) — mirrors pdf_parser/config.py
# --------------------------------------------------------------------------- #
# Single source of truth for the model id. Do NOT add fallbacks.
VISION_MODEL = "qwen/qwen3.6-27b"

# HARD CEILING (see pdf_parser/config.py): this model on the on_demand tier
# rejects any single request that *reserves* more output than its output-tokens-
# per-minute limit (instant 429 "Request too large ... (OTPM): Limit 32000").
# max_completion_tokens must stay strictly below this.
MODEL_OTPM_LIMIT = 32000

# Per-reaction output budget. A features list is tiny, so 8192 is generous and
# leaves headroom under the OTPM ceiling. On a `length` finish the truncation
# detector retries once at MAX_COMPLETION_TOKENS_RETRY. Both stay below the cap.
MAX_COMPLETION_TOKENS = 8192
MAX_COMPLETION_TOKENS_RETRY = 16384

assert MAX_COMPLETION_TOKENS_RETRY < MODEL_OTPM_LIMIT, (
    "MAX_COMPLETION_TOKENS_RETRY must stay below the model OTPM limit or every "
    "request 429s as 'Request too large'."
)

TEMPERATURE = 0  # determinism; lowest the model supports

# qwen3 is a REASONING model. With JSON mode its <think> prefix can make the
# output invalid (400 json_validate_failed, sometimes with an empty generation).
# Disabling reasoning yields clean, fast JSON. Set to None to omit the param.
REASONING_EFFORT = "none"

GROQ_API_KEY_ENV = "GROQ_API_KEY"


def get_groq_api_key() -> str:
    """Return the Groq key or raise a clear error (only the features step needs it)."""
    key = os.environ.get(GROQ_API_KEY_ENV)
    if not key:
        raise RuntimeError(
            f"{GROQ_API_KEY_ENV} not found. Add it to {PROJECT_ROOT / '.env'}."
        )
    return key


# --------------------------------------------------------------------------- #
# Retry / backoff  (transient errors + 429s) — mirrors pdf_parser/config.py
# --------------------------------------------------------------------------- #
MAX_RETRIES = 5            # attempts on transient/429 errors before giving up
BACKOFF_INITIAL = 1.0      # seconds
BACKOFF_FACTOR = 2.0       # exponential multiplier
BACKOFF_MAX = 60.0         # cap per-sleep
BACKOFF_JITTER = 0.25      # +/- fraction of jitter to avoid thundering herd
