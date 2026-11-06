"""Central configuration for the clinical feature extractor.

This module is self-contained. It does NOT import from ``pdf_parser`` or
``reaction_extractor``; it only *reads* the deduplicated vocabulary JSON that the
``reaction_extractor`` module produced, treating it as read-only.

The Groq settings mirror the proven patterns used elsewhere in the project
(client with ``max_retries=0`` + our own backoff, a truncation detector, JSON
mode, ``temperature=0``), but this module targets its own model
(``openai/gpt-oss-120b``) with ``reasoning_effort="medium"``.

The ``.env`` at the *project root* (two levels above this file) is loaded once on
import; only ``GROQ_API_KEY`` is required, and only for the extraction step.
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
INPUT_DIR = PACKAGE_DIR / "input"
OUTPUT_DIR = PACKAGE_DIR / "output"
# Per-stage checkpoint cache. Override with CLF_CACHE_DIR to give a run its own
# isolated cache (e.g. a per-model dir so one model's cached stages never serve
# another model's — the stage cache is keyed by note text, not by model). Default
# is unchanged, so the main pipeline is unaffected.
CACHE_DIR = Path(os.environ.get("CLF_CACHE_DIR") or (OUTPUT_DIR / "cache"))

# Opt-in per-case/per-stage progress logging to stderr (CLF_PROGRESS=1). Off by
# default so the main pipeline's output is unchanged; the eval runs turn it on so
# a long run reports which case/stage it is on instead of sitting silent.
PROGRESS = os.environ.get("CLF_PROGRESS", "").strip().lower() in {"1", "true", "yes", "on"}

# Load the root .env (does nothing if absent; never overrides real env vars).
load_dotenv(PROJECT_ROOT / ".env")

# --------------------------------------------------------------------------- #
# Vocabulary source (READ-ONLY)
# --------------------------------------------------------------------------- #
# The canonical feature list. Produced by the reaction_extractor module; this
# module only reads it and must never write to it.
VOCABULARY_JSON = (
    PROJECT_ROOT
    / "reaction_extractor"
    / "output"
    / "deduplicated_symptoms_labmarkers.json"
)
# The list lives under this key in that file.
VOCABULARY_KEY = "deduplicated_symptoms_labmarkers"

# --------------------------------------------------------------------------- #
# NHSN hierarchical schema source (READ-ONLY)
# --------------------------------------------------------------------------- #
# The NHSN Hemovigilance hierarchical feature schema. For each POSITIVE feature it
# defines a menu of variables in two groups (Timing and Details) that Model 3 fills
# in per feature. This module only *reads* this workbook; it is never written to.
SCHEMA_XLSX = (
    PROJECT_ROOT
    / "NHSN_hierarchical_feature_schema_2.0.xlsx"
)
# The sheet holding the per-feature variable menus, and its column headers.
SCHEMA_SHEET = "Hierarchical Schema"
SCHEMA_COL_FEATURE = "Level 4 - Exact source feature"
SCHEMA_COL_GROUP = "Level 5 - Variable group"
SCHEMA_COL_ORDER = "Variable order"
SCHEMA_COL_VARIABLE_ID = "Variable ID"
SCHEMA_COL_DEFINITION = "Variable prompt / definition"
SCHEMA_COL_RESPONSE_TYPE = "Response type"
SCHEMA_COL_ALLOWED_VALUES = "Allowed values"
SCHEMA_COL_UNITS = "Units"
SCHEMA_COL_NOTES = "Variable notes"
SCHEMA_COL_DEPENDENCIES = "Dependencies"
# The two variable groups Model 3 runs separately for each positive feature.
SCHEMA_GROUP_TIMING = "Timing"
SCHEMA_GROUP_DETAILS = "Details"

# --------------------------------------------------------------------------- #
# Feature validation (Model 1 post-check)
# --------------------------------------------------------------------------- #
# When True (default), every feature Model 1 returns is matched against the
# vocabulary, rewritten to the canonical spelling, and any off-list item is
# dropped + logged. When False, that post-check is bypassed entirely: EVERY feature
# is kept with the model's own spelling and nothing is dropped (favours recall at
# scale). Model 1 still receives the vocabulary in its prompt either way, so it is
# still guided toward the canonical names — this only controls the after-the-fact
# enforcement. Set to False to keep every feature; the code path is always tested.
VALIDATE_FEATURES = True

# Deterministic redundancy trim (runs after validation, before Model 2). When True
# (default), negated "Evidence for absence of ..." features that repeat an identical
# evidence phrase are collapsed to the first occurrence; positive features are never
# touched. Curbs Model 1 over-generating absences from one generic phrase. It only
# removes real repeats, so it is a no-op on inputs without them. Set False to disable.
DEDUPE_ABSENCE_FEATURES = True

# --------------------------------------------------------------------------- #
# Spreadsheet layout
# --------------------------------------------------------------------------- #
SHEET_NAME = "Sheet1"
CLINICAL_TEXT_HEADER = "Clinical Text"        # Column A header
EXTRACTED_FEATURES_HEADER = "Extracted Features"  # Column B header
TREATMENTS_HEADER = "Treatments"              # Column C header (Model 4 output)
PAST_MEDICAL_HISTORY_HEADER = "Past Medical History"  # Column D header (Model 5 output)

INPUT_XLSX_DEFAULT = INPUT_DIR / "Automatic_extraction.xlsx"
OUTPUT_CSV_DEFAULT = OUTPUT_DIR / "extracted.csv"
# The per-input drop log (JSONL, one record per dropped feature) is named after
# the input file, so runs on different files don't clobber each other's logs:
# e.g. "3Stage_test.csv" -> "3Stage_test.dropped.log".
DROPPED_LOG_SUFFIX = ".dropped.log"


def dropped_log_name(input_path) -> str:
    """Return the drop-log filename for a given input file (input stem + suffix)."""
    return f"{Path(input_path).stem}{DROPPED_LOG_SUFFIX}"


# The per-input redundant-absence log (JSONL, one record per absence removed by the
# deterministic dedup step), named after the input file like the drop log.
ABSENCES_LOG_SUFFIX = ".absences.log"


def absences_log_name(input_path) -> str:
    """Return the deduped-absence-log filename for a given input file."""
    return f"{Path(input_path).stem}{ABSENCES_LOG_SUFFIX}"


# The per-input Model-3 schema log (JSONL, one record per variable_id the model
# returned that is not in a feature's Timing/Details menu — dropped, not silent).
SCHEMA_LOG_SUFFIX = ".schema.log"


def schema_log_name(input_path) -> str:
    """Return the schema-drop-log filename for a given input file."""
    return f"{Path(input_path).stem}{SCHEMA_LOG_SUFFIX}"

# --------------------------------------------------------------------------- #
# Cache
# --------------------------------------------------------------------------- #
# Bumped whenever the system prompt or the per-row output handling changes, so a
# stale cache from an older shape can never silently be reused.
#   1 = initial (model returned bare strings / evidence-less objects)
#   2 = user message now pins per-feature objects with an "evidence" field
#   3 = time / global time must carry anchoring context, not a bare duration
#   4 = two-stage pipeline: Model 1 features+evidence, Model 2 attaches time
#   5 = models switched to qwen/qwen3.6-27b (reasoning off); invalidates gpt-oss cache
#   (a v6 Model-1-prompt experiment was reverted back to 5.)
#   6 = Model 3 rewritten: relatedness label replaced by NHSN schema Timing/Details
#       variable extraction (new output shape + new prompts) — invalidates v5 cache
#   7 = Model 3 now also fills negated features: same full Timing call, then keep
#       only the first supported variable; Details stamped deterministically
CACHE_SCHEMA_VERSION = 7

# --------------------------------------------------------------------------- #
# LLM provider (Groq by default; OpenRouter opt-in for Kimi)
# --------------------------------------------------------------------------- #
# The pipeline talks to Groq by default. Setting LLM_PROVIDER=openrouter routes
# every stage through OpenRouter instead (used to run Kimi K3 in the model
# comparison). This is OPT-IN: with the var unset behaviour is byte-identical to
# before, so the public default stays Groq-only and no OpenRouter key is needed.
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "groq").lower()

# OpenRouter (OpenAI-compatible) settings, only consulted when LLM_PROVIDER=openrouter.
OPENROUTER_API_KEY_ENV = "OPENROUTER_API_KEY"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "moonshotai/kimi-k3")
# Pin the underlying provider and disable fallbacks so behaviour is consistent.
# Moonshot (first-party) serves Kimi K3 at its fixed temperature=1 and supports
# JSON output; overridable via OPENROUTER_PROVIDER.
OPENROUTER_PROVIDER = os.environ.get("OPENROUTER_PROVIDER", "moonshotai")
# Kimi K3 is a reasoning model; its hidden reasoning dominates both latency and
# token cost. Cap it (low/medium/high) so runs stay affordable and fast. Bump to
# "medium" via the env var if "low" hurts feature-matching quality.
OPENROUTER_REASONING_EFFORT = os.environ.get("OPENROUTER_REASONING_EFFORT", "low")
# Hard per-call timeout (seconds) so a hung OpenRouter request fails fast instead
# of dangling; our own retry/backoff wrapper still applies on transient errors.
OPENROUTER_TIMEOUT = float(os.environ.get("OPENROUTER_TIMEOUT", "120"))


def get_openrouter_api_key() -> str:
    """Return the OpenRouter key or raise a clear error (only the OpenRouter path needs it)."""
    key = os.environ.get(OPENROUTER_API_KEY_ENV)
    if not key:
        raise RuntimeError(
            f"{OPENROUTER_API_KEY_ENV} not found. Add it to {PROJECT_ROOT / '.env'}."
        )
    return key


# --------------------------------------------------------------------------- #
# Groq / model
# --------------------------------------------------------------------------- #
# Single source of truth for the model ids. Do NOT add fallbacks.
# The pipeline is four-stage: Model 1 extracts features + evidence; Model 2
# attaches contextual time; Model 3 fills in the NHSN schema Timing and Details
# variables per positive feature (run separately per group); Model 4 extracts
# treatments + response. Separate constants so any one stage can be swapped
# independently later without touching the others. All default to qwen; set the
# GROQ_MODEL env var to override every stage at once (used by the model-comparison
# experiment). Unset => identical to before.
_FALLBACK_MODEL = OPENROUTER_MODEL if LLM_PROVIDER == "openrouter" else "qwen/qwen3.6-27b"
_DEFAULT_MODEL = os.environ.get("GROQ_MODEL", _FALLBACK_MODEL)
FEATURES_MODEL = _DEFAULT_MODEL        # Model 1 (features + evidence)
TIME_MODEL = _DEFAULT_MODEL            # Model 2 (time only)
TIMING_SCHEMA_MODEL = _DEFAULT_MODEL   # Model 3, Timing group (NHSN schema fill-in)
DETAILS_SCHEMA_MODEL = _DEFAULT_MODEL  # Model 3, Details group (NHSN schema fill-in)
TREATMENTS_MODEL = _DEFAULT_MODEL      # Model 4 (treatments + response)
# Model 5 (past medical history) is an INDEPENDENT stage: it reads only the note,
# uses no vocabulary, and neither feeds nor depends on the four-stage extract().
PMH_MODEL = _DEFAULT_MODEL           # Model 5 (past medical history)
# Back-compat alias (some call sites / tests refer to a single extraction model).
EXTRACTION_MODEL = FEATURES_MODEL

# Reasoning-effort. qwen3 is a REASONING model whose <think> prefix breaks JSON
# mode (400 json_validate_failed, sometimes an empty generation) — the sibling
# reaction_extractor module documents this and runs qwen with reasoning disabled.
# So with qwen we MUST keep this "none". The gpt-oss models are the opposite: they
# REJECT "none" (400: must be low/medium/high) and the project's gpt-oss triagers
# run at "medium". Because the required value depends on the model, it is exposed
# as the GROQ_REASONING_EFFORT env var (used by the model-comparison experiment to
# pair each model with its own value). Unset => "none", identical to before.
REASONING_EFFORT = os.environ.get("GROQ_REASONING_EFFORT", "none")

TEMPERATURE = 0  # determinism

# Output budget per row. One flat budget per call (no token-escalation retry); a
# `length` finish is a hard error. The value is per-model via the
# GROQ_MAX_COMPLETION_TOKENS env var (default 16384 = qwen's max_completion_tokens).
# The gpt-oss models cap at 65536; the comparison experiment runs them at 50000.
#
# TOKENS_PER_MINUTE is this account's measured Groq rate limit (the
# x-ratelimit-limit-tokens header = 250000). A single request's completion budget
# must stay under it or the request 429s as 'Request too large'.
TOKENS_PER_MINUTE = 250000
MAX_COMPLETION_TOKENS = int(os.environ.get("GROQ_MAX_COMPLETION_TOKENS", "16384"))

assert MAX_COMPLETION_TOKENS < TOKENS_PER_MINUTE, (
    "MAX_COMPLETION_TOKENS must stay below the per-minute token limit or every "
    "request 429s as 'Request too large'."
)

GROQ_API_KEY_ENV = "GROQ_API_KEY"


def get_groq_api_key() -> str:
    """Return the Groq key or raise a clear error (only the extraction step needs it)."""
    key = os.environ.get(GROQ_API_KEY_ENV)
    if not key:
        raise RuntimeError(
            f"{GROQ_API_KEY_ENV} not found. Add it to {PROJECT_ROOT / '.env'}."
        )
    return key


# --------------------------------------------------------------------------- #
# Retry / backoff (transient errors + 429s)
# --------------------------------------------------------------------------- #
MAX_RETRIES = 10           # attempts on transient/429 errors before giving up
BACKOFF_INITIAL = 1.0      # seconds
BACKOFF_FACTOR = 2.0       # exponential multiplier
BACKOFF_MAX = 60.0         # cap per-sleep
BACKOFF_JITTER = 0.25      # +/- fraction of jitter to avoid thundering herd
# When a rate-limit error carries the server's own "retry after" hint (a
# Retry-After header, or OpenRouter's retry_after_seconds), we wait THAT long
# (plus a small margin) instead of guessing with exponential backoff -- landing
# the retry right when the limit clears. Capped so a bad hint can't hang a run.
RETRY_AFTER_MAX = 30.0     # cap the server-suggested wait (seconds)
RETRY_AFTER_MARGIN = 0.5   # small margin added to the suggested wait
