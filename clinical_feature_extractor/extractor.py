"""The AI step: a FIVE-STAGE pipeline per clinical text + deterministic checks.

Stages 1-4 are chained inside ``extract()``; Stage 5 (past medical history) is an
INDEPENDENT method (``extract_past_medical_history``) that reads only the note.

Stage 1 (Model 1, features): the clinical note + the canonical feature list ->
``{feature, evidence}`` objects only (no time). This raw output is validated
against the vocabulary by ``validate_features``.

Stage 2 (Model 2, time): the clinical note + the *already-extracted* numbered
features (with evidence, but NOT the feature list) -> a time for each feature,
stated with its anchoring context. The times are
merged onto Stage-1's features **programmatically** — Stage 1's feature/evidence
are authoritative and Model 2 can never alter or drop them; it only supplies a
time per index (``"uncertain"`` when none is tied).

Stage 3 (Model 3, NHSN schema fill-in): for each POSITIVE feature from Stage 2,
look up its NHSN Hemovigilance variable menu (from the hierarchical schema) and
ask the model to fill in the applicable variables — run SEPARATELY for the two
variable groups, ``Timing`` and ``Details``. Each call sees the note, the one
feature (name + evidence), and that group's menu (variable id, definition,
response type, allowed values, units); it returns only the variables the note
supports, each with a value (and a unit for numeric variables). The results are
merged onto the feature as a nested ``schema`` object **deterministically**: only
variables in that feature's menu are kept (an unknown ``variable_id`` is dropped
and LOGGED), and each kept variable's ``notes``/``dependencies`` are attached from
the schema, never from the model. POSITIVE features get their full Timing and
Details; NEGATED features go through the SAME full Timing call (then kept trimmed
to the first supported variable) and get their single Details variable stamped
deterministically from the already-extracted evidence phrase (no Details LLM call).

Stage 4 (Model 4, treatments): the clinical note + ALL the POSITIVE features found
above (name + evidence + time, so they are easy to locate in the note) -> a raw
list of treatments, each with its response, timing, and a ``feature_list`` flag.
The nested ``schema`` object is stripped before this call, so Model 4 still sees
only name/evidence/time and its prompt shape is unchanged from the earlier
pipeline. This stage is deliberately UNVALIDATED: the model's ``treatments`` list
is stored exactly as returned (no matching, filtering, or reshaping).

Stage 5 (Model 5, past medical history): the clinical note alone -> a raw list of
``{condition, detail}`` objects. It uses no vocabulary and no feature list, and
neither feeds nor depends on Stages 1-4, so it is exposed as its own method.
Like Stage 4 it is deliberately UNVALIDATED (stored exactly as returned).

All stages mirror the project's proven Groq setup: a client with
``max_retries=0`` (our own backoff + jitter on 429/transient), JSON mode,
``temperature=0``, ``reasoning_effort`` from config, and a truncation detector.
Each stage's raw output is cached separately under ``output/cache/`` so re-runs
cost no tokens; ``use_cache=False`` bypasses them all.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import sys
import time
from pathlib import Path
from typing import Optional

from . import config, schema
from .vocabulary import Vocabulary

# Where the positive feature list is injected into the Model 4 (treatments) prompt.
POSITIVE_FEATURES_MARKER = "<Positive Features List>"
# Negated features ("Evidence for absence of ...") describe an ABSENT symptom, so
# they are not passed to Model 4 — you do not treat a symptom that isn't there.
NEGATED_FEATURE_PREFIX = "evidence for absence of"

# Key under which Model 3's filled-in NHSN variables are nested on a positive
# feature: ``{"schema": {"timing": [...], "details": [...]}}``. The two inner keys
# are the output labels for the schema's ``Timing`` / ``Details`` variable groups.
SCHEMA_KEY = "schema"
SCHEMA_GROUP_OUTPUT_KEY = {schema.TIMING: "timing", schema.DETAILS: "details"}


class ExtractionError(RuntimeError):
    """An extraction call failed in a way that must not be silently accepted."""


class TruncationError(ExtractionError):
    """The model output was truncated (finish_reason == 'length') even after retry."""


# --------------------------------------------------------------------------- #
# Shared free helpers (mirror the project's Groq conventions)
# --------------------------------------------------------------------------- #
def _transient_exceptions() -> tuple[type[BaseException], ...]:
    """Groq/OpenAI error types worth retrying (rate limits / transient server errors).

    Both SDKs expose the same-named exception classes; the OpenRouter path uses the
    openai client, so its 429/timeout/connection errors must be caught too or a
    transient rate limit crashes the run instead of backing off.
    """
    types: list[type[BaseException]] = []
    for mod_name in ("groq", "openai"):
        try:
            mod = __import__(mod_name)
        except Exception:  # pragma: no cover - module may be absent
            continue
        for name in ("RateLimitError", "APITimeoutError", "APIConnectionError",
                     "InternalServerError"):
            t = getattr(mod, name, None)
            if isinstance(t, type):
                types.append(t)
    return tuple(types)


def _is_json_validate_failed(exc: BaseException) -> bool:
    """True for Groq's 400 json_validate_failed."""
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        return body.get("error", {}).get("code") == "json_validate_failed"
    return False


def _retry_after_seconds(exc: BaseException):
    """The server's suggested wait (seconds) for a rate-limit error, or None.

    Reads the standard ``Retry-After`` response header (Groq and OpenRouter both
    send it) and, as a fallback, OpenRouter's ``retry_after_seconds`` in the error
    body metadata. Returns None when no usable hint is present.
    """
    resp = getattr(exc, "response", None)
    headers = getattr(resp, "headers", None)
    if headers:
        val = headers.get("retry-after") or headers.get("Retry-After")
        if val:
            try:
                return float(val)
            except (TypeError, ValueError):
                pass
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        meta = body.get("error", {}).get("metadata", {})
        if isinstance(meta, dict) and meta.get("retry_after_seconds") is not None:
            try:
                return float(meta["retry_after_seconds"])
            except (TypeError, ValueError):
                pass
    return None


def _strip_code_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z0-9]*\s*", "", t)
        t = re.sub(r"\s*```$", "", t)
    return t.strip()


def _load_features_system_prompt() -> str:
    """Model 1 system prompt: features + evidence only, no time."""
    return (config.PROMPTS_DIR / "features_system.md").read_text(encoding="utf-8")


def _load_time_system_prompt() -> str:
    """Model 2 system prompt: attach contextual time to already-found features."""
    return (config.PROMPTS_DIR / "time.md").read_text(encoding="utf-8")


def _load_timing_schema_system_prompt() -> str:
    """Model 3 (Timing group): fill the NHSN Timing variables for one feature."""
    return (config.PROMPTS_DIR / "schema_timing.md").read_text(encoding="utf-8")


def _load_details_schema_system_prompt() -> str:
    """Model 3 (Details group): fill the NHSN Details variables for one feature."""
    return (config.PROMPTS_DIR / "schema_details.md").read_text(encoding="utf-8")


def _load_treatments_system_prompt() -> str:
    """Model 4 system prompt: extract treatments/response for the found features."""
    return (config.PROMPTS_DIR / "treatments.md").read_text(encoding="utf-8")


def _load_pmh_system_prompt() -> str:
    """Model 5 system prompt: extract the patient's past medical history from the note."""
    return (config.PROMPTS_DIR / "past_medical_history.md").read_text(encoding="utf-8")


def _text_hash(*parts: str) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode("utf-8"))
        h.update(b"\x00")  # separator so concatenations can't collide
    return h.hexdigest()


def _progress(msg: str) -> None:
    """Emit a timestamped progress line to stderr when CLF_PROGRESS is on.

    No-op by default, so the main pipeline stays silent. Flushed every line so a
    ``tail -f`` on a redirected log shows each case/stage the moment it happens.
    """
    if not config.PROGRESS:
        return
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


def _is_int(value) -> bool:
    """True for a real int (bool is excluded, since bool is an int subclass)."""
    return isinstance(value, int) and not isinstance(value, bool)


# --------------------------------------------------------------------------- #
# Deterministic vocabulary post-check (pure; no I/O, no model)
# --------------------------------------------------------------------------- #
def validate_features(raw_features: list, vocab: Vocabulary) -> dict:
    """Validate a raw ``features`` list against the canonical vocabulary.

    The model is inconsistent about item shape: a feature may arrive as a bare
    string (``"Diarrhea"``) OR as an object (``{"feature": "Diarrhea",
    "evidence": ...}``). Both are handled.

    Rules (deterministic, identical every run):
      * A bare-string item is treated as the feature name: matched
        case-insensitively, and on a hit reconstructed as ``{"feature": canonical}``.
      * An object item with a ``feature`` is matched case-insensitively; on a hit
        the ``feature`` is rewritten to the canonical casing and ALL other keys
        (``evidence``, ``time``, …) are preserved.
      * On a miss the whole item is dropped and recorded.
      * ``evidence`` / ``time`` are free text and are never validated.
      * Any other item shape is dropped and recorded (never silently discarded).

    Returns ``{"features": [...kept...], "dropped": [...records...]}``. Each dropped
    record is ``{"feature": <str|None>, "item": <original item>}``.
    """
    kept: list = []
    dropped: list = []
    if not isinstance(raw_features, list):
        return {"features": kept, "dropped": dropped}

    for item in raw_features:
        # Bare string: the string itself is the candidate feature name.
        if isinstance(item, str):
            canonical = vocab.match(item)
            if canonical is not None:
                kept.append({"feature": canonical})
            else:
                dropped.append({"feature": item, "item": item})
            continue
        # Object carrying a feature: match, normalize casing, preserve all keys.
        if isinstance(item, dict) and isinstance(item.get("feature"), str):
            canonical = vocab.match(item["feature"])
            if canonical is not None:
                normalized = dict(item)          # keep evidence / time / any extras
                normalized["feature"] = canonical  # normalize to canonical casing
                kept.append(normalized)
            else:
                dropped.append({"feature": item["feature"], "item": item})
            continue
        # Neither a string nor a well-formed feature object.
        feat = item.get("feature") if isinstance(item, dict) else None
        dropped.append({"feature": feat if isinstance(feat, str) else None,
                        "item": item})
    return {"features": kept, "dropped": dropped}


def passthrough_features(raw_features: list) -> dict:
    """Keep every feature Model 1 returns — the no-validation path.

    Used when ``config.VALIDATE_FEATURES`` is off. Nothing is matched against the
    vocabulary and nothing is dropped; items are only normalized in SHAPE so the
    downstream stages (time numbering, treatments) still work:
      * a bare string ``"Diarrhea"`` becomes ``{"feature": "Diarrhea"}`` — the
        model's own spelling is preserved, never rewritten to a canonical form;
      * a dict is passed through unchanged (``feature``/``evidence``/any keys kept);
      * any other item is passed through as-is (still never dropped).

    Returns ``{"features": [...], "dropped": []}`` — ``dropped`` is always empty,
    matching ``validate_features``'s shape so callers need no special-casing.
    """
    kept: list = []
    if not isinstance(raw_features, list):
        return {"features": kept, "dropped": []}
    for item in raw_features:
        if isinstance(item, str):
            if item.strip():                       # skip empty/whitespace-only
                kept.append({"feature": item})     # model's spelling, verbatim
        else:
            kept.append(item)                      # dict or other — keep as-is
    return {"features": kept, "dropped": []}


# --------------------------------------------------------------------------- #
# Deterministic Stage-1 + Stage-2 merge (pure; no I/O, no model)
# --------------------------------------------------------------------------- #
UNCERTAIN_TIME = "uncertain"


def merge_time(features: list, time_data: dict) -> list:
    """Attach Stage-2 times onto Stage-1 features, programmatically.

    Stage-1 ``features`` (each a ``{feature, evidence, ...}`` object) are
    AUTHORITATIVE: they are copied through unchanged except for an added ``time``.
    Model 2 supplies times only by 1-based ``index``; it can never rename, drop, or
    reorder a feature here. A feature with no supplied (non-empty) time gets
    ``"uncertain"``.

    ``time_data`` is ``{"times": [{"index": int, "time": str}]}`` (missing/typed-wrong
    parts are tolerated as empty).
    """
    times_by_index: dict[int, str] = {}
    for entry in (time_data.get("times") or []):
        if not isinstance(entry, dict):
            continue
        idx = entry.get("index")
        val = entry.get("time")
        if _is_int(idx) and isinstance(val, str) and val.strip():
            times_by_index.setdefault(idx, val.strip())

    merged: list = []
    for i, feat in enumerate(features, start=1):
        if isinstance(feat, dict) and "feature" in feat:
            obj = dict(feat)                       # authoritative feature/evidence
            obj["time"] = times_by_index.get(i, UNCERTAIN_TIME)
            merged.append(obj)
        else:
            merged.append(feat)                    # pass through (other item)
    return merged


# --------------------------------------------------------------------------- #
# Deterministic Stage-3 schema merge (pure; no I/O, no model)
# --------------------------------------------------------------------------- #
def _coerce_number(value):
    """Return a normalized JSON number for a numeric ``value``, or None if it is
    not numeric.

    Accepts a real number (the model is asked to emit a bare JSON number) or a
    numeric string like ``"6.25"``. Float-representation noise (e.g. 6.2500001) is
    rounded off, and an integral result comes back as an ``int`` so the output
    carries ``59`` rather than ``59.0``. Anything non-numeric (the categorical
    fallbacks ``not_documented`` / ``uncertain`` / ``not_applicable``) returns None
    so the caller can keep it as text.
    """
    if isinstance(value, bool):            # bool is an int subclass; never a measure
        return None
    if isinstance(value, (int, float)):
        num = float(value)
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            num = float(text)
        except ValueError:
            return None
    else:
        return None
    num = round(num, 6)                    # drop float noise; timing needs no more
    return int(num) if num == int(num) else num


def _model_variables_by_id(group_data: dict) -> dict:
    """Map ``variable_id`` -> the model's entry, from ``{"variables": [...]}``.

    First occurrence wins on a duplicate id; malformed entries are ignored.
    """
    by_id: dict = {}
    for entry in (group_data.get("variables") or []):
        if isinstance(entry, dict) and isinstance(entry.get("variable_id"), str):
            by_id.setdefault(entry["variable_id"].strip(), entry)
    return by_id


def merge_schema_group(menu: tuple, group_data: dict) -> tuple[list, list]:
    """Merge one group's model output against that feature's variable menu.

    ``menu`` is the feature's ordered tuple of :class:`~clinical_feature_extractor.
    schema.SchemaVariable` for a group (Timing or Details). Deterministic and
    schema-authoritative: we walk the menu IN ORDER and keep only the variables the
    model supplied a usable value for. The model can never invent a variable — an id
    it returns that is not in the menu is reported in ``unknown_ids`` (for logging),
    never emitted. Each kept entry is lean:
      * ``variable_id`` and ``value`` always;
      * ``unit`` only for a ``numeric`` variable, taken from the model;
      * ``notes`` / ``dependencies`` attached from the SCHEMA (not the model) when
        that variable defines them.

    Returns ``(entries, unknown_ids)``.
    """
    returned = _model_variables_by_id(group_data)
    entries: list = []
    for var in menu:
        got = returned.get(var.variable_id)
        if got is None:
            continue                                   # note said nothing → omit (lean)
        raw_value = got.get("value")
        if var.response_type == "numeric":
            # A numeric variable's value is a real number (e.g. 6.25); its only other
            # legal answers are the categorical fallbacks (not_documented / uncertain
            # / not_applicable), which stay as text. A blank or missing value is omitted.
            number = _coerce_number(raw_value)
            if number is not None:
                value = number
            elif isinstance(raw_value, str) and raw_value.strip():
                value = raw_value.strip()
            else:
                continue                               # no usable value → omit
        else:
            if not (isinstance(raw_value, str) and raw_value.strip()):
                continue                               # no usable value → omit
            value = raw_value.strip()
        out = {"variable_id": var.variable_id, "value": value}
        if var.response_type == "numeric":
            unit = got.get("unit")
            if isinstance(unit, str) and unit.strip():
                out["unit"] = unit.strip()
        if var.notes:
            out["notes"] = var.notes
        if var.dependencies:
            out["dependencies"] = var.dependencies
        entries.append(out)
    menu_ids = {var.variable_id for var in menu}
    unknown_ids = [vid for vid in returned if vid not in menu_ids]
    return entries, unknown_ids


def positive_features(features: list) -> list:
    """Return the positive (non-negated) feature objects from a merged feature list.

    Model 4 (treatments) is fed only POSITIVE features: items that are dicts with a
    ``feature`` key whose name is not a negated ``Evidence for absence of ...``
    form. Items without a ``feature`` key are excluded. Each returned item
    is the original object (``feature``/``evidence``/``time`` preserved), so the
    treatments prompt can use the evidence phrase to locate the feature in the note.
    """
    out: list = []
    for f in features:
        if isinstance(f, dict) and isinstance(f.get("feature"), str):
            if not f["feature"].strip().lower().startswith(NEGATED_FEATURE_PREFIX):
                out.append(f)
    return out


def dedupe_absence_features(features: list) -> list:
    """Collapse redundant negated features that repeat an identical evidence phrase.

    For NEGATED features (``Evidence for absence of ...``) that share the same
    non-empty ``evidence`` string, keep only the FIRST (in list order) and drop the
    rest. POSITIVE features are always kept (presence outweighs absence). Negated
    features with empty/missing evidence are all kept. Order is preserved. Global-time
    and other items pass through untouched.

    Deterministic — pure string logic, no vocabulary, no model. Evidence is matched
    exactly after trimming surrounding whitespace (case-sensitive); at
    ``temperature=0`` this collapses the real repeats (one generic phrase reused as
    the evidence for many absences) without merging genuinely distinct wording. It
    only removes true repeats, so it is a no-op on inputs that have none.
    """
    seen_evidence: set = set()
    out: list = []
    for f in features:
        if (isinstance(f, dict) and isinstance(f.get("feature"), str)
                and f["feature"].strip().lower().startswith(NEGATED_FEATURE_PREFIX)):
            evidence = (f.get("evidence") or "").strip()
            if evidence:
                if evidence in seen_evidence:
                    continue                       # redundant negated on same evidence
                seen_evidence.add(evidence)
        out.append(f)
    return out


# --------------------------------------------------------------------------- #
# The extractor
# --------------------------------------------------------------------------- #
class ClinicalFeatureExtractor:
    """Extracts validated clinical features for one clinical text at a time.

    ``extract(text)`` returns ``{"features": [...], "dropped": [...]}`` where
    ``features`` are vocabulary-validated (canonical casing) and ``dropped`` lists
    every off-list / malformed item for the caller to log. Raw model output is
    cached per clinical text.
    """

    def __init__(self, vocabulary: Vocabulary, client=None,
                 cache_dir: Optional[Path] = None, feature_schema=None):
        self.vocab = vocabulary
        self._client = client
        self.cache_dir = Path(cache_dir) if cache_dir else config.CACHE_DIR
        self._features_system = _load_features_system_prompt()
        self._time_system = _load_time_system_prompt()
        self._timing_schema_system = _load_timing_schema_system_prompt()
        self._details_schema_system = _load_details_schema_system_prompt()
        self._treatments_system = _load_treatments_system_prompt()
        self._pmh_system = _load_pmh_system_prompt()
        # The NHSN schema (Model 3 input). Loaded lazily on first use so an extractor
        # built with a small/fake vocabulary (e.g. in tests) pays nothing until a
        # schema call actually happens; tests may inject one directly. The schema
        # canonicalizes its own feature names against the FULL vocabulary, so it is
        # independent of ``self.vocab``.
        self._feature_schema = feature_schema

    @property
    def feature_schema(self):
        """The NHSN :class:`~clinical_feature_extractor.schema.FeatureSchema` (lazy)."""
        if self._feature_schema is None:
            self._feature_schema = schema.load_schema()
        return self._feature_schema

    # -- client -------------------------------------------------------------- #
    @property
    def client(self):
        if self._client is None:
            if config.LLM_PROVIDER == "openrouter":
                # OpenRouter is OpenAI-API-compatible; used to run Kimi K3.
                import openai
                self._client = openai.OpenAI(
                    base_url=config.OPENROUTER_BASE_URL,
                    api_key=config.get_openrouter_api_key(),
                    max_retries=0,
                    timeout=config.OPENROUTER_TIMEOUT,
                )
            else:
                import groq
                self._client = groq.Groq(api_key=config.get_groq_api_key(), max_retries=0)
        return self._client

    # -- prompts ------------------------------------------------------------- #
    def _vocabulary_block(self) -> str:
        return "\n".join(f"- {item}" for item in self.vocab.items)

    def _build_features_user_message(self, clinical_text: str) -> str:
        """Stage 1 (Model 1): the canonical list + the clinical text, features only."""
        return (
            "Feature list (match against these exactly; return the negated form "
            "verbatim when a feature is stated to be absent):\n"
            f"{self._vocabulary_block()}\n\n"
            "Clinical text:\n"
            f"{clinical_text}\n\n"
            'Return JSON of the shape {"features": [...]}. Each matched feature '
            'MUST be an object, never a bare string, with exactly:\n'
            '  - "feature": the matching item from the list above,\n'
            '  - "evidence": the exact phrase from the clinical text supporting it.\n'
            'Do NOT include any "time" key or any "global time" entry — timing is '
            'handled in a separate step.'
        )

    def _build_time_user_message(self, clinical_text: str, features: list) -> str:
        """Stage 2 (Model 2): the note + the numbered already-found features."""
        lines = []
        for i, feat in enumerate(features, start=1):
            name = feat.get("feature", "") if isinstance(feat, dict) else str(feat)
            ev = feat.get("evidence", "") if isinstance(feat, dict) else ""
            lines.append(f"{i}. Feature: {name}\n   Evidence: {ev}")
        numbered = "\n".join(lines) if lines else "(no features were extracted)"
        return (
            "Clinical note:\n"
            f"{clinical_text}\n\n"
            "Features already extracted from this note (do NOT change them; attach "
            "a time to each by its number):\n"
            f"{numbered}\n\n"
            'Return JSON of the shape {"times": [{"index": <number>, "time": '
            '"<time with anchoring context, or \'uncertain\'>"}]}. Include one '
            '"times" entry for every numbered feature above.'
        )

    def _build_schema_user_message(self, clinical_text: str, feature: dict,
                                   group: str, menu: tuple) -> str:
        """Stage 3 (Model 3): the note + one feature + that group's variable menu.

        ``menu`` is the ordered tuple of :class:`~clinical_feature_extractor.schema.
        SchemaVariable` for ``feature`` in ``group`` (Timing or Details). Each menu
        item is rendered with its definition, response type, allowed values, and (when
        present) its unit menu — everything the model needs to choose a value.
        """
        name = feature.get("feature", "") if isinstance(feature, dict) else str(feature)
        ev = feature.get("evidence", "") if isinstance(feature, dict) else ""
        blocks = []
        for i, var in enumerate(menu, start=1):
            lines = [f"{i}. {var.variable_id}",
                     f"   definition: {var.definition}",
                     f"   response type: {var.response_type}",
                     f"   allowed values: {' | '.join(var.allowed_values)}"]
            if var.units:
                lines.append(f"   units: {' | '.join(var.units)}")
            blocks.append("\n".join(lines))
        menu_text = "\n".join(blocks) if blocks else "(no variables for this group)"
        group_word = SCHEMA_GROUP_OUTPUT_KEY.get(group, group.lower())
        return (
            "Clinical note:\n"
            f"{clinical_text}\n\n"
            f"Feature: {name}\n"
            f'Evidence: "{ev}"\n\n'
            f"{group_word.capitalize()} variables possible for this feature:\n\n"
            f"{menu_text}\n\n"
            'Return JSON of the shape {"variables": [{"variable_id": "...", '
            '"value": "...", "unit": "..." (numeric only)}]}, including only the '
            "variables the note supports."
        )

    def _positive_features_block(self, pos_features: list) -> str:
        """Render the positive features (name + evidence + time) for the marker."""
        lines = []
        for f in pos_features:
            name = f.get("feature", "") if isinstance(f, dict) else str(f)
            parts = [f"- {name}"]
            ev = f.get("evidence") if isinstance(f, dict) else None
            if isinstance(ev, str) and ev.strip():
                parts.append(f'evidence: "{ev.strip()}"')
            tm = f.get("time") if isinstance(f, dict) else None
            if isinstance(tm, str) and tm.strip():
                parts.append(f"time: {tm.strip()}")
            lines.append(" | ".join(parts))
        return "\n".join(lines) if lines else "(no features were extracted from this note)"

    def _build_treatments_system_prompt(self, pos_features: list) -> str:
        """Stage 4 (Model 4) system prompt: the verbatim prompt with the positive
        feature list substituted at its ``<Positive Features List>`` marker."""
        block = self._positive_features_block(pos_features)
        template = self._treatments_system
        if POSITIVE_FEATURES_MARKER in template:
            return template.replace(POSITIVE_FEATURES_MARKER, block)
        return f"{template}\n{block}"

    def _build_treatments_user_message(self, clinical_text: str) -> str:
        """Stage 4 (Model 4): the clinical note (features live in the system prompt)."""
        return f"Clinical text:\n{clinical_text}"

    def _build_pmh_user_message(self, clinical_text: str) -> str:
        """Model 5: the clinical note only (no vocabulary, no feature list)."""
        return f"Clinical text:\n{clinical_text}"

    # -- one raw call -------------------------------------------------------- #
    def _call_once(self, system_prompt: str, user_message: str, max_tokens: int,
                   model: str):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]
        if config.LLM_PROVIDER == "openrouter":
            # Kimi K3 via OpenRouter: temperature is fixed at 1.0 server-side and
            # must be OMITTED; reasoning is returned in a separate field (not in
            # content), so json_object keeps content clean. Pin the provider and
            # disable fallbacks so every call hits the same backend.
            kwargs = dict(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
                extra_body={
                    "provider": {
                        "order": [config.OPENROUTER_PROVIDER],
                        "allow_fallbacks": False,
                    },
                    # Cap the reasoning to keep latency and token cost bounded.
                    "reasoning": {"effort": config.OPENROUTER_REASONING_EFFORT},
                },
            )
            return self.client.chat.completions.create(**kwargs)

        kwargs = dict(
            model=model,
            messages=messages,
            temperature=config.TEMPERATURE,
            max_completion_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
        if config.REASONING_EFFORT is not None:
            kwargs["reasoning_effort"] = config.REASONING_EFFORT
        return self.client.chat.completions.create(**kwargs)

    # -- retry wrapper ------------------------------------------------------- #
    def _sleep_backoff(self, attempt: int, exc=None) -> None:
        # Prefer the server's own retry-after hint (precise) over blind backoff.
        hint = _retry_after_seconds(exc) if exc is not None else None
        if hint is not None:
            delay = min(hint, config.RETRY_AFTER_MAX) + config.RETRY_AFTER_MARGIN
        else:
            delay = min(config.BACKOFF_INITIAL * (config.BACKOFF_FACTOR ** attempt),
                        config.BACKOFF_MAX)
        jitter = delay * config.BACKOFF_JITTER
        time.sleep(max(0.0, delay + random.uniform(-jitter, jitter)))

    def _call_with_retries(self, system_prompt: str, user_message: str,
                           max_tokens: int, model: str):
        transient = _transient_exceptions()
        last_exc: Optional[BaseException] = None
        for attempt in range(config.MAX_RETRIES):
            try:
                return self._call_once(system_prompt, user_message, max_tokens, model)
            except transient as exc:  # type: ignore[misc]
                last_exc = exc
            except Exception as exc:
                if _is_json_validate_failed(exc):
                    last_exc = exc
                else:
                    raise
            if attempt < config.MAX_RETRIES - 1:
                self._sleep_backoff(attempt, last_exc)
        raise ExtractionError(
            f"extraction call failed after {config.MAX_RETRIES} attempts: {last_exc}"
        )

    # -- truncation-aware call ---------------------------------------------- #
    def _call(self, system_prompt: str, user_message: str, model: str) -> str:
        """Call once at the flat completion budget; refuse a truncated answer."""
        resp = self._call_with_retries(system_prompt, user_message,
                                        config.MAX_COMPLETION_TOKENS, model)
        if self._finish_reason(resp) == "length":
            raise TruncationError(
                "extraction output truncated (finish_reason='length') at the "
                "completion token budget; refusing to accept a partial list."
            )
        return self._content(resp)

    @staticmethod
    def _finish_reason(resp) -> Optional[str]:
        try:
            return resp.choices[0].finish_reason
        except Exception:
            return None

    @staticmethod
    def _content(resp) -> str:
        return resp.choices[0].message.content or ""

    # -- parse --------------------------------------------------------------- #
    @staticmethod
    def _parse_features(content: str) -> list:
        """Extract the raw ``features`` list from the model's JSON object.

        Raises ExtractionError if the response is not a JSON object or lacks a
        list-valued ``features`` key — a malformed response never silently
        becomes an empty result.
        """
        try:
            data = json.loads(_strip_code_fences(content))
        except json.JSONDecodeError as exc:
            raise ExtractionError(
                f"extraction response was not valid JSON: {exc}") from exc
        if not isinstance(data, dict) or "features" not in data:
            raise ExtractionError(
                f"extraction response missing 'features' key: {content[:200]!r}")
        features = data["features"]
        if not isinstance(features, list):
            raise ExtractionError(
                f"'features' must be a list, got {type(features).__name__}")
        return features

    @staticmethod
    def _parse_times(content: str) -> dict:
        """Parse Stage-2 JSON into ``{"times": [...]}``.

        Raises ExtractionError if the response is not a JSON object — a malformed
        response never silently becomes an empty result. A missing/typed-wrong
        ``times`` is tolerated as empty (means "no time found").
        """
        try:
            data = json.loads(_strip_code_fences(content))
        except json.JSONDecodeError as exc:
            raise ExtractionError(
                f"time response was not valid JSON: {exc}") from exc
        if not isinstance(data, dict):
            raise ExtractionError(
                f"time response was not a JSON object: {content[:200]!r}")
        times = data.get("times")
        return {"times": times if isinstance(times, list) else []}

    @staticmethod
    def _parse_schema(content: str) -> dict:
        """Parse a Stage-3 group response into ``{"variables": [...]}``.

        Raises ExtractionError if the response is not a JSON object — a malformed
        response never silently becomes an empty result. A missing/typed-wrong
        ``variables`` is tolerated as empty (means the note supported no variable in
        this group).
        """
        try:
            data = json.loads(_strip_code_fences(content))
        except json.JSONDecodeError as exc:
            raise ExtractionError(
                f"schema response was not valid JSON: {exc}") from exc
        if not isinstance(data, dict):
            raise ExtractionError(
                f"schema response was not a JSON object: {content[:200]!r}")
        variables = data.get("variables")
        return {"variables": variables if isinstance(variables, list) else []}

    @staticmethod
    def _parse_treatments(content: str) -> list:
        """Parse Stage-4 JSON into the raw ``treatments`` list — NO validation.

        Per requirement, Model 4's output is taken directly: the treatment items are
        never matched, filtered, or reshaped. We only enforce that the response is a
        JSON object (a malformed response never silently becomes an empty result) and
        return its ``treatments`` list verbatim (empty list if the key is absent).
        """
        try:
            data = json.loads(_strip_code_fences(content))
        except json.JSONDecodeError as exc:
            raise ExtractionError(
                f"treatments response was not valid JSON: {exc}") from exc
        if not isinstance(data, dict):
            raise ExtractionError(
                f"treatments response was not a JSON object: {content[:200]!r}")
        treatments = data.get("treatments")
        return treatments if isinstance(treatments, list) else []

    @staticmethod
    def _parse_pmh(content: str) -> list:
        """Parse Model 5 JSON into the raw ``past_medical_history`` list — NO validation.

        Like the treatments stage, the model's output is taken directly: items are
        never matched, filtered, or reshaped. We only enforce that the response is a
        JSON object (a malformed response never silently becomes an empty result) and
        return its ``past_medical_history`` list verbatim (empty list if absent).
        """
        try:
            data = json.loads(_strip_code_fences(content))
        except json.JSONDecodeError as exc:
            raise ExtractionError(
                f"past-medical-history response was not valid JSON: {exc}") from exc
        if not isinstance(data, dict):
            raise ExtractionError(
                f"past-medical-history response was not a JSON object: {content[:200]!r}")
        pmh = data.get("past_medical_history")
        return pmh if isinstance(pmh, list) else []

    # -- cache (per stage) --------------------------------------------------- #
    def _features_cache_path(self, clinical_text: str) -> Path:
        return self.cache_dir / f"features_{_text_hash(clinical_text)}.json"

    def _time_cache_path(self, clinical_text: str, features: list) -> Path:
        # Prompt-aware: including the time system prompt in the key means a change to
        # time.md invalidates ONLY the time stage (Model 1's features cache and Model
        # 3's treatments cache — keyed differently — stay valid, so they don't re-run).
        key = _text_hash(clinical_text,
                         json.dumps(features, ensure_ascii=False, sort_keys=True),
                         self._time_system)
        return self.cache_dir / f"time_{key}.json"

    def _schema_cache_path(self, clinical_text: str, feature: dict,
                           group: str, system_prompt: str) -> Path:
        # Prompt-aware and per feature per group: the key includes the note, the one
        # feature (name/evidence/time), the group, and that group's system prompt, so
        # editing schema_timing.md / schema_details.md invalidates ONLY that group's
        # cached calls, and each feature's two group calls are cached independently.
        key = _text_hash(clinical_text,
                         json.dumps(feature, ensure_ascii=False, sort_keys=True),
                         group, system_prompt)
        return self.cache_dir / f"schema_{key}.json"

    def _treatments_cache_path(self, clinical_text: str, pos_features: list) -> Path:
        key = _text_hash(clinical_text,
                         json.dumps(pos_features, ensure_ascii=False, sort_keys=True))
        return self.cache_dir / f"treatments_{key}.json"

    def _pmh_cache_path(self, clinical_text: str) -> Path:
        # Prompt-aware (like the time / schema stages): including the PMH system
        # prompt in the key means editing past_medical_history.md invalidates ONLY
        # this stage. Keyed on the note alone otherwise — it uses no other input.
        key = _text_hash(clinical_text, self._pmh_system)
        return self.cache_dir / f"pmh_{key}.json"

    def _read_cache(self, path: Path, payload_key: str):
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
        if not isinstance(data, dict):
            return None
        if data.get("schema_version") != config.CACHE_SCHEMA_VERSION:
            return None
        return data.get(payload_key)

    def _write_cache(self, path: Path, payload_key: str, payload,
                     model: str, clinical_text: str) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"schema_version": config.CACHE_SCHEMA_VERSION,
                        "model": model,
                        "clinical_text": clinical_text,
                        payload_key: payload},
                       ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # -- stages -------------------------------------------------------------- #
    def _features_stage(self, clinical_text: str, use_cache: bool) -> list:
        """Stage 1: return the raw ``[{feature, evidence}, ...]`` list (cached)."""
        path = self._features_cache_path(clinical_text)
        if use_cache:
            cached = self._read_cache(path, "raw_features")
            if isinstance(cached, list):
                return cached
        raw = self._parse_features(self._call(
            self._features_system, self._build_features_user_message(clinical_text),
            config.FEATURES_MODEL))
        self._write_cache(path, "raw_features", raw,
                          config.FEATURES_MODEL, clinical_text)
        return raw

    def _time_stage(self, clinical_text: str, features: list,
                    use_cache: bool) -> dict:
        """Stage 2: return raw ``{"times": [...]}`` (cached)."""
        path = self._time_cache_path(clinical_text, features)
        if use_cache:
            cached = self._read_cache(path, "raw_times")
            if isinstance(cached, dict):
                return cached
        raw = self._parse_times(self._call(
            self._time_system,
            self._build_time_user_message(clinical_text, features),
            config.TIME_MODEL))
        self._write_cache(path, "raw_times", raw, config.TIME_MODEL, clinical_text)
        return raw

    def _schema_group_stage(self, clinical_text: str, feature: dict, group: str,
                            menu: tuple, use_cache: bool) -> dict:
        """Stage 3, one feature + one group: return raw ``{"variables": [...]}`` (cached)."""
        if group == schema.TIMING:
            system, model = self._timing_schema_system, config.TIMING_SCHEMA_MODEL
        else:
            system, model = self._details_schema_system, config.DETAILS_SCHEMA_MODEL
        path = self._schema_cache_path(clinical_text, feature, group, system)
        if use_cache:
            cached = self._read_cache(path, "raw_schema")
            if isinstance(cached, dict):
                return cached
        raw = self._parse_schema(self._call(
            system,
            self._build_schema_user_message(clinical_text, feature, group, menu),
            model))
        self._write_cache(path, "raw_schema", raw, model, clinical_text)
        return raw

    def _treatments_stage(self, clinical_text: str, pos_features: list,
                          use_cache: bool) -> list:
        """Stage 4: return the raw ``treatments`` list (cached), unvalidated."""
        path = self._treatments_cache_path(clinical_text, pos_features)
        if use_cache:
            cached = self._read_cache(path, "raw_treatments")
            if isinstance(cached, list):
                return cached
        raw = self._parse_treatments(self._call(
            self._build_treatments_system_prompt(pos_features),
            self._build_treatments_user_message(clinical_text),
            config.TREATMENTS_MODEL))
        self._write_cache(path, "raw_treatments", raw,
                          config.TREATMENTS_MODEL, clinical_text)
        return raw

    def _pmh_stage(self, clinical_text: str, use_cache: bool) -> list:
        """Model 5: return the raw ``past_medical_history`` list (cached), unvalidated."""
        path = self._pmh_cache_path(clinical_text)
        if use_cache:
            cached = self._read_cache(path, "raw_pmh")
            if isinstance(cached, list):
                return cached
        raw = self._parse_pmh(self._call(
            self._pmh_system,
            self._build_pmh_user_message(clinical_text),
            config.PMH_MODEL))
        self._write_cache(path, "raw_pmh", raw, config.PMH_MODEL, clinical_text)
        return raw

    # -- public -------------------------------------------------------------- #
    def extract(self, clinical_text: str, use_cache: bool = True) -> dict:
        """Four-stage extraction for one clinical text.

        Returns ``{"features": [...], "dropped": [...], "treatments": [...],
        "absences_deduped": <int>, "absences_removed": [...], "schema_dropped":
        [...]}`` where ``features`` come from Stage 1 (vocabulary-validated when
        ``config.VALIDATE_FEATURES`` is on, otherwise kept verbatim; then, when
        ``config.DEDUPE_ABSENCE_FEATURES`` is on, redundant same-evidence absences are
        collapsed) with contextual ``time`` attached (Stage 2), and — for every
        feature — a nested ``schema`` object of NHSN Timing/Details variables filled in
        (Stage 3): POSITIVE features get their full Timing and Details, while NEGATED
        features get the same full Timing call trimmed to its first supported variable
        plus a deterministically stamped Details (no Details call); ``treatments`` is
        Model 4's raw, unvalidated treatments list (Stage
        4), fed the note plus ALL the positive features found above. ``dropped`` lists
        Stage-1 off-list items when validation is on (empty otherwise);
        ``absences_deduped`` counts redundant absences removed and ``absences_removed``
        lists those removed items for logging; ``schema_dropped`` lists every
        ``variable_id`` Model 3 returned that was not in a feature's menu (dropped, for
        logging) — the row is never dropped.
        """
        _progress("  stage 1/5 features: extracting")
        raw_features = self._features_stage(clinical_text, use_cache)
        _progress(f"  stage 1/5 features: {len(raw_features)} raw")
        # The ONLY validation in the pipeline: the Model 1 vocabulary post-check,
        # toggled by config.VALIDATE_FEATURES.
        if config.VALIDATE_FEATURES:
            processed = validate_features(raw_features, self.vocab)
        else:
            processed = passthrough_features(raw_features)
        feats = processed["features"]
        # Deterministic redundancy trim (before Model 2), toggled by config flag:
        # collapse negated features that repeat one evidence phrase. Positives, and
        # thus Model 4's treatments input, are never affected. We also capture WHICH items were
        # removed (by identity — dedupe keeps the same objects) so the caller can log
        # them, not just count them.
        absences_removed: list = []
        if config.DEDUPE_ABSENCE_FEATURES:
            kept = dedupe_absence_features(feats)
            kept_ids = {id(f) for f in kept}
            absences_removed = [f for f in feats if id(f) not in kept_ids]
            feats = kept
        _progress(f"  stage 2/5 time: {len(feats)} features")
        time_data = self._time_stage(clinical_text, feats, use_cache)
        merged = merge_time(feats, time_data)
        # Stage 3 (Model 3, NHSN schema fill-in): fill each feature's Timing and
        # Details variables. The result is attached in place as a nested `schema`
        # object; only variables in that feature's menu are kept, and any unknown
        # variable_id the model returned is captured in `schema_dropped` for logging
        # (never emitted). POSITIVE features are filled in full (a Timing call and a
        # Details call). NEGATED features go through the SAME full Timing call (so the
        # model behaves identically — no new hallucination surface), after which we
        # deterministically keep only the first supported timing variable and drop the
        # rest; their single Details variable is stamped deterministically with no LLM
        # call. Mutates the `merged` dicts, which are copies made by merge_time, so
        # Stage-1/2 state is untouched.
        schema_dropped: list = []
        _progress(f"  stage 3/5 schema: filling {len(merged)} features")
        for _fi, feat in enumerate(merged, 1):
            if not (isinstance(feat, dict) and isinstance(feat.get("feature"), str)):
                continue
            _progress(f"  stage 3/5 schema: feature {_fi}/{len(merged)}")
            is_negated = feat["feature"].strip().lower().startswith(
                NEGATED_FEATURE_PREFIX)
            groups_out: dict = {}
            for group in schema.GROUPS:
                menu = self.feature_schema.variables(feat["feature"], group)
                if not menu:
                    continue
                if is_negated and group == schema.DETAILS:
                    # Deterministic Details for a negated feature: no LLM call. The
                    # single catch-all variable's definition asks for the verbatim
                    # evidence that the feature was absent -- which Model 1 already
                    # extracted as this feature's `evidence` phrase. Stamp that text
                    # (matching the menu's "text value only" shape) and let
                    # merge_schema_group attach the variable's schema notes/dependencies.
                    # If there is no evidence phrase, emit nothing (lean, no placeholder).
                    evidence = feat.get("evidence")
                    entries = []
                    if isinstance(evidence, str) and evidence.strip():
                        entries, _ = merge_schema_group(menu, {"variables": [
                            {"variable_id": menu[0].variable_id,
                             "value": evidence.strip()}]})
                    groups_out[SCHEMA_GROUP_OUTPUT_KEY[group]] = entries
                    continue
                raw = self._schema_group_stage(
                    clinical_text, feat, group, menu, use_cache)
                entries, unknown_ids = merge_schema_group(menu, raw)
                if is_negated and group == schema.TIMING:
                    # A negated feature carries just one timing variable as a symmetry
                    # gesture: the full menu was still sent to the model (identical to
                    # a positive), and we keep only the first supported variable.
                    entries = entries[:1]
                groups_out[SCHEMA_GROUP_OUTPUT_KEY[group]] = entries
                for vid in unknown_ids:
                    schema_dropped.append({"feature": feat["feature"],
                                           "group": group, "variable_id": vid})
            feat[SCHEMA_KEY] = groups_out
        # Stage 4 (Model 4, treatments): ALL positive features are forwarded (there is
        # no longer a relatedness gate). The nested `schema` object is stripped so
        # Model 4's prompt shape (name/evidence/time) is exactly what it was before
        # Stage 3 existed; the schema stays in `merged` (Column B).
        pos = [{k: v for k, v in f.items() if k != SCHEMA_KEY}
               for f in positive_features(merged)]
        _progress(f"  stage 4/5 treatments: {len(pos)} positive features")
        treatments = self._treatments_stage(clinical_text, pos, use_cache)
        return {"features": merged, "dropped": processed["dropped"],
                "treatments": treatments,
                "absences_deduped": len(absences_removed),
                "absences_removed": absences_removed,
                "schema_dropped": schema_dropped}

    def extract_past_medical_history(self, clinical_text: str,
                                     use_cache: bool = True) -> list:
        """Model 5 (INDEPENDENT): the patient's past medical history from the note.

        This is a standalone, additive stage: it reads only the clinical note, uses
        no vocabulary, and neither consumes nor is consumed by ``extract()``'s four
        stages — so running it changes nothing about feature/time/schema/
        treatment extraction. Returns the model's raw, UNVALIDATED list of
        ``{condition, detail}`` objects (stored exactly as returned, like the
        treatments stage). Cached per note under ``output/cache/``.
        """
        _progress("  stage 5/5 past medical history")
        return self._pmh_stage(clinical_text, use_cache)
