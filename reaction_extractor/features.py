"""Column 3 (Features) — the only AI step. Groq, text-only.

For each reaction we send its Column-2 Case-Definition text to the model and get
back a structured object with exactly three fields extracted from that text:
``time_window`` (str | None), ``signs_symptoms`` (list[str] | None), and
``lab_markers`` (list[str] | None). The call mirrors the parser's Groq setup:
``temperature=0``, ``reasoning_effort="none"``, JSON mode, a client with
``max_retries=0`` so retry policy is ours, exponential backoff + jitter on
429/transient errors, and a truncation detector (``finish_reason == "length"``
retries once at a higher budget, then raises — never silently accepted).

Each reaction's result is cached to ``output/cache/<slug>.json`` so re-runs cost
no tokens; ``use_cache=False`` bypasses it. Cache files are stamped with
``config.FEATURES_SCHEMA_VERSION`` and a hit is honored only when it matches, so
a stale cache from an older output shape can never silently be reused.
"""

from __future__ import annotations

import json
import random
import re
import time
from pathlib import Path
from typing import Optional

from . import config


class FeatureError(RuntimeError):
    """A features call failed in a way that must not be silently accepted."""


class TruncationError(FeatureError):
    """The model output was truncated (finish_reason == 'length') even after retry."""


def _transient_exceptions() -> tuple[type[BaseException], ...]:
    """Groq error types worth retrying (rate limits / transient server errors)."""
    try:
        import groq
    except Exception:  # pragma: no cover - groq always present in practice
        return ()
    types: list[type[BaseException]] = []
    for name in ("RateLimitError", "APITimeoutError", "APIConnectionError",
                 "InternalServerError"):
        t = getattr(groq, name, None)
        if isinstance(t, type):
            types.append(t)
    return tuple(types)


def _is_json_validate_failed(exc: BaseException) -> bool:
    """True for Groq's 400 json_validate_failed (reasoning model breaking JSON mode)."""
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        return body.get("error", {}).get("code") == "json_validate_failed"
    return False


def _slug(reaction: str) -> str:
    """Filesystem-safe cache key from a reaction name."""
    s = re.sub(r"[^a-zA-Z0-9]+", "_", reaction).strip("_").lower()
    return s or "reaction"


def _strip_code_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z0-9]*\s*", "", t)
        t = re.sub(r"\s*```$", "", t)
    return t.strip()


def _load_prompt_template() -> str:
    return (config.PROMPTS_DIR / "features.md").read_text(encoding="utf-8")


# The three fields Column 3 extracts, in output order.
FEATURE_KEYS = ("time_window", "signs_symptoms", "lab_markers")
_LIST_FEATURE_KEYS = ("signs_symptoms", "lab_markers")


def empty_features() -> dict:
    """A well-formed features object with every field null."""
    return {k: None for k in FEATURE_KEYS}


# --------------------------------------------------------------------------- #
# "/" term-splitting safety net (Layer 2; the prompt is the primary fix)
# --------------------------------------------------------------------------- #
def _slash_inside_parens(item: str, slash_index: int) -> bool:
    """True if the "/" at ``slash_index`` sits inside an unclosed "(...)"."""
    before = item[:slash_index]
    return before.count("(") > before.count(")")


def _should_split_slash(item: str) -> bool:
    """Decide whether a single "/" joins two independent clinical terms.

    Splits only when every DO-NOT-SPLIT guard passes (see config): exactly one
    "/", no protected substring (``and/or``/``c/w``), no digit on either side
    (ratios, unit equivalences), neither side shorter than SPLIT_MIN_TERM_LEN
    (abbreviations like ``c/w``), the "/" not inside parentheses, and both sides
    purely alphabetic. Anything else is left intact.
    """
    if item.count("/") != 1:
        return False
    low = item.lower()
    if any(sub in low for sub in config.SPLIT_PROTECTED_SUBSTRINGS):
        return False
    if _slash_inside_parens(item, item.index("/")):
        return False
    # The whitespace-delimited token that carries the "/"; split it in two.
    token = next(t for t in item.split() if "/" in t)
    left, right = token.split("/", 1)
    if len(left) < config.SPLIT_MIN_TERM_LEN or len(right) < config.SPLIT_MIN_TERM_LEN:
        return False
    if any(ch.isdigit() for ch in token):
        return False
    return left.isalpha() and right.isalpha()


def _cap_first(text: str) -> str:
    return text[:1].upper() + text[1:] if text else text


def split_slash_terms(item: str) -> list[str]:
    """Split ``"Back/flank pain"`` -> ``["Back pain", "Flank pain"]`` when the "/"
    joins two independent terms; return ``[item]`` unchanged otherwise.

    Trailing tokens after the slash token are shared context appended to both
    parts (``"Back/flank pain"`` -> both keep ``"pain"``), while a slash token
    with nothing after it splits cleanly (``"Chills/rigors"`` -> ``"Chills"`` and
    ``"Rigors"``). Each resulting part has its first letter capitalized.
    """
    if not _should_split_slash(item):
        return [item]
    tokens = item.split()
    idx = next(i for i, t in enumerate(tokens) if "/" in t)
    leading, trailing = tokens[:idx], tokens[idx + 1:]
    left, right = tokens[idx].split("/", 1)
    parts = []
    for term in (left, right):
        parts.append(_cap_first(" ".join(leading + [term] + trailing)))
    return parts


def _has_unclosed_paren(text: str) -> bool:
    """True if ``text`` has more '(' than ')' — a parenthetical left open."""
    return text.count("(") > text.count(")")


def _rejoin_parenthetical_fragments(items: list[str]) -> list[str]:
    """Fold an orphaned clarification back into the item it clarifies.

    A parenthetical or semicolon clarification is one finding, not two. If the
    model emitted the clarifier as its own item — one that starts with '(' or
    ';', or one that closes a '(' the previous item left open — merge it into
    the preceding item rather than keeping a dangling fragment. Items with no
    such signal are left untouched (two clean, unpunctuated terms cannot be
    rejoined here; the prompt is the primary guard for that).
    """
    out: list[str] = []
    for item in items:
        stripped = item.strip()
        if out and (stripped.startswith("(") or stripped.startswith(";")
                    or _has_unclosed_paren(out[-1])):
            sep = "" if stripped[:1] in ";," else " "
            out[-1] = out[-1].rstrip() + sep + stripped
        else:
            out.append(item)
    return out


def normalize_feature_list(items: Optional[list]) -> Optional[list]:
    """Normalize one list field: split independent "/" terms, then rejoin any
    orphaned parenthetical/semicolon clarifications. Order is preserved."""
    if not items:
        return items
    expanded: list[str] = []
    for item in items:
        expanded.extend(split_slash_terms(item))
    return _rejoin_parenthetical_fragments(expanded)


def _normalize_features(features: dict) -> dict:
    """Normalize the two list fields in a parsed features object in place."""
    for key in _LIST_FEATURE_KEYS:
        features[key] = normalize_feature_list(features.get(key))
    return features


def _split_slash_in_list(items: Optional[list]) -> Optional[list]:
    """Apply ``split_slash_terms`` across a list field — split logic only.

    Unlike ``normalize_feature_list`` this does NOT run the parenthetical-rejoin
    pass, so it can never merge across an item that left a "(" unclosed (e.g.
    FNHTR's "Fever (…" line). Order is preserved.
    """
    if not items:
        return items
    out: list[str] = []
    for item in items:
        out.extend(split_slash_terms(item))
    return out


def normalize_output_file(path: Optional[Path] = None) -> dict:
    """Split "/"-joined clinical terms in an existing ``reaction_features.json``.

    Reads the file, applies ``split_slash_terms`` to each reaction's
    ``signs_symptoms`` and ``lab_markers`` only, and overwrites the file in
    place. Deterministic — no Groq call, no network. ``time_window``,
    ``reaction``, ``definition``, and ``source_page`` are never read or written.
    Returns the updated document.
    """
    path = Path(path) if path is not None else \
        config.OUTPUT_DIR / config.OUTPUT_FILENAME_DEFAULT
    document = json.loads(path.read_text(encoding="utf-8"))
    for reaction in document.get("reactions", []):
        feats = reaction.get("features")
        if not isinstance(feats, dict):
            continue
        for key in _LIST_FEATURE_KEYS:
            if key in feats:
                feats[key] = _split_slash_in_list(feats[key])
    path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
    return document


# --------------------------------------------------------------------------- #
# Compound-term splitting for signs_symptoms (generalizable post-processing)
# --------------------------------------------------------------------------- #
# Two structural patterns are split into separate list items; a shared guard
# protects thresholds, connectors and long descriptive phrases. No hardcoded
# strings — the rules generalize to the same shapes in future PDFs.


def _is_split_protected(item: str) -> bool:
    """True if a signs_symptoms item must NEVER be split, regardless of pattern.

    Protects thresholds/measurements (any digit, ``°``, ``%``) and the logical
    connector ``and/or``. Long descriptive phrases are excluded separately by the
    per-pattern word-length ceilings; this is the shared first-line guard both
    pattern functions call before attempting any split.
    """
    if "and/or" in item.lower():
        return True
    if any(ch.isdigit() for ch in item):
        return True
    return any(sym in item for sym in ("°", "%"))


def _split_shared_prefix_list(item: str) -> Optional[list[str]]:
    """Pattern 1 — ``"<prefix> a, b and c"`` -> one item per term with the prefix.

    Requires a comma and exactly one ``and``. The shared prefix is every word
    before the first term (the last word ahead of the first comma); the trailing
    list is split on commas and the ``and``. Returns None (no match) if any term
    exceeds ``SPLIT_PATTERN1_MAX_TERM_WORDS`` words or there is no real prefix.
    """
    if _is_split_protected(item) or "," not in item:
        return None
    if len(re.findall(r"\band\b", item)) != 1:
        return None
    head, tail = item.split(",", 1)
    head_words = head.split()
    if len(head_words) < 2:                       # need "<prefix> <term1>"
        return None
    prefix = " ".join(head_words[:-1])
    terms = [head_words[-1]] + [t.strip() for t in re.split(r",|\band\b", tail) if t.strip()]
    if len(terms) < 2:
        return None
    if any(len(t.split()) > config.SPLIT_PATTERN1_MAX_TERM_WORDS for t in terms):
        return None
    return [f"{prefix} {t}" for t in terms]


def _split_shared_suffix_pair(item: str) -> Optional[list[str]]:
    """Pattern 2 — ``"a and b <prep-phrase>"`` -> two items sharing the suffix.

    Requires exactly one ``and`` and no comma. The two leading nouns (each up to
    ``SPLIT_PATTERN2_MAX_LEADING_WORDS`` words) precede a prepositional suffix
    starting with one of ``SPLIT_PATTERN2_PREPOSITIONS``; the suffix attaches to
    both. Each result is capitalized. Returns None if the shape does not match.
    """
    if _is_split_protected(item) or "," in item:
        return None
    if len(re.findall(r"\band\b", item)) != 1:
        return None
    left, right = re.split(r"\band\b", item, maxsplit=1)
    term1 = left.strip()
    words = right.split()
    if not term1 or len(term1.split()) > config.SPLIT_PATTERN2_MAX_LEADING_WORDS:
        return None
    prep_idx = next((i for i, w in enumerate(words)
                     if w.lower() in config.SPLIT_PATTERN2_PREPOSITIONS), None)
    if not prep_idx:                              # None, or a preposition with no term2 before it
        return None
    term2 = " ".join(words[:prep_idx])
    suffix = " ".join(words[prep_idx:])
    if len(term2.split()) > config.SPLIT_PATTERN2_MAX_LEADING_WORDS or not suffix:
        return None
    return [_cap_first(f"{term1} {suffix}"), _cap_first(f"{term2} {suffix}")]


def _split_symptom_item(item: str) -> list[str]:
    """Apply Pattern 1 then Pattern 2 to one item; unchanged if neither matches."""
    return (_split_shared_prefix_list(item)
            or _split_shared_suffix_pair(item)
            or [item])


def _split_signs_symptoms_list(items: Optional[list]) -> Optional[list]:
    """Expand every item in a signs_symptoms list through both patterns, in order."""
    if not items:
        return items
    out: list[str] = []
    for item in items:
        out.extend(_split_symptom_item(item))
    return out


def split_signs_symptoms(path: Optional[Path] = None) -> dict:
    """Split compound signs_symptoms terms across an existing
    ``reaction_features.json`` and overwrite it in place.

    Deterministic — no Groq call, no network. Applies Pattern 1 / Pattern 2 to
    every reaction's ``signs_symptoms`` only; ``lab_markers``, ``time_window``,
    and all other fields are never read for writing. Returns the updated document.
    """
    path = Path(path) if path is not None else \
        config.OUTPUT_DIR / config.OUTPUT_FILENAME_DEFAULT
    document = json.loads(path.read_text(encoding="utf-8"))
    for reaction in document.get("reactions", []):
        feats = reaction.get("features")
        if isinstance(feats, dict) and "signs_symptoms" in feats:
            feats["signs_symptoms"] = _split_signs_symptoms_list(feats["signs_symptoms"])
    path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
    return document


class FeatureExtractor:
    """Derives the features list for one reaction at a time, with caching."""

    def __init__(self, client=None, cache_dir: Optional[Path] = None):
        self._client = client
        self.cache_dir = Path(cache_dir) if cache_dir else config.CACHE_DIR
        self._template = _load_prompt_template()

    # -- client -------------------------------------------------------------- #
    @property
    def client(self):
        if self._client is None:
            import groq
            self._client = groq.Groq(api_key=config.get_groq_api_key(), max_retries=0)
        return self._client

    # -- prompt -------------------------------------------------------------- #
    def _build_prompt(self, reaction: str, definition: str) -> str:
        return self._template.replace("{reaction}", reaction).replace(
            "{definition}", definition)

    # -- one raw call -------------------------------------------------------- #
    def _call_once(self, prompt: str, max_tokens: int):
        kwargs = dict(
            model=config.VISION_MODEL,
            messages=[
                {"role": "system",
                 "content": "You output only valid JSON matching the requested shape."},
                {"role": "user", "content": prompt},
            ],
            temperature=config.TEMPERATURE,
            max_completion_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
        if config.REASONING_EFFORT is not None:
            kwargs["reasoning_effort"] = config.REASONING_EFFORT
        return self.client.chat.completions.create(**kwargs)

    # -- retry wrapper ------------------------------------------------------- #
    def _sleep_backoff(self, attempt: int) -> None:
        delay = min(config.BACKOFF_INITIAL * (config.BACKOFF_FACTOR ** attempt),
                    config.BACKOFF_MAX)
        jitter = delay * config.BACKOFF_JITTER
        time.sleep(max(0.0, delay + random.uniform(-jitter, jitter)))

    def _call_with_retries(self, prompt: str, max_tokens: int):
        """Return a completion, retrying transient errors and json_validate_failed."""
        transient = _transient_exceptions()
        last_exc: Optional[BaseException] = None
        for attempt in range(config.MAX_RETRIES):
            try:
                return self._call_once(prompt, max_tokens)
            except transient as exc:  # type: ignore[misc]
                last_exc = exc
            except Exception as exc:
                if _is_json_validate_failed(exc):
                    # Treat like an incomplete response: retry, then surface.
                    last_exc = exc
                else:
                    raise
            if attempt < config.MAX_RETRIES - 1:
                self._sleep_backoff(attempt)
        raise FeatureError(
            f"features call failed after {config.MAX_RETRIES} attempts: {last_exc}"
        )

    # -- truncation-aware call ---------------------------------------------- #
    def _call(self, prompt: str) -> str:
        """Call once at the normal budget; on a `length` finish, retry once higher."""
        resp = self._call_with_retries(prompt, config.MAX_COMPLETION_TOKENS)
        if self._finish_reason(resp) == "length":
            resp = self._call_with_retries(prompt, config.MAX_COMPLETION_TOKENS_RETRY)
            if self._finish_reason(resp) == "length":
                raise TruncationError(
                    "features output truncated (finish_reason='length') even at the "
                    "elevated token budget; refusing to accept a partial list."
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
    def _parse_features(content: str) -> dict:
        """Normalize the model's JSON into the fixed 3-key features object.

        ``time_window`` -> trimmed str or None; ``signs_symptoms`` / ``lab_markers``
        -> cleaned list[str] or None (empty -> None). Raises FeatureError only if
        the response is not a JSON object or contains none of the three keys.
        """
        try:
            data = json.loads(_strip_code_fences(content))
        except json.JSONDecodeError as exc:
            raise FeatureError(f"features response was not valid JSON: {exc}") from exc
        if not isinstance(data, dict) or not any(k in data for k in FEATURE_KEYS):
            raise FeatureError(
                f"features response has none of {FEATURE_KEYS}: {content[:200]!r}")

        result = empty_features()
        tw = data.get("time_window")
        if isinstance(tw, str) and tw.strip():
            result["time_window"] = tw.strip()
        for key in _LIST_FEATURE_KEYS:
            value = data.get(key)
            if isinstance(value, list):
                items = [str(x).strip() for x in value if str(x).strip()]
                result[key] = items or None
        return result

    # -- cache --------------------------------------------------------------- #
    def _cache_path(self, reaction: str) -> Path:
        return self.cache_dir / f"{_slug(reaction)}.json"

    def _read_cache(self, reaction: str) -> Optional[dict]:
        path = self._cache_path(reaction)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
        # Honor only current-version, correctly-shaped entries; anything older
        # (e.g. the retired flat list) is ignored and will be recomputed.
        if not isinstance(data, dict):
            return None
        if data.get("schema_version") != config.FEATURES_SCHEMA_VERSION:
            return None
        feats = data.get("features")
        if not isinstance(feats, dict) or set(feats) != set(FEATURE_KEYS):
            return None
        return feats

    def _write_cache(self, reaction: str, features: dict) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache_path(reaction).write_text(
            json.dumps({"schema_version": config.FEATURES_SCHEMA_VERSION,
                        "reaction": reaction, "features": features},
                       ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # -- public -------------------------------------------------------------- #
    def features_for(self, reaction: str, definition: str,
                     use_cache: bool = True) -> dict:
        """Return the features object for one reaction (cache-first unless disabled)."""
        if use_cache:
            cached = self._read_cache(reaction)
            if cached is not None:
                return cached
        content = self._call(self._build_prompt(reaction, definition))
        features = _normalize_features(self._parse_features(content))
        self._write_cache(reaction, features)
        return features
