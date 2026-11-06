"""Double-negative deduplication of the extended symptoms/lab-markers list (Groq).

Reads the already-built ``extended_symptoms_labmarkers.json`` and asks the model
to remove features that are mechanically-generated *double negatives* — an
``Evidence for absence of ...`` wrapper placed around a statement that is itself
already negative (e.g. ``Evidence for absence of absence of clinical signs of
hemolysis``). The meaningful original is kept; only the double-negative is removed.
Two SEPARATE, self-contained files are written: the clean list to
``deduplicated_symptoms_labmarkers.json`` and the removed-items audit trail to
``dedup_removed_items.json``.

The model returns the RETAINED features as a plain JSON list (see
``prompts/dedup.md``); there are NO hardcoded feature strings. The result is
reconciled deterministically in code (``reconcile_retained``): every retained item
must appear VERBATIM in the input (otherwise we raise rather than write a corrupted
list), and the surviving list is rebuilt from the original in original order — so
the model can never reorder, rename, or invent a feature.

The Groq call uses the exact params validated in ``dedup_sandbox/``:
``temperature=0.0``, ``top_p=0.95``, ``reasoning_effort="default"``,
``max_completion_tokens=26271``, streaming, and NO strict JSON mode (the prompt
alone dictates the shape). These are scoped to THIS call so the shared
features-extraction step (which uses ``config.TEMPERATURE`` / ``REASONING_EFFORT``)
is unaffected. A client with ``max_retries=0`` keeps retry policy ours (exponential
backoff + jitter on transient/``json_validate_failed``); a truncation detector
raises on ``finish_reason == "length"``. The raw response is cached keyed by a hash
of the prompt + input list; ``use_cache=False`` bypasses the read.

This module never modifies ``extended_symptoms_labmarkers.json``; it only reads it.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from . import config
# Reuse the features step's shared call machinery — same retry/backoff/truncation
# semantics as the rest of the project, without duplicating the tricky bits.
from .features import (
    FeatureError,
    TruncationError,
    _is_json_validate_failed,
    _strip_code_fences,
    _transient_exceptions,
)

# Bumped whenever the dedup prompt or output shape changes; cache entries stamped
# with this are honored only on a match, so a stale cache can never be reused.
#   1 = paraphrase-removal prompt, {"removals":[...]} output (retired)
#   2 = double-negative prompt, retained-list output + streaming params
DEDUP_SCHEMA_VERSION = 2

# Dedup-specific Groq params — SCOPED HERE (not config.TEMPERATURE / REASONING_EFFORT)
# so the shared features-extraction call is unaffected. These are the exact values
# validated in dedup_sandbox/.
DEDUP_MODEL = config.VISION_MODEL          # "qwen/qwen3.6-27b"
DEDUP_TEMPERATURE = 0.0
DEDUP_TOP_P = 0.95
DEDUP_REASONING_EFFORT = "default"
DEDUP_MAX_COMPLETION_TOKENS = 26271
assert DEDUP_MAX_COMPLETION_TOKENS < config.MODEL_OTPM_LIMIT, (
    "DEDUP_MAX_COMPLETION_TOKENS must stay below the model OTPM limit.")

INPUT_KEY = "extended_symptoms_labmarkers"
OUTPUT_KEY = "deduplicated_symptoms_labmarkers"
REMOVED_KEY = "removed_items"
DEDUP_FILENAME = "deduplicated_symptoms_labmarkers.json"
REMOVED_FILENAME = "dedup_removed_items.json"
SOURCE_FILENAME = "extended_symptoms_labmarkers.json"
CACHE_FILENAME = "dedup_features.json"


def _load_prompt_template() -> str:
    return (config.PROMPTS_DIR / "dedup.md").read_text(encoding="utf-8")


# --------------------------------------------------------------------------- #
# Deterministic reconciliation (pure — no I/O, no model)
# --------------------------------------------------------------------------- #
def reconcile_retained(items: list[str], retained: list) -> dict:
    """Turn the model's RETAINED list into a deduplicated result, safely.

    The model returns the features to KEEP. We reconcile deterministically:
      * ``retained`` must be a list of strings, and every one must appear VERBATIM
        in ``items``; if any does not (the model altered or invented wording), we
        RAISE rather than write a corrupted list;
      * the surviving list is rebuilt from ``items`` in ORIGINAL order, keeping
        exactly the retained items — so the model can never reorder or rename; and
      * ``removed`` = ``items`` − ``retained``.

    Returns the shape the pipeline expects: ``deduplicated`` list, ``removed_items``
    audit trail, ``removed_indices`` (0-based), and ``skipped`` (always empty here —
    anomalies raise instead of being silently skipped).
    """
    if not isinstance(retained, list) or not all(isinstance(x, str) for x in retained):
        raise FeatureError("dedup response must be a JSON list of strings")
    original_set = set(items)
    unknown = [x for x in retained if x not in original_set]
    if unknown:
        raise FeatureError(
            f"dedup returned {len(unknown)} item(s) not present verbatim in the input "
            f"(refusing to write a corrupted list); first: {unknown[0]!r}")
    retained_set = set(retained)
    deduplicated = [x for x in items if x in retained_set]
    removed_positions = [i for i, x in enumerate(items) if x not in retained_set]
    removed_items = [{"removed_value": items[i],
                      "reason": "mechanically-generated double negative"}
                     for i in removed_positions]
    return {
        "deduplicated": deduplicated,
        "removed_items": removed_items,
        "removed_indices": removed_positions,
        "skipped": [],
    }


class FeatureDeduplicator:
    """Runs one VLM dedup pass over the full list, with disk caching."""

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
    def _build_prompt(self, items: list[str]) -> str:
        # The prompt has no placeholder; append the feature list as a JSON array
        # (exactly as validated in dedup_sandbox/).
        return f"{self._template}\n\n{json.dumps(items, ensure_ascii=False, indent=2)}"

    # -- one raw (streaming) call ------------------------------------------- #
    def _call_once(self, prompt: str, max_tokens: int):
        """Create a streaming completion; return ``(content, finish_reason)``."""
        stream = self.client.chat.completions.create(
            model=DEDUP_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=DEDUP_TEMPERATURE,
            max_completion_tokens=max_tokens,
            top_p=DEDUP_TOP_P,
            reasoning_effort=DEDUP_REASONING_EFFORT,
            stream=True,
            stop=None,
        )
        parts: list[str] = []
        finish: Optional[str] = None
        for chunk in stream:
            if not getattr(chunk, "choices", None):
                continue
            ch = chunk.choices[0]
            c = getattr(ch.delta, "content", None)
            if c:
                parts.append(c)
            if ch.finish_reason:
                finish = ch.finish_reason
        return "".join(parts), finish

    # -- retry wrapper ------------------------------------------------------- #
    def _sleep_backoff(self, attempt: int) -> None:
        delay = min(config.BACKOFF_INITIAL * (config.BACKOFF_FACTOR ** attempt),
                    config.BACKOFF_MAX)
        jitter = delay * config.BACKOFF_JITTER
        time.sleep(max(0.0, delay + random.uniform(-jitter, jitter)))

    def _call_with_retries(self, prompt: str, max_tokens: int):
        transient = _transient_exceptions()
        last_exc: Optional[BaseException] = None
        for attempt in range(config.MAX_RETRIES):
            try:
                return self._call_once(prompt, max_tokens)   # (content, finish)
            except transient as exc:  # type: ignore[misc]
                last_exc = exc
            except Exception as exc:
                if _is_json_validate_failed(exc):
                    last_exc = exc
                else:
                    raise
            if attempt < config.MAX_RETRIES - 1:
                self._sleep_backoff(attempt)
        raise FeatureError(
            f"dedup call failed after {config.MAX_RETRIES} attempts: {last_exc}")

    # -- truncation-aware call ---------------------------------------------- #
    def _call(self, prompt: str) -> str:
        content, finish = self._call_with_retries(prompt, DEDUP_MAX_COMPLETION_TOKENS)
        if finish == "length":
            raise TruncationError(
                "dedup output truncated (finish_reason='length') at "
                f"{DEDUP_MAX_COMPLETION_TOKENS} tokens; refusing a partial list.")
        return content

    # -- parse --------------------------------------------------------------- #
    @staticmethod
    def _parse_retained(content: str) -> list:
        """Parse the retained-features JSON list; raise (never silently drop) if malformed."""
        text = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
        text = _strip_code_fences(text)
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            m = re.search(r"\[.*\]", text, flags=re.DOTALL)   # first array in the text
            if not m:
                raise FeatureError(f"dedup response was not valid JSON: {exc}") from exc
            try:
                data = json.loads(m.group(0))
            except json.JSONDecodeError as exc2:
                raise FeatureError(f"dedup response was not valid JSON: {exc2}") from exc2
        if not isinstance(data, list):
            raise FeatureError(
                f"dedup response must be a JSON list, got {type(data).__name__}: "
                f"{content[:200]!r}")
        return data

    # -- cache --------------------------------------------------------------- #
    def _cache_key(self, items: list[str]) -> str:
        h = hashlib.sha256()
        h.update(self._template.encode("utf-8"))
        h.update(b"\x00")
        h.update("\x00".join(items).encode("utf-8"))
        return h.hexdigest()

    def _cache_path(self) -> Path:
        return self.cache_dir / CACHE_FILENAME

    def _read_cache(self, items: list[str]) -> Optional[str]:
        path = self._cache_path()
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
        if not isinstance(data, dict):
            return None
        if data.get("schema_version") != DEDUP_SCHEMA_VERSION:
            return None
        if data.get("input_hash") != self._cache_key(items):
            return None
        raw = data.get("response")
        return raw if isinstance(raw, str) else None

    def _write_cache(self, items: list[str], response: str) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache_path().write_text(
            json.dumps({"schema_version": DEDUP_SCHEMA_VERSION,
                        "input_hash": self._cache_key(items),
                        "response": response},
                       ensure_ascii=False, indent=2),
            encoding="utf-8")

    # -- public -------------------------------------------------------------- #
    def deduplicate(self, items: list[str], use_cache: bool = True) -> dict:
        """Return the reconciled dedup result for ``items`` (cache-first)."""
        raw: Optional[str] = self._read_cache(items) if use_cache else None
        if raw is None:
            raw = self._call(self._build_prompt(items))
            self._write_cache(items, raw)
        return reconcile_retained(items, self._parse_retained(raw))


def build_deduplicated_symptoms_labmarkers(
    input_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
    removed_output_path: Optional[Path] = None,
    use_cache: bool = True,
    deduplicator: Optional[FeatureDeduplicator] = None,
) -> dict:
    """Read the extended file, run the dedup pass, write two parallel files.

    The clean deduplicated list and the removed-items audit trail are written to
    SEPARATE, self-contained files that carry no reference to each other. The
    input file is read-only. Returns a report dict for the CLI to render.
    """
    input_path = Path(input_path) if input_path is not None else \
        config.OUTPUT_DIR / SOURCE_FILENAME
    output_path = Path(output_path) if output_path is not None else \
        config.OUTPUT_DIR / DEDUP_FILENAME
    removed_output_path = Path(removed_output_path) \
        if removed_output_path is not None else config.OUTPUT_DIR / REMOVED_FILENAME

    source_document = json.loads(input_path.read_text(encoding="utf-8"))
    items = source_document.get(INPUT_KEY, [])

    dedup = deduplicator if deduplicator is not None else FeatureDeduplicator()
    result = dedup.deduplicate(items, use_cache=use_cache)

    deduplicated = result["deduplicated"]
    removed_items = result["removed_items"]
    original_count = len(items)
    removed_count = len(removed_items)
    final_count = len(deduplicated)
    # Hard invariant — the counts must be internally consistent.
    assert final_count == original_count - removed_count, (
        f"count mismatch: {final_count} != {original_count} - {removed_count}")

    generated_on = datetime.now(timezone.utc).isoformat()

    # File 1 — the clean deduplicated list (no audit trail).
    document = {
        "metadata": {
            "source_filename": input_path.name,
            "generated_on": generated_on,
            "original_count": original_count,
            "removed_count": removed_count,
            "final_count": final_count,
        },
        OUTPUT_KEY: deduplicated,
    }

    # File 2 — the removed-items audit trail (self-contained, no back-reference).
    removed_document = {
        "metadata": {
            "source_filename": input_path.name,
            "generated_on": generated_on,
            "removed_count": removed_count,
        },
        REMOVED_KEY: removed_items,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
    removed_output_path.parent.mkdir(parents=True, exist_ok=True)
    removed_output_path.write_text(
        json.dumps(removed_document, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "document": document,
        "removed_document": removed_document,
        "result": result,
        "output_path": output_path,
        "removed_output_path": removed_output_path,
    }


def render_report(report: dict) -> str:
    """Render the lightweight validation report as a printable string."""
    document = report["document"]
    meta = document["metadata"]
    skipped = report["result"]["skipped"]

    lines = [
        f"Wrote {meta['final_count']} deduplicated item(s) -> "
        f"{report['output_path']}",
        f"Wrote {meta['removed_count']} removed item(s) -> "
        f"{report['removed_output_path']}",
        f"  original {meta['original_count']} − removed {meta['removed_count']} "
        f"= final {meta['final_count']}",
    ]
    if meta["final_count"] != meta["original_count"] - meta["removed_count"]:
        lines.append("  ⚠ count invariant FAILED.")
    if skipped:
        lines.append(f"  ⚠ {len(skipped)} proposed removal(s) skipped "
                     "(kept for safety):")
        for s in skipped:
            lines.append(f"      - {s['why']}")
    if meta["removed_count"] == 0:
        lines.append("  no removals — list unchanged.")
    return "\n".join(lines)
