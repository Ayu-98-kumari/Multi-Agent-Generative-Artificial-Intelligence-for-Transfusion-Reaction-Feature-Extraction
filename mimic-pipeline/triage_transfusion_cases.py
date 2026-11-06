"""Triage the screened MIMIC-IV summaries into genuine transfusion-reaction cases.

The regex screener (``extract_transfusion_summaries.py``) is recall-oriented: a
match only means the note *mentions* a transfusion reaction. Many of those notes
mention it and then rule it out, or only carry the phrase in a history/checkbox
line. This step reads each screened note with an LLM and grades it:

  * ``genuine``      -- the note describes a transfusion reaction the patient
                        actually had (occurred / suspected-and-treated / worked up
                        as a real event).
  * ``ruled_out``    -- a reaction was considered and actively excluded, or the
                        work-up was negative / not concerning.
  * ``mention_only`` -- the phrase appears with no clinical event (past history,
                        a consent/checkbox line, a generic risk statement).

Every note gets a verdict PLUS a one-sentence reason and a verbatim supporting
quote, so a reject can be audited in seconds without re-reading the whole note.

Design constraints (deliberate):
  * The FULL note text is sent to the model, uncut, no matter how long it is --
    a reaction can be described at the very end of a discharge summary. The note
    is never truncated; if one somehow exceeded the model's context the script
    errors on that row rather than silently trimming.
  * Only the ~843 already-screened summaries are read (the screener's output CSV);
    the 1.1 GB source is never touched here.
  * This script is standalone. It imports nothing from ``pdf_parser``,
    ``reaction_extractor``, or ``clinical_feature_extractor`` and changes nothing
    in them. It only mirrors their proven Groq setup (client with
    ``max_retries=0`` + our own backoff, JSON mode, ``temperature=0``, explicit
    ``reasoning_effort``, a truncation detector).

Code lives in this tracked package; MIMIC data and everything derived from it
(the input CSV and this script's output) stay in the gitignored data directory.
Override the location with $MIMIC_DATA_DIR or --input/--output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

# DUA-covered data (input notes and this script's output) lives outside this
# tracked package.
DATA_DIR = Path(os.environ.get(
    "MIMIC_DATA_DIR", PROJECT_ROOT / "kept-gitignored" / "mimic-iv-note"))

# Load the project-root .env for GROQ_API_KEY (does nothing if absent; never
# overrides a real environment variable).
load_dotenv(PROJECT_ROOT / ".env")

DEFAULT_INPUT = "mimic_transfusion_cases.csv"
DEFAULT_OUTPUT = "mimic_transfusion_cases_triage_output.csv"
# The full audit (verdict + reason + quote + note metadata) is also written as
# JSONL next to the CSV, named after the output stem.
CACHE_DIRNAME = "triage_cache"

TEXT_COLUMN = "text"
ID_COLUMN = "note_id"

# --------------------------------------------------------------------------- #
# Groq / model
# --------------------------------------------------------------------------- #
GROQ_API_KEY_ENV = "GROQ_API_KEY"
MODEL = "openai/gpt-oss-120b"

# gpt-oss is a REASONING model: without an explicit effort it burns the output
# budget on hidden reasoning and can return empty content. "medium" gives it room
# to read a long note and judge, while the visible answer stays tiny.
REASONING_EFFORT = "medium"

TEMPERATURE = 0  # determinism

# Output budget. The visible answer is a few-line JSON object, but gpt-oss's
# hidden reasoning also draws from this budget, so it is generous. On a `length`
# finish the truncation detector retries once at the elevated budget.
MAX_COMPLETION_TOKENS = 8192
MAX_COMPLETION_TOKENS_RETRY = 16384

# Retry / backoff (transient errors + 429 rate limits).
MAX_RETRIES = 8
BACKOFF_INITIAL = 1.0
BACKOFF_FACTOR = 2.0
BACKOFF_MAX = 60.0
BACKOFF_JITTER = 0.25

VERDICTS = ("genuine", "ruled_out", "mention_only")

SYSTEM_PROMPT = """\
You are a hematology chart-review assistant. You read one hospital discharge \
summary and decide whether it describes a genuine blood-transfusion reaction.

A "transfusion reaction" is an adverse reaction to a blood-product transfusion \
(e.g. red cells, platelets, plasma, cryoprecipitate) -- for example febrile \
non-hemolytic reaction, acute or delayed hemolytic reaction, TRALI, TACO, \
allergic/anaphylactic reaction, hypotensive reaction, post-transfusion purpura, \
transfusion-associated dyspnea or GVHD, or a transfusion-transmitted infection.

Classify the note into exactly one verdict:

- "genuine": the patient actually had a transfusion reaction during this or a \
clearly described encounter -- it occurred, or was clinically suspected and \
treated/worked up as a real event (transfusion stopped, reaction protocol run, \
symptoms attributed to the transfusion).
- "ruled_out": a transfusion reaction was considered and actively excluded, or \
the work-up was negative / not concerning / unremarkable. The phrase describes \
something that was checked and found NOT to be a reaction.
- "mention_only": the phrase appears with no actual clinical event in this note \
-- e.g. remote past history, a consent or checkbox line, a generic risk or \
education statement, a differential listed but never pursued.

Read the ENTIRE note before deciding; a reaction may be described only near the \
end. Judge only from what the note says -- do not assume a reaction occurred just \
because blood was given.

Return ONLY a JSON object with exactly these keys:
{
  "verdict": one of "genuine", "ruled_out", "mention_only",
  "reason": a single concise sentence justifying the verdict,
  "supporting_quote": a short verbatim span copied from the note that best \
supports the verdict (copy it exactly; do not paraphrase)
}
"""


class TriageError(RuntimeError):
    """A triage call failed in a way that must not be silently accepted."""


class TruncationError(TriageError):
    """Model output was truncated (finish_reason == 'length') even after retry."""


# --------------------------------------------------------------------------- #
# Groq helpers (mirror the project's conventions; no import from the pipeline)
# --------------------------------------------------------------------------- #
def _transient_exceptions() -> tuple[type[BaseException], ...]:
    """Groq error types worth retrying (rate limits / transient server errors)."""
    try:
        import groq
    except Exception:  # pragma: no cover
        return ()
    types: list[type[BaseException]] = []
    for name in ("RateLimitError", "APITimeoutError", "APIConnectionError",
                 "InternalServerError"):
        t = getattr(groq, name, None)
        if isinstance(t, type):
            types.append(t)
    return tuple(types)


def _is_json_validate_failed(exc: BaseException) -> bool:
    """True for Groq's 400 json_validate_failed (worth one more attempt)."""
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        return body.get("error", {}).get("code") == "json_validate_failed"
    return False


def _strip_code_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z0-9]*\s*", "", t)
        t = re.sub(r"\s*```$", "", t)
    return t.strip()


def _text_hash(*parts: str) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


class Triager:
    """Grades one screened note at a time; caches each verdict for free re-runs."""

    def __init__(self, cache_dir: Path, client=None):
        self.cache_dir = cache_dir
        self._client = client

    # -- client -------------------------------------------------------------- #
    @property
    def client(self):
        if self._client is None:
            import groq
            key = os.environ.get(GROQ_API_KEY_ENV)
            if not key:
                raise TriageError(
                    f"{GROQ_API_KEY_ENV} not found. Add it to {PROJECT_ROOT / '.env'}."
                )
            self._client = groq.Groq(api_key=key, max_retries=0)
        return self._client

    # -- one raw call -------------------------------------------------------- #
    def _call_once(self, user_message: str, max_tokens: int):
        return self.client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=TEMPERATURE,
            max_completion_tokens=max_tokens,
            reasoning_effort=REASONING_EFFORT,
            response_format={"type": "json_object"},
        )

    def _sleep_backoff(self, attempt: int) -> None:
        delay = min(BACKOFF_INITIAL * (BACKOFF_FACTOR ** attempt), BACKOFF_MAX)
        jitter = delay * BACKOFF_JITTER
        time.sleep(max(0.0, delay + random.uniform(-jitter, jitter)))

    def _call_with_retries(self, user_message: str, max_tokens: int):
        transient = _transient_exceptions()
        last_exc: BaseException | None = None
        for attempt in range(MAX_RETRIES):
            try:
                return self._call_once(user_message, max_tokens)
            except transient as exc:  # type: ignore[misc]
                last_exc = exc
            except Exception as exc:
                if _is_json_validate_failed(exc):
                    last_exc = exc
                else:
                    raise
            if attempt < MAX_RETRIES - 1:
                self._sleep_backoff(attempt)
        raise TriageError(f"triage call failed after {MAX_RETRIES} attempts: {last_exc}")

    def _call(self, user_message: str) -> str:
        """Call at the normal budget; on a `length` finish retry once higher."""
        resp = self._call_with_retries(user_message, MAX_COMPLETION_TOKENS)
        if self._finish_reason(resp) == "length":
            resp = self._call_with_retries(user_message, MAX_COMPLETION_TOKENS_RETRY)
            if self._finish_reason(resp) == "length":
                raise TruncationError(
                    "triage output truncated (finish_reason='length') even at the "
                    "elevated budget; refusing to accept a partial verdict."
                )
        return self._content(resp)

    @staticmethod
    def _finish_reason(resp):
        try:
            return resp.choices[0].finish_reason
        except Exception:
            return None

    @staticmethod
    def _content(resp) -> str:
        return resp.choices[0].message.content or ""

    # -- parse --------------------------------------------------------------- #
    @staticmethod
    def _parse(content: str) -> dict:
        try:
            data = json.loads(_strip_code_fences(content))
        except json.JSONDecodeError as exc:
            raise TriageError(f"triage response was not valid JSON: {exc}") from exc
        if not isinstance(data, dict):
            raise TriageError(f"triage response was not a JSON object: {content[:200]!r}")
        verdict = data.get("verdict")
        if verdict not in VERDICTS:
            raise TriageError(f"triage verdict not one of {VERDICTS}: {verdict!r}")
        return {
            "verdict": verdict,
            "reason": data.get("reason") if isinstance(data.get("reason"), str) else "",
            "supporting_quote": (data.get("supporting_quote")
                                 if isinstance(data.get("supporting_quote"), str) else ""),
        }

    # -- cache --------------------------------------------------------------- #
    def _cache_path(self, note_text: str) -> Path:
        # Keyed on the note text AND the prompt, so editing the prompt invalidates
        # the cache automatically (no stale verdicts from an older rubric).
        key = _text_hash(note_text, SYSTEM_PROMPT, MODEL, REASONING_EFFORT)
        return self.cache_dir / f"triage_{key}.json"

    def _read_cache(self, path: Path):
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
        if isinstance(data, dict) and data.get("verdict") in VERDICTS:
            return data
        return None

    def _write_cache(self, path: Path, result: dict) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2),
                        encoding="utf-8")

    # -- public -------------------------------------------------------------- #
    def triage(self, note_text: str, use_cache: bool = True) -> dict:
        """Return ``{"verdict", "reason", "supporting_quote"}`` for one note."""
        path = self._cache_path(note_text)
        if use_cache:
            cached = self._read_cache(path)
            if cached is not None:
                return {k: cached[k] for k in ("verdict", "reason", "supporting_quote")}
        # The whole note goes to the model, uncut.
        result = self._parse(self._call(f"Discharge summary:\n{note_text}"))
        self._write_cache(path, result)
        return result


def run(input_file: Path, output_file: Path, limit: int | None, use_cache: bool) -> None:
    df = pd.read_csv(input_file)
    if TEXT_COLUMN not in df.columns:
        raise SystemExit(f"Input {input_file} has no '{TEXT_COLUMN}' column.")
    if limit is not None:
        df = df.head(limit)

    cache_dir = output_file.parent / CACHE_DIRNAME
    triager = Triager(cache_dir=cache_dir)

    jsonl_file = output_file.with_suffix(".jsonl")

    # Carry a few identifying columns through to the output for traceability.
    id_cols = [c for c in (ID_COLUMN, "subject_id", "hadm_id", "matched_reactions")
               if c in df.columns]

    counts = {v: 0 for v in VERDICTS}
    rows: list[dict] = []
    total = len(df)
    print(f"Triaging {total} screened summaries with {MODEL} "
          f"(reasoning_effort={REASONING_EFFORT})...")

    with jsonl_file.open("w", encoding="utf-8") as jf:
        for i, (_, row) in enumerate(df.iterrows(), start=1):
            note_text = "" if pd.isna(row[TEXT_COLUMN]) else str(row[TEXT_COLUMN])
            note_id = row.get(ID_COLUMN, i)
            try:
                result = triager.triage(note_text, use_cache=use_cache)
            except TriageError as exc:
                # One bad note must not sink the whole run; record and move on.
                print(f"  [{i}/{total}] {note_id}: ERROR -- {exc}", file=sys.stderr)
                result = {"verdict": "error", "reason": str(exc), "supporting_quote": ""}
            else:
                counts[result["verdict"]] += 1

            record = {col: (None if pd.isna(row[col]) else row[col]) for col in id_cols}
            record.update(result)
            rows.append(record)
            jf.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
            jf.flush()

            if i % 25 == 0 or i == total:
                done = counts["genuine"] + counts["ruled_out"] + counts["mention_only"]
                print(f"  [{i}/{total}] graded={done} | "
                      f"genuine={counts['genuine']} "
                      f"ruled_out={counts['ruled_out']} "
                      f"mention_only={counts['mention_only']}")

    pd.DataFrame(rows).to_csv(output_file, index=False)

    print("\nFinished.")
    print(f"Total triaged: {total}")
    for v in VERDICTS:
        print(f"  {v}: {counts[v]}")
    errors = total - sum(counts.values())
    if errors:
        print(f"  errors: {errors} (see stderr; re-run to retry them)")
    print(f"Saved CSV:   {output_file}")
    print(f"Saved JSONL: {jsonl_file}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=DEFAULT_INPUT,
                        help=f"screened summaries CSV (default: {DEFAULT_INPUT})")
    parser.add_argument("--output", default=DEFAULT_OUTPUT,
                        help=f"triage results CSV (default: {DEFAULT_OUTPUT})")
    parser.add_argument("--limit", type=int, default=None,
                        help="triage only the first N notes (for a quick test)")
    parser.add_argument("--no-cache", action="store_true",
                        help="ignore cached verdicts and re-call the model")
    args = parser.parse_args(argv)

    # Relative paths resolve against the data directory, so output never lands in
    # this tracked package by accident.
    input_file = Path(args.input)
    if not input_file.is_absolute():
        input_file = DATA_DIR / input_file
    output_file = Path(args.output)
    if not output_file.is_absolute():
        output_file = DATA_DIR / output_file

    if not input_file.exists():
        raise SystemExit(
            f"Input not found: {input_file}\n"
            f"Run extract_transfusion_summaries.py first, or set $MIMIC_DATA_DIR.")

    run(input_file, output_file, limit=args.limit, use_cache=not args.no_cache)


if __name__ == "__main__":
    main()
