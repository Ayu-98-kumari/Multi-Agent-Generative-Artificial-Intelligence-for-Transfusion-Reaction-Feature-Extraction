"""Pass 2 — page-by-page vision parsing with the Groq VLM.

Each page is rendered to an image and sent to the model together with the
skeleton's context for that page (section, expected tables/figures, layout hint).
The model fills the fixed schema (``PageParse``) — it must not invent keys.

Hard guarantees for medical content:
  * retry with exponential backoff + jitter on transient errors / 429s,
  * a truncation detector: a response that finishes for ``length`` is never
    accepted; we retry once with a higher token budget, then surface a hard
    failure (``TruncationError``),
  * ``temperature=0`` and JSON mode for deterministic, parseable output.

The Groq client is injectable so tests can run with a fake client (no network).
"""

from __future__ import annotations

import base64
import json
import logging
import random
import time
from pathlib import Path
from typing import Iterable, Optional

import fitz  # PyMuPDF

from . import config
from .schema import PageParse, Skeleton, TableBlock

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #
class TruncationError(RuntimeError):
    """Raised when a page response is cut off for length even after a retry."""


class VisionParseError(RuntimeError):
    """Raised when a page response cannot be parsed into the fixed schema."""


# --------------------------------------------------------------------------- #
# Transient errors worth retrying (lazy import keeps the module import-safe)
# --------------------------------------------------------------------------- #
def _transient_exceptions() -> tuple[type[BaseException], ...]:
    try:
        import groq
    except ImportError:  # pragma: no cover
        return ()
    return (
        groq.RateLimitError,
        groq.APIConnectionError,
        groq.APITimeoutError,
        groq.InternalServerError,
    )


# Finish reasons that mean "we did not get complete, valid JSON for this page".
INCOMPLETE_FINISH = {"length", "json_validate_failed"}


def _is_json_validate_failed(exc: BaseException) -> bool:
    """True for a Groq 400 whose error code is json_validate_failed."""
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        return body.get("error", {}).get("code") == "json_validate_failed"
    return False


# --------------------------------------------------------------------------- #
# Prompt loading
# --------------------------------------------------------------------------- #
def _load_prompt(name: str) -> str:
    return (config.PROMPTS_DIR / name).read_text(encoding="utf-8")


# --------------------------------------------------------------------------- #
# Page image + context
# --------------------------------------------------------------------------- #
def render_page_data_url(doc: "fitz.Document", page_index: int, dpi: int) -> str:
    """Render a page (0-based index) to a base64 PNG data URL for the VLM."""
    pix = doc[page_index].get_pixmap(dpi=dpi)
    b64 = base64.b64encode(pix.tobytes("png")).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _continued_table_for_page(skeleton: Skeleton, page_number: int):
    """Deterministically: does this page continue a table that started earlier?

    True when the most recent table before this page is in the same section and
    NO new table/figure starts on any page between it and this page (inclusive).
    This is what lets the dense incident-code Table 8 (pages 19-22, whose
    continuation pages carry no visible header) be parsed as table rows rather
    than prose — without depending on the model's own continued flag, which can
    drop mid-run."""
    items = sorted(skeleton.inventory, key=lambda i: i.page)
    prior_tables = [i for i in items if i.page < page_number and i.kind == "table"]
    if not prior_tables:
        return None
    last = prior_tables[-1]
    # A new table/figure starting in (last.page, page_number] ends the run.
    if any(last.page < i.page <= page_number for i in items):
        return None
    p2s = skeleton.page_to_section()
    if p2s.get(last.page) != p2s.get(page_number):
        return None
    return last


def _continuation_hint(open_info: Optional[dict]) -> str:
    """Instruct a continuation page to transcribe its rows as a table.

    The hint is FIRM (no prose escape) because the list-like incident-code pages
    otherwise default to prose. It is safe: the model still parses a genuinely
    prose page as prose even when told it is a continuation (verified on page 5),
    so a false-positive structural hint does not corrupt prose content."""
    if not open_info:
        return "No table is open from the previous page; parse this page normally."
    cols = open_info.get("columns")
    cols_clause = (f" with columns {cols}" if cols
                   else " using the same columns as that table's first page")
    return (
        f"This page is a CONTINUATION of the table {open_info['label']!r} "
        f"({open_info.get('caption') or 'no caption'}) that began on an earlier "
        f"page and has no repeated header here. Transcribe ALL of its content as "
        f"TABLE ROWS{cols_clause} and set continued_from=true (continued=true if it "
        f"still runs onto the next page). The entries may look like a two-column "
        f"list, but they ARE table rows — output them in the table block, not as "
        f"prose. (If, and only if, this page is genuinely prose, parse it normally.)"
    )


def _page_context(
    skeleton: Skeleton, page_number: int, prev_open: Optional[dict] = None
) -> dict:
    """Assemble the per-page prompt context from the deterministic skeleton."""
    section = next(
        (s for s in skeleton.sections
         if s.page_span[0] <= page_number <= s.page_span[1]),
        None,
    )
    sec_num = section.section_number if section else 0
    sec_title = section.title if section else "Unknown"
    sec_span = section.page_span if section else [page_number, page_number]

    expected = [i for i in skeleton.inventory if i.page == page_number]
    if expected:
        lines = []
        for i in expected:
            kind = "archived reaction table" if i.archived else i.kind
            lines.append(f'- expect a {kind}: "{i.label}" — {i.caption!r}')
        expected_block = "\n".join(lines)
    else:
        expected_block = "- (no specific table/figure starts on this page)"

    page_text = next((p for p in skeleton.pages if p.page == page_number), None)
    if page_text and page_text.is_near_empty:
        layout_hint = "This page appears nearly empty; emit only what is present."
    elif page_text and page_text.two_column:
        layout_hint = ("Two-column layout detected. Preserve correct reading "
                       "order; if it is a parallel-column table, keep each row's "
                       "cells together.")
    else:
        layout_hint = "Single-column layout."

    return {
        "page": page_number,
        "section_number": sec_num,
        "section_title": sec_title,
        "section_span": sec_span,
        "expected_block": expected_block,
        "layout_hint": layout_hint,
        "continuation_hint": _continuation_hint(
            _combine_continuation(skeleton, page_number, prev_open)),
    }


def _combine_continuation(
    skeleton: Skeleton, page_number: int, prev_open: Optional[dict]
) -> Optional[dict]:
    """Prefer the deterministic skeleton signal (with the model's columns if the
    chain is intact); fall back to the model's own continued flag (covers a table
    whose last page also starts a figure, e.g. version history on page 34)."""
    item = _continued_table_for_page(skeleton, page_number)
    if item is None:
        # No deterministic continuation here. We do NOT fall back to the model's
        # own flag: that would mislabel a page that simply follows a single-page
        # table (e.g. page 14 after Table 4) as a continuation.
        return None
    # Carry the previous fragment's columns so every continuation page reuses the
    # table's real header (otherwise the tail page invents columns and validation
    # flags inconsistent row keys).
    cols = prev_open.get("columns") if prev_open else None
    return {"label": item.label, "caption": item.caption, "columns": cols}


def _strip_code_fences(text: str) -> str:
    """Defensively remove ```json fences if the model wraps its output."""
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else t
        if t.endswith("```"):
            t = t[: t.rfind("```")]
    return t.strip()


# --------------------------------------------------------------------------- #
# Parser
# --------------------------------------------------------------------------- #
class GroqVisionParser:
    """Drives the Groq VLM over pages. Inject a client for tests; omit it to use
    a real Groq client built from GROQ_API_KEY."""

    def __init__(
        self,
        client=None,
        *,
        model: str = config.VISION_MODEL,
        max_completion_tokens: int = config.MAX_COMPLETION_TOKENS,
        retry_max_completion_tokens: int = config.MAX_COMPLETION_TOKENS_RETRY,
        dpi: int = config.RENDER_DPI,
        cache_dir: Optional[str | Path] = None,
    ):
        self._client = client
        self.model = model
        self.max_completion_tokens = max_completion_tokens
        self.retry_max_completion_tokens = retry_max_completion_tokens
        self.dpi = dpi
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.system_prompt = _load_prompt("system.md")
        self.user_template = _load_prompt("page_user.md")

    # -- client ----------------------------------------------------------- #
    @property
    def client(self):
        if self._client is None:
            from groq import Groq
            # max_retries=0: our _call_with_retries owns backoff. Otherwise the
            # SDK's own retries stack on top of ours and hammer the endpoint.
            self._client = Groq(api_key=config.get_groq_api_key(), max_retries=0)
        return self._client

    # -- one raw call ----------------------------------------------------- #
    def _call_once(self, messages: list, max_tokens: int) -> tuple[str | None, str]:
        kwargs = dict(
            model=self.model,
            messages=messages,
            temperature=config.TEMPERATURE,
            max_completion_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
        if config.REASONING_EFFORT is not None:
            kwargs["reasoning_effort"] = config.REASONING_EFFORT
        try:
            resp = self.client.chat.completions.create(**kwargs)
        except Exception as exc:
            # Treat a json_validate_failed 400 as an "incomplete response" signal
            # (synthetic finish_reason) so the truncation path can retry it,
            # rather than crashing. Other bad requests are real errors -> re-raise.
            if _is_json_validate_failed(exc):
                return None, "json_validate_failed"
            raise
        choice = resp.choices[0]
        return choice.message.content, choice.finish_reason

    def _call_with_retries(self, messages: list, max_tokens: int) -> tuple[str, str]:
        transient = _transient_exceptions()
        last_exc: Optional[BaseException] = None
        for attempt in range(1, config.MAX_RETRIES + 1):
            try:
                return self._call_once(messages, max_tokens)
            except transient as exc:  # type: ignore[misc]
                last_exc = exc
                if attempt == config.MAX_RETRIES:
                    break
                delay = min(
                    config.BACKOFF_INITIAL * config.BACKOFF_FACTOR ** (attempt - 1),
                    config.BACKOFF_MAX,
                )
                delay *= 1 + random.uniform(-config.BACKOFF_JITTER, config.BACKOFF_JITTER)
                logger.warning(
                    "transient error on attempt %d/%d (%s); retrying in %.1fs",
                    attempt, config.MAX_RETRIES, type(exc).__name__, delay,
                )
                time.sleep(max(0.0, delay))
        raise VisionParseError(
            f"giving up after {config.MAX_RETRIES} attempts: {last_exc!r}"
        )

    # -- messages --------------------------------------------------------- #
    def _build_messages(self, ctx: dict, data_url: str) -> list:
        user_text = self.user_template.format(**ctx)
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": [
                {"type": "text", "text": user_text},
                {"type": "image_url", "image_url": {"url": data_url}},
            ]},
        ]

    # -- public: one page ------------------------------------------------- #
    def parse_page(
        self, skeleton: Skeleton, doc: "fitz.Document", page_number: int,
        prev_open: Optional[dict] = None,
    ) -> PageParse:
        """Parse a single page (1-based). ``prev_open`` carries an open table from
        the previous page so continuations are emitted as tables, not prose.
        Caches by page if cache_dir is set."""
        cached = self._read_cache(skeleton.source_filename, page_number)
        if cached is not None:
            return cached

        ctx = _page_context(skeleton, page_number, prev_open)
        data_url = render_page_data_url(doc, page_number - 1, self.dpi)
        messages = self._build_messages(ctx, data_url)

        content, finish = self._call_with_retries(messages, self.max_completion_tokens)
        if finish in INCOMPLETE_FINISH:
            logger.warning(
                "page %d incomplete (finish_reason=%s); retrying with %d tokens",
                page_number, finish, self.retry_max_completion_tokens,
            )
            content, finish = self._call_with_retries(
                messages, self.retry_max_completion_tokens
            )
            if finish in INCOMPLETE_FINISH:
                raise TruncationError(
                    f"page {page_number} response incomplete (finish_reason="
                    f"{finish}) even at {self.retry_max_completion_tokens} tokens; "
                    f"raise MAX_COMPLETION_TOKENS_RETRY (below MODEL_OTPM_LIMIT) "
                    f"in config.py, or check REASONING_EFFORT"
                )

        page = self._parse_content(content, ctx, page_number)
        self._write_cache(skeleton.source_filename, page_number, page)
        return page

    def _parse_content(self, content: str, ctx: dict, page_number: int) -> PageParse:
        if not content or not content.strip():
            raise VisionParseError(f"page {page_number}: empty model response")
        try:
            data = json.loads(_strip_code_fences(content))
        except json.JSONDecodeError as exc:
            raise VisionParseError(f"page {page_number}: invalid JSON ({exc})") from exc

        # Backfill page/section if the model omitted them; keep its blocks.
        data.setdefault("page", page_number)
        data.setdefault("section_number", ctx["section_number"])
        try:
            return PageParse.model_validate(data)
        except Exception as exc:  # pydantic ValidationError
            raise VisionParseError(
                f"page {page_number}: response does not match schema ({exc})"
            ) from exc

    # -- public: many pages ---------------------------------------------- #
    def parse_pages(
        self,
        skeleton: Skeleton,
        pdf_path: str | Path,
        page_numbers: Optional[Iterable[int]] = None,
    ) -> dict[int, PageParse]:
        """Parse the given pages (default: all). Returns {page_number: PageParse}."""
        doc = fitz.open(pdf_path)
        try:
            pages = (list(page_numbers) if page_numbers is not None
                     else list(range(1, skeleton.total_pages + 1)))
            results: dict[int, PageParse] = {}
            prev_open: Optional[dict] = None
            last_pn: Optional[int] = None
            for pn in pages:
                logger.info("parsing page %d/%d", pn, skeleton.total_pages)
                # Only carry an open table to the immediately following page.
                if last_pn is None or pn != last_pn + 1:
                    prev_open = None
                page = self.parse_page(skeleton, doc, pn, prev_open=prev_open)
                results[pn] = page
                prev_open = self._open_table(page)
                last_pn = pn
            return results
        finally:
            doc.close()

    @staticmethod
    def _open_table(page: PageParse) -> Optional[dict]:
        """Identity (label/caption/columns) of the last table on this page, to
        carry to the next page. Not gated on the model's ``continued`` flag — the
        skeleton decides whether the next page actually continues it; we only need
        the columns so a continuation reuses the real header. None if no table."""
        tables = [b for b in page.blocks if isinstance(b, TableBlock)]
        if not tables:
            return None
        t = tables[-1]
        return {"label": t.label, "caption": t.caption, "columns": list(t.columns)}

    # -- cache ------------------------------------------------------------ #
    def _cache_path(self, source_filename: str, page_number: int) -> Optional[Path]:
        if self.cache_dir is None:
            return None
        stem = Path(source_filename).stem
        return self.cache_dir / stem / f"page_{page_number:03d}.json"

    def _read_cache(self, source_filename: str, page_number: int) -> Optional[PageParse]:
        path = self._cache_path(source_filename, page_number)
        if path and path.exists():
            logger.info("cache hit page %d (%s)", page_number, path)
            return PageParse.model_validate_json(path.read_text(encoding="utf-8"))
        return None

    def _write_cache(self, source_filename: str, page_number: int, page: PageParse):
        path = self._cache_path(source_filename, page_number)
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(page.model_dump_json(indent=2), encoding="utf-8")
