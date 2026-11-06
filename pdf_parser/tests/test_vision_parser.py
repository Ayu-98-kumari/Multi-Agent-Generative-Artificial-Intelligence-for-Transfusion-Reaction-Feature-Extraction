"""Tests for Pass 2 (vision) using a fake Groq client — no network, no spend.

The fake client mimics groq's response shape:
    resp.choices[0].message.content      -> the JSON string
    resp.choices[0].finish_reason        -> "stop" | "length"
A behavior list lets each .create() call return the next canned response (or
raise the next exception), so we can drive retry/truncation paths deterministically.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from pdf_parser import config, vision_parser
from pdf_parser.schema import Skeleton
from pdf_parser.skeleton import build_skeleton
from pdf_parser.vision_parser import (
    GroqVisionParser,
    TruncationError,
    VisionParseError,
    _strip_code_fences,
)

PDF_PATH = config.PROJECT_ROOT / "bv-hv-protocol-current.pdf"
needs_pdf = pytest.mark.skipif(not PDF_PATH.exists(), reason="sample PDF missing")


# --------------------------------------------------------------------------- #
# Fake client
# --------------------------------------------------------------------------- #
def _resp(content: str, finish: str = "stop"):
    msg = SimpleNamespace(content=content)
    return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason=finish)])


class _FakeCompletions:
    def __init__(self, behaviors):
        self.behaviors = list(behaviors)
        self.calls = 0

    def create(self, **kwargs):
        self.kwargs = kwargs
        b = self.behaviors[min(self.calls, len(self.behaviors) - 1)]
        self.calls += 1
        if isinstance(b, Exception):
            raise b
        return b


class FakeClient:
    def __init__(self, behaviors):
        self.chat = SimpleNamespace(completions=_FakeCompletions(behaviors))


VALID_PAGE = json.dumps({
    "page": 3,
    "section_number": 1,
    "blocks": [
        {"type": "prose", "subheading": "Purpose", "text": "Body text.", "page": 3},
        {"type": "figure", "label": "Figure 1", "caption": "Surveillance",
         "description": "A flowchart of the surveillance process.", "page": 3,
         "abbreviations": None},
    ],
})


@pytest.fixture
def no_sleep(monkeypatch):
    monkeypatch.setattr(vision_parser.time, "sleep", lambda *_: None)


@pytest.fixture(scope="module")
def skeleton():
    if not PDF_PATH.exists():
        pytest.skip("sample PDF missing")
    return build_skeleton(PDF_PATH)


# --------------------------------------------------------------------------- #
# Pure helpers (no PDF needed)
# --------------------------------------------------------------------------- #
def test_strip_code_fences():
    assert _strip_code_fences('```json\n{"a":1}\n```') == '{"a":1}'
    assert _strip_code_fences('{"a":1}') == '{"a":1}'


def test_parse_content_backfills_page_and_section():
    parser = GroqVisionParser(client=FakeClient([]))
    ctx = {"section_number": 1}
    body = json.dumps({"blocks": [{"type": "prose", "text": "x", "page": 3}]})
    page = parser._parse_content(body, ctx, page_number=3)
    assert page.page == 3 and page.section_number == 1
    assert page.blocks[0].text == "x"


def test_parse_content_rejects_bad_json():
    parser = GroqVisionParser(client=FakeClient([]))
    with pytest.raises(VisionParseError):
        parser._parse_content("not json", {"section_number": 0}, 1)


def test_parse_content_rejects_off_schema():
    parser = GroqVisionParser(client=FakeClient([]))
    bad = json.dumps({"page": 1, "section_number": 0,
                      "blocks": [{"type": "bogus", "x": 1}]})
    with pytest.raises(VisionParseError):
        parser._parse_content(bad, {"section_number": 0}, 1)


# --------------------------------------------------------------------------- #
# Context building (needs skeleton)
# --------------------------------------------------------------------------- #
@needs_pdf
def test_page_context_surfaces_archived_table(skeleton):
    ctx = vision_parser._page_context(skeleton, 25)  # DHTR archived table
    assert ctx["section_number"] == 6
    assert "Archived: DHTR" in ctx["expected_block"]


@needs_pdf
def test_page_context_two_column_hint(skeleton):
    ctx = vision_parser._page_context(skeleton, 19)  # incident codes
    assert "Two-column" in ctx["layout_hint"]


# --------------------------------------------------------------------------- #
# parse_page: success / truncation / retry  (needs PDF for rendering)
# --------------------------------------------------------------------------- #
@needs_pdf
def test_parse_page_success(skeleton):
    import fitz
    doc = fitz.open(PDF_PATH)
    parser = GroqVisionParser(client=FakeClient([_resp(VALID_PAGE)]))
    page = parser.parse_page(skeleton, doc, 3)
    assert page.page == 3
    assert [b.type for b in page.blocks] == ["prose", "figure"]
    assert page.blocks[1].description  # figure description non-empty
    doc.close()


@needs_pdf
def test_truncation_retries_then_succeeds(skeleton, no_sleep):
    import fitz
    doc = fitz.open(PDF_PATH)
    client = FakeClient([_resp(VALID_PAGE, finish="length"), _resp(VALID_PAGE)])
    parser = GroqVisionParser(client=client)
    page = parser.parse_page(skeleton, doc, 3)
    assert page.page == 3
    assert client.chat.completions.calls == 2  # retried once at higher tokens
    doc.close()


@needs_pdf
def test_truncation_twice_is_hard_failure(skeleton, no_sleep):
    import fitz
    doc = fitz.open(PDF_PATH)
    client = FakeClient([_resp(VALID_PAGE, finish="length"),
                         _resp(VALID_PAGE, finish="length")])
    parser = GroqVisionParser(client=client)
    with pytest.raises(TruncationError):
        parser.parse_page(skeleton, doc, 3)
    doc.close()


@needs_pdf
def test_transient_error_is_retried(skeleton, no_sleep, monkeypatch):
    import fitz
    doc = fitz.open(PDF_PATH)

    class _Transient(Exception):
        pass

    monkeypatch.setattr(vision_parser, "_transient_exceptions", lambda: (_Transient,))
    client = FakeClient([_Transient("429"), _resp(VALID_PAGE)])
    parser = GroqVisionParser(client=client)
    page = parser.parse_page(skeleton, doc, 3)
    assert page.page == 3
    assert client.chat.completions.calls == 2
    doc.close()


def _json_validate_failed_error(failed_generation: str = ""):
    import groq
    import httpx
    resp = httpx.Response(400, request=httpx.Request("POST", "http://x"))
    return groq.BadRequestError(
        "Failed to validate JSON", response=resp,
        body={"error": {"code": "json_validate_failed",
                        "failed_generation": failed_generation}})


@needs_pdf
def test_reasoning_effort_passed_in_request(skeleton):
    import fitz
    doc = fitz.open(PDF_PATH)
    client = FakeClient([_resp(VALID_PAGE)])
    GroqVisionParser(client=client).parse_page(skeleton, doc, 3)
    assert client.chat.completions.kwargs["reasoning_effort"] == config.REASONING_EFFORT
    doc.close()


@needs_pdf
def test_json_validate_failed_retries_then_succeeds(skeleton, no_sleep):
    import fitz
    doc = fitz.open(PDF_PATH)
    client = FakeClient([_json_validate_failed_error(), _resp(VALID_PAGE)])
    parser = GroqVisionParser(client=client)
    page = parser.parse_page(skeleton, doc, 19)
    assert page.page == 3  # from VALID_PAGE fixture
    assert client.chat.completions.calls == 2
    doc.close()


@needs_pdf
def test_json_validate_failed_twice_is_hard_failure(skeleton, no_sleep):
    import fitz
    doc = fitz.open(PDF_PATH)
    client = FakeClient([_json_validate_failed_error(),
                         _json_validate_failed_error()])
    parser = GroqVisionParser(client=client)
    with pytest.raises(TruncationError):
        parser.parse_page(skeleton, doc, 19)
    doc.close()


@needs_pdf
def test_other_bad_request_propagates(skeleton, no_sleep):
    import fitz
    import groq
    import httpx
    doc = fitz.open(PDF_PATH)
    resp = httpx.Response(400, request=httpx.Request("POST", "http://x"))
    other = groq.BadRequestError("nope", response=resp,
                                 body={"error": {"code": "something_else"}})
    parser = GroqVisionParser(client=FakeClient([other]))
    with pytest.raises(groq.BadRequestError):
        parser.parse_page(skeleton, doc, 3)
    doc.close()


TABLE_CONTINUED = json.dumps({
    "page": 19, "section_number": 5, "blocks": [
        {"type": "table", "label": "Table 8", "caption": "Incident Codes",
         "page_span": [19, 19], "columns": ["Code", "Description"],
         "rows": [{"cells": {"Code": "PC 03", "Description": "x"}}],
         "notes": [], "footnotes": [], "continued_from": False, "continued": True}]})


def test_open_table_and_continuation_hint():
    from pdf_parser.schema import PageParse
    from pdf_parser.vision_parser import GroqVisionParser, _continuation_hint
    page = PageParse.model_validate(json.loads(TABLE_CONTINUED))
    open_tbl = GroqVisionParser._open_table(page)
    assert open_tbl["label"] == "Table 8"
    hint = _continuation_hint(open_tbl)
    assert "CONTINUATION" in hint and "Table 8" in hint
    assert "No table is open" in _continuation_hint(None)


@needs_pdf
def test_parse_pages_threads_open_table_to_next_page(skeleton):
    """Page 19 leaves Table 8 open -> page 20's prompt must carry that context so
    the model continues the table instead of emitting prose."""
    client = FakeClient([_resp(TABLE_CONTINUED), _resp(VALID_PAGE)])
    parser = GroqVisionParser(client=client)
    parser.parse_pages(skeleton, PDF_PATH, [19, 20])
    # The last create() was page 20; its user text carries the continuation hint.
    user_text = client.chat.completions.kwargs["messages"][1]["content"][0]["text"]
    assert "CONTINUATION" in user_text
    assert "Table 8" in user_text


@needs_pdf
def test_parse_pages_resets_open_table_on_non_adjacent_pages(skeleton):
    client = FakeClient([_resp(TABLE_CONTINUED), _resp(VALID_PAGE)])
    parser = GroqVisionParser(client=client)
    # Page 6 (Section 2) does not continue any table per the skeleton, and the gap
    # also clears the model-flag carry-over.
    parser.parse_pages(skeleton, PDF_PATH, [19, 6])
    user_text = client.chat.completions.kwargs["messages"][1]["content"][0]["text"]
    assert "No table is open" in user_text


@needs_pdf
def test_higher_token_budget_used_on_retry(skeleton, no_sleep):
    import fitz
    doc = fitz.open(PDF_PATH)
    client = FakeClient([_resp(VALID_PAGE, finish="length"), _resp(VALID_PAGE)])
    parser = GroqVisionParser(client=client)
    parser.parse_page(skeleton, doc, 3)
    # Final call used the elevated retry budget.
    assert client.chat.completions.kwargs["max_completion_tokens"] == \
        config.MAX_COMPLETION_TOKENS_RETRY
    doc.close()
