"""Pipeline + CLI wiring tests with a fake vision parser (no Groq, no spend).

The fake parser returns canned per-page blocks so we can verify that
stitch -> assemble places a cross-page Table 8 as one block in the right
section, assigns block_ids, writes output, and runs validation.
"""

from __future__ import annotations

import pytest

from pdf_parser import cli, config
from pdf_parser.pipeline import Pipeline
from pdf_parser.schema import PageParse, ProseBlock, TableBlock, TableRow
from pdf_parser.skeleton import build_skeleton
from pdf_parser.validation import ValidationReport

PDF_PATH = config.PROJECT_ROOT / "bv-hv-protocol-current.pdf"
needs_pdf = pytest.mark.skipif(not PDF_PATH.exists(), reason="sample PDF missing")


class FakeParser:
    """Returns Table 8 fragments on pages 19-22 (with continuation flags) and a
    prose block on every other page."""

    def parse_pages(self, skeleton, pdf_path, pages=None):
        page_list = (list(pages) if pages is not None
                     else list(range(1, skeleton.total_pages + 1)))
        p2s = skeleton.page_to_section()
        out: dict[int, PageParse] = {}
        for pn in page_list:
            sec = p2s[pn]
            if 19 <= pn <= 22:
                blocks = [TableBlock(
                    label="Table 8", page_span=[pn, pn],
                    columns=["Code", "Desc"],
                    rows=[TableRow(cells={"Code": f"X{pn}", "Desc": "d"})],
                    continued=(pn < 22), continued_from=(pn > 19))]
            else:
                blocks = [ProseBlock(text=f"page {pn} prose", page=pn)]
            out[pn] = PageParse(page=pn, section_number=sec, blocks=blocks)
        return out


@pytest.fixture(scope="module")
def skeleton():
    if not PDF_PATH.exists():
        pytest.skip("sample PDF missing")
    return build_skeleton(PDF_PATH)


@needs_pdf
def test_pipeline_assembles_and_stitches(tmp_path, skeleton):
    pipe = Pipeline(PDF_PATH, output_path=tmp_path / "out.json",
                    work_dir=tmp_path / "wd", parser=FakeParser())
    result = pipe.run()

    doc = result.document
    assert doc is not None
    # One section per skeleton section (front matter + 7).
    assert len(doc.sections) == len(skeleton.sections)

    # Table 8 stitched into a single block in section 5, span [19,22], 4 rows.
    sec5 = next(s for s in doc.sections if s.section_number == 5)
    tables = [b for b in sec5.content if isinstance(b, TableBlock)]
    assert len(tables) == 1
    assert tables[0].label == "Table 8"
    assert tables[0].page_span == [19, 22]
    assert len(tables[0].rows) == 4

    # block_ids assigned, output written, validation ran.
    assert all(b.block_id for s in doc.sections for b in s.content)
    assert result.output_path.exists()
    assert isinstance(result.report, ValidationReport)
    assert result.report.coverage is not None


@needs_pdf
def test_pipeline_single_page(tmp_path, skeleton):
    pipe = Pipeline(PDF_PATH, work_dir=tmp_path / "wd", parser=FakeParser())
    page = pipe.parse_single_page(skeleton, 19)
    assert page.page == 19
    assert page.blocks[0].label == "Table 8"


@needs_pdf
def test_skeleton_is_cached_and_reused(tmp_path, skeleton):
    pipe = Pipeline(PDF_PATH, work_dir=tmp_path / "wd", parser=FakeParser())
    s1 = pipe.load_or_build_skeleton()
    assert pipe.skeleton_path.exists()
    s2 = pipe.load_or_build_skeleton()      # second call loads from disk
    assert s1.total_pages == s2.total_pages == 34


@needs_pdf
def test_cli_skeleton_only(tmp_path, capsys):
    rc = cli.main(["--input", str(PDF_PATH), "--work-dir", str(tmp_path),
                   "--skeleton-only"])
    assert rc == 0
    assert (tmp_path / config.SKELETON_FILENAME).exists()
    out = capsys.readouterr().out
    assert "tables      : 17" in out
    assert "figures     : 3" in out
