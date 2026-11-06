"""Orchestration: skeleton -> vision -> stitch -> assemble -> validate -> write.

Each stage is independently runnable and cached where sensible:
  * the skeleton is written to ``skeleton.json`` and reused unless ``force``,
  * per-page VLM results are cached by the vision parser,
so re-runs are cheap and the VLM pass never re-extracts structure.

The vision parser is injectable (tests pass a fake; the CLI uses a real Groq
client). ``parse_single_page`` isolates one page for a cheap first real call.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

from . import config
from .schema import Document, PageParse, Section, Skeleton, TableBlock
from .skeleton import build_skeleton, write_skeleton
from .stitching import stitch
from .validation import ValidationReport, validate

logger = logging.getLogger(__name__)


def _block_start_page(block) -> int:
    return block.page_span[0] if isinstance(block, TableBlock) else block.page


@dataclass
class RunResult:
    skeleton: Skeleton
    document: Optional[Document] = None
    report: Optional[ValidationReport] = None
    output_path: Optional[Path] = None
    skeleton_path: Optional[Path] = None


class Pipeline:
    def __init__(
        self,
        pdf_path: str | Path,
        *,
        output_path: str | Path | None = None,
        work_dir: str | Path | None = None,
        parser=None,
        coverage_threshold: float = config.COVERAGE_THRESHOLD,
        use_cache: bool = True,
    ):
        self.pdf_path = Path(pdf_path)
        stem = self.pdf_path.stem
        self.work_dir = Path(work_dir) if work_dir else \
            self.pdf_path.parent / f"{stem}_pdfparser"
        self.output_path = Path(output_path) if output_path else \
            self.pdf_path.with_name(f"{stem}.json")
        self.coverage_threshold = coverage_threshold

        if parser is not None:
            self.parser = parser
        else:
            from .vision_parser import GroqVisionParser
            cache_dir = self.work_dir / "pages" if use_cache else None
            self.parser = GroqVisionParser(cache_dir=cache_dir)

    # -- stage 1: skeleton ------------------------------------------------ #
    @property
    def skeleton_path(self) -> Path:
        return self.work_dir / config.SKELETON_FILENAME

    def load_or_build_skeleton(self, force: bool = False) -> Skeleton:
        if not force and self.skeleton_path.exists():
            logger.info("loading cached skeleton %s", self.skeleton_path)
            return Skeleton.model_validate_json(
                self.skeleton_path.read_text(encoding="utf-8"))
        logger.info("building skeleton from %s", self.pdf_path)
        skeleton = build_skeleton(self.pdf_path)
        write_skeleton(skeleton, self.work_dir)
        return skeleton

    # -- stage 2: vision -------------------------------------------------- #
    def parse(
        self, skeleton: Skeleton, pages: Optional[Iterable[int]] = None
    ) -> dict[int, PageParse]:
        return self.parser.parse_pages(skeleton, self.pdf_path, pages)

    def parse_single_page(self, skeleton: Skeleton, page_number: int) -> PageParse:
        """Parse exactly one page (cheap, isolated first real call)."""
        results = self.parser.parse_pages(skeleton, self.pdf_path, [page_number])
        return results[page_number]

    # -- stage 3+4: stitch + assemble ------------------------------------ #
    def assemble(
        self, skeleton: Skeleton, page_results: dict[int, PageParse]
    ) -> Document:
        flat_blocks = stitch(skeleton, page_results)
        p2s = skeleton.page_to_section()
        last_section = skeleton.sections[-1].section_number if skeleton.sections else 0

        buckets: dict[int, list] = {s.section_number: [] for s in skeleton.sections}
        for block in flat_blocks:
            sec = p2s.get(_block_start_page(block), last_section)
            buckets.setdefault(sec, []).append(block)

        sections: list[Section] = []
        for info in skeleton.sections:
            blocks = buckets.get(info.section_number, [])
            for i, block in enumerate(blocks):
                block.block_id = f"s{info.section_number}-b{i}"
            sections.append(Section(
                section_number=info.section_number, title=info.title,
                page_span=info.page_span, content=blocks))

        return Document(
            title=skeleton.title, version=skeleton.version,
            source_filename=skeleton.source_filename,
            total_pages=skeleton.total_pages,
            parsed_on=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            sections=sections)

    # -- stage 5: validate ------------------------------------------------ #
    def validate(self, document: Document, skeleton: Skeleton) -> ValidationReport:
        return validate(document, skeleton, self.coverage_threshold)

    # -- write ------------------------------------------------------------ #
    def write_document(self, document: Document) -> Path:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(
            document.model_dump_json(indent=2), encoding="utf-8")
        return self.output_path

    # -- orchestration ---------------------------------------------------- #
    def run(
        self,
        *,
        pages: Optional[Iterable[int]] = None,
        do_vision: bool = True,
        do_validate: bool = True,
        write_output: bool = True,
        force_skeleton: bool = False,
    ) -> RunResult:
        """Full run (or a subset of stages). Returns artifacts and the report."""
        skeleton = self.load_or_build_skeleton(force=force_skeleton)
        result = RunResult(skeleton=skeleton, skeleton_path=self.skeleton_path)
        if not do_vision:
            return result

        page_results = self.parse(skeleton, pages)
        document = self.assemble(skeleton, page_results)
        result.document = document
        if write_output:
            result.output_path = self.write_document(document)
        if do_validate:
            result.report = self.validate(document, skeleton)
        return result
