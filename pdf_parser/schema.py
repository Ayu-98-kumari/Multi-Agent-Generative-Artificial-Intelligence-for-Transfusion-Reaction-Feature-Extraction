"""Pydantic v2 models — the fixed, predefined shape the whole pipeline targets.

Two families of models:

* Skeleton models (Pass 1 artifact): ``Skeleton``, ``SectionInfo``,
  ``InventoryItem``, ``PageText``. Deterministic, inspectable backbone.
* Document models (final output): ``Document`` -> ``Section`` -> a flat list of
  typed ``ContentBlock``s (``ProseBlock`` | ``TableBlock`` | ``FigureBlock``),
  discriminated on ``type``.

Design rule (decided): the shape reads like a nested document outline but stays
*shallow* — one level of blocks under each section. Page is metadata (a field or
span), never a structural level.
"""

from __future__ import annotations

from typing import Annotated, Literal, Optional, Union

from pydantic import BaseModel, Field, model_validator

# --------------------------------------------------------------------------- #
# Shared
# --------------------------------------------------------------------------- #
PageSpan = Annotated[list[int], Field(min_length=2, max_length=2)]
"""[start, end] inclusive, 1-based printed/PDF page numbers."""


# --------------------------------------------------------------------------- #
# Pass-1 skeleton models
# --------------------------------------------------------------------------- #
class InventoryItem(BaseModel):
    """An expected table or figure, found deterministically in Pass 1."""

    kind: Literal["table", "figure"]
    label: str                      # e.g. "Table 4", "Figure 1", "Archived: DHTR"
    number: int                     # numbered tables/figures; 0 for archived
    caption: str = ""               # text following the label / full reaction name
    page: int                       # 1-based page where the label/heading appears
    # Section-6 archived reaction tables carry no "Table N." prefix; flagged so
    # validation and the VLM treat them as expected tables all the same.
    archived: bool = False


class SectionInfo(BaseModel):
    """A section's identity and page span, derived from TOC + heading regex."""

    section_number: int             # 0 reserved for front matter
    title: str
    page_span: PageSpan


class PageText(BaseModel):
    """Normalized, header/footer-stripped text for one page (coverage baseline)."""

    page: int                       # 1-based
    section_number: int             # owning section (page -> section map)
    text: str
    is_near_empty: bool = False
    two_column: bool = False


class Skeleton(BaseModel):
    """The complete Pass-1 artifact (written to skeleton.json)."""

    source_filename: str
    title: Optional[str] = None
    version: Optional[str] = None
    total_pages: int
    sections: list[SectionInfo]
    inventory: list[InventoryItem]
    pages: list[PageText]

    def expected_tables(self) -> list[InventoryItem]:
        return [i for i in self.inventory if i.kind == "table"]

    def expected_figures(self) -> list[InventoryItem]:
        return [i for i in self.inventory if i.kind == "figure"]

    def page_to_section(self) -> dict[int, int]:
        return {p.page: p.section_number for p in self.pages}


# --------------------------------------------------------------------------- #
# Final document models — typed content blocks
# --------------------------------------------------------------------------- #
class ProseBlock(BaseModel):
    type: Literal["prose"] = "prose"
    block_id: Optional[str] = None          # e.g. "s1-b0" (traceability/RAG)
    subheading: Optional[str] = None        # label on the block, NOT a container
    text: str
    page: int


class TableRow(BaseModel):
    """One row, keyed by column header -> cell text (logic preserved verbatim)."""

    cells: dict[str, str]


class TableBlock(BaseModel):
    type: Literal["table"] = "table"
    block_id: Optional[str] = None
    label: str                              # "Table 4"
    caption: str = ""                       # "Acute hemolytic transfusion reaction (AHTR)"
    page_span: PageSpan
    columns: list[str]
    rows: list[TableRow]
    notes: list[str] = Field(default_factory=list)       # "Note:" lines
    footnotes: list[str] = Field(default_factory=list)
    # Stitching flags (set in Pass 2, consumed by stitching.py).
    continued_from: bool = False            # this fragment continues a prior page
    continued: bool = False                 # this fragment continues onto next page


class FigureBlock(BaseModel):
    type: Literal["figure"] = "figure"
    block_id: Optional[str] = None
    label: str                              # "Figure 1"
    caption: str = ""                       # extracted deterministically where possible
    description: str                        # VLM vision description (must be non-empty)
    page: int
    abbreviations: Optional[str] = None     # legend / "Abbreviations:" line


ContentBlock = Annotated[
    Union[ProseBlock, TableBlock, FigureBlock],
    Field(discriminator="type"),
]


class Section(BaseModel):
    section_number: int
    title: str
    page_span: PageSpan
    content: list[ContentBlock] = Field(default_factory=list)


class PageParse(BaseModel):
    """The VLM's response for a single page: ordered content blocks.

    This is the fixed shape the model must fill (it must not invent keys). The
    pipeline assigns ``block_id`` and reconciles ``section_number`` afterwards."""

    page: int
    section_number: int
    blocks: list[ContentBlock] = Field(default_factory=list)


class Document(BaseModel):
    title: Optional[str] = None
    version: Optional[str] = None
    source_filename: str
    total_pages: int
    parsed_on: Optional[str] = None         # ISO timestamp
    sections: list[Section] = Field(default_factory=list)

    @model_validator(mode="after")
    def _sections_ordered(self) -> "Document":
        nums = [s.section_number for s in self.sections]
        if nums != sorted(nums):
            raise ValueError(f"sections out of order: {nums}")
        return self
