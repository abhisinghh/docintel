"""Core data contract for docintel.

Everything in the pipeline — ingestion, layout detection, OCR, table/chart
extraction, schema extraction, the agent loop, the UI and the RAG/KG export —
speaks this one schema.  It is deliberately close to the "chunks with
grounding" shape used by LandingAI's Agentic Document Extraction, Docling's
DoclingDocument and the OmniDocBench ground-truth format, so that outputs can
be scored against public benchmarks and consumed by downstream RAG systems.

All bounding boxes are stored **normalised to [0, 1]** relative to the page
(x0, y0, x1, y1, origin top-left) so that they are resolution-independent;
the UI multiplies by the rendered page size.
"""
from __future__ import annotations

import uuid
from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


def new_id(prefix: str = "c") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


class ChunkType(str, Enum):
    # Textual
    TEXT = "text"
    TITLE = "title"
    SECTION_HEADER = "section_header"
    LIST_ITEM = "list_item"
    CAPTION = "caption"
    FOOTNOTE = "footnote"
    PAGE_HEADER = "page_header"
    PAGE_FOOTER = "page_footer"
    PAGE_NUMBER = "page_number"
    FORMULA = "formula"
    CODE = "code"
    HANDWRITING = "handwriting"
    FORM_FIELD = "form_field"      # key: value pair (forms, invoices, IDs)
    # Visual
    TABLE = "table"
    FIGURE = "figure"
    CHART = "chart"
    MAP = "map"
    INFOGRAPHIC = "infographic"
    SIGNATURE = "signature"
    STAMP = "stamp"
    BARCODE = "barcode"
    UNKNOWN = "unknown"


class BBox(BaseModel):
    """Normalised bounding box, origin top-left, values in [0, 1]."""

    x0: float
    y0: float
    x1: float
    y1: float

    def to_abs(self, width: float, height: float) -> tuple[float, float, float, float]:
        return self.x0 * width, self.y0 * height, self.x1 * width, self.y1 * height

    @classmethod
    def from_abs(cls, x0: float, y0: float, x1: float, y1: float, width: float, height: float) -> "BBox":
        w, h = max(width, 1e-6), max(height, 1e-6)
        x0, x1 = sorted((x0, x1))
        y0, y1 = sorted((y0, y1))
        return cls(
            x0=max(0.0, min(1.0, x0 / w)),
            y0=max(0.0, min(1.0, y0 / h)),
            x1=max(0.0, min(1.0, x1 / w)),
            y1=max(0.0, min(1.0, y1 / h)),
        )

    def area(self) -> float:
        return max(0.0, self.x1 - self.x0) * max(0.0, self.y1 - self.y0)

    def iou(self, other: "BBox") -> float:
        ix0, iy0 = max(self.x0, other.x0), max(self.y0, other.y0)
        ix1, iy1 = min(self.x1, other.x1), min(self.y1, other.y1)
        inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
        union = self.area() + other.area() - inter
        return inter / union if union > 0 else 0.0

    def union(self, other: "BBox") -> "BBox":
        return BBox(x0=min(self.x0, other.x0), y0=min(self.y0, other.y0),
                    x1=max(self.x1, other.x1), y1=max(self.y1, other.y1))


class Grounding(BaseModel):
    """Where a piece of content lives in the source document."""

    page: int                              # 1-based
    bbox: BBox
    chunk_id: Optional[str] = None
    snippet_path: Optional[str] = None     # cropped image of the region (optional)


class Word(BaseModel):
    text: str
    bbox: BBox
    confidence: float = 1.0


class TableCell(BaseModel):
    row: int
    col: int
    row_span: int = 1
    col_span: int = 1
    text: str = ""
    is_header: bool = False
    bbox: Optional[BBox] = None


class TableData(BaseModel):
    cells: list[TableCell] = Field(default_factory=list)
    n_rows: int = 0
    n_cols: int = 0
    html: Optional[str] = None
    caption: Optional[str] = None

    def to_grid(self) -> list[list[str]]:
        grid = [["" for _ in range(self.n_cols)] for _ in range(self.n_rows)]
        for c in self.cells:
            if 0 <= c.row < self.n_rows and 0 <= c.col < self.n_cols:
                grid[c.row][c.col] = c.text
        return grid

    def to_markdown(self) -> str:
        grid = self.to_grid()
        if not grid:
            return ""
        esc = lambda s: s.replace("|", "\\|").replace("\n", " ")  # noqa: E731
        head = "| " + " | ".join(esc(x) for x in grid[0]) + " |"
        sep = "|" + "|".join(" --- " for _ in grid[0]) + "|"
        body = ["| " + " | ".join(esc(x) for x in row) + " |" for row in grid[1:]]
        return "\n".join([head, sep, *body])

    def to_html(self) -> str:
        if self.html:
            return self.html
        grid = self.to_grid()
        rows = []
        for i, row in enumerate(grid):
            tag = "th" if i == 0 else "td"
            rows.append("<tr>" + "".join(f"<{tag}>{c}</{tag}>" for c in row) + "</tr>")
        return "<table>" + "".join(rows) + "</table>"


class ChartSeries(BaseModel):
    name: str = ""
    x: list[Any] = Field(default_factory=list)
    y: list[Any] = Field(default_factory=list)


class ChartData(BaseModel):
    chart_type: str = "unknown"            # bar | line | pie | scatter | area | map | other
    title: Optional[str] = None
    x_label: Optional[str] = None
    y_label: Optional[str] = None
    series: list[ChartSeries] = Field(default_factory=list)
    summary: Optional[str] = None          # natural-language description (what the chart says)
    raw_table: Optional[TableData] = None  # chart-to-table reconstruction


class Chunk(BaseModel):
    """A layout-aware unit of the document with visual grounding."""

    id: str = Field(default_factory=lambda: new_id("c"))
    doc_id: str = ""
    page: int = 1
    type: ChunkType = ChunkType.TEXT
    text: str = ""                         # markdown-ish plain text (for tables: markdown table)
    bbox: BBox
    reading_order: int = 0
    confidence: float = 1.0
    languages: list[str] = Field(default_factory=list)   # ISO-639 codes, e.g. ["en", "hi"]
    scripts: list[str] = Field(default_factory=list)     # e.g. ["Latin", "Devanagari"]
    source: Literal["native", "ocr", "vlm", "agent"] = "native"
    words: list[Word] = Field(default_factory=list)      # word-level grounding when available
    table: Optional[TableData] = None
    chart: Optional[ChartData] = None
    image_path: Optional[str] = None       # crop of figures/charts/maps
    parent_id: Optional[str] = None
    attributes: dict[str, Any] = Field(default_factory=dict)

    def grounding(self) -> Grounding:
        return Grounding(page=self.page, bbox=self.bbox, chunk_id=self.id, snippet_path=self.image_path)


class Page(BaseModel):
    number: int
    width: float                           # pixel size of the rendered image
    height: float
    image_path: Optional[str] = None
    rotation: int = 0
    chunks: list[Chunk] = Field(default_factory=list)
    languages: list[str] = Field(default_factory=list)
    has_text_layer: bool = False


class FieldValue(BaseModel):
    """A schema-extracted field with provenance.

    value      typed per the schema (number / integer / boolean / ISO-8601 date / enum / array of objects)
    raw_text   the exact source span as it appears in the document
    context    where the value sits: section heading, form label, table row label and column header
    groundings page + bbox (cell/word-level when available) + cropped snippet image
    verified   an independent check found the value in the grounded region
    type_ok    the value satisfied the schema's type/format/enum constraints
    """

    name: str
    value: Any = None
    raw_text: Optional[str] = None
    confidence: float = 0.0
    groundings: list[Grounding] = Field(default_factory=list)
    verified: bool = False
    type_ok: bool = True
    context: dict[str, Any] = Field(default_factory=dict)
    notes: Optional[str] = None


class ExtractionResult(BaseModel):
    schema_name: str = "default"
    fields: list[FieldValue] = Field(default_factory=list)
    data: dict[str, Any] = Field(default_factory=dict)   # plain JSON matching the requested schema

    def field(self, name: str) -> Optional[FieldValue]:
        return next((f for f in self.fields if f.name == name), None)


class Triple(BaseModel):
    subject: str
    predicate: str
    object: str
    subject_type: Optional[str] = None
    object_type: Optional[str] = None
    confidence: float = 1.0
    groundings: list[Grounding] = Field(default_factory=list)


class ParsedDocument(BaseModel):
    doc_id: str = Field(default_factory=lambda: new_id("doc"))
    filename: str = ""
    mime_type: str = ""
    n_pages: int = 0
    pages: list[Page] = Field(default_factory=list)
    markdown: str = ""
    languages: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    extraction: Optional[ExtractionResult] = None
    triples: list[Triple] = Field(default_factory=list)
    pipeline: dict[str, Any] = Field(default_factory=dict)   # which backends ran, timings
    tree: Optional["SectionNode"] = None                      # hierarchical structure (sections → chunks)
    status: str = "parsed"

    # convenience -------------------------------------------------------
    def all_chunks(self) -> list[Chunk]:
        return [c for p in self.pages for c in p.chunks]

    def chunk(self, chunk_id: str) -> Optional[Chunk]:
        return next((c for c in self.all_chunks() if c.id == chunk_id), None)

    def page(self, number: int) -> Optional[Page]:
        return next((p for p in self.pages if p.number == number), None)


class SectionNode(BaseModel):
    """Hierarchical view of the document: title → section headers → content chunks (ids)."""

    id: str = Field(default_factory=lambda: new_id("s"))
    title: str = ""
    level: int = 0
    page: int = 1
    chunk_ids: list[str] = Field(default_factory=list)
    children: list["SectionNode"] = Field(default_factory=list)


def build_tree(doc: "ParsedDocument") -> SectionNode:
    """Group chunks under their headings (title=level 0, section_header=level 1+), set Chunk.parent_id and
    Chunk.attributes['section'], and return the root node. Idempotent."""
    root = SectionNode(title=doc.filename or doc.doc_id, level=-1)
    stack: list[SectionNode] = [root]
    for page in doc.pages:
        for c in sorted(page.chunks, key=lambda c: c.reading_order):
            if c.type in (ChunkType.PAGE_HEADER, ChunkType.PAGE_FOOTER, ChunkType.PAGE_NUMBER):
                continue
            if c.type in (ChunkType.TITLE, ChunkType.SECTION_HEADER):
                level = 0 if c.type == ChunkType.TITLE else 1 + int(c.attributes.get("heading_depth", 0))
                while stack[-1].level >= level:
                    stack.pop()
                node = SectionNode(title=c.text.strip()[:200], level=level, page=c.page, chunk_ids=[c.id])
                stack[-1].children.append(node)
                stack.append(node)
                c.parent_id = stack[-2].id if len(stack) > 1 else None
                c.attributes["section"] = node.title
            else:
                stack[-1].chunk_ids.append(c.id)
                c.parent_id = stack[-1].id if stack[-1] is not root else None
                c.attributes["section"] = stack[-1].title if stack[-1] is not root else ""
    return root


class JobStatus(BaseModel):
    doc_id: str
    status: str = "queued"      # queued | running | parsed | failed
    stage: str = ""
    progress: float = 0.0
    error: Optional[str] = None
