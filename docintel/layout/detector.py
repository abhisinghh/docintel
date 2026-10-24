"""Layout analysis: classify/refine chunk types and detect visual regions (tables, figures, charts).

Two engines:

* ``HeuristicLayout`` – no model. Uses font statistics (from the PDF text
  layer), position on the page and simple regex patterns. Surprisingly solid
  for born-digital PDFs; weak on scans.
* ``DocLayoutYOLO`` – DocLayout-YOLO (DocStructBench weights) via the
  ``doclayout_yolo`` package. Detects: title, plain text, abandon (header/footer),
  figure, figure_caption, table, table_caption, table_footnote, isolate_formula,
  formula_caption. Boxes are merged with text chunks so that text inside a
  detected table/figure region is re-labelled and grouped.

A VLM backend already labels every element, so when the VLM is the OCR engine
the layout stage only has to (a) attach native text to VLM boxes and (b) run
the chart/map classifier on figure regions.
"""
from __future__ import annotations

import re
import statistics
from typing import Optional, Protocol

from PIL import Image

from ..schemas import BBox, Chunk, ChunkType


class LayoutEngine(Protocol):
    name: str

    def refine(self, image: Image.Image, chunks: list[Chunk], drawings: list[tuple] | None = None,
               page: int | None = None, doc_id: str | None = None) -> list[Chunk]: ...


LIST_RE = re.compile(r"^\s*([-•◦▪*]|\d+[.)]|[a-zA-Z][.)]|\(\d+\)|[ivxIVX]+[.)])\s+")
PAGE_NO_RE = re.compile(r"^\s*(page\s*)?\d{1,4}(\s*(of|/)\s*\d{1,4})?\s*$", re.I)
KV_RE = re.compile(r"^\s*[^:\n]{1,60}\s*[:：]\s*\S")


class HeuristicLayout:
    name = "heuristic"

    def refine(self, image: Image.Image, chunks: list[Chunk], drawings: list[tuple] | None = None,
               page: int | None = None, doc_id: str | None = None) -> list[Chunk]:
        W, H = image.size
        # 1. structural regions from geometry (vector drawings + text grids) -----------------
        regions = detect_grid_tables(chunks)
        if drawings:
            regions += detect_vector_regions(chunks, drawings, W, H)
        if regions:
            chunks = merge_regions(chunks, regions, page=page, doc_id=doc_id)
        # 2. per-chunk type refinement --------------------------------------------------------
        sizes = [c.attributes.get("font_size") for c in chunks if c.attributes.get("font_size")]
        body = statistics.median(sizes) if sizes else None
        for c in chunks:
            if c.type not in (ChunkType.TEXT, ChunkType.UNKNOWN):
                continue
            t = c.text.strip()
            fs = c.attributes.get("font_size")
            n_lines = t.count("\n") + 1
            big = bool(body and fs and fs >= body * 1.6 and len(t) < 160)
            if PAGE_NO_RE.match(t) and (c.bbox.y0 > 0.9 or c.bbox.y1 < 0.08):   # bare numbers only count at the page edges
                c.type = ChunkType.PAGE_NUMBER
            elif big:
                c.type = ChunkType.TITLE
            elif c.bbox.y1 < 0.07 and n_lines <= 2 and (not body or not fs or fs <= body * 1.1):
                c.type = ChunkType.PAGE_HEADER
            elif c.bbox.y0 > 0.93 and n_lines <= 2:
                c.type = ChunkType.PAGE_FOOTER
            elif body and fs and (fs >= body * 1.15 or c.attributes.get("bold")) and len(t) < 120 and n_lines <= 2:
                c.type = ChunkType.SECTION_HEADER
            elif LIST_RE.match(t):
                c.type = ChunkType.LIST_ITEM
            elif KV_RE.match(t) and n_lines <= 2 and len(t) < 160:
                c.type = ChunkType.FORM_FIELD
            elif re.match(r"^(fig(ure)?|table|chart|map)\s*\d+", t, re.I):
                c.type = ChunkType.CAPTION
            # likely table: many lines with 3+ aligned runs of whitespace
            elif n_lines >= 3 and sum(1 for ln in t.split("\n") if len(re.findall(r"\s{2,}", ln)) >= 2) >= n_lines * 0.6:
                c.type = ChunkType.TABLE
        return chunks


def _cluster_boxes(boxes: list[BBox], gap: float, gap_y: float | None = None) -> list[BBox]:
    """Union boxes that touch/overlap (within `gap` horizontally / `gap_y` vertically, normalised units)."""
    gap_y = gap if gap_y is None else gap_y
    regions: list[BBox] = []
    for b in sorted(boxes, key=lambda b: (b.y0, b.x0)):
        merged = False
        for i, r in enumerate(regions):
            if b.x0 <= r.x1 + gap and b.x1 >= r.x0 - gap and b.y0 <= r.y1 + gap_y and b.y1 >= r.y0 - gap_y:
                regions[i] = r.union(b)
                merged = True
                break
        if not merged:
            regions.append(b)
    # second pass: merge regions that grew into each other
    changed = True
    while changed:
        changed = False
        out: list[BBox] = []
        for r in regions:
            for i, o in enumerate(out):
                if r.x0 <= o.x1 + gap and r.x1 >= o.x0 - gap and r.y0 <= o.y1 + gap_y and r.y1 >= o.y0 - gap_y:
                    out[i] = o.union(r)
                    changed = True
                    break
            else:
                out.append(r)
        regions = out
    return regions


def _rows(boxes: list[BBox]) -> list[list[BBox]]:
    if not boxes:
        return []
    hs = sorted(b.y1 - b.y0 for b in boxes)
    tol = max(hs[len(hs) // 2] * 0.6, 0.004)
    rows: list[list[BBox]] = []
    for b in sorted(boxes, key=lambda b: b.y0):
        cy = (b.y0 + b.y1) / 2
        for row in rows:
            rcy = sum((x.y0 + x.y1) / 2 for x in row) / len(row)
            if abs(cy - rcy) <= tol:
                row.append(b)
                break
        else:
            rows.append([b])
    return rows


def _gutter_spans(rows: list[list[BBox]], bins: int = 300, thr: int | None = None) -> list[tuple[float, float]]:
    """Interior vertical whitespace gutters shared by all rows (thr = rows allowed to violate), as x-ranges."""
    if not rows:
        return []
    if thr is None:
        thr = len(rows) // 10
    occ = [0] * bins
    for r in rows:
        covered = set()
        for b in r:
            for k in range(int(b.x0 * bins), min(bins, int(b.x1 * bins) + 1)):
                covered.add(k)
        for k in covered:
            occ[k] += 1
    x_min = min(b.x0 for r in rows for b in r) * bins
    x_max = max(b.x1 for r in rows for b in r) * bins
    spans, run = [], 0
    for k in range(bins + 1):
        if k < bins and x_min < k < x_max and occ[k] <= thr:
            run += 1
        else:
            if run >= 2:
                spans.append(((k - run) / bins, k / bins))
            run = 0
    return spans


def _gutters(rows: list[list[BBox]], bins: int = 300, thr: int | None = None) -> int:
    return len(_gutter_spans(rows, bins, thr))


def _row_fill(row: list[BBox], spans: list[tuple[float, float]]) -> float:
    """Fraction of the columns delimited by `spans` that this row has at least one box in."""
    edges = [0.0] + [(a + b) / 2 for a, b in spans] + [1.0]
    filled = 0
    for a, b in zip(edges, edges[1:]):
        if any(a <= (x.x0 + x.x1) / 2 < b for x in row):
            filled += 1
    return filled / (len(edges) - 1)


def _cell_density_ok(window: list[list[BBox]], gutters: int, max_words_per_cell: float = 4.0) -> bool:
    """Table cells hold a few words; prose columns hold many. Rejects multi-column text that merely has gutters."""
    words = sum(len(r) for r in window) / max(1, len(window))
    return words / (gutters + 1) <= max_words_per_cell


def _window_is_table(window: list[list[BBox]], min_rows: int, min_gutters: int) -> bool:
    """Header row may be centred/bold (breaks gutters), so gutters must hold strictly across the *body*;
    the header must have a cell count compatible with the body so a prose line cannot head a table."""
    if len(window) < min_rows:
        return False
    body = window[1:]
    spans = _gutter_spans(body, thr=0)
    g = len(spans)
    if g < min_gutters or not _cell_density_ok(body, g):
        return False
    # every body row must occupy at least half of the columns (a "Key: value" line above a table fills one)
    if any(_row_fill(r, spans) < 0.5 for r in body):
        return False
    if _row_fill(window[0], spans) < 0.5:
        return False
    # header words sitting *inside* the body's whitespace gutters mean the row is prose, not a header
    in_gutter = sum(1 for b in window[0] if any(a <= (b.x0 + b.x1) / 2 <= z for a, z in spans))
    if in_gutter * 2 >= len(window[0]):
        return False
    med = sorted(len(r) for r in body)[len(body) // 2]
    return 0.5 * med <= len(window[0]) <= 1.5 * med + 1


def find_table_windows(boxes: list[BBox], min_rows: int = 3, min_gutters: int = 2) -> list[BBox]:
    """Maximal contiguous runs of word-rows that share ≥min_gutters whitespace columns → table regions."""
    rows = _rows(boxes)
    out: list[BBox] = []
    i = 0
    while i < len(rows):
        j = i
        best = None
        while j < len(rows):
            window = [r for r in rows[i:j + 1] if len(r) >= 2]
            if _window_is_table(window, min_rows, min_gutters):
                best = j
            elif len(window) >= min_rows:
                break
            j += 1
        if best is not None:
            win = rows[i:best + 1]
            # trim sparse leading/trailing rows (titles, totals) that merely sit in the same columns
            mx = max(len(r) for r in win)
            while len(win) > min_rows and len(win[0]) < mx * 0.5:
                win = win[1:]
            while len(win) > min_rows and len(win[-1]) < mx * 0.5:
                win = win[:-1]
            cells = [b for r in win for b in r]
            box = cells[0]
            for b in cells[1:]:
                box = box.union(b)
            out.append(box)
            i = best + 1
        else:
            i += 1
    return out


def detect_grid_tables(chunks: list[Chunk]) -> list[tuple[ChunkType, BBox, float]]:
    """Borderless tables: runs of word-rows that share vertical whitespace gutters."""
    cand = [c for c in chunks if c.type == ChunkType.TEXT and c.words and
            all(len(ln) < 90 for ln in c.text.split("\n"))]
    out = []
    # rows of rendered tables are often 0.02-0.03 page-heights apart, so cluster generously in y;
    # find_table_windows then isolates the tabular rows inside each cluster
    for region in _cluster_boxes([c.bbox for c in cand], gap=0.012, gap_y=0.035):
        inside = [c for c in cand if _contained(c.bbox, region)]
        words = [w.bbox for c in inside for w in c.words]
        for box in find_table_windows(words):
            out.append((ChunkType.TABLE, box, 0.6))
    return out


def detect_vector_regions(chunks: list[Chunk], drawings: list[tuple], W: int, H: int) -> list[tuple[ChunkType, BBox, float]]:
    """Clusters of vector paths = charts/diagrams, or ruled tables when full of gridded words."""
    boxes = [BBox.from_abs(*d, W, H) for d in drawings]
    boxes = [b for b in boxes if b.area() < 0.9]   # ignore page-sized backgrounds
    out = []
    for region in _cluster_boxes(boxes, gap=0.02):      # bars/slices of a chart are often ~1-2 % apart
        if region.area() < 0.01:
            continue
        n_paths = sum(1 for b in boxes if _contained(b, region))
        inside = [c for c in chunks if _contained(c.bbox, region) and c.type == ChunkType.TEXT]
        words = [w.bbox for c in inside for w in c.words]
        rows = [r for r in _rows(words) if len(r) >= 2]
        if _window_is_table(rows, 3, 2):
            out.append((ChunkType.TABLE, region, 0.7))
        elif n_paths >= 3:
            out.append((ChunkType.FIGURE, region, 0.6))
    return out


YOLO_LABELS = {
    "title": ChunkType.TITLE, "plain text": ChunkType.TEXT, "abandon": ChunkType.PAGE_HEADER,
    "figure": ChunkType.FIGURE, "figure_caption": ChunkType.CAPTION, "table": ChunkType.TABLE,
    "table_caption": ChunkType.CAPTION, "table_footnote": ChunkType.FOOTNOTE,
    "isolate_formula": ChunkType.FORMULA, "formula_caption": ChunkType.CAPTION,
}


class DocLayoutYOLO:
    """DocLayout-YOLO region detector (pip install doclayout-yolo huggingface-hub)."""

    name = "doclayout_yolo"

    def __init__(self, model_path: Optional[str] = None, conf: float = 0.25, device: str = "cpu"):
        from doclayout_yolo import YOLOv10  # type: ignore
        if model_path is None:
            from huggingface_hub import hf_hub_download
            model_path = hf_hub_download("juliozhao/DocLayout-YOLO-DocStructBench",
                                         "doclayout_yolo_docstructbench_imgsz1024.pt")
        self.model = YOLOv10(model_path)
        self.conf = conf
        self.device = device

    def detect(self, image: Image.Image) -> list[tuple[ChunkType, BBox, float]]:
        res = self.model.predict(image, imgsz=1024, conf=self.conf, device=self.device, verbose=False)[0]
        W, H = image.size
        out = []
        for box, cls, conf in zip(res.boxes.xyxy.tolist(), res.boxes.cls.tolist(), res.boxes.conf.tolist()):
            label = res.names[int(cls)]
            out.append((YOLO_LABELS.get(label, ChunkType.TEXT), BBox.from_abs(*box, W, H), float(conf)))
        return out

    def refine(self, image: Image.Image, chunks: list[Chunk], drawings: list[tuple] | None = None,
               page: int | None = None, doc_id: str | None = None) -> list[Chunk]:
        regions = self.detect(image)
        return merge_regions(chunks, regions, page=page, doc_id=doc_id)


def merge_regions(chunks: list[Chunk], regions: list[tuple[ChunkType, BBox, float]],
                  page: int | None = None, doc_id: str | None = None) -> list[Chunk]:
    """Group text chunks that fall inside detected table/figure regions into one region chunk."""
    if page is None:
        page = chunks[0].page if chunks else 1
    if doc_id is None:
        doc_id = chunks[0].doc_id if chunks else ""
    visual = _dedupe_regions([(t, b, c) for t, b, c in regions if t in (ChunkType.TABLE, ChunkType.FIGURE, ChunkType.FORMULA)])
    textual = [(t, b, c) for t, b, c in regions if t not in (ChunkType.TABLE, ChunkType.FIGURE, ChunkType.FORMULA)]
    out: list[Chunk] = []
    consumed: set[str] = set()
    for rtype, rbox, rconf in visual:
        inside = [c for c in chunks if c.id not in consumed and _contained(c.bbox, rbox)]
        for c in inside:
            consumed.add(c.id)
        proto = inside[0] if inside else None
        if rtype == ChunkType.TABLE and not inside:
            continue   # a ruled box with nothing inside is not a table
        text = "\n".join(c.text for c in inside)
        words = [w for c in inside for w in c.words]
        out.append(Chunk(
            doc_id=doc_id, page=page, type=rtype, text=text,
            bbox=rbox, confidence=rconf, source=proto.source if proto else "native", words=words,
            reading_order=min((c.reading_order for c in inside), default=9999),
        ))
    for c in chunks:
        if c.id in consumed:
            continue
        # relabel using the best-overlapping textual region
        best = max(textual, key=lambda r: r[1].iou(c.bbox), default=None)
        if best and best[1].iou(c.bbox) > 0.3 and c.type == ChunkType.TEXT:
            c.type = best[0]
        out.append(c)
    out.sort(key=lambda c: (c.page, c.bbox.y0, c.bbox.x0))
    for i, c in enumerate(out):
        c.reading_order = i
    return out


def _dedupe_regions(regions):
    """Merge same-type regions that overlap (e.g. a ruled border and the word grid inside it)."""
    out: list = []
    for t, b, c in sorted(regions, key=lambda r: -r[1].area()):
        for i, (t2, b2, c2) in enumerate(out):
            inter = max(0.0, min(b.x1, b2.x1) - max(b.x0, b2.x0)) * max(0.0, min(b.y1, b2.y1) - max(b.y0, b2.y0))
            if inter / max(min(b.area(), b2.area()), 1e-9) > 0.5:
                if t == t2 or t2 == ChunkType.TABLE:      # same kind (or a ruled box around a word grid): keep the larger
                    out[i] = (t2, b2, max(c, c2))
                    break
                if t == ChunkType.TABLE:                  # a table inside a figure region: keep both, table wins its area
                    continue
                break
        else:
            out.append((t, b, c))
    return out


def _contained(inner: BBox, outer: BBox, tol: float = 0.6) -> bool:
    ix0, iy0 = max(inner.x0, outer.x0), max(inner.y0, outer.y0)
    ix1, iy1 = min(inner.x1, outer.x1), min(inner.y1, outer.y1)
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    return inner.area() > 0 and inter / inner.area() >= tol


def build_layout(settings) -> LayoutEngine:
    if settings.layout.engine == "doclayout_yolo":
        return DocLayoutYOLO(settings.layout.model_path, settings.layout.conf_threshold)
    return HeuristicLayout()


def reading_order(chunks: list[Chunk], column_gap: float = 0.08) -> list[Chunk]:
    """Simple two-column aware reading order: sort by column, then y, then x."""
    if not chunks:
        return chunks
    # two columns exist when a vertical band in the middle of the page is crossed by no (narrow) chunk
    narrow = [c for c in chunks if c.bbox.x1 - c.bbox.x0 <= 0.6]
    split = None
    if len(narrow) >= 4:
        best_gap, best_x = 0.0, None
        edges = sorted({c.bbox.x1 for c in narrow} | {c.bbox.x0 for c in narrow})
        for a, b in zip(edges, edges[1:]):
            mid = (a + b) / 2
            if 0.3 < mid < 0.7 and b - a > best_gap and not any(c.bbox.x0 < mid < c.bbox.x1 for c in narrow):
                best_gap, best_x = b - a, mid
        if best_x is not None and best_gap >= column_gap * 0.5 and \
                sum(1 for c in narrow if c.bbox.x0 >= best_x) >= 2 and sum(1 for c in narrow if c.bbox.x1 <= best_x) >= 2:
            split = best_x
    def key(c: Chunk):
        col = 0 if split is None or c.bbox.x0 < split or c.bbox.x1 - c.bbox.x0 > 0.6 else 1
        return (col, round(c.bbox.y0, 2), c.bbox.x0)
    ordered = sorted(chunks, key=key)
    for i, c in enumerate(ordered):
        c.reading_order = i
    return ordered
