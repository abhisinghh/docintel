"""Table structure recognition.

Strategy (in order of preference):
1. The VLM already returned an HTML table → parsed in the OCR backend.
2. Born-digital PDF: cluster the word boxes inside the table region into
   rows/columns (``words_to_table``) — exact, zero-hallucination, grounded cells.
3. Ask the VLM to convert the cropped table image to HTML (``vlm_table``).
4. Optional: PaddleOCR PP-StructureV3 / Docling TableFormer (wrappers left as hooks).

The TEDS metric in ``docintel.eval.metrics`` scores the HTML against ground truth.
"""
from __future__ import annotations

import re
from typing import Optional

from PIL import Image

from ..schemas import BBox, Chunk, ChunkType, TableCell, TableData, Word


def _cluster_1d(values: list[float], gap: float) -> list[list[int]]:
    """Group sorted indices whose values are within `gap` of the running cluster."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    clusters: list[list[int]] = []
    for i in order:
        if clusters and values[i] - values[clusters[-1][-1]] <= gap:
            clusters[-1].append(i)
        else:
            clusters.append([i])
    return clusters


def words_to_table(words: list[Word], region: Optional[BBox] = None) -> Optional[TableData]:
    """Infer rows/columns from word geometry (ruled or unruled tables with a text layer).

    1. rows   : cluster words by vertical centre
    2. cells  : inside a row, merge horizontally adjacent words (gap ≤ ~1 space) → "Unit price" stays one cell
    3. columns: cluster cells by horizontal overlap across rows (handles left-, centre- and right-aligned columns)
    """
    if len(words) < 4:
        return None
    hs = [w.bbox.y1 - w.bbox.y0 for w in words]
    med_h = sorted(hs)[len(hs) // 2] or 0.01
    ys = [(w.bbox.y0 + w.bbox.y1) / 2 for w in words]
    row_clusters = _cluster_1d(ys, med_h * 0.6)
    # ---- cells -------------------------------------------------------------------------------
    space = med_h * 0.45          # one inter-word space is ~0.25-0.35 of the line height; cell gaps are wider
    rows_cells: list[list[tuple[BBox, str]]] = []
    for idx in row_clusters:
        ws = sorted((words[i] for i in idx), key=lambda w: w.bbox.x0)
        cells: list[tuple[BBox, str]] = []
        for w in ws:
            if cells and w.bbox.x0 - cells[-1][0].x1 <= space:
                b, t = cells[-1]
                cells[-1] = (b.union(w.bbox), f"{t} {w.text}")
            else:
                cells.append((w.bbox, w.text))
        rows_cells.append(cells)
    # ---- columns by overlap ---------------------------------------------------------------------
    # Seed columns from the *body* rows (the header is often a wide/centred cell spanning sub-columns);
    # extents grow only by union with same-column cells, and two cells of one row can never share a column.
    cols: list[list[float]] = []          # [x0, x1] running extents
    col_of: dict[tuple[int, int], int] = {}
    order_rows = list(range(1, len(rows_cells))) + [0] if len(rows_cells) > 1 else [0]
    for r in order_rows:
        cells = rows_cells[r]
        used: set[int] = set()
        for c_i, (b, _) in enumerate(cells):
            best, best_ov = None, 0.0
            for k, (cx0, cx1) in enumerate(cols):
                if k in used:
                    continue
                ov = min(b.x1, cx1) - max(b.x0, cx0)
                if ov > best_ov:
                    best, best_ov = k, ov
            if best is None or best_ov <= 0:
                cols.append([b.x0, b.x1])
                best = len(cols) - 1
            elif r != 0:                      # header cells do not widen body columns
                cols[best][0], cols[best][1] = min(cols[best][0], b.x0), max(cols[best][1], b.x1)
            used.add(best)
            col_of[(r, c_i)] = best
    order = sorted(range(len(cols)), key=lambda k: cols[k][0])
    rank = {k: i for i, k in enumerate(order)}
    n_rows, n_cols = len(rows_cells), len(cols)
    if n_rows < 2 or n_cols < 2:
        return None
    out: list[TableCell] = []
    for r, cells in enumerate(rows_cells):
        for c_i, (b, t) in enumerate(cells):
            out.append(TableCell(row=r, col=rank[col_of[(r, c_i)]], text=t, is_header=(r == 0), bbox=b))
    td = TableData(cells=out, n_rows=n_rows, n_cols=n_cols)
    td.html = td.to_html()
    return td


VLM_TABLE_PROMPT = (
    "Convert this table image to a complete HTML <table>. Preserve all rows, columns, merged cells "
    "(use rowspan/colspan), header rows (<th>) and the original text in its original language/script. "
    "Return only the HTML."
)


def vlm_table(vlm, image: Image.Image, chunk: Chunk) -> Optional[TableData]:
    from ..ocr.backends import _parse_html_table
    html = vlm.read_region(image, chunk.bbox, VLM_TABLE_PROMPT)
    m = re.search(r"<table.*?</table>", html, re.S | re.I)
    return _parse_html_table(m.group(0)) if m else None


def extract_tables(image: Image.Image, chunks: list[Chunk], vlm=None) -> list[Chunk]:
    for c in chunks:
        if c.type != ChunkType.TABLE or c.table is not None:
            continue
        td = words_to_table(c.words, c.bbox) if c.words else None
        if td is None and vlm is not None:
            try:
                td = vlm_table(vlm, image, c)
            except Exception as e:  # pragma: no cover
                c.attributes["table_error"] = str(e)
        if td is not None:
            c.table = td
            c.text = td.to_markdown() or c.text
    return chunks
