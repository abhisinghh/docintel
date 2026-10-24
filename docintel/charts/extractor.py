"""Charts, maps and infographics → structured data + natural-language summary.

Pure-OCR systems throw these away; a RAG system then cannot answer "what was
revenue in Q3 according to the chart on page 4". We therefore:

1. Classify every figure region (chart / map / infographic / photo) — with a
   VLM when available, otherwise with cheap image statistics.
2. For charts run *chart-to-table*: ask the VLM for a JSON with chart type,
   axes, series and the readable data points (the ChartQA / ChartQAPro task).
3. For maps/infographics ask for a structured description: named regions,
   legend categories, values, callouts.
4. Store the result in ``Chunk.chart`` and write a text rendering into
   ``Chunk.text`` so the chunk is searchable and embeddable like any paragraph.

Everything here is grounded: the figure's bbox is the grounding for every
value read from it.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from PIL import Image

from ..schemas import ChartData, ChartSeries, Chunk, ChunkType, TableCell, TableData

CLASSIFY_PROMPT = (
    'Classify this image region. Answer with ONE word from: chart, map, infographic, diagram, photo, logo, '
    'signature, stamp, table, text.'
)

CHART_PROMPT = """Extract the data from this chart. Return ONLY JSON:
{"chart_type": "bar|line|pie|scatter|area|stacked_bar|other",
 "title": "...", "x_label": "...", "y_label": "...",
 "series": [{"name": "...", "x": ["..."], "y": [numbers]}],
 "summary": "2-3 sentences describing what the chart shows, including min/max/trend and any annotated values"}
Read values as precisely as the axis allows. Keep labels in their original language."""

MAP_PROMPT = """Describe this map/infographic for a question-answering system. Return ONLY JSON:
{"kind": "map|infographic|diagram",
 "title": "...",
 "legend": [{"label": "...", "meaning": "..."}],
 "entries": [{"name": "region/item", "value": "value or category", "note": "..."}],
 "summary": "3-5 sentences: what is shown, the extremes, notable patterns, any numbers printed on it"}
Keep names in their original language/script."""


def _heuristic_kind(img: Image.Image) -> str:
    """Charts tend to have few dominant colours and lots of white; photos have many colours."""
    small = img.convert("RGB").resize((64, 64))
    colors = small.getcolors(64 * 64) or []
    n_colors = len(colors)
    white = sum(n for n, c in colors if min(c) > 235) / (64 * 64)
    if n_colors < 400 and white > 0.35:
        return "chart"
    if n_colors < 1500:
        return "infographic"
    return "photo"


def crop_region(image: Image.Image, chunk: Chunk, out_dir: Path) -> Path:
    x0, y0, x1, y1 = chunk.bbox.to_abs(*image.size)
    crop = image.crop((int(x0), int(y0), int(max(x1, x0 + 1)), int(max(y1, y0 + 1))))
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"{chunk.id}.png"
    crop.save(p)
    return p


def _chart_to_table(cd: ChartData) -> Optional[TableData]:
    if not cd.series:
        return None
    xs = cd.series[0].x
    cells = [TableCell(row=0, col=0, text=cd.x_label or "x", is_header=True)]
    for j, s in enumerate(cd.series):
        cells.append(TableCell(row=0, col=j + 1, text=s.name or f"series{j+1}", is_header=True))
    for i, x in enumerate(xs):
        cells.append(TableCell(row=i + 1, col=0, text=str(x)))
        for j, s in enumerate(cd.series):
            v = s.y[i] if i < len(s.y) else ""
            cells.append(TableCell(row=i + 1, col=j + 1, text=str(v)))
    return TableData(cells=cells, n_rows=len(xs) + 1, n_cols=len(cd.series) + 1)


def extract_visuals(image: Image.Image, chunks: list[Chunk], crops_dir: Path, vlm=None) -> list[Chunk]:
    from ..ocr.backends import _extract_json

    for c in chunks:
        if c.type not in (ChunkType.FIGURE, ChunkType.CHART, ChunkType.MAP, ChunkType.INFOGRAPHIC):
            continue
        if c.bbox.area() < 0.004:   # tiny icons / bullets
            continue
        c.image_path = str(crop_region(image, c, crops_dir))
        crop = Image.open(c.image_path)
        kind = None
        if vlm is not None:
            try:
                kind = vlm.chat(crop, CLASSIFY_PROMPT, max_tokens=8).strip().lower().split()[0]
            except Exception as e:  # pragma: no cover
                c.attributes["classify_error"] = str(e)
        if kind is None:
            kind = _heuristic_kind(crop)
        c.attributes["visual_kind"] = kind
        if kind == "chart":
            c.type = ChunkType.CHART
        elif kind == "map":
            c.type = ChunkType.MAP
        elif kind in ("infographic", "diagram"):
            c.type = ChunkType.INFOGRAPHIC
        if vlm is None:
            if not c.text:
                c.text = f"[{kind}]"
            continue
        try:
            if c.type == ChunkType.CHART:
                data = _extract_json(vlm.chat(crop, CHART_PROMPT))
                cd = ChartData(
                    chart_type=data.get("chart_type", "unknown"), title=data.get("title"),
                    x_label=data.get("x_label"), y_label=data.get("y_label"), summary=data.get("summary"),
                    series=[ChartSeries(name=s.get("name", ""), x=s.get("x", []), y=s.get("y", []))
                            for s in data.get("series", []) if isinstance(s, dict)],
                )
                cd.raw_table = _chart_to_table(cd)
                c.chart = cd
                parts = [f"Chart: {cd.title}" if cd.title else "Chart", cd.summary or ""]
                if cd.raw_table:
                    parts.append(cd.raw_table.to_markdown())
                c.text = "\n".join(p for p in parts if p)
            elif c.type in (ChunkType.MAP, ChunkType.INFOGRAPHIC):
                data = _extract_json(vlm.chat(crop, MAP_PROMPT))
                c.attributes["structured"] = data
                lines = [f"{data.get('kind', kind).title()}: {data.get('title') or ''}".strip(), data.get("summary", "")]
                for e in data.get("entries", []):
                    if isinstance(e, dict):
                        lines.append(f"- {e.get('name')}: {e.get('value')} {e.get('note') or ''}".rstrip())
                c.text = "\n".join(l for l in lines if l)
            else:
                c.text = c.text or vlm.chat(crop, "Describe this image in two sentences for a document index.", max_tokens=200)
        except Exception as e:  # pragma: no cover
            c.attributes["visual_error"] = str(e)
            c.text = c.text or f"[{kind}]"
    return chunks


def chart_summary_json(c: Chunk) -> str:
    return json.dumps(c.chart.model_dump(), ensure_ascii=False) if c.chart else "{}"
