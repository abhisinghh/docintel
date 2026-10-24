"""Export to the `ragdoc` ingestion contract (github.com/abhisinghh/RAG_over_doc).

ragdoc's `IngestionPipeline.ingest_parsed(ParsedDocument)` runs figures, tables,
chunking, contextual headers, embedding, indexing and graph extraction on a
pre-parsed document.  Its pydantic `ParsedDocument` looks like:

    {doc_id, source, title, sha256, kind, num_pages, meta, blocks: [Block]}
    Block = {text, page, kind, modality, heading_path, asset, start_s, end_s, line_start, line_end, meta}
    Block.kind     ∈ text | heading | table | list | caption | figure | segment | symbol
    Block.modality ∈ text | table | image | chart | audio | video | code | service

We map our layout-aware chunks onto that, and keep *visual grounding* inside
each block's `meta` (page, normalised bbox, chunk_id, confidence, languages),
which ragdoc copies into chunk metadata → citations can point back to a region.

Optional: pre-extracted KG triples as ragdoc `Node` / `GEdge` records
(`graph.jsonl`) for `GraphStore.upsert_nodes / upsert_edges`.

Usage inside ragdoc:
    from ragdoc.types import ParsedDocument as RD
    from ragdoc.ingestion.pipeline import IngestionPipeline
    pdoc = RD.model_validate(json.load(open("ragdoc_document.json")))
    IngestionPipeline(settings).ingest_parsed(pdoc)
"""
from __future__ import annotations

import base64
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from ..schemas import ChunkType, ParsedDocument

KIND_MAP = {
    ChunkType.TITLE: ("heading", "text"), ChunkType.SECTION_HEADER: ("heading", "text"),
    ChunkType.TEXT: ("text", "text"), ChunkType.LIST_ITEM: ("list", "text"),
    ChunkType.CAPTION: ("caption", "text"), ChunkType.FOOTNOTE: ("text", "text"),
    ChunkType.FORMULA: ("text", "text"), ChunkType.CODE: ("symbol", "code"),
    ChunkType.HANDWRITING: ("text", "text"), ChunkType.FORM_FIELD: ("text", "text"),
    ChunkType.TABLE: ("table", "table"), ChunkType.FIGURE: ("figure", "image"),
    ChunkType.CHART: ("figure", "chart"), ChunkType.MAP: ("figure", "image"),
    ChunkType.INFOGRAPHIC: ("figure", "image"),
}
SKIP = {ChunkType.PAGE_HEADER, ChunkType.PAGE_FOOTER, ChunkType.PAGE_NUMBER, ChunkType.SIGNATURE,
        ChunkType.STAMP, ChunkType.BARCODE}


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:40] or "doc"


def to_ragdoc(doc: ParsedDocument, source_path: str | Path | None = None, embed_images: bool = True) -> dict[str, Any]:
    sha = hashlib.sha256(Path(source_path).read_bytes()).hexdigest() if source_path and Path(source_path).exists() \
        else hashlib.sha256(doc.doc_id.encode()).hexdigest()
    stem = Path(doc.filename).stem
    title = next((c.text.strip() for c in doc.all_chunks() if c.type == ChunkType.TITLE and c.text.strip()), stem)
    heading_path: list[str] = []
    blocks: list[dict[str, Any]] = []
    for p in doc.pages:
        for c in sorted(p.chunks, key=lambda c: c.reading_order):
            if c.type in SKIP:
                continue
            kind, modality = KIND_MAP.get(c.type, ("text", "text"))
            text = c.text.strip()
            if kind == "heading":
                if not text:
                    continue
                level = 0 if c.type == ChunkType.TITLE else 1
                heading_path = heading_path[:level] + [text[:120]]
            if kind == "table" and c.table:
                text = c.table.to_markdown()
            meta: dict[str, Any] = {
                "page": c.page, "bbox": [round(c.bbox.x0, 4), round(c.bbox.y0, 4), round(c.bbox.x1, 4), round(c.bbox.y1, 4)],
                "chunk_id": c.id, "confidence": c.confidence, "languages": c.languages, "scripts": c.scripts,
                "chunk_type": c.type.value, "source": c.source,
            }
            if kind == "figure":
                meta["figure_kind"] = c.attributes.get("visual_kind", c.type.value)
                cap = next((o.text.strip() for o in p.chunks if o.type == ChunkType.CAPTION
                            and abs(o.bbox.y0 - c.bbox.y1) < 0.03), None)
                if cap:
                    meta["caption"] = cap
                if c.chart:
                    meta["chart"] = c.chart.model_dump()
                    text = text or (c.chart.summary or "")
                if embed_images and c.image_path and Path(c.image_path).exists():
                    meta["_image"] = base64.b64encode(Path(c.image_path).read_bytes()).decode()
            if not text and kind != "figure":
                continue
            blocks.append({
                "text": text, "page": c.page, "kind": kind, "modality": modality,
                "heading_path": list(heading_path), "asset": None,
                "start_s": None, "end_s": None, "line_start": None, "line_end": None, "meta": meta,
            })
    return {
        "doc_id": f"{_slug(stem)}-{sha[:8]}", "source": doc.filename, "title": title, "sha256": sha,
        "kind": "document", "num_pages": doc.n_pages, "duration_s": None,
        "meta": {"mime": doc.mime_type, "languages": doc.languages, "parser": "docintel/0.1.0",
                 "docintel_doc_id": doc.doc_id, "pipeline": {k: v for k, v in doc.pipeline.items() if k != "agent_trace"}},
        "blocks": blocks,
    }


def to_ragdoc_graph(doc: ParsedDocument, ragdoc_doc_id: str) -> list[dict[str, Any]]:
    """ragdoc Node / GEdge records. Entity ids are `entity:<normalised name>`; weights 1-10."""
    norm = lambda s: re.sub(r"\s+", " ", s.strip().lower())  # noqa: E731
    nodes: dict[str, dict[str, Any]] = {}
    edges: list[dict[str, Any]] = []
    for t in doc.triples:
        sid, oid = f"entity:{norm(t.subject)}", f"entity:{norm(t.object)}"
        src_ids = [g.chunk_id for g in t.groundings if g.chunk_id]
        for nid, name, typ in ((sid, t.subject, t.subject_type), (oid, t.object, t.object_type)):
            n = nodes.setdefault(nid, {"record": "node", "id": nid, "name": name, "type": typ or "concept",
                                       "description": "", "source_ids": [], "doc_ids": [ragdoc_doc_id], "data": {}})
            n["source_ids"] = sorted(set(n["source_ids"]) | set(src_ids))
        edges.append({"record": "edge", "source": sid, "target": oid,
                      "relation": re.sub(r"[^a-z0-9_]", "_", t.predicate.lower())[:60] or "related_to",
                      "description": "", "weight": max(1, min(10, round(t.confidence * 10))),
                      "source_ids": src_ids, "data": {"groundings": [{"page": g.page, "bbox": g.bbox.model_dump()} for g in t.groundings]}})
    return list(nodes.values()) + edges


def export_ragdoc(doc: ParsedDocument, out_dir: str | Path, source_path: str | Path | None = None) -> dict[str, str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rd = to_ragdoc(doc, source_path)
    (out / "ragdoc_document.json").write_text(json.dumps(rd, ensure_ascii=False), "utf-8")
    (out / "ragdoc_graph.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in to_ragdoc_graph(doc, rd["doc_id"])) + "\n", "utf-8")
    return {"ragdoc_document.json": str(out / "ragdoc_document.json"), "ragdoc_graph.jsonl": str(out / "ragdoc_graph.jsonl")}
