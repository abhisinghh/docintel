"""Export for downstream RAG + knowledge-graph systems (e.g. abhisinghh/RAG_over_doc).

Produces:
* ``chunks.jsonl``  – one record per retrieval unit: {id, doc_id, page, type, text, bbox, languages,
                      metadata}. Shape matches LangChain / LlamaIndex ``Document(page_content, metadata)``.
* ``triples.jsonl`` – knowledge-graph edges with provenance.
* ``graph.cypher``  – Neo4j MERGE statements (nodes: Document, Page, Chunk, Entity).
* ``document.md``   – full markdown.
* ``document.json`` – the complete ParsedDocument.

Chunking policy: layout-aware. Each heading starts a new section; paragraphs
are merged until ``max_chars``; tables, charts, figures and form fields are
*never* split and keep their structured payload in metadata; every record
keeps page + bbox so the answer system can show "where it came from".
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..schemas import ChunkType, ParsedDocument

ATOMIC = {ChunkType.TABLE, ChunkType.CHART, ChunkType.MAP, ChunkType.INFOGRAPHIC, ChunkType.FIGURE, ChunkType.FORM_FIELD}
SKIP = {ChunkType.PAGE_HEADER, ChunkType.PAGE_FOOTER, ChunkType.PAGE_NUMBER}


def rag_chunks(doc: ParsedDocument, max_chars: int = 1200) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    section = ""
    buf: list[Any] = []

    def flush():
        nonlocal buf
        if not buf:
            return
        text = "\n\n".join(c.text.strip() for c in buf if c.text.strip())
        if text:
            b = buf[0].bbox
            for c in buf[1:]:
                b = b.union(c.bbox)
            records.append({
                "id": f"{doc.doc_id}:{buf[0].id}", "doc_id": doc.doc_id, "page": buf[0].page,
                "type": "text", "text": (f"[{section}]\n" if section else "") + text,
                "bbox": [round(b.x0, 4), round(b.y0, 4), round(b.x1, 4), round(b.y1, 4)],
                "languages": sorted({l for c in buf for l in c.languages}),
                "metadata": {"source": doc.filename, "section": section, "chunk_ids": [c.id for c in buf],
                             "pages": sorted({c.page for c in buf})},
            })
        buf = []

    for p in doc.pages:
        for c in sorted(p.chunks, key=lambda c: c.reading_order):
            if c.type in SKIP or not c.text.strip():
                continue
            if c.type in (ChunkType.TITLE, ChunkType.SECTION_HEADER):
                flush()
                section = c.text.strip()[:120]
                continue
            if c.type in ATOMIC:
                flush()
                meta: dict[str, Any] = {"source": doc.filename, "section": section, "chunk_ids": [c.id],
                                        "pages": [c.page], "image_path": c.image_path}
                if c.table:
                    meta["table"] = {"html": c.table.to_html(), "grid": c.table.to_grid()}
                if c.chart:
                    meta["chart"] = c.chart.model_dump()
                if "structured" in c.attributes:
                    meta["structured"] = c.attributes["structured"]
                records.append({
                    "id": f"{doc.doc_id}:{c.id}", "doc_id": doc.doc_id, "page": c.page, "type": c.type.value,
                    "text": (f"[{section}]\n" if section else "") + c.text.strip(),
                    "bbox": [round(c.bbox.x0, 4), round(c.bbox.y0, 4), round(c.bbox.x1, 4), round(c.bbox.y1, 4)],
                    "languages": c.languages, "metadata": meta,
                })
                continue
            if buf and (sum(len(x.text) for x in buf) + len(c.text) > max_chars or buf[0].page != c.page):
                flush()
            buf.append(c)
    flush()
    return records


def neo4j_cypher(doc: ParsedDocument) -> str:
    esc = lambda s: json.dumps(str(s), ensure_ascii=False)  # noqa: E731
    lines = [f"MERGE (d:Document {{id: {esc(doc.doc_id)}}}) SET d.name = {esc(doc.filename)}, d.pages = {doc.n_pages};"]
    for p in doc.pages:
        lines.append(f"MERGE (p:Page {{id: {esc(doc.doc_id + ':' + str(p.number))}}}) SET p.number = {p.number} "
                     f"WITH p MATCH (d:Document {{id: {esc(doc.doc_id)}}}) MERGE (d)-[:HAS_PAGE]->(p);")
    for t in doc.triples:
        g = t.groundings[0] if t.groundings else None
        prov = (f", page: {g.page}, bbox: {json.dumps([g.bbox.x0, g.bbox.y0, g.bbox.x1, g.bbox.y1])}" if g else "")
        rel = "".join(ch if ch.isalnum() else "_" for ch in t.predicate.upper())[:60] or "RELATED_TO"
        lines.append(
            f"MERGE (s:Entity {{name: {esc(t.subject)}}}) MERGE (o:Entity {{name: {esc(t.object)}}}) "
            f"MERGE (s)-[r:{rel} {{doc_id: {esc(doc.doc_id)}{prov}}}]->(o) SET r.confidence = {t.confidence};"
        )
    return "\n".join(lines) + "\n"


def export_all(doc: ParsedDocument, out_dir: str | Path) -> dict[str, str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {}
    (out / "chunks.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rag_chunks(doc)) + "\n", "utf-8")
    (out / "triples.jsonl").write_text(
        "\n".join(json.dumps({**t.model_dump(exclude={"groundings"}),
                              "groundings": [{"page": g.page, "bbox": g.bbox.model_dump(), "chunk_id": g.chunk_id}
                                             for g in t.groundings]}, ensure_ascii=False) for t in doc.triples) + "\n", "utf-8")
    (out / "graph.cypher").write_text(neo4j_cypher(doc), "utf-8")
    (out / "document.md").write_text(doc.markdown, "utf-8")
    (out / "document.json").write_text(doc.model_dump_json(indent=1), "utf-8")
    if doc.extraction:
        (out / "extraction.json").write_text(json.dumps(doc.extraction.model_dump(), ensure_ascii=False, indent=1), "utf-8")
    from .ragdoc import export_ragdoc
    export_ragdoc(doc, out)
    for name in ("chunks.jsonl", "triples.jsonl", "graph.cypher", "document.md", "document.json", "extraction.json",
                 "ragdoc_document.json", "ragdoc_graph.jsonl"):
        if (out / name).exists():
            paths[name] = str(out / name)
    return paths
