"""Knowledge-graph triples with provenance.

Every triple carries the groundings of the chunk(s) it came from, so a graph
query can be traced back to a page region (and the UI can highlight it).

Sources of triples, cheapest first:
* schema fields           (document) --field_name--> value
* tables                  (row key)  --column--> cell       (and table → caption)
* form fields             key --is--> value
* section structure       section --contains--> paragraph  (optional; off by default)
* LLM open IE             entities/relations from prose (only when an LLM is available)
"""
from __future__ import annotations


from ..schemas import ChunkType, ParsedDocument, Triple

OPEN_IE_SYSTEM = (
    "You extract knowledge-graph triples from document text. Return JSON {\"triples\": [{\"subject\":..., "
    "\"predicate\":..., \"object\":..., \"subject_type\":..., \"object_type\":..., \"confidence\":0-1}]}. "
    "Use short canonical predicates (snake_case), keep entity names as written (original script), "
    "and only state facts explicitly supported by the text."
)


def table_triples(doc: ParsedDocument) -> list[Triple]:
    out: list[Triple] = []
    for c in doc.all_chunks():
        if c.type != ChunkType.TABLE or not c.table:
            continue
        grid = c.table.to_grid()
        if len(grid) < 2:
            continue
        header = grid[0]
        for row in grid[1:]:
            key = row[0].strip() if row else ""
            if not key:
                continue
            for j, cell in enumerate(row[1:], start=1):
                if j < len(header) and cell.strip() and header[j].strip():
                    out.append(Triple(subject=key, predicate=header[j].strip(), object=cell.strip(),
                                      subject_type="table_row", object_type="value",
                                      groundings=[c.grounding()]))
        if c.table.caption:
            out.append(Triple(subject=c.id, predicate="has_caption", object=c.table.caption, groundings=[c.grounding()]))
    return out


def form_triples(doc: ParsedDocument) -> list[Triple]:
    out = []
    for c in doc.all_chunks():
        if c.type == ChunkType.FORM_FIELD and ":" in c.text:
            k, v = c.text.split(":", 1)
            if k.strip() and v.strip():
                out.append(Triple(subject=doc.filename or doc.doc_id, predicate=k.strip(), object=v.strip(),
                                  subject_type="document", object_type="value", groundings=[c.grounding()]))
    return out


def field_triples(doc: ParsedDocument) -> list[Triple]:
    if not doc.extraction:
        return []
    return [Triple(subject=doc.filename or doc.doc_id, predicate=f.name, object=str(f.value),
                   subject_type="document", object_type="field_value", confidence=f.confidence,
                   groundings=list(f.groundings))
            for f in doc.extraction.fields if f.value not in (None, "")]


def chart_triples(doc: ParsedDocument) -> list[Triple]:
    out = []
    for c in doc.all_chunks():
        if c.chart and c.chart.series:
            for s in c.chart.series:
                for x, y in zip(s.x, s.y):
                    out.append(Triple(subject=str(x), predicate=s.name or (c.chart.y_label or "value"), object=str(y),
                                      subject_type="chart_category", object_type="number", confidence=0.7,
                                      groundings=[c.grounding()]))
    return out


def llm_triples(doc: ParsedDocument, llm, max_chars: int = 6000, max_calls: int = 40) -> list[Triple]:
    """Open IE on prose chunks, longest first, capped at `max_calls` LLM requests per document."""
    out: list[Triple] = []
    prose = sorted((c for c in doc.all_chunks() if c.type in (ChunkType.TEXT, ChunkType.LIST_ITEM) and len(c.text) >= 80),
                   key=lambda c: -len(c.text))[:max_calls]
    for c in prose:
        try:
            res = llm.complete_json(f"Text (page {c.page}):\n{c.text[:max_chars]}", system=OPEN_IE_SYSTEM)
            for t in res.get("triples", []):
                if all(k in t for k in ("subject", "predicate", "object")):
                    out.append(Triple(subject=str(t["subject"]), predicate=str(t["predicate"]), object=str(t["object"]),
                                      subject_type=t.get("subject_type"), object_type=t.get("object_type"),
                                      confidence=float(t.get("confidence", 0.8)), groundings=[c.grounding()]))
        except Exception as e:
            doc.pipeline.setdefault("warnings", []).append(f"open-IE failed on {c.id}: {e}")
    return out


def extract_triples(doc: ParsedDocument, llm=None) -> list[Triple]:
    triples = field_triples(doc) + table_triples(doc) + form_triples(doc) + chart_triples(doc)
    if llm is not None and getattr(llm, "available", False):
        triples += llm_triples(doc, llm)
    return triples
