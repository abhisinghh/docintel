"""ParsedDocument → Markdown (the format OmniDocBench / olmOCR-bench score, and what most RAG loaders want)."""
from __future__ import annotations

from ..schemas import ChunkType, ParsedDocument

SKIP = {ChunkType.PAGE_HEADER, ChunkType.PAGE_FOOTER, ChunkType.PAGE_NUMBER}


def document_to_markdown(doc: ParsedDocument, page_breaks: bool = True, keep_headers: bool = False) -> str:
    out: list[str] = []
    for p in doc.pages:
        if page_breaks and len(doc.pages) > 1:
            out.append(f"\n<!-- page {p.number} -->\n")
        for c in sorted(p.chunks, key=lambda c: c.reading_order):
            if c.type in SKIP and not keep_headers:
                continue
            t = c.text.strip()
            if not t:
                continue
            if c.type == ChunkType.TITLE:
                out.append(f"# {t}")
            elif c.type == ChunkType.SECTION_HEADER:
                out.append(f"## {t}")
            elif c.type == ChunkType.LIST_ITEM:
                out.append(t if t.lstrip().startswith(("-", "*", "•")) else f"- {t}")
            elif c.type == ChunkType.TABLE:
                out.append(c.table.to_markdown() if c.table else t)
            elif c.type == ChunkType.FORMULA:
                out.append(f"$$\n{t}\n$$")
            elif c.type == ChunkType.CODE:
                out.append(f"```\n{t}\n```")
            elif c.type in (ChunkType.FIGURE, ChunkType.CHART, ChunkType.MAP, ChunkType.INFOGRAPHIC):
                out.append(f"![{c.type.value}]({c.image_path or ''})\n\n{t}")
            elif c.type == ChunkType.CAPTION:
                out.append(f"*{t}*")
            else:
                out.append(t)
    return "\n\n".join(out).strip() + "\n"
