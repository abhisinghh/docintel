"""End-to-end: file → ParsedDocument (chunks with grounding, tables, charts, schema fields, triples)."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable, Optional

from PIL import Image

from .agents.llm import LLM
from .agents.orchestrator import DocumentAgent
from .charts.extractor import extract_visuals
from .config import Settings, get_settings
from .export.markdown import document_to_markdown
from .export.kg import extract_triples
from .ingest.loaders import load_document, native_blocks_to_chunks
from .layout.detector import build_layout, reading_order
from .ocr.backends import TesseractBackend, VLMBackend, build_backend
from .ocr.langdetect import detect_languages, detect_scripts
from .schemas import Chunk, Page, ParsedDocument, build_tree
from .tables.extractor import extract_tables

ProgressFn = Callable[[str, float], None]


class Pipeline:
    def __init__(self, settings: Optional[Settings] = None):
        self.s = settings or get_settings()
        self.s.ensure_dirs()
        self.backend = build_backend(self.s)
        self.vlm: Optional[VLMBackend] = self.backend if isinstance(self.backend, VLMBackend) else None
        # when a VLM/hosted parser is enabled it parses every page (the text layer is still attached for exact word
        # boxes); otherwise born-digital pages use the text layer and scans go to classical OCR
        self.vlm_parses_pages = self.s.vlm.enabled
        self.layout = build_layout(self.s)
        self.llm: Optional[LLM] = LLM(self.s.llm) if self.s.llm.provider != "none" else None
        if self.llm is not None and self.llm.available and self.s.llm.provider != "transformers":
            from .agents.llm import probe
            if not probe(self.llm):             # endpoint not reachable → deterministic offline policies
                self.llm.available = False
                self.llm = None
        self._ocr_fallback = TesseractBackend(self.s.ocr.languages)

    # ------------------------------------------------------------------
    def parse(self, path: str | Path, doc_id: Optional[str] = None, schema: Optional[dict[str, Any]] = None,
              progress: Optional[ProgressFn] = None, build_kg: bool = True,
              mode: Optional[str] = None) -> ParsedDocument:
        """mode: "parse" (no field extraction), "schema" (extract against `schema`), "auto" (propose a schema
        from the document, then extract against it). Defaults: "schema" when a schema is given, else settings."""
        mode = mode or ("schema" if schema else self.s.extraction_mode)
        if mode == "schema" and not schema:
            raise ValueError("extraction mode 'schema' requires a schema")
        if mode == "parse":
            schema = None
        t0 = time.time()
        progress = progress or (lambda stage, p: None)
        doc = ParsedDocument(filename=Path(path).name)
        if doc_id:
            doc.doc_id = doc_id
        pages_dir = self.s.data_dir / "pages" / doc.doc_id
        crops_dir = self.s.data_dir / "crops" / doc.doc_id

        progress("ingest", 0.05)
        loaded = load_document(path, pages_dir, dpi=self.s.render_dpi)
        doc.mime_type, doc.n_pages = loaded.mime_type, len(loaded.pages)
        images: dict[int, Image.Image] = {}
        timings: dict[str, float] = {}

        for i, rp in enumerate(loaded.pages):
            progress(f"page {rp.number}/{doc.n_pages}", 0.1 + 0.6 * i / max(doc.n_pages, 1))
            images[rp.number] = rp.image
            t = time.time()
            # 1. text acquisition --------------------------------------------------------
            if rp.has_text_layer and not self.s.ocr.force_ocr and not self.vlm_parses_pages:
                chunks = native_blocks_to_chunks(rp, doc.doc_id)
                src = "native"
            else:
                try:
                    chunks = self.backend.parse_page(rp.image, rp.number, doc.doc_id)
                    src = self.backend.name
                except Exception as e:   # VLM down → degrade gracefully
                    doc.pipeline.setdefault("warnings", []).append(f"page {rp.number}: {self.backend.name} failed ({e})")
                    chunks = (native_blocks_to_chunks(rp, doc.doc_id) if rp.has_text_layer
                              else self._ocr_fallback.parse_page(rp.image, rp.number, doc.doc_id))
                    src = "fallback"
                # if the VLM parsed the page but a text layer exists, attach exact words for grounding
                if rp.has_text_layer:
                    _attach_native_words(chunks, native_blocks_to_chunks(rp, doc.doc_id))
            timings["text"] = timings.get("text", 0) + time.time() - t
            # 2. layout refinement ---------------------------------------------------------
            t = time.time()
            chunks = self.layout.refine(rp.image, chunks, drawings=rp.native_drawings, page=rp.number, doc_id=doc.doc_id)
            chunks = reading_order(chunks)
            timings["layout"] = timings.get("layout", 0) + time.time() - t
            # 3. tables & visuals ----------------------------------------------------------------
            t = time.time()
            chunks = extract_tables(rp.image, chunks, vlm=self.vlm)
            chunks = extract_visuals(rp.image, chunks, crops_dir, vlm=self.vlm)
            timings["structure"] = timings.get("structure", 0) + time.time() - t
            # 4. language tags ------------------------------------------------------------------
            for c in chunks:
                if not c.languages and c.text:
                    c.languages, c.scripts = detect_languages(c.text), detect_scripts(c.text)
            page_langs = sorted({l for c in chunks for l in c.languages})
            doc.pages.append(Page(number=rp.number, width=rp.width, height=rp.height, image_path=str(rp.image_path),
                                  chunks=chunks, languages=page_langs, has_text_layer=rp.has_text_layer))
            doc.pipeline.setdefault("page_sources", {})[str(rp.number)] = src

        # 5. agent pass + schema extraction -------------------------------------------------------
        progress("agent", 0.75)
        doc.tree = build_tree(doc)
        if mode == "auto":
            from .extraction.discover import suggest_schema
            doc.markdown = document_to_markdown(doc)
            schema = suggest_schema(doc, llm=self.llm)
            doc.metadata["suggested_schema"] = schema
        doc.metadata["extraction_mode"] = mode
        t = time.time()
        agent = DocumentAgent(llm=self.llm, vlm=self.vlm, crops_dir=crops_dir, max_steps=self.s.agent_max_steps,
                              ocr_fallback=self._ocr_fallback)
        agent.run(doc, images, schema=schema if schema and schema.get("properties", schema) else None,
                  verify=self.s.verify_extractions, snippets_dir=crops_dir, strict=self.s.strict_grounding)
        if doc.extraction is not None:
            doc.extraction.schema_name = "auto" if mode == "auto" else doc.extraction.schema_name
        timings["agent"] = time.time() - t

        # 6. document-level outputs ------------------------------------------------------------
        progress("export", 0.9)
        doc.languages = sorted({l for p in doc.pages for l in p.languages})
        doc.tree = build_tree(doc)          # rebuilt: the agent may have re-parsed pages
        doc.markdown = document_to_markdown(doc)
        if build_kg:
            doc.triples = extract_triples(doc, llm=self.llm)
        doc.pipeline.update({
            "ocr_backend": self.backend.name, "layout_engine": self.layout.name,
            "llm": (self.s.llm.model if self.llm else None), "timings_s": {k: round(v, 2) for k, v in timings.items()},
            "total_s": round(time.time() - t0, 2),
        })
        doc.status = "parsed"
        self.save(doc)
        progress("done", 1.0)
        return doc

    # ------------------------------------------------------------------
    def save(self, doc: ParsedDocument) -> Path:
        out = self.s.data_dir / "parsed" / f"{doc.doc_id}.json"
        out.write_text(doc.model_dump_json(indent=1), encoding="utf-8")
        return out

    def load(self, doc_id: str) -> ParsedDocument:
        p = self.s.data_dir / "parsed" / f"{doc_id}.json"
        return ParsedDocument.model_validate_json(p.read_text(encoding="utf-8"))

    def suggest_schema(self, doc: ParsedDocument) -> dict[str, Any]:
        from .extraction.discover import suggest_schema
        schema = suggest_schema(doc, llm=self.llm)
        doc.metadata["suggested_schema"] = schema
        self.save(doc)
        return schema

    def extract(self, doc: ParsedDocument, schema: Optional[dict[str, Any]] = None, schema_name: str = "default") -> ParsedDocument:
        """Run (another) schema extraction on an already parsed document. No schema → propose one first."""
        if not schema:
            schema, schema_name = self.suggest_schema(doc), "auto"
        images = {p.number: Image.open(p.image_path).convert("RGB") for p in doc.pages if p.image_path}
        from .extraction.schema import extract_fields
        doc.extraction = extract_fields(doc, schema, llm=self.llm, vlm=self.vlm, page_images=images,
                                        verify=self.s.verify_extractions, schema_name=schema_name,
                                        snippets_dir=self.s.data_dir / "crops" / doc.doc_id,
                                        strict=self.s.strict_grounding)
        self.save(doc)
        return doc


def _attach_native_words(vlm_chunks: list[Chunk], native_chunks: list[Chunk]) -> None:
    """Give VLM chunks the exact word boxes from the PDF text layer (for word-level grounding)."""
    words = [w for c in native_chunks for w in c.words]
    for c in vlm_chunks:
        if c.words:
            continue
        c.words = [w for w in words if _center_in(w.bbox, c.bbox)]


def _center_in(inner, outer) -> bool:
    cx, cy = (inner.x0 + inner.x1) / 2, (inner.y0 + inner.y1) / 2
    return outer.x0 <= cx <= outer.x1 and outer.y0 <= cy <= outer.y1
