# Architecture

## Principles

1. **One data contract.** Every stage reads and writes `ParsedDocument → Page → Chunk` (`docintel/schemas.py`). A chunk is a
   layout element with a type, text (markdown), normalised bbox, page, confidence, language/script tags, optional
   word-level boxes, and structured payloads for tables and charts. This is the same shape LandingAI ADE, Reducto,
   Docling and OmniDocBench use, so outputs are benchmarkable and swappable.
2. **Visual grounding is not optional.** Every piece of text, every table cell, every extracted field and every KG
   triple points back to `(page, bbox)`. The UI, the RAG citations and the verifier all consume the same groundings.
3. **Models are plug-ins.** OCR/VLM backends, layout detectors and LLM providers are behind small protocols. The
   pipeline runs with zero models (text layer + Tesseract + geometry) and gets better as you plug in dots.ocr,
   DocLayout-YOLO, Qwen3-VL or a frontier model. Nothing in the code depends on a specific vendor.
4. **Agentic = selective re-work, not one giant prompt.** A deterministic first pass produces an *observation*
   (flagged chunks: low confidence, handwriting, bilingual, unparsed table, undescribed figure, garbled text layer).
   The agent then calls tools on those chunks only, bounded by `agent_max_steps`, and records a trace.

## Stages

| Stage | Module | Input → output | Offline behaviour | With models |
|---|---|---|---|---|
| Ingest | `ingest/loaders.py` | file → page PNGs + native text blocks/words + embedded image rects + vector-drawing rects | PyMuPDF; Office via LibreOffice | same |
| Text acquisition | `ocr/backends.py` | page image → chunks | text layer if present, else Tesseract (word boxes, `eng+hin`) | VLM parse (dots.ocr JSON / PaddleOCR-VL / DeepSeek grounding / generic JSON); native words attached to VLM boxes for exact grounding |
| Layout refinement | `layout/detector.py` | chunks → typed chunks + merged regions | font-size/position heuristics, **word-grid gutter detection** for borderless tables, vector-path clusters for charts/diagrams | DocLayout-YOLO regions merged with text chunks |
| Reading order | `layout/detector.py::reading_order` | two-column aware sort | same | VLM order when available |
| Tables | `tables/extractor.py` | table chunk → `TableData` (cells, spans, HTML, markdown) | word-geometry row/column clustering | VLM table→HTML (`read_region`) |
| Visuals | `charts/extractor.py` | figure chunk → crop + classification + `ChartData` / structured map description | colour-statistics classifier | VLM classify → chart-to-table JSON / map-legend JSON / caption |
| Agent | `agents/orchestrator.py` | observation → tool calls | rule policy (table/visual tools, Tesseract re-OCR of garbled regions) | LLM ReAct policy with `zoom_read`, `extract_table`, `describe_visual`, `reread_page` |
| Schema extraction | `extraction/schema.py` | JSON schema → `ExtractionResult` | retrieval + key/value & table heuristics | retrieval → LLM extraction with chunk citations → word-level grounding → independent verification (text match or VLM re-read of the crop) |
| KG | `export/kg.py` | chunks/fields/tables/charts → triples with provenance | deterministic sources | + LLM open-IE on prose |
| Export | `export/{markdown,rag,ragdoc}.py` | markdown, `chunks.jsonl`, `triples.jsonl`, Cypher, ragdoc document + graph | same | same |

## Bilingual and handwriting handling

* `ocr/langdetect.py` tags scripts per chunk from Unicode ranges (Latin, Devanagari, Bengali, Tamil, …). Chunks with
  ≥ 2 scripts are flagged `bilingual`; Tesseract is called with the union of language packs; VLM prompts instruct
  "keep original language, transcribe both".
* Handwriting is a chunk type (`handwriting`), set by the VLM/layout model; the agent zoom-reads handwritten regions
  at 2–3× resolution. For Hindi handwriting specifically, plug a fine-tuned recogniser (`training/htr_devanagari`) in as
  a region reader — the research shows nothing off-the-shelf is reliable.
* Garbled text layers (common with Indic PDFs whose fonts lack Unicode maps) are detected (control characters) and
  re-OCR'd from pixels.

## Grounding model

* All boxes are `[x0, y0, x1, y1]` in `[0, 1]` relative to the page (top-left origin). The UI multiplies by the rendered
  image size, so DPI can change without touching stored data.
* Word boxes come from the PDF text layer or Tesseract; VLM chunks inherit the native words that fall inside their box.
* `extraction.ground_span` fuzzy-matches the extracted value against the chunk's word sequence to produce a tight
  word-level box; it falls back to the chunk box.
* Figure crops are stored (`Chunk.image_path`) so a field extracted from a chart can show the exact crop.

## Extraction modes

`Pipeline.parse(mode=...)`: `parse` skips field extraction entirely; `schema` extracts the supplied JSON Schema;
`auto` calls `extraction/discover.suggest_schema` (form labels → properties with conservative type guessing,
tables → arrays of objects, LLM refinement when available), stores it in `metadata.suggested_schema`, then extracts
against it. The mode is recorded in `metadata.extraction_mode`. Default comes from `settings.extraction_mode`.

## Hierarchy and typed extraction

`schemas.build_tree` turns the flat reading-order chunk list into `title → section → chunks` (`ParsedDocument.tree`),
sets `Chunk.parent_id` and `attributes.section`, and the extractor records that section in each field's `context`.
`extraction/normalize.py` coerces values to the schema's type/format/enum (numbers, ISO dates incl. Hindi month
names and Devanagari digits, booleans, enums, arrays of objects) so heterogeneous documents yield one shape;
failures keep the raw text with `type_ok=false` rather than silently dropping data. Table values are grounded to the
**cell** (bbox + row label + column header) and a cropped snippet PNG is saved for every grounded field.

## Verification

`extract_fields` marks a value **verified** when (a) it is found verbatim (fuzzy ≥ 90) in the grounded chunk text, or
(b) the VLM re-reads the grounded crop and the value is found in that reading. Unverified values keep the LLM's
answer but drop confidence by 40 % and carry a note, so downstream systems can route them to human review.

## API

`POST /api/documents` (upload + optional schema) · `GET /api/documents/{id}` · `/status` · `/pages/{n}` ·
`/crops/{chunk_id}` · `POST /extract` · `GET /chunks` · `GET /search?q=` · `GET /export` (zip) · `DELETE`.
Parsing runs as a background task; the UI polls status.

## Scaling notes

* Parsing is per page and embarrassingly parallel: run N workers, each owning a vLLM client; batch 64–128 page
  images per request on the VLM side.
* Keep page PNGs and crops on object storage; `ParsedDocument` JSON in a document store; chunk records in the vector
  store; triples in the graph store (ragdoc already does the last two).
* Cost reference (research): 1–3 B document VLMs ≈ 1–2 pages/s per L40S/H100-class GPU, roughly $0.5–1 per 1 k pages.
