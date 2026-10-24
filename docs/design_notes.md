# Design notes (moved out of the README)

## Why this design — summary of the research in `models.md` / `benchmarks.md`

* **Off-the-shelf document VLMs now beat OCR pipelines** on clean documents (OmniDocBench top ≈ 96) and emit layout
  JSON with bboxes natively (dots.ocr, PaddleOCR-VL, Infinity-Parser2). We do **not** train a parser; we orchestrate
  them.
* **The "agentic" part is the differentiator, not the OCR model**: ParseBench shows single-pass open models score ≈ 0
  on charts and lose grounding/formatting, while an orchestrated pipeline (plan → tools → verify) scores 85.
  docintel's agent inspects the first pass and calls `zoom_read`, `extract_table`, `describe_visual`, `reread_page`,
  `extract_fields`, `verify_field` only where needed.
* **Bilingual Hindi/English**: printed Devanagari is handled by dots.ocr (best open model on GlotOCR), Bodhan IndicOCR,
  or a frontier VLM; PaddleOCR-VL / DeepSeek-OCR / GLM-OCR / MinerU are unreliable on Devanagari in independent tests.
* **Custom training is required for exactly two things**: Hindi handwriting (no public evidence of any off-the-shelf
  model working; staged TrOCR/Qwen-VL recipe in `training/`) and domain-specific schemas (LoRA on 100–500 labelled
  docs, +10–30 pts). Everything else: use pre-trained models and *evaluate on your own pages*.

## Mapping to LandingAI's "Agentic Document Extraction" capabilities

Every row is covered by a test in `tests/test_ade_parity.py`.

| ADE capability | docintel | Where |
|---|---|---|
| **Structured JSON output** — hierarchical, same shape for PDFs and images | `ParsedDocument` → `pages[]` → `chunks[]` (typed, grounded) **plus** `tree`: title → sections → chunk ids; every chunk has `parent_id` and `attributes.section`. Any input becomes the same JSON | `schemas.py::build_tree`, `GET /api/documents/{id}`, `/tree` |
| **Schema-driven, zero-shot field extraction** | JSON Schema in → values **typed and normalised**: `number`/`integer` (currency, Indian/western grouping, Devanagari digits), `boolean`, `format: date` → ISO-8601 (dd/mm/yyyy, "Oct 14, 2026", "14 अक्टूबर 2026"), `enum` (fuzzy-snapped), `array` of objects (e.g. `line_items` from table rows). Missing → `null`, coercion failure → raw value + `type_ok=false` | `extraction/schema.py`, `extraction/normalize.py`, `POST /api/extract` |
| **Visual grounding** — page, bounding box, cropped snippet | every `FieldValue.groundings[]` has `page`, `bbox` (**table cell** or word-run when available, else chunk) and `snippet_path`; `context` records section, form label, table row label and column header ("12.5 g/dL" is the *Hemoglobin* row, *Result* column, in *Results*) | `extraction/schema.py::ground_cell/ground_span/save_snippet` |
| **Layout-aware parsing** | chunk types incl. `title/section_header/form_field/table/chart/map/infographic/handwriting`; table cells with spans; two-column reading order; charts → data tables; maps/infographics → structured descriptions | `layout/`, `tables/`, `charts/` |
| **Hallucination control / auditability** | independent verification of each value (text match on the grounded region, or VLM re-read of the crop); unverified values lose confidence and are flagged, or are nulled with `strict_grounding: true`; agent trace records every tool call | `_verify`, `config.strict_grounding`, Agent tab |
| **Offered as an API** | async upload + background parse, or synchronous `POST /api/extract` | `api/server.py` |

Where we are behind ADE: model quality (they train specialist DPT models; we orchestrate open ones), line/word-level
grounding with per-word confidence, trained detectors for ID cards / signatures / barcodes, Classify/Split/Jobs APIs,
SDKs, and published benchmark numbers. See the conversation history / `eval_plan.md` for the plan to measure the gap.

## Model providers in detail

| `provider` | What it talks to | Typical use | Config example |
|---|---|---|---|
| `openai` | any **OpenAI-compatible HTTP endpoint** | on-prem vLLM / SGLang / TGI / Ollama / LM Studio **or** hosted APIs (OpenAI, OpenRouter, Together, Fireworks, Gemini's OpenAI endpoint, Mistral) — only `base_url`/`api_key` change | `configs/dots_ocr.yaml` (on-prem), `configs/hosted.yaml` (API) |
| `anthropic` | Anthropic Messages API | Claude as page parser, verifier, agent and extractor | `configs/hosted.yaml` (llm), `configs/landingai.yaml` |
| `transformers` | weights loaded **in-process** with HF transformers | air-gapped / single-GPU / no inference server; any `AutoModelForImageTextToText` checkpoint | `configs/onprem_transformers.yaml` |
| `landingai` | LandingAI Agentic Document Extraction API | hosted parser with grounding; LLM of your choice on top | `configs/landingai.yaml` |
| `none` (LLM only) | — | deterministic offline mode | `configs/offline.yaml` |

The prompt *family* (`generic | dots_ocr | paddleocr_vl | deepseek_ocr`) is independent of the transport, and
`DOCINTEL_LAYOUT_ENGINE` (`heuristic | doclayout_yolo`) chooses the layout detector. `tests/test_providers.py`
exercises the OpenAI-compatible, Anthropic and LandingAI code paths against fake endpoints.
