# Document-parsing models and services (surveyed 22 Oct 2026)

## Big picture

- **Small specialist VLMs (0.8–4 B) top the clean-document leaderboard.** OmniDocBench v1.6 leaders: TeleOCR (1.2 B, 96.9), OvisOCR2 (0.8 B, 96.6), PaddleOCR-VL-1.6 (0.9 B, 96.3), MinerU2.5-Pro (1.2 B, 95.8), GLM-OCR (0.9 B, 95.2). Frontier general VLMs trail (Gemini 3 Pro 92.9, Qwen3-VL-235B 89.8, GPT-5.2 86.6).
- **olmOCR-bench (scan-heavy, English) ranks differently.** Open: Infinity-Parser2-Pro 87.6 (35 B MoE), Infinity-Parser2-Flash 86.0 (2 B), Chandra 2 85.8 (restricted licence), dots.ocr-1.5 / dots.mocr 83.9, LightOnOCR-2-1B 83.2, olmOCR-2 82.4. Hosted: Sarvam Vision 2.1 87.3, Mistral OCR 4 85.2 (self-reported).
- **OmniDocBench is saturating and ~12 % noisy** (PureDocBench audit, May 2026). Under real-world degradation (photos, warping) big general VLMs degrade less than pipeline specialists; Qwen3.5-122B-A10B tops PureDocBench at ~74.
- **"Agentic" benchmarks (ParseBench, Apr 2026)** reward chart-to-data, formatting and grounding: LlamaParse Agentic 84.9 vs single-pass open models (dots.ocr-1.5 55.8, Qwen3-VL-4B 62.0), mostly because open OCR models score ~0 on charts.
- **Commercial products converged on one output shape:** markdown + JSON of typed blocks with normalised bboxes, page index, confidence, and increasingly line/word/cell-level grounding (LandingAI ADE Gen2, Reducto, LlamaParse, Mistral OCR 4, Azure DI). `docintel.schemas` mirrors this.
- **Hindi/Indic is the weak spot of most open models.** Explicit support: PaddleOCR-VL (109 langs incl. 13 Devanagari languages — but independent GlotOCR shows near-zero Devanagari accuracy for 1.5, so verify on your scans), Chandra 2 (Hindi 78.4 on its 43-lang bench), Mistral OCR 4 (170 langs), Bodhan IndicOCR (22 Indian languages, open weights), Sarvam Vision 2.1 (22 Indian languages, API). The Devanagari stress test found DeepSeek-OCR and olmOCR collapse on real Hindi scans; Qwen3-VL-8B was the best open general model, dots.ocr the best open specialist on GlotOCR.

## Open-weight models

| Model | Release | Licence | Size | Handles | Hindi/Indic | Grounding | Output | OmniDocBench v1.6 | olmOCR-bench | Notes |
|---|---|---|---|---|---|---|---|---|---|---|
| **PaddleOCR-VL-1.6** (Baidu) | May 2026 | Apache-2.0 | 0.9 B | text, tables, formulas, charts, seals, spotting; handwriting samples | 109 langs; Devanagari group listed; **independent evidence weak** | layout stage + polygons; JSON | MD + JSON | 96.3 | 80.0 (v1.0) | ~2 GB VRAM; ~45 pages/min on L40S; no fine-tuning support yet |
| **TeleOCR** | Aug 2026 | Apache-2.0 | 1.2 B | text, tables (OTSL spans), formulas, camera pages | n/a | blocks + bbox + rotation | blocks | 96.9 | — | new, thin ecosystem |
| **OvisOCR2** | Jul 2026 | Apache-2.0 | 0.8 B | text, tables, formulas | n/a | region bboxes | MD | 96.6; PureDocBench top | — | single pass |
| **MinerU2.5-Pro / MinerU 4.0** | Apr 2026 | MinerU licence (Apache + terms) | 1.2 B two-stage | text, tables, formulas, figures, order | CN/EN-centric (MORE Hindi 34) | layout bboxes, content_list JSON | MD/HTML/JSON/DOCX | 95.8 | 75.2 | many backends (vLLM, llama.cpp, MLX) |
| **GLM-OCR** (Z.ai) | Feb 2026 | MIT | 0.9 B | text, tables, formulas, seals, code, handwritten; JSON-schema KIE | 100+ claimed; GlotOCR Deva 0.2 | layout stage | MD/HTML/LaTeX/JSON | 95.2 | 75.2 | built-in schema KIE |
| **Infinity-Parser2 Pro/Flash** (infly) | May 2026 | Apache-2.0 | 35 B MoE / 2 B | text, tables, formulas, charts→table, DocVQA | EN/ZH | doc2json bboxes + category | MD/JSON | 94.0 / 92.0 | **87.6 / 86.0** | ParseBench 74.3 (best open) |
| **dots.ocr-1.5 / dots.mocr** (rednote) | Feb–Mar 2026 | MIT (+ attribution) | 3 B | layout + text, tables HTML, formulas, charts/diagrams→SVG, grounding, VQA | multilingual; **best open on Devanagari (GlotOCR 90.2)** | JSON bbox + 11 categories | JSON/MD/SVG | 90.8 | 83.9 | native vLLM; CPU guide |
| **Chandra 2** (Datalab) | Mar 2026 | OpenRAIL-M (restricted) | ~4–5 B | text, tables, forms, checkboxes, handwriting, math, charts→data, Mermaid | 90+; Hindi 78.4 | 15+ block types + bboxes | MD/HTML/JSON | — | 85.8 | best OSS on RealDocBench QA; licence limits |
| **LightOnOCR-2-1B** | Jan 2026 | Apache-2.0 | 1 B | text, MD, figure bboxes | European | figure bboxes | MD | — | 83.2 | fastest in class |
| **olmOCR-2** (AI2) | Oct 2025 | Apache-2.0 | 7 B | text, tables, equations, multi-column | English; collapses on Hindi scans | none | MD | 85.7 | 82.4 | <$200 / 1 M pages |
| **DeepSeek-OCR-2** | Jan 2026 | Apache-2.0 | ~3 B MoE | doc→MD, grounding prompt | **avoid for Indic** (repetition loops) | grounding prompt | MD | 90.3 | 76.3 | very cheap tokens/page |
| **Nanonets-OCR2-3B** | Oct 2025 | verify | ~4 B | tables, LaTeX, charts, signatures, checkboxes, Mermaid | no Indic | none | semantic MD | 83.6 | 69.5 | DocVQA 89.4 |
| **Granite-Docling-258M + Docling** (IBM) | 2025–26 | Apache-2.0 / MIT | 258 M + Heron + TableFormer | layout, tables, code, formulas | Latin-first; Indic via RapidOCR/Tesseract | **per-element provenance (page + bbox)** | JSON/MD/HTML/DocTags | — (ParseBench 50.6) | — | CPU-capable; LangChain/LlamaIndex adapters |
| **Qwen3-VL (2–235 B) / Qwen3.5** (Alibaba) | 2025–26 | Apache-2.0 | 2/4/8/32 B dense, MoE | general VLM: OCR, QwenVL-HTML bboxes, grounding (0–1000), long context | 39 OCR langs; **best open general model on real Hindi scans** (chrF++ 75.2); best on OmniHandwritingOCR | yes | by prompt | 89.8 (235B) | 64.6 (8B) | best LoRA base and verifier |
| **Bodhan IndicOCR** (IIT-M spin-off) | Sep 2026 | open weights (gated) | 33 M layout + 0.8 B OCR | blocks, order, LaTeX, tables; handwriting for 12 Indian langs | **EN + all 22 scheduled Indian languages** | layout blocks | MD/structured | 92.8 (EN) | 82.2 (EN) | Indic-first open option |

## Hosted services

| Service | Latest | Grounding | Hindi | Benchmarks | Price |
|---|---|---|---|---|---|
| **LandingAI ADE Gen2 (DPT-3 Pro/Verity)** | Sep 2026 | line-level (Pro), word-level + per-word confidence (Verity), table cells, Extract V2 atomic citations | not enumerated | ParseBench 45.2 (Gen1) | per output character; Verity < 1¢/page |
| **Reducto Parse r-1** | Sep 2026 | blocks with normalised boxes, page, confidence; per-field citations | 100+ langs | RD-TableBench 90.2; ParseBench 67.8 (legacy) | $0.01/page parse, $0.02 extract |
| **LlamaParse** | 2026 | line/word/cell bboxes (beta) | 100+ | ParseBench 84.9 (Agentic, vendor-run) | 1–45 credits/page |
| **Mistral OCR 4/4.1** | Jun–Aug 2026 | per-block bbox + type; page/word confidence | 170 langs, Hindi "specialised" | olmOCR 85.2; OmniDocBench 93.1 | $4 / 1 k pages |
| **Sarvam Vision 2.1** | Sep 2026 | not documented | 22 Indian languages, handwriting | olmOCR 87.3 (self); Indic bench 87.4 | not public |
| **Azure Document Intelligence v4 / Content Understanding** | 2026 | 4-point polygons | Hindi, Marathi, Tamil printed; no Indic handwriting | ParseBench 59.6 (charts 1.6) | $1.5–30 / 1 k |
| **Google Document AI Layout Parser** | 2026 | bboxes only in v1.0 | n/a | ParseBench 50.4 | $10 / 1 k |

## LandingAI's "OCR → Agentic Document Extraction" post (Sep 2025)

Four waves: (1) OCR digitises characters but flattens layout and fails on scans/handwriting; (2) statistical NLP / early ML needs hand features and solves single subtasks; (3) LLMs/VLMs generalise across layouts but lose provenance, hallucinate plausibly and break schemas; (4) ADE = *visual-AI-first* (treat the page as an image), *agentic* (orchestrator plans, calls specialised tools, verifies) and *data-centric* (production misses feed annotation). Output: `markdown` + `chunks[]` in reading order, each `{id, type ∈ {text, table, figure, marginalia, logo, card, attestation, scan_code}, markdown, grounding{page, box{l,t,r,b} normalised 0–1}}`, plus table/tableCell groundings, confidence and low-confidence spans; companion Extract API returns every value with page + box + crop. Gen2 (Sep 2026) groups blocks under pages, grounds every line (Pro) or word (Verity) and returns character-level citations for extracted values.

`docintel` reproduces this shape with open weights: `Chunk{id,type,text,bbox(0–1),page,confidence,words[]}` → `FieldValue{value,groundings[],verified}`.

## Recommended open-source stack

1. **Render & route** — PyMuPDF at 150–200 DPI; keep the born-digital text layer as a zero-hallucination cross-check; script detection decides the OCR path.
2. **Primary parser (Latin/CJK, clean–moderate scans)** — PaddleOCR-VL-1.6 (Apache-2.0, 0.9 B) or GLM-OCR (MIT, schema-KIE built in); Infinity-Parser2-Flash for olmOCR-bench-style robustness.
3. **Secondary parser for degraded pages, charts and Devanagari** — dots.ocr-1.5 / dots.mocr (MIT; best open Devanagari, layout JSON with bboxes, charts → SVG), with Qwen3-VL-8B / Qwen3.5 as the general-VLM fallback and verifier.
4. **Indic path** — route Devanagari/Bengali/Tamil pages to Bodhan IndicOCR (or dots.ocr), verify with Qwen3-VL; train a per-engine post-corrector for conjunct/matra errors if needed.
5. **Canonical document model** — a Docling-like typed-block schema with per-element provenance (this repo's `ParsedDocument`); markdown carries chunk anchors so text ↔ box links survive into RAG.
6. **Tables** — PaddleOCR-VL / TeleOCR (OTSL spans); TableFormer or word-geometry clustering on CPU.
7. **Agentic loop** — orchestrator builds/takes a schema, extracts with chunk/cell citations, verifies by re-reading the crop and cross-checking the text layer, escalates low-confidence pages to the secondary parser, flags for review.
8. **Serving** — vLLM for all of the above; batch 64–128 pages; ~1–2 pages/s per GPU for 1–3 B models (~$0.5–1 per 1 k pages on an L40S-class card).

## Caveats

Vendor numbers (Sarvam, Mistral, TeleOCR, OvisOCR2, LlamaParse's ParseBench) are self-reported; only the OmniDocBench and olmOCR repo tables are third-party-run. Some aggregator pages list unverifiable scores.

## Sources

landing.ai/blog/ocr-to-agentic-document-extraction-… · docs.landing.ai/ade/ade-json-response.md · docs.landing.ai/ade/ade-groundings · landing.ai/blog/introducing-agentic-document-extraction-gen2 · github.com/opendatalab/OmniDocBench · github.com/allenai/olmocr · arxiv.org/abs/2604.08538 (ParseBench) · arxiv.org/pdf/2605.07492 (PureDocBench) · arxiv.org/pdf/2606.29213 (Devanagari stress test) · huggingface.co/PaddlePaddle/PaddleOCR-VL-1.6 · paddleocr.ai PaddleOCR-VL docs · hackernoon.com (TeleOCR) · arxiv.org/abs/2607.13639 (OvisOCR2) · github.com/opendatalab/MinerU · arxiv.org/abs/2509.22186 · huggingface.co/zai-org/GLM-OCR · arxiv.org/abs/2603.13398 (Qianfan-OCR) · huggingface.co/infly/Infinity-Parser2-Pro · huggingface.co/rednote-hilab/dots.ocr · huggingface.co/rednote-hilab/dots.mocr · huggingface.co/datalab-to/chandra-ocr-2 · huggingface.co/blog/lightonai/lightonocr-2 · huggingface.co/deepseek-ai/DeepSeek-OCR-2 · huggingface.co/nanonets/Nanonets-OCR2-3B · github.com/docling-project/docling · github.com/qwenlm/qwen3-vl · arxiv.org/pdf/2511.21631 · analyticsvidhya.com (Bodhan) · sarvam.ai/blogs/sarvam-vision-2-1 · mistral.ai/news/ocr-4 · reducto.ai/blog/parse-r-1-model · docs.reducto.ai/parse/response-format.md · llamaindex.ai/blog/parsebench · learn.microsoft.com Azure DI model overview · docs.cloud.google.com/document-ai/docs/layout-parse-chunk
