# Evaluation plan

## Train-or-not decision (from `benchmarks.md` / `models.md`)

| Capability | Decision | Rationale |
|---|---|---|
| English layout, reading order, printed text | **Use pre-trained** (dots.ocr / PaddleOCR-VL / Qwen3-VL) | OmniDocBench ≥ 90 for all candidates; olmOCR-bench 80–88 |
| Tables (ruled, borderless, spans) | **Use pre-trained**, add validation rules | TEDS 90+ for PaddleOCR-VL / TeleOCR; Indic tables weaker (MUSTARD 85 TEDS-S) — fine-tune TSR only if your tables are Indic-script heavy |
| Printed Hindi / bilingual | **Use pre-trained, verify on own scans** | dots.ocr 90 Acc@5 (GlotOCR), Gemini 86 chrF++ on degraded scans; avoid PaddleOCR-VL/DeepSeek/GLM/MinerU for Devanagari |
| Charts / infographics / maps | **Use frontier VLM** for extraction (open 8 B models trail by 20–40 pts) | ChartQAPro ~72, MapQA ~50 even for the best |
| English handwriting | **Use pre-trained** (Qwen3-VL-8B, frontier) | IAM-competitive zero-shot |
| **Hindi handwriting** | **Train** (staged TrOCR / Qwen-VL-2B) | no public evidence of any model working; data: synthetic → IIIT-HW-Dev → ≥ 2.5–3 K own lines |
| **Domain schemas** | **LoRA** after 100–500 labelled docs; zero-shot first | +10–30 pts; bboxes as spatial priors help |
| Retrieval over many pages | **Use pre-trained** (ColQwen3 / text + reranker) | ViDoRe V3 ≈ 63 either way; ragdoc already does hybrid retrieval |

## Public benchmarks to run

| Benchmark | Why | Runner | Metric | Target |
|---|---|---|---|---|
| OmniDocBench v1.6 | end-to-end parsing, layout mAP, reading order | `docintel eval omnidocbench` → official scorer | overall / TEDS / NED | ≥ 90 overall with dots.ocr; ≥ 94 with PaddleOCR-VL on EN |
| olmOCR-bench | robustness on scans, tiny text, multi-column | `docintel eval olmocr-bench` → `olmocr.bench.benchmark` | % tests passed | ≥ 80 |
| ParseBench / RealDocBench | **grounding + charts + field QA** — closest to agentic extraction | manual (download, run pipeline, use their harness) | per-rule / per-field | track vs LlamaParse 84.9 / Chandra 86.2 |
| DocVQA / InfographicVQA | document & infographic QA through the parse | `docintel eval docvqa` | ANLS | ≥ 0.85 (parse+LLM) |
| ChartQA / ChartQAPro / CharXiv | chart-to-data quality | `docintel eval chartqa` | relaxed acc | ≥ 0.80 ChartQA; report ChartQAPro |
| MapVerse / MapQA | map understanding | manual | acc | report |
| FUNSD / CORD / SROIE | KV extraction sanity | `docintel eval funsd` | F1 / NED / box F1 | ≥ 0.85 F1 |
| GlotOCR Devanagari slice + Devanagari stress test | printed Hindi | `docintel eval handwriting --data lines.jsonl` (works for printed lines too) | CER / chrF++ | CER ≤ 5 % clean, ≤ 15 % degraded |
| IIIT-HW-Dev, ICDAR-2025 IHDR | Hindi handwriting (before/after training) | `docintel eval handwriting` | CER / WER | CER ≤ 10 % after stage C |
| MP-DocVQA / MMLongBench-Doc | downstream RAG (with ragdoc) | ragdoc `eval` | ANLS / acc | report |

## Private evaluation set (required — no public set covers this)

Build 300–500 pages that mirror production: bilingual Hindi/English forms and letters, degraded scans/photos,
handwritten fields, tables with merged cells, charts and maps. Label with the docintel UI (correct chunk text and
field values; boxes are produced automatically) and export with `training/export_labels.py`. Score with
`docintel eval extraction` (field exact/fuzzy match + grounding hit-rate) and `docintel eval handwriting` (CER/WER on
line crops). Report per-language, per-script and per-source (digital / scan / photo) slices, plus page-level failure
rate and repetition-loop rate for VLM backends.

## Results log

| Date | Config | Benchmark | Score | Notes |
|---|---|---|---|---|
| 2026-10-22 | offline (text layer + Tesseract + heuristics) | sample invoice (smoke) | 4/4 fields exact, all grounded & verified; table 4×5 recovered with "Unit price" as one cell; chart region detected | no models; see `tests/` |
| 2026-10-22 | offline | review edge cases: blank PDF, white/tiny PNG, 3-page two-column PDF (unicode filename), nested schema (object+array+enum), empty/odd schemas, markdown, HTML & DOCX via LibreOffice, 4 concurrent API jobs | all parse without error; lab-report HTML: `hemoglobin_value` = 12.5 grounded to the cell with row *Hemoglobin*, column *Result*, section *Results*; DOCX PO: 2 line items typed; two-column prose not mistaken for a table | `tests/test_ade_parity.py::test_lab_report_cell_grounding` |
| 2026-10-22 | offline | independent code review (separate reviewer agent) + regression suite | 3 critical (path-traversal delete, stored XSS via filename, wrong grounding on rotated pages), 6 high (two-column order, region page index, event-loop blocking, text-file pagination/Unicode, no resource caps, table column merging) and 10 medium issues fixed; 29 tests pass | `tests/test_review_regressions.py` |
