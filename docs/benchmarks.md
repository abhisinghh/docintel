# Benchmark landscape (compiled 22 Oct 2026)

Numbers are as reported by the cited sources on the fetch date; leaderboards move monthly and sites mix model versions, so treat them as indicative. Full source list at the bottom.

## Executive summary

- **Layout-grounded parsing (EN/ZH) is close to saturated** on OmniDocBench (top open models 94–96 overall) and olmOCR-bench (85–88 % pass). Off-the-shelf models are good enough here; the differentiator is bbox-grounding fidelity, which only ParseBench, RealDocBench (layout track), ViDoRe V3 (grounding F1) and OmniDocBench layout mAP score directly.
- **Hindi/Devanagari printed OCR is bimodal.** On rendered sentences dots.ocr and Gemini reach ~90–94 % Acc@5 on Devanagari (GlotOCR) while PaddleOCR-VL-1.5, DeepSeek-OCR-2 and GLM-OCR collapse to ~0–30 % on the same script. On real degraded scans even the best model (Gemini 2.5 Flash) reaches chrF++ ≈ 86; DeepSeek-OCR ≈ 10. Indian-built models (Sarvam Vision 2.1, Bodhan IndicOCR) lead a Sarvam-authored Indic bench (87.4 vs Gemini 79.4).
- **Hindi handwriting has no credible off-the-shelf solution and no public page-level benchmark with VLM results.** Public data is word-level (IIIT-HW-Dev ≈ 95 K words, 12 writers) or very new (DohaScript, ICDAR-2025 IHDR ≈ 10 K camera pages). Custom training is required.
- **Charts, infographics, maps:** DocVQA (97) and ChartQA (90) are saturated; ChartQAPro (~72), CharXiv-reasoning (~62–66), InfographicVQA (~89) and MapQA/MapVerse (~50 proprietary, < 30 open) still discriminate.
- **Schema/KV extraction:** FUNSD/CORD/SROIE are saturated by layout-LMs (F1 0.88–0.96); RealDocBench and ParseBench are where agentic extractors are compared. Domain schemas still need a few hundred labelled docs for LoRA.
- **Doc-RAG:** MMLongBench-Doc (~60 acc), M3DocVQA (~37 F1), ViDoRe V3 (~60–64 nDCG@10) remain hard; MP-DocVQA (0.84 ANLS) is nearly solved by retrieval + VLM.

## End-to-end parsing / layout / grounding

| Benchmark | Measures | Size | Metric | Access | Languages | Representative top scores |
|---|---|---|---|---|---|---|
| **OmniDocBench v1.5→1.6/1.7** (OpenDataLab) | Page→markdown: text, tables, formulas, reading order, layout bboxes (28 block types); handwritten-notes subset ≈ 116 pages | 1,651 pages (v1.0); v1.5 rebalanced EN/ZH | Overall = ((1−text ED)·100 + TEDS + CDM)/3; reading-order ED; layout mAP | HF `opendatalab/OmniDocBench`, Apache-2.0 | EN, ZH | v1.6: PaddleOCR-VL-1.6 96.3, MinerU2.5-Pro 95.8, GLM-OCR 95.2, PaddleOCR-VL-1.5 94.9, Sarvam Vision 2.1 95.0, Gemini 3 Pro 92.9, dots.ocr 90.8, DeepSeek-OCR-2 90.3, Qwen3-VL-235B 89.8 |
| **olmOCR-bench** (AI2) | PDF→text via 7,010 unit tests (presence/absence, order, tables, math) | 1,403 PDFs, 7 categories | % tests passed | HF `allenai/olmOCR-bench`, ODC-BY | EN | Infinity-Parser2-Pro 87.6, Sarvam Vision 2.1 87.3, Chandra-2 85.8, dots.mocr 83.9, olmOCR-v0.4 82.4, PaddleOCR-VL ≈ 80, dots.ocr 79.1, MinerU2.5 75–77, DeepSeek-OCR 75.7 |
| **ParseBench** (LlamaIndex, 2026) | Agent-oriented parsing: tables, charts, faithfulness, formatting, **visual grounding** | ~2,000 enterprise pages, 167 K rules | TableRecordMatch, rule tests, chart point checks | HF + GitHub | EN | LlamaParse Agentic 84.9; Infinity-Parser2-Pro 74.3; most parsers < 6 % on charts |
| **RealDocBench** (Extend AI, 2026) | Field-level QA on regulated docs + 9-class layout detection | 1,356 Q / 581 pages; layout 1,500 pages | per-field accuracy; strict/adjusted F1 | HF `Extend-AI/realdoc-benchmarks` | EN | Extend v2 96.0/90.9; Claude Opus 4.8 93.0/87.1; LlamaParse 92.2/84.5; Chandra-2 (best OSS) 86.2/78.1; layout F1 0.835 |
| **MORE** (Tencent, 2026) | Multilingual parsing, 149 languages incl. Indic | ~1,250 pages | NED, CDM, TEDS | GitHub | 149 incl. Hindi | Hindi: HunyuanOCR 95.9, dots.ocr 95.4, PaddleOCR-VL 95.4, Qwen3-VL-2B 95.3, DeepSeek-OCR 73.7, MinerU2.5 34.0; Indic-family avg 72 (hardest group) |
| **GlotOCR Bench** (2026) | Sentence-image OCR, 158 scripts, clean + degraded | 16,375 sentences (Devanagari 400) | CER, Acc@5 | HF `cis-lmu/glotocr-bench` | 158 scripts | **Devanagari Acc@5:** Gemini 3.1 Flash-Lite 93.5, dots.ocr 90.2, Qwen3-VL-8B 70.8, GPT-4.1 65.8, DeepSeek-OCR-2 28.5, PaddleOCR-VL-1.5 1.0, GLM-OCR 0.2 |
| **Devanagari stress test** (arXiv 2606.29213) | Printed Devanagari on real degraded scans + synthetic | 300 real + 2,000 synthetic | CER/WER/chrF++ | GitHub `Aditya-PS-05/devanagari-ocr-benchmark` | Hindi/Sanskrit | Real chrF++: Gemini 2.5 Flash 86.3, Claude Opus 4.7 82.2, Mistral OCR 77.6, Qwen3-VL-8B 75.2, EasyOCR 58.3, olmOCR-7B 40.5, DeepSeek-OCR 10.4 |
| **Sarvam Indic OCR Bench** (2026) | Indic document OCR, 22 languages | 6,909 samples | accuracy | HF `sarvamai/indic-ocr-bench` | 22 Indic | Sarvam Vision 2.1 87.4, Bodhan IndicOCR 84.9, Gemini 79.4, Google CV 71.8 (vendor-authored) |
| **Bharat Scene Text (BSTD)** | Indic scene text | 6,582 images, 126 K words | TedEval F1, WRR | project site | 11 Indic + EN | E2E WRR: Google OCR 0.41, IndicPhotoOCR 0.36, Surya 0.14, Tesseract 0.02 |
| **OCRBench v2** | 31 scenarios incl. localisation, handwriting | 10 K QA | score/100 | GitHub | EN, ZH | most LMMs < 50 |
| **DocLayNet** (IBM) | Layout detection, 11 classes | 80 K pages | COCO mAP | HF `docling-project/DocLayNet-v1.1`, CDLA-Permissive | EN-centric | Docling heron-101 0.78 |

## Tables

| Benchmark | Measures | Size | Metric | Access | Top scores |
|---|---|---|---|---|---|
| **PubTabNet** | image→HTML (scientific) | 320 K/68 K/9 K | TEDS, TEDS-S | IBM, CDLA-Permissive | MTL-TabNet TEDS-S 97.9 |
| **FinTabNet** | financial-report tables | 88 K/10 K/10 K | TEDS | IBM | MTL-TabNet 98.8 |
| **MUSTARD** (SPRINT 2025) | script-agnostic TSR, 11 Indic scripts | 1,428 tables | TEDS-S | GitHub `IITB-LEAP-OCR/SPRINT` | SPRINT 85.2 vs MTL-TabNet 74.1 |
| **RD-TableBench** (Reducto) | complex real tables, scanned/handwritten/merged | 1,000 tables | hierarchical alignment similarity | HF `reducto/rd-tablebench` | Reducto 90.2 |
| **OmniDocBench tables** | ~1 K tables | TEDS | see above | PaddleOCR-VL-1.5 92.8; Tensorlake 86.8; Textract 80.8; Azure 78.1 |
| **IndoTabVQA** (2026) | cross-lingual table-image QA incl. **Hindi** | 1,910 tables / 6,372 QA | in-match acc | HF `NusaBharat/INDOTABVQA` | GPT-4o 41.1 (Hindi 26.0); LoRA Qwen2.5-VL-7B + bbox priors 48.5 |

## VQA over documents, charts, infographics, maps

| Benchmark | Size | Metric | Access | Top scores |
|---|---|---|---|---|
| **DocVQA** | 12,767 images / 50 K Q | ANLS | `lmms-lab/DocVQA` | Qwen3-VL-235B 97.1, Gemini 2.5 Pro 94.0; human 98.1 |
| **InfographicVQA** | 5,485 infographics / 30 K Q | ANLS | same | Qwen3-VL-235B 89.5, Gemini 2.5 Pro 84.2 |
| **ChartQA** | 20.9 K charts / 32.7 K Q | relaxed acc | `HuggingFaceM4/ChartQA` (GPL-3) | Qwen3-VL 90.3 — saturated |
| **ChartQAPro** | 1,341 charts / 1,948 Q | relaxed acc / EM / ANLS | `ahmed-masry/ChartQAPro` | Claude Opus 4.8 72.3 |
| **CharXiv** | 2,323 charts | EM + judged descriptive | `princeton-nlp/CharXiv` CC-BY-SA | GPT-5 reasoning 81.1; Qwen3-VL-235B 66.1 |
| **MapQA** / **MapVerse** / **MapEval** | 4.2 K / 11.8 K / 700 Q | LLM-judge / acc | project pages | proprietary ≈ 45–50, open < 30; nobody > 67 % on MapEval |
| **OmniHandwritingOCR** (CIKM'26) | 77.5 K EN/ZH handwriting pairs | 1-NED, CER | GitHub | Qwen3-VL-8B 72.2; Nanonets-OCR2 65.5; MinerU2.5 18.3 |

## Key-value / schema extraction

| Benchmark | Size | Metric | Access | Top scores |
|---|---|---|---|---|
| **FUNSD** | 199 forms | entity F1 | non-commercial | LayoutLMv3-large ≈ 0.92 |
| **CORD** | 1,000 receipts | F1 | `naver-clova-ix/cord-v2` CC-BY | LiLT 0.961 |
| **SROIE** | 973 receipts | F1 | ICDAR 2019 | ≈ 0.95 |
| **XFUND** | 1,393 forms, 7 languages, **no Indic** | SER/RE F1 | CC-BY-NC-SA | LiLT 0.79–0.89 |
| **Kleister NDA/Charity** | 540 / 2,778 long docs | F1 | GitHub | LAMBERT 81.8 / 83.6 |

## Multi-page / RAG / retrieval

| Benchmark | Size | Metric | Access | Top scores |
|---|---|---|---|---|
| **MP-DocVQA** | 46 K Q / 48 K pages | ANLS, page acc | `lmms-lab/MP-DocVQA` | M3DocRAG 0.844 |
| **DUDE** | 5 K docs / 41.5 K Q | ANLS, ECE | `jordyvl/DUDE_loader` | human 74.8; models ~50 |
| **MMLongBench-Doc** | 130 docs / 1,062 Q (avg 49 pp) | acc | HF Space | Claude 4.5 Opus 61.9; Qwen3-VL-235B 57.0 |
| **M3DocVQA** | 3,368 PDFs / 2,441 Q | EM, F1 | `bloomberg/m3docrag` | M3DocRAG F1 36.5 |
| **ViDoRe V2 / V3** | V3: 10 corpora, 26 K pages, 3,099 Q × 6 langs, bbox grounding F1 | nDCG@10 | `vidore/*` | nemotron-colembed-vl-8b 63.4; text pipeline + reranker 63.6 |

## Handwriting

| Benchmark | Size | Metric | Access | Notes |
|---|---|---|---|---|
| **IAM** | 13 K lines, 657 writers | CER/WER | registration | HTR-JAND CER 1.23; GPT-4o-mini CER 1.71 |
| **IIIT-HW-Dev (Hindi)** | ~95 K words, 12 writers | CER/WER | cvit.iiit.ac.in | CNN-BiLSTM WER 9.2 / CER 2.0 (in-vocab) |
| **IIIT-INDIC-HW-WORDS** | 868 K words, 135 writers, 10 scripts | CER/WER | cvit.iiit.ac.in / AIKosh | CER 1.0–4.9 |
| **ICDAR 2025 IHDR** | ~10 K camera pages, ~1,000 writers, 10 Indic languages | page-level | academic | only page-level Indic set |
| **DohaScript** (2026) | 531 Hindi pages × 531 writers | none yet | GitHub `KAS4453/DohaScript` | only 29.6 % of pages segment cleanly |
| **AnciDev** | 500 ancient Devanagari pages | CER | ACL | fine-tuned HTR needed |

**Gaps:** no public benchmark scores Hindi *handwriting* with modern VLMs; XFUND and ViDoRe have no Indic; chart/map sets are English-only. A private bilingual, degraded-scan, handwritten-form eval set is unavoidable — see `docs/eval_plan.md`.

## Do off-the-shelf document VLMs handle Devanagari / handwriting?

| Model | Devanagari evidence | Handwriting | Verdict |
|---|---|---|---|
| PaddleOCR-VL 1.5/1.6 (0.9 B, Apache-2.0) | vendor ED 0.097; **independent GlotOCR Acc@5 = 1.0**; dropped from stress test for Latin output | EN/ZH only | best EN/ZH layout+tables; **Devanagari unreliable**; no fine-tuning support yet |
| dots.ocr (1.7 B, MIT) | **best open on Devanagari**: GlotOCR 90.2; MORE Hindi 95.4 | n/a; repetition-loop failure mode | best OSS pick for printed Hindi with bbox layout JSON |
| DeepSeek-OCR / -2 | GlotOCR 28.5; stress test 10.4 | weak | avoid for Indic |
| GLM-OCR (0.9 B, MIT) | GlotOCR 0.2 | internal only | EN/ZH |
| MinerU 2.5 | MORE Hindi 34.0 | worst on OmniHandwritingOCR | not for Indic/HW |
| Chandra-2 (Datalab, OpenRAIL-M) | vendor Hindi 78.4 | marketed, no numbers; best OSS on RealDocBench QA | promising generalist; licence limits commercial use |
| Qwen3-VL 2B–235B (Apache-2.0) | GlotOCR 70.8 (8B); stress 75.2; MORE 95.3 | **best on OmniHandwritingOCR** (72.2) | best general base for LoRA |
| Surya | BSTD WRR 0.14 | — | no public training code |
| Sarvam Vision 2.1 (API) / Bodhan IndicOCR (open weights `bodhan-ai/indic-doc-parser`) | 87.4 / 84.9 on Sarvam bench | claimed, no public numbers | only models built for Indic docs; evaluate Bodhan first |
| Gemini 2.5/3.x, Claude Opus 4.x, GPT-5.x | Gemini best on real Devanagari scans (chrF++ 86) | strong Latin zero-shot | strongest Hindi printed OCR available |

Devanagari-specific failure modes: conjunct clusters, matras, nukta, look-alikes (ब/व, घ/ध, म/भ, द/ध), cross-script hallucination, runaway repetition on low-DPI input.

## Fine-tuning options and data budgets

| Target | Option | Data needed (evidence) |
|---|---|---|
| Printed Devanagari | PP-OCRv5 `devanagari_PP-OCRv5_mobile_rec` retrain | base trained on 3.6 K images; realistic: synthetic lines + 10–50 K real crops |
| Printed Devanagari | LoRA on olmOCR/Qwen-VL | 28.8 K word crops → CER 17.4 % (enclave-scribe) — insufficient alone; need line/page data |
| Hindi handwriting | staged TrOCR-class: 100 K synthetic → 5 K printed → **≥ 2.5–3 K real lines** (8× aug) → CER ≈ 4.9 % on one manuscript domain | first production specialist 5–15 K crops, many writers |
| Hindi handwriting | writer/domain adaptation of an existing Devanagari HTR | gains from 16 lines; ~40 % rel. CER with 256 lines; full FT beats LoRA for TrOCR |
| Domain schemas | LoRA Qwen3-VL 4–8 B with layout bboxes as spatial priors | **100–500 labelled docs per schema** → +10–30 pts over zero-shot |
| Indic tables | fine-tune TSR (SPRINT) or rely on bbox-grounded VLM + validation rules | MUSTARD best 85 TEDS-S |

## Recommendation

Use off-the-shelf (benchmark on your own pages): EN layout/tables/formulas (PaddleOCR-VL, MinerU2.5, dots.ocr, Chandra-2); printed Hindi/bilingual (dots.ocr open, Gemini closed, evaluate Bodhan/Sarvam); charts/infographics/maps (frontier VLMs — open 8 B models trail badly); multi-page retrieval (ColQwen3 / nemotron-colembed or text pipeline + reranker); English handwriting (Qwen3-VL-8B / frontier zero-shot).

Train/fine-tune: **Hindi handwriting** (no off-the-shelf evidence — staged recipe above), **domain schemas** (100–500 docs LoRA), **degraded Devanagari scans** if dots.ocr/Gemini miss the bar, **Indic table structure**.

Evaluation stack: OmniDocBench + olmOCR-bench (parsing), ParseBench/RealDocBench (grounding + field QA), GlotOCR Devanagari slice + Devanagari stress-test (printed Hindi), IIIT-HW-Dev + ICDAR-2025 IHDR (Hindi handwriting), ChartQAPro + CharXiv + InfographicVQA, MapVerse, MMLongBench-Doc + ViDoRe V3, plus the private set.

## Sources

Parsing/layout: github.com/opendatalab/OmniDocBench · arxiv.org/pdf/2412.07626 · benchlm.ai/md/benchmarks/omnidocbench15.md · codesota.com/ocr/benchmark/olmocr-bench · huggingface.co/datasets/allenai/olmOCR-bench · llamaindex.ai/blog/parsebench · arxiv.org/abs/2606.07401 (RealDocBench) · arxiv.org/pdf/2607.02956 (MORE) · alphaxiv.org/abs/2604.12978 (GlotOCR) · arxiv.org/pdf/2606.29213 (Devanagari stress test) · arxiv.org/abs/2501.00321v2 (OCRBench v2) · arxiv.org/pdf/2509.11720 (Docling layout) · tensorlake.ai/blog/benchmarks · landing.ai/blog/docvqa-benchmark · alphaxiv.org/abs/2607.07836 (Infinity-Parser2)

Indic OCR/handwriting: cvit.iiit.ac.in/research/projects/cvit-projects/indic-hw-data · ilocr.iiit.ac.in/icdar_2025_Indic_HDR · arxiv.org/html/2602.18089v1 (DohaScript) · aclanthology.org/2025.bhasha-1.8 (AnciDev) · arxiv.org/pdf/2512.17111 (Old Nepali HTR) · arxiv.org/pdf/2205.02543 · arxiv.org/pdf/2511.23071 (BSTD) · sarvam.ai/blogs/sarvam-vision-2-1 · analyticsvidhya.com/blog/2026/09/bodhan-ai-indic-models · alphaxiv.org/abs/2604.11970 (IndoTabVQA) · huggingface.co/enclavelabs/enclave-scribe-devanagari

Handwriting: codesota.com/benchmark/iam · arxiv.org/pdf/2608.18586 (OmniHandwritingOCR) · arxiv.org/abs/2503.15195v2 · arxiv.org/html/2503.19546v1 · WACV 2025 VisionDocs (LoRA vs FT for HTR)

Tables: arxiv.org/pdf/2503.11932 (SPRINT) · github.com/reductoai/RD-TableBench · arxiv.org/html/2408.09174v2 (TableBench)

VQA/charts/maps: alphaxiv.org/benchmarks/amazon/docvqa · WACV 2022 InfographicVQA · benchmarklist.com/benchmarks/chartqa_pro · aimslab.stanford.edu/measurement-db/charxiv · arxiv.org/pdf/2511.21631 (Qwen3-VL report) · iclr.cc/virtual/2026/10013241 (MapQA) · arxiv.org/html/2602.10518 (MapVerse) · arxiv.org/abs/2501.00316v1 (MapEval)

KIE: arxiv.org/pdf/2202.13669 (LiLT) · arxiv.org/pdf/2408.06345 · arxiv.org/pdf/2105.05796 (Kleister) · hyperscience.ai blog (VLM fine-tuning) · croz.net QLoRA series · dida.do/projects/fewture

RAG/retrieval: arxiv.org/pdf/2212.05935 (MP-DocVQA) · arxiv.org/pdf/2305.08455 (DUDE) · benchmarklist.com/benchmarks/mmlongbench_doc · arxiv.org/html/2411.04952v1 (M3DocRAG) · huggingface.co/blog/manu/vidore-v2 · arxiv.org/pdf/2601.08620v1 (ViDoRe V3)

Models: arxiv.org/pdf/2510.14528 & 2601.21957 (PaddleOCR-VL) · paddleocr.ai PP-OCRv5 multilingual · arxiv.org/pdf/2510.18234 (DeepSeek-OCR) · github.com/datalab-to/chandra · arxiv.org/pdf/2509.22186 (MinerU2.5)
