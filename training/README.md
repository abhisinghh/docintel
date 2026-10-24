# Training recipes

The research (see `docs/benchmarks.md`) says: **do not train** for English layout/tables/printed
Hindi (use dots.ocr / Qwen3-VL / a frontier VLM), **do train** for

1. **Hindi (Devanagari) handwriting** — no off-the-shelf model has public evidence of working.
2. **Domain-specific schema extraction** — LoRA on 100–500 labelled docs gives +10–30 points.
3. (if needed) degraded printed Devanagari scans and Indic table structure.

| Recipe | Script | Base model | Data | Expected |
|---|---|---|---|---|
| Hindi handwriting, staged | `htr_devanagari/train_trocr.py` | TrOCR-base-handwritten or Qwen3-VL-2B | 100 K synthetic lines → IIIT-HW-Dev words → **≥ 2.5–3 K real lines from your forms** (8× aug) | CER ≈ 5 % in-domain (Old-Nepali recipe); plan 5–15 K crops for production |
| Schema extraction LoRA | `schema_lora/train_qwen_vl_lora.py` | Qwen3-VL-4B/8B-Instruct | 100–500 docs labelled with `{field: value}` + page images (bboxes as spatial priors) | +10–30 pts field accuracy vs zero-shot |
| Synthetic Devanagari lines | `htr_devanagari/synth_lines.py` | — | fonts + Hindi corpus (IndicCorp / FLORES) | pre-training stage only |

Label format for all recipes is the docintel `ParsedDocument` JSON (page images + chunks/fields with bboxes), so you can
**label inside the docintel UI** (correct a chunk's text / a field's value) and export training rows with
`python -m training.export_labels data/parsed/*.json --out train.jsonl`.

Hardware: a single 24 GB GPU is enough for TrOCR full fine-tuning and Qwen3-VL-4B LoRA (4-bit); 8B LoRA wants 40–80 GB.
Select checkpoints on **field accuracy / CER on a held-out writer set**, not on loss. Keep a mixed English + printed
Hindi slice in every training mix to avoid regressions on printed text.
