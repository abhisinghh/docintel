"""Staged fine-tuning of a TrOCR-style line recognizer for Devanagari handwriting.

Stage A  synthetic rendered lines (synth_lines.py)           → learns the script
Stage B  IIIT-HW-Dev / IIIT-INDIC-HW-WORDS word crops       → learns handwriting strokes
Stage C  your own labelled lines (≥2.5–3K, many writers)    → learns your domain
Each stage initialises from the previous checkpoint. Full fine-tuning (encoder + decoder) beats LoRA for TrOCR.

Data: JSONL rows {"image": "path/to/line.png", "text": "ग्राहक का नाम"}  (same format docintel's eval runner uses)

    pip install transformers datasets accelerate evaluate jiwer albumentations
    python train_trocr.py --train stageA.jsonl --eval dev.jsonl --base microsoft/trocr-base-handwritten --out ckpt/A
    python train_trocr.py --train stageB.jsonl --eval dev.jsonl --base ckpt/A --out ckpt/B
    python train_trocr.py --train stageC.jsonl --eval dev.jsonl --base ckpt/B --out ckpt/C --epochs 30 --aug 8

Note: the TrOCR tokenizer is RoBERTa/English; for Devanagari either (a) swap the decoder tokenizer for a
byte-level or Indic tokenizer (`--tokenizer ai4bharat/IndicBERTv2-MLM-only`) and resize embeddings, or (b) use a
Qwen3-VL-2B base (already multilingual) with the same data via training/schema_lora. Option (a) is implemented below.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from PIL import Image, ImageFilter, ImageOps
from torch.utils.data import Dataset


def load_rows(p: str):
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


def augment(img: Image.Image) -> Image.Image:
    if random.random() < 0.5:
        img = img.rotate(random.uniform(-2, 2), resample=Image.BICUBIC, fillcolor="white", expand=False)
    if random.random() < 0.4:
        img = img.filter(ImageFilter.GaussianBlur(random.uniform(0.2, 1.0)))
    if random.random() < 0.4:
        img = ImageOps.autocontrast(img, cutoff=random.randint(0, 5))
    if random.random() < 0.3:
        w, h = img.size
        img = img.resize((int(w * random.uniform(0.8, 1.1)), h))
    return img


class LineDataset(Dataset):
    def __init__(self, rows, processor, aug: int = 1, max_len: int = 128):
        self.rows, self.processor, self.aug, self.max_len = rows, processor, aug, max_len

    def __len__(self):
        return len(self.rows) * self.aug

    def __getitem__(self, i):
        r = self.rows[i % len(self.rows)]
        img = Image.open(r["image"]).convert("RGB")
        if self.aug > 1 and i >= len(self.rows):
            img = augment(img)
        pixel = self.processor(images=img, return_tensors="pt").pixel_values[0]
        labels = self.processor.tokenizer(r["text"], padding="max_length", max_length=self.max_len,
                                          truncation=True).input_ids
        labels = [t if t != self.processor.tokenizer.pad_token_id else -100 for t in labels]
        return {"pixel_values": pixel, "labels": torch.tensor(labels)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True); ap.add_argument("--eval", required=True)
    ap.add_argument("--base", default="microsoft/trocr-base-handwritten")
    ap.add_argument("--tokenizer", default="ai4bharat/IndicBERTv2-MLM-only")
    ap.add_argument("--out", required=True); ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=5e-5); ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--aug", type=int, default=1)
    a = ap.parse_args()

    from transformers import (AutoTokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments, TrOCRProcessor,
                              VisionEncoderDecoderModel, default_data_collator)
    import evaluate

    processor = TrOCRProcessor.from_pretrained(a.base)
    model = VisionEncoderDecoderModel.from_pretrained(a.base)
    if a.tokenizer and not Path(a.base).exists():   # first stage: swap in an Indic tokenizer
        tok = AutoTokenizer.from_pretrained(a.tokenizer)
        processor.tokenizer = tok
        model.decoder.resize_token_embeddings(len(tok))
        model.config.decoder_start_token_id = tok.cls_token_id or tok.bos_token_id
        model.config.pad_token_id = tok.pad_token_id
        model.config.eos_token_id = tok.sep_token_id or tok.eos_token_id
        model.config.vocab_size = len(tok)
    model.config.max_length = 128
    model.config.num_beams = 4

    train_ds = LineDataset(load_rows(a.train), processor, aug=a.aug)
    eval_ds = LineDataset(load_rows(a.eval), processor)
    cer = evaluate.load("cer")

    def metrics(pred):
        ids, labels = pred.predictions, pred.label_ids
        labels[labels == -100] = processor.tokenizer.pad_token_id
        p = processor.batch_decode(ids, skip_special_tokens=True)
        g = processor.batch_decode(labels, skip_special_tokens=True)
        return {"cer": cer.compute(predictions=p, references=g)}

    args = Seq2SeqTrainingArguments(
        output_dir=a.out, per_device_train_batch_size=a.bs, per_device_eval_batch_size=a.bs,
        num_train_epochs=a.epochs, learning_rate=a.lr, fp16=torch.cuda.is_available(),
        eval_strategy="epoch", save_strategy="epoch", predict_with_generate=True,
        load_best_model_at_end=True, metric_for_best_model="cer", greater_is_better=False,
        logging_steps=50, save_total_limit=2, warmup_ratio=0.05,
    )
    trainer = Seq2SeqTrainer(model=model, args=args, train_dataset=train_ds, eval_dataset=eval_ds,
                             data_collator=default_data_collator, compute_metrics=metrics)
    trainer.train()
    trainer.save_model(a.out)
    processor.save_pretrained(a.out)


if __name__ == "__main__":
    main()
