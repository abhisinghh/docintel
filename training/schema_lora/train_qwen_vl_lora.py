"""LoRA fine-tuning of Qwen3-VL (or Qwen2.5-VL) for grounded schema extraction.

Each training example = page image + the fields to extract + the expected JSON answer, with the layout bboxes
of candidate chunks passed as text (spatial priors measurably help, esp. for Hindi: IndoTabVQA +4 pts).

Training rows (JSONL), produced by `python -m training.export_labels`:
  {"image": "data/pages/doc_x/page_0001.png",
   "schema": {"invoice_number": "Invoice ID", "total_amount": "Grand total"},
   "chunks": [{"id": "c1", "type": "form_field", "bbox": [x0,y0,x1,y1], "text": "Invoice Number: INV-1"}, ...],
   "answer": {"fields": [{"path": "invoice_number", "value": "INV-1", "chunk_id": "c1", "source_span": "INV-1"}, ...]}}

    pip install transformers>=4.51 peft accelerate bitsandbytes qwen-vl-utils
    python train_qwen_vl_lora.py --train train.jsonl --eval dev.jsonl --base Qwen/Qwen3-VL-4B-Instruct --out lora/invoices

100–500 labelled documents is the practical budget; evaluate with `docintel eval extraction` on a held-out set.
After training, serve the merged model with vLLM and point DOCINTEL_LLM_MODEL at it.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

SYSTEM = ("You are a meticulous document-extraction agent. Extract ONLY values literally present in the document. "
          "Return JSON {\"fields\": [{\"path\", \"value\", \"chunk_id\", \"source_span\", \"confidence\"}]}.")


def build_prompt(row: dict) -> str:
    chunks = "\n".join(f"[chunk {c['id']} | {c['type']} | bbox {c['bbox']}]\n{c['text'][:800]}" for c in row["chunks"])
    spec = "\n".join(f"- {k}: {v}" for k, v in row["schema"].items())
    return f"Document chunks (with normalised bboxes):\n{chunks}\n\nFields to extract:\n{spec}\n\nReturn ONLY JSON."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True); ap.add_argument("--eval")
    ap.add_argument("--base", default="Qwen/Qwen3-VL-4B-Instruct"); ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=3); ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=16); ap.add_argument("--bs", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=8); ap.add_argument("--load_4bit", action="store_true")
    a = ap.parse_args()

    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForImageTextToText, AutoProcessor, Trainer, TrainingArguments
    from PIL import Image

    quant = None
    if a.load_4bit:
        from transformers import BitsAndBytesConfig
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4")
    processor = AutoProcessor.from_pretrained(a.base)
    model = AutoModelForImageTextToText.from_pretrained(a.base, torch_dtype=torch.bfloat16, quantization_config=quant,
                                                        device_map="auto")
    lcfg = LoraConfig(r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.05, task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, lcfg)
    model.print_trainable_parameters()

    def encode(row: dict):
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": build_prompt(row)}]},
                {"role": "assistant", "content": json.dumps(row["answer"], ensure_ascii=False)}]
        text = processor.apply_chat_template(msgs, tokenize=False)
        img = Image.open(row["image"]).convert("RGB")
        enc = processor(text=[text], images=[img], return_tensors="pt", padding=True)
        enc = {k: v[0] for k, v in enc.items()}
        labels = enc["input_ids"].clone()
        # mask everything before the assistant answer
        prompt_len = len(processor(text=[processor.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True)],
                                   images=[img], return_tensors="pt")["input_ids"][0])
        labels[:prompt_len] = -100
        enc["labels"] = labels
        return enc

    class DS(torch.utils.data.Dataset):
        def __init__(self, p):
            self.rows = [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]
        def __len__(self): return len(self.rows)
        def __getitem__(self, i): return encode(self.rows[i])

    def collate(batch):
        keys = batch[0].keys()
        out = {}
        for k in keys:
            if k in ("input_ids", "attention_mask", "labels"):
                pad = -100 if k == "labels" else (processor.tokenizer.pad_token_id if k == "input_ids" else 0)
                out[k] = torch.nn.utils.rnn.pad_sequence([b[k] for b in batch], batch_first=True, padding_value=pad)
            else:
                out[k] = torch.cat([b[k] for b in batch], dim=0) if batch[0][k].dim() > 1 else torch.stack([b[k] for b in batch])
        return out

    args = TrainingArguments(output_dir=a.out, per_device_train_batch_size=a.bs, gradient_accumulation_steps=a.grad_accum,
                             num_train_epochs=a.epochs, learning_rate=a.lr, bf16=True, logging_steps=10,
                             save_strategy="epoch", eval_strategy="epoch" if a.eval else "no", warmup_ratio=0.05,
                             remove_unused_columns=False, gradient_checkpointing=True, report_to="none")
    trainer = Trainer(model=model, args=args, train_dataset=DS(a.train), eval_dataset=DS(a.eval) if a.eval else None,
                      data_collator=collate)
    trainer.train()
    model.save_pretrained(a.out)
    processor.save_pretrained(a.out)
    print("LoRA adapter saved to", a.out, "— merge with `model.merge_and_unload()` before serving with vLLM")


if __name__ == "__main__":
    main()
