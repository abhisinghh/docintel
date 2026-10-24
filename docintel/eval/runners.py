"""Benchmark runners.

Each runner (a) runs the docintel pipeline over a dataset, (b) writes
predictions in the format the benchmark's *official* scorer expects, and
(c) where the metric is simple, computes it directly with ``metrics.py``.

    python -m docintel.eval.runners docvqa      --split validation --limit 200
    python -m docintel.eval.runners chartqa     --limit 200
    python -m docintel.eval.runners omnidocbench --data /path/to/OmniDocBench --out preds/
    python -m docintel.eval.runners olmocr-bench --data /path/to/olmOCR-bench --out preds/
    python -m docintel.eval.runners funsd       --data /path/to/FUNSD/testing_data
    python -m docintel.eval.runners handwriting --data lines.jsonl   (IAM / IIIT-HW-Hindi lines → CER/WER)
    python -m docintel.eval.runners extraction  --data gt.jsonl --schema schema.json   (your own labelled docs)

Official scorers (not vendored here):
  OmniDocBench  https://github.com/opendatalab/OmniDocBench   (pdf_validation.py + configs)
  olmOCR-bench  https://github.com/allenai/olmocr  → `python -m olmocr.bench.benchmark --dir <preds>`
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Optional

from PIL import Image

from ..config import get_settings
from ..pipeline import Pipeline
from ..schemas import BBox, ChunkType
from . import metrics


def _qa_answer(pipe: Pipeline, doc, question: str) -> str:
    """Answer a question from the parsed document (retrieval + LLM; offline: best fuzzy chunk)."""
    from rapidfuzz import fuzz, process
    chunks = [c for c in doc.all_chunks() if c.text.strip()]
    if not chunks:
        return ""
    top = process.extract(question, {c.id: c.text for c in chunks}, scorer=fuzz.token_set_ratio, limit=6)
    ctx = "\n\n".join(doc.chunk(cid).text for _, _, cid in top)
    if pipe.llm is not None and getattr(pipe.llm, "available", False):
        try:
            return pipe.llm.complete(
                f"Document excerpts:\n{ctx}\n\nQuestion: {question}\nAnswer with the shortest exact span from the document.",
                max_tokens=64).strip()
        except Exception:
            pass
    return doc.chunk(top[0][2]).text.split("\n")[0][:80]


def run_docvqa(split: str = "validation", limit: Optional[int] = None, dataset: str = "lmms-lab/DocVQA",
               config: str = "DocVQA") -> dict[str, Any]:
    from datasets import load_dataset
    ds = load_dataset(dataset, config, split=split)
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    pipe = Pipeline(get_settings())
    scores, rows = [], []
    for ex in ds:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            ex["image"].save(f.name)
            doc = pipe.parse(f.name, build_kg=False)
        pred = _qa_answer(pipe, doc, ex["question"])
        s = metrics.anls(pred, ex["answers"])
        scores.append(s)
        rows.append({"qid": ex.get("questionId"), "q": ex["question"], "pred": pred, "gt": ex["answers"], "anls": s})
    return {"benchmark": "DocVQA", "split": split, "n": len(scores), "ANLS": sum(scores) / max(1, len(scores)), "rows": rows}


def run_chartqa(split: str = "test", limit: Optional[int] = None) -> dict[str, Any]:
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceM4/ChartQA", split=split)
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    pipe = Pipeline(get_settings())
    scores, rows = [], []
    for ex in ds:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            ex["image"].save(f.name)
            doc = pipe.parse(f.name, build_kg=False)
        pred = _qa_answer(pipe, doc, ex["query"])
        gt = ex["label"][0] if isinstance(ex["label"], list) else ex["label"]
        s = metrics.relaxed_accuracy(pred, gt)
        scores.append(s)
        rows.append({"q": ex["query"], "pred": pred, "gt": gt, "ok": s})
    return {"benchmark": "ChartQA", "n": len(scores), "relaxed_acc": sum(scores) / max(1, len(scores)), "rows": rows}


def run_omnidocbench(data_dir: str, out_dir: str, limit: Optional[int] = None) -> dict[str, Any]:
    """Write one markdown per page image into out_dir, then score with the official OmniDocBench scripts."""
    pipe = Pipeline(get_settings())
    imgs = sorted(Path(data_dir).glob("images/*.jpg")) + sorted(Path(data_dir).glob("images/*.png"))
    if limit:
        imgs = imgs[:limit]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for p in imgs:
        doc = pipe.parse(p, build_kg=False)
        (out / f"{p.stem}.md").write_text(doc.markdown, "utf-8")
    return {"benchmark": "OmniDocBench", "n": len(imgs), "predictions": str(out),
            "next": "clone opendatalab/OmniDocBench, set prediction dir in configs/end2end.yaml, run pdf_validation.py"}


def run_olmocr_bench(data_dir: str, out_dir: str, limit: Optional[int] = None) -> dict[str, Any]:
    """olmOCR-bench expects <out_dir>/<pdf_stem>_pg<N>_repeat1.md next to bench_data/pdfs."""
    pipe = Pipeline(get_settings())
    pdfs = sorted(Path(data_dir).rglob("*.pdf"))
    if limit:
        pdfs = pdfs[:limit]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for p in pdfs:
        doc = pipe.parse(p, build_kg=False)
        from ..export.markdown import document_to_markdown
        for page in doc.pages:
            md = document_to_markdown(type(doc)(pages=[page]), page_breaks=False)
            (out / f"{p.stem}_pg{page.number}_repeat1.md").write_text(md, "utf-8")
    return {"benchmark": "olmOCR-bench", "n_pdfs": len(pdfs), "predictions": str(out),
            "next": "pip install olmocr[bench]; python -m olmocr.bench.benchmark --dir <bench_data> --candidate <preds>"}


def run_funsd(data_dir: str, limit: Optional[int] = None) -> dict[str, Any]:
    """FUNSD: evaluate word-level text (NED) and entity boxes (IoU F1) against annotations."""
    pipe = Pipeline(get_settings())
    ann_dir, img_dir = Path(data_dir) / "annotations", Path(data_dir) / "images"
    files = sorted(ann_dir.glob("*.json"))
    if limit:
        files = files[:limit]
    neds, f1s = [], []
    for a in files:
        gt = json.loads(a.read_text())
        img = img_dir / (a.stem + ".png")
        im = Image.open(img)
        doc = pipe.parse(img, build_kg=False)
        gt_text = " ".join(w["text"] for f in gt["form"] for w in f["words"])
        pred_text = " ".join(c.text for c in doc.all_chunks())
        neds.append(metrics.ned(pred_text, gt_text))
        gt_boxes = [BBox.from_abs(*f["box"], im.width, im.height) for f in gt["form"]]
        pred_boxes = [c.bbox for c in doc.all_chunks() if c.type != ChunkType.FIGURE]
        f1s.append(metrics.match_boxes(pred_boxes, gt_boxes)["f1"])
    return {"benchmark": "FUNSD", "n": len(files), "mean_NED": sum(neds) / max(1, len(neds)),
            "box_F1@0.5": sum(f1s) / max(1, len(f1s))}


def run_handwriting(lines_jsonl: str, limit: Optional[int] = None) -> dict[str, Any]:
    """Line-level handwriting: JSONL rows {"image": path, "text": gt}. Works for IAM, IIIT-HW-Hindi, your own data."""
    pipe = Pipeline(get_settings())
    rows = [json.loads(l) for l in Path(lines_jsonl).read_text(encoding="utf-8").splitlines() if l.strip()]
    if limit:
        rows = rows[:limit]
    cers, wers = [], []
    for r in rows:
        doc = pipe.parse(r["image"], build_kg=False)
        pred = " ".join(c.text for c in doc.all_chunks()).strip()
        cers.append(metrics.cer(pred, r["text"]))
        wers.append(metrics.wer(pred, r["text"]))
    return {"benchmark": "handwriting", "n": len(rows), "CER": sum(cers) / max(1, len(cers)), "WER": sum(wers) / max(1, len(wers))}


def run_extraction(gt_jsonl: str, schema_path: str, limit: Optional[int] = None) -> dict[str, Any]:
    """Schema extraction on your own labelled docs: JSONL rows {"file": path, "fields": {name: value}, "boxes": {name: [page,x0,y0,x1,y1]}?}."""
    pipe = Pipeline(get_settings())
    schema = json.loads(Path(schema_path).read_text())
    rows = [json.loads(l) for l in Path(gt_jsonl).read_text(encoding="utf-8").splitlines() if l.strip()]
    if limit:
        rows = rows[:limit]
    exact, fuzzy, grounded, n = 0, 0.0, 0, 0
    for r in rows:
        doc = pipe.parse(r["file"], schema=schema, build_kg=False)
        for name, gt in r["fields"].items():
            fv = doc.extraction.field(name) if doc.extraction else None
            pred = "" if fv is None or fv.value is None else str(fv.value)
            n += 1
            exact += int(metrics.normalize_text(pred) == metrics.normalize_text(str(gt)))
            fuzzy += 1 - metrics.ned(pred, str(gt))
            box = (r.get("boxes") or {}).get(name)
            if box and fv and fv.groundings:
                g = fv.groundings[0]
                if g.page == box[0] and g.bbox.iou(BBox(x0=box[1], y0=box[2], x1=box[3], y1=box[4])) >= 0.3:
                    grounded += 1
    return {"benchmark": "schema_extraction", "n_fields": n, "exact_match": exact / max(1, n),
            "fuzzy_match": fuzzy / max(1, n), "grounding_hit@0.3": grounded / max(1, n)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("benchmark", choices=["docvqa", "chartqa", "omnidocbench", "olmocr-bench", "funsd", "handwriting", "extraction"])
    ap.add_argument("--data"); ap.add_argument("--out", default="preds"); ap.add_argument("--split", default=None)
    ap.add_argument("--limit", type=int); ap.add_argument("--schema"); ap.add_argument("--report", default="eval_report.json")
    a = ap.parse_args(argv)
    if a.benchmark == "docvqa":
        res = run_docvqa(a.split or "validation", a.limit)
    elif a.benchmark == "chartqa":
        res = run_chartqa(a.split or "test", a.limit)
    elif a.benchmark == "omnidocbench":
        res = run_omnidocbench(a.data, a.out, a.limit)
    elif a.benchmark == "olmocr-bench":
        res = run_olmocr_bench(a.data, a.out, a.limit)
    elif a.benchmark == "funsd":
        res = run_funsd(a.data, a.limit)
    elif a.benchmark == "handwriting":
        res = run_handwriting(a.data, a.limit)
    else:
        res = run_extraction(a.data, a.schema, a.limit)
    Path(a.report).write_text(json.dumps(res, ensure_ascii=False, indent=1), "utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "rows"}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main(sys.argv[1:])
