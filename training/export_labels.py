"""Turn reviewed docintel ParsedDocument JSON files into training rows.

    python -m training.export_labels data/parsed/*.json --out train.jsonl            # schema-extraction rows
    python -m training.export_labels data/parsed/*.json --out lines.jsonl --mode lines   # OCR line crops (CER training)

Workflow: parse documents → correct texts/fields in the UI (or edit the JSON) → export → train.
Extraction rows need `doc.extraction` to be present (run with --schema or the UI's Extract tab).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

from docintel.schemas import ParsedDocument


def extraction_rows(doc: ParsedDocument):
    if not doc.extraction:
        return
    by_page: dict[int, list] = {}
    for f in doc.extraction.fields:
        if f.value is None or not f.groundings:
            continue
        by_page.setdefault(f.groundings[0].page, []).append(f)
    for page_no, fields in by_page.items():
        page = doc.page(page_no)
        if page is None or not page.image_path:
            continue
        yield {
            "image": page.image_path,
            "schema": {f.name: f.name.replace("_", " ") for f in fields},
            "chunks": [{"id": c.id, "type": c.type.value, "bbox": [round(c.bbox.x0, 3), round(c.bbox.y0, 3), round(c.bbox.x1, 3), round(c.bbox.y1, 3)],
                        "text": c.text} for c in page.chunks if c.text.strip()],
            "answer": {"fields": [{"path": f.name, "value": f.value, "chunk_id": f.groundings[0].chunk_id,
                                   "source_span": f.raw_text or str(f.value)} for f in fields]},
        }


def line_rows(doc: ParsedDocument, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    for page in doc.pages:
        if not page.image_path:
            continue
        img = Image.open(page.image_path)
        for c in page.chunks:
            if not c.text.strip() or c.type.value in ("figure", "chart", "map", "table"):
                continue
            x0, y0, x1, y1 = c.bbox.to_abs(img.width, img.height)
            crop = img.crop((int(x0), int(y0), int(x1), int(y1)))
            p = out_dir / f"{doc.doc_id}_{c.id}.png"
            crop.save(p)
            yield {"image": str(p), "text": c.text, "languages": c.languages, "type": c.type.value}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+"); ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["extraction", "lines"], default="extraction")
    ap.add_argument("--crops", default="training/_crops")
    a = ap.parse_args()
    n = 0
    with open(a.out, "w", encoding="utf-8") as f:
        for p in a.files:
            doc = ParsedDocument.model_validate_json(Path(p).read_text(encoding="utf-8"))
            rows = extraction_rows(doc) if a.mode == "extraction" else line_rows(doc, Path(a.crops))
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                n += 1
    print("wrote", n, "rows to", a.out)


if __name__ == "__main__":
    main()
