import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("DOCINTEL_LLM_PROVIDER", "none")
os.environ.setdefault("DOCINTEL_DATA_DIR", str(ROOT / "tests" / "_data"))


@pytest.fixture(scope="session")
def samples(tmp_path_factory):
    out = tmp_path_factory.mktemp("samples")
    subprocess.run([sys.executable, str(ROOT / "scripts" / "make_samples.py"), str(out)], check=True)
    return out


def test_schemas_bbox_roundtrip():
    from docintel.schemas import BBox
    b = BBox.from_abs(10, 20, 110, 220, 1000, 2000)
    assert b.to_abs(1000, 2000) == (10, 20, 110, 220)
    assert abs(b.iou(b) - 1.0) < 1e-9


def test_metrics():
    from docintel.eval import metrics
    assert metrics.ned("hello world", "hello world") == 0
    assert metrics.anls("Total 42", ["total 42"]) == 1.0
    assert metrics.relaxed_accuracy("41.5", "42") == 1.0
    assert metrics.relaxed_accuracy("50", "42") == 0.0
    t = "<table><tr><th>a</th><th>b</th></tr><tr><td>1</td><td>2</td></tr></table>"
    assert metrics.teds(t, t) == 1.0
    assert 0 < metrics.teds(t, "<table><tr><td>1</td></tr></table>") < 1


def test_langdetect():
    from docintel.ocr.langdetect import detect_languages, detect_scripts, is_bilingual
    assert detect_scripts("Invoice कर चालान") == ["Latin", "Devanagari"] or set(detect_scripts("Invoice कर चालान")) == {"Latin", "Devanagari"}
    assert set(detect_languages("Invoice कर चालान")) == {"en", "hi"}
    assert is_bilingual("Payment due भुगतान देय")


def test_parse_native_pdf(samples):
    from docintel.pipeline import Pipeline
    from docintel.export.rag import rag_chunks
    from docintel.export.ragdoc import to_ragdoc
    pipe = Pipeline()
    doc = pipe.parse(samples / "sample_invoice.pdf", schema={"invoice_number": "Invoice ID", "total_amount": "Grand total"})
    types = {c.type.value for c in doc.all_chunks()}
    assert "table" in types and "title" in types
    assert any(c.type.value == "chart" for c in doc.all_chunks())
    table = next(c for c in doc.all_chunks() if c.type.value == "table")
    assert table.table is not None and table.table.n_rows >= 4
    assert "Cotton saree" in table.text
    ex = doc.extraction
    assert ex.field("invoice_number").value == "INV-2026-00417"
    assert ex.field("total_amount").groundings and ex.field("total_amount").groundings[0].page == 1
    assert ex.field("invoice_number").verified
    recs = rag_chunks(doc)
    assert all({"id", "text", "page", "bbox", "metadata"} <= set(r) for r in recs)
    rd = to_ragdoc(doc, samples / "sample_invoice.pdf")
    assert rd["blocks"] and all(b["kind"] in {"text", "heading", "table", "list", "caption", "figure"} for b in rd["blocks"])
    assert any(b["kind"] == "table" and b["modality"] == "table" for b in rd["blocks"])


def test_parse_scanned_png(samples):
    from docintel.pipeline import Pipeline
    doc = Pipeline().parse(samples / "sample_invoice_scan.png", schema={"invoice_number": "Invoice ID"})
    assert doc.pipeline["page_sources"]["1"] == "tesseract"
    text = " ".join(c.text for c in doc.all_chunks())
    assert "INV-2026-00417" in text
    assert doc.extraction.field("invoice_number").value == "INV-2026-00417"


def test_api_roundtrip(samples):
    from fastapi.testclient import TestClient
    from docintel.api.server import app
    client = TestClient(app)
    with open(samples / "sample_invoice.pdf", "rb") as f:
        r = client.post("/api/documents", files={"file": ("sample_invoice.pdf", f, "application/pdf")})
    assert r.status_code == 200
    doc_id = r.json()["doc_id"]
    # background task runs synchronously in TestClient
    d = client.get(f"/api/documents/{doc_id}").json()
    assert d["n_pages"] == 1 and len(d["pages"][0]["chunks"]) > 3
    assert client.get(f"/api/documents/{doc_id}/pages/1").status_code == 200
    ex = client.post(f"/api/documents/{doc_id}/extract", json={"schema": {"gstin": "GST number"}}).json()
    assert ex["fields"][0]["value"] == "08AAACB1234F1Z5"
    assert client.get(f"/api/documents/{doc_id}/chunks").json()
    assert client.get(f"/api/documents/{doc_id}/export").headers["content-type"] == "application/zip"
    assert client.get("/").status_code == 200
