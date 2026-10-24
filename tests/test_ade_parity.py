"""Checks against the ADE capability list: hierarchical JSON, schema-first typed extraction (incl. arrays),
visual grounding with page + bbox + cropped snippet, context (labels / row / column), verification, single-call API."""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("DOCINTEL_LLM_PROVIDER", "none")
os.environ.setdefault("DOCINTEL_DATA_DIR", str(ROOT / "tests" / "_data"))

SCHEMA = {"type": "object", "properties": {
    "invoice_number": {"type": "string", "description": "Invoice ID"},
    "invoice_date": {"type": "string", "format": "date", "description": "Date issued"},
    "gstin": {"type": "string", "description": "GST identification number"},
    "total_amount": {"type": "number", "description": "Grand total amount"},
    "currency": {"type": "string", "enum": ["INR", "USD", "EUR"], "description": "Currency of the total amount"},
    "line_items": {"type": "array", "items": {"type": "object", "properties": {
        "description": {"type": "string"}, "quantity": {"type": "integer", "description": "Qty"},
        "unit_price": {"type": "number"}, "amount": {"type": "number"}}}},
    "missing_field": {"type": "string", "description": "Something not in the document"},
}}


@pytest.fixture(scope="module")
def samples(tmp_path_factory):
    out = tmp_path_factory.mktemp("samples")
    subprocess.run([sys.executable, str(ROOT / "scripts" / "make_samples.py"), str(out)], check=True)
    return out


def test_normalize():
    from docintel.extraction.normalize import to_bool, to_date, to_number, coerce
    assert to_number("INR 1,07,520.00") == 107520.0
    assert to_number("₹ 850") == 850.0 and to_number("(1,200)") == -1200.0 and to_number("abc") is None
    assert to_date("14/10/2026") == "2026-10-14" and to_date("Oct 14, 2026") == "2026-10-14"
    assert to_date("14 अक्टूबर 2026") == "2026-10-14" and to_date("२०२६-१०-१४") == "2026-10-14"
    assert to_bool("हाँ") is True and to_bool("No") is False
    assert coerce("Indian Rupee", {"type": "string", "enum": ["INR", "USD"]}) == ("Indian Rupee", False, "not one of ['INR', 'USD']")
    assert coerce("inr", {"type": "string", "enum": ["INR", "USD"]})[0] == "INR"
    assert coerce("40", {"type": "integer"}) == (40, True, None)
    assert coerce("forty", {"type": "integer"})[1] is False


def test_hierarchy_typed_fields_grounding(samples):
    from docintel.pipeline import Pipeline
    pipe = Pipeline()
    doc = pipe.parse(samples / "sample_invoice.pdf", schema=SCHEMA)

    # 1. hierarchical JSON: title → sections → chunk ids; every content chunk has a parent + section attribute
    assert doc.tree is not None and doc.tree.children and doc.tree.children[0].level == 0
    sections = {n.title for n in doc.tree.children[0].children}
    assert "Line items" in sections
    assert all(c.attributes.get("section") is not None for c in doc.all_chunks()
               if c.type.value not in ("page_header", "page_footer", "page_number"))

    ex = doc.extraction
    f = {x.name: x for x in ex.fields}
    # 2. schema-first typed output: number, ISO date, array of typed objects, null for missing
    assert f["total_amount"].value == 107520.0 and f["total_amount"].type_ok
    assert f["invoice_date"].value == "2026-10-14"
    assert f["missing_field"].value is None
    items = f["line_items"].value
    assert len(items) == 3 and items[0] == {"description": "Cotton saree (handloom)", "quantity": 40, "unit_price": 850.0, "amount": 34000.0}
    assert f["line_items"].verified and f["line_items"].type_ok
    assert ex.data["line_items"][2]["amount"] == 38400.0

    # 3. visual grounding: page + bbox + cropped snippet for each found value; cell-level for table values
    for name in ("invoice_number", "gstin", "total_amount", "line_items"):
        g = f[name].groundings[0]
        assert g.page == 1 and 0 <= g.bbox.x0 < g.bbox.x1 <= 1
        assert g.snippet_path and Path(g.snippet_path).exists()
    assert f["total_amount"].groundings[0].bbox.area() < 0.02          # the value's own box, not a whole block
    assert "Total" in (f["total_amount"].context.get("label") or f["total_amount"].context.get("row_label") or "")
    assert f["invoice_number"].context.get("label") == "Invoice Number"
    assert f["invoice_number"].context.get("section")
    # 4. verification
    assert f["invoice_number"].verified and f["gstin"].verified


@pytest.mark.skipif(shutil.which("soffice") is None, reason="LibreOffice not installed")
def test_lab_report_cell_grounding(tmp_path):
    """The ADE example: '12.5 g/dL' is the Hemoglobin value in the results table, tied to row + column + section."""
    from docintel.pipeline import Pipeline
    html = tmp_path / "report.html"
    html.write_text("""<h1>Lab Report</h1><p>Patient Name: Asha Verma</p><p>Date: 15 March 2026</p>
<h2>Results</h2><table border=1><tr><th>Test</th><th>Result</th><th>Unit</th><th>Range</th></tr>
<tr><td>Hemoglobin</td><td>12.5</td><td>g/dL</td><td>12-16</td></tr><tr><td>WBC</td><td>7800</td><td>/µL</td><td>4000-11000</td></tr></table>
<p>Remarks: within normal limits</p>""", encoding="utf-8")
    doc = Pipeline().parse(html, schema={"type": "object", "properties": {
        "patient_name": {"type": "string"}, "hemoglobin_value": {"type": "number", "description": "Hemoglobin result"},
        "date": {"type": "string", "format": "date"}}})
    assert any(c.type.value == "table" for c in doc.all_chunks())
    hb = doc.extraction.field("hemoglobin_value")
    assert hb.value == 12.5 and hb.verified and hb.type_ok
    assert hb.context["row_label"] == "Hemoglobin" and hb.context["column_header"] == "Result" and hb.context["section"] == "Results"
    assert hb.groundings[0].bbox.area() < 0.005 and Path(hb.groundings[0].snippet_path).exists()
    assert doc.extraction.field("date").value == "2026-03-15"


def test_strict_mode_nulls_unverified(samples, monkeypatch):
    from docintel.config import Settings
    from docintel.pipeline import Pipeline
    s = Settings()
    s.strict_grounding = True
    doc = Pipeline(s).parse(samples / "sample_invoice.pdf", schema={"invoice_number": "Invoice ID"})
    fv = doc.extraction.field("invoice_number")
    assert fv.value == "INV-2026-00417" and fv.verified     # verified values survive strict mode
    # simulate an unverifiable value
    from docintel.extraction import schema as S
    monkeypatch.setattr(S, "_verify", lambda *a, **k: (False, "forced"))
    doc = Pipeline(s).parse(samples / "sample_invoice.pdf", schema={"invoice_number": "Invoice ID"})
    assert doc.extraction.field("invoice_number").value is None


def test_single_call_api(samples):
    from fastapi.testclient import TestClient
    from docintel.api.server import app
    client = TestClient(app)
    with open(samples / "sample_invoice.pdf", "rb") as fh:
        r = client.post("/api/extract", files={"file": ("sample_invoice.pdf", fh, "application/pdf")},
                        data={"schema": json.dumps(SCHEMA), "name": "invoice"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["extraction"]["schema_name"] == "invoice"
    assert body["extraction"]["data"]["total_amount"] == 107520.0
    assert len(body["extraction"]["data"]["line_items"]) == 3
    assert client.get(body["snippets"]["invoice_number"]).headers["content-type"] == "image/png"
    tree = client.get(f"/api/documents/{body['doc_id']}/tree").json()
    assert tree["children"]


def test_extraction_modes(samples):
    from fastapi.testclient import TestClient
    from docintel.api.server import app
    from docintel.pipeline import Pipeline
    client = TestClient(app)
    pdf = samples / "sample_invoice.pdf"
    # parse only → no extraction
    doc = Pipeline().parse(pdf)
    assert doc.extraction is None and doc.metadata["extraction_mode"] == "parse"
    # schema mode requires a schema
    with pytest.raises(ValueError):
        Pipeline().parse(pdf, mode="schema")
    # auto mode proposes a schema (ids stay strings, dates typed, table → array) and extracts against it
    doc = Pipeline().parse(pdf, mode="auto")
    sc = doc.metadata["suggested_schema"]["properties"]
    assert sc["invoice_number"]["type"] == "string" and sc["invoice_date"].get("format") == "date"
    assert sc["line_items"]["type"] == "array" and "unit_price" in sc["line_items"]["items"]["properties"]
    assert doc.extraction.schema_name == "auto" and doc.extraction.data["invoice_number"] == "INV-2026-00417"
    assert doc.extraction.data["ifsc"] == "SBIN0000648"
    # API: mode is explicit on upload, and suggest-schema works on an already parsed document
    with open(pdf, "rb") as fh:
        r = client.post("/api/documents", files={"file": ("sample_invoice.pdf", fh, "application/pdf")}, data={"mode": "schema"})
    assert r.status_code == 400
    with open(pdf, "rb") as fh:
        r = client.post("/api/documents", files={"file": ("sample_invoice.pdf", fh, "application/pdf")}, data={"mode": "parse"})
    doc_id = r.json()["doc_id"]
    assert r.json()["mode"] == "parse" and client.get(f"/api/documents/{doc_id}").json()["extraction"] is None
    sug = client.post(f"/api/documents/{doc_id}/suggest-schema").json()
    assert "gstin" in sug["properties"]
    ex = client.post(f"/api/documents/{doc_id}/extract", json={}).json()      # no schema → auto
    assert ex["schema_name"] == "auto" and ex["data"]["gstin"] == "08AAACB1234F1Z5" and ex["schema"]["properties"]
    with open(pdf, "rb") as fh:
        r = client.post("/api/extract", files={"file": ("sample_invoice.pdf", fh, "application/pdf")}, data={"mode": "auto"})
    assert r.json()["mode"] == "auto" and r.json()["schema"]["properties"] and r.json()["extraction"]["data"]["line_items"]
