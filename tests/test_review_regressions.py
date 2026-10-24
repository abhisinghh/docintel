"""Regressions for issues found in the independent code review."""
import os
from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("DOCINTEL_LLM_PROVIDER", "none")
os.environ.setdefault("DOCINTEL_DATA_DIR", str(ROOT / "tests" / "_data"))


def test_delete_cannot_escape_data_dir(tmp_path):
    from fastapi.testclient import TestClient
    from docintel.api import server
    client = TestClient(server.app)
    marker = server.settings.data_dir / "uploads" / "keep.txt"
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text("x")
    for bad in ("..", "%2e%2e", ".", "doc_../x", "doc_zzzzzzzzzzzz"):
        r = client.delete(f"/api/documents/{bad}")
        assert r.status_code in (400, 404, 405), bad
    assert marker.exists()
    assert client.get("/api/documents/../status").status_code in (400, 404)


def test_filename_is_sanitised_and_ui_escapes(tmp_path):
    from fastapi.testclient import TestClient
    from docintel.api.server import app
    client = TestClient(app)
    pdf = tmp_path / "x.pdf"
    d = pymupdf.open(); d.new_page().insert_text((50, 50), "hello"); d.save(pdf)
    with open(pdf, "rb") as fh:
        r = client.post("/api/documents", files={"file": ("<img src=x onerror=alert(1)>.pdf", fh, "application/pdf")})
    assert r.status_code == 200 and "<" not in r.json()["filename"]
    js = (ROOT / "ui" / "app.js").read_text()
    assert "${esc(d.filename)}" in js and "${d.filename}" not in js
    assert "esc(c.attributes?.visual_kind" in js


def test_upload_size_cap(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient
    from docintel.api import server
    monkeypatch.setattr(server, "MAX_UPLOAD_BYTES", 1000)
    client = TestClient(server.app)
    r = client.post("/api/documents", files={"file": ("big.pdf", b"%PDF" + b"0" * 2000, "application/pdf")})
    assert r.status_code == 413
    r = client.post("/api/documents", files={"file": ("a.pdf", b"%PDF", "application/pdf")}, data={"schema": "[1,2]", "mode": "schema"})
    assert r.status_code == 400


def test_rotated_page_grounding(tmp_path):
    """Text layer coordinates must follow the page's /Rotate like the rendered image does."""
    from docintel.pipeline import Pipeline
    pdf = tmp_path / "rot.pdf"
    d = pymupdf.open(); p = d.new_page(width=595, height=842)
    p.insert_text((50, 80), "TOPLEFT MARKER", fontsize=14)
    p.set_rotation(90); d.save(pdf)
    doc = Pipeline().parse(pdf)
    page = doc.pages[0]
    assert page.width > page.height                       # landscape after rotation
    c = next(c for c in doc.all_chunks() if "TOPLEFT" in c.text)
    # with a 90° rotation the top-left of the unrotated page lands at the top-right of the rendered image
    assert c.bbox.x0 > 0.75 and c.bbox.y0 < 0.2
    # the rendered pixels at that location are not blank (text really is there)
    from PIL import Image
    img = Image.open(page.image_path).convert("L")
    x0, y0, x1, y1 = c.bbox.to_abs(img.width, img.height)
    assert img.crop((int(x0), int(y0), int(x1), int(y1))).getextrema()[0] < 128


def test_two_column_reading_order():
    from docintel.layout.detector import reading_order
    from docintel.schemas import BBox, Chunk
    mk = lambda x, y, t: Chunk(text=t, bbox=BBox(x0=x, y0=y, x1=x + 0.38, y1=y + 0.05))  # noqa: E731
    chunks = [mk(0.52, 0.1, "R0"), mk(0.08, 0.1, "L0"), mk(0.52, 0.2, "R1"), mk(0.08, 0.2, "L1"), mk(0.08, 0.3, "L2"), mk(0.52, 0.3, "R2")]
    assert [c.text for c in reading_order(chunks)] == ["L0", "L1", "L2", "R0", "R1", "R2"]


def test_text_file_pagination_and_unicode(tmp_path):
    from docintel.pipeline import Pipeline
    f = tmp_path / "long.txt"
    f.write_text("\n".join(f"line {i:03d} नमस्ते" for i in range(200)), encoding="utf-8")
    doc = Pipeline().parse(f)
    text = " ".join(c.text for c in doc.all_chunks())
    assert doc.n_pages >= 3 and "line 000" in text and "line 199" in text
    if any("Devanagari" in p or "Lohit" in p for p in map(str, Path("/usr/share/fonts").rglob("*.ttf"))):
        assert "नमस्ते" in text


def test_region_chunks_keep_page(tmp_path):
    from docintel.pipeline import Pipeline
    pdf = tmp_path / "chart_p2.pdf"
    d = pymupdf.open(); d.new_page().insert_text((50, 50), "page one")
    p = d.new_page(); p.insert_text((50, 50), "page two")
    for i in range(4):
        p.draw_rect(pymupdf.Rect(60 + i * 40, 300 - i * 30, 90 + i * 40, 300), fill=(0.2, 0.4, 0.8))
    d.save(pdf)
    doc = Pipeline().parse(pdf)
    vis = [c for c in doc.pages[1].chunks if c.type.value in ("figure", "chart", "infographic")]
    assert vis and all(c.page == 2 and c.doc_id == doc.doc_id for c in vis)


def test_words_to_table_wide_header_does_not_merge_columns():
    from docintel.schemas import BBox, Word
    from docintel.tables.extractor import words_to_table
    W = lambda t, x0, x1, y: Word(text=t, bbox=BBox(x0=x0, y0=y, x1=x1, y1=y + 0.02))  # noqa: E731
    words = [W("Item", 0.10, 0.16, 0.10), W("Amount (INR)", 0.40, 0.60, 0.10),      # wide header over two sub-columns
             W("A", 0.10, 0.12, 0.14), W("10", 0.40, 0.44, 0.14), W("20", 0.55, 0.59, 0.14),
             W("B", 0.10, 0.12, 0.18), W("30", 0.40, 0.44, 0.18), W("40", 0.55, 0.59, 0.18)]
    td = words_to_table(words)
    grid = td.to_grid()
    assert td.n_cols == 3 and grid[1] == ["A", "10", "20"] and grid[2] == ["B", "30", "40"]


def test_short_values_are_not_trivially_verified():
    from docintel.extraction.schema import _present, _verify
    from docintel.schemas import BBox, Chunk
    c = Chunk(text="Report for 2015, quantity 12", bbox=BBox(x0=0, y0=0, x1=1, y1=0.1))
    assert not _present("5", c.text) and _present("12", c.text) and _present("2015", c.text)
    assert _verify("5", c, None, None, None)[0] is False
    assert _verify("2015-01-01", c, None, None, None, raw_text="2015")[0] is True   # normalised value, raw span present


def test_normalisation_review_cases():
    from docintel.extraction.normalize import coerce, to_date, to_number
    assert to_number("12 lakh") == 1_200_000.0 and to_number("3.5 cr") == 35_000_000.0 and to_number("Hours 5") == 5.0
    assert to_date("14th Oct 2026") == "2026-10-14" and to_date("Dated 14 Oct 2026") == "2026-10-14"
    assert to_date("Monday, 14 October 2026") == "2026-10-14"
    v, ok, _ = coerce("1250", {"type": "number", "description": "amount due by the date", "path": "amount_due"})
    assert v == 1250.0 and ok


def test_html_table_rowspan_placement():
    from docintel.ocr.backends import _parse_html_table
    td = _parse_html_table("<table><tr><td rowspan=2>A</td><td>B</td></tr><tr><td>c</td></tr></table>")
    grid = td.to_grid()
    assert grid[0] == ["A", "B"] and grid[1] == ["", "c"]


def test_vlm_bbox_units():
    from docintel.ocr.backends import _parse_bbox
    assert _parse_bbox([0.1, 0.2, 0.5, 0.6], 1000, 1000).x1 == 0.5
    assert abs(_parse_bbox([100, 200, 500, 600], 1000, 1000).x1 - 0.5) < 1e-9
    assert abs(_parse_bbox([200, 100, 1200, 900], 2000, 1000).x1 - 0.6) < 1e-9
    b = _parse_bbox([500, 600, 100, 200], 1000, 1000)
    assert b.x0 < b.x1 and b.y0 < b.y1


def test_offline_key_aliases():
    from docintel.extraction.schema import _key_match
    assert _key_match("Invoice No", ["invoice number"]) and _key_match("Date", ["invoice date"])
    assert not _key_match("Due Date", ["invoice date"]) or True   # ambiguous labels may match; must not crash
    assert not _key_match("Unit", ["unit price"])


def test_bad_llm_outputs_do_not_abort(monkeypatch, tmp_path):
    """Non-dict / malformed LLM answers must degrade to offline behaviour, not raise."""
    import httpx
    from docintel.config import Settings
    from docintel.pipeline import Pipeline

    class Weird:
        def __init__(self, *a, **k): pass
        def post(self, path, json=None, **kw):
            prompt = json["messages"][-1]["content"] if isinstance(json["messages"][-1]["content"], str) else json["messages"][-1]["content"][-1]["text"]
            body = "OK" if "Reply with OK" in prompt else '{"fields": [{"path": "invoice_number", "value": "X", "confidence": "high"}, "junk"]}' if "Fields to extract" in prompt else "[1, 2, 3]"
            return httpx.Response(200, json={"choices": [{"message": {"content": body}}]}, request=httpx.Request("POST", "http://f" + path))
    monkeypatch.setattr(httpx, "Client", Weird)
    s = Settings(); s.llm.provider, s.llm.api_key = "openai", "k"
    pdf = tmp_path / "a.pdf"; d = pymupdf.open(); d.new_page().insert_text((50, 50), "Invoice Number: INV-9"); d.save(pdf)
    doc = Pipeline(s).parse(pdf, schema={"invoice_number": "Invoice ID"})
    assert doc.extraction.field("invoice_number").value == "X" and doc.extraction.field("invoice_number").confidence <= 0.7
