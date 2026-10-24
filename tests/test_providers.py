"""Exercise the model-backed code paths (VLM page parsing, chart extraction, LLM schema extraction, agent loop)
against fake OpenAI-compatible / Anthropic endpoints, so they are tested without a GPU or API key."""
import json
import os
import subprocess
import sys
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("DOCINTEL_DATA_DIR", str(ROOT / "tests" / "_data"))

PAGE_JSON = json.dumps([
    {"type": "title", "bbox": [50, 40, 600, 80], "text": "TAX INVOICE / कर चालान", "confidence": 0.98},
    {"type": "form_field", "bbox": [50, 100, 500, 125], "text": "Invoice Number: INV-2026-00417", "confidence": 0.97},
    {"type": "table", "bbox": [50, 200, 900, 400], "confidence": 0.95,
     "text": "<table><tr><th>Item</th><th>Qty</th></tr><tr><td>Cotton saree</td><td>40</td></tr></table>"},
    {"type": "chart", "bbox": [50, 450, 600, 700], "text": "bar chart of monthly sales", "confidence": 0.9},
    {"type": "handwriting", "bbox": [50, 720, 600, 760], "text": "भुगतान 30 दिनों में", "confidence": 0.6},
    {"type": "text", "bbox": [50, 780, 900, 860], "confidence": 0.95,
     "text": "Bharat Textiles Pvt. Ltd. issued this invoice to the Rajasthan Handloom Cooperative in Kota for handloom goods supplied in October 2026."},
])
CHART_JSON = json.dumps({"chart_type": "bar", "title": "Monthly sales", "x_label": "Month", "y_label": "INR lakh",
                         "series": [{"name": "sales", "x": ["Jun", "Jul"], "y": [4.2, 5.1]}], "summary": "Sales rose from Jun to Jul."})
EXTRACT_JSON = json.dumps({"fields": [{"path": "invoice_number", "value": "INV-2026-00417", "chunk_id": None,
                                       "source_span": "INV-2026-00417", "confidence": 0.95}]})
TRIPLES_JSON = json.dumps({"triples": [{"subject": "Bharat Textiles", "predicate": "issued", "object": "INV-2026-00417", "confidence": 0.9}]})


def fake_reply(prompt: str) -> str:
    if "return ONLY a JSON array" in prompt:
        return PAGE_JSON
    if "Classify this image region" in prompt:
        return "chart"
    if "Extract the data from this chart" in prompt:
        return CHART_JSON
    if "Fields to extract" in prompt:
        return EXTRACT_JSON
    if "knowledge-graph triples" in prompt or "Text (page" in prompt:
        return TRIPLES_JSON
    if "orchestrator" in prompt or "Observation:" in prompt:
        return json.dumps({"tool": "finish", "reason": "ok"})
    if "Transcribe" in prompt:
        return "भुगतान 30 दिनों में देय"
    if "Reply with OK" in prompt:
        return "OK"
    return PAGE_JSON


class FakeClient:
    """Stands in for httpx.Client; answers both OpenAI chat/completions and Anthropic messages shapes."""

    def __init__(self, *a, **kw):
        self.calls = []

    def post(self, path, json=None, files=None, **kw):
        self.calls.append(path)
        if path.endswith("/messages"):
            prompt = json["messages"][-1]["content"][-1]["text"]
            if json.get("system"):
                prompt = json["system"] + "\n" + prompt
            body = {"content": [{"type": "text", "text": fake_reply(prompt)}]}
        else:
            msgs = json["messages"]
            prompt = "\n".join(m["content"] if isinstance(m["content"], str) else m["content"][-1]["text"] for m in msgs)
            body = {"choices": [{"message": {"content": fake_reply(prompt)}}]}
        return httpx.Response(200, json=body, request=httpx.Request("POST", "http://fake" + path))


@pytest.fixture(scope="module")
def sample(tmp_path_factory):
    out = tmp_path_factory.mktemp("s")
    subprocess.run([sys.executable, str(ROOT / "scripts" / "make_samples.py"), str(out)], check=True)
    return out / "sample_invoice_scan.png"


@pytest.mark.parametrize("vlm_provider,llm_provider", [("openai", "openai"), ("anthropic", "anthropic")])
def test_vlm_and_llm_paths(monkeypatch, sample, vlm_provider, llm_provider):
    monkeypatch.setattr(httpx, "Client", FakeClient)
    from docintel.config import Settings
    from docintel.pipeline import Pipeline
    s = Settings()
    s.vlm.enabled, s.vlm.provider, s.vlm.family, s.vlm.api_key = True, vlm_provider, "generic", "k"
    s.llm.provider, s.llm.api_key = llm_provider, "k"
    pipe = Pipeline(s)
    doc = pipe.parse(sample, schema={"invoice_number": "Invoice ID"})
    types = {c.type.value for c in doc.all_chunks()}
    assert {"title", "form_field", "table", "chart", "handwriting"} <= types
    assert doc.pipeline["page_sources"]["1"].startswith("vlm:" + vlm_provider)
    table = next(c for c in doc.all_chunks() if c.type.value == "table")
    assert table.table.n_rows == 2 and "Cotton saree" in table.text
    chart = next(c for c in doc.all_chunks() if c.type.value == "chart")
    assert chart.chart and chart.chart.series[0].y == [4.2, 5.1] and chart.chart.raw_table.n_rows == 3
    title = next(c for c in doc.all_chunks() if c.type.value == "title")
    assert set(title.languages) == {"en", "hi"}
    fv = doc.extraction.field("invoice_number")
    assert fv.value == "INV-2026-00417" and fv.groundings and fv.verified
    assert any(t.predicate == "issued" for t in doc.triples)
    assert "agent_trace" in doc.pipeline


def test_landingai_backend_mapping(monkeypatch, sample):
    class ADEClient(FakeClient):
        def post(self, path, json=None, files=None, **kw):
            body = {"data": {"chunks": [
                {"chunk_type": "text", "markdown": "Invoice Number: INV-1", "grounding": [{"page": 0, "box": {"left": 0.1, "top": 0.1, "right": 0.5, "bottom": 0.15}}]},
                {"chunk_type": "table", "markdown": "<table><tr><td>a</td><td>b</td></tr></table>", "grounding": [{"page": 0, "box": {"left": 0.1, "top": 0.2, "right": 0.9, "bottom": 0.4}}]},
            ]}}
            return httpx.Response(200, json=body, request=httpx.Request("POST", "http://fake" + path))
    monkeypatch.setattr(httpx, "Client", ADEClient)
    from docintel.config import Settings
    from docintel.pipeline import Pipeline
    s = Settings()
    s.vlm.enabled, s.vlm.provider, s.vlm.api_key, s.llm.provider = True, "landingai", "k", "none"
    doc = Pipeline(s).parse(sample)
    assert doc.pipeline["ocr_backend"] == "landingai"
    assert [c.type.value for c in doc.all_chunks()][:2] == ["form_field", "table"]  # heuristics relabel "k: v" lines
    assert abs(doc.all_chunks()[0].bbox.x1 - 0.5) < 1e-9
