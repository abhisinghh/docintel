"""Schema discovery ("build schema"): propose a JSON Schema from a parsed document when the user has none.

Modes of use
  parse    – no extraction at all (default); the user only wants the grounded parse / markdown / RAG export
  schema   – the user supplies a schema (ADE-style schema-first extraction)
  auto     – no schema: propose one from the document (this module), then extract against it; the proposed
             schema is returned so the user can edit it and re-run deterministically

Offline proposal (no LLM): every `form_field` / "Label: value" line becomes a property named from the label
(snake_case), with the type guessed from the value (number / date / boolean / string); every table becomes an
`array` of objects whose properties come from the header row. With an LLM, the proposal is refined: better names,
descriptions, enums for repeated categorical values, and fields the heuristics missed.
"""
from __future__ import annotations

import re
from typing import Any

from ..schemas import ChunkType, ParsedDocument
from .normalize import to_bool, to_date, to_number

KV_RE = re.compile(r"^\s*([^:|\n]{2,60}?)\s*[:：]\s*(.+?)\s*$")
STOP = {"the", "of", "and", "no", "in", "for"}
NUMERIC_RE = re.compile(r"^\(?[-+]?\s*(₹|\$|€|£|rs\.?|inr|usd|eur)?\s*\d[\d,]*(\.\d+)?\s*(%|₹|\$|€|£|inr|usd|eur|lakh|crore)?\)?$", re.I)
CAPTION_RE = re.compile(r"^(fig(ure)?|table|chart|map|annexure|appendix)\s*\d*", re.I)


def snake(label: str, fallback: str = "field") -> str:
    s = re.sub(r"[^\w\s]", " ", label.lower())
    toks = [t for t in s.split() if t and t not in STOP]
    s = "_".join(toks)[:48].strip("_")
    return s or fallback


def _clean_label(label: str) -> str | None:
    label = re.sub(r"^[\s\-•*·–—]+", "", label).strip()
    if not label or any(ord(ch) < 32 for ch in label) or "\ufffd" in label:
        return None          # garbled text layer
    if CAPTION_RE.match(label):
        return None          # captions are not fields
    letters = sum(ch.isalpha() for ch in label)
    if letters < 0.6 * len(label.replace(" ", "")):
        return None          # OCR noise such as "0000 O00 OO0000"
    return label


ID_LABEL_RE = re.compile(r"\b(phone|mobile|tel|fax|pin|zip|postal|code|id|no\.?|number|account|a/c|ifsc|gstin|pan|aadhaar|"
                         r"passport|licen[cs]e|reference|ref|invoice|order|policy|serial|sku|isbn|iban|swift|cin|tan|uid)\b", re.I)


def guess_type(value: str, label: str = "") -> dict[str, Any]:
    """Conservative: only call something a number/date/boolean when the *whole* value looks like one,
    so identifiers such as INV-2026-00417 or SBIN0000648 stay strings. Labels that denote identifiers
    (phone, PIN, account no., GSTIN ...) are always strings even when the value is all digits."""
    v = value.strip()
    if label and ID_LABEL_RE.search(label) and not re.search(r"amount|total|price|qty|quantity|rate|count|date|time|dated|period", label, re.I):
        return {"type": "string"}
    if to_bool(v) is not None and v.lower() not in ("1", "0", ""):
        return {"type": "boolean"}
    if NUMERIC_RE.match(v) and to_number(v) is not None:
        return {"type": "integer"} if re.fullmatch(r"[-+]?\d{1,9}", v) else {"type": "number"}
    if to_date(v) and re.search(r"\d", v) and not NUMERIC_RE.match(v) and len(v) <= 40:
        return {"type": "string", "format": "date"}
    return {"type": "string"}


def suggest_schema(doc: ParsedDocument, llm=None, max_fields: int = 40) -> dict[str, Any]:
    props: dict[str, Any] = {}
    examples: dict[str, str] = {}
    for c in doc.all_chunks():
        if c.type in (ChunkType.PAGE_HEADER, ChunkType.PAGE_FOOTER, ChunkType.PAGE_NUMBER):
            continue
        if c.table and c.table.n_rows >= 2:
            header = [h.strip() for h in c.table.to_grid()[0]]
            cols = {snake(h, f"col_{i+1}"): h for i, h in enumerate(header) if h and len(h) < 40}
            if len(cols) >= 2:
                sample = c.table.to_grid()[1]
                item_props = {}
                for i, h in enumerate(header):
                    key = snake(h, f"col_{i+1}")
                    if h and key in cols and key not in item_props:
                        item_props[key] = {**guess_type(sample[i] if i < len(sample) else "", h), "description": h}
                name = snake(c.attributes.get("section") or c.table.caption or "table_rows")
                name = f"{name}_rows" if name in props else name
                props[name] = {"type": "array", "description": f"Rows of the table under '{c.attributes.get('section') or ''}'".strip(),
                               "items": {"type": "object", "properties": item_props}}
            continue
        if c.type in (ChunkType.FORM_FIELD, ChunkType.TEXT, ChunkType.LIST_ITEM):
            for line in c.text.split("\n"):
                m = KV_RE.match(line)
                if not m:
                    continue
                label, value = _clean_label(m.group(1)), m.group(2)
                if not label or len(value) > 120 or len(label.split()) > 6:
                    continue
                key = snake(label)
                if key in props or len(props) >= max_fields:
                    continue
                props[key] = {**guess_type(value, label), "description": label.strip()}
                examples[key] = value
    schema = {"type": "object", "properties": props, "_source": "heuristic"}

    if llm is not None and getattr(llm, "available", False):
        try:
            md = doc.markdown[:12000]
            prompt = (
                "Propose a JSON Schema (draft-07 style: type/properties/description/format/enum/items) of the fields a "
                "downstream system would want to extract from this document. Use snake_case names, add a one-line "
                "description per field, use format 'date' for dates, 'number'/'integer' for amounts and counts, "
                "'array' of objects for repeating rows (tables, line items), and 'enum' for small categorical sets. "
                "Keep labels in the document's language in the descriptions. Do not include page furniture.\n\n"
                f"Heuristic draft (improve it, keep good fields):\n{schema}\n\nDocument (markdown):\n{md}"
            )
            out = llm.complete_json(prompt)
            if isinstance(out, dict) and out.get("properties"):
                out.pop("_source", None)
                return {"type": "object", "properties": out["properties"], "_source": "llm"}
        except Exception as e:  # keep the heuristic draft
            schema["_llm_error"] = str(e)
    return schema
