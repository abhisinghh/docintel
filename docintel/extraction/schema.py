"""Schema-driven field extraction with visual grounding (ADE-style).

Input : a ParsedDocument + a JSON Schema (or a flat {field: description} dict).
Output: ExtractionResult whose every FieldValue carries
        - a value typed per the schema (number / integer / boolean / date → ISO-8601 / enum / array of objects)
        - the raw source span,
        - groundings: page + bbox (table *cell* or word-run when available, else the chunk) + cropped snippet image,
        - context: section heading, form label, table row label / column header,
        - verified flag from an independent re-check, type_ok flag from coercion.

Flow (plan → retrieve → extract → ground → verify → normalise):
1. *Plan*: flatten the schema into leaf fields; arrays of objects (e.g. invoice_items) stay one field.
2. *Retrieve*: rank chunks by lexical similarity to the field name/description (rapidfuzz) + all tables/form fields.
3. *Extract*: one LLM call for all fields with candidate chunks (ids included) → value + chunk_id + span.
   Array fields: the LLM maps table rows to item objects; offline, columns are matched to item properties by header.
   Offline fallback for scalars: key/value and table-cell heuristics. The pipeline works with no LLM.
4. *Ground*: locate the span in the chunk's table cells (→ cell bbox + row/column labels) or words (→ tight box).
5. *Verify*: textual match against the grounded chunk, else VLM re-read of the crop. Unverified → confidence × 0.6
   (or value nulled when strict=True).
6. *Normalise*: coerce to the declared type/format/enum; failures keep the raw value with type_ok=False.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Optional

from PIL import Image
from rapidfuzz import fuzz, process

from ..schemas import BBox, Chunk, ChunkType, ExtractionResult, FieldValue, Grounding, ParsedDocument
from .normalize import coerce

# ------------------------------------------------------------------ schema handling


def flatten_schema(schema: dict[str, Any], prefix: str = "") -> list[dict[str, Any]]:
    """JSON-Schema → list of leaf fields {path, type, format, description, enum, items}.
    Objects are flattened with dotted paths; arrays are kept as a single (list-valued) field."""
    fields: list[dict[str, Any]] = []
    if "properties" in schema:
        for name, sub in schema["properties"].items():
            path = f"{prefix}{name}"
            if not isinstance(sub, dict):                 # {"properties": {"x": "a description"}}
                sub = {"type": "string", "description": str(sub)}
            if sub.get("type") == "object" and "properties" in sub:
                fields.extend(flatten_schema(sub, path + "."))
            else:
                fields.append({"path": path, "type": sub.get("type", "string"), "format": sub.get("format"),
                               "description": sub.get("description", ""), "enum": sub.get("enum"),
                               "items": sub.get("items")})
    else:  # flat {field: description}
        for name, desc in schema.items():
            fields.append({"path": f"{prefix}{name}", "type": "string", "format": None,
                           "description": desc if isinstance(desc, str) else "", "enum": None, "items": None})
    return fields


def _set_path(d: dict, path: str, value: Any) -> None:
    parts = path.split(".")
    for p in parts[:-1]:
        d = d.setdefault(p, {})
    d[parts[-1]] = value


def _label(field: dict[str, Any]) -> str:
    return field["path"].split(".")[-1].replace("_", " ")


# ------------------------------------------------------------------ retrieval


def candidate_chunks(doc: ParsedDocument, field: dict[str, Any], k: int = 8) -> list[Chunk]:
    chunks = [c for c in doc.all_chunks() if c.text.strip()]
    if not chunks:
        return []
    query = f"{_label(field)} {field['path'].replace('.', ' ')} {field['description']}"
    scored = process.extract(query, {c.id: c.text for c in chunks}, scorer=fuzz.token_set_ratio, limit=k)
    ids = {cid for _, _, cid in scored}
    for c in chunks:   # values hide in tables and key/value rows
        if c.type in (ChunkType.FORM_FIELD, ChunkType.TABLE) and len(ids) < k + 6:
            ids.add(c.id)
    return [c for c in chunks if c.id in ids]


# ------------------------------------------------------------------ grounding


def ground_span(chunk: Chunk, span: str) -> Optional[BBox]:
    """Find `span` among the chunk's words and return the union bbox of the best-matching run."""
    if not span or not chunk.words:
        return None
    target = re.sub(r"\s+", " ", span).strip().lower()
    words = chunk.words
    n = len(target.split())
    best, best_score = None, 0
    for i in range(len(words)):
        for L in range(max(1, n - 1), min(len(words), i + n + 2) - i + 1):
            cand = " ".join(w.text for w in words[i:i + L]).lower()
            score = fuzz.ratio(cand, target)
            if score > best_score:
                best, best_score = (i, i + L), score
            if score == 100:
                break
    if best and best_score >= 80:
        run = words[best[0]:best[1]]
        box = run[0].bbox
        for w in run[1:]:
            box = box.union(w.bbox)
        return box
    return None


def ground_cell(chunk: Chunk, span: str) -> tuple[Optional[BBox], dict[str, Any]]:
    """If the chunk is a table, find the cell holding `span`; return its bbox and row/column labels."""
    if not chunk.table or not span:
        return None, {}
    grid = chunk.table.to_grid()
    best, best_score = None, 0
    for c in chunk.table.cells:
        if not c.text.strip():
            continue
        if _present(span, c.text) and (len(span) >= 6 or c.text.strip().lower() == span.strip().lower()):
            s = 100 if c.text.strip().lower() == span.strip().lower() else 90
        else:
            s = fuzz.ratio(span.lower(), c.text.lower())
        if s > best_score:
            best, best_score = c, s
    if best is None or best_score < 85:
        return None, {}
    ctx: dict[str, Any] = {"row": best.row, "col": best.col}
    if grid and best.row < len(grid):
        ctx["row_label"] = next((x for x in grid[best.row] if x.strip()), "") if best.col > 0 else ""
        ctx["column_header"] = grid[0][best.col] if best.col < len(grid[0]) else ""
    if chunk.table.caption:
        ctx["table_caption"] = chunk.table.caption
    return best.bbox, ctx


def context_for(chunk: Chunk, span: str) -> dict[str, Any]:
    ctx: dict[str, Any] = {}
    if chunk.attributes.get("section"):
        ctx["section"] = chunk.attributes["section"]
    if chunk.type == ChunkType.FORM_FIELD and ":" in chunk.text:
        ctx["label"] = chunk.text.split(":", 1)[0].strip()
    else:   # "Label: value" on the same line as the span
        for line in chunk.text.split("\n"):
            if span and span.lower() in line.lower() and ":" in line:
                ctx["label"] = line.split(":", 1)[0].strip()[:80]
                break
    ctx["chunk_type"] = chunk.type.value
    return ctx


def save_snippet(page_image: Image.Image, bbox: BBox, out: Path, pad: int = 6) -> str:
    x0, y0, x1, y1 = bbox.to_abs(*page_image.size)
    crop = page_image.crop((max(0, int(x0) - pad), max(0, int(y0) - pad),
                            min(page_image.width, int(x1) + pad), min(page_image.height, int(y1) + pad)))
    out.parent.mkdir(parents=True, exist_ok=True)
    crop.save(out)
    return str(out)


# ------------------------------------------------------------------ offline fallback


_STOP = {"the", "of", "and", "a", "an", "in", "for", "with", "to", "value", "result", "info", "details"}
_ALIAS = {"no": "number", "num": "number", "nbr": "number", "dt": "date", "amt": "amount", "qty": "quantity",
          "desc": "description", "addr": "address", "tel": "phone", "mob": "phone", "dob": "birth", "ref": "reference"}
NEXT_KEY_RE = re.compile(r"\s{2,}[A-Za-z\u0900-\u097F][\w ()/.-]{1,40}[:：]\s|\s+(?=[A-Z][a-z]+(?: [A-Z][a-z]+)?[:：]\s)")


def _value_only(v: str) -> str:
    """'1,250.50 and Date: 15/03/2026' → '1,250.50 and' is still wrong; cut at the next 'Key:' on the line."""
    m = NEXT_KEY_RE.search(v)
    return v[:m.start()].strip() if m else v.strip()


def _key_match(cell: str, keys: list[str]) -> bool:
    """Does this label cell name the field?  Token based: at least 2/3 of the key's content tokens must appear
    (fuzzily) in the label, so 'Invoice ID' does not match 'Invoice Date' and 'Unit' does not match 'unit price'."""
    norm = lambda t: _ALIAS.get(t, t)  # noqa: E731
    cell_toks = [norm(t) for t in re.sub(r"[^\w\s]", " ", cell.lower()).split() if t]
    if not cell_toks or len("".join(cell_toks)) < 3:
        return False
    for k in keys:
        k_toks = [norm(t) for t in re.sub(r"[^\w\s]", " ", k.lower()).split() if t and t not in _STOP]
        if not k_toks:
            continue
        present = sum(1 for kt in k_toks if any(fuzz.ratio(kt, ct) >= 85 or (len(kt) >= 4 and kt in ct) for ct in cell_toks))
        if present * 3 >= len(k_toks) * 2 and present >= 1 and len(cell_toks) <= len(k_toks) + 3:   # ≥ 2/3 of key tokens
            return True
        # a bare label equal to the field's head noun ("Date" for invoice_date) when nothing else qualifies the label
        if len(cell_toks) == 1 and len(k_toks) >= 2 and cell_toks[0] == k_toks[-1]:
            return True
    return False


def _offline_scalar(field: dict[str, Any], cands: list[Chunk]) -> Optional[tuple[Any, Chunk, str]]:
    name = _label(field)
    keys = [name] + [w for w in field["description"].split(",") if w.strip()][:3]
    if name in ("title", "document title", "heading"):
        for c in cands:
            if c.type == ChunkType.TITLE and c.text.strip():
                return c.text.strip(), c, c.text.strip()

    def kv(line: str):
        # a line may hold several pairs: "Amount: 1,250.50   Date: 15/03/2026"
        for m in re.finditer(r"([^:|]{1,60}?)\s*[:：]\s*([^:：]+?)(?=\s{2,}[^:：]{1,40}[:：]|\s+[A-Z][a-z]+[:：]|$)", line.strip()):
            label = m.group(1).strip()
            label = label.split("  ")[-1] if "  " in label else label
            if _key_match(label, keys):
                return _value_only(m.group(2))
        return None

    for c in cands:
        lines = c.text.split("\n") if not c.table else [" ".join(x for x in row if x.strip()) for row in c.table.to_grid()]
        for line in lines:
            v = kv(line)
            if v:
                return v, c, v
        if c.table:
            for row in c.table.to_grid():
                cells = [x for x in row if x.strip()]
                for i, cell in enumerate(cells[:-1]):
                    if _key_match(cell, keys) and cells[i + 1]:
                        return cells[i + 1], c, cells[i + 1]
    return None


def _offline_array(field: dict[str, Any], doc: ParsedDocument) -> Optional[tuple[list[dict], Chunk]]:
    """Map the best table's columns onto the item schema's properties by header similarity."""
    props = ((field.get("items") or {}).get("properties")) or {}
    tables = [c for c in doc.all_chunks() if c.table and c.table.n_rows >= 2]
    if not props or not tables:
        return None
    best, best_score, best_map = None, 0, {}
    for t in tables:
        header = t.table.to_grid()[0]
        mapping, score = {}, 0
        for prop, sub in props.items():
            q = f"{prop.replace('_', ' ')} {sub.get('description', '')}"
            hit = process.extractOne(q, {i: h for i, h in enumerate(header) if h.strip()}, scorer=fuzz.token_set_ratio)
            if hit and hit[1] >= 60:
                mapping[prop] = hit[2]
                score += hit[1]
        if len(mapping) >= max(1, len(props) // 2) and score > best_score:
            best, best_score, best_map = t, score, mapping
    if best is None:
        return None
    from .normalize import to_number
    rows = []
    for row in best.table.to_grid()[1:]:
        item = {p: (row[i] if i < len(row) else "") for p, i in best_map.items()}
        filled = sum(1 for v in item.values() if v.strip())
        if filled < max(1, len(item) / 2):
            continue
        # drop summary rows (Subtotal / Total / GST) that land in numeric columns with non-numeric text
        if any(props[p].get("type") in ("number", "integer") and v.strip() and to_number(v) is None
               for p, v in item.items()):
            continue
        rows.append(item)
    return rows, best


# ------------------------------------------------------------------ main

EXTRACT_SYSTEM = (
    "You are a meticulous document-extraction agent. Extract ONLY values that are literally present in the "
    "provided document chunks. Never guess. If a value is absent return null. Keep original language/script "
    "unless the field asks for normalisation (dates → ISO 8601, numbers → plain digits). For array fields return "
    "a list of objects built from the table rows or repeated blocks, one object per row."
)


def _best_chunk_for(span: str, cands: list[Chunk]) -> Optional[Chunk]:
    if not span or not cands:
        return None
    return max(cands, key=lambda c: fuzz.partial_ratio(span.lower(), c.text.lower()))


def _present(needle: str, hay: str) -> bool:
    """Is `needle` in `hay`?  Short needles must match a whole token (so '5' is not 'verified' by '2015')."""
    n, h = needle.lower().strip(), hay.lower()
    if not n:
        return False
    if len(n) < 6 or n.isdigit():
        toks = re.findall(r"[\w.,/-]+", h)
        return any(t == n or t.strip(".,") == n for t in toks)
    return fuzz.partial_ratio(n, h) >= 90


def _verify(value: Any, chunk: Chunk, page_image: Optional[Image.Image], bbox: Optional[BBox], vlm,
            raw_text: Optional[str] = None) -> tuple[bool, Optional[str]]:
    """Verify the *source span* (what the model says it read), falling back to the value itself; normalised
    values (ISO dates, bare numbers) legitimately differ from the page text."""
    if isinstance(value, list):   # arrays: every item value must be found in the grounded chunk
        cells = [str(x) for r in value for x in (r.values() if isinstance(r, dict) else [r]) if str(x).strip()]
        if not cells:
            return False, "empty list"
        found = sum(1 for c in cells if _present(c, chunk.text))
        return (found / len(cells) >= 0.9), f"{found}/{len(cells)} item values found in source table"
    for cand in (raw_text, str(value)):
        if cand and _present(str(cand), chunk.text):
            return True, "matched chunk text"
    val = str(raw_text or value)
    if vlm is not None and page_image is not None and bbox is not None:
        try:
            seen = vlm.read_region(page_image, bbox)
            if fuzz.partial_ratio(val.lower(), seen.lower()) >= 85:
                return True, "matched VLM re-read"
            return False, f"VLM re-read differs: {seen[:120]}"
        except Exception as e:  # pragma: no cover
            return False, f"verify error: {e}"
    return False, "could not verify"


def extract_fields(doc: ParsedDocument, schema: dict[str, Any], llm=None, vlm=None,
                   page_images: Optional[dict[int, Image.Image]] = None, verify: bool = True,
                   schema_name: str = "default", snippets_dir: Optional[Path] = None,
                   strict: bool = False) -> ExtractionResult:
    fields = flatten_schema(schema)
    result = ExtractionResult(schema_name=schema_name)
    page_images = page_images or {}
    if not doc.tree:
        from ..schemas import build_tree
        doc.tree = build_tree(doc)

    # ---- extract (batched LLM call) -------------------------------------------------
    plan: dict[str, Any] = {}
    cand_map: dict[str, list[Chunk]] = {f["path"]: candidate_chunks(doc, f) for f in fields}
    if llm is not None and getattr(llm, "available", True):
        all_cands: dict[str, Chunk] = {c.id: c for cs in cand_map.values() for c in cs}
        for c in doc.all_chunks():   # arrays usually live in tables → always include them
            if c.table:
                all_cands[c.id] = c
        ctx = "\n\n".join(f"[chunk {c.id} | page {c.page} | {c.type.value}]\n{c.text[:2500]}" for c in all_cands.values())
        spec = []
        for f in fields:
            line = (f"- {f['path']} ({f['type']}{', format ' + f['format'] if f.get('format') else ''}"
                    f"{', one of ' + str(f['enum']) if f['enum'] else ''}): {f['description']}")
            if f["type"] == "array" and (f.get("items") or {}).get("properties"):
                line += " — items: " + ", ".join(f"{k} ({v.get('type', 'string')}: {v.get('description', '')})"
                                                 for k, v in f["items"]["properties"].items())
            spec.append(line)
        prompt = (
            f"Document chunks:\n{ctx}\n\nFields to extract:\n" + "\n".join(spec) + "\n\n"
            "Return JSON: {\"fields\": [{\"path\": ..., \"value\": ..., \"chunk_id\": ..., "
            "\"source_span\": \"the exact text in the chunk the value was read from\", \"confidence\": 0-1}]}"
        )
        try:
            out = llm.complete_json(prompt, system=EXTRACT_SYSTEM)
            items = out.get("fields", []) if isinstance(out, dict) else out
            for item in items or []:
                if isinstance(item, dict) and item.get("path"):
                    plan[str(item["path"])] = item
        except Exception as e:
            result.data["_llm_error"] = str(e)

    # ---- ground + verify + normalise --------------------------------------------------------
    for f in fields:
        path = f["path"]
        item = plan.get(path)
        value, chunk, span, conf = None, None, None, 0.0
        if item and item.get("value") not in (None, "", []):
            value = item["value"]
            try:
                conf = max(0.0, min(1.0, float(item.get("confidence", 0.7))))
            except (TypeError, ValueError):
                conf = 0.7
            span = item.get("source_span") or (str(value) if not isinstance(value, list) else "")
            chunk = doc.chunk(item.get("chunk_id") or "") or _best_chunk_for(span, cand_map[path] or doc.all_chunks())
        elif f["type"] == "array":
            off = _offline_array(f, doc)
            if off:
                value, chunk = off
                conf, span = 0.5, ""
        else:
            off = _offline_scalar(f, cand_map[path])
            if off:
                value, chunk, span = off
                conf = 0.5
        fv = FieldValue(name=path, value=value, raw_text=span or None, confidence=round(conf, 3))
        if chunk is not None:
            box, cell_ctx = ground_cell(chunk, span or "")
            if box is None:
                box = ground_span(chunk, span or "") or chunk.bbox
            fv.context = {**context_for(chunk, span or ""), **cell_ctx}
            if cell_ctx:
                fv.context.pop("label", None)   # row/column labels are the context for table cells
            g = Grounding(page=chunk.page, bbox=box, chunk_id=chunk.id, snippet_path=chunk.image_path)
            img = page_images.get(chunk.page)
            if snippets_dir is not None and img is not None:
                safe = re.sub(r"[^A-Za-z0-9_.-]", "_", path)
                g.snippet_path = save_snippet(img, box, snippets_dir / f"field_{safe}.png")
            fv.groundings.append(g)
            if verify and value is not None:
                fv.verified, note = _verify(value, chunk, img, box, vlm, raw_text=span)
                fv.notes = note
                if not fv.verified:
                    fv.confidence = round(fv.confidence * 0.6, 3)
                    if strict:
                        fv.value, fv.notes = None, f"{note}; nulled (strict grounding)"
        if fv.value is not None:
            typed, ok, note = coerce(fv.value, f)
            fv.value, fv.type_ok = typed, ok
            if note:
                fv.notes = f"{fv.notes}; {note}" if fv.notes else note
        result.fields.append(fv)
        _set_path(result.data, path, fv.value)
    return result
