"""FastAPI service + static UI.

POST /api/documents                 upload (multipart: file, mode=parse|schema|auto, optional schema JSON) → {doc_id}
                                    parse  = layout/OCR/grounding only (default)
                                    schema = also extract the fields of the supplied JSON schema
                                    auto   = propose a schema from the document, then extract against it
POST /api/documents/{id}/suggest-schema   propose a schema for an already parsed document (editable, re-runnable)
GET  /api/documents                 list
GET  /api/documents/{id}            full ParsedDocument (status while running)
GET  /api/documents/{id}/status     job status
GET  /api/documents/{id}/pages/{n}  page image (PNG)
GET  /api/documents/{id}/crops/{chunk_id}  figure/chart crop
POST /api/documents/{id}/extract    body: {"schema": {...}, "name": "invoice"} → ExtractionResult with groundings
POST /api/extract                   ADE-style single call: multipart file + schema → {doc_id, extraction, markdown}
GET  /api/documents/{id}/tree       hierarchical structure (title → sections → chunk ids)
GET  /api/documents/{id}/snippets/{field}   cropped image snippet of an extracted field's grounding
GET  /api/documents/{id}/export     zip with chunks.jsonl / triples.jsonl / graph.cypher / document.md
GET  /api/documents/{id}/chunks     RAG records (what RAG_over_doc ingests)
GET  /api/documents/{id}/search?q=  grounded search inside the document
DELETE /api/documents/{id}
"""
from __future__ import annotations

import io
import json
import os
import re
import shutil
import threading
import zipfile
from pathlib import Path
from typing import Any, Optional

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from ..config import get_settings
from ..ingest.loaders import IMAGE_EXT, OFFICE_EXT, TEXT_EXT
from ..export.rag import export_all, rag_chunks
from ..pipeline import Pipeline
from ..schemas import JobStatus, ParsedDocument, new_id

app = FastAPI(title="docintel", version="0.1.0")
settings = get_settings()
UI_DIR = Path(__file__).resolve().parents[2] / "ui"

_pipeline: Optional[Pipeline] = None
_jobs: dict[str, JobStatus] = {}
_lock = threading.Lock()


def pipeline() -> Pipeline:
    global _pipeline
    with _pipeline_lock:                      # one model load, even under concurrent first requests
        if _pipeline is None:
            _pipeline = Pipeline(settings)
    return _pipeline


def _parsed_path(doc_id: str) -> Path:
    return settings.data_dir / "parsed" / f"{doc_id}.json"


SUPPORTED_EXT = {".pdf"} | IMAGE_EXT | OFFICE_EXT | TEXT_EXT
SAFE_ID = re.compile(r"^(doc|c|s)_[0-9a-f]{12}$")          # ids are always <prefix>_<12 hex>; never '.' or '..'
MAX_UPLOAD_BYTES = int(os.getenv("DOCINTEL_MAX_UPLOAD_MB", "50")) * 1024 * 1024
_pipeline_lock = threading.Lock()


def _check_upload(file: UploadFile) -> Path:
    name = Path(file.filename or "upload").name
    name = re.sub(r"[^\w. \-\u0080-\uffff]", "_", name)[:120] or "upload"   # keep unicode, drop shell/HTML metachars
    if Path(name).suffix.lower() not in SUPPORTED_EXT:
        raise HTTPException(415, f"unsupported file type {Path(name).suffix!r}; supported: {sorted(SUPPORTED_EXT)}")
    return Path(name)


async def _read_upload(file: UploadFile) -> bytes:
    data = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"file larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB")
    return data


def _parse_schema(schema: Optional[str]) -> Optional[dict[str, Any]]:
    if not schema:
        return None
    try:
        obj = json.loads(schema)
    except json.JSONDecodeError as e:
        raise HTTPException(400, f"schema is not valid JSON: {e}") from e
    if not isinstance(obj, dict) or not obj:
        raise HTTPException(400, "schema must be a non-empty JSON object (JSON Schema or {field: description})")
    return obj


def _safe(seg: str) -> str:
    if not SAFE_ID.match(seg):
        raise HTTPException(400, "invalid id")
    return seg


def _load(doc_id: str) -> ParsedDocument:
    _safe(doc_id)
    p = _parsed_path(doc_id)
    if not p.exists():
        raise HTTPException(404, "document not found or not parsed yet")
    return ParsedDocument.model_validate_json(p.read_text(encoding="utf-8"))


def _run_job(doc_id: str, path: Path, schema: Optional[dict[str, Any]], mode: str = "parse"):
    def prog(stage: str, p: float):
        with _lock:
            _jobs[doc_id].stage, _jobs[doc_id].progress, _jobs[doc_id].status = stage, p, "running"
    try:
        pipeline().parse(path, doc_id=doc_id, schema=schema, progress=prog, mode=mode)
        with _lock:
            _jobs[doc_id].status, _jobs[doc_id].progress = "parsed", 1.0
    except Exception as e:  # pragma: no cover
        with _lock:
            _jobs[doc_id].status, _jobs[doc_id].error = "failed", str(e)


# ------------------------------------------------------------------ routes


@app.post("/api/documents")
async def upload(background: BackgroundTasks, file: UploadFile = File(...), schema: Optional[str] = Form(None),
                 mode: Optional[str] = Form(None)):
    mode = mode or ("schema" if schema else "parse")
    if mode not in ("parse", "schema", "auto"):
        raise HTTPException(400, "mode must be parse | schema | auto")
    if mode == "schema" and not schema:
        raise HTTPException(400, "mode=schema requires a schema")
    name = _check_upload(file)
    schema_obj = _parse_schema(schema)
    data = await _read_upload(file)
    doc_id = new_id("doc")
    dest = settings.data_dir / "uploads" / doc_id
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / name
    path.write_bytes(data)
    with _lock:
        _jobs[doc_id] = JobStatus(doc_id=doc_id, status="queued")
    background.add_task(_run_job, doc_id, path, schema_obj, mode)
    return {"doc_id": doc_id, "filename": path.name, "mode": mode}


@app.post("/api/extract")
async def extract_sync(file: UploadFile = File(...), schema: Optional[str] = Form(None), name: str = Form("default"),
                       include_document: bool = Form(False), mode: Optional[str] = Form(None)):
    """One call: upload a document (+ schema, or mode=auto to discover one), get typed, grounded, verified fields back."""
    mode = mode or ("schema" if schema else "auto")
    if mode not in ("parse", "schema", "auto"):
        raise HTTPException(400, "mode must be parse | schema | auto")
    fname = _check_upload(file)
    schema_obj = _parse_schema(schema)
    data = await _read_upload(file)
    doc_id = new_id("doc")
    dest = settings.data_dir / "uploads" / doc_id
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / fname
    path.write_bytes(data)
    from starlette.concurrency import run_in_threadpool
    doc = await run_in_threadpool(pipeline().parse, path, doc_id=doc_id, schema=schema_obj, build_kg=False, mode=mode)
    if doc.extraction and mode != "auto":
        doc.extraction.schema_name = name
    out: dict[str, Any] = {"doc_id": doc.doc_id, "n_pages": doc.n_pages, "languages": doc.languages, "mode": mode,
                           "schema": doc.metadata.get("suggested_schema") if mode == "auto" else (json.loads(schema) if schema else None),
                           "extraction": doc.extraction.model_dump() if doc.extraction else None,
                           "snippets": {f.name: f"/api/documents/{doc.doc_id}/snippets/{f.name}"
                                        for f in (doc.extraction.fields if doc.extraction else []) if f.groundings}}
    if include_document:
        out["document"] = doc.model_dump()
    else:
        out["markdown"] = doc.markdown
    return out


@app.post("/api/documents/{doc_id}/suggest-schema")
def suggest_schema(doc_id: str):
    return pipeline().suggest_schema(_load(doc_id))


@app.get("/api/documents/{doc_id}/tree")
def tree(doc_id: str):
    doc = _load(doc_id)
    if doc.tree is None:
        from ..schemas import build_tree
        doc.tree = build_tree(doc)
    return doc.tree


@app.get("/api/documents/{doc_id}/snippets/{field}")
def snippet(doc_id: str, field: str):
    import re as _re
    p = settings.data_dir / "crops" / _safe(doc_id) / f"field_{_re.sub(r'[^A-Za-z0-9_.-]', '_', field)[:80]}.png"
    if not p.exists():
        raise HTTPException(404)
    return FileResponse(p, media_type="image/png")


@app.get("/api/documents")
def list_documents():
    items = []
    for p in sorted((settings.data_dir / "parsed").glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            items.append({"doc_id": d["doc_id"], "filename": d["filename"], "n_pages": d["n_pages"],
                          "languages": d.get("languages", []), "status": d.get("status"),
                          "mode": (d.get("metadata") or {}).get("extraction_mode", "parse"),
                          "n_fields": len((d.get("extraction") or {}).get("fields", [])),
                          "n_chunks": sum(len(pg["chunks"]) for pg in d["pages"])})
        except Exception:
            continue
    with _lock:
        jobs = list(_jobs.values())
    for j in jobs:
        if j.status in ("queued", "running", "failed") and not any(i["doc_id"] == j.doc_id for i in items):
            items.insert(0, {"doc_id": j.doc_id, "filename": "(processing)", "n_pages": 0, "status": j.status,
                             "stage": j.stage, "progress": j.progress, "error": j.error})
    return items


@app.get("/api/documents/{doc_id}/status")
def status(doc_id: str):
    _safe(doc_id)
    with _lock:
        if doc_id in _jobs:
            return _jobs[doc_id].model_copy()
    if _parsed_path(doc_id).exists():
        return JobStatus(doc_id=doc_id, status="parsed", progress=1.0)
    raise HTTPException(404, "unknown document")


@app.get("/api/documents/{doc_id}")
def get_document(doc_id: str):
    _safe(doc_id)
    with _lock:
        j = _jobs.get(doc_id)
    if j and j.status != "parsed":
        return JSONResponse({"doc_id": doc_id, "status": j.status, "stage": j.stage, "progress": j.progress, "error": j.error})
    return _load(doc_id)


@app.get("/api/documents/{doc_id}/pages/{n}")
def page_image(doc_id: str, n: int):
    p = settings.data_dir / "pages" / _safe(doc_id) / f"page_{n:04d}.png"
    if not p.exists():
        raise HTTPException(404)
    return FileResponse(p, media_type="image/png")


@app.get("/api/documents/{doc_id}/crops/{chunk_id}")
def crop_image(doc_id: str, chunk_id: str):
    p = settings.data_dir / "crops" / _safe(doc_id) / f"{_safe(chunk_id)}.png"
    if not p.exists():
        raise HTTPException(404)
    return FileResponse(p, media_type="image/png")


@app.post("/api/documents/{doc_id}/extract")
def extract(doc_id: str, body: dict[str, Any]):          # sync → runs in the threadpool, never blocks the event loop
    schema = body.get("schema") or body.get("schema_") or None   # omitted → auto-discover a schema first
    if schema is not None and (not isinstance(schema, dict) or not schema):
        raise HTTPException(400, "schema must be a non-empty JSON object")
    doc = _load(doc_id)
    doc = pipeline().extract(doc, schema, schema_name=body.get("name", "default"))
    return {**doc.extraction.model_dump(), "schema": schema or doc.metadata.get("suggested_schema")}


@app.get("/api/documents/{doc_id}/chunks")
def chunks(doc_id: str):
    return rag_chunks(_load(doc_id))


@app.get("/api/documents/{doc_id}/search")
def search(doc_id: str, q: str, k: int = 8):
    from rapidfuzz import fuzz, process
    doc = _load(doc_id)
    cands = {c.id: c for c in doc.all_chunks() if c.text.strip()}
    hits = process.extract(q, {cid: c.text for cid, c in cands.items()}, scorer=fuzz.partial_token_set_ratio, limit=k)
    return [{"chunk_id": cid, "score": s, "page": cands[cid].page, "type": cands[cid].type.value,
             "bbox": cands[cid].bbox.model_dump(), "text": cands[cid].text[:400]} for _, s, cid in hits]


@app.get("/api/documents/{doc_id}/export")
def export(doc_id: str):
    doc = _load(doc_id)
    out = settings.data_dir / "exports" / doc_id
    paths = export_all(doc, out)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, p in paths.items():
            z.write(p, name)
    buf.seek(0)
    return StreamingResponse(buf, media_type="application/zip",
                             headers={"Content-Disposition": f'attachment; filename="{doc_id}_export.zip"'})


@app.delete("/api/documents/{doc_id}")
def delete(doc_id: str):
    _safe(doc_id)
    for sub in ("uploads", "pages", "crops", "exports"):
        shutil.rmtree(settings.data_dir / sub / doc_id, ignore_errors=True)
    _parsed_path(doc_id).unlink(missing_ok=True)
    with _lock:
        _jobs.pop(doc_id, None)
    return {"deleted": doc_id}


@app.get("/api/health")
def health():
    p = pipeline()
    return {"ok": True, "ocr_backend": p.backend.name, "layout": p.layout.name,
            "llm": p.s.llm.model if p.llm else None, "vlm": p.s.vlm.model if p.vlm else None}


@app.get("/", response_class=HTMLResponse)
def index():
    return (UI_DIR / "index.html").read_text(encoding="utf-8")


if UI_DIR.exists():
    app.mount("/ui", StaticFiles(directory=str(UI_DIR)), name="ui")
