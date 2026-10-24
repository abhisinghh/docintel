"""docintel command line.

  docintel parse  <file> [--schema schema.json] [--out exports/]   parse + export RAG/KG bundle
  docintel extract <doc_id> --schema schema.json                    run schema extraction on a parsed doc
  docintel serve  [--port 8080]                                     API + UI
  docintel eval   <benchmark> ...                                   see docintel.eval.runners
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import get_settings


def main(argv=None):
    ap = argparse.ArgumentParser(prog="docintel")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("parse"); p.add_argument("file"); p.add_argument("--schema", help="JSON schema file → mode=schema")
    p.add_argument("--mode", choices=["parse", "schema", "auto"], help="parse only (default), extract with --schema, or auto-discover fields")
    p.add_argument("--out"); p.add_argument("--no-kg", action="store_true")
    e = sub.add_parser("extract"); e.add_argument("doc_id"); e.add_argument("--schema", required=True)
    s = sub.add_parser("serve"); s.add_argument("--host", default="0.0.0.0"); s.add_argument("--port", type=int, default=8080)
    v = sub.add_parser("eval"); v.add_argument("args", nargs=argparse.REMAINDER)
    a = ap.parse_args(argv)

    if a.cmd == "parse":
        from .export.rag import export_all
        from .pipeline import Pipeline
        schema = json.loads(Path(a.schema).read_text()) if a.schema else None
        pipe = Pipeline(get_settings())
        doc = pipe.parse(a.file, schema=schema, build_kg=not a.no_kg, mode=a.mode,
                         progress=lambda st, pr: print(f"  [{pr:4.0%}] {st}", file=sys.stderr))
        out = a.out or str(get_settings().data_dir / "exports" / doc.doc_id)
        paths = export_all(doc, out)
        print(json.dumps({"doc_id": doc.doc_id, "pages": doc.n_pages, "chunks": len(doc.all_chunks()),
                          "languages": doc.languages, "triples": len(doc.triples), "exports": paths,
                          "extraction_mode": doc.metadata.get("extraction_mode"),
                          "suggested_schema": doc.metadata.get("suggested_schema"),
                          "fields": {f.name: f.value for f in doc.extraction.fields} if doc.extraction else None,
                          "pipeline": {k: v for k, v in doc.pipeline.items() if k != "agent_trace"}}, indent=1, ensure_ascii=False))
    elif a.cmd == "extract":
        from .pipeline import Pipeline
        pipe = Pipeline(get_settings())
        doc = pipe.extract(pipe.load(a.doc_id), json.loads(Path(a.schema).read_text()))
        print(json.dumps(doc.extraction.model_dump(), indent=1, ensure_ascii=False))
    elif a.cmd == "serve":
        import uvicorn
        uvicorn.run("docintel.api.server:app", host=a.host, port=a.port, reload=False)
    elif a.cmd == "eval":
        from .eval.runners import main as eval_main
        eval_main(a.args)


if __name__ == "__main__":
    main()
