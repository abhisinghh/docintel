"""Agentic orchestration on top of the deterministic pipeline.

"Agentic document extraction" ≠ "one VLM call per page". The agent inspects
the first-pass result and decides, page by page, which *tools* to apply:

  zoom_read(chunk_id)        re-OCR a low-confidence / handwritten / tiny-text region at 2-3× zoom
  extract_table(chunk_id)    run table structure recognition on a region
  describe_visual(chunk_id)  chart-to-table / map description for a figure region
  reread_page(page)          full page re-parse with the VLM (e.g. text layer was garbage)
  extract_fields(schema)     grounded schema extraction
  verify_field(field)        independent visual re-check of a value
  finish(summary)

Policy: if an LLM is configured, a ReAct-style loop (JSON actions, bounded
by `agent_max_steps`) chooses tools from an *observation* summary of the
page (chunk types, confidences, scripts, flags). Without an LLM the same
tools are driven by a deterministic rule policy, so the project runs fully
offline — the quality just scales with the models you plug in.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from PIL import Image

from ..charts.extractor import extract_visuals
from ..extraction.schema import extract_fields
from ..ocr.langdetect import detect_languages, detect_scripts
from ..schemas import ChunkType, ParsedDocument
from ..tables.extractor import extract_tables

LOW_CONF = 0.75


@dataclass
class AgentTrace:
    steps: list[dict[str, Any]] = field(default_factory=list)

    def log(self, action: str, **kw: Any) -> None:
        self.steps.append({"action": action, **kw})


class DocumentAgent:
    def __init__(self, llm=None, vlm=None, crops_dir: Path = Path("./data/crops"), max_steps: int = 12, ocr_fallback=None):
        self.llm = llm if (llm is not None and getattr(llm, "available", True)) else None
        self.vlm = vlm
        self.ocr_fallback = ocr_fallback   # classical OCR used for garbled regions when no VLM is available
        self.crops_dir = crops_dir
        self.max_steps = max_steps
        self.trace = AgentTrace()

    # ------------------------------------------------------------------ tools
    def tool_zoom_read(self, doc: ParsedDocument, images: dict[int, Image.Image], chunk_id: str) -> str:
        c = doc.chunk(chunk_id)
        if c is None:
            return "no such chunk"
        if self.vlm is None:
            if self.ocr_fallback is None:
                return "unavailable"
            img = images[c.page]
            x0, y0, x1, y1 = c.bbox.to_abs(*img.size)
            crop = img.crop((int(x0) - 4, int(y0) - 4, int(x1) + 4, int(y1) + 4))
            crop = crop.resize((crop.width * 2, crop.height * 2))
            sub = self.ocr_fallback.parse_page(crop, c.page, doc.doc_id)
            text = "\n".join(s.text for s in sub).strip()
            if text:
                c.attributes["first_pass_text"] = c.text
                c.text, c.source = text, "ocr"
                c.confidence = min(0.7, sum(s.confidence for s in sub) / max(1, len(sub)))
                c.languages, c.scripts = detect_languages(text), detect_scripts(text)
            return text[:200] or "empty"
        instr = ("Transcribe all text in this region exactly as written, including handwriting. "
                 "Keep the original language and script; if two languages are present transcribe both.")
        text = self.vlm.read_region(images[c.page], c.bbox, instr).strip()
        if text:
            c.attributes["first_pass_text"] = c.text
            c.text, c.source, c.confidence = text, "agent", max(c.confidence, 0.85)
            c.languages, c.scripts = detect_languages(text), detect_scripts(text)
        return text[:200]

    def tool_extract_table(self, doc: ParsedDocument, images: dict[int, Image.Image], chunk_id: str) -> str:
        c = doc.chunk(chunk_id)
        if c is None:
            return "no such chunk"
        extract_tables(images[c.page], [c], vlm=self.vlm)      # geometry first, then one guarded VLM attempt
        return f"{c.table.n_rows}x{c.table.n_cols}" if c.table else "failed"

    def tool_describe_visual(self, doc: ParsedDocument, images: dict[int, Image.Image], chunk_id: str) -> str:
        c = doc.chunk(chunk_id)
        if c is None:
            return "no such chunk"
        extract_visuals(images[c.page], [c], self.crops_dir, vlm=self.vlm)
        return c.text[:200]

    def tool_reread_page(self, doc: ParsedDocument, images: dict[int, Image.Image], page: int) -> str:
        if self.vlm is None:
            return "unavailable"
        p = doc.page(page)
        if p is None:
            return "no such page"
        new = self.vlm.parse_page(images[page], page, doc.doc_id)
        for i, c in enumerate(new):
            c.reading_order = i
        p.chunks = new
        return f"{len(new)} chunks"

    # ------------------------------------------------------------------ observation
    @staticmethod
    def observe(doc: ParsedDocument) -> dict[str, Any]:
        obs = {"pages": []}
        for p in doc.pages:
            flags = []
            for c in p.chunks:
                f = []
                if c.confidence < LOW_CONF and c.type not in (ChunkType.FIGURE,):
                    f.append("low_conf")
                if c.type == ChunkType.HANDWRITING:
                    f.append("handwriting")
                if len(c.scripts) > 1:
                    f.append("bilingual")
                if c.type == ChunkType.TABLE and c.table is None:
                    f.append("table_unparsed")
                if c.type in (ChunkType.FIGURE, ChunkType.CHART, ChunkType.MAP, ChunkType.INFOGRAPHIC) and not c.text.strip("[]").strip():
                    f.append("visual_undescribed")
                if c.text and len(c.text) < 3 and c.bbox.area() > 0.02:
                    f.append("suspiciously_empty")
                if c.text and sum(1 for ch in c.text if ord(ch) < 32 and ch not in "\n\t" or ch == "\ufffd") > 0.1 * len(c.text):
                    f.append("garbled")   # text layer without glyph mapping (common with Indic fonts)
                if f:
                    flags.append({"chunk_id": c.id, "type": c.type.value, "flags": f, "conf": c.confidence})
            obs["pages"].append({"page": p.number, "n_chunks": len(p.chunks), "has_text_layer": p.has_text_layer,
                                 "flagged": flags[:40]})
        return obs

    # ------------------------------------------------------------------ policies
    def _rule_policy(self, doc: ParsedDocument, images: dict[int, Image.Image]) -> None:
        obs = self.observe(doc)
        for p in obs["pages"]:
            for fl in p["flagged"]:
                flags = fl["flags"]
                if "table_unparsed" in flags:
                    self._safe_call("extract_table", self.tool_extract_table, doc, images, fl["chunk_id"])
                if "visual_undescribed" in flags:
                    self._safe_call("describe_visual", self.tool_describe_visual, doc, images, fl["chunk_id"])
                if ("low_conf" in flags or "handwriting" in flags or "suspiciously_empty" in flags or "garbled" in flags) \
                        and (self.vlm or ("garbled" in flags and self.ocr_fallback)):
                    self._safe_call("zoom_read", self.tool_zoom_read, doc, images, fl["chunk_id"])

    def _safe_call(self, name: str, fn, doc, images, chunk_id: str) -> None:
        try:
            self.trace.log(name, chunk_id=chunk_id, result=str(fn(doc, images, chunk_id))[:200])
        except Exception as e:                       # a failing tool must never abort the parse
            self.trace.log(name, chunk_id=chunk_id, error=str(e)[:200])

    def _llm_policy(self, doc: ParsedDocument, images: dict[int, Image.Image]) -> None:
        tools: dict[str, Callable[..., str]] = {
            "zoom_read": lambda a: self.tool_zoom_read(doc, images, a["chunk_id"]),
            "extract_table": lambda a: self.tool_extract_table(doc, images, a["chunk_id"]),
            "describe_visual": lambda a: self.tool_describe_visual(doc, images, a["chunk_id"]),
            "reread_page": lambda a: self.tool_reread_page(doc, images, int(a["page"])),
        }
        system = (
            "You are the orchestrator of a document-parsing pipeline. You see an observation of the current parse "
            "(flagged chunks) and the tool results so far. Choose the next tool to improve fidelity, or finish. "
            "Respond ONLY with JSON: {\"tool\": name, \"args\": {...}, \"reason\": str} or {\"tool\": \"finish\"}. "
            f"Tools: {list(tools)} (+ finish). Only call tools on flagged chunks. Never call the same tool twice on one chunk."
        )
        history: list[dict[str, Any]] = []
        for _ in range(self.max_steps):
            obs = self.observe(doc)
            prompt = f"Observation:\n{json.dumps(obs, ensure_ascii=False)[:6000]}\n\nHistory:\n{json.dumps(history)[-3000:]}"
            try:
                decision = self.llm.complete_json(prompt, system=system)
            except Exception as e:
                self.trace.log("llm_error", error=str(e))
                break
            if not isinstance(decision, dict):
                self.trace.log("finish", reason="non-dict decision")
                break
            tool = decision.get("tool", "finish")
            if tool == "finish" or tool not in tools:
                self.trace.log("finish", reason=decision.get("reason"))
                break
            args = decision.get("args") if isinstance(decision.get("args"), dict) else {}
            args = {k: v for k, v in args.items() if k not in ("action", "result", "reason", "error")}
            key = (tool, json.dumps(args, sort_keys=True, default=str))
            if any((h["tool"], json.dumps(h["args"], sort_keys=True)) == key for h in history):
                break
            try:
                res = tools[tool](args)
            except Exception as e:
                res = f"error: {e}"
            history.append({"tool": tool, "args": args, "result": str(res)[:200]})
            self.trace.log(tool, **args, result=str(res)[:200], reason=decision.get("reason"))

    # ------------------------------------------------------------------ run
    def run(self, doc: ParsedDocument, images: dict[int, Image.Image],
            schema: Optional[dict[str, Any]] = None, verify: bool = True,
            snippets_dir: Optional[Path] = None, strict: bool = False) -> ParsedDocument:
        if self.llm is not None:
            self._llm_policy(doc, images)
        else:
            self._rule_policy(doc, images)
        if schema:
            doc.extraction = extract_fields(doc, schema, llm=self.llm, vlm=self.vlm, page_images=images, verify=verify,
                                            snippets_dir=snippets_dir or self.crops_dir, strict=strict)
            self.trace.log("extract_fields", n_fields=len(doc.extraction.fields),
                           n_verified=sum(1 for f in doc.extraction.fields if f.verified))
        doc.pipeline["agent_trace"] = self.trace.steps
        return doc
