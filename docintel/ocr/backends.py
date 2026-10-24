"""OCR / page-parsing backends.

Every backend implements ``parse_page(image, page_number, doc_id) -> list[Chunk]``
and returns chunks with *normalised* bounding boxes.

Backends
--------
TesseractBackend   CPU, classical OCR with word boxes; multilingual via language packs
                   (eng+hin for English/Hindi). Baseline / fallback, weak on handwriting.
VLMBackend         Document VLM served through any OpenAI-compatible endpoint (vLLM,
                   SGLang, Ollama, hosted APIs). Prompt families for dots.ocr,
                   PaddleOCR-VL, DeepSeek-OCR, and a generic JSON prompt usable with
                   Qwen3-VL / GPT / Claude. This is the "agentic document extraction"
                   workhorse: one pass gives layout type + text + bbox per element,
                   reads handwriting and bilingual text, and emits tables as HTML.
DoclingBackend     Optional wrapper around IBM Docling (layout + TableFormer + OCR).
"""
from __future__ import annotations

import base64
import io
import json
import re
from typing import Optional, Protocol

import httpx
from PIL import Image

from ..config import VLMConfig
from ..schemas import BBox, Chunk, ChunkType, TableData, Word
from .langdetect import detect_languages, detect_scripts

# ------------------------------------------------------------------ protocol


class OCRBackend(Protocol):
    name: str

    def parse_page(self, image: Image.Image, page_number: int, doc_id: str) -> list[Chunk]: ...


# ------------------------------------------------------------------ tesseract


class TesseractBackend:
    name = "tesseract"

    def __init__(self, languages: list[str] | None = None, psm: int = 3):
        self.languages = languages or ["eng"]
        self.psm = psm

    def _available_langs(self) -> set[str]:
        try:
            import pytesseract
            return set(pytesseract.get_languages(config=""))
        except Exception:
            return {"eng"}

    def parse_page(self, image: Image.Image, page_number: int, doc_id: str) -> list[Chunk]:
        import pytesseract

        avail = self._available_langs()
        langs = [l for l in self.languages if l in avail] or ["eng"]
        data = pytesseract.image_to_data(
            image, lang="+".join(langs), config=f"--psm {self.psm}", output_type=pytesseract.Output.DICT
        )
        W, H = image.size
        # group words into blocks (tesseract block_num/par_num)
        groups: dict[tuple[int, int], list[dict]] = {}
        for i in range(len(data["text"])):
            txt = (data["text"][i] or "").strip()
            if not txt:
                continue
            key = (data["block_num"][i], data["par_num"][i])
            groups.setdefault(key, []).append({
                "text": txt, "conf": float(data["conf"][i]) if data["conf"][i] not in ("-1", -1) else 0.0,
                "x": data["left"][i], "y": data["top"][i], "w": data["width"][i], "h": data["height"][i],
                "line": data["line_num"][i],
            })
        chunks: list[Chunk] = []
        for order, (key, words) in enumerate(sorted(groups.items(), key=lambda kv: (min(w["y"] for w in kv[1]), min(w["x"] for w in kv[1])))):
            x0 = min(w["x"] for w in words); y0 = min(w["y"] for w in words)
            x1 = max(w["x"] + w["w"] for w in words); y1 = max(w["y"] + w["h"] for w in words)
            # rebuild line structure
            lines: dict[int, list[dict]] = {}
            for w in words:
                lines.setdefault(w["line"], []).append(w)
            text = "\n".join(" ".join(w["text"] for w in sorted(ln, key=lambda w: w["x"])) for _, ln in sorted(lines.items()))
            conf = sum(w["conf"] for w in words) / (100.0 * len(words))
            chunks.append(Chunk(
                doc_id=doc_id, page=page_number, type=ChunkType.TEXT, text=text,
                bbox=BBox.from_abs(x0, y0, x1, y1, W, H), reading_order=order, confidence=round(conf, 3),
                source="ocr", languages=detect_languages(text), scripts=detect_scripts(text),
                words=[Word(text=w["text"], confidence=w["conf"] / 100.0,
                            bbox=BBox.from_abs(w["x"], w["y"], w["x"] + w["w"], w["y"] + w["h"], W, H)) for w in words],
            ))
        return chunks


# ------------------------------------------------------------------ VLM

GENERIC_PROMPT = """You are a document parsing model. Read this page image and return ONLY a JSON array.
Each element is a layout block in reading order with:
  "type": one of ["title","section_header","text","list_item","caption","footnote","page_header","page_footer","page_number","formula","code","handwriting","form_field","table","figure","chart","map","infographic","signature","stamp"]
  "bbox": [x0,y0,x1,y1] as integers in a 0-1000 coordinate system relative to the page (origin top-left)
  "text": the exact transcribed text (keep original language/script; for bilingual text keep both; for tables return an HTML <table>; for charts give a short description plus the data values you can read; for figures give a caption/description; for form_field use "key: value")
  "confidence": 0-1
Transcribe handwriting faithfully. Do not translate. Do not omit any text on the page."""

DOTS_OCR_PROMPT = (
    "Please output the layout information from the PDF image, including each layout element's bbox, "
    "its category, and the corresponding text content within the bbox.\n\n"
    "1. Bbox format: [x1, y1, x2, y2]\n\n"
    "2. Layout Categories: The possible categories are ['Caption', 'Footnote', 'Formula', 'List-item', "
    "'Page-footer', 'Page-header', 'Picture', 'Section-header', 'Table', 'Text', 'Title'].\n\n"
    "3. Text Extraction & Formatting Rules:\n    - Picture: For the 'Picture' category, the text field should be omitted.\n"
    "    - Formula: Format its text as LaTeX.\n    - Table: Format its text as HTML.\n"
    "    - All Others (Text, Title, etc.): Format their text as Markdown.\n\n"
    "4. Constraints:\n    - The output text must be the original text from the image, with no translation.\n"
    "    - All layout elements must be sorted according to human reading order.\n\n"
    "5. Final Output: The entire output must be a single JSON object."
)

DOTS_TYPE_MAP = {
    "Caption": ChunkType.CAPTION, "Footnote": ChunkType.FOOTNOTE, "Formula": ChunkType.FORMULA,
    "List-item": ChunkType.LIST_ITEM, "Page-footer": ChunkType.PAGE_FOOTER, "Page-header": ChunkType.PAGE_HEADER,
    "Picture": ChunkType.FIGURE, "Section-header": ChunkType.SECTION_HEADER, "Table": ChunkType.TABLE,
    "Text": ChunkType.TEXT, "Title": ChunkType.TITLE,
}


def _img_b64(image: Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    image.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode()


def _extract_json(s: str):
    s = s.strip()
    s = re.sub(r"^```(?:json)?", "", s).strip()
    s = re.sub(r"```$", "", s).strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        m = re.search(r"(\[.*\]|\{.*\})", s, re.S)
        if m:
            return json.loads(m.group(1))
        raise


def _parse_html_table(html: str) -> Optional[TableData]:
    rows = re.findall(r"<tr[^>]*>(.*?)</tr>", html, re.S | re.I)
    if not rows:
        return None
    cells, n_cols = [], 0
    occupied: set[tuple[int, int]] = set()           # (row, col) slots taken by row/col spans above or to the left
    for r, row in enumerate(rows):
        c = 0
        for m in re.finditer(r"<(t[dh])([^>]*)>(.*?)</t[dh]>", row, re.S | re.I):
            while (r, c) in occupied:
                c += 1
            tag, attrs, inner = m.group(1).lower(), m.group(2), m.group(3)
            text = re.sub(r"<[^>]+>", " ", inner)
            text = re.sub(r"\s+", " ", text).strip()
            cs = max(1, int((re.search(r'colspan="?(\d+)', attrs) or [None, "1"])[1]))
            rs = max(1, int((re.search(r'rowspan="?(\d+)', attrs) or [None, "1"])[1]))
            cells.append(dict(row=r, col=c, col_span=cs, row_span=rs, text=text, is_header=(tag == "th" or r == 0)))
            for dr in range(rs):
                for dc in range(cs):
                    occupied.add((r + dr, c + dc))
            c += cs
        while (r, c) in occupied:
            c += 1
        n_cols = max(n_cols, c)
    from ..schemas import TableCell
    return TableData(cells=[TableCell(**c) for c in cells], n_rows=len(rows), n_cols=n_cols, html=html)


class VLMBackend:
    """Document VLM behind one of three transports (see VLMConfig.provider):
    OpenAI-compatible HTTP (on-prem vLLM/SGLang/Ollama or hosted APIs), Anthropic Messages API,
    or in-process HF transformers. The prompt family is independent of the transport."""

    name = "vlm"

    def __init__(self, cfg: VLMConfig):
        self.cfg = cfg
        self.name = f"vlm:{cfg.provider}:{cfg.family}:{cfg.model}"
        self._transport = make_transport(cfg.provider, cfg.model, cfg.base_url, cfg.api_key, cfg.timeout, cfg.device)

    # -- raw call ------------------------------------------------------------------
    def chat(self, image: Image.Image, prompt: str, max_tokens: Optional[int] = None) -> str:
        return self._transport(image, prompt, max_tokens or self.cfg.max_tokens)

    def prompt_for_family(self) -> str:
        if self.cfg.family == "dots_ocr":
            return DOTS_OCR_PROMPT
        if self.cfg.family == "deepseek_ocr":
            return "<|grounding|>Convert the document to markdown. "
        if self.cfg.family == "paddleocr_vl":
            # PaddleOCR-VL is normally driven by its own pipeline (layout → per-element recognition);
            # through vLLM we ask for the generic JSON layout output.
            return GENERIC_PROMPT
        return GENERIC_PROMPT

    # -- page parse ------------------------------------------------------------------
    def parse_page(self, image: Image.Image, page_number: int, doc_id: str) -> list[Chunk]:
        raw = self.chat(image, self.prompt_for_family())
        W, H = image.size
        if self.cfg.family == "deepseek_ocr":
            return self._parse_deepseek(raw, W, H, page_number, doc_id)
        data = _extract_json(raw)
        if isinstance(data, dict):
            data = data.get("elements") or data.get("blocks") or data.get("layout") or next(
                (v for v in data.values() if isinstance(v, list)), [])
        if not isinstance(data, list):
            raise ValueError("VLM did not return a list of layout elements")
        chunks: list[Chunk] = []
        for order, el in enumerate(data):
            if not isinstance(el, dict):
                continue
            try:
                bbox = _parse_bbox(el.get("bbox") or el.get("box"), W, H, pixel=(self.cfg.family == "dots_ocr"))
                raw_type = str(el.get("type") or el.get("category") or "text")
                ctype = DOTS_TYPE_MAP.get(raw_type) or _coerce_type(raw_type)
                text = str(el.get("text") or "")
                table = _parse_html_table(text) if ctype == ChunkType.TABLE and "<table" in text.lower() else None
                md_text = table.to_markdown() if table else text
                try:
                    conf = max(0.0, min(1.0, float(el.get("confidence") if el.get("confidence") is not None else 0.9)))
                except (TypeError, ValueError):
                    conf = 0.9
                chunks.append(Chunk(
                    doc_id=doc_id, page=page_number, type=ctype, text=md_text, bbox=bbox, reading_order=order,
                    confidence=conf, source="vlm", table=table,
                    languages=detect_languages(md_text), scripts=detect_scripts(md_text),
                ))
            except Exception:          # one malformed element must not lose the page
                continue
        return chunks

    def _parse_deepseek(self, raw: str, W: int, H: int, page_number: int, doc_id: str) -> list[Chunk]:
        # DeepSeek-OCR grounding format: <|ref|>type<|/ref|><|det|>[[x0,y0,x1,y1]]<|/det|> text ...  (0-999 coords)
        pat = re.compile(r"<\|ref\|>(.*?)<\|/ref\|><\|det\|>\[\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]\]<\|/det\|>", re.S)
        chunks, order = [], 0
        matches = list(pat.finditer(raw))
        for i, m in enumerate(matches):
            end = matches[i + 1].start() if i + 1 < len(matches) else len(raw)
            text = raw[m.end():end].strip()
            x0, y0, x1, y1 = (int(m.group(k)) / 999 for k in range(2, 6))
            chunks.append(Chunk(doc_id=doc_id, page=page_number, type=_coerce_type(m.group(1)), text=text,
                                bbox=BBox(x0=x0, y0=y0, x1=x1, y1=y1), reading_order=order, source="vlm",
                                languages=detect_languages(text), scripts=detect_scripts(text)))
            order += 1
        if not chunks:
            chunks.append(Chunk(doc_id=doc_id, page=page_number, type=ChunkType.TEXT, text=raw,
                                bbox=BBox(x0=0, y0=0, x1=1, y1=1), source="vlm"))
        return chunks

    # -- region-level helpers used by the agent -----------------------------------------
    def read_region(self, image: Image.Image, bbox: BBox, instruction: str = "Transcribe all text exactly. Keep the original language.") -> str:
        x0, y0, x1, y1 = bbox.to_abs(*image.size)
        pad = 8
        crop = image.crop((max(0, x0 - pad), max(0, y0 - pad), min(image.width, x1 + pad), min(image.height, y1 + pad)))
        if crop.width < 2 or crop.height < 2:
            return ""
        # upscale small crops so the VLM can read fine print / handwriting
        if crop.width < 800:
            s = 800 / crop.width
            crop = crop.resize((int(crop.width * s), int(crop.height * s)))
        return self.chat(crop, instruction, max_tokens=4096)


# ------------------------------------------------------------------ transports


def make_transport(provider: str, model: str, base_url: str, api_key: str, timeout: float, device: str = "auto"):
    """Return f(image, prompt, max_tokens) -> str for the chosen provider."""
    if provider == "anthropic":
        client = httpx.Client(base_url="https://api.anthropic.com", timeout=timeout,
                              headers={"x-api-key": api_key, "anthropic-version": "2023-06-01"})

        def call(image, prompt, max_tokens):
            payload = {"model": model, "max_tokens": max_tokens, "temperature": 0.0, "messages": [{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": _img_b64(image)}},
                {"type": "text", "text": prompt}]}]}
            r = client.post("/v1/messages", json=payload)
            r.raise_for_status()
            return "".join(b.get("text", "") for b in r.json()["content"])
        return call
    if provider == "transformers":
        return TransformersVLM(model, device).chat
    # default: OpenAI-compatible chat/completions
    client = httpx.Client(base_url=base_url, timeout=timeout, headers={"Authorization": f"Bearer {api_key}"})

    def call(image, prompt, max_tokens):
        payload = {"model": model, "temperature": 0.0, "max_tokens": max_tokens, "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_img_b64(image)}"}},
            {"type": "text", "text": prompt}]}]}
        r = client.post("/chat/completions", json=payload)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
    return call


class TransformersVLM:
    """In-process HF transformers inference (no server). Works for Qwen2.5/3-VL, dots.ocr, PaddleOCR-VL,
    GLM-OCR and other AutoModelForImageTextToText checkpoints. pip install -e '.[vlm]'."""

    def __init__(self, model: str, device: str = "auto"):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        self.processor = AutoProcessor.from_pretrained(model, trust_remote_code=True)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model, torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map=device, trust_remote_code=True)
        self.model.eval()

    def chat(self, image: Image.Image, prompt: str, max_tokens: int = 4096) -> str:
        import torch
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        text = self.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[image], return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
        return self.processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]


class LandingAIBackend:
    """LandingAI Agentic Document Extraction as a hosted parser (API key in DOCINTEL_VLM_API_KEY).

    Posts the page image to the ADE parse endpoint and maps returned chunks/blocks (type, markdown,
    grounding{page, box{l,t,r,b}} normalised 0-1) onto docintel chunks. Parses both the v1 `chunks[]`
    and the Gen2 `pages[].blocks[]` shapes defensively; check docs.landing.ai if the field names change.
    """

    name = "landingai"

    def __init__(self, cfg: VLMConfig):
        self.cfg = cfg
        base = cfg.base_url if "landing.ai" in (cfg.base_url or "") else "https://api.va.landing.ai"
        self._client = httpx.Client(base_url=base, timeout=cfg.timeout, headers={"Authorization": f"Basic {cfg.api_key}"})
        self.read_region = None   # no region re-read API; agent falls back to the LLM

    def parse_page(self, image: Image.Image, page_number: int, doc_id: str) -> list[Chunk]:
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        r = self._client.post("/v1/tools/agentic-document-analysis", files={"image": ("page.png", buf.getvalue(), "image/png")})
        r.raise_for_status()
        data = r.json().get("data", r.json())
        items = data.get("chunks") or [b for pg in data.get("pages", []) for b in pg.get("blocks", [])]
        type_map = {"text": ChunkType.TEXT, "table": ChunkType.TABLE, "figure": ChunkType.FIGURE, "marginalia": ChunkType.PAGE_HEADER,
                    "logo": ChunkType.FIGURE, "card": ChunkType.FORM_FIELD, "attestation": ChunkType.SIGNATURE, "scan_code": ChunkType.BARCODE,
                    "title": ChunkType.TITLE, "section_header": ChunkType.SECTION_HEADER, "form_field": ChunkType.FORM_FIELD}
        chunks = []
        for order, it in enumerate(items):
            g = (it.get("grounding") or [{}])
            g = g[0] if isinstance(g, list) else g
            box = g.get("box") or it.get("box") or {}
            bbox = BBox(x0=float(box.get("left", box.get("l", 0))), y0=float(box.get("top", box.get("t", 0))),
                        x1=float(box.get("right", box.get("r", 1))), y1=float(box.get("bottom", box.get("b", 1))))
            text = it.get("markdown") or it.get("text") or ""
            ctype = type_map.get(str(it.get("chunk_type") or it.get("type") or "text").lower(), ChunkType.TEXT)
            table = _parse_html_table(text) if ctype == ChunkType.TABLE and "<table" in text.lower() else None
            chunks.append(Chunk(doc_id=doc_id, page=page_number, type=ctype, text=table.to_markdown() if table else text, bbox=bbox,
                                reading_order=order, confidence=float(it.get("confidence", 0.9)), source="vlm", table=table,
                                languages=detect_languages(text), scripts=detect_scripts(text)))
        return chunks


def _parse_bbox(bb, W: int, H: int, pixel: bool = False) -> BBox:
    """Accept [x0,y0,x1,y1] in pixels, 0-1000 or 0-1 units; order corners and clamp. Missing → whole page."""
    if not bb or len(bb) < 4:
        return BBox(x0=0, y0=0, x1=1, y1=1)
    x0, y0, x1, y1 = (float(v) for v in bb[:4])
    x0, x1 = min(x0, x1), max(x0, x1)
    y0, y1 = min(y0, y1), max(y0, y1)
    mx = max(x1, y1)
    if pixel or mx > 1000:
        return BBox.from_abs(x0, y0, x1, y1, W, H)
    if mx <= 1.0:
        return BBox(x0=x0, y0=y0, x1=x1, y1=y1)
    return BBox(x0=x0 / 1000, y0=y0 / 1000, x1=x1 / 1000, y1=y1 / 1000)


def _coerce_type(raw: str) -> ChunkType:
    r = raw.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "header": ChunkType.SECTION_HEADER, "heading": ChunkType.SECTION_HEADER, "paragraph": ChunkType.TEXT,
        "picture": ChunkType.FIGURE, "image": ChunkType.FIGURE, "list": ChunkType.LIST_ITEM,
        "equation": ChunkType.FORMULA, "footer": ChunkType.PAGE_FOOTER, "key_value": ChunkType.FORM_FIELD,
        "kv": ChunkType.FORM_FIELD, "graph": ChunkType.CHART, "plot": ChunkType.CHART,
    }
    if r in aliases:
        return aliases[r]
    try:
        return ChunkType(r)
    except ValueError:
        return ChunkType.TEXT


# ------------------------------------------------------------------ docling (optional)


class DoclingBackend:
    """Wrap IBM Docling's converter so its layout + TableFormer output fits our schema."""

    name = "docling"

    def __init__(self):
        from docling.document_converter import DocumentConverter  # noqa: F401
        self._conv = DocumentConverter()

    def parse_page(self, image: Image.Image, page_number: int, doc_id: str) -> list[Chunk]:  # pragma: no cover
        import tempfile
        from docling_core.types.doc import DocItemLabel  # type: ignore

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            image.save(f.name)
            res = self._conv.convert(f.name)
        W, H = image.size
        out = []
        for order, (item, _lvl) in enumerate(res.document.iterate_items()):
            prov = item.prov[0] if getattr(item, "prov", None) else None
            if prov is None:
                continue
            b = prov.bbox.to_top_left_origin(page_height=H)
            label = str(getattr(item, "label", DocItemLabel.TEXT)).lower()
            text = getattr(item, "text", "") or ""
            out.append(Chunk(doc_id=doc_id, page=page_number, type=_coerce_type(label), text=text, source="vlm",
                             bbox=BBox.from_abs(b.l, b.t, b.r, b.b, W, H), reading_order=order))
        return out


def build_backend(settings) -> OCRBackend:
    if settings.vlm.enabled:
        if settings.vlm.provider == "landingai":
            return LandingAIBackend(settings.vlm)
        return VLMBackend(settings.vlm)
    if settings.ocr.engine == "docling":
        return DoclingBackend()
    return TesseractBackend(languages=settings.ocr.languages)
