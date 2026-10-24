"""Turn *any* input document into rendered page images plus (if available) a native text layer.

Supported directly: PDF, PNG/JPG/TIFF/BMP/WEBP, plain text / markdown.
Office formats (docx/pptx/xlsx/odt) and HTML are converted to PDF with
LibreOffice when it is installed (``soffice --headless --convert-to pdf``),
which keeps the pipeline uniform: every page becomes an image that the layout
model and VLM can look at, and born-digital PDFs keep their text layer for
free (zero-hallucination text + exact bounding boxes).
"""
from __future__ import annotations

import mimetypes
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pymupdf  # PyMuPDF
from PIL import Image

from ..schemas import BBox, Chunk, ChunkType, Word

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp", ".gif"}
OFFICE_EXT = {".docx", ".doc", ".pptx", ".ppt", ".xlsx", ".xls", ".odt", ".odp", ".ods", ".rtf", ".html", ".htm"}
TEXT_EXT = {".txt", ".md", ".csv", ".json"}


@dataclass
class NativeBlock:
    """A text block from the PDF text layer with its lines/words (absolute pixel coords at render size)."""

    bbox: tuple[float, float, float, float]
    text: str
    font_size: float
    is_bold: bool
    words: list[Word] = field(default_factory=list)


@dataclass
class RenderedPage:
    number: int
    image: Image.Image
    image_path: Path
    width: int
    height: int
    native_blocks: list[NativeBlock] = field(default_factory=list)
    native_images: list[tuple[float, float, float, float]] = field(default_factory=list)  # embedded image rects (abs)
    native_drawings: list[tuple[float, float, float, float]] = field(default_factory=list)  # vector path rects (abs)
    has_text_layer: bool = False


@dataclass
class LoadedDocument:
    path: Path
    mime_type: str
    pages: list[RenderedPage]
    pdf_path: Optional[Path] = None


def _convert_office_to_pdf(src: Path, out_dir: Path) -> Path:
    soffice = shutil.which("soffice") or shutil.which("libreoffice")
    if not soffice:
        raise RuntimeError(
            f"Cannot convert {src.suffix} without LibreOffice. Install libreoffice or convert to PDF first."
        )
    subprocess.run(
        [soffice, "--headless", "--convert-to", "pdf", "--outdir", str(out_dir), str(src)],
        check=True, capture_output=True, timeout=300,
    )
    out = out_dir / (src.stem + ".pdf")
    if not out.exists():
        raise RuntimeError(f"LibreOffice did not produce {out}")
    return out


UNICODE_FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf",   # Latin + Devanagari (Docker image installs it)
    "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/lohit-devanagari/Lohit-Devanagari.ttf",
)


def _text_to_pdf(src: Path, out_dir: Path, lines_per_page: int = 60) -> Path:
    """Render a plain-text file to PDF, paginating by lines, with a Unicode font when one is installed."""
    doc = pymupdf.open()
    text = src.read_text(encoding="utf-8", errors="replace")
    lines: list[str] = []
    for raw in text.splitlines() or [""]:
        while len(raw) > 110:
            cut = raw.rfind(" ", 0, 110)
            cut = cut if cut > 40 else 110
            lines.append(raw[:cut])
            raw = raw[cut:].lstrip()
        lines.append(raw)
    fontfile = next((f for f in UNICODE_FONT_CANDIDATES if Path(f).exists()), None)
    for start in range(0, max(1, len(lines)), lines_per_page):
        page = doc.new_page()
        kw = {"fontname": "uni", "fontfile": fontfile} if fontfile else {"fontname": "helv"}
        if fontfile:
            page.insert_font(fontname="uni", fontfile=fontfile)
        y = 50
        for ln in lines[start:start + lines_per_page]:
            page.insert_text((40, y), ln, fontsize=10, **kw)
            y += 12.5
    out = out_dir / (src.stem + ".pdf")
    doc.save(out)
    return out


def _images_to_pdf(paths: list[Path], out_dir: Path) -> Path:
    doc = pymupdf.open()
    for p in paths:
        img = pymupdf.open(str(p))
        rect = img[0].rect
        pdfbytes = img.convert_to_pdf()
        img.close()
        imgpdf = pymupdf.open("pdf", pdfbytes)
        page = doc.new_page(width=rect.width, height=rect.height)
        page.show_pdf_page(rect, imgpdf, 0)
    out = out_dir / (paths[0].stem + ".pdf")
    doc.save(out)
    return out


def to_pdf(path: Path, work_dir: Path) -> Path:
    ext = path.suffix.lower()
    if ext == ".pdf":
        return path
    if ext in IMAGE_EXT:
        return _images_to_pdf([path], work_dir)
    if ext in OFFICE_EXT:
        return _convert_office_to_pdf(path, work_dir)
    if ext in TEXT_EXT:
        return _text_to_pdf(path, work_dir)
    raise ValueError(f"Unsupported file type: {ext}")


def _extract_native(page: pymupdf.Page, scale: float) -> tuple[list[NativeBlock], list[tuple[float, float, float, float]]]:
    """Text-layer blocks/words in *rendered* pixel coordinates.

    PyMuPDF returns text coordinates in the unrotated page space while ``get_pixmap`` honours the page's
    /Rotate entry, so every rect is mapped through ``page.rotation_matrix`` first (identity for unrotated pages)."""
    blocks: list[NativeBlock] = []
    images: list[tuple[float, float, float, float]] = []
    rot = page.rotation_matrix

    def rect(bb) -> tuple[float, float, float, float]:
        r = pymupdf.Rect(bb) * rot
        return r.x0 * scale, r.y0 * scale, r.x1 * scale, r.y1 * scale

    d = page.get_text("dict")
    for b in d.get("blocks", []):
        x0, y0, x1, y1 = rect(b["bbox"])
        if b.get("type") == 1:  # image
            images.append((x0, y0, x1, y1))
            continue
        lines_text, sizes, bold_flags = [], [], []
        for line in b.get("lines", []):
            spans = line.get("spans", [])
            line_txt = "".join(s["text"] for s in spans)
            if not line_txt.strip():
                continue
            lines_text.append(line_txt)
            for s in spans:
                sizes.append(s.get("size", 10))
                bold_flags.append("bold" in s.get("font", "").lower() or bool(s.get("flags", 0) & 16))
        text = "\n".join(lines_text).strip()
        if not text:
            continue
        blocks.append(NativeBlock(
            bbox=(x0, y0, x1, y1), text=text,
            font_size=(sum(sizes) / len(sizes)) if sizes else 10.0,
            is_bold=(sum(bold_flags) > len(bold_flags) / 2) if bold_flags else False,
        ))
    # word-level grounding
    W, H = page.rect.width * scale, page.rect.height * scale      # page.rect is already the rotated size
    block_boxes = [BBox.from_abs(*blk.bbox, W, H) for blk in blocks]
    for wx0, wy0, wx1, wy1, wtxt, _bno, _lno, _wno in page.get_text("words"):
        wb = BBox.from_abs(*rect((wx0, wy0, wx1, wy1)), W, H)
        # attach to the block with the highest overlap
        best, best_iou = None, 0.0
        for blk, bx in zip(blocks, block_boxes):
            inter = max(0, min(wb.x1, bx.x1) - max(wb.x0, bx.x0)) * max(0, min(wb.y1, bx.y1) - max(wb.y0, bx.y0))
            if inter > best_iou:
                best, best_iou = blk, inter
        if best is not None and best_iou > 0:
            best.words.append(Word(text=wtxt, bbox=wb))
    return blocks, images


def load_document(path: str | Path, out_dir: str | Path, dpi: int = 150, max_side: int = 4000) -> LoadedDocument:
    """Render every page to PNG at `dpi` (longest side capped at `max_side` px) and pull the native text layer."""
    path = Path(path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mime = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
    work = Path(tempfile.mkdtemp(prefix="docintel_", dir=out_dir))
    pdf_path = to_pdf(path, work)

    doc = pymupdf.open(str(pdf_path))
    pages: list[RenderedPage] = []
    for i, page in enumerate(doc):
        scale = dpi / 72.0
        # cap the render so a huge scan/photo cannot allocate gigabytes (longest side ≤ max_side px)
        longest = max(page.rect.width, page.rect.height) * scale
        if longest > max_side:
            scale *= max_side / longest
        pix = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), alpha=False)
        img_path = out_dir / f"page_{i + 1:04d}.png"
        pix.save(str(img_path))
        image = Image.open(img_path).convert("RGB")
        blocks, images = _extract_native(page, scale)
        drawings = []
        try:
            rot = page.rotation_matrix
            for d in page.get_drawings():
                r = d.get("rect")
                if r is not None and r.width * r.height > 4:
                    r = pymupdf.Rect(r) * rot
                    drawings.append((r.x0 * scale, r.y0 * scale, r.x1 * scale, r.y1 * scale))
        except Exception:
            pass
        text_chars = sum(len(b.text) for b in blocks)
        pages.append(RenderedPage(
            number=i + 1, image=image, image_path=img_path, width=pix.width, height=pix.height,
            native_blocks=blocks, native_images=images, native_drawings=drawings,
            # a scanned PDF usually has 0 chars; some have junk OCR layers — require a little real text
            has_text_layer=text_chars > 20,
        ))
    doc.close()
    if pdf_path != path and work.exists():     # converted copy no longer needed once rendered
        shutil.rmtree(work, ignore_errors=True)
        pdf_path = None
    return LoadedDocument(path=path, mime_type=mime, pages=pages, pdf_path=pdf_path)


def native_blocks_to_chunks(rp: RenderedPage, doc_id: str) -> list[Chunk]:
    """Convert text-layer blocks into chunks (types are refined later by the layout stage)."""
    chunks = []
    for order, b in enumerate(rp.native_blocks):
        chunks.append(Chunk(
            doc_id=doc_id, page=rp.number, type=ChunkType.TEXT, text=b.text,
            bbox=BBox.from_abs(*b.bbox, rp.width, rp.height), reading_order=order,
            confidence=1.0, source="native", words=b.words,
            attributes={"font_size": round(b.font_size, 1), "bold": b.is_bold},
        ))
    for rect in rp.native_images:
        chunks.append(Chunk(
            doc_id=doc_id, page=rp.number, type=ChunkType.FIGURE, text="",
            bbox=BBox.from_abs(*rect, rp.width, rp.height), reading_order=len(chunks),
            confidence=1.0, source="native",
        ))
    return chunks
