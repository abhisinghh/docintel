"""Render synthetic Devanagari / bilingual text lines for stage-A pretraining.

    python synth_lines.py --corpus hindi_sentences.txt --fonts fonts/ --n 100000 --out synth/

Uses every .ttf in --fonts (mix Noto Sans/Serif Devanagari, Lohit, Mukta, Hind, Tiro, Kalam/handwriting-style
fonts such as 'Kalam', 'Baloo', 'Yatra One'). Adds blur, rotation, noise, background texture, and random
English tokens so the model sees code-mixed lines. Writes line PNGs + a JSONL manifest.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

ENGLISH = ["Name", "Date", "Signature", "Amount", "Rs.", "No.", "Address", "Mobile", "PAN", "Aadhaar", "Total", "2026"]


def render(text: str, font_path: str, size: int = 40) -> Image.Image:
    font = ImageFont.truetype(font_path, size)
    dummy = Image.new("L", (10, 10))
    x0, y0, x1, y1 = ImageDraw.Draw(dummy).textbbox((0, 0), text, font=font)
    pad = 16
    img = Image.new("L", (x1 - x0 + 2 * pad, y1 - y0 + 2 * pad), random.randint(225, 255))
    d = ImageDraw.Draw(img)
    d.text((pad - x0, pad - y0), text, font=font, fill=random.randint(0, 60))
    if random.random() < 0.5:
        img = img.rotate(random.uniform(-2.5, 2.5), resample=Image.BICUBIC, fillcolor=255, expand=True)
    if random.random() < 0.5:
        img = img.filter(ImageFilter.GaussianBlur(random.uniform(0.3, 1.2)))
    if random.random() < 0.4:   # salt noise
        px = img.load()
        for _ in range(int(img.width * img.height * 0.002)):
            px[random.randrange(img.width), random.randrange(img.height)] = random.randint(0, 255)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True); ap.add_argument("--fonts", required=True)
    ap.add_argument("--n", type=int, default=10000); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    fonts = [str(p) for p in Path(a.fonts).rglob("*.ttf")]
    lines = [l.strip() for l in Path(a.corpus).read_text(encoding="utf-8").splitlines() if l.strip()]
    with open(out / "manifest.jsonl", "w", encoding="utf-8") as mf:
        for i in range(a.n):
            text = random.choice(lines)
            words = text.split()
            if random.random() < 0.3 and words:   # code-mixing
                words.insert(random.randrange(len(words) + 1), random.choice(ENGLISH))
            text = " ".join(words[: random.randint(2, 10)])
            img = render(text, random.choice(fonts), size=random.randint(28, 48))
            p = out / f"{i:07d}.png"
            img.save(p)
            mf.write(json.dumps({"image": str(p), "text": text}, ensure_ascii=False) + "\n")
    print("wrote", a.n, "lines to", out)


if __name__ == "__main__":
    main()
