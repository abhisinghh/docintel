"""Generate sample documents for smoke tests: a born-digital bilingual invoice PDF (with a table and a
bar chart) and a scanned-looking PNG version of the same page (so the OCR path is exercised too)."""
from __future__ import annotations

import sys
from pathlib import Path

import pymupdf
from PIL import Image, ImageFilter

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "samples")
OUT.mkdir(parents=True, exist_ok=True)

doc = pymupdf.open()
page = doc.new_page(width=595, height=842)
hindi_font = None
for cand in ("/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf",
             "/usr/share/fonts/truetype/lohit-devanagari/Lohit-Devanagari.ttf",
             "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
    if Path(cand).exists():
        hindi_font = cand
        break

y = 50
page.insert_text((50, y), "TAX INVOICE / कर चालान", fontsize=20, fontname="helv")
y += 30
page.insert_text((50, y), "Bharat Textiles Pvt. Ltd.", fontsize=12, fontname="helv")
y += 16
page.insert_text((50, y), "Invoice Number: INV-2026-00417", fontsize=10, fontname="helv"); y += 14
page.insert_text((50, y), "Invoice Date: 2026-10-14", fontsize=10, fontname="helv"); y += 14
page.insert_text((50, y), "GSTIN: 08AAACB1234F1Z5", fontsize=10, fontname="helv"); y += 14
page.insert_text((50, y), "Customer: Rajasthan Handloom Cooperative, Kota", fontsize=10, fontname="helv"); y += 24

page.insert_text((50, y), "Line items", fontsize=13, fontname="helv"); y += 18
# table
cols = [50, 90, 300, 380, 460, 545]
headers = ["#", "Description", "Qty", "Unit price", "Amount"]
rows = [["1", "Cotton saree (handloom)", "40", "850.00", "34000.00"],
        ["2", "Silk dupatta", "25", "1200.00", "30000.00"],
        ["3", "Block-print bedsheet", "60", "640.00", "38400.00"]]
page.draw_rect(pymupdf.Rect(cols[0], y - 12, cols[-1], y + 14 * (len(rows) + 1)), width=0.6)
for i, h in enumerate(headers):
    page.insert_text((cols[i] + 3, y), h, fontsize=9, fontname="hebo")
for r in rows:
    y += 14
    for i, cell in enumerate(r):
        page.insert_text((cols[i] + 3, y), cell, fontsize=9, fontname="helv")
y += 22
page.insert_text((300, y), "Subtotal: 102400.00", fontsize=10, fontname="helv"); y += 14
page.insert_text((300, y), "GST (5%): 5120.00", fontsize=10, fontname="helv"); y += 14
page.insert_text((300, y), "Total Amount: INR 107520.00", fontsize=11, fontname="hebo"); y += 30

# bar chart
page.insert_text((50, y), "Figure 1: Monthly sales 2026 (INR lakh)", fontsize=10, fontname="helv"); y += 10
chart = pymupdf.Rect(50, y, 350, y + 150)
page.draw_rect(chart, color=(0.3, 0.3, 0.3), width=0.5)
vals = [("Jun", 4.2), ("Jul", 5.1), ("Aug", 6.3), ("Sep", 5.8), ("Oct", 7.4)]
bw = 40
for i, (m, v) in enumerate(vals):
    x0 = chart.x0 + 25 + i * 60
    page.draw_rect(pymupdf.Rect(x0, chart.y1 - 20 - v * 15, x0 + bw, chart.y1 - 20), color=(0.2, 0.4, 0.8), fill=(0.2, 0.4, 0.8))
    page.insert_text((x0 + 10, chart.y1 - 6), m, fontsize=8, fontname="helv")
    page.insert_text((x0 + 8, chart.y1 - 24 - v * 15), str(v), fontsize=8, fontname="helv")
y = chart.y1 + 30

if hindi_font:
    page.insert_font(fontname="hindi", fontfile=hindi_font)
    page.insert_text((50, y), "नियम एवं शर्तें: भुगतान 30 दिनों के भीतर देय है।", fontsize=10, fontname="hindi")
    y += 14
page.insert_text((50, y), "Terms: Payment due within 30 days. Goods once sold will not be taken back.", fontsize=10, fontname="helv")
y += 14
page.insert_text((50, y), "- Bank: State Bank of India, Kota Main Branch", fontsize=10, fontname="helv"); y += 12
page.insert_text((50, y), "- IFSC: SBIN0000648", fontsize=10, fontname="helv")
page.insert_text((280, 820), "Page 1 of 1", fontsize=8, fontname="helv")

pdf = OUT / "sample_invoice.pdf"
doc.save(pdf)
print("wrote", pdf)

# scanned version
pix = doc[0].get_pixmap(matrix=pymupdf.Matrix(2, 2))
img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
img = img.rotate(0.6, resample=Image.BICUBIC, fillcolor="white").filter(ImageFilter.GaussianBlur(0.6))
png = OUT / "sample_invoice_scan.png"
img.convert("L").save(png)
print("wrote", png)
