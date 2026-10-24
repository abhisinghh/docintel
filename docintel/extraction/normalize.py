"""Type / format enforcement for schema-driven extraction.

The LLM (or the offline heuristics) returns strings as they appear in the document; this module coerces them to
the schema's declared type so every document yields the *same* shape:

  number / integer  "INR 1,07,520.00" → 107520.0        (currency symbols, Indian & western grouping removed)
  boolean           "Yes" / "✓" / "हाँ" → True
  string+format     "14/10/2026", "Oct 14, 2026", "14 अक्टूबर 2026" → "2026-10-14"   (format: date)
  enum              nearest allowed value by fuzzy match (≥ 80) else None
  array of objects  list of coerced objects

A value that cannot be coerced is kept as the raw string, `type_ok=False` and a note is attached, so nothing is
silently dropped and reviewers can see exactly what failed.
"""
from __future__ import annotations

import re
from datetime import date, datetime
from typing import Any, Optional

from rapidfuzz import fuzz, process

MONTHS = {m: i + 1 for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])}
MONTHS.update({"sept": 9, "january": 1, "february": 2, "march": 3, "april": 4, "june": 6, "july": 7, "august": 8,
               "september": 9, "october": 10, "november": 11, "december": 12})
HINDI_MONTHS = {"जनवरी": 1, "फ़रवरी": 2, "फरवरी": 2, "मार्च": 3, "अप्रैल": 4, "मई": 5, "जून": 6, "जुलाई": 7, "अगस्त": 8,
                "सितंबर": 9, "सितम्बर": 9, "अक्टूबर": 10, "अक्तूबर": 10, "नवंबर": 11, "नवम्बर": 11, "दिसंबर": 12, "दिसम्बर": 12}
DEVANAGARI_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")
TRUE = {"yes", "y", "true", "✓", "✔", "x", "checked", "हाँ", "हां", "haan", "1"}
FALSE = {"no", "n", "false", "unchecked", "नहीं", "nahi", "0", ""}


def to_number(v: Any) -> Optional[float]:
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return float(v)
    s = str(v).translate(DEVANAGARI_DIGITS)
    mult = 1.0
    if re.search(r"\b(lakhs?|lacs?)\b", s, re.I):
        mult = 1e5
    elif re.search(r"\b(crores?|cr)\b", s, re.I):
        mult = 1e7
    elif re.search(r"\b(k|thousand)\b", s, re.I):
        mult = 1e3
    elif re.search(r"\b(mn|million)\b", s, re.I):
        mult = 1e6
    s = re.sub(r"[₹$€£¥]|\b(rs\.?|inr|usd|eur|gbp|lakhs?|lacs?|crores?|cr|k|thousand|mn|million)\b", "", s, flags=re.I)
    neg = ("(" in s and ")" in s) or s.strip().startswith("-")
    m = re.search(r"\d[\d,]*(?:\.\d+)?|\.\d+", s)
    if not m:
        return None
    try:
        n = float(m.group(0).replace(",", "")) * mult
    except ValueError:
        return None
    return -n if neg else n


def to_date(v: Any) -> Optional[str]:
    if isinstance(v, (date, datetime)):
        return v.strftime("%Y-%m-%d")
    s = str(v).translate(DEVANAGARI_DIGITS).strip()
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
        return s
    for pat in (r"(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})",):           # ISO-ish
        m = re.search(pat, s)
        if m:
            y, mo, d = map(int, m.groups())
            return _fmt(y, mo, d)
    m = re.search(r"(\d{1,2})[-/.](\d{1,2})[-/.](\d{2,4})", s)       # dd/mm/yyyy (Indian & EU default)
    if m:
        d, mo, y = map(int, m.groups())
        if mo > 12 and d <= 12:
            d, mo = mo, d
        y = y + 2000 if y < 100 else y
        return _fmt(y, mo, d)
    tokens = re.findall(r"[A-Za-zऀ-ॿ]+|\d+", s)
    nums = [int(t) for t in tokens if t.isdigit()]
    mon = next((m for m in (MONTHS.get(t.lower().rstrip(".")) or HINDI_MONTHS.get(t) for t in tokens if not t.isdigit()) if m), None)
    if mon and len(nums) >= 2:
        d = next((n for n in nums if n <= 31), None)
        y = next((n for n in nums if n > 31), None)
        if d and y:
            return _fmt(y + 2000 if y < 100 else y, mon, d)
    return None


def _fmt(y: int, m: int, d: int) -> Optional[str]:
    try:
        return date(y, m, d).strftime("%Y-%m-%d")
    except ValueError:
        return None


def to_bool(v: Any) -> Optional[bool]:
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in TRUE:
        return True
    if s in FALSE:
        return False
    return None


def to_enum(v: Any, options: list) -> Optional[Any]:
    if v in options:
        return v
    best = process.extractOne(str(v).lower(), {o: str(o).lower() for o in options}, scorer=fuzz.token_set_ratio)
    return best[2] if best and best[1] >= 80 else None


def coerce(value: Any, field: dict[str, Any]) -> tuple[Any, bool, Optional[str]]:
    """Return (typed_value, type_ok, note)."""
    if value is None or value == "":
        return None, True, None
    t = field.get("type", "string")
    fmt = field.get("format")
    if field.get("enum"):
        e = to_enum(value, field["enum"])
        return (e, True, None) if e is not None else (value, False, f"not one of {field['enum']}")
    if t in ("number", "integer"):
        n = to_number(value)
        if n is None:
            return value, False, "not numeric"
        if t == "integer":
            return int(round(n)), abs(n - round(n)) < 1e-9, None if abs(n - round(n)) < 1e-9 else "rounded to integer"
        return n, True, None
    if t == "boolean":
        b = to_bool(value)
        return (b, True, None) if b is not None else (value, False, "not boolean")
    if t == "array":
        if not isinstance(value, list):
            return value, False, "expected a list"
        items = field.get("items") or {}
        props = items.get("properties") or {}
        out = []
        ok_all = True
        for row in value:
            if isinstance(row, dict) and props:
                typed = {}
                for k, sub in props.items():
                    tv, ok, _ = coerce(row.get(k), {"type": sub.get("type", "string"), "format": sub.get("format"), "enum": sub.get("enum")})
                    typed[k] = tv
                    ok_all &= ok
                out.append(typed)
            else:
                out.append(row)
        return out, ok_all, None if ok_all else "some item fields failed type coercion"
    if fmt in ("date", "date-time") or re.search(r"(^|[_.\s])(date|dated|dob)($|[_.\s])", field.get("path", ""), re.I):
        d = to_date(value)
        return (d, True, None) if d else (str(value), False, "not a recognisable date")
    return str(value) if not isinstance(value, (dict, list)) else value, True, None
