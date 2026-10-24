"""Cheap script/language detection from Unicode ranges (no model needed).

Good enough to tag chunks as Latin / Devanagari / Arabic / CJK ... and to
decide which OCR language pack or VLM prompt to use. Replace with fastText
`lid.176` for finer-grained language IDs if needed.
"""
from __future__ import annotations

import unicodedata
from collections import Counter

SCRIPT_TO_LANG = {
    "Latin": "en", "Devanagari": "hi", "Bengali": "bn", "Gurmukhi": "pa", "Gujarati": "gu",
    "Tamil": "ta", "Telugu": "te", "Kannada": "kn", "Malayalam": "ml", "Oriya": "or",
    "Arabic": "ar", "Cyrillic": "ru", "Han": "zh", "Hiragana": "ja", "Katakana": "ja", "Hangul": "ko",
    "Thai": "th", "Hebrew": "he", "Greek": "el",
}

# tesseract language pack for each script (used when engine=tesseract)
SCRIPT_TO_TESS = {
    "Latin": "eng", "Devanagari": "hin", "Bengali": "ben", "Gurmukhi": "pan", "Gujarati": "guj",
    "Tamil": "tam", "Telugu": "tel", "Kannada": "kan", "Malayalam": "mal", "Oriya": "ori",
    "Arabic": "ara", "Cyrillic": "rus", "Han": "chi_sim", "Hiragana": "jpn", "Katakana": "jpn",
    "Hangul": "kor", "Thai": "tha", "Hebrew": "heb", "Greek": "ell",
}


def _script_of(ch: str) -> str | None:
    if not ch.isalpha():
        return None
    try:
        name = unicodedata.name(ch)
    except ValueError:
        return None
    first = name.split(" ")[0]
    if first in ("LATIN",):
        return "Latin"
    if first == "DEVANAGARI":
        return "Devanagari"
    if first == "CJK":
        return "Han"
    for s in ("BENGALI", "GURMUKHI", "GUJARATI", "TAMIL", "TELUGU", "KANNADA", "MALAYALAM", "ORIYA",
              "ARABIC", "CYRILLIC", "HIRAGANA", "KATAKANA", "HANGUL", "THAI", "HEBREW", "GREEK"):
        if first == s:
            return s.capitalize()
    return None


def detect_scripts(text: str, min_share: float = 0.05) -> list[str]:
    counts = Counter(s for s in (_script_of(c) for c in text) if s)
    total = sum(counts.values())
    if total == 0:
        return []
    return [s for s, n in counts.most_common() if n / total >= min_share]


def detect_languages(text: str) -> list[str]:
    langs = []
    for s in detect_scripts(text):
        lang = SCRIPT_TO_LANG.get(s)
        if lang and lang not in langs:
            langs.append(lang)
    return langs


def is_bilingual(text: str) -> bool:
    return len(detect_scripts(text, min_share=0.15)) >= 2
