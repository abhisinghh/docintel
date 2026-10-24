"""Metrics used across document-AI benchmarks.

* normalized edit distance (NED)   – OmniDocBench text / formula, OCR line accuracy
* CER / WER                        – handwriting (IAM, IIIT-HW-Hindi)
* ANLS                             – DocVQA / InfographicVQA / MP-DocVQA
* relaxed accuracy                 – ChartQA (5 % numeric tolerance)
* TEDS (+ structure-only TEDS)     – table structure (PubTabNet, FinTabNet, OmniDocBench tables)
* bbox mAP-lite / IoU matching     – layout grounding
No GPU, no heavy deps; TEDS uses a plain recursive tree edit distance (APTED-equivalent cost model).
"""
from __future__ import annotations

import re
from functools import lru_cache
from typing import Iterable, Sequence

from rapidfuzz.distance import Levenshtein

from ..schemas import BBox


# ------------------------------------------------------------------ text


def normalize_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def ned(pred: str, gt: str, normalize: bool = True) -> float:
    """Normalized edit distance in [0, 1] (0 = identical)."""
    if normalize:
        pred, gt = normalize_text(pred), normalize_text(gt)
    if not pred and not gt:
        return 0.0
    return Levenshtein.distance(pred, gt) / max(len(pred), len(gt))


def cer(pred: str, gt: str) -> float:
    return Levenshtein.distance(pred, gt) / max(1, len(gt))


def wer(pred: str, gt: str) -> float:
    p, g = pred.split(), gt.split()
    return Levenshtein.distance(p, g) / max(1, len(g))


def anls(pred: str, gts: Sequence[str], tau: float = 0.5) -> float:
    """Average Normalized Levenshtein Similarity (DocVQA)."""
    best = 0.0
    p = normalize_text(pred)
    for gt in gts:
        g = normalize_text(gt)
        if not p and not g:
            return 1.0
        d = Levenshtein.distance(p, g) / max(len(p), len(g), 1)
        s = 1 - d
        best = max(best, s if s >= tau else 0.0)
    return best


def relaxed_accuracy(pred: str, gt: str, tol: float = 0.05) -> float:
    """ChartQA metric: numeric answers within ±5 %, else exact (case-insensitive) match."""
    def num(x: str):
        m = re.search(r"-?\d+(?:[.,]\d+)?", x.replace(",", ""))
        return float(m.group(0)) if m else None
    pn, gn = num(pred), num(gt)
    if pn is not None and gn is not None:
        return 1.0 if abs(pn - gn) <= tol * abs(gn) or (gn == 0 and pn == 0) else 0.0
    return 1.0 if normalize_text(pred) == normalize_text(gt) else 0.0


# ------------------------------------------------------------------ TEDS


class _Node:
    __slots__ = ("tag", "text", "children", "colspan", "rowspan")

    def __init__(self, tag, text="", colspan=1, rowspan=1):
        self.tag, self.text, self.colspan, self.rowspan = tag, text, colspan, rowspan
        self.children: list[_Node] = []


def _html_to_tree(html: str, structure_only: bool = False) -> _Node:
    """Table HTML → tree of tr/td nodes (spans kept as attributes, as in the PubTabNet TEDS reference)."""
    from ..ocr.backends import _parse_html_table
    root = _Node("table")
    td = _parse_html_table(html)
    if td is None:
        return root
    by_row: dict[int, list] = {}
    for c in td.cells:
        by_row.setdefault(c.row, []).append(c)
    for r in sorted(by_row):
        tr = _Node("tr")
        for c in sorted(by_row[r], key=lambda c: c.col):
            tr.children.append(_Node("td", "" if structure_only else normalize_text(c.text), c.col_span, c.row_span))
        root.children.append(tr)
    return root


def _count(n: _Node) -> int:
    return 1 + sum(_count(c) for c in n.children)


def _node_cost(a: _Node | None, b: _Node | None) -> float:
    if a is None or b is None:
        return 1.0
    if a.tag != b.tag or a.colspan != b.colspan or a.rowspan != b.rowspan:
        return 1.0
    if a.tag == "td":
        return ned(a.text, b.text, normalize=False) if (a.text or b.text) else 0.0
    return 0.0


def _ted(a: _Node, b: _Node) -> float:
    """Simple tree edit distance (Zhang–Shasha-style recursion on ordered forests; fine for table sizes)."""

    @lru_cache(maxsize=None)
    def forest(fa: tuple, fb: tuple) -> float:
        if not fa and not fb:
            return 0.0
        if not fa:
            return sum(_count(n) for n in fb)
        if not fb:
            return sum(_count(n) for n in fa)
        x, y = fa[-1], fb[-1]
        d1 = forest(fa[:-1], fb) + _count(x)                       # delete x
        d2 = forest(fa, fb[:-1]) + _count(y)                       # insert y
        d3 = forest(fa[:-1], fb[:-1]) + forest(tuple(x.children), tuple(y.children)) + _node_cost(x, y)
        return min(d1, d2, d3)

    return forest((a,), (b,))


def teds(pred_html: str, gt_html: str, structure_only: bool = False) -> float:
    """Tree-Edit-Distance-based Similarity (Zhong et al., PubTabNet).

    Note: this is a compact ordered-forest edit distance with subtree deletion cost = subtree size, which is
    an upper bound on the APTED distance used by the official scorer (identical for most table edits, slightly
    pessimistic when whole rows are inserted/deleted). For publishable numbers run the official PubTabNet
    scorer (`pip install apted`); this implementation is for quick regression checks and large tables may hit
    the recursion limit."""
    import sys
    sys.setrecursionlimit(max(sys.getrecursionlimit(), 20000))
    a, b = _html_to_tree(pred_html, structure_only), _html_to_tree(gt_html, structure_only)
    n = max(_count(a), _count(b))
    if n == 0:
        return 1.0
    return 1.0 - _ted(a, b) / n


# ------------------------------------------------------------------ layout / grounding


def match_boxes(pred: Iterable[BBox], gt: Iterable[BBox], iou_thr: float = 0.5) -> dict[str, float]:
    pred, gt = list(pred), list(gt)
    used = set()
    tp = 0
    for p in pred:
        best, best_iou = None, 0.0
        for j, g in enumerate(gt):
            if j in used:
                continue
            v = p.iou(g)
            if v > best_iou:
                best, best_iou = j, v
        if best is not None and best_iou >= iou_thr:
            used.add(best)
            tp += 1
    prec = tp / len(pred) if pred else 0.0
    rec = tp / len(gt) if gt else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"precision": prec, "recall": rec, "f1": f1, "tp": tp, "n_pred": len(pred), "n_gt": len(gt)}
