"""Rule-based contract segmentation (initial heuristic for Phase 2)."""

from __future__ import annotations

import re
from typing import Any

# Heading-like lines: numbered sections, Section/Article keywords, or short ALL CAPS lines
_HEADING_LINE = re.compile(
    r"^(?:SECTION|Section|SECTIONS|Article|ARTICLE|\d+(?:\.\d+)*[\.)])\s+.+\s*$",
    re.MULTILINE,
)


def split_clauses(contract_text: str | Any) -> list[str]:
    """
    Segment raw contract text using newlines, heading heuristics, and sentence punctuation.

    This is a **baseline** splitter (not legal-grade). It is reproducible and good for
    before/after visualization in ``02_clause_segmentation.ipynb``.
    """
    if contract_text is None:
        return []
    text = str(contract_text).replace("\r\n", "\n").replace("\r", "\n").strip()
    if text.lower() == "nan":
        return []
    if not text:
        return []

    segments: list[str] = []
    for block in re.split(r"\n\s*\n+", text):
        block = block.strip()
        if not block:
            continue
        lines = [ln.strip() for ln in block.split("\n") if ln.strip()]
        if not lines:
            continue
        buf: list[str] = []
        for line in lines:
            if _is_heading_line(line):
                if buf:
                    segments.extend(_split_on_punctuation(_flush(buf)))
                    buf = []
                segments.append(line)
            else:
                buf.append(line)
        if buf:
            segments.extend(_split_on_punctuation(_flush(buf)))

    out = [s.strip() for s in segments if s and s.strip()]
    return out


def _is_heading_line(line: str) -> bool:
    if len(line) < 3:
        return False
    if _HEADING_LINE.match(line):
        return True
    # Short shouty titles (common in exhibits)
    if line.isupper() and len(line) <= 160 and len(line.split()) <= 14:
        return True
    return False


def _flush(buf: list[str]) -> str:
    return " ".join(s for s in buf if s).strip()


def _split_on_punctuation(paragraph: str) -> list[str]:
    """Split on `. ` / `; ` / `: ` boundaries while keeping segments non-empty."""
    if not paragraph:
        return []
    parts = re.split(r"(?<=[.;:])\s+", paragraph)
    return [p.strip() for p in parts if p.strip()]
