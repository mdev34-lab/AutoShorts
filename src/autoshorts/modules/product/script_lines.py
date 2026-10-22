"""Preserve the script's line structure for scene<->sentence mapping.

Each non-empty script line is ONE scene unit (one narration sentence per
scene). Punctuation splitting only happens WITHIN a line; sentences are
never merged across lines.
"""
from __future__ import annotations

import re


def script_lines(raw_text: str) -> list[str]:
    """Non-empty script lines with internal whitespace collapsed."""
    lines = []
    for line in raw_text.splitlines():
        cleaned = re.sub(r"\s+", " ", line).strip()
        if cleaned:
            lines.append(cleaned)
    return lines


def split_line_sentences(line: str) -> list[str]:
    """Split a line into sentences on terminal punctuation; ensure a terminator."""
    pieces = [p.strip() for p in re.split(r"(?<=[.!?])\s+", line) if p.strip()]
    return [p if p[-1] in ".!?…" else p + "." for p in pieces]
