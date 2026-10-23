"""Fact atoms extracted from identity-verified product pages.

The slug and the page title are untrusted prose. A fact is narrated only
when it is a structured atom (a number with a known unit bound to its own
attribute, or a keyword feature) found in a single clause of a verified
page that also proves the exact model. Clauses that contain any negation,
price, promotion, certification or origin wording are skipped entirely, and
every narrated atom is attributed to the page host it came from.

This is a deliberately narrow, finite vocabulary: it cannot prove a claim is
true, only that a verified page states it in a clause the parser can bind.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from html.parser import HTMLParser
from urllib.parse import urlsplit

from .script_guard import (
    _ATTRIBUTE_WORDS,
    _NEGATION_WORDS,
    _PRICE_PATTERN,
    _RESOLUTION_PATTERN,
    _UNGROUNDABLE_PATTERN,
    _UNIT_PATTERN,
    _WORD_RE,
    _canon_unit,
    _features_in,
    _normalize,
)

_SAFE_UNITS = {
    "gb", "tb", "mb", "w", "kw", "wh", "mah", "hz", "khz", "mhz", "ghz",
    "mp", "px", "pol", "kg", "cm", "mm", "ansi",
}  # fmt: skip
_ATTRIBUTE_LABEL = {
    "memory": "memória",
    "storage": "armazenamento",
    "weight": "peso",
    "load": "capacidade de carga",
    "power": "potência",
    "brightness": "brilho",
    "battery": "bateria",
    "screen": "tela",
    "display": "display",
}
_FEATURE_LABEL = {
    "bluetooth": "Bluetooth",
    "wi-fi": "Wi-Fi",
    "wifi": "Wi-Fi",
    "android": "Android",
    "hdmi": "HDMI",
    "laser": "laser",
    "oled": "OLED",
    "amoled": "AMOLED",
    "qled": "QLED",
    "lcd": "LCD",
    "led": "LED",
    "ips": "IPS",
    "bateria": "bateria",
    "battery": "bateria",
}
_CLAUSE_SPLIT = re.compile(
    r"[,;:|/•()\[\]\n]|\s[-–—]\s|\s(?:e|and|com|with|\+)\s", re.IGNORECASE
)
_MAX_CLAUSE = 120


@dataclass(frozen=True)
class Atom:
    key: tuple  # ("spec", number, unit, attribute) or ("feature", name)
    surface: str
    source: str  # page host

    @property
    def is_spec(self) -> bool:
        return self.key[0] == "spec"


def host_label(url: str) -> str:
    host = (urlsplit(url).hostname or "").lower()
    return host[4:] if host.startswith("www.") else host


_AMBIGUOUS = object()


def _bound_attribute(clause: str, start: int, end: int, window: int = 2):
    """Attribute word nearest to the number inside ONE clause, or _AMBIGUOUS
    when two different attributes are equally near ("suporta 2kg peso")."""
    before = _WORD_RE.findall(_normalize(clause[:start]))[-window:]
    after = _WORD_RE.findall(_normalize(clause[end:]))[:window]
    near_before = near_after = None
    for d, word in enumerate(reversed(before), start=1):
        if word in _ATTRIBUTE_WORDS:
            near_before = (d, _ATTRIBUTE_WORDS[word])
            break
    for d, word in enumerate(after, start=1):
        if word in _ATTRIBUTE_WORDS:
            near_after = (d, _ATTRIBUTE_WORDS[word])
            break
    if near_before and near_after:
        if near_before[1] == near_after[1]:
            return near_before[1]
        if near_before[0] == near_after[0]:
            return _AMBIGUOUS
        return min(near_before, near_after)[1]
    found = near_before or near_after
    return found[1] if found else None


def _clauses(text: str) -> list[str]:
    return [c.strip() for c in _CLAUSE_SPLIT.split(text or "") if c and c.strip()]


def _has_negation(clause: str) -> bool:
    return any(w in _NEGATION_WORDS for w in _WORD_RE.findall(_normalize(clause)))


def atoms_from_text(text: str, source: str) -> list[Atom]:
    atoms: dict[tuple, Atom] = {}
    clauses = _clauses(text)
    for i, clause in enumerate(clauses):
        if len(clause) > _MAX_CLAUSE:
            continue
        # a negation here or in the previous clause may scope this one
        if _has_negation(clause) or (i > 0 and _has_negation(clauses[i - 1])):
            continue
        norm = _normalize(clause)
        if _UNGROUNDABLE_PATTERN.search(norm) or _PRICE_PATTERN.search(norm):
            continue
        for pattern, unit_group in ((_UNIT_PATTERN, 2), (_RESOLUTION_PATTERN, 2)):
            for m in pattern.finditer(clause):
                number = m.group(1).replace(",", ".")
                unit = _canon_unit(m.group(unit_group))
                resolution = pattern is _RESOLUTION_PATTERN
                if not resolution and unit not in _SAFE_UNITS:
                    continue
                # Resolutions (4K, 1080p) never take an attribute word.
                attribute = (
                    None if resolution else _bound_attribute(clause, m.start(), m.end())
                )
                if attribute is _AMBIGUOUS:
                    continue
                spec = m.group(0).replace(" ", "")
                if attribute:
                    surface = f"{spec} de {_ATTRIBUTE_LABEL[attribute]}"
                elif resolution:
                    surface = f"resolução {spec}"
                else:
                    surface = spec
                key = ("spec", number, unit, attribute)
                atoms.setdefault(key, Atom(key, surface, source))
        for feature in _features_in(clause):
            if feature in ("bateria", "battery"):
                continue  # every device has one; day-long claims are not atoms
            name = _FEATURE_LABEL.get(feature, feature)
            key = ("feature", name.casefold())
            atoms.setdefault(key, Atom(key, name, source))
    return list(atoms.values())


class _BodyLines(HTMLParser):
    """Short text lines from list/table/description cells inside <main>/<article>."""

    _SKIP = {"script", "style", "nav", "footer", "header", "aside", "noscript"}
    _LINE = {"li", "td", "th", "dd", "p"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.lines: list[str] = []
        self._main = 0
        self._skip = 0
        self._buf: list[str] | None = None

    def handle_starttag(self, tag, attrs):
        if tag in ("main", "article"):
            self._main += 1
        elif tag in self._SKIP:
            self._skip += 1
        elif tag in self._LINE and self._main and not self._skip:
            self._buf = []

    def handle_endtag(self, tag):
        if tag in ("main", "article"):
            self._main = max(0, self._main - 1)
        elif tag in self._SKIP:
            self._skip = max(0, self._skip - 1)
        elif tag in self._LINE and self._buf is not None:
            line = " ".join("".join(self._buf).split())
            if line:
                self.lines.append(line)
            self._buf = None

    def handle_data(self, data):
        if self._buf is not None and not self._skip:
            self._buf.append(data)


def page_atoms(page) -> list[Atom]:
    """Atoms from a verified page: identity fields plus bounded body spec lines."""
    source = host_label(page.url)
    texts = [page.title, page.h1, getattr(page, "meta_title", "")]
    parser = _BodyLines()
    try:
        parser.feed(getattr(page, "html_text", "") or "")
    except Exception:  # malformed HTML: identity fields only
        parser.lines = []
    texts.extend(line for line in parser.lines[:200] if len(line) <= _MAX_CLAUSE)
    merged: dict[tuple, Atom] = {}
    for text in texts:
        if isinstance(text, str) and text:
            for atom in atoms_from_text(text, source):
                merged.setdefault(atom.key, atom)
    return list(merged.values())
