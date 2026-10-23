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

import html
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
                if attribute is None and not resolution:
                    continue  # a bare "7mm" or "5W" says nothing narratable
                spec = m.group(0).replace(" ", "")
                unit_text = m.group(unit_group)
                if len(unit_text) > 2 and unit_text.isalpha():
                    spec = f"{number} {unit_text}"  # "290 ANSI", read aloud
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

    def __init__(self, whole_page: bool = False) -> None:
        super().__init__(convert_charrefs=True)
        self.lines: list[str] = []
        # Pages without <main>/<article> (many storefront themes) are read as
        # a whole; nav/header/footer/aside/script are still skipped.
        self._main = 1 if whole_page else 0
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


_META_RE = re.compile(
    r"<meta\s[^>]*?(?:name|property)=[\"'](?:description|og:description|"
    r"twitter:description)[\"'][^>]*?content=[\"']([^\"']*)[\"']",
    re.I,
)


def _meta_descriptions(html_text: str) -> list[str]:
    """Publisher-declared page descriptions (same page, same host)."""
    seen: list[str] = []
    for raw in _META_RE.findall(html_text):
        text = " ".join(html.unescape(raw).split())
        if text and text not in seen:
            seen.append(text)
    return [t[: _MAX_CLAUSE * 4] for t in seen]


_OTHER_MODEL = re.compile(r"\b[a-z]{1,3}\d{3}", re.I)


def _code_pattern(code: str) -> re.Pattern:
    body = re.escape(code).replace("plus", r"(?:plus|\+)")
    return re.compile(body, re.I)


def _disputed_near_code(html_text: str, codes: list[str]) -> set[tuple]:
    """(unit, other number) pairs stated right after the model code anywhere on
    the page, nav and menus included (e.g. a product menu listing the same
    model with another brightness). Used only to REMOVE claims."""
    text = re.sub(
        r"<script.*?</script>|<style.*?</style>", " ", html_text, flags=re.S | re.I
    )
    text = " ".join(html.unescape(re.sub(r"<[^>]+>", " ", text)).split())
    seen: set[tuple] = set()
    for code in codes:
        for m in _code_pattern(code).finditer(text):
            window = text[m.end() : m.end() + 70]
            cut = _OTHER_MODEL.search(window)
            if cut:
                window = window[: cut.start()]
            for um in _UNIT_PATTERN.finditer(window):
                seen.add((_canon_unit(um.group(2)), um.group(1).replace(",", ".")))
    return seen


def page_atoms(page) -> list[Atom]:
    """Atoms from a verified page: identity fields plus bounded body spec lines."""
    source = host_label(page.url)
    texts = [page.title, page.h1, getattr(page, "meta_title", "")]
    html_text = getattr(page, "html_text", "") or ""
    texts.extend(_meta_descriptions(html_text))
    has_region = re.search(r"<(main|article)[\s>]", html_text, re.I) is not None
    parser = _BodyLines(whole_page=not has_region)
    try:
        parser.feed(html_text)
    except Exception:  # malformed HTML: identity fields only
        parser.lines = []
    texts.extend(line for line in parser.lines[:200] if len(line) <= _MAX_CLAUSE)
    merged: dict[tuple, Atom] = {}
    for text in texts:
        if isinstance(text, str) and text:
            for atom in atoms_from_text(text, source):
                merged.setdefault(atom.key, atom)
    atoms = list(merged.values())
    codes = list(getattr(page, "identity_codes", None) or [])
    if codes:
        disputed = _disputed_near_code(html_text, codes)
        atoms = [
            a
            for a in atoms
            if not (
                a.key[0] == "spec"
                and any(u == a.key[2] and n != a.key[1] for u, n in disputed)
            )
        ]
    return atoms


def reconcile_atoms(atoms: list[Atom]) -> list[Atom]:
    """Drop disputed or uncorroborated numeric claims; never pick a side.

    - Resolution atoms (720P, 4K): dropped when two atoms disagree.
    - Attribute-bound specs (brightness, memory, storage, weight...): dropped
      when any collected atom states another value for the same attribute.
      Source COUNT is not truth (hosts copy each other or describe other
      variants), so agreement is not required, only absence of dispute.
    - Features carry no number and pass through.
    """
    groups: dict[tuple, list[Atom]] = {}
    for atom in atoms:
        if atom.key[0] != "spec":
            continue
        _kind, _number, unit, attribute = atom.key
        group = ("resolution",) if attribute is None else ("attr", attribute)
        groups.setdefault(group, []).append(atom)
    keep: list[Atom] = []
    for atom in atoms:
        if atom.key[0] != "spec":
            keep.append(atom)
            continue
        attribute = atom.key[3]
        group = ("resolution",) if attribute is None else ("attr", attribute)
        members = groups[group]
        values = {a.key[1:3] for a in members}
        if len(values) > 1:
            continue  # sources disagree: omit, do not choose
        keep.append(atom)
    return keep
