"""Deterministic product narration built only from the verified name.

The showcase script is composed from facts parsed out of the verified
product name - there is no free text generation in this path, so no
fabricated specification, feature, certification or benefit can reach the
narration. Connective phrasing carries no product facts: a spec is narrated
with an attribute word ("8GB de RAM") only when the name itself binds the
two; otherwise the bare verified spec is used ("com 8GB").
"""

from __future__ import annotations

import re

from .provenance import primary_model_code
from .script_guard import (
    _ATTRIBUTE_WORDS,
    _RESOLUTION_PATTERN,
    _UNIT_PATTERN,
    _WORD_RE,
    _features_in,
    _normalize,
)

# Attribute words are narrated exactly as written in the verified name.
_ATTRIBUTE_SURFACE = re.compile(
    r"\b("
    + "|".join(sorted({w for w in _ATTRIBUTE_WORDS}, key=len, reverse=True))
    + r")\b",
    re.IGNORECASE,
)


def _surface_attribute(name: str, start: int, end: int, window: int = 5) -> str | None:
    """Attribute word as written in the name, within `window` words of a match."""
    before = list(_WORD_RE.finditer(_normalize(name[:start])))[-window:]
    after = list(_WORD_RE.finditer(_normalize(name[end:])))[:window]
    words = [m.group(0) for m in reversed(before)] + [m.group(0) for m in after]
    for word in words:
        if word in _ATTRIBUTE_WORDS:
            # Return the original-cased word from the name itself.
            original = re.search(rf"\b{re.escape(word)}\b", name, re.IGNORECASE)
            return original.group(0) if original else word
    return None


def _verified_specs(name: str) -> list[str]:
    """Spec clauses grounded verbatim in the name, in order of appearance."""
    clauses: list[str] = []
    for m in _UNIT_PATTERN.finditer(name):
        spec = m.group(0).strip()
        attribute = _surface_attribute(name, m.start(), m.end())
        if attribute:
            clauses.append(f"{spec} de {attribute}")
        else:
            clauses.append(spec)
    for m in _RESOLUTION_PATTERN.finditer(name):
        clauses.append(f"resolução {m.group(0).strip()}")
    return list(dict.fromkeys(clauses))


def _verified_features(name: str) -> list[str]:
    features = []
    for feature in sorted(_features_in(name)):
        original = re.search(
            rf"(?<![a-z0-9]){re.escape(feature)}(?![a-z0-9])", name, re.IGNORECASE
        )
        features.append(original.group(0) if original else feature)
    return features


def _surface_code(name: str, code: str) -> str:
    """The model code as written in the name (original casing and spacing)."""
    tokens = list(re.finditer(r"[A-Za-z0-9]+", name))
    normalized = [_normalize(t.group(0)) for t in tokens]
    for start in range(len(tokens)):
        concatenated = ""
        for end in range(start, min(len(tokens), start + 4)):
            concatenated += normalized[end]
            if concatenated == code:
                return " ".join(t.group(0) for t in tokens[start : end + 1])
    return code


def compose_showcase_script(product_name: str) -> list[str]:
    """Build the 3-paragraph showcase script from verified name facts only."""
    name = " ".join(product_name.split())
    primary = _surface_code(name, primary_model_code(name))
    specs = _verified_specs(name)
    features = _verified_features(name)

    hook = f"Conheça {name}."

    facts: list[str] = []
    if specs:
        facts.append(
            "Traz "
            + ", ".join(specs[:-1])
            + (" e " if len(specs) > 1 else "")
            + specs[-1]
            + "."
        )
    if features:
        facts.append("Com " + " e ".join(features) + ".")
    if not facts:
        facts.append(f"Vale conferir o {primary} de perto.")

    close = f"Confira todos os detalhes do {primary} no link."

    return [hook, *facts, close]
