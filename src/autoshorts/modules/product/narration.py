"""Deterministic product narration that carries no unverified claims.

There is no free text generation in this path. The product name comes from
a user-controlled URL slug, so only the model identifier is narrated; no
specification, feature, certification or benefit parsed from the slug.
"""

from __future__ import annotations

import re

from .provenance import primary_model_code
from .script_guard import _normalize


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


def _join(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " e " + items[-1]


def compose_showcase_script(product_name: str, atoms: list | None = None) -> list[str]:
    """Showcase script from the model identifier plus attributed fact atoms.

    Only the identifier and structured atoms (see evidence.py) are narrated,
    each attributed to the verified page it came from; there is no free text.
    Without atoms the script contains no product claim at all.
    """
    name = " ".join(product_name.split())
    primary = _surface_code(name, primary_model_code(name))
    paragraphs = [f"Conheça o {primary}."]
    by_source: dict[str, list] = {}
    narrated: set = set()
    for atom in atoms or []:
        if atom.key in narrated:
            continue  # same fact from another host: say it once
        narrated.add(atom.key)
        by_source.setdefault(atom.source, []).append(atom)
    for source, group in list(by_source.items())[:2]:
        specs = [a.surface for a in group if a.is_spec][:4]
        feats = [a.surface for a in group if not a.is_spec][:4]
        if specs:
            paragraphs.append(
                f"Segundo o site {source}, o {primary} traz {_join(specs)}."
            )
        if feats:
            paragraphs.append(f"Segundo o site {source}, ele conta com {_join(feats)}.")
    if len(paragraphs) == 1:
        paragraphs.append(f"Veja o {primary} em detalhe, nas imagens a seguir.")
    paragraphs.append(f"Confira todos os detalhes do {primary} no link.")
    return paragraphs
