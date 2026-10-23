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


def compose_showcase_script(product_name: str) -> list[str]:
    """Build the 3-paragraph showcase script without any slug-derived claim.

    The product name is a user-controlled URL slug, and narration is
    synthesized before any source page is verified. Specs, features and
    negations ("sem Bluetooth") parsed from it are therefore claims, not
    facts, and are never narrated. Only the model identifier is spoken: it
    names the product and is exactly what the image pipeline later proves
    against external sources. Connective phrasing carries no product facts.
    """
    name = " ".join(product_name.split())
    primary = _surface_code(name, primary_model_code(name))
    return [
        f"Conheça o {primary}.",
        f"Veja o {primary} em detalhe, nas imagens a seguir.",
        f"Confira todos os detalhes do {primary} no link.",
    ]
