"""Longer narration copy written by the LLM from verified fact atoms only.

The model receives ONLY the attributed atoms. Its output is never trusted:
every line must pass the atom guard AND a lexical containment check - each
content word must come from an atom surface, the page host, the model code or
a small neutral connective list. Prose that adds a qualitative or factual word
(incrível, potente, rápido, ...) is dropped line by line. Anything dropped or
missing falls back to the deterministic atom script, so this can lengthen the
copy but never widen what it claims.
"""

from __future__ import annotations

import os
import re

import requests

from ..config import API_KEY, API_URL
from .errors import ProductPipelineError
from .script_guard import _normalize, assert_script_within_atoms

# The repo-wide MODEL_TEXT default needs paid credits on the dedicated route;
# a reasoning model needs a large token budget or it returns empty text.
COPY_MODEL = os.getenv("AUTOSHORTS_COPY_MODEL", "openai")

# Neutral words with no product claim. Deliberately small.
CONNECTIVES = set(
    "a o as os um uma de do da dos das e em no na nos nas com para por que se "
    "segundo site ele ela isso este esta esse essa traz tem conta oferece "
    "mostra informa lista aponta ver veja confira conheca ainda tambem alem "
    "disso outro outra mais link detalhes detalhe imagens seguir todos todas "
    "pagina oficial especificacoes ficha tecnica produto modelo cita destaca "
    "inclui possui sobre ao aos pela pelo".split()
)
_WORD = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")


def allowed_vocabulary(atoms: list, code: str) -> set[str]:
    vocab = set(CONNECTIVES)
    pieces = [a.surface for a in atoms] + [a.source for a in atoms] + [code]
    for piece in pieces:
        vocab.update(_WORD.findall(_normalize(piece)))
        vocab.update(re.findall(r"[a-z0-9]+", _normalize(piece)))
    return vocab


def line_is_contained(line: str, vocab: set[str]) -> bool:
    return all(
        tok in vocab or all(p in vocab for p in tok.split("-"))
        for tok in _WORD.findall(_normalize(line))
    )


def _prompt(code: str, atoms: list) -> str:
    facts = "\n".join(f"- {a.surface} (fonte: {a.source})" for a in atoms)
    return (
        "Escreva um roteiro curto em português do Brasil para narrar um vídeo "
        f"vertical do produto {code}. Use SOMENTE os fatos abaixo, atribuídos "
        "ao site de origem (ex.: 'Segundo o site X, ...'). Proibido: adjetivos "
        "de opinião, comparações, preço, benefícios, qualquer fato que não "
        "esteja na lista. Escreva de 6 a 9 frases curtas, uma por linha, sem "
        "numeração, repetindo e reorganizando os fatos se precisar, e termine "
        "chamando para ver os detalhes no link.\n\nFatos:\n" + facts
    )


def _call(prompt: str, timeout: float = 60.0) -> str:
    resp = requests.post(
        API_URL,
        headers={"Authorization": f"Bearer {API_KEY}"},
        json={
            "model": COPY_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
            "max_tokens": 2500,
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"] or ""


def polish_script(
    base: list[str], atoms: list, code: str, *, caller=_call, attempts: int = 2
) -> list[str]:
    """Return the base script extended with contained LLM lines, or base."""
    if not atoms or not API_KEY and caller is _call:
        return base
    vocab = allowed_vocabulary(atoms, code)
    kept: list[str] = []
    for _ in range(attempts):
        try:
            raw = caller(_prompt(code, atoms))
        except Exception:
            continue
        for line in raw.splitlines():
            line = re.sub(r"^[\s\-*\d.)]+", "", line).strip()
            if len(line.split()) < 4 or line in kept:
                continue
            if not line_is_contained(line, vocab):
                continue
            try:
                assert_script_within_atoms([line], atoms)
            except ProductPipelineError:
                continue
            kept.append(line)
        if len(kept) >= 6:
            break
    if len(kept) < 3:
        return base
    return [base[0], *kept[:9], base[-1]]
