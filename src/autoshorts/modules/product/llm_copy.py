"""Narration lines written by the LLM from verified fact atoms, in a CLOSED grammar.

The model receives only the attributed atoms. Its output is never trusted and
is not filtered by a word blacklist: a line is accepted only if, once the
product name, the page host and whole atom surfaces are replaced by
placeholders, it matches exactly

    [segundo o site HOST,] [além disso|também] [o] (PRODUTO|ele|ela)
    [também|ainda] VERBO ATOM ((e|,) ATOM)*

with VERBO from a short list of neutral possession verbs. That form can only
assert "the product has <atom>, <atom>": no negation, comparison, relation
between facts, adjective or new noun can occur, so meaning and polarity are
fixed by construction. What the model contributes is grouping, order, the
choice of verb and where the source is named. Anything else is dropped line
by line; fewer than three accepted lines falls back to the deterministic
script. Richer prose (benefits, comparisons, explanations) would need a real
entailment check against the atoms and is not provided here.
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

_VERBS = "traz|tem|conta com|oferece|possui|inclui|vem com|apresenta|destaca|lista|cita"
_WORD = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")


def _squash(text: str) -> str:
    text = _normalize(text).replace(",", " , ")
    text = re.sub(r"[.!?;:]+(?=\s|$)", " ", text)
    return " ".join(text.split())


def line_matches_grammar(line: str, atoms: list, code: str) -> bool:
    """True only for lines of the closed form documented above."""
    work = f" {_squash(line)} "
    sources = sorted({a.source for a in atoms}, key=len, reverse=True)
    named_hosts: set[str] = set()
    for src in sources:
        token = f" {_squash(src)} "
        if token in work:
            named_hosts.add(src)
            work = work.replace(token, " HOST ")
    if len(named_hosts) > 1:
        return False
    found_surfaces = [
        sf for sf in {_squash(a.surface) for a in atoms} if f" {sf} " in work + " "
    ]
    surfaces = sorted({_squash(a.surface) for a in atoms}, key=len, reverse=True)
    for surface in surfaces:
        work = re.sub(rf"(?<![a-z0-9]){re.escape(surface)}(?![a-z0-9])", " ATOM ", work)
    work = re.sub(
        rf"(?<![a-z0-9]){re.escape(_squash(code))}(?![a-z0-9])", " PRODUTO ", work
    )
    work = " ".join(work.split())
    pattern = (
        r"(segundo o site HOST ,? )?(alem disso ,? |tambem ,? )?(o )?"
        r"(PRODUTO|ele|ela) ((tambem|ainda) )?(" + _VERBS + r") "
        r"ATOM( (, e|e|,) ATOM)*"
    )
    if re.fullmatch(pattern, work) is None:
        return False
    if named_hosts:
        host = next(iter(named_hosts))
        # every fact in the line must really come from the host it names
        stated = {
            _squash(a.surface) for a in atoms if _squash(a.surface) in found_surfaces
        }
        backed = {_squash(a.surface) for a in atoms if a.source == host}
        if not stated <= backed:
            return False
    return True


def _prompt(code: str, atoms: list) -> str:
    facts = "\n".join(f"- {a.surface} (fonte: {a.source})" for a in atoms)
    return (
        "Escreva um roteiro curto em português do Brasil para narrar um vídeo "
        f"vertical do produto {code}. Use SOMENTE os fatos abaixo, atribuídos "
        "ao site de origem na PRIMEIRA frase de fatos e de novo só na última. "
        "Cada frase tem EXATAMENTE esta forma (sem colchetes), por exemplo:\n"
        f"Segundo o site SITE, o {code} tem FATO1, FATO2 e FATO3.\n"
        f"O {code} também traz FATO4 e FATO5.\n"
        "O verbo é um de: traz, tem, conta com, oferece, possui, inclui, vem com, "
        "apresenta, destaca. Copie cada FATO exatamente como está na lista, sem "
        "reformular. Nada além disso: sem adjetivos, comparações, negações, preço "
        "ou outras palavras. Escreva de 4 a 8 frases, de 1 a 3 fatos cada, uma por "
        "linha, sem numeração, sem abertura nem chamada final. Use todos os "
        "fatos, cada um uma vez.\n\nFatos:\n" + facts
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
    kept: list[str] = []
    for _ in range(attempts):
        try:
            raw = caller(_prompt(code, atoms))
        except Exception:
            continue
        for line in raw.splitlines():
            line = re.sub(r"^[\s\-*\d.)]+", "", line).strip()
            if line in kept:
                continue
            if not line_matches_grammar(line, atoms, code):
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
