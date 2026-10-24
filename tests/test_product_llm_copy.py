"""LLM copy is accepted only inside the closed grammar: product + neutral
possession verb + verbatim atom surfaces. Meaning and polarity are fixed by
construction, not by a word blacklist."""

import pytest

from autoshorts.modules.product.evidence import Atom, atoms_from_text
from autoshorts.modules.product.llm_copy import line_matches_grammar, polish_script
from autoshorts.modules.product.narration import compose_showcase_script

ATOMS = atoms_from_text(
    "HY300 Pro 4K Wi-Fi 8GB RAM 128GB de armazenamento", "a.example.org"
)
BASE = compose_showcase_script("Projetor HY300 Pro", ATOMS)
CODE = "HY300 Pro"


def _caller(text):
    return lambda _prompt: text


def test_closed_form_lines_are_kept_and_extend_the_script():
    raw = "\n".join(
        [
            "Segundo o site a.example.org, o HY300 Pro tem resolução 4K.",
            "- Segundo o site a.example.org, o HY300 Pro conta com Wi-Fi.",
            "1. O HY300 Pro também traz 8GB de memória e 128GB de armazenamento.",
        ]
    )
    out = polish_script(BASE, ATOMS, CODE, caller=_caller(raw))
    assert out != BASE and out[0] == BASE[0] and out[-1] == BASE[-1]
    assert "128GB de armazenamento" in " ".join(out)


@pytest.mark.parametrize(
    "line",
    [
        # reviewer probes: comparisons and polarity reversals reusing atom words
        "O HY300 Pro tem mais memória que outro modelo.",
        "O HY300 Pro tem mais armazenamento que outro modelo.",
        "O HY300 Pro tem mais potência que outro modelo.",
        "HY300 Pro: no Bluetooth.",
        "O HY300 Pro não tem Wi-Fi.",
        "O HY300 Pro tem Wi-Fi mas sem 8GB de memória.",
        # relation between facts, not possession by the product
        "O Wi-Fi tem 8GB de memória.",
        "O HY300 Pro tem 8GB de memória que o Wi-Fi.",
        # opinion / added facts / unlisted values
        "O HY300 Pro é incrível e muito potente.",
        "Segundo o site a.example.org, o HY300 Pro tem 16GB de memória.",
        "Segundo o site a.example.org, o HY300 Pro tem Bluetooth.",
        # unknown host named
        "Segundo o site b.example.net, o HY300 Pro tem Wi-Fi.",
    ],
)
def test_grammar_rejects_unsupported_meaning(line):
    assert not line_matches_grammar(line, ATOMS, CODE), line


@pytest.mark.parametrize(
    "line",
    [
        "O HY300 Pro tem Wi-Fi.",
        "Segundo o site a.example.org, ele conta com Wi-Fi e resolução 4K.",
        "O HY300 Pro traz 8GB de memória, 128GB de armazenamento e Wi-Fi.",
    ],
)
def test_grammar_accepts_plain_possession_of_atoms(line):
    assert line_matches_grammar(line, ATOMS, CODE), line


def test_host_attribution_must_match_the_fact_source():
    mixed = ATOMS + [Atom(("feature", "bluetooth"), "Bluetooth", "b.example.net")]
    assert not line_matches_grammar(
        "Segundo o site a.example.org, o HY300 Pro tem Bluetooth.", mixed, CODE
    )
    assert line_matches_grammar(
        "Segundo o site b.example.net, o HY300 Pro tem Bluetooth.", mixed, CODE
    )


def test_compact_two_line_output_that_covers_every_fact_is_accepted():
    covered = "\n".join(
        [
            "Segundo o site a.example.org, o HY300 Pro tem resolução 4K, Wi-Fi e 8GB de memória.",
            "O HY300 Pro também traz 128GB de armazenamento.",
        ]
    )
    out = polish_script(BASE, ATOMS, CODE, caller=_caller(covered))
    assert len(out) == 4 and out != BASE


def test_valid_lines_that_drop_a_required_fact_fall_back():
    partial = "O HY300 Pro tem Wi-Fi."
    assert polish_script(BASE, ATOMS, CODE, caller=_caller(partial)) == BASE


def test_invalid_or_empty_output_falls_back_to_base():
    raw = "\n".join(
        [
            "O HY300 Pro tem mais memória que outro modelo.",
            "HY300 Pro: no Bluetooth.",
        ]
    )
    assert polish_script(BASE, ATOMS, CODE, caller=_caller(raw)) == BASE
    assert polish_script(BASE, ATOMS, CODE, caller=_caller("")) == BASE


def test_failing_model_falls_back_to_base():
    def boom(_p):
        raise RuntimeError("no credits")

    assert polish_script(BASE, ATOMS, CODE, caller=boom) == BASE


def test_coverage_requires_the_same_fact_identity_not_similar_words():
    """A line reusing the words 'memória' and '8GB' for another fact must not
    count as covering the 8GB-memory fact; the 16GB claim is also not a fact."""
    out = polish_script(
        BASE,
        ATOMS,
        CODE,
        caller=_caller(
            "Segundo o site a.example.org, o HY300 Pro tem Wi-Fi e resolução 4K.\n"
            "O HY300 Pro tem 16GB de memória e 128GB de armazenamento."
        ),
    )
    assert out == BASE


def test_attribution_to_the_base_source_must_survive():
    no_host = "\n".join(
        [
            "O HY300 Pro tem resolução 4K, Wi-Fi e 8GB de memória.",
            "O HY300 Pro também traz 128GB de armazenamento.",
        ]
    )
    assert polish_script(BASE, ATOMS, CODE, caller=_caller(no_host)) == BASE
