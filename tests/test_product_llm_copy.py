"""LLM copy is only kept when every word is contained in the evidence."""

from autoshorts.modules.product.evidence import atoms_from_text
from autoshorts.modules.product.llm_copy import line_is_contained, polish_script
from autoshorts.modules.product.narration import compose_showcase_script

ATOMS = atoms_from_text(
    "HY300 Pro 4K Wi-Fi 8GB RAM 128GB de armazenamento", "a.example.org"
)
BASE = compose_showcase_script("Projetor HY300 Pro", ATOMS)


def _caller(text):
    return lambda _prompt: text


def test_contained_lines_are_kept_and_extend_the_script():
    raw = "\n".join(
        [
            "Segundo o site a.example.org, o HY300 Pro tem resolução 4K.",
            "- Segundo o site a.example.org, o HY300 Pro conta com Wi-Fi.",
            "1. Segundo o site a.example.org, o HY300 Pro tem 8GB de memória.",
        ]
    )
    out = polish_script(BASE, ATOMS, "HY300 Pro", caller=_caller(raw))
    assert out[0] == BASE[0] and out[-1] == BASE[-1]
    assert len(out) == 5


def test_opinion_and_added_facts_are_dropped_line_by_line():
    raw = "\n".join(
        [
            "O HY300 Pro é incrível e muito potente demais.",
            "Segundo o site a.example.org, o HY300 Pro tem 16GB de memória.",
            "Segundo o site a.example.org, o HY300 Pro tem Bluetooth.",
            "Segundo o site a.example.org, o HY300 Pro tem resolução 4K.",
        ]
    )
    # only one clean line survives (< 3): fall back to the deterministic base
    assert polish_script(BASE, ATOMS, "HY300 Pro", caller=_caller(raw)) == BASE


def test_failing_model_falls_back_to_base():
    def boom(_p):
        raise RuntimeError("no credits")

    assert polish_script(BASE, ATOMS, "HY300 Pro", caller=boom) == BASE


def test_containment_rejects_unlisted_words():
    from autoshorts.modules.product.llm_copy import allowed_vocabulary

    vocab = allowed_vocabulary(ATOMS, "HY300 Pro")
    assert line_is_contained(
        "Segundo o site a.example.org, ele conta com Wi-Fi.", vocab
    )
    assert not line_is_contained("Ele é rápido e silencioso.", vocab)
