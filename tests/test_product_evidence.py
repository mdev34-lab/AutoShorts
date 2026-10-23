"""Fact atoms: only bindable, non-negated, attributed claims are extracted."""

import pytest

from autoshorts.modules.product.errors import ProductPipelineError
from autoshorts.modules.product.evidence import atoms_from_text
from autoshorts.modules.product.narration import compose_showcase_script
from autoshorts.modules.product.script_guard import assert_script_within_atoms


def _surfaces(text):
    return sorted(a.surface for a in atoms_from_text(text, "loja.example.org"))


@pytest.mark.parametrize(
    "text",
    [
        "Projetor HY300 Pro sem Bluetooth",
        "Projetor HY300 Pro não tem suporte para Bluetooth",
        "Projetor HY300 Pro nem Bluetooth nem HDMI",
        "Fabricado no Japão, certificado CE, alumínio aeronáutico",
        "Bateria para o dia todo",
        "R$ 99 de desconto 20W",
    ],
)
def test_unsupported_or_negated_clauses_yield_no_atoms(text):
    assert _surfaces(text) == []


def test_resolution_never_takes_a_neighbouring_attribute():
    assert _surfaces("8GB RAM Wi-Fi 4K") == ["8GB de memória", "Wi-Fi", "resolução 4K"]


def test_multi_spec_binds_each_quantity_to_its_own_attribute():
    assert _surfaces("8GB de RAM e 256GB de armazenamento") == [
        "256GB de armazenamento",
        "8GB de memória",
    ]


def test_load_versus_weight_ambiguity_is_dropped_not_guessed():
    assert _surfaces("Suporta 2kg peso 1kg") == ["1kg de peso"]


def test_narration_is_attributed_and_deduplicated():
    atoms = atoms_from_text("4K Wi-Fi 8GB RAM", "a.example.org") + atoms_from_text(
        "4K Wi-Fi 8GB RAM", "b.example.org"
    )
    script = compose_showcase_script("Projetor HY300 Pro", atoms)
    text = " ".join(script)
    assert "Segundo o site a.example.org" in text
    assert "b.example.org" not in text
    assert_script_within_atoms(script, atoms)


def test_guard_rejects_spec_and_feature_claims_outside_the_atoms():
    """Finite classes only (specs, known features, certification/price terms):
    the real protection is that the script is composed from atoms alone."""
    atoms = atoms_from_text("8GB RAM Wi-Fi 4K", "a.example.org")
    for bad in (
        "O projetor traz 256GB de armazenamento.",
        "O projetor tem Bluetooth.",
        "O projetor traz 8GB de armazenamento.",
    ):
        with pytest.raises(ProductPipelineError):
            assert_script_within_atoms([bad], atoms)
