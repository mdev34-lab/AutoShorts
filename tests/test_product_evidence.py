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


class _Page:
    url = "https://www.loja.example.org/p/hy300"
    title = "HY300 Smart Projector"
    h1 = "HY300 Smart Projector"
    meta_title = ""

    def __init__(self, html_text):
        self.html_text = html_text


def test_storefront_page_without_main_uses_meta_description_and_skips_junk():
    from autoshorts.modules.product.evidence import page_atoms

    html = (
        '<html><head><meta name="description" content="HY300 delivers 720P HD, '
        '290 ANSI brightness, WiFi 6, Bluetooth 5.4"></head><body>'
        "<nav><li>Frete 99W</li></nav>"
        "<div><p>Gap 7mm entre as peças</p><p>Alto-falante 1*5W</p></div>"
        "</body></html>"
    )
    surfaces = sorted(a.surface for a in page_atoms(_Page(html)))
    assert "290 ANSI de brilho" in surfaces and "resolução 720P" in surfaces
    # bare, attribute-less quantities are never narratable atoms
    assert not any("7mm" in s or "5W" in s or "99W" in s for s in surfaces)


def test_conflicting_numeric_claims_are_omitted_not_chosen():
    from autoshorts.modules.product.evidence import reconcile_atoms

    a = atoms_from_text("290 ANSI brightness, 720P HD, WiFi 6", "a.example.org")
    b = atoms_from_text("170 ANSI brightness, 720P HD, Bluetooth", "b.example.org")
    out = {x.surface for x in reconcile_atoms(a + b)}
    assert not any("ANSI" in s for s in out)  # 290 vs 170: neither is chosen
    assert "resolução 720P" in out and "Wi-Fi" in out and "Bluetooth" in out
    # source COUNT is not truth: an undisputed single-source number is kept
    assert any("290 ANSI" in x.surface for x in reconcile_atoms(a))


def test_same_model_listed_with_another_value_elsewhere_on_the_page_is_disputed():
    from autoshorts.modules.product.evidence import page_atoms

    class P(_Page):
        identity_codes = ["hy300proplus"]

    html = (
        '<html><head><meta name="description" content="HY300Pro+ delivers 720P HD, '
        '290 ANSI brightness, WiFi 6"></head><body>'
        "<nav>HY320 260ANSI 4K HY300Pro+ HOT 720P 200ANSI 4K Wifi6 HY300Pro 720P "
        "150ANSI</nav><div><p>Pronto</p></div></body></html>"
    )
    surfaces = {a.surface for a in page_atoms(P(html))}
    assert not any("ANSI" in s for s in surfaces)  # 290 vs 200 for the same model
    assert "resolução 720P" in surfaces  # no dispute on resolution
