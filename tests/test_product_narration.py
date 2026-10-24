"""The composed script may only narrate facts verified in the product name."""

from autoshorts.modules.product.narration import compose_showcase_script
from autoshorts.modules.product.script_guard import assert_script_within_verified_facts

NAMES = [
    "MAGCUBIC HY300ProPlus Projector 290 ANSI",
    "Notebook Y2 8GB RAM",
    "Caixa de Som X200 20W Bluetooth",
    "Projetor HY300 Pro 4K Wi-Fi",
    "Smartphone X100 Pro",
    "Fone JBL Flip 6 Bluetooth 30W",
]


def test_composed_script_passes_its_own_guard():
    for name in NAMES:
        paragraphs = compose_showcase_script(name)
        assert len(paragraphs) >= 3
        assert_script_within_verified_facts(paragraphs, name)


def test_hook_narrates_only_the_model_identifier():
    paragraphs = compose_showcase_script("Projetor HY300 Pro 4K Wi-Fi")
    assert paragraphs[0] == "Conheça o HY300 Pro."
    assert "4K" not in " ".join(paragraphs)


def test_negated_slug_claim_is_never_inverted():
    joined = " ".join(compose_showcase_script("Projetor HY300 Pro sem Bluetooth"))
    assert "Bluetooth" not in joined


def test_unverified_slug_claims_are_not_repeated():
    name = (
        "Projetor HY300 Pro aluminio aeronautico Fabricado no Japao "
        "certificado CE dois terabytes bateria o dia todo"
    )
    joined = " ".join(compose_showcase_script(name)).lower()
    for claim in ("alumin", "japao", "certificado", "terabyte", "bateria", "dia todo"):
        assert claim not in joined


def test_multi_spec_name_does_not_abort_or_cross_bind():
    name = "Notebook Y2 8GB de RAM e 256GB de armazenamento"
    paragraphs = compose_showcase_script(name)
    assert_script_within_verified_facts(paragraphs, name)
    assert "GB" not in " ".join(paragraphs)


def test_model_code_narrated_as_written_in_name():
    paragraphs = compose_showcase_script("Projetor HY300 Pro 4K Wi-Fi")
    assert any("HY300 Pro" in p for p in paragraphs)
    assert not any("hy300pro" in p for p in paragraphs)


def test_identifier_only_script_has_no_fact_claims():
    paragraphs = compose_showcase_script("Smartphone X100 Pro")
    joined = " ".join(paragraphs)
    for forbidden in ("bateria", "OLED", "garantia", "Anatel", "horas", "GB"):
        assert forbidden not in joined
