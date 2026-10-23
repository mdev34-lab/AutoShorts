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


def test_hook_narrates_the_verified_name_verbatim():
    paragraphs = compose_showcase_script("Projetor HY300 Pro 4K Wi-Fi")
    assert paragraphs[0] == "Conheça Projetor HY300 Pro 4K Wi-Fi."


def test_attribute_binding_in_narration():
    paragraphs = compose_showcase_script("Notebook Y2 8GB RAM")
    assert any("8GB de RAM" in p for p in paragraphs)
    assert not any("armazenamento" in p for p in paragraphs)


def test_bare_spec_gets_no_invented_attribute():
    paragraphs = compose_showcase_script("Caixa de Som X200 20W Bluetooth")
    assert any("Traz 20W." in p for p in paragraphs)
    assert not any("potência" in p for p in paragraphs)


def test_model_code_narrated_as_written_in_name():
    paragraphs = compose_showcase_script("Projetor HY300 Pro 4K Wi-Fi")
    assert any("HY300 Pro" in p for p in paragraphs)
    assert not any("hy300pro" in p for p in paragraphs)


def test_no_facts_means_no_fact_claims():
    paragraphs = compose_showcase_script("Smartphone X100 Pro")
    joined = " ".join(paragraphs)
    for forbidden in ("bateria", "OLED", "garantia", "Anatel", "horas", "GB"):
        assert forbidden not in joined


def test_portuguese_negation_forms_never_narrated_as_features():
    for name in (
        "Projetor HY300 Pro não tem Bluetooth",
        "Projetor HY300 Pro nem Bluetooth",
        "Projetor HY300 Pro sem Bluetooth",
    ):
        joined = " ".join(compose_showcase_script(name)[1:])
        assert "Com Bluetooth" not in joined
