import pytest

from autoshorts.modules.product.errors import ProductPipelineError
from autoshorts.modules.product.script_guard import assert_script_within_verified_facts

NAME = "MAGCUBIC HY300ProPlus Projector 290 ANSI"


def test_grounded_script_passes():
    paragraphs = [
        "Conheça o Magcubic HY300ProPlus, projetor com 290 ANSI de brilho.",
        "Design compacto pra levar pra qualquer lugar.",
        "Perfeito pra sessão de cinema em casa.",
    ]
    assert_script_within_verified_facts(paragraphs, NAME)


def test_fabricated_spec_fails_closed():
    paragraphs = [
        "Conheça o Magcubic HY300ProPlus.",
        "São 128GB de armazenamento interno pra todos os seus apps.",
        "Compre já o seu.",
    ]
    with pytest.raises(ProductPipelineError, match="128"):
        assert_script_within_verified_facts(paragraphs, NAME)


def test_price_mention_fails_closed():
    paragraphs = [
        "Conheça o Magcubic HY300ProPlus.",
        "Por apenas R$ 499 você leva.",
        "Aproveite essa oferta.",
    ]
    with pytest.raises(ProductPipelineError, match="price/discount"):
        assert_script_within_verified_facts(paragraphs, NAME)


def test_discount_mention_fails_closed():
    paragraphs = [
        "Conheça o Magcubic HY300ProPlus.",
        "Com 50% de desconto essa semana.",
        "Não perca.",
    ]
    with pytest.raises(ProductPipelineError):
        assert_script_within_verified_facts(paragraphs, NAME)


def test_model_code_digits_do_not_ground_specs():
    # The 300 in "HY300" is part of the model code; it grounds no unit claim.
    for claim in (
        "Com bateria de 300mAh de longa duração.",
        "Suporta até 300kg com facilidade.",
    ):
        with pytest.raises(ProductPipelineError, match="300"):
            assert_script_within_verified_facts([claim], "Projetor HY300 Pro Portátil")


def test_grounded_unit_claim_still_passes():
    paragraphs = ["São 20W de potência com Bluetooth pra ouvir em qualquer cômodo."]
    assert_script_within_verified_facts(paragraphs, "Caixa de Som X200 20W Bluetooth")


def test_unverified_resolution_and_features_fail_closed():
    with pytest.raises(ProductPipelineError):
        assert_script_within_verified_facts(
            ["Resolução 4K e Bluetooth integrado."], "Projetor HY300 Pro Portátil"
        )
    with pytest.raises(ProductPipelineError):
        assert_script_within_verified_facts(
            ["Wi-Fi de última geração."], "Projetor HY300 Pro Portátil"
        )


def test_grounded_resolution_and_features_pass():
    paragraphs = ["Resolução 4K com Wi-Fi integrado pra espelhar a tela."]
    assert_script_within_verified_facts(paragraphs, "Projetor HY300 Pro 4K Wi-Fi")
