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


def test_spelled_out_quantities_fail_closed():
    for claim in (
        "Com dois terabytes de armazenamento.",
        "With two Terabytes of storage.",
        "São cinco horas de bateria.",
    ):
        with pytest.raises(ProductPipelineError):
            assert_script_within_verified_facts([claim], "Smartphone X100 Pro")


def test_display_tech_and_battery_claims_fail_closed():
    for claim in (
        "Tela OLED de tirar o fôlego.",
        "Long battery life for days.",
        "Bateria que dura o dia todo.",
    ):
        with pytest.raises(ProductPipelineError):
            assert_script_within_verified_facts([claim], "Smartphone X100 Pro")


def test_empty_product_name_fails_closed():
    with pytest.raises(ProductPipelineError, match="No verified product name"):
        assert_script_within_verified_facts(["Qualquer coisa."], "   ")


def test_unit_aliases_ground_equivalent_claims():
    # "10h" in the name grounds "10 horas"; "20 Watts" grounds "20W".
    assert_script_within_verified_facts(
        ["Bateria de 10 horas de autonomia."], "Notebook Y2 10h Bateria"
    )
    assert_script_within_verified_facts(
        ["São 20W de potência."], "Caixa X200 20 Watts Bluetooth"
    )
    assert_script_within_verified_facts(
        ["Tela OLED linda."], "Smart TV Z9 OLED 55 Polegadas"
    )


def test_ungroundable_claim_classes_always_rejected():
    # A product slug can never verify certifications, warranty or grade claims.
    for claim in (
        "Certificado pela Anatel e Inmetro.",
        "Garantia de cinco anos.",
        "Produto com garantia vitalícia.",
        "Lifetime warranty included.",
        "Estrutura de alumínio aeroespacial.",
    ):
        with pytest.raises(ProductPipelineError):
            assert_script_within_verified_facts([claim], "Drone D10 Pro")


def test_attribute_binding_blocks_cross_attribute_reuse():
    # "8GB RAM" in the name must not ground "8GB de armazenamento".
    with pytest.raises(ProductPipelineError, match="storage"):
        assert_script_within_verified_facts(
            ["São 8GB de armazenamento interno."], "Notebook Y2 8GB RAM"
        )
    # "suporta 2kg" must not ground "pesa 2kg".
    with pytest.raises(ProductPipelineError, match="weight"):
        assert_script_within_verified_facts(
            ["Pesa apenas 2kg."], "Suporte de TV Suporta 2kg"
        )
    # Same attribute binds fine.
    assert_script_within_verified_facts(
        ["São 8GB de RAM pra multitarefa."], "Notebook Y2 8GB RAM"
    )
    assert_script_within_verified_facts(
        ["Suporta até 2kg com segurança."], "Suporte de TV Suporta 2kg"
    )
