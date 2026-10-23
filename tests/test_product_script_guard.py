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
