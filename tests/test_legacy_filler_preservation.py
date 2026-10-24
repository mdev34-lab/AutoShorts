"""Style heuristics must not delete narration facts (#23)."""

from unittest.mock import Mock, patch

import pytest

from autoshorts.modules.script_generator import ScriptGenerator

FACT_PARAGRAPHS = [
    "Na entrevista, o treinador disse que fica uma lição: defender em bloco.",
    "O museu que vale a pena conhecer guarda o troféu original.",
    "O capitão chamou a derrota de lição que vale para a temporada.",
    "Até a publicação do documento, ninguém sabia quem havia assinado.",
    "O segredo do título foi a defesa sólida, com 12 gols sofridos em 38 jogos.",
    "O livro A Verdade Escondida recebeu o prêmio literário.",
    "O single Você Não Vai Acreditar encerrou o show.",
    "Poucos conhecem o nome original do estádio, registrado na escritura.",
    "Pouca gente sabe que a final foi decidida nos pênaltis.",
    "Muita gente não sabe que o goleiro também foi capitão.",
    "O que poucos sabem é que a equipe jogou a final sem reservas.",
    "O árbitro explicou a decisão neste vídeo publicado pela federação.",
    "Vamos Explorar é o título do programa premiado pela emissora.",
    "A empresa Exemplo Ltda foi fundada em 1998.",
    "A companhia Exemplo S.A. anunciou o fechamento da fábrica.",
    "O banco foi autorizado como instituição de pagamento pelo regulador.",
    "A sentença identificou a associação como pessoa jurídica responsável.",
]


@pytest.mark.parametrize("paragraph", FACT_PARAGRAPHS)
def test_legacy_substring_never_drops_a_fact(paragraph):
    assert ScriptGenerator._is_filler(paragraph)  # proves the old rule matches
    assert ScriptGenerator._validate_paragraphs([paragraph]) == [paragraph]


def test_fact_retention_does_not_depend_on_digits_or_attribution():
    paragraphs = FACT_PARAGRAPHS[7:10]
    assert ScriptGenerator._ensure_paragraph_count(
        ScriptGenerator._validate_paragraphs(paragraphs), 5
    ) == paragraphs


def test_validation_still_strips_markdown_and_ignores_non_strings():
    assert ScriptGenerator._validate_paragraphs(
        [None, 42, "**O segredo** foi defender bem."]
    ) == ["O segredo foi defender bem."]


def test_style_warning_does_not_echo_private_narration():
    with patch("autoshorts.modules.script_generator.log") as log:
        result = ScriptGenerator._validate_paragraphs(FACT_PARAGRAPHS)
    assert result == FACT_PARAGRAPHS
    assert any("legacy filler" in call.args[0] for call in log.call_args_list)
    assert all("Exemplo" not in call.args[0] for call in log.call_args_list)


def test_generate_script_keeps_five_paragraphs_without_repair():
    gen = ScriptGenerator(web_search=False)
    gen.web_search = True
    gen.searcher = Mock()
    gen.searcher.search_with_queries.return_value = [{"text": "source"}]
    paragraphs = FACT_PARAGRAPHS[0:5]
    with (
        patch.object(gen, "_generate_search_queries", return_value=["query"]),
        patch.object(gen, "_make_text_api_call", return_value=paragraphs),
        patch.object(gen, "_verify_factual_claims", side_effect=lambda ps, _: ps),
        patch.object(gen, "_generate_title_from_script", return_value="title"),
        patch.object(gen, "_repair_paragraphs") as repair,
    ):
        assert gen.generate_script("a temporada") == paragraphs
    repair.assert_not_called()
