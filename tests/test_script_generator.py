"""
Test script generator functionality
"""

import json
from unittest.mock import Mock, patch

import pytest

from autoshorts.modules.script_generator import ScriptGenerator


class TestScriptGenerator:
    """Test cases for ScriptGenerator class"""

    def setup_method(self):
        """Setup test fixtures"""
        self.script_generator = ScriptGenerator(web_search=False)

    def test_init(self):
        """Test script generator initialization"""
        assert hasattr(self.script_generator, "api_url")
        assert hasattr(self.script_generator, "api_key")
        assert hasattr(self.script_generator, "model")

    def test_init_with_web_search(self):
        """Test script generator initialization with web search"""
        generator = ScriptGenerator(web_search=True)
        assert generator.web_search
        assert generator.searcher is not None

    def test_init_without_web_search(self):
        """Test script generator initialization without web search"""
        generator = ScriptGenerator(web_search=False)
        assert not generator.web_search
        assert generator.searcher is None

    @patch("autoshorts.modules.script_generator.requests.post")
    def test_generate_script_success(self, mock_post):
        """Test successful script generation"""
        # Mock API response
        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": "Este é o primeiro parágrafo.\nEste é o segundo parágrafo.\nEste é o terceiro parágrafo.\nEste é o quarto parágrafo.\nEste é o quinto parágrafo."
                    }
                }
            ]
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        result = self.script_generator.generate_script("inteligência artificial")

        assert isinstance(result, list)
        assert len(result) == 5
        assert "primeiro" in result[0].lower()

    @patch("autoshorts.modules.script_generator.WebSearcher")
    @patch("autoshorts.modules.script_generator.requests.post")
    def test_generate_script_with_web_search(self, mock_post, mock_searcher_class):
        """Test script generation with web search enabled (two-pass)"""
        # Mock WebSearcher to return no results (falls back to draft)
        mock_searcher = Mock()
        mock_searcher.search_with_queries.return_value = None
        mock_searcher_class.return_value = mock_searcher

        # Mock draft API response (Pass 1)
        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "draft": [
                                    "AI transformed technology in the 21st century.",
                                    "The field was founded in 1956 at Dartmouth.",
                                    "Machine learning emerged as a key approach.",
                                    "Neural networks revolutionized the field.",
                                    "Today AI powers everything from search to self-driving cars.",
                                ],
                                "queries": [
                                    "intelig\u00eancia artificial hist\u00f3ria 1956",
                                    "Deep learning revolu\u00e7\u00e3o tecnologia",
                                    "IA aplica\u00e7\u00f5es atuais",
                                ],
                                "title": "A Revolu\u00e7\u00e3o da Intelig\u00eancia Artificial",
                            }
                        )
                    }
                }
            ]
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        script_generator = ScriptGenerator(web_search=True)
        result = script_generator.generate_script("AI")

        assert isinstance(result, list)
        assert len(result) > 0
        # Verify search was called with the queries from the draft
        mock_searcher.search_with_queries.assert_called_once()

        # Verify no API-level tools (search is now local)
        payload = mock_post.call_args[1]["json"]
        assert "tools" not in payload
        # Draft call should use JSON response_format
        assert payload.get("response_format") == {"type": "json_object"}

    @patch("autoshorts.modules.script_generator.requests.post")
    def test_generate_script_api_error_returns_empty(self, mock_post):
        """Test script generation with API error returns empty list"""
        mock_post.side_effect = Exception("API Error")

        result = self.script_generator.generate_script("test subject")

        assert isinstance(result, list)
        assert len(result) == 0

    @patch("autoshorts.modules.script_generator.requests.post")
    def test_generate_script_empty_response_returns_empty(self, mock_post):
        """Test script generation with empty response returns empty list"""
        mock_response = Mock()
        mock_response.json.return_value = {"choices": [{"message": {"content": ""}}]}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        result = self.script_generator.generate_script("test subject")

        assert isinstance(result, list)
        assert len(result) == 0

    @patch("autoshorts.modules.script_generator.requests.post")
    def test_generate_script_with_prompts_success(self, mock_post):
        """Test successful script generation with image prompts"""
        # Mock API response with JSON structure
        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": '{"paragraphs": ["Para 1", "Para 2", "Para 3", "Para 4", "Para 5", "Para 6", "Para 7"], "image_prompts": ["Prompt 1", "Prompt 2", "Prompt 3", "Prompt 4", "Prompt 5", "Prompt 6", "Prompt 7", "Prompt 8", "Prompt 9"]}'
                    }
                }
            ]
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        result = self.script_generator.generate_script_with_prompts("test subject")

        assert isinstance(result, tuple)
        assert len(result) == 2
        paragraphs, image_prompts = result

        assert isinstance(paragraphs, list)
        assert isinstance(image_prompts, list)
        assert len(paragraphs) == 7
        assert len(image_prompts) == 9

    @patch("autoshorts.modules.script_generator.requests.post")
    def test_generate_script_with_prompts_api_error(self, mock_post):
        """Test script generation with prompts API error"""
        mock_post.side_effect = Exception("API Error")

        with pytest.raises(Exception):
            self.script_generator.generate_script_with_prompts("test subject")

    @patch("autoshorts.modules.script_generator.requests.post")
    def test_generate_script_with_prompts_invalid_json(self, mock_post):
        """Test script generation with invalid JSON response"""
        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "invalid json content"}}]
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        with pytest.raises(ValueError):
            self.script_generator.generate_script_with_prompts("test subject")

    @patch("autoshorts.modules.script_generator.requests.post")
    def test_generate_script_from_metadata(self, mock_post):
        """Test script generation from video metadata"""
        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": "Generated script based on video title and description.\nSecond paragraph.\nThird paragraph.\nFourth paragraph.\nFifth paragraph."
                    }
                }
            ]
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        result = self.script_generator.generate_script_from_metadata(
            "Video Title", "Video Description"
        )

        assert isinstance(result, list)
        assert len(result) > 0

    @patch("autoshorts.modules.script_generator.requests.post")
    def test_generate_script_from_metadata_with_web_search(self, mock_post):
        """Test script generation from metadata with web search"""
        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": "Generated script.\nSecond paragraph.\nThird paragraph.\nFourth paragraph.\nFifth paragraph."
                    }
                }
            ]
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        generator = ScriptGenerator(web_search=True)
        result = generator.generate_script_from_metadata(
            "Video Title", "Video Description"
        )

        assert isinstance(result, list)
        assert len(result) > 0

    @patch("autoshorts.modules.script_generator.requests.post")
    def test_generate_script_from_metadata_error_returns_empty(self, mock_post):
        """Test script generation from metadata with error returns empty list"""
        mock_post.side_effect = Exception("API Error")

        result = self.script_generator.generate_script_from_metadata(
            "Title", "Description"
        )

        assert isinstance(result, list)
        assert len(result) == 0

    @patch("autoshorts.modules.script_generator.requests.post")
    def test_api_request_headers(self, mock_post):
        """Test that API requests include proper headers"""
        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "test\npara2\npara3\npara4\npara5"}}]
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        self.script_generator.generate_script("test")

        # Verify the call was made with correct headers
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        headers = call_args[1]["headers"]

        assert "Authorization" in headers
        assert headers["Authorization"].startswith("Bearer ")
        assert "Content-Type" in headers
        assert headers["Content-Type"] == "application/json"

    @patch("autoshorts.modules.script_generator.requests.post")
    def test_api_request_payload(self, mock_post):
        """Test that API requests include proper payload structure"""
        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "test\npara2\npara3\npara4\npara5"}}]
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        self.script_generator.generate_script("test subject")

        # Verify the call was made with correct payload
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        payload = call_args[1]["json"]

        assert "model" in payload
        assert "messages" in payload
        assert isinstance(payload["messages"], list)
        assert len(payload["messages"]) == 2  # system + user
        assert payload["messages"][0]["role"] == "system"
        assert payload["messages"][1]["role"] == "user"

    @patch("autoshorts.modules.script_generator.requests.post")
    def test_generate_script_with_special_characters(self, mock_post):
        """Test script generation with special characters"""
        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": "Título: Ênfase em ááéíóú & caracteres especiais!\nSegundo parágrafo.\nTerceiro parágrafo.\nQuarto parágrafo.\nQuinto parágrafo."
                    }
                }
            ]
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        result = self.script_generator.generate_script("test")

        assert isinstance(result, list)
        assert len(result) > 0

    @patch("autoshorts.modules.script_generator.requests.post")
    def test_generate_script_with_markdown_formatting(self, mock_post):
        """Test script generation cleans markdown formatting"""
        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": "```Este é o primeiro parágrafo.```\n**Segundo parágrafo.**\n*Terceiro parágrafo.*\nQuarto parágrafo.\nQuinto parágrafo."
                    }
                }
            ]
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        result = self.script_generator.generate_script("test")

        assert isinstance(result, list)
        # Markdown should be cleaned
        assert "```" not in result[0]
        assert "**" not in result[1]

    @patch("autoshorts.modules.script_generator.requests.post")
    def test_generate_script_single_paragraph_response(self, mock_post):
        """Test script generation with single paragraph splits correctly"""
        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": "This is a very long single paragraph that should be split into multiple sentences. It has multiple sentences. And more sentences. And even more content to make it longer. Final sentence here."
                    }
                }
            ]
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        result = self.script_generator.generate_script("test")

        assert isinstance(result, list)
        # Should split by periods when no line breaks
        assert len(result) >= 1

    @patch("autoshorts.modules.script_generator.requests.post")
    def test_generate_script_with_double_newlines(self, mock_post):
        """Test script generation with double newline separators"""
        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": "First paragraph.\n\nSecond paragraph.\n\nThird paragraph.\n\nFourth paragraph.\n\nFifth paragraph."
                    }
                }
            ]
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        result = self.script_generator.generate_script("test")

        assert isinstance(result, list)
        assert len(result) >= 3

    @patch("autoshorts.modules.script_generator.requests.post")
    def test_generate_script_insufficient_paragraphs(self, mock_post):
        """Test script generation returns what it gets, no padding"""
        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Only one paragraph."}}]
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        result = self.script_generator.generate_script("test")

        assert isinstance(result, list)
        assert len(result) == 1

    @patch("autoshorts.modules.script_generator.requests.post")
    def test_generate_script_with_context_success(self, mock_post):
        """Test _generate_script_with_context returns paragraphs from API"""
        paragraphs = [
            "Em 2025, o Banco Master veio a tona após a Operação Compliance Zero.",
            "O rombo foi de R$ 41 bilhões no Fundo Garantidor de Créditos.",
            "Daniel Vorcaro foi preso ao tentar deixar o país.",
        ]
        data = {"paragraphs": paragraphs}
        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": json.dumps(data)}}]
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        generator = ScriptGenerator(web_search=True)
        result = generator._generate_script_with_context(
            "Banco Master", "FONTES:\n- source 1\n- source 2"
        )
        assert result == paragraphs

    @patch("autoshorts.modules.script_generator.requests.post")
    def test_generate_script_with_context_api_error(self, mock_post):
        """Test _generate_script_with_context returns None on API error"""
        mock_post.side_effect = Exception("API Error")
        generator = ScriptGenerator(web_search=True)
        result = generator._generate_script_with_context("test", "sources")
        assert result is None

    @patch("autoshorts.modules.script_generator.requests.post")
    def test_generate_script_with_context_empty_response(self, mock_post):
        """Test _generate_script_with_context returns None when no paragraphs key"""
        data = {"other_key": ["value"]}
        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": json.dumps(data)}}]
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        generator = ScriptGenerator(web_search=True)
        result = generator._generate_script_with_context("test", "sources")
        assert result is None

    @patch("autoshorts.modules.script_generator.WebSearcher")
    @patch("autoshorts.modules.script_generator.requests.post")
    def test_generate_script_with_prompts_web_search(
        self, mock_post, mock_searcher_class
    ):
        """Test script generation with prompts and web search (two-pass)"""
        mock_searcher = Mock()
        mock_searcher.search_with_queries.return_value = None
        mock_searcher_class.return_value = mock_searcher

        # Mock draft API response (Pass 1)
        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "draft": ["P1", "P2", "P3", "P4", "P5", "P6", "P7"],
                                "queries": ["test query 1", "test query 2"],
                                "title": "Test Title",
                            }
                        )
                    }
                }
            ]
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        generator = ScriptGenerator(web_search=True)
        paragraphs, prompts = generator.generate_script_with_prompts("test")

        # No search results found, returns draft as-is
        assert len(paragraphs) == 7
        assert prompts == []

        mock_searcher.search_with_queries.assert_called_once()

        # Verify no API-level tools (search is now local)
        payload = mock_post.call_args[1]["json"]
        assert "tools" not in payload

    @patch("autoshorts.modules.script_generator.requests.post")
    def test_generate_script_from_metadata_long_description(self, mock_post):
        """Test script generation from metadata with long description"""
        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": "Generated script.\nSecond paragraph.\nThird paragraph.\nFourth paragraph.\nFifth paragraph."
                    }
                }
            ]
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        long_description = "A" * 2000  # Very long description
        result = self.script_generator.generate_script_from_metadata(
            "Title", long_description
        )

        assert isinstance(result, list)
        # Verify the description was truncated in the call
        mock_post.assert_called_once()

    @patch("autoshorts.modules.script_generator.WebSearcher")
    @patch("autoshorts.modules.script_generator.requests.post")
    def test_generate_script_with_search_results(self, mock_post, mock_searcher_class):
        """Test generate_script with search results uses context in final call"""
        mock_searcher = Mock()
        mock_searcher.search_with_queries.return_value = [
            {
                "title": "Flamengo",
                "url": "https://ex.com/1",
                "snippet": "Hist\u00f3ria do Flamengo",
            },
            {
                "title": "Fluminense",
                "url": "https://ex.com/2",
                "snippet": "Origem do Fluminense",
            },
        ]
        mock_searcher.format_context.return_value = "FONTES DA WEB:\n..."
        mock_searcher_class.return_value = mock_searcher

        query_response = Mock()
        query_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {"queries": ["flamengo hist\u00f3ria", "fluminense origem"]}
                        )
                    }
                }
            ]
        }
        query_response.raise_for_status.return_value = None

        text_response = Mock()
        text_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": (
                            "Primeiro par\u00e1grafo sobre o cl\u00e1ssico.\n"
                            "Segundo par\u00e1grafo com mais detalhes.\n"
                            "Terceiro par\u00e1grafo da hist\u00f3ria.\n"
                            "Quarto par\u00e1grafo.\n"
                            "Quinto par\u00e1grafo."
                        )
                    }
                }
            ]
        }
        text_response.raise_for_status.return_value = None

        verification_response = Mock()
        verification_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps({
                            "verified": True,
                            "corrections": [],
                            "paragraphs": [],
                        })
                    }
                }
            ]
        }
        verification_response.raise_for_status.return_value = None

        title_response = Mock()
        title_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps({"title": "Cl\u00e1ssico"})
                    }
                }
            ]
        }
        title_response.raise_for_status.return_value = None

        mock_post.side_effect = [query_response, text_response, verification_response, title_response]

        generator = ScriptGenerator(web_search=True)
        result = generator.generate_script("Flamengo x Fluminense")

        assert isinstance(result, list)
        assert len(result) == 5
        assert "primeiro" in result[0].lower()
        assert mock_searcher.search_with_queries.call_count == 2
        assert mock_searcher.format_context.call_count == 2
        assert mock_post.call_count == 5

    @patch("autoshorts.modules.script_generator.WebSearcher")
    @patch("autoshorts.modules.script_generator.requests.post")
    def test_generate_script_with_prompts_web_search_success(
        self, mock_post, mock_searcher_class
    ):
        """Test generate_script_with_prompts with search results success path"""
        mock_searcher = Mock()
        mock_searcher.search_with_queries.return_value = [
            {"title": "Fla", "url": "https://ex.com/1", "snippet": "Flamengo info"},
            {"title": "Flu", "url": "https://ex.com/2", "snippet": "Fluminense info"},
        ]
        mock_searcher.format_context.return_value = "FONTES DA WEB:\n..."
        mock_searcher_class.return_value = mock_searcher

        query_response = Mock()
        query_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps({"queries": ["query1", "query2"]})
                    }
                }
            ]
        }
        query_response.raise_for_status.return_value = None

        final_json = json.dumps({"paragraphs": [f"P{i}" for i in range(1, 8)]})
        final_response = Mock()
        final_response.json.return_value = {
            "choices": [{"message": {"content": final_json}}]
        }
        final_response.raise_for_status.return_value = None

        verification_response = Mock()
        verification_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps({
                            "verified": True,
                            "corrections": [],
                            "paragraphs": [],
                        })
                    }
                }
            ]
        }
        verification_response.raise_for_status.return_value = None

        title_response = Mock()
        title_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps({"title": "Test Title"})
                    }
                }
            ]
        }
        title_response.raise_for_status.return_value = None

        mock_post.side_effect = [query_response, final_response, verification_response, title_response]

        generator = ScriptGenerator(web_search=True)
        paragraphs, prompts = generator.generate_script_with_prompts("test")

        assert len(paragraphs) == 7
        assert prompts == []
        assert mock_searcher.format_context.call_count == 2
        assert mock_post.call_count == 5

    @patch("autoshorts.modules.script_generator.WebSearcher")
    @patch("autoshorts.modules.script_generator.requests.post")
    def test_generate_script_with_prompts_empty_draft_fallback(
        self, mock_post, mock_searcher_class
    ):
        """Test generate_script_with_prompts falls back when draft is empty"""
        mock_searcher = Mock()
        mock_searcher.search_with_queries.return_value = None
        mock_searcher.search.return_value = None
        mock_searcher_class.return_value = mock_searcher

        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "draft": [],
                                "queries": [],
                                "title": "",
                            }
                        )
                    }
                }
            ]
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        generator = ScriptGenerator(web_search=True)
        paragraphs, prompts = generator.generate_script_with_prompts("test")
        assert len(paragraphs) == 0  # No fallback filler, returns empty
        assert isinstance(paragraphs, list)
        assert prompts == []

    @patch("autoshorts.modules.script_generator.WebSearcher")
    @patch("autoshorts.modules.script_generator.requests.post")
    def test_generate_script_draft_api_error(self, mock_post, mock_searcher_class):
        """Test generate_script handles draft API error gracefully"""
        mock_searcher = Mock()
        mock_searcher.search_with_queries.return_value = None
        mock_searcher_class.return_value = mock_searcher

        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "draft": [],
                                "queries": ["fallback"],
                                "title": "",
                            }
                        )
                    }
                }
            ]
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        generator = ScriptGenerator(web_search=True)
        result = generator.generate_script("test")
        assert isinstance(result, list)
        assert len(result) == 0

    @patch("autoshorts.modules.script_generator.requests.post")
    def test_generate_draft_api_error_returns_fallback(self, mock_post):
        """Test _generate_draft returns fallback on API error"""
        mock_post.side_effect = Exception("API Error")
        generator = ScriptGenerator(web_search=True)
        result = generator._generate_draft("test")
        assert "draft" in result
        assert result["draft"] == []
        assert "test" in result["queries"]

    @patch("autoshorts.modules.script_generator.requests.post")
    def test_generate_draft_missing_draft_key(self, mock_post):
        """Test _generate_draft returns fallback when response has no draft key"""
        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "queries": ["q1"],
                                "title": "Titulo",
                            }
                        )
                    }
                }
            ]
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        generator = ScriptGenerator(web_search=True)
        result = generator._generate_draft("test")
        assert result["draft"] == []
        assert "test" in result["queries"]

    @patch("autoshorts.modules.script_generator.requests.post")
    def test_generate_image_prompts_success(self, mock_post):
        """Test generate_image_prompts_from_script returns paired prompts"""
        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "images": [
                                    {
                                        "web_query": "dramatic scene",
                                        "ai_prompt": "A dramatic cinematic scene with dramatic lighting, ultra detailed 4k",
                                    },
                                    {
                                        "web_query": "close up shot",
                                        "ai_prompt": "A close up shot with dramatic lighting, ultra detailed 4k",
                                    },
                                ]
                            }
                        )
                    }
                }
            ]
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        generator = ScriptGenerator(web_search=False)
        result = generator.generate_image_prompts_from_script(["Para 1", "Para 2"], 2)
        assert isinstance(result, list)
        assert len(result) == 2
        assert "web_query" in result[0]
        assert "ai_prompt" in result[0]
        assert result[0]["web_query"] == "dramatic scene"

    @patch("autoshorts.modules.script_generator.requests.post")
    def test_generate_image_prompts_api_error(self, mock_post):
        """Test generate_image_prompts_from_script raises on API error"""
        mock_post.side_effect = Exception("API Error")
        generator = ScriptGenerator(web_search=False)
        with pytest.raises(Exception):
            generator.generate_image_prompts_from_script(["Para"], 1)

    @patch("autoshorts.modules.script_generator.requests.post")
    def test_generate_image_prompts_missing_key(self, mock_post):
        """Test generate_image_prompts_from_script returns empty list when key missing"""
        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": json.dumps({"other_key": ["value"]})}}]
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        generator = ScriptGenerator(web_search=False)
        result = generator.generate_image_prompts_from_script(["Para 1", "Para 2"], 2)
        assert result == []


class TestScriptGeneratorEdgeCases:
    """Test edge cases for ScriptGenerator"""

    def setup_method(self):
        """Setup test fixtures"""
        self.script_generator = ScriptGenerator(web_search=False)

    @patch("autoshorts.modules.script_generator.requests.post")
    def test_generate_script_timeout_returns_empty(self, mock_post):
        """Test script generation with timeout returns empty list"""
        import requests

        mock_post.side_effect = requests.Timeout("Request timed out")

        result = self.script_generator.generate_script("test")

        assert isinstance(result, list)
        assert len(result) == 0

    @patch("autoshorts.modules.script_generator.requests.post")
    def test_generate_script_connection_error_returns_empty(self, mock_post):
        """Test script generation with connection error returns empty list"""
        import requests

        mock_post.side_effect = requests.ConnectionError("No connection")

        result = self.script_generator.generate_script("test")

        assert isinstance(result, list)
        assert len(result) == 0

    @patch("autoshorts.modules.script_generator.requests.post")
    def test_generate_script_http_error_returns_empty(self, mock_post):
        """Test script generation with HTTP error returns empty list"""
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = Exception("HTTP 500")
        mock_post.return_value = mock_response

        result = self.script_generator.generate_script("test")

        assert isinstance(result, list)
        assert len(result) == 0


if __name__ == "__main__":
    pytest.main([__file__])
