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
        self.script_generator = ScriptGenerator()

    def test_init(self):
        """Test script generator initialization"""
        assert hasattr(self.script_generator, "api_url")
        assert hasattr(self.script_generator, "api_key")
        assert hasattr(self.script_generator, "model")

    def test_init_with_web_search(self):
        """Test script generator initialization with web search"""
        generator = ScriptGenerator(web_search=True)
        assert generator.web_search
        assert generator.model == "gemini-fast"
        assert generator.tools is not None

    def test_init_without_web_search(self):
        """Test script generator initialization without web search"""
        generator = ScriptGenerator(web_search=False)
        assert not generator.web_search
        assert generator.tools is None

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

    @patch("autoshorts.modules.script_generator.requests.post")
    def test_generate_script_with_web_search(self, mock_post):
        """Test script generation with web search enabled"""
        # Mock API response
        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": "Web search result about artificial intelligence.\nSecond paragraph here.\nThird paragraph.\nFourth paragraph.\nFifth paragraph."
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

        # Verify tools were included in the request
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        payload = call_args[1]["json"]
        assert "tools" in payload

    @patch("autoshorts.modules.script_generator.requests.post")
    def test_generate_script_api_error_fallback(self, mock_post):
        """Test script generation with API error returns fallback"""
        mock_post.side_effect = Exception("API Error")

        # Should return fallback script instead of raising
        result = self.script_generator.generate_script("test subject")

        assert isinstance(result, list)
        assert len(result) == 5
        assert "vídeo interessante" in result[0].lower()

    @patch("autoshorts.modules.script_generator.requests.post")
    def test_generate_script_empty_response(self, mock_post):
        """Test script generation with empty response"""
        mock_response = Mock()
        mock_response.json.return_value = {"choices": [{"message": {"content": ""}}]}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        result = self.script_generator.generate_script("test subject")

        assert isinstance(result, list)
        assert len(result) == 5  # Should return fallback

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

        with pytest.raises(json.JSONDecodeError):
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
    def test_generate_script_from_metadata_error(self, mock_post):
        """Test script generation from metadata with error returns fallback"""
        mock_post.side_effect = Exception("API Error")

        result = self.script_generator.generate_script_from_metadata(
            "Title", "Description"
        )

        assert isinstance(result, list)
        assert len(result) == 5

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
        """Test script generation pads insufficient paragraphs"""
        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Only one paragraph."}}]
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        result = self.script_generator.generate_script("test")

        assert isinstance(result, list)
        assert len(result) == 5  # Should be padded to 5

    @patch("autoshorts.modules.script_generator.requests.post")
    def test_generate_script_with_prompts_web_search(self, mock_post):
        """Test script generation with prompts and web search"""
        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": '{"paragraphs": ["P1", "P2", "P3", "P4", "P5", "P6", "P7"], "image_prompts": ["I1", "I2", "I3", "I4", "I5", "I6", "I7", "I8", "I9"]}'
                    }
                }
            ]
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        generator = ScriptGenerator(web_search=True)
        paragraphs, prompts = generator.generate_script_with_prompts("test")

        # Verify tools were included
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        payload = call_args[1]["json"]
        assert "tools" in payload

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


class TestScriptGeneratorEdgeCases:
    """Test edge cases for ScriptGenerator"""

    def setup_method(self):
        """Setup test fixtures"""
        self.script_generator = ScriptGenerator()

    @patch("autoshorts.modules.script_generator.requests.post")
    def test_generate_script_timeout(self, mock_post):
        """Test script generation with timeout returns fallback"""
        import requests

        mock_post.side_effect = requests.Timeout("Request timed out")

        result = self.script_generator.generate_script("test")

        assert isinstance(result, list)
        assert len(result) == 5

    @patch("autoshorts.modules.script_generator.requests.post")
    def test_generate_script_connection_error(self, mock_post):
        """Test script generation with connection error returns fallback"""
        import requests

        mock_post.side_effect = requests.ConnectionError("No connection")

        result = self.script_generator.generate_script("test")

        assert isinstance(result, list)
        assert len(result) == 5

    @patch("autoshorts.modules.script_generator.requests.post")
    def test_generate_script_http_error(self, mock_post):
        """Test script generation with HTTP error returns fallback"""
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = Exception("HTTP 500")
        mock_post.return_value = mock_response

        result = self.script_generator.generate_script("test")

        assert isinstance(result, list)
        assert len(result) == 5


if __name__ == "__main__":
    pytest.main([__file__])
