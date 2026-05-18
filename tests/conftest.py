"""
Test configuration and fixtures for pytest
"""

import shutil
import sys
import tempfile
from pathlib import Path

import pytest

# Add src to path for all tests
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture(autouse=True)
def isolate_image_cache():
    """Patch IMAGE_CACHE_DIR to a unique temp dir per test to prevent cross-test pollution."""
    from unittest.mock import patch
    cache_dir = Path(tempfile.mkdtemp())
    with patch("autoshorts.generators.explainer.IMAGE_CACHE_DIR", cache_dir):
        yield
    if cache_dir.exists():
        shutil.rmtree(cache_dir)


@pytest.fixture(autouse=True)
def prevent_web_search_network():
    """Prevent WebSearcher from making real DDGS network calls in tests."""
    from unittest.mock import patch
    with patch("ddgs.DDGS") as mock_ddgs:
        mock_instance = mock_ddgs.return_value
        mock_instance.text.return_value = []
        yield


@pytest.fixture
def temp_dir():
    """Provide a temporary directory for tests"""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    if temp_path.exists():
        shutil.rmtree(temp_path)


@pytest.fixture
def sample_paragraphs():
    """Provide sample paragraphs for testing"""
    return [
        "Este é o primeiro parágrafo de teste.",
        "Este é o segundo parágrafo de teste.",
        "Este é o terceiro parágrafo de teste.",
    ]


@pytest.fixture
def sample_prompts():
    """Provide sample image prompts for testing"""
    return [
        "A beautiful landscape with mountains",
        "A futuristic city skyline",
        "A serene beach at sunset",
    ]


@pytest.fixture
def mock_api_response():
    """Provide a mock API response for testing"""
    return {
        "choices": [
            {
                "message": {
                    "content": "Este é o primeiro parágrafo.\nEste é o segundo parágrafo.\nEste é o terceiro parágrafo."
                }
            }
        ]
    }


@pytest.fixture
def mock_vtt_content():
    """Provide sample VTT content for testing"""
    return """WEBVTT

00:00:00.000 --> 00:00:02.000
Primeiro subtítulo

00:00:02.000 --> 00:00:04.000
Segundo subtítulo

00:00:04.000 --> 00:00:06.000
Terceiro subtítulo
"""
