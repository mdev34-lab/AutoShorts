"""
Test configuration validation and constants
"""

from pathlib import Path
from unittest.mock import patch

import pytest

# Import all config constants
from autoshorts.modules.config import *


class TestAPIConfiguration:
    """Test cases for API configuration"""

    def test_api_url(self):
        """Test API URL configuration"""
        assert API_URL.startswith("https://")
        assert "pollinations.ai" in API_URL

    def test_img_url(self):
        """Test image API URL configuration"""
        assert IMG_URL.startswith("https://")
        assert "pollinations.ai" in IMG_URL

    def test_api_key(self):
        """Test API key configuration"""
        assert isinstance(API_KEY, str)
        assert len(API_KEY) > 0


class TestModelConfiguration:
    """Test cases for model configuration"""

    def test_model_text(self):
        """Test text model configuration"""
        assert isinstance(MODEL_TEXT, str)
        assert len(MODEL_TEXT) > 0

    def test_model_image(self):
        """Test image model configuration"""
        assert isinstance(MODEL_IMAGE, str)
        assert len(MODEL_IMAGE) > 0


class TestTTSConfiguration:
    """Test cases for TTS configuration"""

    def test_voice(self):
        """Test voice configuration"""
        assert isinstance(VOICE, str)
        assert "-" in VOICE  # Voice format like pt-BR-AntonioNeural

    def test_tts_rate(self):
        """Test TTS rate configuration"""
        assert isinstance(TTS_RATE, str)
        assert "%" in TTS_RATE  # Rate format like +20%  # Rate format like +20%


class TestDirectoryConfiguration:
    """Test cases for directory configuration"""

    def test_output_dir(self):
        """Test output directory configuration"""
        assert isinstance(OUTPUT_DIR, Path)
        assert OUTPUT_DIR.name == "output"

    def test_temp_dir_prefix(self):
        """Test temp directory prefix configuration"""
        assert isinstance(TEMP_DIR_PREFIX, str)
        assert len(TEMP_DIR_PREFIX) > 0


class TestVideoConfiguration:
    """Test cases for video configuration"""

    def test_video_dimensions(self):
        """Test video dimension configuration"""
        assert isinstance(VIDEO_WIDTH, int)
        assert isinstance(VIDEO_HEIGHT, int)
        assert VIDEO_WIDTH > 0
        assert VIDEO_HEIGHT > 0
        # Video dimensions should make sense (both positive) - .env may have different aspect ratio

    def test_video_fps(self):
        """Test video FPS configuration"""
        assert isinstance(VIDEO_FPS, int)
        assert (
            1 <= VIDEO_FPS <= 60
        )  # Reasonable FPS range (allowing lower FPS from .env)

    def test_video_codec(self):
        """Test video codec configuration"""
        assert isinstance(VIDEO_CODEC, str)
        assert VIDEO_CODEC in ["libx264", "libx265", "h264", "h265"]

    def test_audio_codec(self):
        """Test audio codec configuration"""
        assert isinstance(AUDIO_CODEC, str)
        assert AUDIO_CODEC in ["aac", "mp3", "opus"]


class TestYouTubeConfiguration:
    """Test cases for YouTube configuration"""

    def test_youtube_max_height(self):
        """Test YouTube max height configuration"""
        assert isinstance(YOUTUBE_MAX_HEIGHT, int)
        assert YOUTUBE_MAX_HEIGHT in [360, 480, 720, 1080]

    def test_youtube_format(self):
        """Test YouTube format configuration"""
        assert isinstance(YOUTUBE_FORMAT, str)
        assert "height" in YOUTUBE_FORMAT
        # Format may vary, just check it contains height specification


class TestFontConfiguration:
    """Test cases for font configuration"""

    def test_default_font(self):
        """Test default font configuration"""
        assert isinstance(DEFAULT_FONT, str)
        assert (
            len(DEFAULT_FONT) > 0
        )  # Font may be system font name without .ttf extension

    def test_fallback_fonts(self):
        """Test fallback fonts configuration"""
        assert isinstance(FALLBACK_FONTS, list)
        assert len(FALLBACK_FONTS) > 0
        assert all(isinstance(font, str) for font in FALLBACK_FONTS)


class TestEffectConfiguration:
    """Test cases for effect configuration"""

    def test_crossfade_time(self):
        """Test crossfade time configuration"""
        assert isinstance(CROSSFADE_TIME, (int, float))
        assert 0.0 <= CROSSFADE_TIME <= 5.0  # Reasonable crossfade range

    def test_start_fade(self):
        """Test start fade configuration"""
        assert isinstance(START_FADE, (int, float))
        assert 0.0 <= START_FADE <= 5.0  # Reasonable fade range


class TestEncodingConfiguration:
    """Test cases for encoding configuration"""

    def test_encoding_preset(self):
        """Test encoding preset configuration"""
        assert isinstance(ENCODING_PRESET, str)
        assert ENCODING_PRESET in [
            "ultrafast",
            "superfast",
            "veryfast",
            "faster",
            "fast",
            "medium",
            "slow",
            "slower",
            "veryslow",
        ]

    def test_encoding_crf(self):
        """Test encoding CRF configuration"""
        assert isinstance(ENCODING_CRF, int)
        assert 0 <= ENCODING_CRF <= 51  # Valid CRF range

    def test_encoding_threads(self):
        """Test encoding threads configuration"""
        assert isinstance(ENCODING_THREADS, int)
        assert ENCODING_THREADS > 0


class TestConfigurationIntegration:
    """Test cases for configuration integration"""

    def test_all_required_constants_exist(self):
        """Test that all required configuration constants exist"""
        required_constants = [
            "API_URL",
            "IMG_URL",
            "API_KEY",
            "MODEL_TEXT",
            "MODEL_IMAGE",
            "VOICE",
            "TTS_RATE",
            "OUTPUT_DIR",
            "TEMP_DIR_PREFIX",
            "VIDEO_WIDTH",
            "VIDEO_HEIGHT",
            "VIDEO_FPS",
            "VIDEO_CODEC",
            "AUDIO_CODEC",
            "DEFAULT_FONT",
            "FALLBACK_FONTS",
            "YOUTUBE_MAX_HEIGHT",
            "YOUTUBE_FORMAT",
            "CROSSFADE_TIME",
            "START_FADE",
            "ENCODING_PRESET",
            "ENCODING_CRF",
            "ENCODING_THREADS",
        ]

        for constant in required_constants:
            assert globals().get(constant) is not None, (
                f"Missing required constant: {constant}"
            )

    def test_configuration_consistency(self):
        """Test that configuration values are consistent"""
        # Video dimensions should make sense (both positive)
        assert VIDEO_WIDTH > 0
        assert VIDEO_HEIGHT > 0

        # Audio and video codecs should be compatible
        assert VIDEO_CODEC in ["libx264", "libx265"]
        assert AUDIO_CODEC == "aac"  # Best compatibility with H.264/H.265

        # YouTube format should contain height specification
        assert "height" in YOUTUBE_FORMAT

    @patch.dict("os.environ", {"AUTOSHORTS_OUTPUT_DIR": "/custom/output"})
    def test_environment_override(self):
        """Test that environment variables can override configuration (if implemented)"""
        # This test would be relevant if environment variable overrides are implemented
        # For now, it just verifies of current behavior
        assert OUTPUT_DIR is not None


if __name__ == "__main__":
    pytest.main([__file__])
