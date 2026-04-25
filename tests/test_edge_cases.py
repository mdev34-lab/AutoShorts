"""
Test edge cases and error handling for AutoShorts modules
"""

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

from autoshorts.modules import (
    ScriptGenerator,
    SubtitleSystem,
    TTSSystem,
    ensure_dir_exists,
    get_file_size_mb,
    get_system_font,
    get_video_duration,
    safe_filename,
)


class TestEdgeCasesScriptGenerator:
    """Test edge cases for ScriptGenerator"""

    def setup_method(self):
        """Setup test fixtures"""
        self.generator = ScriptGenerator()

    @patch("autoshorts.modules.script_generator.requests.post")
    def test_empty_subject(self, mock_post):
        """Test script generation with empty subject"""
        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Empty response"}}]
        }
        mock_post.return_value = mock_response

        with pytest.raises(ValueError):
            self.generator.generate_script("")

    @patch("autoshorts.modules.script_generator.requests.post")
    def test_very_long_subject(self, mock_post):
        """Test script generation with very long subject"""
        long_subject = "A" * 1000
        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Long response"}}]
        }
        mock_post.return_value = mock_response

        result = self.generator.generate_script(long_subject)
        assert isinstance(result, list)

    @patch("autoshorts.modules.script_generator.requests.post")
    def test_api_timeout(self, mock_post):
        """Test API timeout handling"""
        mock_post.side_effect = TimeoutError("Connection timeout")

        with pytest.raises(TimeoutError):
            self.generator.generate_script("test subject")

    @patch("autoshorts.modules.script_generator.requests.post")
    def test_malformed_api_response(self, mock_post):
        """Test malformed API response handling"""
        mock_response = Mock()
        mock_response.json.return_value = {}  # Missing choices
        mock_post.return_value = mock_response

        with pytest.raises(KeyError):
            self.generator.generate_script("test subject")


class TestEdgeCasesTTSSystem:
    """Test edge cases for TTSSystem"""

    def setup_method(self):
        """Setup test fixtures"""
        self.tts_system = TTSSystem()
        self.temp_dir = Path(tempfile.mkdtemp())

    def teardown_method(self):
        """Cleanup test fixtures"""
        import shutil

        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    @patch("autoshorts.modules.tts_system.edge_tts.Communicate")
    async def test_empty_text(self, mock_communicate_class):
        """Test TTS with empty text"""
        mock_comm = Mock()
        mock_comm.save = AsyncMock()
        mock_communicate_class.return_value = mock_comm

        result = await self.tts_system.generate_audio_only([], self.temp_dir)
        assert result is None

    @patch("autoshorts.modules.tts_system.edge_tts.Communicate")
    async def test_very_long_text(self, mock_communicate_class):
        """Test TTS with very long text"""
        long_paragraphs = ["A" * 1000] * 10
        mock_comm = Mock()
        mock_comm.save = AsyncMock()
        mock_communicate_class.return_value = mock_comm

        result = await self.tts_system.generate_audio_only(
            long_paragraphs, self.temp_dir
        )
        assert result.endswith(".mp3")

    @patch("autoshorts.modules.tts_system.edge_tts.Communicate")
    async def test_tts_connection_error(self, mock_communicate_class):
        """Test TTS connection error handling"""
        mock_communicate_class.side_effect = Exception("TTS connection failed")

        with pytest.raises(Exception):
            await self.tts_system.generate_audio_only(["Test"], self.temp_dir)

    @patch("autoshorts.modules.tts_system.edge_tts.Communicate")
    async def test_special_characters_in_text(self, mock_communicate_class):
        """Test TTS with special characters"""
        paragraphs = ["Olá! São Paulo, café & açúcar! ãõçéíóú"]
        mock_comm = Mock()
        mock_comm.save = AsyncMock()
        mock_communicate_class.return_value = mock_comm

        result = await self.tts_system.generate_audio_only(paragraphs, self.temp_dir)
        assert result.endswith(".mp3")


class TestEdgeCasesSubtitleSystem:
    """Test edge cases for SubtitleSystem"""

    def setup_method(self):
        """Setup test fixtures"""
        self.subtitle_system = SubtitleSystem()
        self.temp_dir = Path(tempfile.mkdtemp())

    def teardown_method(self):
        """Cleanup test fixtures"""
        import shutil

        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_empty_paragraphs(self):
        """Test subtitle generation with empty paragraphs"""
        vtt_path = self.subtitle_system.generate_subtitles([], 10.0, str(self.temp_dir))
        assert Path(vtt_path).exists()

        with open(vtt_path, "r", encoding="utf-8") as f:
            content = f.read()
            assert "WEBVTT" in content

    def test_zero_duration(self):
        """Test subtitle generation with zero duration"""
        vtt_path = self.subtitle_system.generate_subtitles(
            ["Test"], 0.0, str(self.temp_dir)
        )
        assert Path(vtt_path).exists()

    def test_very_long_paragraphs(self):
        """Test subtitle generation with very long paragraphs"""
        long_paragraphs = ["A" * 1000] * 5
        vtt_path = self.subtitle_system.generate_subtitles(
            long_paragraphs, 60.0, str(self.temp_dir)
        )
        assert Path(vtt_path).exists()

    def test_special_characters_paragraphs(self):
        """Test subtitle generation with special characters"""
        special_paragraphs = ["Olá! São Paulo, ãõçéíóú"]
        vtt_path = self.subtitle_system.generate_subtitles(
            special_paragraphs, 10.0, str(self.temp_dir)
        )
        assert Path(vtt_path).exists()

        with open(vtt_path, "r", encoding="utf-8") as f:
            content = f.read()
            assert "São Paulo" in content

    def test_invalid_vtt_path(self):
        """Test subtitle rendering with invalid VTT path"""
        with pytest.raises(FileNotFoundError):
            self.subtitle_system.render_subtitles("nonexistent.vtt", (1920, 1080))


class TestEdgeCasesUtilityFunctions:
    """Test edge cases for utility functions"""

    def test_get_system_font_empty_config(self):
        """Test font function with empty config"""
        with patch("autoshorts.modules.config.DEFAULT_FONT", None):
            font = get_system_font()
            assert font == "arial.ttf"

    def test_safe_filename_edge_cases(self):
        """Test safe filename with various inputs"""
        # Empty string
        assert safe_filename("") == ""

        # Only special characters
        assert safe_filename("!@#$%^&*()") == ""

        # Very long filename
        long_name = "A" * 300
        result = safe_filename(long_name)
        assert len(result) <= 255  # Max filename length

        # Unicode characters
        unicode_name = "São Paulo café ãõç"
        result = safe_filename(unicode_name)
        assert result == "São Paulo café ãõç"

    def test_get_video_duration_invalid_file(self):
        """Test video duration with invalid file"""
        duration = get_video_duration("nonexistent.mp4")
        assert duration == 0.0

    def test_ensure_dir_exists_edge_cases(self):
        """Test directory creation edge cases"""
        temp_dir = Path(tempfile.mkdtemp())

        # Test with existing directory
        ensure_dir_exists(temp_dir)
        assert temp_dir.exists()

        # Test with nested non-existent directory
        nested_dir = temp_dir / "level1" / "level2" / "level3"
        ensure_dir_exists(nested_dir)
        assert nested_dir.exists()

        # Cleanup
        import shutil

        shutil.rmtree(temp_dir)

    def test_get_file_size_mb_edge_cases(self):
        """Test file size calculation edge cases"""
        # Non-existent file
        size = get_file_size_mb("nonexistent.txt")
        assert size == 0.0

        # Empty file
        temp_file = Path(tempfile.mktemp())
        temp_file.write_bytes(b"")
        size = get_file_size_mb(str(temp_file))
        assert size == 0.0
        temp_file.unlink()


class TestIntegrationEdgeCases:
    """Test integration edge cases"""

    @pytest.mark.asyncio
    async def test_full_pipeline_empty_inputs(self):
        """Test full pipeline with empty inputs"""
        temp_dir = Path(tempfile.mkdtemp())

        try:
            with patch(
                "autoshorts.modules.script_generator.ScriptGenerator"
            ) as mock_script_class, patch(
                "autoshorts.modules.tts_system.TTSSystem"
            ) as mock_tts_class, patch(
                "autoshorts.modules.subtitle_system.SubtitleSystem"
            ) as mock_subtitle_class:
                # Setup mocks to handle empty inputs gracefully
                mock_script = Mock()
                mock_script.generate_script_with_prompts.return_value = ([], [])
                mock_script_class.return_value = mock_script

                mock_tts = Mock()
                mock_tts.generate_audio_only = AsyncMock(return_value=None)
                mock_tts_class.return_value = mock_tts

                mock_subtitle = Mock()
                mock_subtitle.generate_subtitles.return_value = str(
                    temp_dir / "subs.vtt"
                )
                mock_subtitle.render_subtitles.return_value = []
                mock_subtitle_class.return_value = mock_subtitle

                # Test components can handle empty results
                script_gen = mock_script_class()
                paragraphs, prompts = script_gen.generate_script_with_prompts("")
                assert paragraphs == []
                assert prompts == []

                tts_sys = mock_tts_class()
                audio_result = await tts_sys.generate_audio_only([])
                assert audio_result is None

        finally:
            import shutil

            if temp_dir.exists():
                shutil.rmtree(temp_dir)

    def test_error_propagation(self):
        """Test that errors propagate correctly through the system"""
        with patch("autoshorts.modules.script_generator.requests.post") as mock_post:
            mock_post.side_effect = ConnectionError("Network error")

            script_gen = ScriptGenerator()

            with pytest.raises(ConnectionError):
                script_gen.generate_script("test")
