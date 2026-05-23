"""
Test edge cases and error handling for AutoShorts modules
"""

import subprocess
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
        self.generator = ScriptGenerator()

    @patch("autoshorts.modules.script_generator.requests.post")
    def test_empty_subject(self, mock_post):
        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Empty response"}}]
        }
        mock_post.return_value = mock_response

        result = self.generator.generate_script("")
        assert isinstance(result, list)
        assert len(result) > 0

    @patch("autoshorts.modules.script_generator.requests.post")
    def test_very_long_subject(self, mock_post):
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
        mock_post.side_effect = TimeoutError("Connection timeout")

        result = self.generator.generate_script("test subject")
        assert isinstance(result, list)
        assert len(result) == 5

    @patch("autoshorts.modules.script_generator.requests.post")
    def test_malformed_api_response(self, mock_post):
        mock_response = Mock()
        mock_response.json.return_value = {}
        mock_post.return_value = mock_response

        result = self.generator.generate_script("test subject")
        assert isinstance(result, list)
        assert len(result) == 5


class TestEdgeCasesTTSSystem:
    """Test edge cases for TTSSystem"""

    def setup_method(self):
        self.tts_system = TTSSystem()
        self.temp_dir = Path(tempfile.mkdtemp())

    def teardown_method(self):
        import shutil

        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    @patch("autoshorts.modules.tts_system.edge_tts.Communicate")
    async def test_empty_text(self, mock_communicate_class):
        mock_comm = Mock()
        mock_comm.save = AsyncMock()
        mock_communicate_class.return_value = mock_comm

        result = await self.tts_system.generate_audio_only([], self.temp_dir)
        assert result.endswith(".mp3")

    @patch("autoshorts.modules.tts_system.edge_tts.Communicate")
    async def test_very_long_text(self, mock_communicate_class):
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
        mock_communicate_class.side_effect = Exception("TTS connection failed")

        with pytest.raises(Exception):
            await self.tts_system.generate_audio_only(["Test"], self.temp_dir)

    @patch("autoshorts.modules.tts_system.edge_tts.Communicate")
    async def test_special_characters_in_text(self, mock_communicate_class):
        paragraphs = ["Olá! São Paulo, café & açúcar! ãõçéíóú"]
        mock_comm = Mock()
        mock_comm.save = AsyncMock()
        mock_communicate_class.return_value = mock_comm

        result = await self.tts_system.generate_audio_only(paragraphs, self.temp_dir)
        assert result.endswith(".mp3")


class TestEdgeCasesSubtitleSystem:
    """Test edge cases for SubtitleSystem"""

    def setup_method(self):
        self.subtitle_system = SubtitleSystem()
        self.temp_dir = Path(tempfile.mkdtemp())

    def teardown_method(self):
        import shutil

        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_empty_paragraphs(self):
        vtt_path = self.subtitle_system.generate_subtitles([], 10.0, str(self.temp_dir))
        assert Path(vtt_path).exists()

        with open(vtt_path, "r", encoding="utf-8") as f:
            content = f.read()
            assert "WEBVTT" in content

    def test_zero_duration(self):
        vtt_path = self.subtitle_system.generate_subtitles(
            ["Test"], 0.0, str(self.temp_dir)
        )
        assert Path(vtt_path).exists()

    def test_very_long_paragraphs(self):
        long_paragraphs = ["A" * 1000] * 5
        vtt_path = self.subtitle_system.generate_subtitles(
            long_paragraphs, 60.0, str(self.temp_dir)
        )
        assert Path(vtt_path).exists()

    def test_special_characters_paragraphs(self):
        special_paragraphs = ["Olá! São Paulo, ãõçéíóú"]
        vtt_path = self.subtitle_system.generate_subtitles(
            special_paragraphs, 10.0, str(self.temp_dir)
        )
        assert Path(vtt_path).exists()

        with open(vtt_path, "r", encoding="utf-8") as f:
            content = f.read()
            assert "São Paulo" in content

    def test_invalid_vtt_path(self):
        result = self.subtitle_system.render_subtitles("nonexistent.vtt", (1920, 1080))
        assert result == []


class TestEdgeCasesUtilityFunctions:
    """Test edge cases for utility functions"""

    def test_get_system_font_empty_config(self):
        with patch("autoshorts.modules.utils.DEFAULT_FONT", "arial.ttf"):
            font = get_system_font()
            assert font == "arial.ttf"

    def test_safe_filename_edge_cases(self):
        assert safe_filename("") == ""
        assert safe_filename("!@#$%^&*()") is not None
        long_name = "A" * 300
        result = safe_filename(long_name)
        assert len(result) <= 50
        unicode_name = "São Paulo café ãõç"
        result = safe_filename(unicode_name)
        assert result == "São_Paulo_café_ãõç"

    def test_get_video_duration_invalid_file(self):
        with patch("autoshorts.modules.utils.subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.CalledProcessError(1, "ffprobe")
            with pytest.raises(subprocess.CalledProcessError):
                get_video_duration("nonexistent.mp4")

    def test_ensure_dir_exists_edge_cases(self):
        temp_dir = Path(tempfile.mkdtemp())
        ensure_dir_exists(temp_dir)
        assert temp_dir.exists()
        nested_dir = temp_dir / "level1" / "level2" / "level3"
        ensure_dir_exists(nested_dir)
        assert nested_dir.exists()
        import shutil

        shutil.rmtree(temp_dir)

    def test_get_file_size_mb_edge_cases(self):
        size = get_file_size_mb("nonexistent.txt")
        assert size == 0.0
        temp_file = Path(tempfile.mktemp())
        temp_file.write_bytes(b"")
        size = get_file_size_mb(str(temp_file))
        assert size == 0.0
        temp_file.unlink()


class TestIntegrationEdgeCases:
    """Test integration edge cases"""

    @pytest.mark.asyncio
    async def test_full_pipeline_empty_inputs(self):
        temp_dir = Path(tempfile.mkdtemp())

        try:
            with (
                patch(
                    "autoshorts.modules.script_generator.ScriptGenerator"
                ) as mock_script_class,
                patch("autoshorts.modules.tts_system.TTSSystem") as mock_tts_class,
                patch(
                    "autoshorts.modules.subtitle_system.SubtitleSystem"
                ) as mock_subtitle_class,
            ):
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
        with patch("autoshorts.modules.script_generator.requests.post") as mock_post:
            mock_post.side_effect = ConnectionError("Network error")

            script_gen = ScriptGenerator()

            result = script_gen.generate_script("test")
            assert isinstance(result, list)
            assert len(result) == 5
