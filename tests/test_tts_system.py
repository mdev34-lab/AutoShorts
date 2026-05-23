"""
Test TTS (Text-to-Speech) functionality
"""

import sys
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

# Add src to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from autoshorts.modules.tts_system import TTSSystem


class TestTTSSystem:
    """Test cases for TTSSystem class"""

    def setup_method(self):
        """Setup test fixtures"""
        self.temp_dir = Path(tempfile.mkdtemp())

    def teardown_method(self):
        """Cleanup test fixtures"""
        if self.temp_dir.exists():
            import shutil

            shutil.rmtree(self.temp_dir)

    def test_init(self):
        """Test TTS system initialization"""
        tts_system = TTSSystem()
        assert hasattr(tts_system, "voice")
        assert hasattr(tts_system, "rate")
        assert hasattr(tts_system, "subtitle_system")

    @pytest.mark.asyncio
    async def test_generate_audio_only(self):
        """Test audio generation only"""
        paragraphs = ["Este é um teste de geração de áudio."]

        # Create a proper async mock
        async def mock_save(*args, **kwargs):
            pass

        mock_comm = Mock()
        mock_comm.save = mock_save

        with patch(
            "autoshorts.modules.tts_system.edge_tts.Communicate", return_value=mock_comm
        ):
            tts_system = TTSSystem()
            result = await tts_system.generate_audio_only(paragraphs, self.temp_dir)

            # Should return path to generated audio file
            assert result is not None
            assert result.endswith(".mp3")

    @pytest.mark.asyncio
    async def test_generate_audio_and_subtitles(self):
        """Test combined audio and subtitle generation"""
        paragraphs = ["Primeiro parágrafo.", "Segundo parágrafo."]

        # Create a proper async mock
        async def mock_save(*args, **kwargs):
            pass

        mock_comm = Mock()
        mock_comm.save = mock_save

        mock_subprocess_result = Mock()
        mock_subprocess_result.stdout = "10.0"
        mock_subprocess_result.stderr = ""

        with (
            patch(
                "autoshorts.modules.tts_system.edge_tts.Communicate",
                return_value=mock_comm,
            ),
            patch(
                "autoshorts.modules.tts_system.subprocess.run",
                return_value=mock_subprocess_result,
            ),
        ):
            tts_system = TTSSystem()
            (
                audio_path,
                subtitle_path,
                duration,
            ) = await tts_system.generate_audio_and_subtitles(paragraphs, self.temp_dir)

            # Should return paths to both files and duration
            assert audio_path is not None
            assert subtitle_path is not None
            assert audio_path.endswith(".mp3")
            assert subtitle_path.endswith(".vtt")
            assert duration == 10.0

    @pytest.mark.asyncio
    async def test_generate_audio_only_with_special_characters(self):
        """Test audio generation with special characters"""
        paragraphs = ['Texto com "aspas" e caracteres especiais!']

        async def mock_save(*args, **kwargs):
            pass

        mock_comm = Mock()
        mock_comm.save = mock_save

        with patch(
            "autoshorts.modules.tts_system.edge_tts.Communicate", return_value=mock_comm
        ):
            tts_system = TTSSystem()
            result = await tts_system.generate_audio_only(paragraphs, self.temp_dir)

            assert result is not None
            assert result.endswith(".mp3")

    @pytest.mark.asyncio
    async def test_generate_audio_only_multiple_paragraphs(self):
        """Test audio generation with multiple paragraphs"""
        paragraphs = [
            "Primeiro parágrafo do texto.",
            "Segundo parágrafo do texto.",
            "Terceiro parágrafo do texto.",
        ]

        async def mock_save(*args, **kwargs):
            pass

        mock_comm = Mock()
        mock_comm.save = mock_save

        with patch(
            "autoshorts.modules.tts_system.edge_tts.Communicate", return_value=mock_comm
        ):
            tts_system = TTSSystem()
            result = await tts_system.generate_audio_only(paragraphs, self.temp_dir)

            assert result is not None

    @pytest.mark.asyncio
    async def test_generate_audio_only_empty_paragraphs(self):
        """Test handling of empty paragraphs"""
        paragraphs = []

        async def mock_save(*args, **kwargs):
            pass

        mock_comm = Mock()
        mock_comm.save = mock_save

        with patch(
            "autoshorts.modules.tts_system.edge_tts.Communicate", return_value=mock_comm
        ):
            tts_system = TTSSystem()
            result = await tts_system.generate_audio_only(paragraphs, self.temp_dir)

            # Should still return a path even for empty input
            assert result is not None

    @pytest.mark.asyncio
    async def test_generate_audio_and_subtitles_with_mocked_subtitle_system(self):
        """Test combined generation with mocked subtitle system"""
        paragraphs = ["Test paragraph."]

        async def mock_save(*args, **kwargs):
            pass

        mock_comm = Mock()
        mock_comm.save = mock_save

        mock_subprocess_result = Mock()
        mock_subprocess_result.stdout = "5.5"
        mock_subprocess_result.stderr = ""

        mock_subtitle_system = Mock()
        mock_subtitle_system.generate_subtitles = Mock(
            return_value=str(self.temp_dir / "subs.vtt")
        )

        with (
            patch(
                "autoshorts.modules.tts_system.edge_tts.Communicate",
                return_value=mock_comm,
            ),
            patch(
                "autoshorts.modules.tts_system.subprocess.run",
                return_value=mock_subprocess_result,
            ),
        ):
            tts_system = TTSSystem()
            tts_system.subtitle_system = mock_subtitle_system

            (
                audio_path,
                subtitle_path,
                duration,
            ) = await tts_system.generate_audio_and_subtitles(paragraphs, self.temp_dir)

            assert audio_path is not None
            assert duration == 5.5

    @pytest.mark.asyncio
    async def test_tts_error_handling(self):
        """Test error handling in TTS generation"""
        paragraphs = ["Teste de erro"]

        with patch(
            "autoshorts.modules.tts_system.edge_tts.Communicate",
            side_effect=Exception("TTS Error"),
        ):
            tts_system = TTSSystem()
            with pytest.raises(Exception):
                await tts_system.generate_audio_only(paragraphs, self.temp_dir)

    @pytest.mark.asyncio
    async def test_generate_audio_and_subtitles_error_handling(self):
        """Test error handling in combined generation"""
        paragraphs = ["Test paragraph"]

        with patch(
            "autoshorts.modules.tts_system.edge_tts.Communicate",
            side_effect=Exception("TTS Error"),
        ):
            tts_system = TTSSystem()
            with pytest.raises(Exception):
                await tts_system.generate_audio_and_subtitles(paragraphs, self.temp_dir)


class TestTTSSystemIntegration:
    """Integration tests for TTSSystem"""

    def setup_method(self):
        """Setup test fixtures"""
        self.temp_dir = Path(tempfile.mkdtemp())

    def teardown_method(self):
        """Cleanup test fixtures"""
        if self.temp_dir.exists():
            import shutil

            shutil.rmtree(self.temp_dir)

    @pytest.mark.asyncio
    async def test_full_workflow(self):
        """Test full TTS workflow"""
        paragraphs = [
            "Este é o primeiro parágrafo de teste.",
            "Este é o segundo parágrafo de teste.",
        ]

        async def mock_save(*args, **kwargs):
            # Simulate creating the audio file
            audio_path = (
                args[0]
                if args
                else kwargs.get("path", str(self.temp_dir / "narration.mp3"))
            )
            Path(audio_path).write_bytes(b"fake audio")

        mock_comm = Mock()
        mock_comm.save = mock_save

        mock_subprocess_result = Mock()
        mock_subprocess_result.stdout = "15.0"
        mock_subprocess_result.stderr = ""

        with (
            patch(
                "autoshorts.modules.tts_system.edge_tts.Communicate",
                return_value=mock_comm,
            ),
            patch(
                "autoshorts.modules.tts_system.subprocess.run",
                return_value=mock_subprocess_result,
            ),
        ):
            tts_system = TTSSystem()
            (
                audio_path,
                subtitle_path,
                duration,
            ) = await tts_system.generate_audio_and_subtitles(paragraphs, self.temp_dir)

            assert audio_path is not None
            assert subtitle_path is not None
            assert duration == 15.0


if __name__ == "__main__":
    pytest.main([__file__])
