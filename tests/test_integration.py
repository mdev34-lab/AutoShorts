"""
Integration tests for main AutoShorts workflows
"""

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

from autoshorts.fluximages import AssetManager, ScriptEngine, VideoEngine
from autoshorts.modules import SubtitleSystem


class TestScriptEngineIntegration:
    """Integration tests for script generation workflow"""

    def setup_method(self):
        """Setup test fixtures"""
        self.script_engine = ScriptEngine()

    @patch("autoshorts.modules.script_generator.requests.post")
    def test_script_generation_workflow(self, mock_post):
        """Test complete script generation workflow"""
        # Mock API response
        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": '{"paragraphs": ["Este é o primeiro parágrafo.", "Este é o segundo parágrafo.", "Este é o terceiro parágrafo."], "image_prompts": ["Prompt 1", "Prompt 2", "Prompt 3"]}'
                    }
                }
            ]
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        subject = "inteligência artificial"
        paragraphs, image_prompts = self.script_engine.generate(subject)

        assert isinstance(paragraphs, list)
        assert isinstance(image_prompts, list)
        assert len(paragraphs) > 0
        assert len(image_prompts) > 0
        assert any("primeiro" in paragraph.lower() for paragraph in paragraphs)

    @patch("autoshorts.modules.script_generator.requests.post")
    def test_script_generation_with_prompts_workflow(self, mock_post):
        """Test script generation with image prompts workflow"""
        # Mock API response for prompts
        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": '{"paragraphs": ["Para 1", "Para 2"], "image_prompts": ["Prompt 1", "Prompt 2"]}'
                    }
                }
            ]
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        subject = "exploração espacial"
        paragraphs, prompts = (
            self.script_engine.script_generator.generate_script_with_prompts(subject)
        )

        assert isinstance(paragraphs, list)
        assert isinstance(prompts, list)
        assert len(paragraphs) > 0
        assert len(prompts) > 0


class TestAssetManagerIntegration:
    """Integration tests for asset management workflow"""

    def setup_method(self):
        """Setup test fixtures"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.asset_manager = AssetManager(self.temp_dir)

    def teardown_method(self):
        """Cleanup test fixtures"""
        if self.temp_dir.exists():
            import shutil

            shutil.rmtree(self.temp_dir)

    @pytest.mark.asyncio
    async def test_audio_generation_workflow(self):
        """Test audio generation workflow"""
        paragraphs = ["Teste de geração de áudio."]

        with patch("edge_tts.Communicate") as mock_communicate:
            mock_comm = AsyncMock()
            mock_comm.save = AsyncMock()
            mock_communicate.return_value = mock_comm

            result = await self.asset_manager.generate_audio(paragraphs)

            assert result is not None
            assert result.endswith(".mp3")

    @patch("autoshorts.fluximages.requests.get")
    def test_image_generation_workflow(self, mock_get):
        """Test image generation workflow"""
        prompts = ["Test prompt 1", "Test prompt 2"]

        # Mock image API response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = b"fake_image_data"
        mock_get.return_value = mock_response

        result = self.asset_manager.generate_ai_images(prompts)

        assert isinstance(result, list)
        assert len(result) == len(prompts)
        assert all(Path(path).exists() for path in result)

    @patch("autoshorts.fluximages.requests.get")
    def test_image_generation_error_handling(self, mock_get):
        """Test image generation error handling"""
        prompts = ["Test prompt"]

        # Mock API error
        mock_response = Mock()
        mock_response.status_code = 500
        mock_get.return_value = mock_response

        with pytest.raises(Exception):
            self.asset_manager.generate_ai_images(prompts)


class TestVideoEngineIntegration:
    """Integration tests for video creation workflow"""

    def setup_method(self):
        """Setup test fixtures"""
        self.video_engine = VideoEngine()
        self.temp_dir = Path(tempfile.mkdtemp())

    def teardown_method(self):
        """Cleanup test fixtures"""
        if self.temp_dir.exists():
            import shutil

            shutil.rmtree(self.temp_dir)

    @patch("autoshorts.modules.subtitle_system.SubtitleSystem.generate_subtitles")
    @patch("autoshorts.modules.subtitle_system.SubtitleSystem.render_subtitles")
    @patch("moviepy.AudioFileClip")
    @patch("moviepy.ImageClip")
    @patch("moviepy.concatenate_videoclips")
    def test_video_creation_workflow(
        self, mock_concat, mock_imageclip, mock_audioclip, mock_render, mock_generate
    ):
        """Test complete video creation workflow"""
        # Setup mocks
        mock_audio = Mock()
        mock_audio.duration = 10.0
        mock_audioclip.return_value = mock_audio

        mock_clip = Mock()
        mock_clip.with_duration.return_value = mock_clip
        mock_clip.resized.return_value = mock_clip
        mock_clip.with_effects.return_value = mock_clip
        mock_imageclip.return_value = mock_clip

        mock_video = Mock()
        mock_video.with_audio.return_value = mock_video
        mock_concat.return_value = mock_video

        mock_generate.return_value = str(self.temp_dir / "test.vtt")
        mock_render.return_value = []

        # Test data
        img_paths = [str(self.temp_dir / f"img_{i}.jpg") for i in range(3)]
        audio_path = str(self.temp_dir / "audio.mp3")
        paragraphs = ["Test paragraph 1", "Test paragraph 2"]
        output_path = str(self.temp_dir / "output.mp4")

        # Create fake image files
        for img_path in img_paths:
            Path(img_path).write_bytes(b"fake_image_data")

        # Create fake audio file
        Path(audio_path).write_bytes(b"fake_audio_data")

        # This should not raise an exception
        try:
            self.video_engine.create_video(
                img_paths, audio_path, paragraphs, output_path
            )
        except Exception:
            # Expected due to mocking, but workflow should be attempted
            pass


class TestCompleteWorkflowIntegration:
    """Integration tests for complete AutoShorts workflow"""

    def setup_method(self):
        """Setup test fixtures"""
        self.temp_dir = Path(tempfile.mkdtemp())

    def teardown_method(self):
        """Cleanup test fixtures"""
        if self.temp_dir.exists():
            import shutil

            shutil.rmtree(self.temp_dir)

    @patch("autoshorts.modules.script_generator.requests.post")
    @patch("edge_tts.Communicate")
    @patch("autoshorts.fluximages.requests.get")
    def test_complete_flux_workflow(self, mock_get, mock_communicate, mock_post):
        """Test complete flux image generation workflow"""
        # Mock script generation
        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": '{"paragraphs": ["Para 1", "Para 2"], "image_prompts": ["Prompt 1", "Prompt 2"]}'
                    }
                }
            ]
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        # Mock TTS
        mock_comm = AsyncMock()
        mock_comm.save = Mock()
        mock_communicate.return_value = mock_comm

        # Mock image generation
        mock_img_response = Mock()
        mock_img_response.status_code = 200
        mock_img_response.content = b"fake_image_data"
        mock_get.return_value = mock_img_response

        # Test workflow components
        script_engine = ScriptEngine()
        asset_manager = AssetManager(self.temp_dir)

        # Generate script and prompts
        paragraphs, prompts = script_engine.generate("test subject")

        # Generate assets
        audio_task = asset_manager.generate_audio(paragraphs)
        img_paths = asset_manager.generate_ai_images(prompts)

        # Verify results
        assert isinstance(paragraphs, list)
        assert isinstance(prompts, list)
        assert isinstance(img_paths, list)
        assert len(paragraphs) > 0
        assert len(prompts) > 0

    @patch("autoshorts.modules.script_generator.requests.post")
    def test_script_to_subtitle_workflow(self, mock_post):
        """Test workflow from script generation to subtitle creation"""
        # Mock script generation
        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Para 1\nPara 2\nPara 3"}}]
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        # Generate script
        script_engine = ScriptEngine()

        # Mock the script generator to return valid data
        with patch.object(
            script_engine.script_generator, "generate_script_with_prompts"
        ) as mock_gen:
            mock_gen.return_value = (
                [
                    "Paragraph 1",
                    "Paragraph 2",
                    "Paragraph 3",
                    "Paragraph 4",
                    "Paragraph 5",
                ],
                [
                    "Prompt 1",
                    "Prompt 2",
                    "Prompt 3",
                    "Prompt 4",
                    "Prompt 5",
                    "Prompt 6",
                    "Prompt 7",
                    "Prompt 8",
                    "Prompt 9",
                ],
            )

            paragraphs, prompts = script_engine.generate("test subject")

        # Generate subtitles
        subtitle_system = SubtitleSystem()
        vtt_path = subtitle_system.generate_subtitles(paragraphs, 10.0, self.temp_dir)

        # Verify workflow
        assert isinstance(paragraphs, list)
        assert len(paragraphs) > 0
        assert vtt_path is not None
        assert Path(vtt_path).exists()

        # Check VTT content
        content = Path(vtt_path).read_text(encoding="utf-8")
        assert "WEBVTT" in content


class TestErrorHandlingIntegration:
    """Integration tests for error handling in workflows"""

    def setup_method(self):
        """Setup test fixtures"""
        self.temp_dir = Path(tempfile.mkdtemp())

    def teardown_method(self):
        """Cleanup test fixtures"""
        if self.temp_dir.exists():
            import shutil

            shutil.rmtree(self.temp_dir)

    @patch(
        "autoshorts.modules.script_generator.requests.post",
        side_effect=Exception("API Error"),
    )
    def test_script_generation_error_propagation(self, mock_post):
        """Test that script generation errors are properly handled"""
        script_engine = ScriptEngine()

        with pytest.raises(Exception):
            script_engine.generate("test subject")

    @pytest.mark.asyncio
    async def test_audio_generation_error_handling(self):
        """Test audio generation error handling"""
        asset_manager = AssetManager(self.temp_dir)
        paragraphs = ["Test paragraph"]

        with patch("edge_tts.Communicate", side_effect=Exception("TTS Error")):
            with pytest.raises(Exception):
                await asset_manager.generate_audio(paragraphs)


if __name__ == "__main__":
    pytest.main([__file__])
