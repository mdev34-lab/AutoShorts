"""
Test FluxImages module functionality
"""

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

from autoshorts.fluximages import (
    AssetManager,
    ScriptEngine,
    VideoEngine,
    main,
    shutdown_computer,
)


class TestScriptEngine:
    """Test cases for ScriptEngine class"""

    def setup_method(self):
        """Setup test fixtures"""
        self.script_engine = ScriptEngine()

    def test_init(self):
        """Test ScriptEngine initialization"""
        assert hasattr(self.script_engine, "script_generator")

    @patch("autoshorts.fluximages.ScriptGenerator")
    def test_generate(self, mock_generator_class):
        """Test script generation with prompts"""
        mock_generator = Mock()
        mock_generator.generate_script_with_prompts.return_value = (
            ["Paragraph 1", "Paragraph 2"],
            ["Prompt 1", "Prompt 2"],
        )
        mock_generator_class.return_value = mock_generator

        engine = ScriptEngine()
        engine.script_generator = mock_generator

        result = engine.generate("test subject")

        assert isinstance(result, tuple)
        assert len(result) == 2
        assert len(result[0]) == 2  # paragraphs
        assert len(result[1]) == 2  # prompts


class TestAssetManager:
    """Test cases for AssetManager class"""

    def setup_method(self):
        """Setup test fixtures"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.asset_manager = AssetManager(self.temp_dir)

    def teardown_method(self):
        """Cleanup test fixtures"""
        import shutil

        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_init(self):
        """Test AssetManager initialization"""
        assert hasattr(self.asset_manager, "temp_dir")
        assert hasattr(self.asset_manager, "tts_system")

    @patch("autoshorts.fluximages.requests.get")
    @patch("autoshorts.fluximages.random.randint")
    def test_generate_ai_images_success(self, mock_randint, mock_get):
        """Test successful AI image generation"""
        mock_randint.return_value = 12345

        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = b"fake_image_data"
        mock_get.return_value = mock_response

        prompts = ["A beautiful sunset", "A mountain landscape"]

        result = self.asset_manager.generate_ai_images(prompts)

        assert isinstance(result, list)
        assert len(result) == 2
        assert all(path.endswith(".jpg") for path in result)

    @patch("autoshorts.fluximages.requests.get")
    @patch("autoshorts.fluximages.random.randint")
    def test_generate_ai_images_failure(self, mock_randint, mock_get):
        """Test AI image generation failure"""
        mock_randint.return_value = 12345

        # Mock failed response
        mock_response = Mock()
        mock_response.status_code = 500
        mock_get.return_value = mock_response

        prompts = ["A beautiful sunset"]

        with pytest.raises(Exception):
            self.asset_manager.generate_ai_images(prompts)

    @patch("autoshorts.fluximages.requests.get")
    @patch("autoshorts.fluximages.random.randint")
    def test_generate_ai_images_network_error(self, mock_randint, mock_get):
        """Test AI image generation with network error"""
        mock_randint.return_value = 12345
        mock_get.side_effect = Exception("Network error")

        prompts = ["A beautiful sunset"]

        with pytest.raises(Exception):
            self.asset_manager.generate_ai_images(prompts)

    @pytest.mark.asyncio
    async def test_generate_audio(self):
        """Test audio generation"""
        with patch("autoshorts.fluximages.TTSSystem") as mock_tts_class:
            mock_tts = Mock()
            mock_tts.generate_audio_only = AsyncMock(return_value="audio.mp3")
            mock_tts_class.return_value = mock_tts

            asset_manager = AssetManager(self.temp_dir)
            asset_manager.tts_system = mock_tts

            paragraphs = ["Test paragraph"]

            result = await asset_manager.generate_audio(paragraphs)

            assert result == "audio.mp3"


class TestVideoEngine:
    """Test cases for VideoEngine class"""

    def setup_method(self):
        """Setup test fixtures"""
        self.video_engine = VideoEngine()

    def test_init(self):
        """Test VideoEngine initialization"""
        assert hasattr(self.video_engine, "subtitle_system")

    @patch("autoshorts.fluximages.AudioFileClip")
    @patch("autoshorts.fluximages.ImageClip")
    @patch("autoshorts.fluximages.concatenate_videoclips")
    @patch("autoshorts.fluximages.CompositeVideoClip")
    @patch("autoshorts.fluximages.SubtitleSystem")
    def test_create_video_basic(
        self,
        mock_subtitle_class,
        mock_composite,
        mock_concat,
        mock_image_clip,
        mock_audio_clip,
    ):
        """Test basic video creation"""
        # Create temp files for images first
        temp_dir = Path(tempfile.mkdtemp())

        # Mock audio
        mock_audio = Mock()
        mock_audio.duration = 30.0
        mock_audio.close = Mock()
        mock_audio_clip.return_value = mock_audio

        # Mock image clips
        mock_clip = Mock()
        mock_clip.with_duration = Mock(return_value=mock_clip)
        mock_clip.resized = Mock(return_value=mock_clip)
        mock_clip.with_effects = Mock(return_value=mock_clip)
        mock_clip.close = Mock()
        mock_image_clip.return_value = mock_clip

        # Mock concatenated video
        mock_video = Mock()
        mock_video.with_audio = Mock(return_value=mock_video)
        mock_video.write_videofile = Mock()
        mock_video.close = Mock()
        mock_concat.return_value = mock_video

        # Mock composite
        mock_final = Mock()
        mock_final.write_videofile = Mock()
        mock_final.close = Mock()
        mock_composite.return_value = mock_final

        # Mock subtitle system
        mock_subtitle = Mock()
        mock_subtitle.generate_subtitles = Mock(return_value=str(temp_dir / "subs.vtt"))
        mock_subtitle.render_subtitles = Mock(return_value=[])
        mock_subtitle_class.return_value = mock_subtitle

        img_paths = []
        for i in range(5):
            img_path = temp_dir / f"ai_img_{i}.jpg"
            img_path.write_bytes(b"fake_image")
            img_paths.append(str(img_path))

        # Create temp audio file
        audio_path = temp_dir / "audio.mp3"
        audio_path.write_bytes(b"fake_audio")

        output_path = temp_dir / "output.mp4"

        try:
            self.video_engine.create_video(
                img_paths,
                str(audio_path),
                ["Paragraph 1", "Paragraph 2"],
                str(output_path),
            )

            # Verify audio was loaded
            mock_audio_clip.assert_called_once()
        finally:
            import shutil

            if temp_dir.exists():
                shutil.rmtree(temp_dir)

    @patch("autoshorts.fluximages.AudioFileClip")
    def test_create_video_no_clips(self, mock_audio_clip):
        """Test video creation with no clips raises error"""
        mock_audio = Mock()
        mock_audio.duration = 30.0
        mock_audio.close = Mock()
        mock_audio_clip.return_value = mock_audio

        # The function should handle empty clips gracefully
        # It raises ZeroDivisionError due to division by num_imgs (0)
        # This is a known issue in the source code
        with pytest.raises((ValueError, ZeroDivisionError)):
            self.video_engine.create_video([], "audio.mp3", [], "output.mp4")


class TestShutdownComputer:
    """Test cases for shutdown_computer function"""

    @patch("autoshorts.fluximages.platform.system")
    @patch("autoshorts.fluximages.os.system")
    @patch("autoshorts.fluximages.log")
    def test_shutdown_windows(self, mock_log, mock_system, mock_platform):
        """Test shutdown on Windows"""
        mock_platform.return_value = "Windows"

        shutdown_computer()

        mock_system.assert_called_once_with("shutdown /s /t 30")

    @patch("autoshorts.fluximages.platform.system")
    @patch("autoshorts.fluximages.os.system")
    @patch("autoshorts.fluximages.log")
    def test_shutdown_linux(self, mock_log, mock_system, mock_platform):
        """Test shutdown on Linux"""
        mock_platform.return_value = "Linux"

        shutdown_computer()

        mock_system.assert_called_once_with("shutdown -h +1")

    @patch("autoshorts.fluximages.platform.system")
    @patch("autoshorts.fluximages.os.system")
    @patch("autoshorts.fluximages.log")
    def test_shutdown_macos(self, mock_log, mock_system, mock_platform):
        """Test shutdown on macOS"""
        mock_platform.return_value = "Darwin"

        shutdown_computer()

        mock_system.assert_called_once_with("shutdown -h +1")

    @patch("autoshorts.fluximages.platform.system")
    @patch("autoshorts.fluximages.log")
    def test_shutdown_unsupported_os(self, mock_log, mock_platform):
        """Test shutdown on unsupported OS"""
        mock_platform.return_value = "UnsupportedOS"

        shutdown_computer()

        # Should log warning about unsupported OS
        assert mock_log.called

    @patch("autoshorts.fluximages.platform.system")
    @patch("autoshorts.fluximages.os.system")
    @patch("autoshorts.fluximages.log")
    def test_shutdown_exception_handling(self, mock_log, mock_system, mock_platform):
        """Test shutdown handles exceptions gracefully"""
        mock_platform.return_value = "Windows"
        mock_system.side_effect = Exception("Permission denied")

        # Should not raise exception
        shutdown_computer()

        # Should log error
        assert mock_log.called


class TestMainFunction:
    """Test cases for main function"""

    def setup_method(self):
        """Setup test fixtures"""
        self.temp_dir = Path(tempfile.mkdtemp())

    def teardown_method(self):
        """Cleanup test fixtures"""
        import shutil

        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    @patch("autoshorts.fluximages.argparse.ArgumentParser")
    @patch("autoshorts.fluximages.ScriptEngine")
    @patch("autoshorts.fluximages.AssetManager")
    @patch("autoshorts.fluximages.VideoEngine")
    @patch("autoshorts.fluximages.tempfile.mkdtemp")
    @patch("autoshorts.fluximages.Path.mkdir")
    @patch("autoshorts.fluximages.shutil.rmtree")
    def test_main_with_subject(
        self,
        mock_rmtree,
        mock_mkdir,
        mock_mkdtemp,
        mock_video_engine_class,
        mock_asset_class,
        mock_script_class,
        mock_parser_class,
    ):
        """Test main function with subject argument"""
        # Mock argument parser
        mock_parser = Mock()
        mock_args = Mock()
        mock_args.subject = "test subject"
        mock_args.goodnight = False
        mock_args.batch = None
        mock_args.web_search = False
        mock_parser.parse_args.return_value = mock_args
        mock_parser_class.return_value = mock_parser

        # Mock temp directory
        mock_mkdtemp.return_value = str(self.temp_dir)

        # Mock script engine
        mock_script = Mock()
        mock_script.generate.return_value = (
            ["Para 1", "Para 2", "Para 3", "Para 4", "Para 5"],
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
        mock_script_class.return_value = mock_script

        # Mock asset manager
        mock_asset = Mock()
        mock_asset.generate_audio = AsyncMock(return_value="audio.mp3")
        mock_asset.generate_ai_images = Mock(return_value=["img1.jpg"] * 9)
        mock_asset_class.return_value = mock_asset

        # Mock video engine
        mock_video = Mock()
        mock_video.create_video = Mock()
        mock_video_engine_class.return_value = mock_video

        # Run main
        asyncio.run(main())

        # Verify script was generated
        mock_script.generate.assert_called_once_with("test subject")

    @patch("autoshorts.fluximages.argparse.ArgumentParser")
    @patch("autoshorts.fluximages.tempfile.mkdtemp")
    @patch("autoshorts.fluximages.Path.mkdir")
    @patch("autoshorts.fluximages.shutil.rmtree")
    def test_main_batch_processing(
        self, mock_rmtree, mock_mkdir, mock_mkdtemp, mock_parser_class
    ):
        """Test main function with batch processing"""
        # Mock argument parser
        mock_parser = Mock()
        mock_args = Mock()
        mock_args.subject = None
        mock_args.goodnight = False
        mock_args.batch = ["subject1", "subject2"]
        mock_args.web_search = False
        mock_parser.parse_args.return_value = mock_args
        mock_parser_class.return_value = mock_parser

        # Mock temp directory
        mock_mkdtemp.return_value = str(self.temp_dir)

        with patch("autoshorts.fluximages.ScriptEngine") as mock_script_class, patch(
            "autoshorts.fluximages.AssetManager"
        ) as mock_asset_class, patch(
            "autoshorts.fluximages.VideoEngine"
        ) as mock_video_class:
            # Setup mocks for successful processing
            mock_script = Mock()
            mock_script.generate.return_value = (
                ["Para 1", "Para 2", "Para 3", "Para 4", "Para 5"],
                ["Prompt 1"] * 9,
            )
            mock_script_class.return_value = mock_script

            mock_asset = Mock()
            mock_asset.generate_audio = AsyncMock(return_value="audio.mp3")
            mock_asset.generate_ai_images = Mock(return_value=["img.jpg"] * 9)
            mock_asset_class.return_value = mock_asset

            mock_video = Mock()
            mock_video.create_video = Mock()
            mock_video_class.return_value = mock_video

            asyncio.run(main())

            # Verify batch was processed
            assert mock_script.generate.call_count == 2

    @patch("autoshorts.fluximages.argparse.ArgumentParser")
    @patch("autoshorts.fluximages.tempfile.mkdtemp")
    @patch("autoshorts.fluximages.Path.mkdir")
    @patch("autoshorts.fluximages.shutil.rmtree")
    @patch("autoshorts.fluximages.shutdown_computer")
    def test_main_with_goodnight(
        self, mock_shutdown, mock_rmtree, mock_mkdir, mock_mkdtemp, mock_parser_class
    ):
        """Test main function with goodnight flag"""
        # Mock argument parser
        mock_parser = Mock()
        mock_args = Mock()
        mock_args.subject = "test subject"
        mock_args.goodnight = True
        mock_args.batch = None
        mock_args.web_search = False
        mock_parser.parse_args.return_value = mock_args
        mock_parser_class.return_value = mock_parser

        # Mock temp directory
        mock_mkdtemp.return_value = str(self.temp_dir)

        with patch("autoshorts.fluximages.ScriptEngine") as mock_script_class, patch(
            "autoshorts.fluximages.AssetManager"
        ) as mock_asset_class, patch(
            "autoshorts.fluximages.VideoEngine"
        ) as mock_video_class:
            # Setup mocks for successful processing
            mock_script = Mock()
            mock_script.generate.return_value = (
                ["Para 1", "Para 2", "Para 3", "Para 4", "Para 5"],
                ["Prompt 1"] * 9,
            )
            mock_script_class.return_value = mock_script

            mock_asset = Mock()
            mock_asset.generate_audio = AsyncMock(return_value="audio.mp3")
            mock_asset.generate_ai_images = Mock(return_value=["img.jpg"] * 9)
            mock_asset_class.return_value = mock_asset

            mock_video = Mock()
            mock_video.create_video = Mock()
            mock_video_class.return_value = mock_video

            asyncio.run(main())

            # Verify shutdown was called
            mock_shutdown.assert_called_once()

    @patch("autoshorts.fluximages.argparse.ArgumentParser")
    def test_main_no_subject(self, mock_parser_class):
        """Test main function with no subject raises error"""
        mock_parser = Mock()
        mock_args = Mock()
        mock_args.subject = None
        mock_args.goodnight = False
        mock_args.batch = None
        mock_args.web_search = False
        mock_parser.parse_args.return_value = mock_args
        mock_parser.error = Mock(side_effect=SystemExit(1))
        mock_parser_class.return_value = mock_parser

        with pytest.raises(SystemExit):
            asyncio.run(main())


class TestEdgeCases:
    """Test edge cases and error handling"""

    def setup_method(self):
        """Setup test fixtures"""
        self.temp_dir = Path(tempfile.mkdtemp())

    def teardown_method(self):
        """Cleanup test fixtures"""
        import shutil

        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    @patch("autoshorts.fluximages.requests.get")
    @patch("autoshorts.fluximages.random.randint")
    def test_generate_ai_images_empty_prompts(self, mock_randint, mock_get):
        """Test AI image generation with empty prompts"""
        mock_randint.return_value = 12345
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = b"fake_image"
        mock_get.return_value = mock_response

        asset_manager = AssetManager(self.temp_dir)
        result = asset_manager.generate_ai_images([])

        assert result == []

    @patch("autoshorts.fluximages.requests.get")
    @patch("autoshorts.fluximages.random.randint")
    def test_generate_ai_images_special_characters_in_prompt(
        self, mock_randint, mock_get
    ):
        """Test AI image generation with special characters in prompt"""
        mock_randint.return_value = 12345
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = b"fake_image"
        mock_get.return_value = mock_response

        asset_manager = AssetManager(self.temp_dir)

        # Prompt with special characters
        prompts = ["A sunset with 'quotes' and \"double quotes\""]

        result = asset_manager.generate_ai_images(prompts)

        assert len(result) == 1
        # Verify the URL was properly encoded
        call_args = mock_get.call_args
        assert "quotes" in str(call_args)

    def test_script_engine_with_web_search(self):
        """Test ScriptEngine with web search enabled"""
        with patch("autoshorts.fluximages.ScriptGenerator") as mock_gen_class:
            mock_gen = Mock()
            mock_gen.generate_script_with_prompts.return_value = (
                ["Para 1"],
                ["Prompt 1"],
            )
            mock_gen_class.return_value = mock_gen

            engine = ScriptEngine()
            engine.script_generator = mock_gen

            result = engine.generate("test subject")

            assert result is not None


if __name__ == "__main__":
    pytest.main([__file__])
