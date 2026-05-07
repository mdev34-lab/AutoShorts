#!/usr/bin/env python3
"""
Unit tests for AutoShorts experimental mode functionality.
"""

import shutil
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

from autoshorts.experimental import (
    IMAGE_BOUNCE_INTERVAL,
    IMAGE_OVERLAY_DURATION,
    ExperimentalYouTubeProcessor,
)
from autoshorts.modules import (
    IMAGE_FADE_IN_TIME,
    IMAGE_FADE_OUT_TIME,
)


class TestExperimentalYouTubeProcessor:
    """Test suite for ExperimentalYouTubeProcessor class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.processor = ExperimentalYouTubeProcessor()
        self.processor.temp_dir = self.temp_dir

    def teardown_method(self):
        """Clean up test fixtures."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_init(self):
        """Test processor initialization."""
        assert self.processor.script_generator is not None
        assert self.processor.tts_system is not None
        assert self.processor.subtitle_system is not None
        assert self.processor.temp_dir.exists()

    def test_bounce_effect(self):
        """Test bounce effect function."""
        duration = 1.0
        max_zoom = 1.05

        # Test at start (t=0)
        scale_start = self.processor._apply_bounce_effect(Mock(), duration)
        assert scale_start is not None

        # Test that the effect is applied correctly
        mock_clip = Mock()
        mock_clip.with_effects.return_value = mock_clip
        result = self.processor._apply_bounce_effect(mock_clip, duration)
        assert result is not None
        mock_clip.with_effects.assert_called_once()

    @patch("autoshorts.experimental.requests.post")
    def test_generate_search_query(self, mock_post):
        """Test AI search query generation."""
        # Mock API response
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": "artificial intelligence explained documentary"
                    }
                }
            ]
        }
        mock_post.return_value = mock_response

        query = self.processor._generate_search_query("artificial intelligence")

        assert isinstance(query, str)
        assert len(query) > 10
        assert (
            "artificial intelligence" not in query.lower() or len(query) > 30
        )  # Should be more specific
        mock_post.assert_called_once()

    @patch("autoshorts.experimental.yt_dlp.YoutubeDL")
    def test_download_youtube_video(self, mock_ydl):
        """Test YouTube video download."""
        # Mock yt-dlp response
        mock_info = {
            "entries": [
                {
                    "id": "test123",
                    "webpage_url": "https://youtube.com/watch?v=test123",
                    "title": "Test Video",
                    "description": "Test description",
                    "duration": 120,
                    "availability": "public",
                }
            ]
        }

        mock_ydl_instance = Mock()
        mock_ydl_instance.extract_info.return_value = mock_info
        mock_ydl.return_value.__enter__.return_value = mock_ydl_instance

        # Mock download
        with patch.object(
            self.processor, "_generate_search_query", return_value="test query"
        ):
            with patch("autoshorts.experimental.yt_dlp.YoutubeDL") as mock_download:
                mock_download_instance = Mock()
                mock_download.return_value.__enter__.return_value = (
                    mock_download_instance
                )

                # Create a dummy video file
                dummy_video = self.temp_dir / "source_video.mp4"
                dummy_video.touch()

                result = self.processor._download_youtube_video("test subject")

                assert isinstance(result, tuple)
                assert len(result) == 3
                assert isinstance(result[0], str)  # video path
                assert isinstance(result[1], str)  # title
                assert isinstance(result[2], str)  # description

    @patch("autoshorts.experimental.requests.get")
    def test_generate_ai_images(self, mock_get):
        """Test AI image generation."""
        # Mock image generation response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = b"fake_image_data"
        mock_get.return_value = mock_response

        script_paragraphs = ["This is a test paragraph about artificial intelligence."]
        num_images = 3

        with patch("urllib.parse.quote", return_value="safe_prompt"):
            img_paths = self.processor._generate_ai_images(
                "AI", script_paragraphs, num_images
            )

        assert len(img_paths) == num_images
        assert all(Path(path).exists() for path in img_paths)
        assert all(path.endswith(".jpg") for path in img_paths)
        assert mock_get.call_count == num_images

    def test_generate_ai_images_failure(self):
        """Test AI image generation failure handling."""
        # Mock failed response
        with patch("autoshorts.experimental.requests.get") as mock_get:
            mock_response = Mock()
            mock_response.status_code = 500
            mock_get.return_value = mock_response

            with patch("urllib.parse.quote", return_value="safe_prompt"):
                with pytest.raises(Exception, match="Failed to generate image"):
                    self.processor._generate_ai_images("AI", ["test"], 1)

    @patch("autoshorts.experimental.VideoFileClip")
    @patch("autoshorts.experimental.AudioFileClip")
    @patch("autoshorts.experimental.ImageClip")
    def test_create_video_with_overlays(
        self, mock_image_clip, mock_audio_clip, mock_video_clip
    ):
        """Test video creation with overlays."""
        # Mock background video
        mock_bg_video = Mock()
        mock_bg_video.size = (1920, 1080)  # Landscape
        mock_bg_video.duration = 20.0
        mock_bg_video.cropped.return_value = mock_bg_video
        mock_bg_video.resized.return_value = mock_bg_video
        mock_bg_video.subclipped.return_value = mock_bg_video
        mock_video_clip.return_value = mock_bg_video

        # Mock audio
        mock_audio = Mock()
        mock_audio.duration = 15.0
        mock_audio_clip.return_value = mock_audio

        # Mock image clips
        mock_img_clip = Mock()
        mock_img_clip.with_duration.return_value = mock_img_clip
        mock_img_clip.resized.return_value = mock_img_clip
        mock_img_clip.with_position.return_value = mock_img_clip
        mock_img_clip.with_start.return_value = mock_img_clip
        mock_img_clip.with_effects.return_value = mock_img_clip
        mock_image_clip.return_value = mock_img_clip

        # Create dummy image files
        img_paths = []
        for i in range(3):
            img_path = self.temp_dir / f"img_{i}.jpg"
            img_path.touch()
            img_paths.append(str(img_path))

        audio_path = self.temp_dir / "test_audio.mp3"
        audio_path.touch()
        bg_video_path = self.temp_dir / "bg_video.mp4"
        bg_video_path.touch()

        result = self.processor._create_video_with_overlays(
            str(bg_video_path), img_paths, str(audio_path), 15.0
        )

        assert result is not None
        # Should create 3 overlays for 15 seconds (every 5 seconds)
        assert mock_image_clip.call_count == 3
        mock_bg_video.subclipped.assert_called_once_with(0, 15.0)

    def test_create_video_with_overlays_no_images(self):
        """Test video creation with overlays when no images provided."""
        # Create dummy files
        audio_path = self.temp_dir / "test_audio.mp3"
        audio_path.touch()
        bg_video_path = self.temp_dir / "bg_video.mp4"
        bg_video_path.touch()

        # Mock background video
        mock_bg_video = Mock()
        mock_bg_video.size = (1080, 1920)
        mock_bg_video.duration = 10.0
        mock_bg_video.resized.return_value = mock_bg_video
        mock_bg_video.subclipped.return_value = mock_bg_video
        mock_bg_video.with_audio.return_value = mock_bg_video

        with patch("autoshorts.experimental.VideoFileClip", return_value=mock_bg_video):
            with patch("autoshorts.experimental.AudioFileClip"):
                result = self.processor._create_video_with_overlays(
                    str(bg_video_path), [], str(audio_path), 10.0
                )

        assert result is not None
        # Should still work with just background video
        mock_bg_video.with_audio.assert_called_once()

    @patch(
        "autoshorts.experimental.ExperimentalYouTubeProcessor._download_youtube_video"
    )
    @patch("autoshorts.experimental.ExperimentalYouTubeProcessor._generate_ai_images")
    @patch(
        "autoshorts.experimental.ExperimentalYouTubeProcessor._create_video_with_overlays"
    )
    async def test_process_experimental_video_success(
        self, mock_create_video, mock_generate_images, mock_download
    ):
        """Test successful experimental video processing."""
        # Mock YouTube download
        mock_video_path = self.temp_dir / "source_video.mp4"
        mock_video_path.touch()
        mock_download.return_value = (
            str(mock_video_path),
            "Test Title",
            "Test Description",
        )

        # Mock script generation
        mock_script = ["This is a test paragraph.", "Another test paragraph."]
        self.processor.script_generator = Mock()
        self.processor.script_generator.generate_script_from_metadata = Mock(
            return_value=mock_script
        )

        # Mock TTS generation
        mock_audio_path = self.temp_dir / "audio.mp3"
        mock_vtt_path = self.temp_dir / "subtitles.vtt"
        mock_audio_path.touch()
        mock_vtt_path.touch()
        self.processor.tts_system.generate_audio_and_subtitles = AsyncMock(
            return_value=(str(mock_audio_path), str(mock_vtt_path), 10.0)
        )

        # Mock image generation
        img_paths = []
        for i in range(3):
            img_path = self.temp_dir / f"img_{i}.jpg"
            img_path.touch()
            img_paths.append(str(img_path))
        mock_generate_images.return_value = img_paths

        # Mock video creation
        mock_final_video = Mock()
        mock_final_video.write_videofile = Mock()
        mock_final_video.close = Mock()
        mock_create_video.return_value = mock_final_video

        # Mock subtitle rendering
        self.processor.subtitle_system.render_subtitles.return_value = []

        output_path = self.temp_dir / "output.mp4"
        result = await self.processor.process_experimental_video(
            "test subject", str(output_path)
        )

        assert result is True
        mock_download.assert_called_once_with("test subject")
        mock_generate_images.assert_called_once()
        mock_create_video.assert_called_once()
        mock_final_video.write_videofile.assert_called_once()

    @patch(
        "autoshorts.experimental.ExperimentalYouTubeProcessor._download_youtube_video"
    )
    async def test_process_experimental_video_failure(self, mock_download):
        """Test experimental video processing failure."""
        # Mock download failure
        mock_download.side_effect = Exception("Download failed")

        output_path = self.temp_dir / "output.mp4"
        result = await self.processor.process_experimental_video(
            "test subject", str(output_path)
        )

        assert result is False


class TestExperimentalModeIntegration:
    """Integration tests for experimental mode."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())

    def teardown_method(self):
        """Clean up test fixtures."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_image_overlay_constants(self):
        """Test that overlay constants are set correctly."""
        assert IMAGE_OVERLAY_DURATION == 1.0
        assert IMAGE_BOUNCE_INTERVAL == 5.0
        assert IMAGE_FADE_IN_TIME == 0.2
        assert IMAGE_FADE_OUT_TIME == 0.2

    def test_processor_with_real_files(self):
        """Test processor with real temporary files."""
        processor = ExperimentalYouTubeProcessor()
        processor.temp_dir = self.temp_dir

        # Test bounce effect with real parameters
        duration = 1.0
        max_zoom = 1.05

        # Test that bounce effect works
        mock_clip = Mock()
        mock_clip.with_effects.return_value = mock_clip
        result = processor._apply_bounce_effect(mock_clip, duration)

        assert result is not None
        mock_clip.with_effects.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
