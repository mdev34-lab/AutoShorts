"""
Test coverage for experimental.py module
"""

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
    MAX_ZOOM_FACTOR,
)


class TestExperimentalConstants:
    """Test experimental mode constants"""

    def test_overlay_duration(self):
        """Test overlay duration constant"""
        assert IMAGE_OVERLAY_DURATION == 1.0

    def test_bounce_interval(self):
        """Test bounce interval constant"""
        assert IMAGE_BOUNCE_INTERVAL == 5.0

    def test_fade_times(self):
        """Test fade time constants"""
        assert IMAGE_FADE_IN_TIME == 0.2
        assert IMAGE_FADE_OUT_TIME == 0.2

    def test_max_zoom_factor(self):
        """Test max zoom factor constant"""
        assert MAX_ZOOM_FACTOR == 1.05


class TestExperimentalYouTubeProcessor:
    """Test ExperimentalYouTubeProcessor class"""

    def setup_method(self):
        """Setup test fixtures"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.processor = ExperimentalYouTubeProcessor()

    def teardown_method(self):
        """Cleanup test fixtures"""
        import shutil

        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_init(self):
        """Test processor initialization"""
        assert hasattr(self.processor, "script_generator")
        assert hasattr(self.processor, "tts_system")
        assert hasattr(self.processor, "subtitle_system")

    @patch("autoshorts.experimental.yt_dlp.YoutubeDL")
    def test_download_youtube_video_success(self, mock_ydl_class):
        """Test successful YouTube video download"""
        mock_ydl = Mock()
        mock_ydl.extract_info.return_value = {
            "title": "Test Video",
            "duration": 300,
            "formats": [{"url": "https://example.com/video.mp4"}],
        }
        mock_ydl_class.return_value.__enter__.return_value = mock_ydl

        result = self.processor._download_youtube_video(
            "https://youtube.com/watch?v=test"
        )

        assert isinstance(result, dict)
        assert result["title"] == "Test Video"
        assert result["duration"] == 300

    @patch("autoshorts.experimental.yt_dlp.YoutubeDL")
    def test_download_youtube_video_failure(self, mock_ydl_class):
        """Test YouTube video download failure"""
        mock_ydl = Mock()
        mock_ydl.extract_info.side_effect = Exception("Download failed")
        mock_ydl_class.return_value.__enter__.return_value = mock_ydl

        with pytest.raises(Exception):
            self.processor._download_youtube_video("https://youtube.com/watch?v=test")

    @patch("autoshorts.experimental.requests.get")
    def test_generate_ai_image_success(self, mock_get):
        """Test successful AI image generation"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = b"fake_image_data"
        mock_get.return_value = mock_response

        result = self.processor._generate_ai_image("test prompt")

        assert result is not None
        assert result.endswith(".jpg")

    @patch("autoshorts.experimental.requests.get")
    def test_generate_ai_image_failure(self, mock_get):
        """Test AI image generation failure"""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_get.return_value = mock_response

        with pytest.raises(Exception):
            self.processor._generate_ai_image("test prompt")

    @patch("autoshorts.experimental.VideoFileClip")
    @patch("autoshorts.experimental.AudioFileClip")
    @patch("autoshorts.experimental.CompositeVideoClip")
    def test_create_video_with_overlays(
        self, mock_composite, mock_audio_clip, mock_video_clip
    ):
        """Test video creation with overlays"""
        # Mock video clip
        mock_video = Mock()
        mock_video.duration = 30.0
        mock_video.close = Mock()
        mock_video_clip.return_value = mock_video

        # Mock audio clip
        mock_audio = Mock()
        mock_audio.duration = 30.0
        mock_audio.close = Mock()
        mock_audio_clip.return_value = mock_audio

        # Mock composite
        mock_final = Mock()
        mock_final.write_videofile = Mock()
        mock_final.close = Mock()
        mock_composite.return_value = mock_final

        # Create temp files
        img_paths = []
        for i in range(3):
            img_path = self.temp_dir / f"ai_img_{i}.jpg"
            img_path.write_bytes(b"fake_image")
            img_paths.append(str(img_path))

        audio_path = self.temp_dir / "audio.mp3"
        audio_path.write_bytes(b"fake_audio")

        output_path = self.temp_dir / "output.mp4"

        self.processor._create_video_with_overlays(
            img_paths, str(audio_path), str(output_path)
        )

        # Verify video creation was attempted
        mock_composite.assert_called_once()
        mock_final.write_videofile.assert_called_once_with(
            str(output_path),
            fps=30,
            codec="libx264",
            audio_codec="aac",
            preset="medium",
            crf=23,
            threads=4,
        )

    def test_calculate_image_timing(self):
        """Test image timing calculations"""
        video_duration = 30.0
        image_count = 3

        timings = self.processor._calculate_image_timing(video_duration, image_count)

        assert len(timings) == image_count
        # Verify timings are within video duration
        for start, duration in timings:
            assert 0 <= start <= video_duration
            assert duration == IMAGE_OVERLAY_DURATION

    def test_apply_bounce_effect(self):
        """Test bounce effect application"""

        # Create a mock image clip
        mock_clip = Mock()
        mock_clip.with_effects = Mock(return_value=mock_clip)
        mock_clip.with_duration = Mock(return_value=mock_clip)

        result = self.processor._apply_bounce_effect(mock_clip, 5.0)

        # Verify effect was applied
        assert result is not None
        mock_clip.with_duration.assert_called_with(5.0)

    @patch("autoshorts.experimental.SubtitleSystem")
    def test_render_subtitles(self, mock_subtitle_class):
        """Test subtitle rendering"""
        mock_subtitle = Mock()
        mock_subtitle.render_subtitles.return_value = [Mock(), Mock()]
        mock_subtitle_class.return_value = mock_subtitle

        result = self.processor._render_subtitles("test.vtt", (1920, 1080))

        assert isinstance(result, list)
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_process_experimental_video(self):
        """Test main experimental video processing workflow"""
        with patch.object(
            self.processor, "_download_youtube_video"
        ) as mock_download, patch.object(
            self.processor, "script_generator"
        ) as mock_script_gen, patch.object(
            self.processor, "tts_system"
        ) as mock_tts, patch.object(
            self.processor, "_generate_ai_image"
        ) as mock_gen_image, patch.object(
            self.processor, "_create_video_with_overlays"
        ) as mock_create_video:
            # Setup mocks
            mock_download.return_value = {"title": "Test Video", "duration": 60}
            mock_script_gen.generate_script_with_prompts.return_value = (
                ["Paragraph 1", "Paragraph 2"],
                ["Prompt 1", "Prompt 2"],
            )
            mock_tts.generate_audio_only = AsyncMock(return_value="audio.mp3")
            mock_gen_image.side_effect = ["img1.jpg", "img2.jpg"]
            mock_create_video.return_value = "output.mp4"

            result = await self.processor.process_experimental_video(
                "test subject", "output.mp4"
            )

            assert result == "output.mp4"
            mock_download.assert_called_once()
            mock_script_gen.generate_script_with_prompts.assert_called_once_with(
                "test subject"
            )
            assert mock_tts.generate_audio_only.call_count == 1
            assert mock_gen_image.call_count == 2
            mock_create_video.assert_called_once()
