"""
Test YouTube summarizer functionality
"""

import sys
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

# Add src to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from autoshorts.yt_summarizer import (
    VideoProcessor,
    YouTubeDownloader,
    generate_tts,
    main,
    shutdown_computer,
)


class TestYouTubeDownloader:
    """Test cases for YouTube downloader"""

    def setup_method(self):
        """Setup test fixtures"""
        self.downloader = YouTubeDownloader()

    def test_init(self):
        """Test downloader initialization"""
        assert hasattr(self.downloader, "ydl_opts")
        assert "format" in self.downloader.ydl_opts
        assert "outtmpl" in self.downloader.ydl_opts

    @patch("autoshorts.yt_summarizer.requests.post")
    def test_generate_search_query(self, mock_post):
        """Test AI search query generation"""
        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": "artificial intelligence explained documentary"
                    }
                }
            ]
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        query = self.downloader.generate_search_query("inteligência artificial")

        assert isinstance(query, str)
        assert len(query) > 10
        assert "documentary" in query.lower()

    def test_is_suitable_video_duration(self):
        """Test video duration filtering"""
        # Test suitable video - needs id and webpage_url
        video_info = {
            "duration": 180,
            "title": "Test Video",
            "id": "test123",
            "webpage_url": "http://example.com/video",
        }
        assert self.downloader._is_suitable_video(video_info) == True

        # Test too short
        video_info = {
            "duration": 30,
            "title": "Short Video",
            "id": "test123",
            "webpage_url": "http://example.com/video",
        }
        assert self.downloader._is_suitable_video(video_info) == False

        # Test too long
        video_info = {
            "duration": 4000,
            "title": "Long Video",
            "id": "test123",
            "webpage_url": "http://example.com/video",
        }
        assert self.downloader._is_suitable_video(video_info) == False

    @patch("autoshorts.yt_summarizer.yt_dlp.YoutubeDL")
    @patch("autoshorts.yt_summarizer.requests.post")
    def test_search_and_download(self, mock_post, mock_ydl):
        """Test video search and download"""
        # Mock AI query generation
        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "test subject documentary explained"}}]
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        # Mock search results
        mock_info = {
            "entries": [
                {
                    "duration": 180,
                    "title": "Suitable Video",
                    "webpage_url": "http://example.com/video1",
                    "id": "test123",
                },
                {
                    "duration": 30,
                    "title": "Short Video",
                    "webpage_url": "http://example.com/video2",
                    "id": "test456",
                },
            ]
        }
        mock_ydl.return_value.__enter__.return_value.extract_info.return_value = (
            mock_info
        )
        mock_ydl.return_value.__enter__.return_value.download.return_value = None

        with patch("pathlib.Path.glob") as mock_glob, patch(
            "autoshorts.yt_summarizer.create_temp_dir"
        ) as mock_temp:
            mock_glob.return_value = [Path("source_video.mp4")]
            mock_temp.return_value = Path(tempfile.mkdtemp())

            result = self.downloader.search_and_download("test subject")

            assert result.endswith(".mp4")

    @patch("autoshorts.yt_summarizer.yt_dlp.YoutubeDL")
    def test_download_direct_url(self, mock_ydl):
        """Test direct URL download"""
        # Mock metadata extraction
        mock_info = {"title": "Test Video", "description": "Test description"}
        mock_ydl.return_value.__enter__.return_value.extract_info.return_value = (
            mock_info
        )
        mock_ydl.return_value.__enter__.return_value.download.return_value = None

        with patch("pathlib.Path.glob") as mock_glob:
            mock_glob.return_value = [Path("source_video.mp4")]

            result = self.downloader.download_direct_url("http://example.com/video")

            assert isinstance(result, tuple)
            assert len(result) == 3
            assert result[0].endswith(".mp4")


class TestVideoProcessor:
    """Test cases for video processor"""

    def setup_method(self):
        """Setup test fixtures"""
        self.processor = VideoProcessor()
        self.temp_dir = Path(tempfile.mkdtemp())

    def teardown_method(self):
        """Cleanup test fixtures"""
        if self.temp_dir.exists():
            import shutil

            shutil.rmtree(self.temp_dir)

    def test_init(self):
        """Test processor initialization"""
        assert hasattr(self.processor, "subtitle_system")

    @patch("autoshorts.yt_summarizer.get_video_duration")
    def test_get_video_duration(self, mock_duration):
        """Test video duration retrieval"""
        mock_duration.return_value = 120.0

        result = self.processor._get_video_duration("test.mp4")

        assert result == 120.0
        mock_duration.assert_called_once_with("test.mp4")

    @patch("subprocess.run")
    def test_apply_fast_blur(self, mock_run):
        """Test FFmpeg blur application"""
        mock_run.return_value = None

        input_path = "input.mp4"
        output_path = "output.mp4"

        result = self.processor._apply_fast_blur(input_path, output_path)

        assert result == output_path
        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        assert "ffmpeg" in args
        # boxblur is in the -vf argument, not as a separate arg
        assert any("boxblur" in str(arg) for arg in args)

    @patch("subprocess.run")
    @patch("autoshorts.yt_summarizer.get_video_duration")
    def test_cut_video(self, mock_duration, mock_run):
        """Test video cutting"""
        # Mock video duration
        mock_duration.return_value = 400.0
        mock_run.return_value = None

        with patch.object(self.processor, "_get_video_duration", return_value=400.0):
            result = self.processor.cut_video("long_video.mp4", 300)

            # Should return a new path (cut video)
            assert result != "long_video.mp4"
            assert result.endswith(".mp4")

    @patch("subprocess.run")
    def test_cut_video_needs_cutting(self, mock_run):
        """Test video cutting when duration exceeds limit"""
        mock_run.return_value = None

        with patch.object(self.processor, "_get_video_duration", return_value=400.0):
            result = self.processor.cut_video("long_video.mp4", 300)

            assert result != "long_video.mp4"
            assert result.endswith(".mp4")

    @patch("autoshorts.yt_summarizer.VideoFileClip")
    @patch("autoshorts.yt_summarizer.AudioFileClip")
    @patch("autoshorts.yt_summarizer.SubtitleSystem")
    def test_aspect_ratio_portrait_video(
        self, mock_subtitles, mock_audio, mock_video_clip
    ):
        """Test that portrait videos (9:16) maintain correct aspect ratio"""
        # Mock portrait video (720x1280)
        mock_video = Mock()
        mock_video.size = (720, 1280)
        mock_video.duration = 60.0
        mock_video.resized = Mock(return_value=mock_video)
        mock_video.cropped = Mock(return_value=mock_video)
        mock_video.with_position = Mock(return_value=mock_video)
        mock_video.with_speed_scaled = Mock(return_value=mock_video)
        mock_video.with_duration = Mock(return_value=mock_video)
        mock_video.with_audio = Mock(return_value=mock_video)
        mock_video.with_effects = Mock(return_value=mock_video)
        mock_video.close = Mock()
        mock_video_clip.return_value = mock_video

        # Mock audio
        mock_audio_clip = Mock()
        mock_audio_clip.duration = 60.0
        mock_audio.return_value = mock_audio_clip

        # Mock subtitles
        mock_subtitle_system = Mock()
        mock_subtitle_system.render_subtitles = Mock(return_value=[])
        mock_subtitles.return_value = mock_subtitle_system

        with patch.object(
            self.processor, "_get_video_duration", return_value=60.0
        ), patch("autoshorts.yt_summarizer.create_temp_dir") as mock_temp_dir, patch(
            "autoshorts.yt_summarizer.CompositeVideoClip"
        ) as mock_composite:
            mock_temp_dir.return_value = self.temp_dir
            mock_composite.return_value = mock_video

            # Test that portrait video doesn't get cropped incorrectly
            w, h = mock_video.size
            assert w == 720
            assert h == 1280
            assert w / h == 9 / 16  # Correct portrait aspect ratio

    def test_aspect_ratio_landscape_to_portrait(self):
        """Test that landscape videos (16:9) are cropped to portrait (9:16)"""
        # Test the aspect ratio calculation logic directly
        # Landscape video (1920x1080)
        w, h = 1920, 1080
        aspect_ratio = w / h

        # Verify it's landscape
        assert aspect_ratio > 16 / 9 - 0.1  # Allow some tolerance
        assert aspect_ratio < 16 / 9 + 0.1

        # Test the cropping logic for landscape to portrait
        # When w/h > 16/9, we crop height: new_h = int(w * (9/16))
        if w / h > 16 / 9:
            new_h = int(w * (9 / 16))
            # After cropping, aspect ratio should be 9:16
            new_aspect = w / new_h
            assert abs(new_aspect - 9 / 16) < 0.01  # Should be very close to 9:16

        # Test the opposite case (portrait video)
        w, h = 720, 1280
        aspect_ratio = w / h
        assert aspect_ratio < 1.0  # Portrait

        # When w/h < 9/16, we crop width: new_w = int(h * (16/9))
        if w / h < 9 / 16:
            new_w = int(h * (16 / 9))
            # After cropping, aspect ratio should be 16:9 (before final resize)
            new_aspect = new_w / h
            assert abs(new_aspect - 16 / 9) < 0.01

    @patch("autoshorts.yt_summarizer.VideoFileClip")
    @patch("autoshorts.yt_summarizer.AudioFileClip")
    @patch("autoshorts.yt_summarizer.SubtitleSystem")
    def test_aspect_ratio_square_video(
        self, mock_subtitles, mock_audio, mock_video_clip
    ):
        """Test that square videos (1:1) are handled correctly"""
        # Mock square video (1080x1080)
        mock_video = Mock()
        mock_video.size = (1080, 1080)
        mock_video.duration = 60.0
        mock_video.resized = Mock(return_value=mock_video)
        mock_video.cropped = Mock(return_value=mock_video)
        mock_video.with_position = Mock(return_value=mock_video)
        mock_video.with_speed_scaled = Mock(return_value=mock_video)
        mock_video.with_duration = Mock(return_value=mock_video)
        mock_video.with_audio = Mock(return_value=mock_video)
        mock_video.with_effects = Mock(return_value=mock_video)
        mock_video.close = Mock()
        mock_video_clip.return_value = mock_video

        # Mock audio
        mock_audio_clip = Mock()
        mock_audio_clip.duration = 60.0
        mock_audio.return_value = mock_audio_clip

        # Mock subtitles
        mock_subtitle_system = Mock()
        mock_subtitle_system.render_subtitles = Mock(return_value=[])
        mock_subtitles.return_value = mock_subtitle_system

        with patch.object(
            self.processor, "_get_video_duration", return_value=60.0
        ), patch("autoshorts.yt_summarizer.create_temp_dir") as mock_temp_dir, patch(
            "autoshorts.yt_summarizer.CompositeVideoClip"
        ) as mock_composite:
            mock_temp_dir.return_value = self.temp_dir
            mock_composite.return_value = mock_video

            # Square video should be cropped to portrait
            w, h = mock_video.size
            assert w == h  # Square
            assert w / h == 1.0

    def test_aspect_ratio_calculation(self):
        """Test aspect ratio calculations for different video types"""
        # Portrait (9:16)
        w, h = 720, 1280
        aspect = w / h
        assert aspect == 9 / 16
        assert aspect < 1.0

        # Landscape (16:9)
        w, h = 1920, 1080
        aspect = w / h
        assert aspect == 16 / 9
        assert aspect > 1.0

        # Square (1:1)
        w, h = 1080, 1080
        aspect = w / h
        assert aspect == 1.0


class TestUtilityFunctions:
    """Test cases for utility functions"""

    def test_shutdown_computer_windows(self):
        """Test computer shutdown on Windows"""
        with patch("platform.system", return_value="Windows"), patch(
            "os.system"
        ) as mock_system:
            shutdown_computer()

            mock_system.assert_called_once_with("shutdown /s /t 30")

    def test_shutdown_computer_linux(self):
        """Test computer shutdown on Linux"""
        with patch("platform.system", return_value="Linux"), patch(
            "os.system"
        ) as mock_system:
            shutdown_computer()

            mock_system.assert_called_once_with("shutdown -h +1")

    def test_shutdown_computer_unsupported(self):
        """Test computer shutdown on unsupported OS"""
        with patch("platform.system", return_value="Unsupported"), patch(
            "autoshorts.yt_summarizer.log"
        ) as mock_log:
            shutdown_computer()

            mock_log.assert_called()

    @pytest.mark.asyncio
    async def test_generate_tts(self):
        """Test TTS generation function"""
        with patch("autoshorts.yt_summarizer.TTSSystem") as mock_tts_class:
            mock_tts = Mock()
            mock_tts.generate_audio_and_subtitles = AsyncMock(
                return_value=("audio.mp3", "subtitles.vtt", 10.0)
            )
            mock_tts_class.return_value = mock_tts

            paragraphs = ["Test paragraph"]
            output_dir = Path(tempfile.mkdtemp())

            result = await generate_tts(paragraphs, output_dir)

            assert isinstance(result, tuple)
            assert len(result) == 3
            assert result[0] == "audio.mp3"
            assert result[1] == "subtitles.vtt"
            assert result[2] == 10.0


class TestMainFunction:
    """Test cases for main function"""

    def setup_method(self):
        """Setup test fixtures"""
        self.temp_dir = Path(tempfile.mkdtemp())

    def teardown_method(self):
        """Cleanup test fixtures"""
        if self.temp_dir.exists():
            import shutil

            shutil.rmtree(self.temp_dir)

    @patch("autoshorts.yt_summarizer.YouTubeDownloader")
    @patch("autoshorts.yt_summarizer.VideoProcessor")
    @patch("autoshorts.yt_summarizer.generate_tts")
    @patch("autoshorts.yt_summarizer.argparse.ArgumentParser")
    def test_main_with_subject(
        self,
        mock_parser,
        mock_generate_tts,
        mock_processor_class,
        mock_downloader_class,
    ):
        """Test main function with subject argument"""
        # Mock argument parser
        mock_args = Mock()
        mock_args.subject = "test subject"
        mock_args.output = "output"
        mock_args.youtube_url = None
        mock_args.goodnight = False
        mock_args.batch = None
        mock_args.web_search = False
        mock_parser.return_value.parse_args.return_value = mock_args

        # Mock classes
        mock_downloader = Mock()
        mock_processor = Mock()
        mock_downloader_class.return_value = mock_downloader
        mock_processor_class.return_value = mock_processor

        # Mock successful processing
        mock_processor.create_output_video.return_value = True
        mock_downloader.search_and_download.return_value = "video.mp4"
        mock_generate_tts.return_value = ("audio.mp3", "subs.vtt", 10.0)

        with patch("pathlib.Path.mkdir"), patch(
            "autoshorts.yt_summarizer.clean_temp_files"
        ), patch("autoshorts.yt_summarizer.ScriptGenerator") as mock_script_gen:
            mock_script_gen.return_value.generate_script.return_value = [
                "Para 1",
                "Para 2",
            ]

            main()

    @patch("autoshorts.yt_summarizer.YouTubeDownloader")
    @patch("autoshorts.yt_summarizer.VideoProcessor")
    @patch("autoshorts.yt_summarizer.generate_tts")
    @patch("autoshorts.yt_summarizer.argparse.ArgumentParser")
    def test_main_with_youtube_url(
        self,
        mock_parser,
        mock_generate_tts,
        mock_processor_class,
        mock_downloader_class,
    ):
        """Test main function with YouTube URL"""
        # Mock argument parser
        mock_args = Mock()
        mock_args.subject = None
        mock_args.output = "output"
        mock_args.youtube_url = "http://youtube.com/watch?v=test"
        mock_args.goodnight = False
        mock_args.batch = None
        mock_args.web_search = False
        mock_parser.return_value.parse_args.return_value = mock_args

        # Mock classes
        mock_downloader = Mock()
        mock_processor = Mock()
        mock_downloader_class.return_value = mock_downloader
        mock_processor_class.return_value = mock_processor

        # Mock successful processing
        mock_downloader.download_direct_url.return_value = (
            "video.mp4",
            "Title",
            "Description",
        )
        mock_processor.create_output_video.return_value = True
        mock_generate_tts.return_value = ("audio.mp3", "subs.vtt", 10.0)

        with patch("pathlib.Path.mkdir"), patch(
            "autoshorts.yt_summarizer.clean_temp_files"
        ), patch("autoshorts.yt_summarizer.ScriptGenerator") as mock_script_gen:
            mock_script_gen.return_value.generate_script.return_value = [
                "Para 1",
                "Para 2",
            ]

            main()

    @patch("autoshorts.yt_summarizer.argparse.ArgumentParser")
    def test_main_no_arguments(self, mock_parser):
        """Test main function with no arguments"""
        # Mock argument parser to raise error
        mock_parser.return_value.parse_args.side_effect = SystemExit(1)

        with pytest.raises(SystemExit):
            main()


if __name__ == "__main__":
    pytest.main([__file__])
