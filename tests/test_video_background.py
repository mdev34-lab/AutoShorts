"""
Tests for VideoBackgroundManager (YouTube search, filtering, error handling).
"""

from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from autoshorts.modules.video_background import VideoBackgroundManager


class TestIsSuitableVideo:
    """Test video suitability filtering."""

    def setup_method(self):
        self.manager = VideoBackgroundManager()

    def test_suitable_video(self):
        info = {
            "duration": 120,
            "title": "Great Video",
            "id": "abc123",
            "webpage_url": "https://youtube.com/watch?v=abc123",
        }
        assert self.manager._is_suitable_video(info) is True

    def test_private_video(self):
        info = {
            "duration": 120,
            "title": "Private",
            "id": "abc123",
            "webpage_url": "url",
            "availability": "private",
        }
        assert self.manager._is_suitable_video(info) is False

    def test_unavailable_video(self):
        info = {
            "duration": 120,
            "title": "Unavailable",
            "id": "abc123",
            "webpage_url": "url",
            "availability": "unavailable",
        }
        assert self.manager._is_suitable_video(info) is False

    def test_missing_id(self):
        info = {"duration": 120, "title": "No ID", "webpage_url": "url"}
        assert self.manager._is_suitable_video(info) is False

    def test_missing_url(self):
        info = {"duration": 120, "title": "No URL", "id": "abc123"}
        assert self.manager._is_suitable_video(info) is False

    def test_too_short(self):
        info = {"duration": 10, "title": "Short", "id": "abc123", "webpage_url": "url"}
        assert self.manager._is_suitable_video(info, min_duration=30) is False

    def test_too_long(self):
        info = {"duration": 9999, "title": "Long", "id": "abc123", "webpage_url": "url"}
        assert self.manager._is_suitable_video(info, max_duration=3600) is False

    def test_exactly_min_duration(self):
        info = {"duration": 60, "title": "OK", "id": "abc123", "webpage_url": "url"}
        assert self.manager._is_suitable_video(info, min_duration=60) is True

    def test_exactly_max_duration(self):
        info = {"duration": 600, "title": "OK", "id": "abc123", "webpage_url": "url"}
        assert self.manager._is_suitable_video(info, max_duration=600) is True


class TestExtractErrorMessage:
    """Test error message extraction from yt-dlp exceptions."""

    def setup_method(self):
        self.manager = VideoBackgroundManager()

    def test_exception_with_msg_attr(self):
        exc = Exception()
        exc.msg = "video unavailable"
        assert "unavailable" in self.manager._extract_error_message(exc)

    def test_exception_with_excn_msg(self):
        exc = Exception()
        exc.excn_msg = "private video"
        assert "private" in self.manager._extract_error_message(exc)

    def test_dict_exception_with_msg(self):
        exc = {"msg": "some error", "error": "another"}
        result = self.manager._extract_error_message(exc)
        assert result == "some error"

    def test_dict_exception_fallback_to_error(self):
        exc = {"error": "fallback error"}
        result = self.manager._extract_error_message(exc)
        assert result == "fallback error"

    def test_empty_dict(self):
        exc = {}
        result = self.manager._extract_error_message(exc)
        assert isinstance(result, str)

    def test_list_exception(self):
        exc = [{"msg": "list error"}]
        result = self.manager._extract_error_message(exc)
        assert result == "list error"

    def test_plain_string_fallback(self):
        exc = Exception("plain error")
        result = self.manager._extract_error_message(exc)
        assert result == "plain error"


class TestGenerateSearchQuery:
    """Test AI-powered search query generation."""

    def setup_method(self):
        self.manager = VideoBackgroundManager()

    @patch("autoshorts.modules.video_background.requests.post")
    def test_generate_query_success(self, mock_post):
        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "flamingo documentary explained"}}]
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        query = self.manager.generate_search_query("flamingo")

        assert isinstance(query, str)
        assert len(query) > 10
        assert "documentary" in query.lower()

    @patch("autoshorts.modules.video_background.requests.post")
    def test_generate_query_removes_quotes(self, mock_post):
        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": '"flamingo history documentary"'}}]
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        query = self.manager.generate_search_query("flamingo")
        assert '"' not in query

    @patch("autoshorts.modules.video_background.requests.post")
    def test_generate_query_too_generic_raises(self, mock_post):
        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "short"}}]
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        with pytest.raises(Exception):
            self.manager.generate_search_query("flamingo")

    @patch("autoshorts.modules.video_background.requests.post")
    def test_generate_query_api_error_raises(self, mock_post):
        mock_post.side_effect = Exception("API timeout")

        with pytest.raises(Exception):
            self.manager.generate_search_query("flamingo")


class TestDownloadFromUrl:
    """Test direct URL download."""

    def setup_method(self):
        self.manager = VideoBackgroundManager()
        self.manager.temp_dir = MagicMock()

    @patch("autoshorts.modules.video_background.yt_dlp.YoutubeDL")
    def test_download_from_url_success(self, mock_ydl_class):
        mock_ydl_instance = MagicMock()
        mock_ydl_class.return_value.__enter__.return_value = mock_ydl_instance

        mock_ydl_instance.extract_info.side_effect = [
            {"title": "Test Video", "description": "A test"},
            None,
        ]

        self.manager.temp_dir.glob.return_value = [Path("source_video.mp4")]
        result = self.manager.download_from_url("https://youtube.com/watch?v=test")
        assert isinstance(result, tuple)
        assert len(result) == 3

    @patch("autoshorts.modules.video_background.yt_dlp.YoutubeDL")
    def test_download_from_url_no_file_raises(self, mock_ydl_class):
        mock_ydl_instance = MagicMock()
        mock_ydl_class.return_value.__enter__.return_value = mock_ydl_instance
        mock_ydl_instance.extract_info.return_value = {
            "title": "Test",
            "description": "",
        }

        self.manager.temp_dir.glob.return_value = []
        with pytest.raises(FileNotFoundError):
            self.manager.download_from_url("https://youtube.com/watch?v=test")


class TestSearchAndDownload:
    """Test the search-and-download orchestration."""

    def setup_method(self):
        self.manager = VideoBackgroundManager()

    @patch.object(VideoBackgroundManager, "_search_with_ddg")
    @patch.object(VideoBackgroundManager, "generate_search_query")
    def test_ddg_success_returns_path(self, mock_gen_query, mock_ddg):
        mock_gen_query.return_value = "test query"
        mock_ddg.return_value = "/path/to/video.mp4"

        result = self.manager.search_and_download("test")
        assert result == "/path/to/video.mp4"
        mock_ddg.assert_called_once_with("test query", "test")

    @patch.object(VideoBackgroundManager, "_search_with_ddg")
    @patch.object(VideoBackgroundManager, "_search_with_ytdlp")
    @patch.object(VideoBackgroundManager, "generate_search_query")
    def test_ddg_failure_falls_back_to_ytdlp(
        self, mock_gen_query, mock_ytdlp, mock_ddg
    ):
        mock_gen_query.return_value = "test query"
        mock_ddg.return_value = None
        mock_ytdlp.return_value = "/path/to/video.mp4"

        result = self.manager.search_and_download("test")
        assert result == "/path/to/video.mp4"
        mock_ytdlp.assert_called_once_with("test query", "test")
