"""Query optimization is optional; failures must not abort footage flow (#9)."""

from unittest.mock import Mock, patch

import pytest
import requests

from autoshorts.modules.video_background import VideoBackgroundManager


@pytest.mark.parametrize(
    "payload",
    [
        {"choices": [{"message": {"content": None}}]},
        {"choices": []},
        {"choices": [{"message": {}}]},
        {"choices": [{"message": {"content": "short"}}]},
        {"choices": [{"message": {"content": ["not text"]}}]},
    ],
)
def test_invalid_model_query_uses_unchanged_subject(payload):
    response = Mock()
    response.json.return_value = payload
    with patch("autoshorts.modules.video_background.requests.post", return_value=response):
        assert VideoBackgroundManager().generate_search_query("História do futebol") == "História do futebol"


@pytest.mark.parametrize("error", [requests.Timeout("timeout"), requests.HTTPError("503")])
def test_transport_errors_fall_back_without_retrying_model(error):
    with patch("autoshorts.modules.video_background.requests.post", side_effect=error) as post:
        assert VideoBackgroundManager().generate_search_query("futebol") == "futebol"
    post.assert_called_once()


def test_fallback_log_does_not_include_external_exception_body():
    with (
        patch("autoshorts.modules.video_background.requests.post", side_effect=ValueError("private response body")),
        patch("autoshorts.modules.video_background.log") as log,
    ):
        assert VideoBackgroundManager().generate_search_query("futebol") == "futebol"
    warnings = [c for c in log.call_args_list if len(c.args) > 1 and c.args[1] == "WARNING"]
    assert warnings
    assert all("private response body" not in c.args[0] for c in log.call_args_list)


def test_null_query_continues_through_real_search_orchestration():
    response = Mock()
    response.json.return_value = {"choices": [{"message": {"content": None}}]}
    manager = VideoBackgroundManager()
    with (
        patch("autoshorts.modules.video_background.requests.post", return_value=response),
        patch.object(manager, "_search_with_ddg", return_value="/fixture/video.mp4") as ddg,
        patch.object(manager, "_search_with_ytdlp") as ytdlp,
    ):
        assert manager.search_and_download("História do futebol") == "/fixture/video.mp4"
    ddg.assert_called_once_with("História do futebol", "História do futebol")
    ytdlp.assert_not_called()
