from unittest.mock import patch

import pytest

from autoshorts.modules.product import network
from autoshorts.modules.product.errors import ProviderError, SourceBlockedError


class _StreamResponse:
    def __init__(self, chunks, status_code=200, headers=None):
        self.status_code = status_code
        self.headers = headers or {"content-type": "application/octet-stream"}
        self._chunks = chunks

    def iter_content(self, size):
        return iter(self._chunks)

    @property
    def content(self):
        raise AssertionError("streamed body must never be materialized via .content")


class _Session:
    trust_env = False

    def __init__(self, response):
        self._response = response

    def request(self, method, url, allow_redirects=False, timeout=None, **kwargs):
        return self._response


def test_streamed_response_never_materialized_for_challenge_check():
    resp = _StreamResponse([b"x" * 1000])
    out = network.safe_request(
        _Session(resp), "GET", "https://example.com/big", operation="t", stream=True
    )
    assert out is resp


def test_safe_get_enforces_byte_cap():
    big = _StreamResponse([b"y" * (1024 * 1024)] * 5)
    with patch.object(network, "safe_request", return_value=big):
        with pytest.raises(ProviderError, match="byte limit"):
            network.safe_get(
                "https://example.com/f", operation="t", max_bytes=4 * 1024 * 1024
            )


def test_safe_get_scans_capped_bytes_for_challenge_markers():
    marker = next(iter(network._CAPTCHA_MARKERS)).encode()
    challenge = _StreamResponse(
        [b"<html>please verify " + marker + b"</html>"],
        headers={"content-type": "text/html"},
    )
    with patch.object(network, "safe_request", return_value=challenge):
        with pytest.raises(SourceBlockedError, match="CAPTCHA"):
            network.safe_get(
                "https://example.com/", operation="t", max_bytes=4 * 1024 * 1024
            )


def test_redirect_without_location_fails_closed():
    resp = _StreamResponse([], status_code=302, headers={})
    with pytest.raises(SourceBlockedError, match="without Location"):
        network.safe_request(
            _Session(resp), "GET", "https://example.com/r", operation="t"
        )
