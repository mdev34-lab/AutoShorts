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


class _CountingStream(_StreamResponse):
    def __init__(self, chunks, **kw):
        super().__init__(chunks, **kw)
        self.pulled = 0

    def iter_content(self, size):
        for chunk in self._chunks:
            self.pulled += 1
            yield chunk


def test_byte_cap_terminates_stream_early():
    # 1 MiB chunks, 4 MiB cap: the read must stop at the 5th chunk, not drain 10.
    big = _CountingStream([b"y" * (1024 * 1024)] * 10)
    with patch.object(network, "safe_request", return_value=big):
        with pytest.raises(ProviderError, match="byte limit"):
            network.safe_get(
                "https://example.com/f", operation="t", max_bytes=4 * 1024 * 1024
            )
    assert big.pulled == 5


def test_challenge_markers_rejected_even_in_octet_stream():
    # safe_request skips the probe for streamed bodies, so safe_get's scan of
    # the capped bytes is the only challenge check - it runs for any MIME type.
    marker = next(iter(network._CAPTCHA_MARKERS)).encode()
    challenge = _StreamResponse(
        [b"\x89binary-prefix " + marker + b" \x00\xff"],
        headers={"content-type": "application/octet-stream"},
    )
    with patch.object(network, "safe_request", return_value=challenge):
        with pytest.raises(SourceBlockedError, match="CAPTCHA"):
            network.safe_get(
                "https://example.com/f", operation="t", max_bytes=4 * 1024 * 1024
            )


def test_fetch_page_evidence_cap_terminates_early():
    from autoshorts.modules.product import provenance

    chunk = b"<html>" + b"z" * (64 * 1024 - 6)  # exactly one 64 KiB read
    page = _CountingStream([chunk] * 100, headers={"content-type": "text/html"})
    with patch.object(provenance, "safe_request", return_value=page):
        with pytest.raises(provenance.MediaProcessingError, match="exceeded 4 MiB"):
            provenance.fetch_page_evidence("https://example.com/p")
    # 4 MiB / 64 KiB = 64 allowed chunks; the 65th read trips the cap.
    assert page.pulled == 65
