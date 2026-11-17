"""Regression tests for the 2026-11-17 silent-hang failure.

A render slept 59 minutes on recv: requests' timeout only bounds connect and
inter-byte reads, so a trickling response never timed out, and block-buffered
stdout plus a SIGTERM kill left a 0-byte log. These tests pin the fixes:
total wall-clock deadline on text API calls and flushed log lines.
"""

import json
import threading
import time
from http.server import BaseHTTPRequestHandler

import pytest

from autoshorts.modules.script_generator import (
    TotalDeadlineExceeded,
    _post_with_deadline,
)


class _TrickleHandler(BaseHTTPRequestHandler):
    """Sends one byte every 0.4s: inter-byte gaps small, total time huge."""

    def do_POST(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        for _ in range(50):
            try:
                self.wfile.write(b" ")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                return
            time.sleep(0.4)

    def log_message(self, *args):
        pass


class _FastHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.dumps({"choices": [{"message": {"content": "ok"}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


def _serve(handler):
    from http.server import ThreadingHTTPServer

    srv = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    srv.daemon_threads = True
    srv.block_on_close = False
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


class TestPostWithDeadline:
    def test_trickling_response_aborted_by_total_deadline(self):
        srv = _serve(_TrickleHandler)
        try:
            url = f"http://127.0.0.1:{srv.server_address[1]}/"
            t0 = time.monotonic()
            with pytest.raises(TotalDeadlineExceeded):
                _post_with_deadline(
                    url,
                    headers={},
                    payload={},
                    read_timeout=5,
                    total_deadline=1,
                )
            elapsed = time.monotonic() - t0
            # deadline + at most one in-flight chunk read; far below the 20s
            # the trickle would take unbounded
            assert elapsed < 8
        finally:
            srv.server_close()

    def test_fast_response_returned_complete(self):
        srv = _serve(_FastHandler)
        try:
            url = f"http://127.0.0.1:{srv.server_address[1]}/"
            body = _post_with_deadline(
                url,
                headers={},
                payload={},
                read_timeout=5,
                total_deadline=10,
            )
            assert json.loads(body)["choices"][0]["message"]["content"] == "ok"
        finally:
            srv.server_close()


class TestLogFlush:
    def test_log_prints_with_flush(self, monkeypatch):
        from autoshorts.modules import logging_system

        calls = []
        monkeypatch.setattr("builtins.print", lambda *a, **kw: calls.append((a, kw)))
        logging_system.log("hello")
        assert calls and calls[0][1].get("flush") is True


class _SilentBodyHandler(BaseHTTPRequestHandler):
    """Sends headers, then goes completely silent (no bytes ever)."""

    def do_POST(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        time.sleep(10)  # outlives the test timeouts

    def log_message(self, *args):
        pass


class _SlowHeadersHandler(BaseHTTPRequestHandler):
    """Accepts the connection but sends nothing - not even headers."""

    def do_POST(self):
        time.sleep(10)

    def log_message(self, *args):
        pass


class TestPostWithDeadlineEdgeCases:
    def test_silent_body_fails_on_inter_byte_read_timeout(self):
        """No bytes mid-body: bounded by the inter-byte read timeout.

        With stream=True, requests surfaces a mid-body read timeout as
        ConnectionError wrapping urllib3's ReadTimeoutError (a ReadTimeout is
        only raised while waiting for the response headers). Both are
        RequestException subclasses, so the existing retry/failure paths
        handle them.
        """
        import requests

        srv = _serve(_SilentBodyHandler)
        try:
            url = f"http://127.0.0.1:{srv.server_address[1]}/"
            t0 = time.monotonic()
            with pytest.raises(requests.exceptions.ConnectionError):
                _post_with_deadline(
                    url, headers={}, payload={}, read_timeout=2, total_deadline=30
                )
            assert time.monotonic() - t0 < 10
        finally:
            srv.server_close()

    def test_slow_headers_fail_on_read_timeout(self):
        """Headers never arrive: bounded by the read timeout on first byte."""
        import requests

        srv = _serve(_SlowHeadersHandler)
        try:
            url = f"http://127.0.0.1:{srv.server_address[1]}/"
            t0 = time.monotonic()
            with pytest.raises(requests.exceptions.ReadTimeout):
                _post_with_deadline(
                    url, headers={}, payload={}, read_timeout=2, total_deadline=30
                )
            assert time.monotonic() - t0 < 10
        finally:
            srv.server_close()

    def test_deadline_overshoot_bounded_by_one_read_timeout(self):
        """A blocked read started just before the deadline may overshoot by at
        most one read_timeout - the documented worst case."""
        srv = _serve(_TrickleHandler)  # one byte every 0.4s
        try:
            url = f"http://127.0.0.1:{srv.server_address[1]}/"
            t0 = time.monotonic()
            with pytest.raises(TotalDeadlineExceeded):
                _post_with_deadline(
                    url, headers={}, payload={}, read_timeout=3, total_deadline=2
                )
            assert time.monotonic() - t0 < 2 + 3 + 1  # deadline + read + slack
        finally:
            srv.server_close()


class TestPostFailurePropagation:
    def test_connect_failure_propagates_unmasked(self, monkeypatch):
        """A requests.post failure must surface as itself, not a NameError
        from a finally-block close() on an unassigned response."""
        import requests as req

        from autoshorts.modules import script_generator

        def boom(*args, **kwargs):
            raise req.exceptions.ConnectTimeout("refused")

        monkeypatch.setattr(script_generator.requests, "post", boom)
        with pytest.raises(req.exceptions.ConnectTimeout):
            _post_with_deadline(
                "http://127.0.0.1:1/",
                headers={},
                payload={},
                read_timeout=2,
                total_deadline=5,
            )
