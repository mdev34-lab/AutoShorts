"""Regression tests for the 2026-11-17 silent-hang failure.

A render slept 59 minutes on recv: requests' timeout only bounds connect and
inter-byte reads, so a trickling response never timed out, and block-buffered
stdout plus a SIGTERM kill left a 0-byte log. These tests pin the fixes:
total wall-clock deadline on text API calls and flushed log lines.
"""

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

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
    srv = HTTPServer(("127.0.0.1", 0), handler)
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
            srv.shutdown()

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
            srv.shutdown()


class TestLogFlush:
    def test_log_prints_with_flush(self, monkeypatch):
        from autoshorts.modules import logging_system

        calls = []
        monkeypatch.setattr("builtins.print", lambda *a, **kw: calls.append((a, kw)))
        logging_system.log("hello")
        assert calls and calls[0][1].get("flush") is True
