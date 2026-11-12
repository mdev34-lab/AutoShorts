"""Fake-only coverage for the filesystem RPC client and request executor.

No network, thread or supervisor: a fake producer answers requests from inside
the client's injected ``sleep``, and a fake ``public_http.py`` transport file
(enforcing the actual ``(host, prefix)`` allowlist contract) stands in for the
real bounded client. One bounded test child proves a FIFO response cannot block.
"""

from __future__ import annotations

import ast
import base64
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parents[1] / "src" / "autoshorts"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


rpc = _load("resume_rpc_under_test", _SRC / "resume_rpc.py")
ROOT = "https://duckduckgo.com/"
IMAGES = "https://duckduckgo.com/i.js"
IMAGE_PARAMS = {
    "o": "json",
    "q": "Flamengo",
    "l": "br-pt",
    "vqd": "4-123",
    "p": "1",
    "ct": "AT",
}
PHOTO = "https://upload.wikimedia.org/wikipedia/commons/a/ab/photo.jpg"
ALLOW = ["https://upload.wikimedia.org/wikipedia/commons/"]
ALLOW_ENTRIES = [["upload.wikimedia.org", "/wikipedia/commons/"]]  # JSON-logged tuples
BODY = b"abc"


def _ok(request, body=BODY):
    return {
        "id": request["id"],
        "op": request["op"],
        "ok": True,
        "size": len(body),
        "sha256": hashlib.sha256(body).hexdigest(),
        "body": base64.b64encode(body).decode("ascii"),
    }


class Producer:
    """Fake supervisor driven by the client's injected clock and sleep."""

    def __init__(self, directory, respond):
        self.directory = Path(directory)
        self.respond = respond
        self.now = 0.0
        self.sleeps = 0
        self.requests = []

    def clock(self):
        return self.now

    def sleep(self, seconds):
        assert seconds == rpc.POLL_SECONDS
        self.sleeps += 1
        self.now += seconds
        request = self.directory / rpc.REQUEST_NAME
        response = self.directory / rpc.RESPONSE_NAME
        if self.respond is None or not request.exists() or os.path.lexists(response):
            return
        parsed = rpc.validate_request(request.read_bytes())
        self.requests.append(parsed)
        answer = self.respond(parsed)
        if callable(answer):
            answer(response)
            return
        raw = (
            answer if isinstance(answer, bytes) else json.dumps(answer).encode("utf-8")
        )
        temp = self.directory / ".response.tmp"
        temp.write_bytes(raw)
        os.replace(temp, response)


@pytest.fixture
def directory(tmp_path):
    path = tmp_path.resolve() / "rpc"
    path.mkdir()
    os.chmod(path, 0o700)
    return str(path)


@pytest.fixture
def make(directory):
    def build(respond=lambda request: _ok(request), wait=40.0):
        producer = Producer(directory, respond)
        client = rpc.FileRPC(
            directory, wait_seconds=wait, clock=producer.clock, sleep=producer.sleep
        )
        return client, producer

    return build


def _root(client, **kwargs):
    options = {"max_bytes": 3, "timeout": 10.0, "max_redirects": 0, **kwargs}
    return client.get_ddg(ROOT, {"q": "Flamengo"}, **options)


def _empty(directory):
    return sorted(os.listdir(directory)) == []


# --- client roundtrips ----------------------------------------------------


def test_ddg_roundtrip_publishes_exact_request_and_cleans_up(make, directory):
    client, producer = make(lambda request: _ok(request, b"<html>ok</html>"))
    body = client.get_ddg(
        ROOT, {"q": "Flamengo"}, max_bytes=262_144, timeout=10.0, max_redirects=0
    )
    assert body == b"<html>ok</html>"
    assert producer.requests == [
        {
            "version": 1,
            "id": 1,
            "op": "ddg",
            "url": ROOT,
            "params": {"q": "Flamengo"},
            "max_bytes": 262_144,
            "timeout": 10.0,
            "max_redirects": 0,
        }
    ]
    assert _empty(directory)


def test_ddg_defaults_use_route_caps(make):
    client, producer = make()
    client.get_ddg(IMAGES, dict(IMAGE_PARAMS))
    assert producer.requests[0]["max_bytes"] == 1_048_576
    assert producer.requests[0]["timeout"] == 10.0
    assert producer.requests[0]["max_redirects"] == 0


def test_ids_are_sequential_and_one_request_at_a_time(make, directory):
    client, producer = make()
    for _ in range(3):
        assert _root(client) == BODY
        assert _empty(directory)
    assert [request["id"] for request in producer.requests] == [1, 2, 3]


def test_photo_request_carries_no_allowlist(make):
    client, producer = make(lambda request: _ok(request, b"\xff\xd8jpeg"))
    assert (
        client.get_photo(PHOTO, max_bytes=1000, timeout=15, max_redirects=3)
        == b"\xff\xd8jpeg"
    )
    request = producer.requests[0]
    assert request == {
        "version": 1,
        "id": 1,
        "op": "photo",
        "url": PHOTO,
        "max_bytes": 1000,
        "timeout": 15.0,
        "max_redirects": 3,
    }
    assert "allowlist" not in request


@pytest.mark.parametrize("allowlist", [ALLOW, [], ["https://evil.example/"], ()])
def test_caller_cannot_supply_photo_allowlist(make, directory, allowlist):
    client, producer = make()
    with pytest.raises(rpc.RPCRequestError, match="authoritative"):
        client.get_photo(PHOTO, allowlist)
    assert producer.requests == [] and _empty(directory)


def test_fixed_ddg_search_adapter_works_over_file_rpc(make):
    ddg = _load("resume_ddg_under_test", _SRC / "modules" / "resume_ddg.py")
    results = {
        "results": [
            {
                "image": "https://img.example/a.jpg",
                "title": "Flamengo",
                "url": "https://p.example/a",
                "height": 800,
                "width": 1200,
            }
        ]
    }

    def respond(request):
        if request["url"] == ROOT:
            return _ok(request, b'<html><script>vqd="4-123"</script></html>')
        assert request["params"]["vqd"] == "4-123"
        return _ok(request, json.dumps(results).encode("utf-8"))

    client, producer = make(respond)
    assert ddg.FixedDDGSearch(client.get_ddg)("Flamengo") == results["results"]
    assert [(r["url"], r["max_bytes"]) for r in producer.requests] == [
        (ROOT, 262_144),
        (IMAGES, 1_048_576),
    ]


def test_unknown_keywords_are_not_accepted(make):
    client, _ = make()
    with pytest.raises(TypeError):
        client.get_ddg(ROOT, {"q": "x"}, headers={"Authorization": "x"})
    with pytest.raises(TypeError):
        client.get_photo(PHOTO, method="POST")
    with pytest.raises(TypeError):
        client.get_ddg(ROOT, {"q": "x"}, 3)


# --- client-side request validation ---------------------------------------


def _params(**changes):
    params = dict(IMAGE_PARAMS)
    for key, value in changes.items():
        if value is None:
            params.pop(key)
        else:
            params[key] = value
    return params


@pytest.mark.parametrize(
    "url, params, kwargs",
    [
        ("http://duckduckgo.com/", {"q": "x"}, {}),
        ("https://duckduckgo.com", {"q": "x"}, {}),
        ("https://duckduckgo.com/i.js?o=json", _params(), {}),
        ("https://duckduckgo.com/d.js", {"q": "x"}, {}),
        ("https://html.duckduckgo.com/html/", {"q": "x"}, {}),
        ("https://example.com/", {"q": "x"}, {}),
        (["https://duckduckgo.com/"], {"q": "x"}, {}),
        (ROOT, None, {}),
        (ROOT, {}, {}),
        (ROOT, {"q": "x", "kl": "us-en"}, {}),
        (ROOT, [("q", "x")], {}),
        (ROOT, {"q": ""}, {}),
        (ROOT, {"q": "   "}, {}),
        (ROOT, {"q": "x" * 2001}, {}),
        (ROOT, {"q": "a\x00b"}, {}),
        (ROOT, {"q": "a\nb"}, {}),
        (ROOT, {"q": 5}, {}),
        (IMAGES, _params(o="xml"), {}),
        (IMAGES, _params(l="us-en"), {}),
        (IMAGES, _params(p="2"), {}),
        (IMAGES, _params(ct="US"), {}),
        (IMAGES, _params(p=1), {}),
        (IMAGES, _params(vqd=None), {}),
        (IMAGES, _params(vqd='4"x'), {}),
        (IMAGES, _params(vqd="4 x"), {}),
        (IMAGES, _params(vqd="v" * 257), {}),
        (IMAGES, {**IMAGE_PARAMS, "s": "100"}, {}),
        (ROOT, {"q": "x"}, {"max_bytes": 262_145}),
        (IMAGES, _params(), {"max_bytes": 1_048_577}),
        (ROOT, {"q": "x"}, {"max_bytes": 0}),
        (ROOT, {"q": "x"}, {"max_bytes": True}),
        (ROOT, {"q": "x"}, {"max_bytes": 1.5}),
        (ROOT, {"q": "x"}, {"timeout": 10.01}),
        (ROOT, {"q": "x"}, {"timeout": 0}),
        (ROOT, {"q": "x"}, {"timeout": True}),
        (ROOT, {"q": "x"}, {"timeout": float("nan")}),
        (ROOT, {"q": "x"}, {"timeout": "10"}),
        (ROOT, {"q": "x"}, {"max_redirects": 1}),
        (ROOT, {"q": "x"}, {"max_redirects": False}),
        (ROOT, {"q": "x"}, {"max_redirects": -1}),
    ],
)
def test_invalid_ddg_requests_never_publish_or_consume_ids(
    make, directory, url, params, kwargs
):
    client, producer = make()
    with pytest.raises(rpc.RPCRequestError):
        client.get_ddg(url, params, **kwargs)
    assert _empty(directory) and producer.requests == []
    _root(client)
    assert producer.requests[0]["id"] == 1


@pytest.mark.parametrize(
    "url, kwargs",
    [
        ("http://upload.wikimedia.org/x.jpg", {}),
        ("https://user:pw@upload.wikimedia.org/x.jpg", {}),
        ("https://upload.wikimedia.org:443/x.jpg", {}),
        ("https://Upload.Wikimedia.org/x.jpg", {}),
        ("https://upload.wikimedia.org/x.jpg#frag", {}),
        ("https://upload.wikimedia.org/a/../b.jpg", {}),
        ("https://upload.wikimedia.org/a/./b.jpg", {}),
        ("https://upload.wikimedia.org/a/%2e%2e/b.jpg", {}),
        ("https://upload.wikimedia.org/a%2Fb.jpg", {}),
        ("https://upload.wikimedia.org/a\\b.jpg", {}),
        ("https://upload.wikimedia.org//b.jpg", {}),
        ("https://upload.wikimedia.org/a b.jpg", {}),
        ("https://upload.wikimedia.org/ä.jpg", {}),
        ("https://127.0.0.1/x.jpg", {}),
        ("https://[::1]/x.jpg", {}),
        ("https://localhost/x.jpg", {}),
        ("https://*.wikimedia.org/x.jpg", {}),
        ("https:///x.jpg", {}),
        ("", {}),
        (None, {}),
        ("https://upload.wikimedia.org/" + "a" * 2048, {}),
        (PHOTO, {"max_bytes": 10 * 1024 * 1024 + 1}),
        (PHOTO, {"max_bytes": 0}),
        (PHOTO, {"timeout": 15.01}),
        (PHOTO, {"timeout": False}),
        (PHOTO, {"max_redirects": 4}),
        (PHOTO, {"max_redirects": True}),
    ],
)
def test_invalid_photo_requests_never_publish(make, directory, url, kwargs):
    client, producer = make()
    with pytest.raises(rpc.RPCRequestError):
        client.get_photo(url, **kwargs)
    assert _empty(directory) and producer.requests == []


def test_photo_caps_at_limits_are_accepted(make):
    client, producer = make()
    client.get_photo(
        PHOTO + "?width=800", max_bytes=10 * 1024 * 1024, timeout=15, max_redirects=0
    )
    assert producer.requests[0]["max_bytes"] == 10 * 1024 * 1024


def test_request_json_over_16kib_is_rejected(make, directory):
    client, producer = make()
    query = "\U0001f600" * 2000  # within 2000 chars, but escaped JSON exceeds 16 KiB
    with pytest.raises(rpc.RPCRequestError, match="bytes"):
        client.get_ddg(ROOT, {"q": query})
    assert _empty(directory) and producer.requests == []


# --- response validation --------------------------------------------------


def _symlink_to_valid(request):
    def create(response):
        target = response.parent.parent / "elsewhere.json"
        target.write_text(json.dumps(_ok(request)), encoding="utf-8")
        os.symlink(target, response)

    return create


@pytest.mark.parametrize(
    "respond",
    [
        lambda r: {**_ok(r), "id": r["id"] + 1},
        lambda r: {**_ok(r), "id": 0},
        lambda r: {**_ok(r), "id": True},
        lambda r: {**_ok(r), "id": "1"},
        lambda r: {**_ok(r), "op": "photo"},
        lambda r: {**_ok(r), "ok": 1},
        lambda r: {**_ok(r), "extra": 1},
        lambda r: {k: v for k, v in _ok(r).items() if k != "sha256"},
        lambda r: {**_ok(r), "body": "!!!!"},
        lambda r: {**_ok(r, b"ab"), "body": "YWJ="},
        lambda r: {**_ok(r), "body": base64.b64encode(BODY).decode() + "\n"},
        lambda r: {**_ok(r), "body": 123},
        lambda r: {**_ok(r), "size": 4},
        lambda r: {**_ok(r), "size": True},
        lambda r: {**_ok(r, b""), "size": 0},
        lambda r: _ok(r, b"abcd"),
        lambda r: {**_ok(r), "sha256": "0" * 64},
        lambda r: {**_ok(r), "sha256": _ok(r)["sha256"].upper()},
        lambda r: {"id": r["id"], "op": r["op"], "ok": False, "error": ""},
        lambda r: {"id": r["id"], "op": r["op"], "ok": False, "error": "e" * 1025},
        lambda r: {"id": r["id"], "op": r["op"], "ok": False, "error": 503},
        lambda r: {"id": r["id"], "op": r["op"], "ok": False, "error": "x", "size": 1},
        lambda r: b"",
        lambda r: b"{",
        lambda r: b'{"id": NaN}',
        lambda r: b'{"id": 1, "id": 1, "op": "ddg", "ok": false, "error": "x"}',
        lambda r: b"[]",
        lambda r: b"\xff\xfe",
        lambda r: b" " * (4 * 1 + 4096 + 1),
        _symlink_to_valid,
        lambda r: lambda response: response.mkdir(),
    ],
)
def test_bad_responses_fail_closed_and_disable_client(make, respond):
    client, _ = make(respond)
    with pytest.raises(rpc.RPCProtocolError):
        _root(client)
    with pytest.raises(rpc.RPCProtocolError, match="unusable"):
        _root(client)


def test_remote_error_is_raised_and_client_stays_usable(make, directory):
    answers = iter(
        [
            lambda r: {
                "id": r["id"],
                "op": r["op"],
                "ok": False,
                "error": "upstream HTTP 503",
            },
            lambda r: _ok(r),
        ]
    )
    client, producer = make(lambda request: next(answers)(request))
    with pytest.raises(rpc.RPCRemoteError, match="upstream HTTP 503"):
        _root(client)
    assert _empty(directory)
    assert _root(client) == BODY
    assert [request["id"] for request in producer.requests] == [1, 2]


def test_missing_response_times_out_with_bounded_wait(make, directory):
    client, producer = make(respond=None, wait=0.2)
    with pytest.raises(rpc.RPCTimeoutError):
        _root(client)
    assert 4 <= producer.sleeps <= 5
    assert 0.2 <= producer.now < 0.3
    assert _empty(directory), "the unanswered request is withdrawn"
    with pytest.raises(rpc.RPCProtocolError, match="unusable"):
        _root(client)


def test_vanished_request_is_a_protocol_error(make):
    def respond(request):
        def create(response):
            response.write_text(json.dumps(_ok(request)), encoding="utf-8")
            (response.parent / rpc.REQUEST_NAME).unlink()

        return create

    client, _ = make(respond)
    with pytest.raises(rpc.RPCProtocolError, match="file operation failed"):
        _root(client)


FIFO_CHILD = r"""
import importlib.util
import os
import sys

spec = importlib.util.spec_from_file_location("resume_rpc_fifo_child", sys.argv[1])
rpc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rpc)
directory = sys.argv[2]
now = [0.0]


def sleep(seconds):
    now[0] += seconds
    response = os.path.join(directory, rpc.RESPONSE_NAME)
    if os.path.exists(os.path.join(directory, rpc.REQUEST_NAME)) and not os.path.lexists(response):
        os.mkfifo(response, 0o600)  # writerless FIFO: a blocking open() would hang forever


client = rpc.FileRPC(directory, wait_seconds=1.0, clock=lambda: now[0], sleep=sleep)
try:
    client.get_ddg("https://duckduckgo.com/", {"q": "x"}, max_bytes=3)
except rpc.RPCProtocolError as exc:
    print("rejected:", exc)
else:
    print("accepted")
"""


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="POSIX FIFO required")
def test_fifo_response_is_rejected_without_blocking(directory):
    # Separate child with a hard timeout: a blocking open() fails the test instead of hanging it.
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            FIFO_CHILD,
            str(_SRC / "resume_rpc.py"),
            directory,
        ],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.startswith("rejected: Response must be a regular file"), (
        result.stdout
    )


@pytest.mark.parametrize("name", [rpc.REQUEST_NAME, rpc.RESPONSE_NAME, ".request.tmp"])
def test_stale_files_block_publication(make, directory, name):
    stale = Path(directory) / name
    stale.write_text("{}", encoding="utf-8")
    client, producer = make()
    with pytest.raises(rpc.RPCProtocolError, match="Stale"):
        _root(client)
    assert sorted(os.listdir(directory)) == [name]
    assert stale.read_text(encoding="utf-8") == "{}"
    assert producer.requests == []


def test_at_most_256_requests_per_run(make):
    client, producer = make()
    for _ in range(rpc.MAX_REQUESTS):
        _root(client)
    with pytest.raises(rpc.RPCLimitError):
        _root(client)
    assert [r["id"] for r in producer.requests] == list(range(1, 257))


@pytest.mark.parametrize("wait", [0, -1, 40.01, True, float("inf"), "40"])
def test_client_wait_is_bounded(directory, wait):
    with pytest.raises(rpc.RPCRequestError):
        rpc.FileRPC(directory, wait_seconds=wait)


def test_client_directory_must_be_private_canonical_directory(tmp_path):
    base = tmp_path.resolve()
    private = base / "private"
    private.mkdir()
    os.chmod(private, 0o700)
    shared = base / "shared"
    shared.mkdir()
    os.chmod(shared, 0o755)
    regular = base / "file"
    regular.write_text("x", encoding="utf-8")
    link = base / "link"
    os.symlink(private, link)
    rpc.FileRPC(str(private))
    for bad in (
        "rpc",
        str(base / "missing"),
        str(regular),
        str(link),
        str(shared),
        private,
        str(private) + "/",
        str(private / ".." / "private"),
        None,
    ):
        with pytest.raises(rpc.RPCRequestError):
            rpc.FileRPC(bad)


# --- validate_request -----------------------------------------------------


def _request(**changes):
    request = {
        "version": 1,
        "id": 1,
        "op": "ddg",
        "url": IMAGES,
        "params": dict(IMAGE_PARAMS),
        "max_bytes": 1_048_576,
        "timeout": 10,
        "max_redirects": 0,
    }
    for key, value in changes.items():
        if value is ...:
            request.pop(key)
        else:
            request[key] = value
    return request


def _photo_request(**changes):
    request = {
        "version": 1,
        "id": 2,
        "op": "photo",
        "url": PHOTO,
        "max_bytes": 5000,
        "timeout": 15,
        "max_redirects": 3,
    }
    request.update(changes)
    return request


def _raw(request):
    return json.dumps(request).encode("utf-8")


def test_validate_request_normalizes_valid_requests():
    assert rpc.validate_request(_raw(_request())) == {**_request(), "timeout": 10.0}
    assert rpc.validate_request(_raw(_photo_request())) == {
        **_photo_request(),
        "timeout": 15.0,
    }


@pytest.mark.parametrize(
    "raw",
    [
        _raw(_request()).decode("utf-8"),
        b"",
        b" " * (16 * 1024 + 1),
        b"\xff",
        b"[]",
        b'{"version": 1, "version": 1}',
        _raw(_request()).replace(b'"timeout": 10', b'"timeout": NaN'),
        _raw(_request(version=2)),
        _raw(_request(version=True)),
        _raw(_request(id=0)),
        _raw(_request(id=257)),
        _raw(_request(id=True)),
        _raw(_request(id="1")),
        _raw(_request(op="http")),
        _raw(_request(op="exec")),
        _raw(_request(op=None)),
        _raw(_request(params=...)),
        _raw(_request(method="POST")),
        _raw(_request(headers={"Authorization": "Bearer x"})),
        _raw(_request(auth="user:pass")),
        _raw(_request(cookies="a=b")),
        _raw(_request(module="os")),
        _raw(_request(executable="/bin/sh")),
        _raw(_request(env={"API_KEY": "x"})),
        _raw(_request(output_path="/tmp/x")),
        _raw(_request(transport_path="/tmp/public_http.py")),
        _raw(_request(allowlist=ALLOW)),
        _raw(_request(url="https://api.openai.com/v1/images")),
        _raw(_photo_request(allowlist=ALLOW)),
        _raw(_photo_request(params={"q": "x"})),
        _raw(_photo_request(url="https://localhost/x.jpg")),
        _raw(_photo_request(max_bytes=10 * 1024 * 1024 + 1)),
    ],
)
def test_validate_request_rejects_everything_outside_the_schema(raw):
    with pytest.raises(rpc.RPCRequestError):
        rpc.validate_request(raw)


def test_fixed_routes_and_caps():
    assert set(rpc.DDG_ROUTES) == {ROOT, IMAGES}
    assert all(cap <= 1024 * 1024 for cap, _ in rpc.DDG_ROUTES.values())
    assert rpc.PHOTO_MAX_BYTES == 10 * 1024 * 1024
    assert (rpc.DDG_MAX_TIMEOUT, rpc.PHOTO_MAX_TIMEOUT, rpc.PHOTO_MAX_REDIRECTS) == (
        10.0,
        15.0,
        3,
    )
    assert (rpc.MAX_REQUEST_BYTES, rpc.MAX_REQUESTS, rpc.MAX_WAIT_SECONDS) == (
        16384,
        256,
        40.0,
    )
    assert rpc.POLL_SECONDS == 0.05


# --- execute_request with a fake transport file ---------------------------


FAKE_TRANSPORT = """
import json
from pathlib import Path

_LOG = Path(__file__).with_name("calls.jsonl")
RESULT = {result!r}


def _log(entry):
    with _LOG.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(entry) + "\\n")


_log({{"call": "import"}})


def _result():
    if RESULT == "raise":
        raise RuntimeError("upstream refused")
    return RESULT


_contract_validate = validate_photo_allowlist


def validate_photo_allowlist(allowlist):
    _log({{"call": "validate", "allowlist": list(allowlist)}})
    return _contract_validate(allowlist)


class BoundedPublicClient:
    def get_ddg(self, url, params, *, max_bytes, timeout, max_redirects):
        _log({{"call": "ddg", "url": url, "params": params, "max_bytes": max_bytes,
              "timeout": timeout, "max_redirects": max_redirects}})
        return _result()

    def get_photo(self, url, allowlist, *, max_bytes, timeout, max_redirects):
        validated = _contract_validate(allowlist)
        validate_photo_target(url, validated)
        _log({{"call": "photo", "url": url, "allowlist": list(allowlist), "max_bytes": max_bytes,
              "timeout": timeout, "max_redirects": max_redirects}})
        return _result()
"""

# Prepended to FAKE_TRANSPORT. validate_photo_allowlist is copied verbatim from the
# attached actual public_http API; the helpers below it are stand-ins for code not shown.
ALLOWLIST_CONTRACT = r'''from __future__ import annotations

import re
from collections.abc import Sequence
from urllib.parse import urlsplit

_SHARED_STORAGE_HOSTS = ("storage.googleapis.com", "s3.amazonaws.com")
_DNS_LABEL = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?")


class AllowlistDeniedError(ValueError):
    pass


def _is_valid_dns_host(host):
    labels = host.split(".")
    return len(host) <= 253 and len(labels) >= 2 and all(_DNS_LABEL.fullmatch(label) for label in labels)


def validate_photo_allowlist(allowlist: Sequence[tuple[str, str]]) -> list[tuple[str, str]]:
    """Validate and normalize explicit (host, pathprefix) tuples."""
    if not allowlist:
        raise AllowlistDeniedError("Explicit photo allowlist must be supplied and non-empty")

    cleaned: list[tuple[str, str]] = []
    for entry in allowlist:
        if not isinstance(entry, (tuple, list)) or len(entry) != 2:
            raise AllowlistDeniedError(f"Each allowlist entry must be a (host, pathprefix) tuple, got {entry!r}")
        host, prefix = entry
        if not isinstance(host, str) or not isinstance(prefix, str):
            raise AllowlistDeniedError("Host and pathprefix must be strings")
        host_clean = host.strip().lower().rstrip(".")
        prefix_clean = prefix.strip()

        if not host_clean or "*" in host_clean or "?" in host_clean:
            raise AllowlistDeniedError(f"Wildcards and empty hosts forbidden in photo allowlist: {host!r}")
        if not _is_valid_dns_host(host_clean):
            raise AllowlistDeniedError(f"Invalid DNS host in photo allowlist: {host!r}")
        if any(tok in prefix_clean for tok in ("..", "%", "?", "#", "\\")):
            raise AllowlistDeniedError(f"Unsafe characters in photo allowlist prefix: {prefix!r}")
        if not prefix_clean or prefix_clean == "/":
            if host_clean in _SHARED_STORAGE_HOSTS or any(host_clean.endswith("." + s) for s in _SHARED_STORAGE_HOSTS):
                raise AllowlistDeniedError(
                    f"Shared storage host {host_clean!r} requires explicit specific pathprefix, not root '/'"
                )
        if not prefix_clean.startswith("/"):
            prefix_clean = "/" + prefix_clean

        cleaned.append((host_clean, prefix_clean))
    return cleaned


def validate_photo_target(url, validated_allowlist):
    parts = urlsplit(url)
    if not any(parts.hostname == host and parts.path.startswith(prefix) for host, prefix in validated_allowlist):
        raise AllowlistDeniedError(f"Photo target outside allowlist: {url!r}")
'''


@pytest.fixture
def transport(tmp_path):
    path = tmp_path.resolve() / "src" / "autoshorts" / "modules" / "public_http.py"

    def build(result=b"public-bytes", source=None):
        path.parent.mkdir(parents=True, exist_ok=True)
        fake = ALLOWLIST_CONTRACT + FAKE_TRANSPORT.format(result=result)
        path.write_text(source if source is not None else fake, encoding="utf-8")
        return str(path)

    return build


def _calls(transport_path):
    log = Path(transport_path).with_name("calls.jsonl")
    if not log.exists():
        return []
    return [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]


def test_execute_ddg_calls_only_get_ddg_with_fixed_limits(transport):
    path = transport()
    before = {name for name in sys.modules if name.startswith("autoshorts")}
    body = rpc.execute_request(
        _raw(_request()), transport_path=path, photo_allowlist=ALLOW
    )
    assert body == b"public-bytes"
    assert _calls(path) == [
        {"call": "import"},
        {
            "call": "ddg",
            "url": IMAGES,
            "params": IMAGE_PARAMS,
            "max_bytes": 1_048_576,
            "timeout": 10.0,
            "max_redirects": 0,
        },
    ]
    assert {name for name in sys.modules if name.startswith("autoshorts")} == before
    assert "_autoshorts_rpc_public_http" not in sys.modules


def test_execute_photo_uses_authoritative_allowlist(transport):
    path = transport(b"\xff\xd8jpeg")
    body = rpc.execute_request(
        _raw(_photo_request()), transport_path=path, photo_allowlist=tuple(ALLOW)
    )
    assert body == b"\xff\xd8jpeg"
    assert _calls(path) == [
        {"call": "import"},
        {"call": "validate", "allowlist": ALLOW_ENTRIES},
        {
            "call": "photo",
            "url": PHOTO,
            "allowlist": ALLOW_ENTRIES,
            "max_bytes": 5000,
            "timeout": 15.0,
            "max_redirects": 3,
        },
    ]


COUNTEREXAMPLE_URL = "https://photos.example.invalid/club/a.jpg"
COUNTEREXAMPLE_ALLOW = ["https://photos.example.invalid/club/"]


def test_raw_counterexample_passes_parsed_tuples_to_actual_contract(transport):
    path = transport(b"\xff\xd8jpeg")
    raw = json.dumps(
        dict(
            version=1,
            id=1,
            op="photo",
            url=COUNTEREXAMPLE_URL,
            max_bytes=5000,
            timeout=15,
            max_redirects=3,
        )
    ).encode()
    body = rpc.execute_request(
        raw, transport_path=path, photo_allowlist=COUNTEREXAMPLE_ALLOW
    )
    assert body == b"\xff\xd8jpeg"
    entries = [["photos.example.invalid", "/club/"]]
    assert _calls(path) == [
        {"call": "import"},
        {"call": "validate", "allowlist": entries},
        {
            "call": "photo",
            "url": COUNTEREXAMPLE_URL,
            "allowlist": entries,
            "max_bytes": 5000,
            "timeout": 15.0,
            "max_redirects": 3,
        },
    ]


def test_fake_transport_enforces_actual_tuple_contract(transport):
    fake = _load("public_http_contract_fake", transport())
    with pytest.raises(
        fake.AllowlistDeniedError, match=r"must be a \(host, pathprefix\) tuple"
    ):
        fake.validate_photo_allowlist(COUNTEREXAMPLE_ALLOW)
    with pytest.raises(
        fake.AllowlistDeniedError, match=r"must be a \(host, pathprefix\) tuple"
    ):
        fake.BoundedPublicClient().get_photo(
            COUNTEREXAMPLE_URL,
            COUNTEREXAMPLE_ALLOW,
            max_bytes=5000,
            timeout=15.0,
            max_redirects=3,
        )
    entries = [("photos.example.invalid", "/club/")]
    assert fake.validate_photo_allowlist(entries) == entries


def test_actual_contract_rejection_fails_closed(transport):
    path = transport()
    url = "https://upload.wikimedia.org/a%20b/x.jpg"
    with pytest.raises(
        rpc.RPCTransportError, match="AllowlistDeniedError: Unsafe characters"
    ):
        rpc.execute_request(
            _raw(_photo_request(url=url)),
            transport_path=path,
            photo_allowlist=["https://upload.wikimedia.org/a%20b/"],
        )
    assert [call["call"] for call in _calls(path)] == ["import", "validate"]


@pytest.mark.parametrize(
    "url",
    [
        "https://evil.example/wikipedia/commons/a.jpg",
        "https://upload.wikimedia.org/other/a.jpg",
        "https://upload.wikimedia.org/wikipedia/commonsX/a.jpg",
        "https://a.upload.wikimedia.org/wikipedia/commons/a.jpg",
        "https://upload.wikimedia.org.evil.example/wikipedia/commons/a.jpg",
        "https://upload.wikimedia.org/wikipedia/commons",
    ],
)
def test_photo_outside_allowlist_never_loads_transport(transport, url):
    path = transport()
    with pytest.raises(rpc.RPCRequestError, match="allowlist"):
        rpc.execute_request(
            _raw(_photo_request(url=url)), transport_path=path, photo_allowlist=ALLOW
        )
    assert _calls(path) == []


def test_request_cannot_override_allowlist(transport):
    path = transport()
    raw = _raw(
        _photo_request(
            url="https://evil.example/a.jpg", allowlist=["https://evil.example/"]
        )
    )
    with pytest.raises(rpc.RPCRequestError):
        rpc.execute_request(raw, transport_path=path, photo_allowlist=ALLOW)
    assert _calls(path) == []


@pytest.mark.parametrize(
    "allowlist",
    [
        ALLOW[0],
        [],
        None,
        [None],
        ["http://upload.wikimedia.org/wikipedia/commons/"],
        ["https://upload.wikimedia.org/wikipedia/commons"],
        ["https://upload.wikimedia.org/wikipedia/commons/?a=1"],
        ["https://upload.wikimedia.org/wikipedia/commons/?"],
        ["https://1.2.3.4/x/"],
        ["https://*.wikimedia.org/x/"],
        ["https://upload.wikimedia.org:443/x/"],
        ["https://upload.wikimedia.org/a/../"],
        ALLOW + ALLOW,
        [f"https://host{i}.example/p/" for i in range(65)],
    ],
)
def test_invalid_authoritative_allowlist_fails_before_transport(transport, allowlist):
    path = transport()
    with pytest.raises(rpc.RPCRequestError):
        rpc.execute_request(
            _raw(_photo_request()), transport_path=path, photo_allowlist=allowlist
        )
    assert _calls(path) == []


def test_invalid_request_fails_before_transport(transport):
    path = transport()
    with pytest.raises(rpc.RPCRequestError):
        rpc.execute_request(
            _raw(_request(method="POST")), transport_path=path, photo_allowlist=ALLOW
        )
    with pytest.raises(rpc.RPCRequestError):
        rpc.execute_request(_request(), transport_path=path, photo_allowlist=ALLOW)
    assert _calls(path) == []


@pytest.mark.parametrize(
    "result, match",
    [
        (b"", "nonempty"),
        ("text", "nonempty"),
        (b"abcd", "nonempty"),
        ("raise", "upstream refused"),
        (None, "nonempty"),
    ],
)
def test_transport_results_are_checked(transport, result, match):
    path = transport(result)
    raw = _raw(_request(url=ROOT, params={"q": "x"}, max_bytes=3))
    with pytest.raises(rpc.RPCTransportError, match=match):
        rpc.execute_request(raw, transport_path=path, photo_allowlist=ALLOW)


def test_transport_path_must_be_trusted_public_http_file(transport, tmp_path):
    path = transport()
    base = tmp_path.resolve()
    wrong_dir = base / "other" / "modules" / "public_http.py"
    wrong_dir.parent.mkdir(parents=True)
    wrong_dir.write_text(FAKE_TRANSPORT.format(result=b"x"), encoding="utf-8")
    wrong_name = Path(path).with_name("other_http.py")
    wrong_name.write_text("X = 1\n", encoding="utf-8")
    link_dir = base / "link" / "autoshorts" / "modules"
    link_dir.mkdir(parents=True)
    os.symlink(path, link_dir / "public_http.py")
    folder = base / "dir" / "autoshorts" / "modules" / "public_http.py"
    folder.mkdir(parents=True)
    for bad in (
        "src/autoshorts/modules/public_http.py",
        str(wrong_dir),
        str(wrong_name),
        str(link_dir / "public_http.py"),
        str(folder),
        str(base / "missing" / "autoshorts" / "modules" / "public_http.py"),
        Path(path),
        None,
    ):
        with pytest.raises(rpc.RPCRequestError):
            rpc.execute_request(
                _raw(_request()), transport_path=bad, photo_allowlist=ALLOW
            )
    assert _calls(path) == []


@pytest.mark.parametrize(
    "source, raw",
    [
        ("X = 1\n", _raw(_request())),
        ("raise ImportError('broken transport')\n", _raw(_request())),
        ("class BoundedPublicClient:\n    pass\n", _raw(_photo_request())),
    ],
)
def test_incomplete_or_broken_transport_fails_closed(transport, source, raw):
    path = transport(source=source)
    with pytest.raises(rpc.RPCTransportError):
        rpc.execute_request(raw, transport_path=path, photo_allowlist=ALLOW)
    assert "_autoshorts_rpc_public_http" not in sys.modules


# --- static scope ---------------------------------------------------------


def test_module_is_stdlib_only_without_process_thread_or_network():
    source = (_SRC / "resume_rpc.py").read_text("utf-8")
    imported = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imported |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            imported.add((node.module or "").split(".")[0])
    assert imported <= {
        "__future__",
        "base64",
        "hashlib",
        "importlib",
        "json",
        "math",
        "os",
        "stat",
        "sys",
        "time",
        "urllib",
    }
    for forbidden in (
        "subprocess",
        "threading",
        "socket",
        "Popen",
        "urlopen",
        "import requests",
        "run_with_public_fetch",
        "autoshorts.modules",
    ):
        assert forbidden not in source
