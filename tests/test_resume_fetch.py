"""Fake-only coverage for the one-request public fetch helper.

No network, process or thread. The helper is copied beside a fake or the real
``resume_rpc.py`` so its same-directory rule holds; real-RPC tests use a fake
``public_http.py`` transport file. Import/execute sentinel logs prove what ran.
"""

from __future__ import annotations

import ast
import base64
import hashlib
import importlib.util
import json
import os
import stat
import sys
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parents[1] / "src" / "autoshorts"
FETCH = _SRC / "resume_fetch.py"
REAL_RPC = _SRC / "resume_rpc.py"
TRANSPORT_ARG = "/trusted/autoshorts/modules/public_http.py"
ALLOW = ["https://upload.wikimedia.org/wikipedia/commons/"]
PHOTO = "https://upload.wikimedia.org/wikipedia/commons/a/ab/photo.jpg"
FAKE_REQUEST = {"version": 1, "id": 7, "op": "ddg", "max_bytes": 3}
OK = ["body", "id", "ok", "op", "sha256", "size"]
ERR = ["error", "id", "ok", "op"]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


FAKE_RPC = """
import json
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_OK_KEYS = frozenset({ok_keys!r})
_ERROR_KEYS = frozenset({error_keys!r})
RESULT = {result!r}
SQUAT = {squat!r}


def _log(name, entry):
    with (_HERE / name).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(entry) + "\\n")


_log("imported.jsonl", {{"call": "import"}})


def validate_request(raw):
    data = json.loads(raw.decode("utf-8"))
    if type(data) is not dict or set(data) != {{"version", "id", "op", "max_bytes"}} or data["version"] != 1:
        raise ValueError("fake schema violation")
    return data


def execute_request(raw, *, transport_path, photo_allowlist):
    _log("executed.jsonl", {{"raw": raw.decode("utf-8"), "transport_path": transport_path,
                             "photo_allowlist": list(photo_allowlist)}})
    if SQUAT:
        Path(SQUAT).write_text("existing", encoding="utf-8")
    if RESULT == "raise":
        raise RuntimeError("upstream refused " + "x" * 5000)
    return RESULT
"""

FAKE_TRANSPORT = """
import json
from pathlib import Path

_LOG = Path(__file__).with_name("transport.jsonl")
MODE = {mode!r}


def _log(entry):
    with _LOG.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(entry) + "\\n")


_log({{"call": "import"}})


def validate_photo_allowlist(allowlist):
    if not allowlist or any(not isinstance(entry, (tuple, list)) or len(entry) != 2 for entry in allowlist):
        raise ValueError("Each allowlist entry must be a (host, pathprefix) tuple")
    return [tuple(entry) for entry in allowlist]


class BoundedPublicClient:
    def get_ddg(self, url, params, *, max_bytes, timeout, max_redirects):
        _log({{"call": "ddg", "url": url, "params": params}})
        if MODE == "raise":
            raise RuntimeError("upstream HTTP 503")
        return b"<html>ok</html>"

    def get_photo(self, url, allowlist, *, max_bytes, timeout, max_redirects):
        validate_photo_allowlist(allowlist)
        _log({{"call": "photo", "url": url, "allowlist": [list(entry) for entry in allowlist]}})
        return b"\\xff\\xd8jpeg"
"""


class Env:
    """Helper copy in ``bin/`` beside its ``resume_rpc.py``; private RPC dir in ``rpc/``."""

    def __init__(self, base):
        self.base = base
        self.bin = base / "bin"
        self.dir = base / "rpc"
        self.modules = base / "src" / "autoshorts" / "modules"
        self.bin.mkdir()
        self.dir.mkdir()
        os.chmod(self.dir, 0o700)
        (self.bin / "resume_fetch.py").write_bytes(FETCH.read_bytes())
        self.fetch = _load("resume_fetch_under_test", self.bin / "resume_fetch.py")
        self.rpc = str(self.bin / "resume_rpc.py")
        self.request = str(self.dir / "request.json")
        self.response = str(self.dir / "response.json")

    def fake_rpc(self, result=b"abc", squat="", ok_keys=OK, error_keys=ERR):
        source = FAKE_RPC.format(
            result=result, squat=squat, ok_keys=ok_keys, error_keys=error_keys
        )
        Path(self.rpc).write_text(source, encoding="utf-8")

    def real_rpc(self, mode="ok"):
        Path(self.rpc).write_bytes(REAL_RPC.read_bytes())
        self.modules.mkdir(parents=True)
        transport = self.modules / "public_http.py"
        transport.write_text(FAKE_TRANSPORT.format(mode=mode), encoding="utf-8")
        return str(transport)

    def put_request(self, request=FAKE_REQUEST):
        raw = (
            request
            if isinstance(request, bytes)
            else json.dumps(request).encode("utf-8")
        )
        Path(self.request).write_bytes(raw)

    def run(self, **overrides):
        options = {
            "rpc_path": self.rpc,
            "transport_path": TRANSPORT_ARG,
            "photo_allowlist": ALLOW,
            **overrides,
        }
        return self.fetch.fetch_once(self.request, self.response, **options)

    def argv(self, allowlist_json=json.dumps(ALLOW)):
        return [
            "--request",
            self.request,
            "--response",
            self.response,
            "--rpc",
            self.rpc,
            "--transport",
            TRANSPORT_ARG,
            "--allowlist-json",
            allowlist_json,
        ]

    def log(self, name, where=None):
        path = (where or self.bin) / name
        if not path.exists():
            return []
        return [
            json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        ]

    def listing(self):
        return sorted(os.listdir(self.dir))

    def response_json(self):
        return json.loads(Path(self.response).read_text(encoding="utf-8"))

    def untouched(self):
        return self.log("executed.jsonl") == [] and not os.path.lexists(self.response)


@pytest.fixture
def env(tmp_path):
    return Env(tmp_path.resolve())


# --- success and typed error responses ------------------------------------


def test_success_publishes_exact_private_ok_response(env):
    env.fake_rpc(b"abc")
    env.put_request()
    assert env.run() is None
    assert env.response_json() == {
        "id": 7,
        "op": "ddg",
        "ok": True,
        "size": 3,
        "sha256": hashlib.sha256(b"abc").hexdigest(),
        "body": base64.b64encode(b"abc").decode("ascii"),
    }
    assert stat.S_IMODE(os.lstat(env.response).st_mode) == 0o600
    assert os.lstat(env.response).st_nlink == 1
    assert env.listing() == ["request.json", "response.json"], (
        "request kept for the client; temp removed"
    )
    assert env.log("executed.jsonl") == [
        {
            "raw": json.dumps(FAKE_REQUEST),
            "transport_path": TRANSPORT_ARG,
            "photo_allowlist": ALLOW,
        },
    ]
    assert "_autoshorts_fetch_rpc" not in sys.modules


def test_executor_failure_publishes_bounded_error_and_raises(env):
    env.fake_rpc("raise")
    env.put_request()
    with pytest.raises(env.fetch.FetchExecutionError, match="upstream refused"):
        env.run()
    data = env.response_json()
    assert (
        sorted(data) == ERR
        and data["ok"] is False
        and (data["id"], data["op"]) == (7, "ddg")
    )
    assert (
        data["error"].startswith("RuntimeError: upstream refused")
        and len(data["error"]) == 1024
    )
    assert env.listing() == ["request.json", "response.json"]


@pytest.mark.parametrize("result", [b"", "text", None, b"abcd", bytearray(b"abc")])
def test_invalid_executor_result_becomes_error_response(env, result):
    env.fake_rpc(result)
    env.put_request()
    with pytest.raises(env.fetch.FetchExecutionError, match="nonempty bytes"):
        env.run()
    data = env.response_json()
    assert (
        data["ok"] is False and "Executor must return nonempty bytes" in data["error"]
    )


# --- rejected requests publish nothing -------------------------------------


@pytest.mark.parametrize(
    "raw",
    [
        b"",
        b"{",
        b"[]",
        b"\xff",
        json.dumps({**FAKE_REQUEST, "headers": {"Authorization": "x"}}).encode(),
        json.dumps({**FAKE_REQUEST, "transport_path": "/tmp/evil.py"}).encode(),
        json.dumps({**FAKE_REQUEST, "version": 2}).encode(),
        b" " * (16 * 1024 + 1),
    ],
)
def test_malformed_request_never_executes_or_publishes(env, raw):
    env.fake_rpc()
    env.put_request(raw)
    with pytest.raises(env.fetch.FetchRejectedError):
        env.run()
    assert env.listing() == ["request.json"] and env.untouched()


def _symlink_request(env):
    target = env.base / "elsewhere.json"
    target.write_text(json.dumps(FAKE_REQUEST), encoding="utf-8")
    os.symlink(target, env.request)


@pytest.mark.parametrize(
    "make",
    [
        _symlink_request,
        lambda env: os.mkfifo(env.request, 0o600),  # writerless FIFO: must not block
        lambda env: os.mkdir(env.request),
        lambda env: None,
        lambda env: env.put_request(
            json.dumps(FAKE_REQUEST).encode() + b" " * (16 * 1024)
        ),
    ],
)
def test_unsafe_request_files_are_rejected_before_rpc_load(env, make):
    env.fake_rpc()
    make(env)
    with pytest.raises(env.fetch.FetchRejectedError):
        env.run()
    assert env.untouched() and env.log("imported.jsonl") == []


# --- response slot ---------------------------------------------------------


@pytest.mark.parametrize("name", ["response.json", ".response.tmp"])
def test_stale_response_files_block_before_execution(env, name):
    env.fake_rpc()
    env.put_request()
    stale = env.dir / name
    stale.write_text("existing", encoding="utf-8")
    with pytest.raises(env.fetch.FetchPublishError, match="Stale"):
        env.run()
    assert stale.read_text(encoding="utf-8") == "existing"
    assert env.log("executed.jsonl") == []


def test_symlinked_response_slot_is_never_followed(env):
    env.fake_rpc()
    env.put_request()
    target = env.base / "target.json"
    target.write_text("keep", encoding="utf-8")
    os.symlink(target, env.response)
    with pytest.raises(env.fetch.FetchPublishError):
        env.run()
    assert target.read_text(encoding="utf-8") == "keep" and os.path.islink(env.response)
    assert env.log("executed.jsonl") == []


def test_response_appearing_during_execution_is_preserved(env):
    env.fake_rpc(b"abc", squat=env.response)
    env.put_request()
    with pytest.raises(env.fetch.FetchPublishError, match="Cannot publish"):
        env.run()
    assert Path(env.response).read_text(encoding="utf-8") == "existing"
    assert env.listing() == ["request.json", "response.json"], "temp file cleaned up"
    assert len(env.log("executed.jsonl")) == 1


# --- trusted path arguments ------------------------------------------------


def test_request_and_response_paths_are_fixed_private_and_canonical(env):
    env.fake_rpc()
    env.put_request()
    base, folder = env.base, str(env.dir)
    other = base / "other"
    other.mkdir()
    os.chmod(other, 0o700)
    shared = base / "shared"
    shared.mkdir()
    os.chmod(shared, 0o755)
    (shared / "request.json").write_text(json.dumps(FAKE_REQUEST), encoding="utf-8")
    link = base / "link"
    os.symlink(env.dir, link)
    cases = [
        (folder + "/req.json", env.response),
        (env.request, folder + "/resp.json"),
        (env.request, str(other / "response.json")),
        ("rpc/request.json", "rpc/response.json"),
        (folder + "/../rpc/request.json", folder + "/../rpc/response.json"),
        (folder + "/./request.json", folder + "/./response.json"),
        (str(link / "request.json"), str(link / "response.json")),
        (str(shared / "request.json"), str(shared / "response.json")),
        (
            str(base / "missing" / "request.json"),
            str(base / "missing" / "response.json"),
        ),
        (Path(env.request), env.response),
        (None, env.response),
    ]
    for request, response in cases:
        with pytest.raises(env.fetch.FetchRejectedError):
            env.fetch.fetch_once(
                request,
                response,
                rpc_path=env.rpc,
                transport_path=TRANSPORT_ARG,
                photo_allowlist=ALLOW,
            )
    assert env.untouched() and env.log("imported.jsonl") == []
    assert not os.path.lexists(shared / "response.json")


def test_rpc_path_must_be_canonical_resume_rpc_beside_helper(env):
    env.fake_rpc()
    env.put_request()
    elsewhere = env.base / "elsewhere"
    elsewhere.mkdir()
    source = Path(env.rpc).read_text(encoding="utf-8")
    (elsewhere / "resume_rpc.py").write_text(source, encoding="utf-8")
    (env.bin / "other_rpc.py").write_text(source, encoding="utf-8")
    for bad in (
        str(elsewhere / "resume_rpc.py"),
        str(env.bin / "other_rpc.py"),
        "resume_rpc.py",
        env.rpc + "/",
        str(env.bin) + "/./resume_rpc.py",
        Path(env.rpc),
        None,
    ):
        with pytest.raises(env.fetch.FetchRejectedError):
            env.run(rpc_path=bad)
    os.unlink(env.rpc)
    os.symlink(elsewhere / "resume_rpc.py", env.rpc)
    with pytest.raises(env.fetch.FetchRejectedError):
        env.run()
    os.unlink(env.rpc)
    os.mkdir(env.rpc)
    with pytest.raises(env.fetch.FetchRejectedError):
        env.run()
    assert env.untouched()
    assert (
        env.log("imported.jsonl") == [] and env.log("imported.jsonl", elsewhere) == []
    )


@pytest.mark.parametrize("ok_keys, error_keys", [(OK + ["extra"], ERR), (OK, ERR[:-1])])
def test_rpc_module_with_other_protocol_is_rejected(env, ok_keys, error_keys):
    env.fake_rpc(ok_keys=ok_keys, error_keys=error_keys)
    env.put_request()
    with pytest.raises(env.fetch.FetchRejectedError, match="protocol"):
        env.run()
    assert env.untouched()


def test_broken_rpc_module_is_rejected(env):
    Path(env.rpc).write_text("def broken(:\n", encoding="utf-8")
    env.put_request()
    with pytest.raises(env.fetch.FetchRejectedError, match="Cannot load rpc module"):
        env.run()
    assert not os.path.lexists(env.response)
    assert "_autoshorts_fetch_rpc" not in sys.modules


# --- CLI -------------------------------------------------------------------


def test_cli_exit_codes(env):
    env.fake_rpc(b"abc")
    env.put_request()
    assert env.fetch.main(env.argv()) == 0
    assert env.response_json()["ok"] is True
    assert env.fetch.main(env.argv()) == 3, "occupied response slot"
    assert len(env.log("executed.jsonl")) == 1


def test_cli_failed_fetch_exits_1_with_error_response(env):
    env.fake_rpc("raise")
    env.put_request()
    assert env.fetch.main(env.argv()) == 1
    assert env.response_json()["ok"] is False


def test_cli_malformed_request_exits_2_without_response(env):
    env.fake_rpc()
    env.put_request({**FAKE_REQUEST, "version": 2})
    assert env.fetch.main(env.argv()) == 2
    assert env.listing() == ["request.json"] and env.untouched()


@pytest.mark.parametrize(
    "allowlist_json",
    [
        "",
        "{}",
        "[]",
        "[1]",
        '["https://a.example/", null]',
        "NaN",
        "[NaN]",
        '"https://a.example/"',
        "[" * 50_000,
        "[" * 70_000,
    ],
)
def test_cli_rejects_bad_allowlist_json_before_anything(env, allowlist_json):
    env.fake_rpc()
    env.put_request()
    assert env.fetch.main(env.argv(allowlist_json)) == 2
    assert env.untouched() and env.log("imported.jsonl") == []


@pytest.mark.parametrize(
    "mutate",
    [
        lambda argv: argv[:-2],
        lambda argv: [
            "--allowlist" if arg == "--allowlist-json" else arg for arg in argv
        ],
        lambda argv: argv + ["--header", "Authorization: x"],
        lambda argv: argv + ["--url", "https://evil.example/"],
        lambda argv: argv + ["extra"],
    ],
)
def test_cli_accepts_only_fixed_flags(env, mutate):
    env.fake_rpc()
    env.put_request()
    with pytest.raises(SystemExit) as info:
        env.fetch.main(mutate(env.argv()))
    assert info.value.code == 2
    assert env.untouched()


# --- real resume_rpc with a fake transport ---------------------------------


def _client(env, rpc, on_request):
    now = [0.0]

    def sleep(seconds):
        now[0] += seconds
        if os.path.exists(env.request) and not os.path.lexists(env.response):
            on_request()

    return rpc.FileRPC(
        str(env.dir), wait_seconds=1.0, clock=lambda: now[0], sleep=sleep
    )


def test_helper_constants_match_real_rpc_protocol(env):
    env.real_rpc()
    rpc = _load("resume_rpc_copy_under_test", env.rpc)
    assert (rpc.REQUEST_NAME, rpc.RESPONSE_NAME) == (
        env.fetch.REQUEST_NAME,
        env.fetch.RESPONSE_NAME,
    )
    assert (rpc._OK_KEYS, rpc._ERROR_KEYS) == (env.fetch.OK_KEYS, env.fetch.ERROR_KEYS)
    assert rpc.MAX_REQUEST_BYTES == env.fetch.MAX_REQUEST_BYTES
    assert rpc.MAX_ERROR_CHARS == env.fetch.MAX_ERROR_CHARS


def test_real_rpc_roundtrip_through_file_rpc(env):
    transport = env.real_rpc()
    rpc = _load("resume_rpc_copy_under_test", env.rpc)
    before = {name for name in sys.modules if name.startswith("autoshorts")}
    client = _client(env, rpc, lambda: env.run(transport_path=transport))
    assert (
        client.get_ddg("https://duckduckgo.com/", {"q": "Flamengo"})
        == b"<html>ok</html>"
    )
    assert client.get_photo(PHOTO) == b"\xff\xd8jpeg"
    assert env.listing() == []
    assert env.log("transport.jsonl", env.modules) == [
        {"call": "import"},
        {"call": "ddg", "url": "https://duckduckgo.com/", "params": {"q": "Flamengo"}},
        {"call": "import"},
        {
            "call": "photo",
            "url": PHOTO,
            "allowlist": [["upload.wikimedia.org", "/wikipedia/commons/"]],
        },
    ]
    assert {name for name in sys.modules if name.startswith("autoshorts")} == before
    assert "_autoshorts_fetch_rpc" not in sys.modules


def test_real_rpc_transport_error_reaches_client_as_remote_error(env):
    transport = env.real_rpc(mode="raise")
    rpc = _load("resume_rpc_copy_under_test", env.rpc)

    def answer():
        with pytest.raises(env.fetch.FetchExecutionError, match="upstream HTTP 503"):
            env.run(transport_path=transport)

    client = _client(env, rpc, answer)
    with pytest.raises(rpc.RPCRemoteError, match="upstream HTTP 503"):
        client.get_ddg("https://duckduckgo.com/", {"q": "Flamengo"})
    assert env.listing() == []


def test_real_rpc_rejects_out_of_schema_request_before_transport_import(env):
    transport = env.real_rpc()
    env.put_request(
        {
            "version": 1,
            "id": 1,
            "op": "ddg",
            "url": "https://duckduckgo.com/",
            "params": {"q": "x"},
            "max_bytes": 3,
            "timeout": 10,
            "max_redirects": 0,
            "headers": {"Authorization": "x"},
        }
    )
    with pytest.raises(env.fetch.FetchRejectedError, match="Malformed request"):
        env.run(transport_path=transport)
    assert env.listing() == ["request.json"]
    assert env.log("transport.jsonl", env.modules) == []


# --- static scope ----------------------------------------------------------


def test_helper_is_stdlib_only_without_process_thread_network_or_env():
    source = FETCH.read_text("utf-8")
    imported = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imported |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            imported.add((node.module or "").split(".")[0])
    assert imported <= {
        "__future__",
        "argparse",
        "base64",
        "hashlib",
        "importlib",
        "json",
        "os",
        "stat",
        "sys",
    }
    for forbidden in (
        "subprocess",
        "threading",
        "socket",
        "urllib",
        "import requests",
        "Popen",
        "os.system",
        "os.environ",
        "getenv",
        "import autoshorts",
        "from autoshorts",
        "public_http",
        "eval(",
        "pickle",
    ):
        assert forbidden not in source
