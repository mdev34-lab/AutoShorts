"""Coverage for the standalone offline process boundary.

Fake-launch tests prove only the command, environment, validation and result
handling. Mechanical guarantees (namespace, guards, group kill) are asserted
only by tests that run real processes; namespace tests skip when this host
cannot create a nested user+network namespace.
"""

from __future__ import annotations

import ast
import errno
import functools
import importlib.util
import json
import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

_MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "src" / "autoshorts" / "resume_boundary.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "resume_boundary_under_test", _MODULE_PATH
)
rb = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(rb)

PARENT_NS = "net:[4026531840]"
CHILD_NS = "net:[4026532999]"
HOSTILE_ENV = {
    "HOME": "/home/real-user",
    "PATH": "/evil/bin:/usr/bin",
    "HTTP_PROXY": "http://proxy.invalid:3128",
    "HTTPS_PROXY": "http://proxy.invalid:3128",
    "ALL_PROXY": "socks5://proxy.invalid:1080",
    "NO_PROXY": "*",
    "API_KEY": "secret-api-key",
    "OPENAI_API_KEY": "sk-secret",
    "POLLINATIONS_TOKEN": "secret-token",
    "AWS_SECRET_ACCESS_KEY": "secret-aws",
    "STRIPE_BILLING_KEY": "secret-billing",
    "SESSION_COOKIE": "secret-cookie",
    "PYTHONPATH": "/evil/site",
    "PYTHONHOME": "/evil/home",
    "PYTHONSTARTUP": "/evil/startup.py",
    "LD_PRELOAD": "/evil/lib.so",
    "PYTHON_DOTENV_DISABLED": "0",
}
EXPECTED_KEYS = {
    "PATH",
    "LANG",
    "LC_ALL",
    "HOME",
    "TMPDIR",
    "TMP",
    "TEMP",
    "XDG_CACHE_HOME",
    "XDG_CONFIG_HOME",
    "XDG_DATA_HOME",
    "PYTHON_DOTENV_DISABLED",
    "API_KEY",
    "AUTOSHORTS_OFFLINE_BOUNDARY",
    "AUTOSHORTS_BOUNDARY_READY",
    "AUTOSHORTS_BOUNDARY_SOURCE_ROOT",
    "AUTOSHORTS_PARENT_NETNS",
}


def _write(path: Path, source: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(source), encoding="utf-8")
    return path


@pytest.fixture
def layout(tmp_path):
    root = tmp_path.resolve() / "src"
    worker = _write(root / "pkg" / "worker.py", "print('worker')\n")
    return root, worker


# --- fake launch: command, environment, result handling -------------------


class FakeLaunch:
    def __init__(
        self,
        *,
        ready=CHILD_NS,
        returncode=0,
        timed_out=False,
        stdout=b"ok\n",
        stderr=b"",
    ):
        self.ready, self.returncode, self.timed_out = ready, returncode, timed_out
        self.stdout, self.stderr = stdout, stderr
        self.calls = []

    def __call__(self, command, env, cwd, stdout_path, stderr_path, timeout):
        base = Path(stdout_path).parent
        self.calls.append(
            {
                "command": command,
                "env": dict(env),
                "cwd": Path(cwd),
                "timeout": timeout,
                "base": base,
                "bootstrap": Path(command[5]).read_text("utf-8"),
                "dirs": {
                    name: (base / name).is_dir() for name in ("home", "tmp", "work")
                },
                "base_mode": os.stat(base).st_mode & 0o777,
                "modes": {
                    name: os.stat(base / name).st_mode & 0o777
                    for name in ("home", "tmp", "work")
                },
            }
        )
        Path(stdout_path).write_bytes(self.stdout)
        Path(stderr_path).write_bytes(self.stderr)
        if self.ready is not None:
            Path(env["AUTOSHORTS_BOUNDARY_READY"]).write_text(
                self.ready, encoding="ascii"
            )
        return self.timed_out, self.returncode


@pytest.fixture
def fake(monkeypatch):
    def install(**kwargs):
        launch = FakeLaunch(**kwargs)
        monkeypatch.setattr(rb, "_require_namespace_support", lambda: PARENT_NS)
        monkeypatch.setattr(rb, "_launch", launch)
        return launch

    return install


def test_exact_namespace_command_and_private_layout(fake, layout):
    root, worker = layout
    launch = fake()
    result = rb.run_offline(
        worker, ["--manifest", "/data/m.json", ""], timeout=30, source_root=root
    )
    call = launch.calls[0]
    base = call["base"]
    assert call["command"] == [
        "/usr/bin/unshare",
        "-Urn",
        sys.executable,
        "-I",
        "-B",
        str(base / "bootstrap.py"),
        str(worker),
        "--manifest",
        "/data/m.json",
        "",
    ]
    assert call["bootstrap"] == rb.BOOTSTRAP_SOURCE
    assert call["cwd"] == base / "work"
    assert call["timeout"] == 30.0 and type(call["timeout"]) is float
    assert call["dirs"] == {"home": True, "tmp": True, "work": True}
    assert isinstance(result, subprocess.CompletedProcess)
    assert result.returncode == 0 and result.stdout == "ok\n" and result.stderr == ""
    assert not base.exists(), "temporary directory and bootstrap must be removed"


def test_private_layout_is_fixed_canonical_tmp_despite_ambient_tmpdir_symlink(
    fake, layout, tmp_path, monkeypatch
):
    root, worker = layout
    redirected = tmp_path.resolve() / "ambient-target"
    redirected.mkdir()
    ambient = tmp_path.resolve() / "ambient-link"
    ambient.symlink_to(redirected, target_is_directory=True)
    monkeypatch.setenv("TMPDIR", str(ambient))
    monkeypatch.setattr(rb.tempfile, "tempdir", None)
    launch = fake()

    rb.run_offline(worker, [], timeout=5, source_root=root)

    call = launch.calls[0]
    base = call["base"]
    assert base.parent == Path("/tmp")
    assert base == Path(os.path.realpath(base))
    assert call["base_mode"] & 0o077 == 0
    assert all(mode & 0o077 == 0 for mode in call["modes"].values())
    assert call["env"]["HOME"] == str(base / "home")
    assert call["env"]["TMPDIR"] == str(base / "tmp")
    assert call["cwd"] == base / "work"
    assert not base.exists()


def test_environment_is_built_from_scratch_despite_hostile_parent(
    fake, layout, monkeypatch
):
    for key, value in HOSTILE_ENV.items():
        monkeypatch.setenv(key, value)
    root, worker = layout
    launch = fake()
    rb.run_offline(worker, [], timeout=5, source_root=root)
    env, base = launch.calls[0]["env"], launch.calls[0]["base"]
    assert set(env) == EXPECTED_KEYS
    assert env == {
        "PATH": "/usr/bin:/bin",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "HOME": str(base / "home"),
        "TMPDIR": str(base / "tmp"),
        "TMP": str(base / "tmp"),
        "TEMP": str(base / "tmp"),
        "XDG_CACHE_HOME": str(base / "home" / ".cache"),
        "XDG_CONFIG_HOME": str(base / "home" / ".config"),
        "XDG_DATA_HOME": str(base / "home" / ".local" / "share"),
        "PYTHON_DOTENV_DISABLED": "1",
        "API_KEY": "",
        "AUTOSHORTS_OFFLINE_BOUNDARY": "1",
        "AUTOSHORTS_BOUNDARY_READY": str(base / "ready"),
        "AUTOSHORTS_BOUNDARY_SOURCE_ROOT": str(root),
        "AUTOSHORTS_PARENT_NETNS": PARENT_NS,
    }
    joined = json.dumps(env)
    for secret in HOSTILE_ENV.values():
        if secret not in ("*", "0"):
            assert secret not in joined
    assert "AUTOSHORTS_NETWORK_GUARD" not in env, (
        "only the guarded bootstrap sets the marker"
    )


@pytest.mark.parametrize("ready", [None, PARENT_NS, "garbage", ""])
def test_missing_or_invalid_readiness_is_unavailable_never_success(fake, layout, ready):
    root, worker = layout
    launch = fake(ready=ready, returncode=0)
    with pytest.raises(rb.BoundaryUnavailableError, match="not established"):
        rb.run_offline(worker, [], timeout=5, source_root=root)
    assert not launch.calls[0]["base"].exists()


def test_guard_failure_exit_is_unavailable(fake, layout):
    root, worker = layout
    fake(
        ready=None,
        returncode=86,
        stderr=b"AUTOSHORTS_BOUNDARY_GUARD_FAILED: not separate\n",
    )
    with pytest.raises(rb.BoundaryUnavailableError, match="GUARD_FAILED") as caught:
        rb.run_offline(worker, [], timeout=5, source_root=root)
    assert caught.value.returncode == 86


def test_nonzero_exit_is_failure_with_bounded_diagnostics(fake, layout):
    root, worker = layout
    noise = b"x" * 20000 + b"\nTraceback: worker exploded\n"
    fake(returncode=3, stderr=noise)
    with pytest.raises(rb.BoundaryFailureError, match="exit code 3") as caught:
        rb.run_offline(worker, [], timeout=5, source_root=root)
    error = caught.value
    assert error.returncode == 3
    assert len(error.stderr) == rb.DIAGNOSTIC_CHARS
    assert error.stderr.endswith("Traceback: worker exploded\n")
    assert len(str(error)) < rb.DIAGNOSTIC_CHARS + 200


def test_timeout_result_raises_timeout(fake, layout):
    root, worker = layout
    fake(timed_out=True, returncode=-9)
    with pytest.raises(rb.BoundaryTimeoutError, match="process group was killed"):
        rb.run_offline(worker, [], timeout=1.5, source_root=root)


@pytest.mark.parametrize("stream", ["stdout", "stderr"])
def test_output_overflow_is_rejected(fake, layout, stream):
    root, worker = layout
    fake(**{stream: b"y" * (rb.MAX_OUTPUT_BYTES + 1)})
    with pytest.raises(rb.BoundaryOutputError):
        rb.run_offline(worker, [], timeout=5, source_root=root)


def test_output_at_limit_is_returned(fake, layout):
    root, worker = layout
    fake(stdout=b"z" * rb.MAX_OUTPUT_BYTES)
    result = rb.run_offline(worker, [], timeout=5, source_root=root)
    assert len(result.stdout) == rb.MAX_OUTPUT_BYTES


# --- fail closed before launch --------------------------------------------


def test_missing_unshare_fails_closed_without_launch(layout, monkeypatch):
    root, worker = layout
    called = []
    monkeypatch.setattr(rb, "_launch", lambda *a: called.append(a))
    monkeypatch.setattr(rb, "UNSHARE", "/nonexistent/unshare")
    with pytest.raises(rb.BoundaryUnavailableError, match="not available"):
        rb.run_offline(worker, [], timeout=5, source_root=root)
    assert called == []


def test_non_linux_fails_closed_without_launch(layout, monkeypatch):
    root, worker = layout
    called = []
    monkeypatch.setattr(rb, "_launch", lambda *a: called.append(a))
    monkeypatch.setattr(rb.sys, "platform", "darwin")
    with pytest.raises(rb.BoundaryUnavailableError, match="Linux"):
        rb.run_offline(worker, [], timeout=5, source_root=root)
    assert called == []


def test_unstartable_namespace_command_is_unavailable(tmp_path):
    with pytest.raises(rb.BoundaryUnavailableError, match="Could not start"):
        rb._launch(
            ["/nonexistent/unshare", "-Urn", "true"],
            {"PATH": "/usr/bin:/bin"},
            tmp_path,
            tmp_path / "out",
            tmp_path / "err",
            5,
        )


# --- strict input validation ----------------------------------------------


def _bad_worker_cases(root, worker):
    (root / "pkg" / "notes.txt").write_text("x", encoding="utf-8")
    (root / "pkg" / "dir.py").mkdir()
    link = root / "pkg" / "link.py"
    try:
        os.symlink(worker, link)
    except OSError:
        link = None
    cases = [
        True,
        None,
        7,
        "",
        "pkg/worker.py",
        str(root / "pkg" / "missing.py"),
        str(root / "pkg" / "notes.txt"),
        str(root / "pkg" / "dir.py"),
        str(root / "pkg" / ".." / "pkg" / "worker.py"),
        str(worker) + "\x00",
    ]
    if link is not None:
        cases.append(str(link))
    return cases


def test_invalid_worker_paths_are_rejected(fake, layout):
    root, worker = layout
    launch = fake()
    for bad in _bad_worker_cases(root, worker):
        with pytest.raises(rb.BoundaryInputError):
            rb.run_offline(bad, [], timeout=5, source_root=root)
    assert launch.calls == []


def test_invalid_source_roots_are_rejected(fake, layout, tmp_path):
    root, worker = layout
    launch = fake()
    other = tmp_path.resolve() / "other"
    other.mkdir()
    for bad in (
        True,
        None,
        "src",
        str(root / "missing"),
        str(worker),
        str(other),
        str(root / "pkg" / ".."),
        str(worker.parent / "worker.py"),
    ):
        with pytest.raises(rb.BoundaryInputError):
            rb.run_offline(worker, [], timeout=5, source_root=bad)
    assert launch.calls == []


@pytest.mark.parametrize(
    "args",
    [
        ("a",),
        None,
        "abc",
        [1],
        [b"a"],
        [True],
        ["ok", None],
        ["a\x00b"],
        ["x" * (4096 + 1)],
        ["a"] * 65,
    ],
)
def test_invalid_args_are_rejected(fake, layout, args):
    root, worker = layout
    launch = fake()
    with pytest.raises(rb.BoundaryInputError):
        rb.run_offline(worker, args, timeout=5, source_root=root)
    assert launch.calls == []


def test_args_at_limits_are_accepted(fake, layout):
    root, worker = layout
    fake()
    rb.run_offline(worker, ["x" * 4096] * 64, timeout=5, source_root=root)


@pytest.mark.parametrize(
    "timeout",
    [True, False, 0, -1, 0.0, 7200.001, 7201, float("nan"), float("inf"), "5", None],
)
def test_invalid_timeouts_are_rejected(fake, layout, timeout):
    root, worker = layout
    launch = fake()
    with pytest.raises(rb.BoundaryInputError):
        rb.run_offline(worker, [], timeout=timeout, source_root=root)
    assert launch.calls == []


@pytest.mark.parametrize("timeout", [0.001, 1, 7200, 7200.0])
def test_valid_timeouts_are_accepted(fake, layout, timeout):
    root, worker = layout
    fake()
    rb.run_offline(worker, [], timeout=timeout, source_root=root)


@pytest.mark.parametrize(
    "overrides", [{"VIDEO_FPS": "30"}, {"API_KEY": "x"}, [], "A=1", ()]
)
def test_env_overrides_are_not_supported(fake, layout, overrides):
    root, worker = layout
    launch = fake()
    with pytest.raises(rb.BoundaryInputError, match="overrides"):
        rb.run_offline(worker, [], timeout=5, source_root=root, env_overrides=overrides)
    assert launch.calls == []


def test_empty_overrides_are_equivalent_to_none(fake, layout):
    root, worker = layout
    launch = fake()
    rb.run_offline(worker, [], timeout=5, source_root=root, env_overrides={})
    assert set(launch.calls[0]["env"]) == EXPECTED_KEYS


# --- static source properties ---------------------------------------------


def test_module_imports_only_stdlib_and_never_shells_out():
    tree = ast.parse(_MODULE_PATH.read_text("utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            imported.add((node.module or "").split(".")[0])
    assert imported <= set(sys.stdlib_module_names) | {"__future__"}
    for node in ast.walk(tree):
        if isinstance(node, ast.keyword) and node.arg == "shell":
            assert isinstance(node.value, ast.Constant) and node.value.value is False


def test_bootstrap_is_stdlib_only_and_guards_precede_worker():
    tree = ast.parse(rb.BOOTSTRAP_SOURCE)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module)
    assert imported == {"os", "sys", "socket", "types", "runpy"}
    main = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_main"
    )
    order = [ast.unparse(statement) for statement in main.body]
    position = {
        key: next(i for i, text in enumerate(order) if key in text)
        for key in (
            "_verify_namespace()",
            "_install_guards()",
            "_stub_dotenv()",
            "AUTOSHORTS_NETWORK_GUARD",
            "sys.path.insert",
            "runpy.run_path",
        )
    }
    assert (
        position["_verify_namespace()"]
        < position["_install_guards()"]
        < position["_stub_dotenv()"]
        < position["AUTOSHORTS_NETWORK_GUARD"]
        < position["sys.path.insert"]
        < position["runpy.run_path"]
    )
    assert "run_name='__main__'" in order[position["runpy.run_path"]]
    assert rb.GUARD_EXIT_CODE == 86 and "os._exit(86)" in rb.BOOTSTRAP_SOURCE


def test_bootstrap_outside_a_new_namespace_fails_closed_before_worker(tmp_path):
    """Real process, no namespace: the bootstrap must refuse to run the worker."""
    sentinel = tmp_path / "worker-ran"
    worker = _write(tmp_path / "worker.py", f"open({str(sentinel)!r}, 'w').close()\n")
    bootstrap = _write(tmp_path / "bootstrap.py", rb.BOOTSTRAP_SOURCE)
    env = {
        "PATH": "/usr/bin:/bin",
        "AUTOSHORTS_PARENT_NETNS": os.readlink("/proc/self/ns/net"),
        "AUTOSHORTS_BOUNDARY_READY": str(tmp_path / "ready"),
        "AUTOSHORTS_BOUNDARY_SOURCE_ROOT": str(tmp_path),
    }
    done = subprocess.run(
        [sys.executable, "-I", str(bootstrap), str(worker)],
        env=env,
        capture_output=True,
        timeout=60,
    )
    assert done.returncode == 86
    assert b"AUTOSHORTS_BOUNDARY_GUARD_FAILED" in done.stderr
    assert not sentinel.exists()
    assert not (tmp_path / "ready").exists()


# --- real process-group kill and bounded reads (no namespace needed) -------


def _gone(pid, seconds=10.0):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        try:
            state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
        except (FileNotFoundError, ProcessLookupError):
            return True
        if state in ("Z", "X"):
            return True
        time.sleep(0.05)
    return False


def _spawning_parent(pid_file: Path) -> str:
    return textwrap.dedent(f"""
        import subprocess, sys, time
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
        open({str(pid_file)!r}, "w").write(str(child.pid))
        time.sleep(120)
    """)


def _spawning_flood_parent(pid_file: Path, stream: str) -> str:
    return textwrap.dedent(f"""
        import subprocess, sys, time
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
        open({str(pid_file)!r}, "w").write(str(child.pid))
        target = getattr(sys, {stream!r})
        target.write("x" * {rb.MAX_OUTPUT_BYTES + 1})
        target.flush()
        time.sleep(120)
    """)


def test_launch_timeout_kills_whole_process_group(tmp_path):
    pid_file = tmp_path / "grandchild.pid"
    command = [sys.executable, "-c", _spawning_parent(pid_file)]
    started = time.monotonic()
    timed_out, returncode = rb._launch(
        command,
        {"PATH": "/usr/bin:/bin"},
        tmp_path,
        tmp_path / "out",
        tmp_path / "err",
        3,
    )
    assert timed_out is True
    assert returncode == -9
    assert time.monotonic() - started < 30
    assert pid_file.exists()
    assert _gone(int(pid_file.read_text()))


@pytest.mark.parametrize("stream", ["stdout", "stderr"])
def test_launch_inflight_overflow_kills_group_and_reaps_promptly(tmp_path, stream):
    pid_file = tmp_path / "grandchild.pid"
    command = [sys.executable, "-c", _spawning_flood_parent(pid_file, stream)]
    started = time.monotonic()
    with pytest.raises(rb.BoundaryOutputError) as caught:
        rb._launch(
            command,
            {"PATH": "/usr/bin:/bin"},
            tmp_path,
            tmp_path / "out",
            tmp_path / "err",
            60,
        )
    assert caught.value.returncode == -9
    assert time.monotonic() - started < 30
    assert pid_file.exists()
    assert _gone(int(pid_file.read_text()))


def test_launch_kills_leftover_group_members_after_normal_exit(tmp_path):
    pid_file = tmp_path / "grandchild.pid"
    script = textwrap.dedent(f"""
        import subprocess, sys
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
        open({str(pid_file)!r}, "w").write(str(child.pid))
    """)
    timed_out, returncode = rb._launch(
        [sys.executable, "-c", script],
        {"PATH": "/usr/bin:/bin"},
        tmp_path,
        tmp_path / "out",
        tmp_path / "err",
        60,
    )
    assert (timed_out, returncode) == (False, 0)
    assert _gone(int(pid_file.read_text()))


@pytest.mark.parametrize("stream", ["stdout", "stderr"])
def test_launch_accepts_exact_per_stream_output_limit(tmp_path, stream):
    other = "stderr" if stream == "stdout" else "stdout"
    script = f"import sys; sys.{stream}.write('z' * {rb.MAX_OUTPUT_BYTES}); sys.{stream}.flush()"
    timed_out, returncode = rb._launch(
        [sys.executable, "-c", script],
        {"PATH": "/usr/bin:/bin"},
        tmp_path,
        tmp_path / "out",
        tmp_path / "err",
        60,
    )
    assert (timed_out, returncode) == (False, 0)
    raw, over = rb._read_capped(tmp_path / ("out" if stream == "stdout" else "err"))
    assert len(raw) == rb.MAX_OUTPUT_BYTES and over is False
    assert rb._read_capped(tmp_path / ("err" if other == "stderr" else "out")) == (
        b"",
        False,
    )


def test_launch_writes_output_to_files_and_reads_are_capped(tmp_path):
    script = (
        f"import sys; sys.stderr.write('err'); sys.stderr.flush(); "
        f"sys.stdout.write('a' * {rb.MAX_OUTPUT_BYTES + 10})"
    )
    with pytest.raises(rb.BoundaryOutputError) as caught:
        rb._launch(
            [sys.executable, "-c", script],
            {"PATH": "/usr/bin:/bin"},
            tmp_path,
            tmp_path / "out",
            tmp_path / "err",
            60,
        )
    assert caught.value.returncode in (0, -9)
    assert len(caught.value.stdout) == rb.DIAGNOSTIC_CHARS
    assert caught.value.stderr == "err"
    raw, over = rb._read_capped(tmp_path / "out")
    assert over is True and len(raw) == rb.MAX_OUTPUT_BYTES
    assert rb._read_capped(tmp_path / "err") == (b"err", False)


def test_launch_refuses_existing_output_files(tmp_path):
    (tmp_path / "out").write_bytes(b"old")
    with pytest.raises(FileExistsError):
        rb._launch(
            [sys.executable, "-c", "pass"],
            {},
            tmp_path,
            tmp_path / "out",
            tmp_path / "err",
            5,
        )
    assert (tmp_path / "out").read_bytes() == b"old"


# --- real namespace integration (skips only without the capability) ------


@functools.lru_cache(maxsize=1)
def _namespace_capability():
    if not sys.platform.startswith("linux") or not os.access(rb.UNSHARE, os.X_OK):
        return f"{rb.UNSHARE} unavailable"
    try:
        probe = subprocess.run(
            [rb.UNSHARE, "-Urn", sys.executable, "-I", "-c", "pass"],
            env={"PATH": "/usr/bin:/bin"},
            capture_output=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"unshare probe failed: {exc}"
    if probe.returncode != 0:
        return (
            "user+network namespace not permitted: "
            + probe.stderr.decode(errors="replace")[-300:]
        )
    return None


@pytest.fixture
def namespace(tmp_path):
    reason = _namespace_capability()
    if reason:
        pytest.skip(reason)
    root = tmp_path.resolve() / "src"
    return root


GUARD_WORKER = """
    import json, os, socket, sys, _socket
    import pkg.helper as helper
    import dotenv

    def denied(action):
        try:
            action()
        except PermissionError as exc:
            return "PermissionError"
        except Exception as exc:
            reason = getattr(exc, "reason", None)
            return "wrapped-PermissionError" if isinstance(reason, PermissionError) else repr(exc)
        return "ALLOWED"

    def urllib_open():
        import urllib.request
        urllib.request.urlopen("http://127.0.0.1:9/", timeout=2)

    print(json.dumps({
        "helper": helper.AT_IMPORT,
        "marker": os.environ.get("AUTOSHORTS_NETWORK_GUARD"),
        "dotenv_disabled": os.environ.get("PYTHON_DOTENV_DISABLED"),
        "dotenv_stub": getattr(dotenv, "AUTOSHORTS_OFFLINE_STUB", False),
        "load_dotenv": dotenv.load_dotenv(),
        "dotenv_values": dotenv.dotenv_values(),
        "sys_path0": sys.path[0],
        "argv": sys.argv,
        "env": dict(os.environ),
        "create_connection": denied(lambda: socket.create_connection(("127.0.0.1", 9), timeout=2)),
        "getaddrinfo": denied(lambda: socket.getaddrinfo("localhost", 80)),
        "socket_connect": denied(lambda: socket.socket().connect(("127.0.0.1", 9))),
        "raw_socket_connect": denied(lambda: _socket.socket().connect(("127.0.0.1", 9))),
        "raw_getaddrinfo": denied(lambda: _socket.getaddrinfo("localhost", 80)),
        "urllib": denied(urllib_open),
    }))
"""

HELPER = """
    import os, sys
    AT_IMPORT = {
        "marker": os.environ.get("AUTOSHORTS_NETWORK_GUARD"),
        "dotenv_stub": getattr(sys.modules.get("dotenv"), "AUTOSHORTS_OFFLINE_STUB", False),
    }
"""

NATIVE_WORKER = """
import subprocess, sys
probe = '''
import errno, json, os, socket
def attempt(address):
    s = socket.socket()
    s.settimeout(3)
    try:
        s.connect(address)
        return "CONNECTED"
    except OSError as exc:
        return errno.errorcode.get(exc.errno, str(exc.errno))
    finally:
        s.close()
with open("/proc/net/dev") as f:
    devices = sorted(l.split(":", 1)[0].strip() for l in f.read().splitlines()[2:] if ":" in l)
with open("/proc/net/route") as f:
    routes = sorted(l.split()[0] for l in f.read().splitlines()[1:] if l.split())
print(json.dumps({
    "ns": os.readlink("/proc/self/ns/net"),
    "devices": devices,
    "routes": routes,
    "loopback": attempt(("127.0.0.1", 9)),
    "test_net": attempt(("192.0.2.1", 9)),
}))
'''
# A fresh interpreter without the bootstrap's Python guards: only the
# kernel namespace stands between it and the network.
done = subprocess.run([sys.executable, "-I", "-S", "-c", probe], capture_output=True, timeout=30)
sys.stdout.write(done.stdout.decode())
sys.stderr.write(done.stderr.decode())
sys.exit(done.returncode)
"""


def test_real_guards_precede_worker_and_env_is_clean(namespace, monkeypatch):
    for key, value in HOSTILE_ENV.items():
        monkeypatch.setenv(key, value)
    root = namespace
    _write(root / "pkg" / "__init__.py", "")
    _write(root / "pkg" / "helper.py", HELPER)
    worker = _write(root / "pkg" / "worker.py", GUARD_WORKER)
    result = rb.run_offline(worker, ["--flag", "value"], timeout=120, source_root=root)
    data = json.loads(result.stdout)
    assert data["helper"] == {"marker": "1", "dotenv_stub": True}
    assert data["marker"] == "1" and data["dotenv_disabled"] == "1"
    assert data["dotenv_stub"] is True
    assert data["load_dotenv"] is False and data["dotenv_values"] == {}
    assert data["sys_path0"] == str(root)
    assert data["argv"] == [str(worker), "--flag", "value"]
    for key in (
        "create_connection",
        "getaddrinfo",
        "socket_connect",
        "raw_socket_connect",
        "raw_getaddrinfo",
    ):
        assert data[key] == "PermissionError", key
    assert data["urllib"] in ("PermissionError", "wrapped-PermissionError")
    env = data["env"]
    assert set(env) == EXPECTED_KEYS | {"AUTOSHORTS_NETWORK_GUARD"}
    assert env["API_KEY"] == "" and env["PATH"] == "/usr/bin:/bin"
    assert env["HOME"] != HOSTILE_ENV["HOME"] and "autoshorts-boundary-" in env["HOME"]
    joined = json.dumps(env)
    for secret in HOSTILE_ENV.values():
        if secret not in ("*", "0"):
            assert secret not in joined


def test_real_imported_synthetic_source_does_not_write_pycache(namespace):
    root = namespace
    _write(root / "pkg" / "__init__.py", "")
    _write(root / "pkg" / "helper.py", "VALUE = 42\n")
    worker = _write(
        root / "pkg" / "worker.py", "import pkg.helper\nprint(pkg.helper.VALUE)\n"
    )

    result = rb.run_offline(worker, [], timeout=120, source_root=root)

    assert result.stdout == "42\n"
    assert not list(root.rglob("__pycache__"))
    assert not list(root.rglob("*.pyc"))


def test_real_native_child_has_separate_namespace_and_no_route(namespace):
    root = namespace
    worker = _write(root / "pkg" / "native_worker.py", NATIVE_WORKER)
    result = rb.run_offline(worker, [], timeout=120, source_root=root)
    data = json.loads(result.stdout)
    assert data["ns"].startswith("net:[")
    assert data["ns"] != os.readlink("/proc/self/ns/net")
    assert set(data["devices"]) <= {"lo"}
    assert set(data["routes"]) <= {"lo"}
    assert data["loopback"] != "CONNECTED"
    assert data["test_net"] != "CONNECTED"
    assert data["test_net"] in {
        errno.errorcode[errno.ENETUNREACH],
        errno.errorcode[errno.EHOSTUNREACH],
        errno.errorcode[errno.EADDRNOTAVAIL],
    }


def test_real_timeout_kills_namespace_process_group(namespace):
    root = namespace
    pid_file = root.parent / "grandchild.pid"
    worker = _write(root / "pkg" / "sleepy.py", _spawning_parent(pid_file))
    with pytest.raises(rb.BoundaryTimeoutError):
        rb.run_offline(worker, [], timeout=5, source_root=root)
    assert pid_file.exists()
    assert _gone(int(pid_file.read_text()))


def test_real_output_overflow_is_rejected(namespace):
    root = namespace
    worker = _write(
        root / "pkg" / "noisy.py",
        f"import sys\nsys.stdout.write('n' * {rb.MAX_OUTPUT_BYTES + 1})\n",
    )
    with pytest.raises(rb.BoundaryOutputError):
        rb.run_offline(worker, [], timeout=120, source_root=root)


def test_real_nonzero_exit_is_failure(namespace):
    root = namespace
    worker = _write(
        root / "pkg" / "failing.py",
        "import sys\nsys.stderr.write('saved input rejected')\nsys.exit(3)\n",
    )
    with pytest.raises(rb.BoundaryFailureError, match="saved input rejected") as caught:
        rb.run_offline(worker, [], timeout=120, source_root=root)
    assert caught.value.returncode == 3


def test_real_success_returns_output(namespace):
    root = namespace
    worker = _write(
        root / "pkg" / "echo.py", "import sys\nprint('|'.join(sys.argv[1:]))\n"
    )
    result = rb.run_offline(worker, ["a b", "", "-x"], timeout=120, source_root=root)
    assert result.returncode == 0
    assert result.stdout == "a b||-x\n"
