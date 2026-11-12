"""Standalone offline process boundary for saved-input workers.

Standard library only. Load this file by direct path (importlib), never via
``autoshorts`` package initialization, so no application code or native
network client is imported in the parent.

``run_offline`` launches a supplied worker script inside a genuine Linux
network namespace (``/usr/bin/unshare -Urn``) with a credential-free
environment built from scratch. A generated bootstrap verifies the separate
namespace and the absence of non-loopback interfaces/routes, installs
unconditional socket and audit-hook denials and a no-op ``dotenv`` module, and
only then exposes the worker's source root and runs the worker. If any of this
cannot be established the call fails closed; there is no weaker fallback.

Output size is checked periodically, not constrained by a filesystem quota; a
fast or bursty writer can exceed the configured limit between polls. The
wall-clock timeout kills the spawned process group. A worker process that
deliberately calls ``setsid`` leaves that group; it still stays inside the
network namespace.
"""

from __future__ import annotations

import math
import os
import signal
import stat
import subprocess
import sys
import tempfile
import time
from pathlib import Path

UNSHARE = "/usr/bin/unshare"
NAMESPACE_FLAGS = "-Urn"
SAFE_PATH = "/usr/bin:/bin"
MAX_TIMEOUT_SECONDS = 7200
MAX_ARGS = 64
MAX_ARG_CHARS = 4096
MAX_OUTPUT_BYTES = 1024 * 1024
DIAGNOSTIC_CHARS = 4096
GUARD_EXIT_CODE = 86
_POLL_SECONDS = 0.05

BOOTSTRAP_SOURCE = r'''"""Generated autoshorts offline bootstrap: guards run before any worker import."""
import os
import sys

_DENIAL = "network access denied by autoshorts offline boundary"
_DENIED_EVENTS = frozenset({
    "socket.bind",
    "socket.connect",
    "socket.getaddrinfo",
    "socket.gethostbyaddr",
    "socket.gethostbyname",
    "socket.getnameinfo",
    "socket.sendmsg",
    "socket.sendto",
    "urllib.Request",
    "http.client.connect",
    "ftplib.connect",
    "imaplib.open",
    "nntplib.connect",
    "poplib.connect",
    "smtplib.connect",
    "telnetlib.Telnet.open",
    "webbrowser.open",
})


def _fail(message):
    sys.stderr.write("AUTOSHORTS_BOUNDARY_GUARD_FAILED: " + message + "\n")
    sys.stderr.flush()
    os._exit(86)


def _routes(path, device_column):
    try:
        with open(path, encoding="ascii", errors="replace") as stream:
            lines = stream.read().splitlines()
    except FileNotFoundError:
        return set()
    except OSError as exc:
        _fail("cannot read routes: %s" % exc)
    if path.endswith("/route"):
        lines = lines[1:]
    return {line.split()[device_column] for line in lines if line.split()}


def _verify_namespace():
    parent = os.environ.get("AUTOSHORTS_PARENT_NETNS", "")
    try:
        current = os.readlink("/proc/self/ns/net")
    except OSError as exc:
        _fail("cannot read network namespace: %s" % exc)
    if not parent.startswith("net:[") or not current.startswith("net:[") or current == parent:
        _fail("not running in a separate network namespace")
    try:
        with open("/proc/net/dev", encoding="ascii", errors="replace") as stream:
            lines = stream.read().splitlines()[2:]
    except OSError as exc:
        _fail("cannot list network interfaces: %s" % exc)
    devices = {line.split(":", 1)[0].strip() for line in lines if ":" in line}
    devices |= _routes("/proc/net/route", 0)
    devices |= _routes("/proc/net/ipv6_route", -1)
    if devices - {"lo"}:
        _fail("non-loopback interface or route present: %s" % ",".join(sorted(devices - {"lo"})))
    return current


def _audit(event, args):
    if event in _DENIED_EVENTS:
        raise PermissionError(_DENIAL + ": " + event)


def _deny(*args, **kwargs):
    raise PermissionError(_DENIAL)


def _install_guards():
    sys.addaudithook(_audit)
    import socket

    for name in ("create_connection", "create_server", "getaddrinfo", "gethostbyname",
                 "gethostbyname_ex", "gethostbyaddr", "getnameinfo"):
        if hasattr(socket, name):
            setattr(socket, name, _deny)
    for name in ("bind", "connect", "connect_ex", "sendmsg", "sendto"):
        setattr(socket.socket, name, _deny)


def _stub_dotenv():
    import types

    def load_dotenv(*args, **kwargs):
        return False

    def dotenv_values(*args, **kwargs):
        return {}

    def find_dotenv(*args, **kwargs):
        return ""

    package = types.ModuleType("dotenv")
    main = types.ModuleType("dotenv.main")
    for module in (package, main):
        module.load_dotenv = load_dotenv
        module.dotenv_values = dotenv_values
        module.find_dotenv = find_dotenv
        module.AUTOSHORTS_OFFLINE_STUB = True
    package.__path__ = []
    package.main = main
    sys.modules["dotenv"] = package
    sys.modules["dotenv.main"] = main


def _main():
    if len(sys.argv) < 2:
        _fail("missing worker path")
    namespace = _verify_namespace()
    _install_guards()
    _stub_dotenv()
    os.environ["PYTHON_DOTENV_DISABLED"] = "1"
    os.environ["AUTOSHORTS_NETWORK_GUARD"] = "1"
    ready = os.environ.get("AUTOSHORTS_BOUNDARY_READY", "")
    source_root = os.environ.get("AUTOSHORTS_BOUNDARY_SOURCE_ROOT", "")
    if not ready or not source_root:
        _fail("missing boundary environment")
    try:
        with open(ready, "x", encoding="ascii") as stream:
            stream.write(namespace)
    except OSError as exc:
        _fail("cannot write readiness marker: %s" % exc)
    worker = sys.argv[1]
    sys.argv = [worker] + sys.argv[2:]
    # The worker's source root becomes importable only now, after every guard.
    sys.path.insert(0, source_root)
    import runpy

    runpy.run_path(worker, run_name="__main__")


_main()
'''


class BoundaryError(RuntimeError):
    """The offline boundary refused, could not establish, or saw a failed run."""

    def __init__(self, message, *, returncode=None, stdout="", stderr=""):
        super().__init__(message)
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


class BoundaryInputError(BoundaryError, ValueError):
    """Invalid worker path, source root, arguments, timeout or overrides."""


class BoundaryUnavailableError(BoundaryError):
    """The network namespace or the in-process guards were not established."""


class BoundaryTimeoutError(BoundaryError):
    """The worker exceeded its wall-clock limit; its process group was killed."""


class BoundaryFailureError(BoundaryError):
    """The worker ran inside the boundary but exited unsuccessfully."""


class BoundaryOutputError(BoundaryError):
    """The worker produced more output than the boundary returns."""


def _canonical_path(value, label):
    if isinstance(value, bool) or not isinstance(value, (str, Path)):
        raise BoundaryInputError(f"{label} must be a path")
    text = str(value)
    if not text or "\x00" in text or not os.path.isabs(text):
        raise BoundaryInputError(f"{label} must be an absolute path")
    if os.path.realpath(text) != text:
        raise BoundaryInputError(f"{label} must be canonical, without symlinks or '..'")
    return Path(text)


def _worker(value):
    path = _canonical_path(value, "worker_path")
    if path.suffix != ".py":
        raise BoundaryInputError("worker_path must be a .py file")
    try:
        mode = os.lstat(path).st_mode
    except OSError as exc:
        raise BoundaryInputError(f"worker_path does not exist: {path}") from exc
    if not stat.S_ISREG(mode):
        raise BoundaryInputError("worker_path must be a regular file")
    return path


def _source_root(value, worker):
    path = _canonical_path(value, "source_root")
    try:
        mode = os.lstat(path).st_mode
    except OSError as exc:
        raise BoundaryInputError(f"source_root does not exist: {path}") from exc
    if not stat.S_ISDIR(mode):
        raise BoundaryInputError("source_root must be a directory")
    if path not in worker.parents:
        raise BoundaryInputError("worker_path must be inside source_root")
    return path


def _arguments(value):
    if type(value) is not list or len(value) > MAX_ARGS:
        raise BoundaryInputError(f"args must be a list of at most {MAX_ARGS} strings")
    for item in value:
        if type(item) is not str or len(item) > MAX_ARG_CHARS or "\x00" in item:
            raise BoundaryInputError(
                f"each arg must be a string of at most {MAX_ARG_CHARS} chars"
            )
    return list(value)


def _timeout(value):
    if type(value) not in (int, float):
        raise BoundaryInputError("timeout must be an int or float, not bool")
    seconds = float(value)
    if not math.isfinite(seconds) or not 0 < seconds <= MAX_TIMEOUT_SECONDS:
        raise BoundaryInputError(
            f"timeout must be finite, > 0 and <= {MAX_TIMEOUT_SECONDS}"
        )
    return seconds


def _require_namespace_support():
    """Parent network-namespace identity; fail closed when unshare is unusable."""
    if not sys.platform.startswith("linux"):
        raise BoundaryUnavailableError("Linux network namespaces are required")
    if not (os.path.isfile(UNSHARE) and os.access(UNSHARE, os.X_OK)):
        raise BoundaryUnavailableError(f"{UNSHARE} is not available")
    executable = sys.executable
    if (
        not executable
        or not os.path.isabs(executable)
        or not os.path.isfile(executable)
    ):
        raise BoundaryUnavailableError("An absolute Python executable is required")
    try:
        namespace = os.readlink("/proc/self/ns/net")
    except OSError as exc:
        raise BoundaryUnavailableError(
            "Cannot read the parent network namespace"
        ) from exc
    if not namespace.startswith("net:["):
        raise BoundaryUnavailableError("Unexpected parent network namespace identity")
    return namespace


def _environment(home, tmp, ready, source_root, parent_namespace):
    """The complete child environment; nothing is inherited from the parent."""
    return {
        "PATH": SAFE_PATH,
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "HOME": str(home),
        "TMPDIR": str(tmp),
        "TMP": str(tmp),
        "TEMP": str(tmp),
        "XDG_CACHE_HOME": str(home / ".cache"),
        "XDG_CONFIG_HOME": str(home / ".config"),
        "XDG_DATA_HOME": str(home / ".local" / "share"),
        "PYTHON_DOTENV_DISABLED": "1",
        "API_KEY": "",
        "AUTOSHORTS_OFFLINE_BOUNDARY": "1",
        "AUTOSHORTS_BOUNDARY_READY": str(ready),
        "AUTOSHORTS_BOUNDARY_SOURCE_ROOT": str(source_root),
        "AUTOSHORTS_PARENT_NETNS": parent_namespace,
    }


def _command(bootstrap, worker, arguments):
    return [
        UNSHARE,
        NAMESPACE_FLAGS,
        sys.executable,
        "-I",
        "-B",
        str(bootstrap),
        str(worker),
        *arguments,
    ]


def _kill_group(pgid):
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def _launch(command, env, cwd, stdout_path, stderr_path, timeout):
    """Run in a new session; always SIGKILL its whole group before reaping.

    Returns ``(timed_out, returncode)``. The leader stays an unreaped zombie
    (``WNOWAIT``) until the group is killed, so its pgid cannot be reused.
    Output-size polling is fail-closed but is not a disk quota: a fast writer
    can overshoot the per-stream limit between polls.
    """
    with open(stdout_path, "xb") as stdout, open(stderr_path, "xb") as stderr:
        try:
            process = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                env=env,
                cwd=cwd,
                shell=False,
                close_fds=True,
                start_new_session=True,
            )
        except OSError as exc:
            raise BoundaryUnavailableError(
                f"Could not start the network namespace: {exc}"
            ) from exc
        timed_out = False
        output_overflow = False
        try:
            deadline = time.monotonic() + timeout
            while True:
                try:
                    exited = os.waitid(
                        os.P_PID, process.pid, os.WEXITED | os.WNOWAIT | os.WNOHANG
                    )
                except ChildProcessError:
                    exited = True
                sizes = (
                    os.fstat(stdout.fileno()).st_size,
                    os.fstat(stderr.fileno()).st_size,
                )
                if any(size > MAX_OUTPUT_BYTES for size in sizes):
                    output_overflow = True
                    break
                if exited is not None:
                    break
                if time.monotonic() >= deadline:
                    timed_out = True
                    break
                time.sleep(_POLL_SECONDS)
        finally:
            try:
                _kill_group(process.pid)
            finally:
                process.wait()
    if output_overflow:
        stdout_raw, _ = _read_capped(stdout_path)
        stderr_raw, _ = _read_capped(stderr_path)
        stdout_text = stdout_raw.decode("utf-8", errors="replace")
        stderr_text = stderr_raw.decode("utf-8", errors="replace")
        raise BoundaryOutputError(
            f"Process output exceeded {MAX_OUTPUT_BYTES} bytes",
            returncode=process.returncode,
            stdout=_tail(stdout_text),
            stderr=_tail(stderr_text),
        )
    return timed_out, process.returncode


def _read_capped(path):
    try:
        with open(path, "rb") as stream:
            raw = stream.read(MAX_OUTPUT_BYTES + 1)
    except FileNotFoundError:
        return b"", False
    return raw[:MAX_OUTPUT_BYTES], len(raw) > MAX_OUTPUT_BYTES


def _read_ready(path):
    try:
        with open(path, "rb") as stream:
            return stream.read(256).decode("ascii")
    except (OSError, UnicodeDecodeError):
        return None


def _tail(text):
    return text[-DIAGNOSTIC_CHARS:]


def run_offline(worker_path, args, *, timeout, source_root, env_overrides=None):
    """Run ``worker_path`` with ``args`` offline; return its bounded output.

    Raises a ``BoundaryError`` subclass for invalid input, unavailable
    isolation, timeout, nonzero exit or output above ``MAX_OUTPUT_BYTES``.
    """
    worker = _worker(worker_path)
    root = _source_root(source_root, worker)
    arguments = _arguments(args)
    seconds = _timeout(timeout)
    if env_overrides is not None and (type(env_overrides) is not dict or env_overrides):
        raise BoundaryInputError(
            "Environment overrides are not supported by this boundary"
        )
    parent_namespace = _require_namespace_support()

    with tempfile.TemporaryDirectory(
        prefix="autoshorts-boundary-", dir="/tmp", ignore_cleanup_errors=True
    ) as name:
        base = Path(os.path.realpath(name))
        home, tmp, work = base / "home", base / "tmp", base / "work"
        for directory in (home, tmp, work):
            directory.mkdir(mode=0o700)
        bootstrap = base / "bootstrap.py"
        bootstrap.write_text(BOOTSTRAP_SOURCE, encoding="utf-8")
        ready = base / "ready"
        command = _command(bootstrap, worker, arguments)
        env = _environment(home, tmp, ready, root, parent_namespace)
        timed_out, returncode = _launch(
            command, env, work, base / "stdout", base / "stderr", seconds
        )
        stdout_raw, stdout_over = _read_capped(base / "stdout")
        stderr_raw, stderr_over = _read_capped(base / "stderr")
        child_namespace = _read_ready(ready)

    stdout = stdout_raw.decode("utf-8", errors="replace")
    stderr = stderr_raw.decode("utf-8", errors="replace")
    details = {
        "returncode": returncode,
        "stdout": _tail(stdout),
        "stderr": _tail(stderr),
    }
    if timed_out:
        raise BoundaryTimeoutError(
            f"Worker exceeded {seconds:g}s; its process group was killed. stderr tail:\n{_tail(stderr)}",
            **details,
        )
    if (
        child_namespace is None
        or not child_namespace.startswith("net:[")
        or child_namespace == parent_namespace
    ):
        raise BoundaryUnavailableError(
            f"Network namespace/guards were not established (exit {returncode}). "
            f"stderr tail:\n{_tail(stderr)}",
            **details,
        )
    if returncode != 0:
        raise BoundaryFailureError(
            f"Offline worker failed with exit code {returncode}. stderr tail:\n{_tail(stderr)}",
            **details,
        )
    if stdout_over or stderr_over:
        raise BoundaryOutputError(
            f"Worker output exceeded {MAX_OUTPUT_BYTES} bytes", **details
        )
    return subprocess.CompletedProcess(command, 0, stdout, stderr)
