"""Source-only supervisor for the reviewed public-fetch filesystem RPC.

Requires the qualified resume_boundary.py, resume_rpc.py and resume_fetch.py
beside this file. Load by direct filepath, never through the application.
The boundary's unchanged bootstrap establishes the offline namespace/guards.
Only the separate, credential-free one-request helper loads public_http.py.

The offline group is stopped while a helper runs: FileRPC cannot consume its
response before the supervisor checks the helper's exit and typed response.
The client still owns deletion of both slots. No threads or network clients
are used here. This function must exclusively own its children's wait status.

This is not a filesystem/account sandbox. In particular, processes that call
setsid can escape process-group cleanup (not their network namespace), and
clean HOME/config environment variables do not revoke filesystem permissions.
Output is capped when checked/read, not by a filesystem quota: temporary-disk
usage can grow between checks, particularly during the bounded helper launch.
"""

from __future__ import annotations

import importlib.util
import json
import os
import signal
import stat
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

_HERE = Path(os.path.realpath(__file__)).parent
_HELPER_SECONDS = 30.0
_MAX_ALLOWLIST_JSON_CHARS = 64 * 1024


def _local_file(name):
    path = str(_HERE / name)
    if os.path.realpath(path) != path or not stat.S_ISREG(os.lstat(path).st_mode):
        raise ValueError(f"{name} must be a canonical regular sibling of this file")
    return path


def _load_local(name, module_name):
    spec = importlib.util.spec_from_file_location(module_name, _local_file(name))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {name} by filepath")
    module = importlib.util.module_from_spec(spec)
    previous = sys.modules.get(module_name)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        if previous is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = previous
    return module


boundary = _load_local("resume_boundary.py", "_autoshorts_supervisor_boundary")
rpc = _load_local("resume_rpc.py", "_autoshorts_supervisor_rpc")

if TYPE_CHECKING:
    from .resume_boundary import BoundaryError as _BoundaryErrorBase
    from .resume_boundary import BoundaryTimeoutError as _BoundaryTimeoutErrorBase
else:
    # Keep the actual filepath-loaded classes, not separately imported copies.
    _BoundaryErrorBase = boundary.BoundaryError
    _BoundaryTimeoutErrorBase = boundary.BoundaryTimeoutError


class SupervisorProtocolError(_BoundaryErrorBase):
    """An unsafe, changed, replayed or out-of-order RPC slot was observed."""


class SupervisorHelperError(_BoundaryErrorBase):
    """The one-request helper failed its exit/output/response contract."""


class SupervisorHelperTimeoutError(_BoundaryTimeoutErrorBase):
    """A helper exceeded its per-request deadline; its group was killed."""


def _remaining(deadline):
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise boundary.BoundaryTimeoutError("Whole-run deadline exceeded")
    return remaining


@contextmanager
def _snapshot(path, limit):
    """Bounded nonblocking read; retain the descriptor through the caller's use."""
    flags = os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW | os.O_CLOEXEC
    try:
        descriptor = os.open(path, flags)
    except FileNotFoundError:
        yield None
        return
    except OSError as exc:
        raise SupervisorProtocolError(f"Cannot safely open {path.name}: {exc}") from exc
    try:
        try:
            info = os.fstat(descriptor)
            if not stat.S_ISREG(info.st_mode) or not 0 < info.st_size <= limit:
                raise SupervisorProtocolError(
                    f"{path.name} must be a regular file of 1..{limit} bytes"
                )
            raw = b""
            while len(raw) <= limit:
                chunk = os.read(descriptor, limit + 1 - len(raw))
                if not chunk:
                    break
                raw += chunk
            if not 0 < len(raw) <= limit or len(raw) != info.st_size:
                raise SupervisorProtocolError(
                    f"{path.name} changed size or exceeded its limit"
                )
        except OSError as exc:
            raise SupervisorProtocolError(f"Cannot read {path.name}: {exc}") from exc
        yield raw, (info.st_dev, info.st_ino), descriptor
    finally:
        os.close(descriptor)


def _exists(path):
    try:
        os.lstat(path)
    except FileNotFoundError:
        return False
    return True


def _ready(path, parent_namespace):
    try:
        with _snapshot(path, 256) as snapshot:
            namespace = None if snapshot is None else snapshot[0].decode("ascii")
        current = os.readlink("/proc/self/ns/net")
    except (SupervisorProtocolError, OSError, UnicodeDecodeError) as exc:
        raise boundary.BoundaryUnavailableError(
            "Invalid boundary readiness marker"
        ) from exc
    if (
        namespace is None
        or not namespace.startswith("net:[")
        or not namespace.endswith("]")
        or not namespace[5:-1].isdigit()
        or namespace in (parent_namespace, current)
        or current != parent_namespace
    ):
        raise boundary.BoundaryUnavailableError(
            "Network namespace/guards were not established"
        )


def _rpc_identity(directory, identity):
    try:
        if rpc._private_directory(str(directory)) != identity:
            raise SupervisorProtocolError("RPC directory changed identity")
    except rpc.RPCError as exc:
        raise SupervisorProtocolError(
            "RPC directory is no longer canonical and private"
        ) from exc


def _exited(process):
    # Never poll()/wait() a leader before killing its group: retain its pgid.
    return (
        os.waitid(os.P_PID, process.pid, os.WEXITED | os.WNOWAIT | os.WNOHANG)
        is not None
    )


def _freeze(process, deadline):
    if _exited(process):
        return False
    os.killpg(process.pid, signal.SIGSTOP)
    while True:
        _remaining(deadline)
        state = os.waitid(
            os.P_PID, process.pid, os.WSTOPPED | os.WEXITED | os.WNOWAIT | os.WNOHANG
        )
        if state is not None:
            return state.si_code == os.CLD_STOPPED
        time.sleep(min(boundary._POLL_SECONDS, _remaining(deadline)))


def _request(raw, next_id):
    try:
        request = rpc.validate_request(raw)
    except Exception as exc:
        raise SupervisorProtocolError(
            f"Rejected request: {type(exc).__name__}: {exc}"
        ) from exc
    if request["id"] != next_id:
        raise SupervisorProtocolError(
            f"Expected request id {next_id}, not {request['id']}"
        )
    # Valid photo candidates use execute_request's authoritative allowlist check
    # before transport import. Its typed denial is an ordinary helper failure;
    # _run_helper still verifies the exit, request snapshot and response below.
    return request


def _helper_environment(home, tmp):
    # Independently constructed: no parent HOME, proxies, credentials or PYTHONPATH.
    return {
        "PATH": "/usr/bin:/bin",
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
    }


def _run_helper(
    base,
    directory,
    identity,
    approved,
    request,
    helper,
    rpc_path,
    transport,
    allowlist_json,
    deadline,
):
    helper_deadline = min(deadline, time.monotonic() + _HELPER_SECONDS)
    folder = base / f"fetch-{request['id']:03d}"
    folder.mkdir(mode=0o700)
    home, tmp, work = folder / "home", folder / "tmp", folder / "work"
    for path in (home, tmp, work):
        path.mkdir(mode=0o700)
    command = [
        sys.executable,
        "-I",
        "-B",
        helper,
        "--request",
        str(directory / rpc.REQUEST_NAME),
        "--response",
        str(directory / rpc.RESPONSE_NAME),
        "--rpc",
        rpc_path,
        "--transport",
        transport,
        "--allowlist-json",
        allowlist_json,
    ]
    budget = helper_deadline - time.monotonic()
    _remaining(deadline)
    if budget <= 0:
        raise SupervisorHelperTimeoutError("Helper setup exhausted its deadline")
    # The qualified launcher always kills the helper group BEFORE reaping,
    # including on successful exit, timeout, or an exception. No helper threads.
    timed_out, code = boundary._launch(
        command,
        _helper_environment(home, tmp),
        work,
        folder / "stdout",
        folder / "stderr",
        budget,
    )
    _remaining(deadline)
    if timed_out or time.monotonic() >= helper_deadline:
        raise SupervisorHelperTimeoutError("Public-fetch helper exceeded its deadline")
    _, stdout_over = boundary._read_capped(folder / "stdout")
    stderr, stderr_over = boundary._read_capped(folder / "stderr")
    if stdout_over or stderr_over:
        raise boundary.BoundaryOutputError(
            "Helper output exceeded the 1 MiB per-stream limit"
        )
    if code not in (0, 1):
        tail = boundary._tail(stderr.decode("utf-8", errors="replace"))
        raise SupervisorHelperError(f"Public-fetch helper exited {code}: {tail}")
    _rpc_identity(directory, identity)
    with _snapshot(directory / rpc.REQUEST_NAME, rpc.MAX_REQUEST_BYTES) as current:
        if current is None or current[:2] != approved[:2]:
            raise SupervisorProtocolError(
                "Request disappeared or changed before response acceptance"
            )
    limit = 4 * ((request["max_bytes"] + 2) // 3) + 4096
    with _snapshot(directory / rpc.RESPONSE_NAME, limit) as response:
        if response is None:
            raise SupervisorHelperError("Helper left no response")
        try:
            rpc._parse_response(
                response[0], request["id"], request["op"], request["max_bytes"]
            )
        except rpc.RPCRemoteError:
            expected_code = 1
        except Exception as exc:
            raise SupervisorHelperError(
                "Helper left an invalid typed response"
            ) from exc
        else:
            expected_code = 0
    if code != expected_code:
        raise SupervisorHelperError("Helper exit does not match its typed response")
    # Never unlink either slot: FileRPC consumes the validated response itself.


def _serve(
    process,
    base,
    directory,
    ready,
    parent_namespace,
    helper,
    rpc_path,
    transport,
    allowlist_json,
    deadline,
    outputs,
):
    identity = rpc._private_directory(str(directory))
    pending = None  # (raw bytes, inode identity, pinned descriptor), already answered
    next_id = 1
    try:
        while True:
            _remaining(deadline)
            if any(
                os.fstat(stream.fileno()).st_size > boundary.MAX_OUTPUT_BYTES
                for stream in outputs
            ):
                raise boundary.BoundaryOutputError(
                    "Worker output exceeded the 1 MiB per-stream limit"
                )
            if not _freeze(process, deadline):
                return
            # Stop the client to close the publish/consume race for helper exit 1.
            # As with cleanup, deliberately escaped process groups are not covered.
            _rpc_identity(directory, identity)
            with _snapshot(
                directory / rpc.REQUEST_NAME, rpc.MAX_REQUEST_BYTES
            ) as snapshot:
                response_present = _exists(directory / rpc.RESPONSE_NAME)
                if snapshot is None:
                    if response_present:
                        raise SupervisorProtocolError(
                            "Response present without its request"
                        )
                    if pending is not None:
                        os.close(pending[2])
                        pending = None
                elif pending is not None and snapshot[:2] == pending[:2]:
                    # Response may just have been unlinked, before request unlink.
                    # Never dispatch this ID again, even during that cleanup gap.
                    pass
                else:
                    if response_present:
                        raise SupervisorProtocolError(
                            "Request changed before the response was consumed"
                        )
                    request = _request(snapshot[0], next_id)
                    _ready(ready, parent_namespace)
                    _run_helper(
                        base,
                        directory,
                        identity,
                        snapshot,
                        request,
                        helper,
                        rpc_path,
                        transport,
                        allowlist_json,
                        deadline,
                    )
                    pin = os.dup(
                        snapshot[2]
                    )  # Pin old inode against reuse/replayed replacement.
                    if pending is not None:
                        os.close(pending[2])
                    pending = snapshot[0], snapshot[1], pin
                    next_id += 1
            _remaining(deadline)
            if _exited(process):
                return
            os.killpg(process.pid, signal.SIGCONT)
            time.sleep(min(boundary._POLL_SECONDS, _remaining(deadline)))
    finally:
        if pending is not None:
            os.close(pending[2])
        # On exceptions leave the offline group stopped for the caller's kill.


def _outputs(base):
    stdout, stdout_over = boundary._read_capped(base / "stdout")
    stderr, stderr_over = boundary._read_capped(base / "stderr")
    return (
        stdout.decode("utf-8", errors="replace"),
        stderr.decode("utf-8", errors="replace"),
        stdout_over or stderr_over,
    )


def run_with_public_fetch(worker_path, args, *, timeout, source_root, photo_allowlist):
    """Return the offline worker's CompletedProcess, or raise a BoundaryError.

    No executable, helper path, transport path, environment or RPC directory is
    caller-selectable. ``photo_allowlist`` is trusted configuration, never DDG
    results. ``--rpc-dir`` (including abbreviations) and ``--`` are reserved.
    Timeout includes setup and every helper; no successful return past deadline.
    """
    started = time.monotonic()
    worker = boundary._worker(worker_path)
    root = boundary._source_root(source_root, worker)
    arguments = boundary._arguments(args)
    try:
        seconds = boundary._timeout(timeout)
    except OverflowError as exc:
        raise boundary.BoundaryInputError(
            "timeout is outside the supported range"
        ) from exc
    deadline = started + seconds
    if len(arguments) > boundary.MAX_ARGS - 2:
        raise boundary.BoundaryInputError(
            "args must leave room for the two trusted RPC arguments"
        )
    for argument in arguments:
        flag = argument.partition("=")[0]
        if argument == "--" or (flag.startswith("--") and "--rpc-dir".startswith(flag)):
            raise boundary.BoundaryInputError(
                "-- and --rpc-dir overrides/abbreviations are reserved"
            )
    try:
        _local_file("resume_boundary.py")
        rpc_path = _local_file("resume_rpc.py")
        helper = _local_file("resume_fetch.py")
        transport = rpc._transport_file(
            str(root / "autoshorts" / "modules" / "public_http.py")
        )
        entries = rpc._allowlist(photo_allowlist)
    except (OSError, ValueError, rpc.RPCError) as exc:
        raise boundary.BoundaryInputError(
            f"Invalid trusted configuration: {exc}"
        ) from exc
    allowlist_json = json.dumps(
        [f"https://{host}{prefix}" for host, prefix in entries],
        ensure_ascii=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    if len(allowlist_json) > _MAX_ALLOWLIST_JSON_CHARS:
        raise boundary.BoundaryInputError(
            "photo_allowlist exceeds the helper CLI limit"
        )
    parent_namespace = boundary._require_namespace_support()
    if signal.getsignal(signal.SIGCHLD) != signal.SIG_DFL:
        raise boundary.BoundaryUnavailableError(
            "Supervisor requires exclusive, default SIGCHLD handling"
        )
    _remaining(deadline)

    with tempfile.TemporaryDirectory(
        prefix="autoshorts-supervisor-", dir="/tmp", ignore_cleanup_errors=True
    ) as name:
        base = Path(os.path.realpath(name))
        home, tmp, work, directory = (
            base / leaf for leaf in ("home", "tmp", "work", "rpc")
        )
        for path in (home, tmp, work, directory):
            path.mkdir(mode=0o700)
        bootstrap, ready = base / "bootstrap.py", base / "ready"
        bootstrap.write_text(boundary.BOOTSTRAP_SOURCE, encoding="utf-8")
        command = boundary._command(
            bootstrap, worker, [*arguments, "--rpc-dir", str(directory)]
        )
        env = boundary._environment(home, tmp, ready, root, parent_namespace)
        process = None
        try:
            with (
                open(base / "stdout", "xb") as stdout,
                open(base / "stderr", "xb") as stderr,
            ):
                _remaining(deadline)
                try:
                    try:
                        process = subprocess.Popen(
                            command,
                            stdin=subprocess.DEVNULL,
                            stdout=stdout,
                            stderr=stderr,
                            env=env,
                            cwd=work,
                            shell=False,
                            close_fds=True,
                            start_new_session=True,
                        )
                    except OSError as exc:
                        raise boundary.BoundaryUnavailableError(
                            f"Could not start offline worker: {exc}"
                        ) from exc
                    _serve(
                        process,
                        base,
                        directory,
                        ready,
                        parent_namespace,
                        helper,
                        rpc_path,
                        transport,
                        allowlist_json,
                        deadline,
                        (stdout, stderr),
                    )
                finally:
                    if process is not None:
                        try:
                            boundary._kill_group(process.pid)
                        finally:
                            process.wait()
            stdout_text, stderr_text, overflow = _outputs(base)
            _remaining(deadline)
            _ready(ready, parent_namespace)
            if process.returncode != 0:
                raise boundary.BoundaryFailureError(
                    f"Offline worker exited {process.returncode}"
                )
            if overflow:
                raise boundary.BoundaryOutputError(
                    "Worker output exceeded the 1 MiB per-stream limit"
                )
            if any(
                _exists(directory / leaf)
                for leaf in (
                    rpc.REQUEST_NAME,
                    rpc.RESPONSE_NAME,
                    ".request.tmp",
                    ".response.tmp",
                )
            ):
                raise SupervisorProtocolError("Worker exited with undrained RPC slots")
            _remaining(deadline)
            return subprocess.CompletedProcess(command, 0, stdout_text, stderr_text)
        except boundary.BoundaryError as exc:
            stdout_text, stderr_text, _ = _outputs(base)
            exc.returncode = None if process is None else process.returncode
            exc.stdout, exc.stderr = (
                boundary._tail(stdout_text),
                boundary._tail(stderr_text),
            )
            raise
