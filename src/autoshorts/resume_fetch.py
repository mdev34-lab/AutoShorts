"""One-request public fetch helper for the filesystem RPC.

Standard library only; load or run by direct file path, never via the app
package. ``fetch_once`` (and the ``main`` CLI) answers exactly one request:

1. every argument is a trusted supervisor argument, never read from the request;
2. ``rpc_path`` must be the canonical ``resume_rpc.py`` beside this file and is
   loaded directly by file path;
3. ``request.json`` and ``response.json`` must share one canonical private (0700)
   directory owned by this user; the request is opened without following links
   or blocking, must be a regular file of at most 16 KiB and is validated by the
   RPC module before the executor is called;
4. a rejected request publishes nothing (CLI exit 2); a valid request that fails
   publishes a bounded typed error response (exit 1); success publishes the typed
   bytes response (exit 0); an occupied or unpublishable response slot exits 3.
   Responses are complete 0600 temp files hard-linked into place, so an existing
   response is never overwritten or followed.

NOT established here (remaining gap): the future supervisor must run this helper
in a fresh clean environment inside a hard-killed process group.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.util
import json
import os
import stat
import sys

REQUEST_NAME = "request.json"
RESPONSE_NAME = "response.json"
RPC_NAME = "resume_rpc.py"
_TEMP_NAME = ".response.tmp"
MAX_REQUEST_BYTES = 16 * 1024
MAX_ERROR_CHARS = 1024
MAX_ALLOWLIST_JSON_CHARS = 64 * 1024
OK_KEYS = frozenset({"id", "op", "ok", "size", "sha256", "body"})
ERROR_KEYS = frozenset({"id", "op", "ok", "error"})
EXIT_OK, EXIT_FETCH_FAILED, EXIT_REJECTED, EXIT_PUBLISH_FAILED = 0, 1, 2, 3
_RPC_MODULE = "_autoshorts_fetch_rpc"


class FetchError(RuntimeError):
    """Base class for helper failures."""


class FetchRejectedError(FetchError, ValueError):
    """Trusted arguments, helper files or the request were rejected; nothing was published."""


class FetchPublishError(FetchError):
    """The response slot is occupied or the response could not be published."""


class FetchExecutionError(FetchError):
    """A valid request failed; a typed error response was published."""


def _rpc_file(value):
    here = os.path.dirname(os.path.realpath(__file__))
    if (
        type(value) is not str
        or not os.path.isabs(value)
        or os.path.realpath(value) != value
    ):
        raise FetchRejectedError("rpc_path must be a canonical absolute path string")
    if os.path.basename(value) != RPC_NAME or os.path.dirname(value) != here:
        raise FetchRejectedError(f"rpc_path must be the {RPC_NAME} beside this helper")
    try:
        mode = os.lstat(value).st_mode
    except OSError as exc:
        raise FetchRejectedError("rpc_path does not exist") from exc
    if not stat.S_ISREG(mode):
        raise FetchRejectedError("rpc_path must be a regular file")
    return value


def _slot(request_path, response_path):
    """Return the private directory holding the fixed request/response names."""
    for value in (request_path, response_path):
        if (
            type(value) is not str
            or not os.path.isabs(value)
            or os.path.normpath(value) != value
        ):
            raise FetchRejectedError(
                "request/response paths must be normalized absolute strings"
            )
    directory = os.path.dirname(request_path)
    if (
        os.path.basename(request_path) != REQUEST_NAME
        or os.path.basename(response_path) != RESPONSE_NAME
        or os.path.dirname(response_path) != directory
    ):
        raise FetchRejectedError(
            f"Paths must be {REQUEST_NAME} and {RESPONSE_NAME} in one directory"
        )
    if os.path.realpath(directory) != directory:
        raise FetchRejectedError("RPC directory must be canonical")
    try:
        info = os.lstat(directory)
    except OSError as exc:
        raise FetchRejectedError("RPC directory does not exist") from exc
    if (
        not stat.S_ISDIR(info.st_mode)
        or info.st_uid != os.getuid()
        or info.st_mode & 0o077
    ):
        raise FetchRejectedError(
            "RPC directory must be a private (0700) directory owned by this user"
        )
    return directory


def _read_request(path):
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise FetchRejectedError(
            f"Cannot open request safely: {exc.strerror or exc}"
        ) from exc
    raw = b""
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_REQUEST_BYTES:
            raise FetchRejectedError(
                f"Request must be a regular file of at most {MAX_REQUEST_BYTES} bytes"
            )
        while len(raw) <= MAX_REQUEST_BYTES:
            chunk = os.read(descriptor, MAX_REQUEST_BYTES + 1 - len(raw))
            if not chunk:
                break
            raw += chunk
    except OSError as exc:
        raise FetchRejectedError(f"Cannot read request: {exc.strerror or exc}") from exc
    finally:
        os.close(descriptor)
    if not 0 < len(raw) <= MAX_REQUEST_BYTES:
        raise FetchRejectedError(f"Request must be 1..{MAX_REQUEST_BYTES} bytes")
    return raw


def _load_rpc(path):
    spec = importlib.util.spec_from_file_location(_RPC_MODULE, path)
    if spec is None or spec.loader is None:
        raise FetchRejectedError("Cannot load rpc module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[_RPC_MODULE] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise FetchRejectedError(
            f"Cannot load rpc module: {type(exc).__name__}"
        ) from exc
    finally:
        sys.modules.pop(_RPC_MODULE, None)
    if (
        not callable(getattr(module, "validate_request", None))
        or not callable(getattr(module, "execute_request", None))
        or getattr(module, "_OK_KEYS", None) != OK_KEYS
        or getattr(module, "_ERROR_KEYS", None) != ERROR_KEYS
    ):
        raise FetchRejectedError("rpc module does not provide the expected protocol")
    return module


def _bounded(text):
    return text[:MAX_ERROR_CHARS]


def _publish(directory, payload, keys):
    if set(payload) != keys:
        raise FetchPublishError("Response does not match the protocol keys")
    raw = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    temp = os.path.join(directory, _TEMP_NAME)
    created = False
    try:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC
        descriptor = os.open(temp, flags, 0o600)
        created = True
        try:
            os.fchmod(descriptor, 0o600)
            view = memoryview(raw)
            while view:
                view = view[os.write(descriptor, view) :]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        # link() publishes the complete file, never replaces an existing response
        # and (follow_symlinks=False) never follows a link.
        os.link(temp, os.path.join(directory, RESPONSE_NAME), follow_symlinks=False)
    except OSError as exc:
        raise FetchPublishError(
            f"Cannot publish response: {exc.strerror or exc}"
        ) from exc
    finally:
        if created:
            try:
                os.unlink(temp)
            except OSError:
                pass


def fetch_once(
    request_path, response_path, *, rpc_path, transport_path, photo_allowlist
):
    """Answer one request with the trusted RPC executor; return None on success.

    Raises ``FetchRejectedError`` (nothing published), ``FetchPublishError``
    (slot occupied or publication failed) or ``FetchExecutionError`` (a typed
    error response was published for a valid request).
    """
    rpc_file = _rpc_file(rpc_path)
    directory = _slot(request_path, response_path)
    raw = _read_request(request_path)
    rpc = _load_rpc(rpc_file)
    try:
        request = rpc.validate_request(raw)
    except Exception as exc:
        raise FetchRejectedError(_bounded(f"Malformed request: {exc}")) from exc
    if (
        type(request) is not dict
        or type(request.get("id")) is not int
        or request.get("op") not in ("ddg", "photo")
        or type(request.get("max_bytes")) is not int
    ):
        raise FetchRejectedError("rpc module returned an invalid request")
    for name in (RESPONSE_NAME, _TEMP_NAME):
        if os.path.lexists(os.path.join(directory, name)):
            raise FetchPublishError(f"Stale response file present: {name}")
    head = {"id": request["id"], "op": request["op"]}
    try:
        body = rpc.execute_request(
            raw, transport_path=transport_path, photo_allowlist=photo_allowlist
        )
        if type(body) is not bytes or not 0 < len(body) <= request["max_bytes"]:
            raise FetchError("Executor must return nonempty bytes within max_bytes")
    except Exception as exc:
        message = _bounded(f"{type(exc).__name__}: {exc}")
        _publish(directory, {**head, "ok": False, "error": message}, ERROR_KEYS)
        raise FetchExecutionError(message) from exc
    _publish(
        directory,
        {
            **head,
            "ok": True,
            "size": len(body),
            "sha256": hashlib.sha256(body).hexdigest(),
            "body": base64.b64encode(body).decode("ascii"),
        },
        OK_KEYS,
    )


def _allowlist_json(text):
    def constant(value):
        raise ValueError(f"Invalid JSON constant: {value}")

    if len(text) > MAX_ALLOWLIST_JSON_CHARS:
        raise FetchRejectedError("--allowlist-json is too long")
    try:
        value = json.loads(text, parse_constant=constant)
    except (ValueError, RecursionError) as exc:
        raise FetchRejectedError("--allowlist-json must be valid JSON") from exc
    if (
        type(value) is not list
        or not value
        or any(type(entry) is not str for entry in value)
    ):
        raise FetchRejectedError(
            "--allowlist-json must be a nonempty JSON list of strings"
        )
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="resume_fetch.py",
        allow_abbrev=False,
        description="Answer one filesystem-RPC public fetch request.",
    )
    for flag in ("--request", "--response", "--rpc", "--transport", "--allowlist-json"):
        parser.add_argument(flag, required=True)
    args = parser.parse_args(argv)
    try:
        fetch_once(
            args.request,
            args.response,
            rpc_path=args.rpc,
            transport_path=args.transport,
            photo_allowlist=_allowlist_json(args.allowlist_json),
        )
    except FetchExecutionError as exc:
        print(
            f"resume_fetch: fetch failed, error response published: {exc}",
            file=sys.stderr,
        )
        return EXIT_FETCH_FAILED
    except FetchPublishError as exc:
        print(f"resume_fetch: {exc}", file=sys.stderr)
        return EXIT_PUBLISH_FAILED
    except FetchRejectedError as exc:
        print(f"resume_fetch: rejected, nothing published: {exc}", file=sys.stderr)
        return EXIT_REJECTED
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
