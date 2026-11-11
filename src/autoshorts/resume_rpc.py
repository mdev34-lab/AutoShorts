"""Filesystem RPC for bounded public bytes requested by an offline worker.

Standard library only; load by direct file path, never via the ``autoshorts``
package. This slice provides only:

* ``FileRPC``: the offline worker's client. It publishes one strictly typed
  request at a time as ``request.json`` in a private directory and waits (hard
  deadline <= 40 s, 50 ms polling) for ``response.json``.
* ``validate_request(raw)``: strict request schema for the future supervisor.
* ``execute_request(raw, *, transport_path, photo_allowlist)``: for a future
  separately isolated, clean-environment one-request process. It loads only the
  trusted transport file and calls only ``get_ddg``/``get_photo``.

Protocol: the supervisor reads ``request.json`` and atomically renames its
answer to ``response.json``; it never deletes either file. The client removes
both after reading. Requests carry no free-form operation, method, header,
credential, module, path, executable or environment field; DDG uses fixed
routes and parameters; photos use the supervisor's authoritative allowlist.

NOT implemented here (remaining gaps): the supervisor, its process isolation,
clean environment and hard kill of online helpers.
"""

from __future__ import annotations

import base64
import hashlib
import importlib.util
import json
import math
import os
import stat
import sys
import time
from urllib.parse import urlsplit

PROTOCOL_VERSION = 1
REQUEST_NAME = "request.json"
RESPONSE_NAME = "response.json"
_TEMP_NAME = ".request.tmp"
MAX_REQUEST_BYTES = 16 * 1024
MAX_REQUESTS = 256
MAX_WAIT_SECONDS = 40.0
POLL_SECONDS = 0.05
MAX_ERROR_CHARS = 1024
MAX_URL_CHARS = 2048
MAX_QUERY_CHARS = 2000
MAX_TOKEN_CHARS = 256
MAX_ALLOWLIST_ENTRIES = 64

DDG_ROOT_URL = "https://duckduckgo.com/"
DDG_IMAGES_URL = "https://duckduckgo.com/i.js"
# Route -> (byte cap, fixed parameter values; None marks a validated variable).
DDG_ROUTES = {
    DDG_ROOT_URL: (262_144, {"q": None}),
    DDG_IMAGES_URL: (
        1_048_576,
        {"o": "json", "q": None, "l": "br-pt", "vqd": None, "p": "1", "ct": "AT"},
    ),
}
DDG_MAX_TIMEOUT = 10.0
PHOTO_MAX_BYTES = 10 * 1024 * 1024
PHOTO_MAX_TIMEOUT = 15.0
PHOTO_MAX_REDIRECTS = 3

_TRANSPORT_MODULE = "_autoshorts_rpc_public_http"
_DDG_KEYS = frozenset(
    {"version", "id", "op", "url", "params", "max_bytes", "timeout", "max_redirects"}
)
_PHOTO_KEYS = _DDG_KEYS - {"params"}
_OK_KEYS = frozenset({"id", "op", "ok", "size", "sha256", "body"})
_ERROR_KEYS = frozenset({"id", "op", "ok", "error"})


class RPCError(RuntimeError):
    """Base class for file RPC failures."""


class RPCRequestError(RPCError, ValueError):
    """A request, configuration or argument is outside the fixed schema."""


class RPCProtocolError(RPCError):
    """Missing, stale, malformed or over-limit RPC files; the client stops."""


class RPCTimeoutError(RPCProtocolError):
    """No response before the hard client deadline."""


class RPCLimitError(RPCError):
    """More requests than allowed for one run."""


class RPCRemoteError(RPCError):
    """The supervisor reported that the bounded fetch failed."""


class RPCTransportError(RPCError):
    """The trusted transport could not be loaded or did not return valid bytes."""


# --- shared strict validators ---------------------------------------------


def _int(value, low, high, label, error=RPCRequestError):
    if type(value) is not int or not low <= value <= high:
        raise error(f"{label} must be an exact integer from {low} through {high}")
    return value


def _seconds(value, high, label):
    if (
        type(value) not in (int, float)
        or not math.isfinite(value)
        or not 0 < value <= high
    ):
        raise RPCRequestError(f"{label} must be a finite number > 0 and <= {high}")
    return float(value)


def _query(value):
    if (
        type(value) is not str
        or not value.strip()
        or len(value) > MAX_QUERY_CHARS
        or any(not char.isprintable() and char != "\t" for char in value)
    ):
        raise RPCRequestError(
            f"q must be a printable nonblank string of at most {MAX_QUERY_CHARS} chars"
        )
    return value


def _token(value):
    if (
        type(value) is not str
        or not value
        or len(value) > MAX_TOKEN_CHARS
        or any(
            not char.isprintable() or char.isspace() or char in "\"'<>&"
            for char in value
        )
    ):
        raise RPCRequestError(
            "vqd must be a short printable token without spaces or quotes"
        )
    return value


def _ddg(url, params, max_bytes, timeout, max_redirects):
    if type(url) is not str or url not in DDG_ROUTES:
        raise RPCRequestError("DDG url must be one of the fixed routes")
    cap, fixed = DDG_ROUTES[url]
    if type(params) is not dict or set(params) != set(fixed):
        raise RPCRequestError(f"DDG params must have exactly the keys {sorted(fixed)}")
    for key, expected in fixed.items():
        value = params[key]
        if expected is not None and (type(value) is not str or value != expected):
            raise RPCRequestError(f"DDG param {key} must be {expected!r}")
    _query(params["q"])
    if "vqd" in params:
        _token(params["vqd"])
    return {
        "op": "ddg",
        "url": url,
        "params": dict(params),
        "max_bytes": _int(max_bytes, 1, cap, "DDG max_bytes"),
        "timeout": _seconds(timeout, DDG_MAX_TIMEOUT, "DDG timeout"),
        "max_redirects": _int(max_redirects, 0, 0, "DDG max_redirects"),
    }


def _https_parts(value, label):
    """(host, path) of a strict https URL; no userinfo, port, fragment or traversal."""
    if (
        type(value) is not str
        or not 0 < len(value) <= MAX_URL_CHARS
        or any(ord(char) <= 32 or ord(char) >= 127 for char in value)
        or "\\" in value
    ):
        raise RPCRequestError(
            f"{label} must be a printable ASCII URL of at most {MAX_URL_CHARS} chars"
        )
    try:
        parts = urlsplit(value)
        port = parts.port
    except ValueError as exc:
        raise RPCRequestError(f"Malformed {label}") from exc
    host = parts.hostname or ""
    if (
        parts.scheme != "https"
        or parts.fragment
        or port is not None
        or parts.netloc != host
    ):
        raise RPCRequestError(
            f"{label} must be https with a plain lowercase host and no fragment"
        )
    if (
        "." not in host
        or host.replace(".", "").isdigit()
        or host.startswith(".")
        or host.endswith(".")
        or any(char not in "abcdefghijklmnopqrstuvwxyz0123456789.-" for char in host)
    ):
        raise RPCRequestError(
            f"{label} host must be a plain DNS name (no wildcard or IP)"
        )
    path = parts.path
    lowered = path.lower()
    segments = path.split("/")[1:]
    if (
        not path.startswith("/")
        or "//" in path
        or any(seg in (".", "..") for seg in segments)
        or any(code in lowered for code in ("%2e", "%2f", "%5c", "%00"))
    ):
        raise RPCRequestError(
            f"{label} path must be absolute without traversal or encoded separators"
        )
    return host, path, parts.query


def _photo(url, max_bytes, timeout, max_redirects):
    _https_parts(url, "photo url")
    return {
        "op": "photo",
        "url": url,
        "max_bytes": _int(max_bytes, 1, PHOTO_MAX_BYTES, "photo max_bytes"),
        "timeout": _seconds(timeout, PHOTO_MAX_TIMEOUT, "photo timeout"),
        "max_redirects": _int(
            max_redirects, 0, PHOTO_MAX_REDIRECTS, "photo max_redirects"
        ),
    }


def _allowlist(value):
    """Parse authoritative ``https://host/prefix/`` strings into ``(host, prefix)`` tuples."""
    if type(value) not in (list, tuple) or not 0 < len(value) <= MAX_ALLOWLIST_ENTRIES:
        raise RPCRequestError(
            f"photo_allowlist must be a list/tuple of 1..{MAX_ALLOWLIST_ENTRIES} entries"
        )
    entries = []
    for entry in value:
        host, path, query = _https_parts(entry, "photo_allowlist entry")
        if query or "?" in entry or not path.endswith("/"):
            raise RPCRequestError(
                "photo_allowlist entries must be query-free prefixes ending in '/'"
            )
        entries.append((host, path))
    if len(set(entries)) != len(entries):
        raise RPCRequestError("photo_allowlist entries must be unique")
    return entries


def _duplicate_free(pairs, error):
    result = {}
    for key, value in pairs:
        if key in result:
            raise error(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _loads(raw, error):
    def constant(value):
        raise error(f"Invalid JSON constant: {value}")

    try:
        return json.loads(
            raw.decode("utf-8"),
            parse_constant=constant,
            object_pairs_hook=lambda pairs: _duplicate_free(pairs, error),
        )
    except (UnicodeDecodeError, ValueError) as exc:
        if isinstance(exc, error):
            raise
        raise error("Malformed JSON") from exc


def _encode(payload):
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def validate_request(raw):
    """Strictly parse one request file; return a normalized request dict."""
    if type(raw) is not bytes or not 0 < len(raw) <= MAX_REQUEST_BYTES:
        raise RPCRequestError(f"Request must be 1..{MAX_REQUEST_BYTES} bytes")
    data = _loads(raw, RPCRequestError)
    if type(data) is not dict:
        raise RPCRequestError("Request root must be an object")
    op = data.get("op")
    keys = {"ddg": _DDG_KEYS, "photo": _PHOTO_KEYS}.get(op) if type(op) is str else None
    if keys is None or set(data) != keys:
        raise RPCRequestError("Request must be exactly a ddg or photo request")
    _int(data["version"], PROTOCOL_VERSION, PROTOCOL_VERSION, "version")
    request_id = _int(data["id"], 1, MAX_REQUESTS, "id")
    if op == "ddg":
        fields = _ddg(
            data["url"],
            data["params"],
            data["max_bytes"],
            data["timeout"],
            data["max_redirects"],
        )
    else:
        fields = _photo(
            data["url"], data["max_bytes"], data["timeout"], data["max_redirects"]
        )
    return {"version": PROTOCOL_VERSION, "id": request_id, **fields}


# --- one-request executor (future isolated online process) ---------------


def _transport_file(value):
    if (
        type(value) is not str
        or not os.path.isabs(value)
        or os.path.realpath(value) != value
    ):
        raise RPCRequestError("transport_path must be a canonical absolute path string")
    parts = value.split(os.sep)
    if parts[-3:] != ["autoshorts", "modules", "public_http.py"]:
        raise RPCRequestError(
            "transport_path must be .../autoshorts/modules/public_http.py"
        )
    try:
        mode = os.lstat(value).st_mode
    except OSError as exc:
        raise RPCRequestError("transport_path does not exist") from exc
    if not stat.S_ISREG(mode):
        raise RPCRequestError("transport_path must be a regular file")
    return value


def _load_transport(path):
    spec = importlib.util.spec_from_file_location(_TRANSPORT_MODULE, path)
    if spec is None or spec.loader is None:
        raise RPCTransportError("Cannot load transport file")
    module = importlib.util.module_from_spec(spec)
    sys.modules[_TRANSPORT_MODULE] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise RPCTransportError(
            f"Transport import failed: {type(exc).__name__}"
        ) from exc
    finally:
        sys.modules.pop(_TRANSPORT_MODULE, None)
    return module


def execute_request(raw, *, transport_path, photo_allowlist):
    """Validate ``raw`` and perform it with the trusted transport; return bytes.

    Only for a separately isolated clean-environment process. The request
    cannot name a transport, method or allowlist; ``photo_allowlist`` from the
    trusted caller is authoritative and checked before any transport call.
    """
    request = validate_request(raw)
    path = _transport_file(transport_path)
    entries = _allowlist(photo_allowlist)
    if request["op"] == "photo":
        host, url_path, _ = _https_parts(request["url"], "photo url")
        if not any(
            host == allowed and url_path.startswith(prefix)
            for allowed, prefix in entries
        ):
            raise RPCRequestError("photo url is outside the supervisor photo_allowlist")
    module = _load_transport(path)
    client_class = getattr(module, "BoundedPublicClient", None)
    validate = getattr(module, "validate_photo_allowlist", None)
    if not callable(client_class) or (
        request["op"] == "photo" and not callable(validate)
    ):
        raise RPCTransportError(
            "Transport lacks BoundedPublicClient/validate_photo_allowlist"
        )
    limits = {
        "max_bytes": request["max_bytes"],
        "timeout": request["timeout"],
        "max_redirects": request["max_redirects"],
    }
    try:
        client = client_class()
        if request["op"] == "ddg":
            body = client.get_ddg(request["url"], dict(request["params"]), **limits)
        else:
            validate(list(entries))
            body = client.get_photo(request["url"], list(entries), **limits)
    except Exception as exc:
        raise RPCTransportError(
            f"{type(exc).__name__}: {exc}"[:MAX_ERROR_CHARS]
        ) from exc
    if type(body) is not bytes or not 0 < len(body) <= request["max_bytes"]:
        raise RPCTransportError("Transport must return nonempty bytes within max_bytes")
    return body


# --- offline worker client ------------------------------------------------


def _private_directory(value):
    if (
        type(value) is not str
        or not os.path.isabs(value)
        or os.path.realpath(value) != value
    ):
        raise RPCRequestError("RPC directory must be a canonical absolute path string")
    try:
        info = os.lstat(value)
    except OSError as exc:
        raise RPCRequestError("RPC directory does not exist") from exc
    if (
        not stat.S_ISDIR(info.st_mode)
        or info.st_uid != os.getuid()
        or info.st_mode & 0o077
    ):
        raise RPCRequestError(
            "RPC directory must be a private (0700) directory owned by this user"
        )
    return info.st_dev, info.st_ino


class FileRPC:
    """One-outstanding-request client with fixed file names in a private dir."""

    def __init__(
        self,
        directory,
        *,
        wait_seconds=MAX_WAIT_SECONDS,
        clock=time.monotonic,
        sleep=time.sleep,
    ):
        self._identity = _private_directory(directory)
        self._directory = directory
        self._wait = _seconds(wait_seconds, MAX_WAIT_SECONDS, "wait_seconds")
        if not callable(clock) or not callable(sleep):
            raise RPCRequestError("clock and sleep must be callable")
        self._clock, self._sleep = clock, sleep
        self._next_id = 1
        self._broken = False

    def get_ddg(
        self,
        url,
        params=None,
        *,
        max_bytes=None,
        timeout=DDG_MAX_TIMEOUT,
        max_redirects=0,
    ):
        if max_bytes is None and type(url) is str and url in DDG_ROUTES:
            max_bytes = DDG_ROUTES[url][0]
        return self._call(_ddg(url, params, max_bytes, timeout, max_redirects))

    def get_photo(
        self,
        url,
        allowlist=None,
        *,
        max_bytes=PHOTO_MAX_BYTES,
        timeout=PHOTO_MAX_TIMEOUT,
        max_redirects=PHOTO_MAX_REDIRECTS,
    ):
        if allowlist is not None:
            raise RPCRequestError(
                "The supervisor photo allowlist is authoritative; do not pass one"
            )
        return self._call(_photo(url, max_bytes, timeout, max_redirects))

    def _path(self, name):
        return os.path.join(self._directory, name)

    def _call(self, fields):
        if self._broken:
            raise RPCProtocolError(
                "FileRPC is unusable after an earlier protocol failure"
            )
        if self._next_id > MAX_REQUESTS:
            raise RPCLimitError(f"At most {MAX_REQUESTS} requests per run")
        raw = _encode({"version": PROTOCOL_VERSION, "id": self._next_id, **fields})
        validate_request(raw)
        request_id = self._next_id
        self._next_id += 1
        try:
            self._publish(raw)
            return self._await(request_id, fields["op"], fields["max_bytes"])
        except RPCRemoteError:
            raise
        except OSError as exc:
            self._broken = True
            raise RPCProtocolError(f"RPC file operation failed: {exc}") from exc
        except BaseException:
            self._broken = True
            raise

    def _publish(self, raw):
        if _private_directory(self._directory) != self._identity:
            raise RPCProtocolError("RPC directory changed")
        for name in (REQUEST_NAME, RESPONSE_NAME, _TEMP_NAME):
            if os.path.lexists(self._path(name)):
                raise RPCProtocolError(f"Stale RPC file present: {name}")
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
        descriptor = os.open(self._path(_TEMP_NAME), flags, 0o600)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
            # link refuses an existing request; the request appears complete.
            os.link(self._path(_TEMP_NAME), self._path(REQUEST_NAME))
        finally:
            try:
                os.unlink(self._path(_TEMP_NAME))
            except FileNotFoundError:
                pass

    def _await(self, request_id, op, max_bytes):
        deadline = self._clock() + self._wait
        limit = 4 * math.ceil(max_bytes / 3) + 4096
        while True:
            try:
                # O_NONBLOCK: a FIFO must not block open() past the deadline before fstat.
                flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
                descriptor = os.open(self._path(RESPONSE_NAME), flags)
            except FileNotFoundError:
                if self._clock() >= deadline:
                    try:
                        os.unlink(self._path(REQUEST_NAME))
                    except FileNotFoundError:
                        pass
                    raise RPCTimeoutError(
                        f"No response to request {request_id} within {self._wait:g}s"
                    ) from None
                self._sleep(POLL_SECONDS)
                continue
            except OSError as exc:
                raise RPCProtocolError(
                    "Response must be a regular file, not a symlink"
                ) from exc
            try:
                if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                    raise RPCProtocolError("Response must be a regular file")
                stream = os.fdopen(descriptor, "rb")
            except BaseException:
                os.close(descriptor)
                raise
            with stream:
                raw = stream.read(limit + 1)
            # The client owns cleanup; a vanished request is a protocol error.
            os.unlink(self._path(RESPONSE_NAME))
            os.unlink(self._path(REQUEST_NAME))
            if len(raw) > limit:
                raise RPCProtocolError("Response file exceeds its size limit")
            return _parse_response(raw, request_id, op, max_bytes)


def _parse_response(raw, request_id, op, max_bytes):
    data = _loads(raw, RPCProtocolError)
    if type(data) is not dict or type(data.get("ok")) is not bool:
        raise RPCProtocolError("Response must be an object with boolean ok")
    if set(data) != (_OK_KEYS if data["ok"] else _ERROR_KEYS):
        raise RPCProtocolError("Response has unexpected or missing keys")
    if type(data["id"]) is not int or data["id"] != request_id or data["op"] != op:
        raise RPCProtocolError("Stale or mismatched response id/op")
    if not data["ok"]:
        error = data["error"]
        if type(error) is not str or not error.strip() or len(error) > MAX_ERROR_CHARS:
            raise RPCProtocolError("Response error must be a short nonblank string")
        raise RPCRemoteError(f"Public fetch failed: {error}")
    size = _int(data["size"], 1, max_bytes, "Response size", RPCProtocolError)
    digest, encoded = data["sha256"], data["body"]
    if (
        type(digest) is not str
        or len(digest) != 64
        or any(c not in "0123456789abcdef" for c in digest)
    ):
        raise RPCProtocolError("Response sha256 must be 64 lowercase hex chars")
    if type(encoded) is not str:
        raise RPCProtocolError("Response body must be base64 text")
    try:
        body = base64.b64decode(encoded, validate=True)
    except ValueError as exc:
        raise RPCProtocolError("Invalid base64 response body") from exc
    if base64.b64encode(body).decode("ascii") != encoded:
        raise RPCProtocolError("Non-canonical base64 response body")
    if len(body) != size or hashlib.sha256(body).hexdigest() != digest:
        raise RPCProtocolError("Response body size or sha256 mismatch")
    return body
