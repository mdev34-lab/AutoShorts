"""Bounded public HTTP transport with strict safety boundaries.

Enforces:
- requests.Session with trust_env=False (no ambient credentials/proxies).
- HTTPS on port 443 only, zero URL credentials.
- HTTP GET only.
- Exact duckduckgo.com endpoint validation for search (/ or /i.js).
- Explicit tuple(host, pathprefix) photo allowlist supplied by caller (no wildcards, no shared storage roots).
- Hop-by-hop redirect revalidation.
- Zero cookies or auth headers stored or transmitted.
- Strict bytecap and timecap bounds.
"""

from __future__ import annotations

import ipaddress
import re
import socket
import time
from typing import Any, Sequence, cast
from urllib.parse import urljoin, urlsplit

import requests
import requests.adapters
import urllib3
import urllib3.connection
import urllib3.connectionpool
import urllib3.poolmanager


class PublicHttpError(ValueError):
    """Base error for public HTTP boundary violations."""


class URLSecurityError(PublicHttpError):
    """URL violates HTTPS/port/credentials or endpoint constraints."""


class AllowlistDeniedError(PublicHttpError):
    """Target host or path is not permitted by explicit allowlist."""


class RedirectDeniedError(PublicHttpError):
    """Redirect target violated safety boundary or hop limit."""


class ByteCapExceededError(PublicHttpError):
    """Response size exceeded maximum byte cap."""


class TimeCapExceededError(PublicHttpError):
    """Request exceeded timeout cap."""


_HOST_RE = re.compile(r"^[a-z0-9-]+(\.[a-z0-9-]+)+$")
_FORBIDDEN_PERCENT_SEQUENCES = ("%2e", "%2f", "%5c", "%25")

MAX_BYTES_CEILING = 20_971_520
MAX_TIMEOUT_CEILING = 30.0
MAX_REDIRECTS_CEILING = 5


def _is_valid_dns_host(host: str) -> bool:
    if not _HOST_RE.fullmatch(host):
        return False
    last_label = host.rsplit(".", 1)[-1]
    if not any(ch.isalpha() for ch in last_label):
        return False
    if (
        host == "localhost"
        or host.endswith(".localhost")
        or host.endswith(".local")
        or host.endswith(".internal")
    ):
        return False
    try:
        ipaddress.ip_address(host)
    except ValueError:
        return True
    return False


def validate_caps(max_bytes, timeout, max_redirects):
    """Validate bounded client caps against module ceilings."""
    if not (0 < max_bytes <= MAX_BYTES_CEILING):
        raise PublicHttpError(
            f"max_bytes must be >0 and <= {MAX_BYTES_CEILING}, got {max_bytes!r}"
        )
    if not (0 < timeout <= MAX_TIMEOUT_CEILING):
        raise PublicHttpError(
            f"timeout must be >0 and <= {MAX_TIMEOUT_CEILING}, got {timeout!r}"
        )
    if not (0 <= max_redirects <= MAX_REDIRECTS_CEILING):
        raise PublicHttpError(
            f"max_redirects must be >=0 and <= {MAX_REDIRECTS_CEILING}, got {max_redirects!r}"
        )


_SHARED_STORAGE_HOSTS = frozenset(
    {
        "s3.amazonaws.com",
        "storage.googleapis.com",
        "blob.core.windows.net",
        "cloudfront.net",
        "objects.githubusercontent.com",
        "r2.cloudflarestorage.com",
    }
)


def validate_photo_allowlist(
    allowlist: Sequence[tuple[str, str]],
) -> list[tuple[str, str]]:
    """Validate and normalize explicit (host, pathprefix) tuples."""
    if not allowlist:
        raise AllowlistDeniedError(
            "Explicit photo allowlist must be supplied and non-empty"
        )

    cleaned: list[tuple[str, str]] = []
    for entry in allowlist:
        if not isinstance(entry, (tuple, list)) or len(entry) != 2:
            raise AllowlistDeniedError(
                f"Each allowlist entry must be a (host, pathprefix) tuple, got {entry!r}"
            )
        host, prefix = entry
        if not isinstance(host, str) or not isinstance(prefix, str):
            raise AllowlistDeniedError("Host and pathprefix must be strings")
        host_clean = host.strip().lower().rstrip(".")
        prefix_clean = prefix.strip()

        if not host_clean or "*" in host_clean or "?" in host_clean:
            raise AllowlistDeniedError(
                f"Wildcards and empty hosts forbidden in photo allowlist: {host!r}"
            )
        if not _is_valid_dns_host(host_clean):
            raise AllowlistDeniedError(f"Invalid DNS host in photo allowlist: {host!r}")
        if any(tok in prefix_clean for tok in ("..", "%", "?", "#", "\\")):
            raise AllowlistDeniedError(
                f"Unsafe characters in photo allowlist prefix: {prefix!r}"
            )
        if not prefix_clean or prefix_clean == "/":
            if host_clean in _SHARED_STORAGE_HOSTS or any(
                host_clean.endswith("." + s) for s in _SHARED_STORAGE_HOSTS
            ):
                raise AllowlistDeniedError(
                    f"Shared storage host {host_clean!r} requires explicit specific pathprefix, not root '/'"
                )
        if not prefix_clean.startswith("/"):
            prefix_clean = "/" + prefix_clean

        cleaned.append((host_clean, prefix_clean))
    return cleaned


def assert_safe_https_url(url: str) -> tuple[str, str]:
    """Ensure URL is HTTPS on port 443 with no embedded userinfo/credentials."""
    if not isinstance(url, str) or not url.strip():
        raise URLSecurityError("URL must be a non-empty string")

    if any(ch.isspace() for ch in url):
        raise URLSecurityError("URL must not contain whitespace")
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in url):
        raise URLSecurityError("URL must not contain control characters")
    if "\\" in url:
        raise URLSecurityError("URL must not contain backslash")
    if "#" in url:
        raise URLSecurityError("URL must not contain '#'")

    parts = urlsplit(url)
    scheme = parts.scheme.lower()
    if scheme != "https":
        raise URLSecurityError(f"Only HTTPS is permitted, got scheme {scheme!r}")

    if parts.username or parts.password:
        raise URLSecurityError(
            "URL credentials (username/password) are strictly forbidden"
        )

    host = (parts.hostname or "").lower().rstrip(".")
    if not host:
        raise URLSecurityError("URL missing valid hostname")

    if not _is_valid_dns_host(host):
        raise URLSecurityError(f"Host is not a valid DNS name: {host!r}")

    port = parts.port
    if port is not None and port != 443:
        raise URLSecurityError(f"Only port 443 is permitted for HTTPS, got port {port}")

    path = parts.path or "/"
    lowered_path = path.lower()
    for seq in _FORBIDDEN_PERCENT_SEQUENCES:
        if seq in lowered_path:
            raise URLSecurityError(
                f"Path contains forbidden percent-encoded sequence {seq!r}"
            )
    for segment in path.split("/"):
        if segment in (".", ".."):
            raise URLSecurityError(f"Path contains forbidden dot segment {segment!r}")

    return host, path


def validate_ddg_endpoint(url: str) -> None:
    """Verify URL targets exact duckduckgo.com host with path / or /i.js."""
    host, path = assert_safe_https_url(url)
    if host != "duckduckgo.com":
        raise URLSecurityError(
            f"DuckDuckGo search requires exact host 'duckduckgo.com', got {host!r}"
        )
    if path not in ("/", "/i.js"):
        raise URLSecurityError(
            f"DuckDuckGo search requires exact path '/' or '/i.js', got {path!r}"
        )


def validate_photo_target(
    url: str, allowlist: Sequence[tuple[str, str]]
) -> tuple[str, str]:
    """Verify URL matches an explicit (host, pathprefix) allowlist entry."""
    host, path = assert_safe_https_url(url)
    normalized_allowlist = validate_photo_allowlist(allowlist)

    matched = False
    for allowed_host, allowed_prefix in normalized_allowlist:
        if host == allowed_host and (
            path == allowed_prefix.rstrip("/")
            or path.startswith(
                allowed_prefix if allowed_prefix.endswith("/") else allowed_prefix + "/"
            )
        ):
            matched = True
            break

    if not matched:
        raise AllowlistDeniedError(
            f"Target URL ({host}{path}) not permitted by explicit photo allowlist"
        )
    return host, path


def _default_resolver(host: str, port: int) -> list[str]:
    # AF_UNSPEC stream lookups return IP sockaddrs whose first field is text.
    # A cast never coerces malformed values; _resolve_public still rejects them.
    return [
        cast(str, ai[4][0])
        for ai in socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    ]


def _resolve_public(host, port, resolver=None):
    """Resolve host and return the first publicly routable IP; deny mixed/private."""
    if resolver is None:
        resolver = _default_resolver
    try:
        answers = resolver(host, port)
    except Exception as exc:
        raise URLSecurityError(f"DNS resolution failed for {host!r}: {exc}") from exc

    if not answers:
        raise URLSecurityError(f"DNS resolution returned no answers for {host!r}")

    first_ip = None
    for answer in answers:
        if not isinstance(answer, str):
            raise URLSecurityError(f"Unparsable DNS answer: {answer!r}")
        try:
            addr = ipaddress.ip_address(answer)
        except ValueError:
            raise URLSecurityError(f"Unparsable DNS answer: {answer!r}") from None

        if isinstance(addr, ipaddress.IPv6Address) and addr.ipv4_mapped is not None:
            mapped = addr.ipv4_mapped
            if not mapped.is_global:
                raise URLSecurityError(
                    f"DNS answer is not publicly routable: {answer!r}"
                )

        if not addr.is_global or addr.is_multicast:
            raise URLSecurityError(f"DNS answer is not publicly routable: {answer!r}")

        if first_ip is None:
            first_ip = answer

    return first_ip


class _PinnedHTTPSConnection(urllib3.connection.HTTPSConnection):
    """HTTPSConnection that pins to a pre-validated public IP."""

    resolver = staticmethod(_default_resolver)

    def __init__(self, *args, resolver=None, **kwargs):
        super().__init__(*args, **kwargs)
        if resolver is not None:
            self.resolver = resolver

    def _new_conn(self):
        resolver = getattr(self, "resolver", None)
        if resolver is None:
            resolver = getattr(type(self), "resolver", None)
        if resolver is None:
            resolver = _default_resolver

        host_for_dns = self._dns_host if hasattr(self, "_dns_host") else self.host
        validated_ip = _resolve_public(host_for_dns, self.port, resolver)

        sock = socket.create_connection((validated_ip, self.port), self.timeout)

        try:
            peer = sock.getpeername()[0]
            peer_addr = ipaddress.ip_address(peer)
            validated_addr = ipaddress.ip_address(validated_ip)
        except (OSError, ValueError):
            sock.close()
            raise

        if peer_addr != validated_addr or not peer_addr.is_global:
            sock.close()
            raise URLSecurityError(
                f"Connected peer {peer!r} does not match validated IP {validated_ip!r}"
            )

        return sock


class _DenyHttpAdapter(requests.adapters.HTTPAdapter):
    """HTTPAdapter that denies all plain-HTTP requests."""

    def send(
        self, request, stream=False, timeout=None, verify=True, cert=None, proxies=None
    ):
        raise URLSecurityError("Plain HTTP is not permitted")


class _PinnedAdapter(requests.adapters.HTTPAdapter):
    """HTTPAdapter that pins HTTPS connections to validated public IPs."""

    def __init__(self, resolver=None, *args, **kwargs):
        self.resolver = resolver if resolver is not None else _default_resolver
        super().__init__(*args, **kwargs)

    def init_poolmanager(self, connections, maxsize, block=False, **pool_kwargs):
        bound_resolver = self.resolver

        class _Conn(_PinnedHTTPSConnection):
            resolver = staticmethod(bound_resolver)

        class _Pool(urllib3.connectionpool.HTTPSConnectionPool):
            ConnectionCls = _Conn

        super().init_poolmanager(connections, maxsize, block, **pool_kwargs)
        self.poolmanager.pool_classes_by_scheme = {
            **urllib3.poolmanager.pool_classes_by_scheme,
            "https": _Pool,
        }


class BoundedPublicClient:
    """Bounded public HTTP client with no ambient credentials, strict caps, and hop revalidation."""

    def __init__(self, resolver=None):
        if resolver is None:
            resolver = _default_resolver
        self.resolver = resolver
        self.session = requests.Session()
        self.session.mount("https://", _PinnedAdapter(resolver))
        self.session.mount("http://", _DenyHttpAdapter())
        self._reset_session()

    def _reset_session(self):
        session = self.session
        session.trust_env = False
        session.auth = None
        session.proxies.clear()
        session.headers.clear()
        session.cookies.clear()
        session.verify = True

    def get_ddg(
        self,
        url: str,
        params: dict[str, Any] | None = None,
        *,
        max_bytes: int = 1_048_576,
        timeout: float = 10.0,
        max_redirects: int = 3,
    ) -> bytes:
        """Execute bounded GET against exact duckduckgo.com / or /i.js endpoint."""
        validate_caps(max_bytes, timeout, max_redirects)
        validate_ddg_endpoint(url)
        return self._bounded_get(
            url,
            params=params,
            validator=validate_ddg_endpoint,
            max_bytes=max_bytes,
            timeout=timeout,
            max_redirects=max_redirects,
        )

    def get_photo(
        self,
        url: str,
        allowlist: Sequence[tuple[str, str]],
        *,
        max_bytes: int = 10_485_760,
        timeout: float = 15.0,
        max_redirects: int = 3,
    ) -> bytes:
        """Execute bounded GET to download photo strictly matching supplied allowlist."""
        validate_caps(max_bytes, timeout, max_redirects)
        validated_allowlist = validate_photo_allowlist(allowlist)
        validate_photo_target(url, validated_allowlist)

        def _val(target_url: str):
            validate_photo_target(target_url, validated_allowlist)

        return self._bounded_get(
            url,
            params=None,
            validator=_val,
            max_bytes=max_bytes,
            timeout=timeout,
            max_redirects=max_redirects,
            require_image_content_type=True,
        )

    def _bounded_get(
        self,
        url: str,
        *,
        params: dict[str, Any] | None,
        validator,
        max_bytes: int,
        timeout: float,
        max_redirects: int,
        require_image_content_type: bool = False,
    ) -> bytes:
        deadline = time.monotonic() + timeout
        current_url = url
        current_params = params

        for hop in range(max_redirects + 1):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeCapExceededError("Request exceeded timeout cap")
            validator(current_url)

            self._reset_session()
            headers = {
                "User-Agent": "autoshorts/1.0 (public bounded client; no auth)",
                "Accept-Encoding": "identity",
            }

            try:
                resp = self.session.request(
                    "GET",
                    current_url,
                    params=current_params,
                    headers=headers,
                    allow_redirects=False,
                    timeout=(remaining, remaining),
                    stream=True,
                    proxies={},
                )
            except requests.Timeout as exc:
                raise TimeCapExceededError(
                    f"Request timed out ({timeout}s): {exc}"
                ) from exc
            except requests.RequestException as exc:
                raise PublicHttpError(f"HTTP request failed: {exc}") from exc

            self.session.cookies.clear()

            if resp.status_code in (301, 302, 303, 307, 308):
                resp.close()
                if hop >= max_redirects:
                    raise RedirectDeniedError(
                        f"Redirect limit ({max_redirects}) exceeded"
                    )
                loc = resp.headers.get("Location")
                if not loc:
                    raise RedirectDeniedError(
                        f"Redirect missing Location header at {current_url}"
                    )
                next_url = urljoin(current_url, loc)
                validator(next_url)
                current_url = next_url
                current_params = None
                continue

            if resp.status_code != 200:
                resp.close()
                raise PublicHttpError(
                    f"Server returned HTTP {resp.status_code} for {current_url}"
                )

            if require_image_content_type:
                ctype = (resp.headers.get("Content-Type") or "").lower()
                if not ctype.startswith("image/"):
                    resp.close()
                    raise PublicHttpError(
                        f"Expected image/* content-type, got {ctype!r}"
                    )

            cl_header = resp.headers.get("Content-Length")
            if cl_header is not None:
                try:
                    cl_value = int(cl_header)
                except (TypeError, ValueError):
                    cl_value = None
                if cl_value is not None and cl_value > max_bytes:
                    resp.close()
                    raise ByteCapExceededError(
                        f"Response exceeded maximum byte cap of {max_bytes} bytes"
                    )

            chunks: list[bytes] = []
            total = 0
            try:
                for chunk in resp.iter_content(chunk_size=32768):
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeCapExceededError("Request exceeded timeout cap")
                    if not chunk:
                        continue
                    total += len(chunk)
                    if total > max_bytes:
                        raise ByteCapExceededError(
                            f"Response exceeded maximum byte cap of {max_bytes} bytes"
                        )
                    chunks.append(chunk)
            finally:
                resp.close()

            if not chunks:
                raise PublicHttpError("Response body was empty")

            return b"".join(chunks)

        raise RedirectDeniedError("Redirect limit exceeded")
