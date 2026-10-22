"""HTTP access with a negative allowlist, controlled redirects and fail-fast anti-bot.

Policy: the product pipeline never requests Shopee pages or the Shopee CDN
(the slug is the only product data source). shp.ee may only be accessed by
the short-link resolver, which stops at the Shopee redirect without fetching
the destination. Any CAPTCHA/challenge page aborts instead of retrying.
"""
from __future__ import annotations

import re
from urllib.parse import urljoin, urlsplit

import requests

from .errors import InvalidProductLink, ProviderError, SourceBlocked

_REDIRECTS = {301, 302, 303, 307, 308}
_SHOPEE_RE = re.compile(r"(?:^|\.)shopee\.", re.IGNORECASE)
_CAPTCHA_MARKERS = (
    # Match active interstitials, not ordinary sites that load optional CAPTCHA scripts/forms.
    "verify you are human", "prove you are human", "robot check", "unusual traffic",
    "challenge-platform", "cf-chl-", "checking your browser", "access denied",
    "captcha challenge", "complete the captcha", "complete captcha", "captcha verification",
    "please solve the captcha", "bot verification required", "sign in to confirm you’re not a bot",
)


def normalized_host(url: str) -> str:
    return (urlsplit(url).hostname or "").rstrip(".").lower()


def is_shopee_host(host: str) -> bool:
    return bool(_SHOPEE_RE.search(host.lower()))


def is_shopee_cdn_host(host: str) -> bool:
    host = host.lower().rstrip(".")
    return is_shopee_host(host) or host == "susercontent.com" or host.endswith(".susercontent.com")


def is_shp_ee_host(host: str) -> bool:
    return host.lower().rstrip(".") == "shp.ee"


def assert_safe_request_url(url: str, *, operation: str, allow_shp_ee: bool = False) -> str:
    parts = urlsplit(url)
    host = (parts.hostname or "").lower().rstrip(".")
    if parts.scheme.lower() != "https" or not host:
        raise SourceBlocked(f"Unsafe/invalid URL for {operation}; HTTPS only.")
    if is_shopee_cdn_host(host):
        raise SourceBlocked(f"Blocked by policy: this pipeline never accesses {host}.")
    if is_shp_ee_host(host) and not allow_shp_ee:
        raise SourceBlocked("shp.ee may only be accessed by the short-link redirect resolver.")
    return host


def _looks_like_challenge(response: requests.Response) -> bool:
    ctype = (response.headers.get("content-type") or "").lower()
    if "text/html" not in ctype and "application/json" not in ctype and "text/plain" not in ctype:
        return False
    try:
        body = response.content[:256_000].decode("utf-8", errors="ignore").lower()
    except Exception:
        return False
    return any(marker in body for marker in _CAPTCHA_MARKERS)


def safe_request(session: requests.Session, method: str, url: str, *, operation: str,
                 timeout: tuple[int, int] = (12, 35),
                 allow_shp_ee: bool = False, stop_at_shopee_redirect: bool = False,
                 max_redirects: int = 5, **kwargs) -> requests.Response:
    """Request without automatic redirects; never follow into Shopee/CDN Shopee."""
    current = url
    for _hop in range(max_redirects + 1):
        host = assert_safe_request_url(current, operation=operation, allow_shp_ee=allow_shp_ee)
        if allow_shp_ee and not is_shp_ee_host(host):
            raise InvalidProductLink("The shp.ee redirect pointed to a host the resolver does not allow.")
        response = session.request(method, current, allow_redirects=False, timeout=timeout, **kwargs)
        if response.status_code in _REDIRECTS:
            location = response.headers.get("location")
            if not location:
                raise SourceBlocked(f"Redirect without Location at {host}; stopped without following.")
            target = urljoin(current, location)
            target_host = normalized_host(target)
            if is_shopee_cdn_host(target_host):
                if stop_at_shopee_redirect:
                    # Hand the 3xx back to the resolver; no request was made to the target.
                    response.headers["x-shopee-short-target"] = target
                    return response
                raise SourceBlocked(f"Redirect to {target_host} blocked; no request was made there.")
            if allow_shp_ee and not is_shp_ee_host(target_host):
                raise InvalidProductLink("shp.ee redirected to something that is not a Shopee product link.")
            current = target
            continue
        if response.status_code in (403, 429):
            raise SourceBlocked(f"Source blocked (HTTP {response.status_code}) at {host}; no retry/CAPTCHA/proxy.")
        if _looks_like_challenge(response):
            raise SourceBlocked(f"Challenge/CAPTCHA page detected at {host}; aborting instead of solving.")
        return response
    raise SourceBlocked("Redirect limit exceeded; chain stopped.")


def safe_get(url: str, *, operation: str,
             max_bytes: int | None = None, expected_content_prefix: str | None = None) -> tuple[bytes, requests.Response]:
    session = requests.Session()
    session.trust_env = False
    response = safe_request(session, "GET", url, operation=operation, stream=True)
    if response.status_code >= 400:
        raise SourceBlocked(f"Source answered HTTP {response.status_code} ({normalized_host(url)}).")
    content_type = (response.headers.get("content-type") or "").lower()
    if expected_content_prefix and not content_type.startswith(expected_content_prefix):
        if content_type.startswith("text/html"):
            raise SourceBlocked(f"{normalized_host(url)} returned HTML instead of the expected file.")
    chunks: list[bytes] = []
    total = 0
    for chunk in response.iter_content(1024 * 128):
        if not chunk:
            continue
        total += len(chunk)
        if max_bytes and total > max_bytes:
            raise ProviderError(f"File exceeded the {max_bytes}-byte limit.")
        chunks.append(chunk)
    data = b"".join(chunks)
    sample = data[:256_000].decode("utf-8", errors="ignore").lower()
    if any(marker in sample for marker in _CAPTCHA_MARKERS):
        raise SourceBlocked(f"Challenge/CAPTCHA detected in content from {normalized_host(url)}.")
    return data, response


def resolve_shp_ee(short_url: str) -> str:
    """Resolve only redirects hosted on shp.ee; never request the Shopee destination."""
    host = normalized_host(short_url)
    if not is_shp_ee_host(host):
        raise InvalidProductLink("The resolver only accepts short links on the shp.ee host.")
    session = requests.Session()
    session.trust_env = False
    response = safe_request(
        session, "GET", short_url, operation="resolve_shp_ee",
        allow_shp_ee=True, stop_at_shopee_redirect=True, stream=True,
        headers={"User-Agent": "autoshorts/1.0 (short-link resolver; no Shopee fetch)"},
    )
    target = response.headers.get("x-shopee-short-target")
    if target:
        return target
    location = response.headers.get("location")
    if location:
        target = urljoin(short_url, location)
        if is_shopee_host(normalized_host(target)):
            return target
    raise InvalidProductLink("shp.ee did not return a direct redirect to a Shopee product; nothing beyond shp.ee was accessed.")
