"""Public image-source discovery; Shopee domains are never consulted or returned."""
from __future__ import annotations

import os
from dataclasses import dataclass
from urllib.parse import urlsplit

from .errors import ProviderError, SourceBlocked
from .network import is_shopee_cdn_host

_PROXY_ENV_NAMES = {"http_proxy", "https_proxy", "all_proxy", "ftp_proxy", "ddgs_proxy"}


@dataclass(frozen=True)
class ImageCandidate:
    image_url: str
    page_url: str
    title: str
    source: str
    aliexpress: bool = False


def search_image_candidates(product_name: str, *, limit: int = 12) -> list[ImageCandidate]:
    """Public image search; candidates from Shopee or its CDN are filtered out."""
    try:
        from ddgs import DDGS
    except ImportError as exc:
        raise ProviderError("Install the ddgs package to search alternative image sources.") from exc

    query = f"{product_name} product photos"
    saved_proxy_env = {k: v for k, v in os.environ.items() if k.lower() in _PROXY_ENV_NAMES}
    try:
        for key in list(os.environ):
            if key.lower() in _PROXY_ENV_NAMES:
                os.environ.pop(key, None)
        raw = list(DDGS(proxy=None, timeout=10).images(query, max_results=limit))
    except Exception as exc:
        text = str(exc).lower()
        if any(mark in text for mark in ("captcha", "403", "429", "rate limit", "blocked", "robot")):
            raise SourceBlocked("DuckDuckGo blocked the image search; aborting without retry/CAPTCHA/proxy.") from exc
        if "no results" in text:
            return []
        raise ProviderError(f"Public image search failed: {exc}") from exc
    finally:
        for key, value in saved_proxy_env.items():
            os.environ[key] = value

    candidates: list[ImageCandidate] = []
    for entry in raw:
        image_url = str(entry.get("image") or "")
        page_url = str(entry.get("url") or "")
        title = str(entry.get("title") or "(untitled)")
        if not image_url.startswith("https://") or not page_url.startswith("https://"):
            continue
        image_host = (urlsplit(image_url).hostname or "").lower()
        page_host = (urlsplit(page_url).hostname or "").lower()
        # Shopee and its CDN are excluded even when the search engine returns them.
        if is_shopee_cdn_host(image_host) or is_shopee_cdn_host(page_host):
            continue
        aliexpress = "aliexpress." in page_host or page_host.endswith("alicdn.com") or ".alicdn.com" in image_host
        source = page_host or image_host
        candidates.append(ImageCandidate(image_url, page_url, title, source, aliexpress))
    return candidates
