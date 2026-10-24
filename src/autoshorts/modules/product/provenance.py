"""Fail-closed product/source identity checks for selected images.

Every image must come from a public page that proves the exact model code
from the product slug in its title/H1, and the image URL must appear in
that page's gallery with alt text labelled with the same model code. Any
doubt raises ProductNotFoundError and nothing is downloaded.
"""

from __future__ import annotations

import html
import re
import unicodedata
from dataclasses import dataclass, field
from html.parser import HTMLParser
from urllib.parse import urljoin, urlsplit

import requests

from .errors import MediaProcessingError, ProductNotFoundError, SourceBlockedError
from .network import _CAPTCHA_MARKERS, safe_request

_MAX_PAGE_BYTES = 4 * 1024 * 1024
_MODEL_SUFFIXES = {
    "pro",
    "plus",
    "max",
    "ultra",
    "gt",
    "neo",
    "lite",
    "mini",
    "bt",
    "se",
    "fe",
}
_PREFIX_STOPWORDS = {
    "the",
    "for",
    "with",
    "model",
    "series",
    "product",
    "projector",
    "projetor",
    "jbl",
    "magcubic",
}
_URL_RE = re.compile(r"(?:https?:)?//[^\s,]+|/[^\s,]+", re.IGNORECASE)
# Numeric listing IDs embedded in the slug (-i.<shop>.<item>) are not product
# model codes and must never be required from the source page.
_SHOPEE_ITEM_ID = re.compile(r"^i\d{6,}$")
# Resolution specs (1080p, 4K, 8K) are descriptors, not model identifiers;
# they must never seed or extend a model code ("Portátil 4K" != "portatil4k").
_RESOLUTION_TOKEN = re.compile(r"^(?:\d{3,4}p|[48]k)$")


def _words(text: str) -> list[str]:
    text = html.unescape(text or "").replace("+", " plus ")
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return re.findall(r"[a-z0-9]+", text.casefold())


def _is_suffix_run(token: str) -> bool:
    """True for a variant written glued together, e.g. 'proplus' = pro + plus."""
    if token in _MODEL_SUFFIXES:
        return True
    return any(
        token.startswith(suffix) and _is_suffix_run(token[len(suffix) :])
        for suffix in _MODEL_SUFFIXES
        if suffix != token
    )


def model_codes(text: str) -> list[str]:
    """Model-like alphanumeric codes, retaining explicit variant suffixes such as PLUS."""
    tokens = _words(text)
    codes: list[str] = []
    for index, token in enumerate(tokens):
        if not any(ch.isdigit() for ch in token):
            continue
        if _SHOPEE_ITEM_ID.match(token) or _RESOLUTION_TOKEN.match(token):
            continue
        code = token
        following = index + 1
        # 'mini' is a category descriptor ("mini projector"), not a variant suffix.
        while (
            following < len(tokens)
            and _is_suffix_run(tokens[following])
            and tokens[following] != "mini"
        ):
            code += tokens[following]
            following += 1
        if (
            index > 0
            and tokens[index - 1].isalpha()
            and tokens[index - 1] not in _PREFIX_STOPWORDS
        ):
            prefix = tokens[index - 1]
            if len(prefix) <= 8:
                codes.append(prefix + code)
        if len(code) >= 4 and any(ch.isalpha() for ch in code):
            codes.append(code)
    # Separated prefixes, e.g. "HY 300 Pro Plus".
    for index, token in enumerate(tokens[:-1]):
        if not token.isalpha() or len(token) > 5 or token in _PREFIX_STOPWORDS:
            continue
        next_token = tokens[index + 1]
        if (
            next_token.isdigit()
            and len(next_token) >= 2
            and not _RESOLUTION_TOKEN.match(token + next_token)
        ):
            code = token + next_token
            following = index + 2
            while following < len(tokens) and _is_suffix_run(tokens[following]):
                code += tokens[following]
                following += 1
            if len(code) >= 4:
                codes.append(code)
    return list(dict.fromkeys(codes))


def primary_model_code(product_name: str) -> str:
    codes = model_codes(product_name)
    if not codes:
        raise ProductNotFoundError(
            "No verifiable model identifier found in the product slug; "
            "media is not approved without an exact match in an external source."
        )
    longest = max(len(code) for code in codes)
    # When the slug yields variants of the SAME model via prefix stopwords
    # (e.g. 'hy300proplus' and 'magcubichy300proplus'), the canonical variant
    # is the longest one WITHOUT a brand prefix. Different brands still yield
    # distinct codes and are still rejected.
    candidates = [code for code in codes if len(code) == longest]
    if len(candidates) > 1:
        brand_words = {w for w in _words(product_name) if w.isalpha() and len(w) >= 4}

        def _prefix_len(code: str) -> int:
            for brand in sorted(brand_words, key=len, reverse=True):
                if code.startswith(brand) and len(code) > len(brand):
                    rest = code[len(brand) :]
                    if any(ch.isdigit() for ch in rest):
                        return len(brand)
            return 0

        min_prefix = min(_prefix_len(code) for code in candidates)
        candidates = [code for code in candidates if _prefix_len(code) == min_prefix]
    return min(candidates)


@dataclass
class PageEvidence:
    url: str
    status: int
    title: str
    h1: str
    meta_title: str
    identity_codes: list[str]
    images: list[dict[str, str]] = field(default_factory=list)
    links: list[str] = field(default_factory=list)
    html_text: str = ""

    @property
    def identity_text(self) -> str:
        return " ".join(x for x in (self.title, self.h1, self.meta_title) if x)


class _PageParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.title_parts: list[str] = []
        self.h1_parts: list[str] = []
        self.title_depth = 0
        self.h1_depth = 0
        self.meta_title = ""
        self.meta_images: list[str] = []
        self.images: list[dict[str, str]] = []
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = {key.lower(): (value or "") for key, value in attrs}
        if tag.lower() == "title":
            self.title_depth += 1
        if tag.lower() == "h1":
            self.h1_depth += 1
        if tag.lower() == "meta":
            key = (
                attributes.get("property") or attributes.get("name") or ""
            ).casefold()
            content = attributes.get("content", "")
            if key in {"og:title", "twitter:title"} and content and not self.meta_title:
                self.meta_title = content
            elif key in {"og:image", "twitter:image"} and content:
                self.meta_images.append(content)
        if tag.lower() == "img":
            sources = []
            for key in ("src", "data-src", "data-original", "srcset", "data-srcset"):
                if attributes.get(key):
                    sources.extend(_URL_RE.findall(attributes[key]))
            if sources:
                self.images.append(
                    {"alt": attributes.get("alt", ""), "sources": "\n".join(sources)}
                )
        if tag.lower() == "a" and attributes.get("href"):
            self.links.append(attributes["href"])
        if tag.lower() in {"iframe", "video", "source"}:
            for key in ("src", "data-src", "srcset"):
                if attributes.get(key):
                    self.links.extend(_URL_RE.findall(attributes[key]))

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "title" and self.title_depth:
            self.title_depth -= 1
        if tag.lower() == "h1" and self.h1_depth:
            self.h1_depth -= 1

    def handle_data(self, data: str) -> None:
        if self.title_depth:
            self.title_parts.append(data)
        if self.h1_depth and not self.h1_parts:
            self.h1_parts.append(data)


def _parse_page_html(raw: str, url: str, status: int) -> PageEvidence:
    """Parse already-fetched HTML into page evidence (no new request)."""
    parser = _PageParser()
    try:
        parser.feed(raw)
        parser.close()
    except Exception as exc:
        raise MediaProcessingError(f"Could not parse the evidence page: {exc}") from exc
    raw_title = re.sub(r"\s+", " ", html.unescape(" ".join(parser.title_parts))).strip()
    h1 = re.sub(r"\s+", " ", html.unescape(" ".join(parser.h1_parts))).strip()
    title = raw_title or re.sub(r"\s+", " ", html.unescape(parser.meta_title)).strip()
    identity = " ".join(x for x in (title, h1, parser.meta_title) if x)
    return PageEvidence(
        url=url,
        status=status,
        title=title,
        h1=h1,
        meta_title=parser.meta_title,
        identity_codes=model_codes(identity),
        images=parser.images,
        links=parser.links,
        html_text=raw,
    )


def fetch_page_evidence(
    url: str, *, operation: str = "verify_product_source_page"
) -> PageEvidence:
    session = requests.Session()
    session.trust_env = False
    response = safe_request(
        session,
        "GET",
        url,
        operation=operation,
        timeout=(12, 45),
        stream=True,
        headers={
            "User-Agent": "autoshorts/1.0 (public-source identity check; no Shopee fetch)"
        },
    )
    if response.status_code >= 400:
        raise ProductNotFoundError(
            f"The evidence page answered HTTP {response.status_code}; "
            "no image was downloaded or used without product verification."
        )
    content_type = (response.headers.get("content-type") or "").casefold()
    if "text/html" not in content_type:
        raise ProductNotFoundError(
            f"The evidence URL did not return HTML ({content_type or 'missing MIME'}); "
            "cannot verify the model and the image association."
        )
    chunks: list[bytes] = []
    size = 0
    for chunk in response.iter_content(64 * 1024):
        if not chunk:
            continue
        size += len(chunk)
        if size > _MAX_PAGE_BYTES:
            raise MediaProcessingError(
                "Evidence page exceeded 4 MiB; failing without accepting unverified media."
            )
        chunks.append(chunk)
    raw = b"".join(chunks).decode(response.encoding or "utf-8", errors="replace")
    sample = raw[:256_000].casefold()
    if any(marker in sample for marker in _CAPTCHA_MARKERS):
        raise SourceBlockedError(
            "Challenge/CAPTCHA detected in the evidence page; aborting instead of solving."
        )
    return _parse_page_html(raw, url=response.url, status=response.status_code)


_STRICT_CODE = re.compile(r"[a-z]{1,4}\d{3,}[a-z]*")


# Generic spec words that precede a number ("HD 290 ANSI", "USB 3"): they are
# not product family prefixes, so "hd290" must not read as another model.
_SPEC_WORDS = frozenset(
    "hd fhd uhd usb hdmi ram rom lcd led ips dc ac ansi gb tb mb wifi".split()
)


def other_model_codes(codes: list[str], expected: str) -> list[str]:
    """Codes naming a DIFFERENT model than `expected`. A brand-prefixed form of
    the same code ('magcubichy300pro') is the same model; a comparison title
    ('HY300 Pro vs HY300 ProPlus') carries a second model and is ambiguous."""
    strict = [
        c
        for c in codes
        if _STRICT_CODE.fullmatch(c)
        and re.match(r"[a-z]+", c).group() not in _SPEC_WORDS
    ]
    return [c for c in strict if c != expected and not c.endswith(expected)]


def verify_product_identity(
    product_name: str, page: PageEvidence, *, label: str
) -> dict:
    expected = primary_model_code(product_name)
    # Every identity field that names a model must name THIS model: a page
    # whose H1 says "HY300 Pro" but whose title says "HY300 ProPlus" is
    # ambiguous about the variant and is rejected, not merged.
    for field_name, text in (
        ("title", page.title),
        ("h1", page.h1),
        ("meta title", page.meta_title),
    ):
        codes = model_codes(text) if isinstance(text, str) and text else []
        if codes and (expected not in codes or other_model_codes(codes, expected)):
            raise ProductNotFoundError(
                f"Match rejected for {label}: the page {field_name} names a different "
                f"model variant (codes seen: {codes}), the slug requires {expected!r}. "
                "No media will be downloaded."
            )
    if expected not in page.identity_codes:
        raise ProductNotFoundError(
            f"Match rejected for {label}: the slug requires the exact model {expected!r}, "
            f"but the page title/H1 does not prove it (codes seen: {page.identity_codes or 'none'}). "
            "No media will be downloaded."
        )
    return {
        "expected_model_code": expected,
        "page_model_codes": page.identity_codes,
        "page_title": page.title,
        "page_h1": page.h1,
        "page_url": page.url,
        "identity_match": True,
    }


def _resource_key(url: str, page_url: str) -> tuple[str, int, str, str]:
    absolute = urljoin(page_url, html.unescape(url.strip()))
    parts = urlsplit(absolute)
    host = (parts.hostname or "").casefold().rstrip(".")
    scheme = parts.scheme.lower()
    port = parts.port or (443 if scheme == "https" else 80 if scheme == "http" else 0)
    path = re.sub(r"/+", "/", parts.path)
    query = "&".join(sorted(parts.query.split("&"))) if parts.query else ""
    return host, port, path, query


def verify_image_sources(
    candidates, product_name: str, *, prefetched_page: PageEvidence | None = None
) -> list[dict]:
    """Require an exact model in the source page and a matching, model-labelled gallery image.

    prefetched_page: evidence of a page already fetched and validated in this same
    operation; the same fail-closed checks run over it with no new request.
    """
    if not candidates:
        raise ProductNotFoundError("No images can be validated: empty list.")
    expected = primary_model_code(product_name)
    page_cache: dict[str, PageEvidence] = {}
    if prefetched_page is not None:
        page_cache[prefetched_page.url] = prefetched_page
    evidence: list[dict] = []
    for candidate in candidates:
        page_url = candidate.page_url
        if not page_url:
            raise ProductNotFoundError(
                "Image without a verifiable source page/URL; failing without rendering. "
                "Use images from a public page that identifies the exact model."
            )
        page = page_cache.get(page_url)
        if page is None:
            page = fetch_page_evidence(page_url)
        page_cache[page_url] = page
        identity_evidence = verify_product_identity(product_name, page, label="photo")
        wanted = _resource_key(candidate.image_url, page.url)
        matched_alt = None
        for image in page.images:
            if expected not in model_codes(image.get("alt", "")):
                continue
            source_urls = _URL_RE.findall(image.get("sources", ""))
            if any(_resource_key(src, page.url) == wanted for src in source_urls):
                matched_alt = image.get("alt", "")
                break
        if matched_alt is None:
            raise ProductNotFoundError(
                f"Match rejected for photo {candidate.title!r}: the image does not appear in "
                f"the page gallery with alt/model {expected!r}. No photos will be downloaded."
            )
        evidence.append(
            {
                **identity_evidence,
                "image_url": candidate.image_url,
                "image_host": urlsplit(candidate.image_url).hostname,
                "image_gallery_alt": matched_alt,
                "image_bound_to_page": True,
                "verification_method": "exact model in title/H1 and image URL bound to a model-labelled img alt",
            }
        )
    return evidence
