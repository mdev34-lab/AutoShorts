"""Local validation and extraction of the product slug. Never fetches the Shopee page."""

from __future__ import annotations

import html
import re
import unicodedata
from dataclasses import dataclass
from urllib.parse import unquote, urlsplit

from .errors import InvalidProductLinkError
from .network import is_shopee_host, is_shp_ee_host, normalized_host, resolve_shp_ee

_ITEM_SUFFIX = re.compile(r"-i\.\d+\.\d+$", re.IGNORECASE)
_WHITESPACE = re.compile(r"\s+")


@dataclass(frozen=True)
class Product:
    input_url: str
    resolved_url: str
    slug: str
    name: str
    shop_id: str | None = None
    item_id: str | None = None


def _parse_product_path(url: str) -> Product:
    parts = urlsplit(url)
    host = (parts.hostname or "").lower()
    if parts.scheme.lower() not in {"https", "http"} or not is_shopee_host(host):
        raise InvalidProductLinkError(
            "Input must be a product link on shopee.* (or a resolvable shp.ee short link)."
        )
    raw_path = unquote(parts.path).strip("/")
    if not raw_path:
        raise InvalidProductLinkError(
            "The Shopee link has no product slug in its path."
        )
    segment = raw_path.split("/")[-1]
    if segment.lower() in {"product", "search", "cart", "checkout", "list"}:
        raise InvalidProductLinkError(
            "The path looks like a category/search page, not an individual product link."
        )
    match = re.search(r"-i\.(\d+)\.(\d+)$", segment, re.IGNORECASE)
    shop_id, item_id = (match.group(1), match.group(2)) if match else (None, None)
    if len(segment) > 300:
        raise InvalidProductLinkError(
            "The product slug is unreasonably long (>300 chars); refusing to "
            "treat it as a product name."
        )
    slug = _ITEM_SUFFIX.sub("", segment)
    # The slug is an UNTRUSTED identifier, not evidence: keep it bounded and
    # markup-free; facts are verified against the source page downstream.
    name = html.unescape(slug.replace("-", " ").replace("_", " "))
    name = re.sub(r"<[^>]*>", " ", name)  # strip any markup smuggled via %3C...
    name = "".join(ch for ch in name if ch.isprintable() or ch.isspace())
    name = _WHITESPACE.sub(" ", name).strip(" .")
    if len(name) > 200:
        raise InvalidProductLinkError(
            "The slug-derived product name exceeds 200 chars; refusing to "
            "narrate an unbounded user-supplied string."
        )
    if len(name) < 3 or not any(ch.isalpha() for ch in name):
        raise InvalidProductLinkError(
            "Could not extract a usable product name from the slug."
        )
    return Product(
        input_url=url,
        resolved_url=url,
        slug=slug,
        name=name,
        shop_id=shop_id,
        item_id=item_id,
    )


def parse_product_link(url: str) -> Product:
    if not isinstance(url, str) or not url.strip():
        raise InvalidProductLinkError("Provide a Shopee product link.")
    url = url.strip()
    host = normalized_host(url)
    if is_shp_ee_host(host):
        resolved = resolve_shp_ee(url)
        product = _parse_product_path(resolved)
        return Product(
            input_url=url,
            resolved_url=resolved,
            slug=product.slug,
            name=product.name,
            shop_id=product.shop_id,
            item_id=product.item_id,
        )
    return _parse_product_path(url)


def normalized_product_query(name: str) -> str:
    """Search text derived from the slug; strips only common promo signs/words."""
    text = unicodedata.normalize("NFKC", name)
    text = re.sub(r"[^\w\s+.-]", " ", text, flags=re.UNICODE)
    text = _WHITESPACE.sub(" ", text).strip()
    return text
