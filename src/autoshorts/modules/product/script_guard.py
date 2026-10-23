"""Fail-closed guard for product showcase scripts.

The narration is generated from the slug-derived product name only. A prompt
instruction is not a guarantee, so every generated script is checked before
use: any specification-like claim, resolution/feature mention or
price/discount mention that is not grounded in the verified product name
aborts the video before any media is downloaded.

Grounding is unit-aware: a number only grounds a claim together with the unit
it appears with in the product name. Digits that are part of a model code
(e.g. the 300 in "HY300") ground nothing, so a fabricated "300mAh" battery
claim is rejected even when the model code contains "300".
"""

from __future__ import annotations

import re
import unicodedata

from .errors import ProductPipelineError

_UNIT_PATTERN = re.compile(
    r"\b(\d+(?:[.,]\d+)?)\s?(gb|tb|mb|mp|mpx|px|kw|wh|mah|ah|khz|mhz|ghz|hz|"
    r"w|v|amp|ma|kg|mg|ml|cl|cm|mm|km|g|l|m|a|ansi|polegadas?|pol|\"|″)\b",
    re.IGNORECASE,
)
# Resolution-style claims are always written attached to the number (4K, 1080p).
_RESOLUTION_PATTERN = re.compile(r"\b(\d+(?:[.,]\d+)?)(k|p)\b", re.IGNORECASE)
_PRICE_PATTERN = re.compile(
    r"(r\$\s?\d|\b\d+(?:[.,]\d+)?\s?%|\bpor\s+apenas\b|\bde\s+\S+\s+por\b|"
    r"\bdesconto\b|\bcupom\b|\bfrete\s+gr[áa]tis\b)",
    re.IGNORECASE,
)
# Feature claims that must appear verbatim in the verified product name.
_FEATURE_KEYWORDS = ("bluetooth", "wi-fi", "wifi", "android", "hdmi", "laser")


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.casefold())
    return "".join(ch for ch in text if not unicodedata.combining(ch))


def _spec_pairs(text: str) -> set[tuple[str, str]]:
    """(number, unit) pairs grounded by the text, separators normalized."""
    pairs = {
        (m.group(1).replace(",", "."), m.group(2).casefold())
        for m in _UNIT_PATTERN.finditer(text)
    }
    pairs.update(
        (m.group(1).replace(",", "."), m.group(2).casefold())
        for m in _RESOLUTION_PATTERN.finditer(text)
    )
    return pairs


def _features_in(text: str) -> set[str]:
    normalized = _normalize(text)
    return {
        kw for kw in _FEATURE_KEYWORDS
        if re.search(rf"(?<![a-z0-9]){re.escape(kw)}(?![a-z0-9])", normalized)
    }


def assert_script_within_verified_facts(
    paragraphs: list[str], product_name: str
) -> None:
    """Raise ProductPipelineError unless every spec/price claim traces to the name."""
    grounded = _spec_pairs(product_name)
    grounded_features = _features_in(product_name)
    text = " ".join(paragraphs)
    for number, unit in _spec_pairs(text):
        if (number, unit) not in grounded:
            raise ProductPipelineError(
                f"Script mentions specification {number}{unit} that is not present "
                "in the verified product name; aborting instead of narrating "
                "unverified specs."
            )
    for feature in sorted(_features_in(text) - grounded_features):
        raise ProductPipelineError(
            f"Script claims feature {feature!r} that is not present in the "
            "verified product name; aborting instead of narrating unverified "
            "features."
        )
    price = _PRICE_PATTERN.search(text)
    if price:
        raise ProductPipelineError(
            f"Script mentions price/discount ({price.group(0).strip()!r}); "
            "showcase narration must stay price-free."
        )
