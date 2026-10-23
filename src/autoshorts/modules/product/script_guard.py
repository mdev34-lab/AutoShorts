"""Fail-closed guard for product showcase scripts.

The narration is generated from the slug-derived product name only. A prompt
instruction is not a guarantee, so every generated script is checked before
use: any specification-like claim or price/discount mention that is not
grounded in the verified product name aborts the video before any media is
downloaded.
"""

from __future__ import annotations

import re

from .errors import ProductPipelineError

_UNIT_PATTERN = re.compile(
    r"\b\d+(?:[.,]\d+)?\s?(?:gb|tb|mb|mp|mpx|px|kw|wh|mah|ah|khz|mhz|ghz|hz|"
    r"w|v|amp|ma|kg|mg|ml|cl|cm|mm|km|g|l|m|a|ansi|polegadas?|pol|\"|″)\b",
    re.IGNORECASE,
)
_PRICE_PATTERN = re.compile(
    r"(r\$\s?\d|\b\d+(?:[.,]\d+)?\s?%|\bpor\s+apenas\b|\bde\s+\S+\s+por\b|"
    r"\bdesconto\b|\bcupom\b|\bfrete\s+gr[áa]tis\b)",
    re.IGNORECASE,
)
_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)?")


def _numbers_in(text: str) -> set[str]:
    return set(_NUMBER_RE.findall(text.casefold()))


def assert_script_within_verified_facts(
    paragraphs: list[str], product_name: str
) -> None:
    """Raise ProductPipelineError unless every spec/price claim traces to the name."""
    grounded = _numbers_in(product_name)
    text = " ".join(paragraphs)
    for match in _UNIT_PATTERN.finditer(text):
        token = match.group(0).strip()
        number = _NUMBER_RE.match(token)
        if number and number.group(0) not in grounded:
            raise ProductPipelineError(
                f"Script mentions specification {token!r} that is not present in "
                "the verified product name; aborting instead of narrating "
                "unverified specs."
            )
    price = _PRICE_PATTERN.search(text)
    if price:
        raise ProductPipelineError(
            f"Script mentions price/discount ({price.group(0).strip()!r}); "
            "showcase narration must stay price-free."
        )
