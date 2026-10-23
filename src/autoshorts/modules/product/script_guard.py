"""Fail-closed guard for product showcase scripts.

The narration is generated from the slug-derived product name only. A prompt
instruction is not a guarantee, so every generated script is checked before
use: any specification-like claim, resolution/feature mention or
price/discount mention that is not grounded in the verified product name
aborts the video before any media is downloaded.

Grounding is unit-aware: a number only grounds a claim together with the unit
it appears with in the product name. Digits that are part of a model code
(e.g. the 300 in "HY300") ground nothing, so a fabricated "300mAh" battery
claim is rejected even when the model code contains "300". Spelled-out
quantities ("two terabytes") are checked the same way as digits.
"""

from __future__ import annotations

import re
import unicodedata

from .errors import ProductPipelineError

_UNIT_ALIASES = {
    "hora": "h",
    "horas": "h",
    "hour": "h",
    "hours": "h",
    "watt": "w",
    "watts": "w",
    "gigabyte": "gb",
    "gigabytes": "gb",
    "giga": "gb",
    "gigas": "gb",
    "terabyte": "tb",
    "terabytes": "tb",
    "polegada": "pol",
    "polegadas": "pol",
}

_UNIT_PATTERN = re.compile(
    r"\b(\d+(?:[.,]\d+)?)\s?(gb|tb|mb|mp|mpx|px|kw|wh|mah|ah|khz|mhz|ghz|hz|"
    r"watts?|w|v|amp|ma|kg|mg|ml|cl|cm|mm|km|g|l|m|a|ansi|horas?|hours?|h|"
    r"gigabytes?|gigas?|terabytes?|polegadas?|pol|\"|″)\b",
    re.IGNORECASE,
)
# Resolution-style claims are always written attached to the number (4K, 1080p).
_RESOLUTION_PATTERN = re.compile(r"\b(\d+(?:[.,]\d+)?)(k|p)\b", re.IGNORECASE)

_NUMBER_WORDS = {
    "um": "1",
    "uma": "1",
    "one": "1",
    "dois": "2",
    "duas": "2",
    "two": "2",
    "três": "3",
    "tres": "3",
    "three": "3",
    "quatro": "4",
    "four": "4",
    "cinco": "5",
    "five": "5",
    "seis": "6",
    "six": "6",
    "sete": "7",
    "seven": "7",
    "oito": "8",
    "eight": "8",
    "nove": "9",
    "nine": "9",
    "dez": "10",
    "ten": "10",
    "onze": "11",
    "eleven": "11",
    "doze": "12",
    "twelve": "12",
    "treze": "13",
    "thirteen": "13",
    "catorze": "14",
    "quatorze": "14",
    "fourteen": "14",
    "quinze": "15",
    "fifteen": "15",
    "vinte": "20",
    "twenty": "20",
    "trinta": "30",
    "thirty": "30",
}
_SPELLED_PATTERN = re.compile(
    r"\b(" + "|".join(_NUMBER_WORDS) + r")\s+(gb|tb|mb|mah|ah|kw|wh|"
    r"watts?|kg|g|l|ml|horas?|hours?|polegadas?|pol|"
    r"gigabytes?|gigas?|terabytes?)\b",
    re.IGNORECASE,
)

_PRICE_PATTERN = re.compile(
    r"(r\$\s?\d|\b\d+(?:[.,]\d+)?\s?%|\bpor\s+apenas\b|\bde\s+\S+\s+por\b|"
    r"\bdesconto\b|\bcupom\b|\bfrete\s+gr[áa]tis\b)",
    re.IGNORECASE,
)
# Feature claims that must appear verbatim in the verified product name.
_FEATURE_KEYWORDS = (
    "bluetooth",
    "wi-fi",
    "wifi",
    "android",
    "hdmi",
    "laser",
    "oled",
    "amoled",
    "qled",
    "lcd",
    "led",
    "ips",
    "bateria",
    "battery",
)


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.casefold())
    return "".join(ch for ch in text if not unicodedata.combining(ch))


def _canon_unit(unit: str) -> str:
    return _UNIT_ALIASES.get(_normalize(unit), _normalize(unit))


def _spec_pairs(text: str) -> set[tuple[str, str]]:
    """(number, unit) pairs grounded by the text, separators/units normalized."""
    pairs = {
        (m.group(1).replace(",", "."), _canon_unit(m.group(2)))
        for m in _UNIT_PATTERN.finditer(text)
    }
    pairs.update(
        (m.group(1).replace(",", "."), m.group(2).casefold())
        for m in _RESOLUTION_PATTERN.finditer(text)
    )
    pairs.update(
        (_NUMBER_WORDS[_normalize(m.group(1))], _canon_unit(m.group(2)))
        for m in _SPELLED_PATTERN.finditer(text)
    )
    return pairs


def _features_in(text: str) -> set[str]:
    normalized = _normalize(text)
    return {
        kw
        for kw in _FEATURE_KEYWORDS
        if re.search(rf"(?<![a-z0-9]){re.escape(kw)}(?![a-z0-9])", normalized)
    }


def assert_script_within_verified_facts(
    paragraphs: list[str], product_name: str
) -> None:
    """Raise ProductPipelineError unless every spec/price claim traces to the name."""
    if not product_name.strip():
        raise ProductPipelineError(
            "No verified product name available; aborting instead of narrating "
            "claims that cannot be grounded."
        )
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
