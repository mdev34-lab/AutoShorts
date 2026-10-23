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
# Claims a slug-derived product name can never verify: certifications, warranty
# terms and grade/origin claims. Rejected categorically, regardless of wording.
_UNGROUNDABLE_PATTERN = re.compile(
    r"(?<![a-z0-9])(anatel|inmetro|fcc|fda|iso\s?9001|garantia|warranty|"
    r"vitalicio|lifetime|aerospace|aeroespacial|military|militar|medico|"
    r"medical|hospitalar)(?![a-z0-9])"
)

# Attribute context for binding unit claims: "8GB RAM" must not ground "8GB de
# armazenamento", nor "suporta 2kg" ground "pesa 2kg".
_ATTRIBUTE_WORDS = {
    "ram": "memory",
    "memoria": "memory",
    "armazenamento": "storage",
    "storage": "storage",
    "disco": "storage",
    "ssd": "storage",
    "hd": "storage",
    "peso": "weight",
    "pesa": "weight",
    "weighs": "weight",
    "weight": "weight",
    "suporta": "load",
    "supports": "load",
    "carga": "load",
    "capacidade": "load",
    "load": "load",
    "potencia": "power",
    "power": "power",
    "brilho": "brightness",
    "brightness": "brightness",
    "bateria": "battery",
    "battery": "battery",
    "autonomia": "battery",
    "tela": "screen",
    "screen": "screen",
    "display": "display",
}

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


_WORD_RE = re.compile(r"[a-z0-9]+")


def _attribute_near(text: str, start: int, end: int, window: int = 5) -> str | None:
    """First recognized attribute word within `window` words around a match."""
    before = _WORD_RE.findall(_normalize(text[:start]))[-window:]
    after = _WORD_RE.findall(_normalize(text[end:]))[:window]
    for word in list(reversed(before)) + after:
        attribute = _ATTRIBUTE_WORDS.get(word)
        if attribute:
            return attribute
    return None


def _spec_claims(text: str) -> set[tuple[str, str, str | None]]:
    """(number, unit, attribute) claims, separators/units normalized."""
    claims: set[tuple[str, str, str | None]] = set()
    for m in _UNIT_PATTERN.finditer(text):
        claims.add(
            (
                m.group(1).replace(",", "."),
                _canon_unit(m.group(2)),
                _attribute_near(text, m.start(), m.end()),
            )
        )
    for m in _RESOLUTION_PATTERN.finditer(text):
        claims.add(
            (
                m.group(1).replace(",", "."),
                m.group(2).casefold(),
                _attribute_near(text, m.start(), m.end()),
            )
        )
    for m in _SPELLED_PATTERN.finditer(text):
        claims.add(
            (
                _NUMBER_WORDS[_normalize(m.group(1))],
                _canon_unit(m.group(2)),
                _attribute_near(text, m.start(), m.end()),
            )
        )
    return claims


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
    grounded = _spec_claims(product_name)
    grounded_features = _features_in(product_name)
    text = " ".join(paragraphs)
    ungroundable = _UNGROUNDABLE_PATTERN.search(_normalize(text))
    if ungroundable:
        raise ProductPipelineError(
            f"Script makes a {ungroundable.group(0)!r} claim; certification, "
            "warranty and grade/origin claims can never be verified from a "
            "product name - aborting."
        )
    for number, unit, attribute in _spec_claims(text):
        candidates = {(n, u) for n, u, _a in grounded}
        if (number, unit) not in candidates:
            raise ProductPipelineError(
                f"Script mentions specification {number}{unit} that is not present "
                "in the verified product name; aborting instead of narrating "
                "unverified specs."
            )
        if attribute is not None:
            name_attributes = {
                a for n, u, a in grounded if (n, u) == (number, unit) and a is not None
            }
            if name_attributes and attribute not in name_attributes:
                raise ProductPipelineError(
                    f"Script binds {number}{unit} to {attribute!r} but the verified "
                    f"product name binds it to {sorted(name_attributes)!r}; "
                    "aborting instead of narrating a mismatched attribute."
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
