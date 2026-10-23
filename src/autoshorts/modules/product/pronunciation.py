"""Pronunciation respellings for TTS only; subtitles keep the original spelling.

`prepare_for_tts(text, extra)` returns the text to send to the TTS engine.
Order:
1. dictionary (built-in default + optional user JSON file; user wins);
2. decimals ("5.4"/"1,6" -> "cinco vírgula quatro"), the point gets lost otherwise;
3. alphanumeric codes (HY300, 720P) -> letters by name + number in words;
4. bare integers stay as digits (the engine reads them fine; spelling them
   out makes the video longer).
The dictionary is {"spelling": "respelling"}; keys match whole words,
case-insensitive. Extra file via env AUTOSHORTS_PRONUNCIATION_FILE.
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

DEFAULT_DICTIONARY: dict[str, str] = {
    "Magcubic": "Mag quiúbic",
    "Wi-Fi": "uai fai",
    "WiFi": "uai fai",
    "Bluetooth": "blu tuf",
    "keystone": "quistoun",
    "ANSI": "ânsi",
    "Android": "Andróide",
    "Pro": "Pró",
    "Plus": "Plás",
    "HDMI": "agá dê ême i",
    "USB": "u ésse bê",
    "LED": "léd",
    "Full HD": "full agá dê",
}

LETTERS = {
    "a": "á",
    "b": "bê",
    "c": "cê",
    "d": "dê",
    "e": "é",
    "f": "éfe",
    "g": "gê",
    "h": "agá",
    "i": "í",
    "j": "jota",
    "k": "cá",
    "l": "éle",
    "m": "ême",
    "n": "ene",
    "o": "ó",
    "p": "pê",
    "q": "quê",
    "r": "érre",
    "s": "ésse",
    "t": "tê",
    "u": "u",
    "v": "vê",
    "w": "dáblio",
    "x": "xis",
    "y": "ípsilon",
    "z": "zê",
}


def _collapse(text: str) -> str:
    folded = unicodedata.normalize("NFKD", text)
    return re.sub(r"[^a-z0-9]", "", folded.lower())


def validate_respelling(key: str, value: str) -> None:
    """Reject a respelling that is not a pure phonetic rewrite of its key.

    A respelling reaches the TTS after the script guard ran, so it must not
    be able to introduce facts. It must stay close in size to the key
    (spelled-out letters and numbers still fit), and may not contain a
    spec, feature or certification claim that the key does not have.
    """
    from .script_guard import (
        _RESOLUTION_PATTERN,
        _UNGROUNDABLE_PATTERN,
        _UNIT_PATTERN,
        _features_in,
    )

    k, v = _collapse(key), _collapse(value)
    if not k or not v or len(v) > 3 * len(k) + 2:
        raise ValueError(
            f"Pronunciation respelling for {key!r} must be phonetic only "
            "(too long to be a respelling of the key)."
        )
    for probe in (_UNGROUNDABLE_PATTERN, _UNIT_PATTERN, _RESOLUTION_PATTERN):
        if probe.search(value) and not probe.search(key):
            raise ValueError(
                f"Pronunciation respelling for {key!r} adds a factual claim."
            )
    if _features_in(value) - _features_in(key):
        raise ValueError(f"Pronunciation respelling for {key!r} adds a feature.")


def load_dictionary(path: str | Path | None = None) -> dict[str, str]:
    merged = dict(DEFAULT_DICTIONARY)
    if path:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, dict) or not all(
            isinstance(k, str) and isinstance(v, str) for k, v in data.items()
        ):
            raise ValueError(
                "Pronunciation file must be a JSON object {spelling: respelling}."
            )
        for key, value in data.items():
            validate_respelling(key, value)
        merged.update(data)
    return merged


def _num(n: str) -> str:
    try:
        from num2words import num2words

        return num2words(int(n), lang="pt_BR")
    except Exception:
        return n  # digits are read acceptably on their own


def _digits(n: str) -> str:
    return " ".join(_num(d) for d in n)


def _spell(letters: str) -> str:
    return " ".join(LETTERS.get(c.lower(), c) for c in letters)


def prepare_for_tts(text: str, dictionary: dict[str, str] | None = None) -> str:
    d = DEFAULT_DICTIONARY if dictionary is None else dictionary
    out = text
    for key in sorted(d, key=len, reverse=True):
        replacement = d[key]

        def _replace(_m: re.Match[str], _v: str = replacement) -> str:
            return _v

        out = re.sub(
            rf"(?<![\w-]){re.escape(key)}(?![\w-])",
            _replace,
            out,
            flags=re.I,
        )
    out = re.sub(
        r"(\d+)[.,](\d+)",
        lambda m: f"{_num(m.group(1))} vírgula {_digits(m.group(2))}",
        out,
    )
    # codes: 1-3 letters + digits (HY300) or digits + 1-2 uppercase letters (720P)
    out = re.sub(
        r"\b([A-Za-z]{1,3})(\d+)\b",
        lambda m: f"{_spell(m.group(1))} {_num(m.group(2))}",
        out,
    )
    out = re.sub(
        r"\b(\d+)([A-Z]{1,2})\b",
        lambda m: f"{_num(m.group(1))} {_spell(m.group(2))}",
        out,
    )
    # bare integers stay as digits
    return re.sub(r"\s+", " ", out).strip()
