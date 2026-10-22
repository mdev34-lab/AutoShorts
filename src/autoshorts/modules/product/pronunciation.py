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
