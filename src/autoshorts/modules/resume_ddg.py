"""Fixed-endpoint DDG image search using only an injected bounded GET."""

from __future__ import annotations

import json
from collections.abc import Callable

_ROOT_URL = "https://duckduckgo.com/"
_IMAGES_URL = "https://duckduckgo.com/i.js"
_HTML_MAX_BYTES = 262_144
_JSON_MAX_BYTES = 1_048_576
_TIMEOUT_SECONDS = 10.0
_MAX_TOKEN_BYTES = 256


class DDGSearchError(ValueError):
    """Invalid search input or an unusable DDG response."""


def _reject_challenge_html(raw: bytes) -> None:
    lowered = raw.lower()
    if any(
        marker in lowered
        for marker in (
            b"captcha",
            b"challenge",
            b"anomaly.js",
            b"anomaly-modal",
            b"verify you are human",
            b"verify that you are human",
            b"bots use duckduckgo",
        )
    ):
        raise DDGSearchError("DDG challenge or CAPTCHA HTML is prohibited")


def _extract_vqd(html: bytes) -> str:
    """Parse the installed double-quoted, ampersand and single-quoted forms."""
    _reject_challenge_html(html)
    for opening, closing in (
        (b'vqd="', b'"'),
        (b"vqd=", b"&"),
        (b"vqd='", b"'"),
    ):
        start = html.find(opening)
        if start < 0:
            continue
        start += len(opening)
        # The unquoted form must not swallow a quoted token and later markup.
        if opening == b"vqd=" and html[start : start + 1] in (b'"', b"'"):
            continue
        end = html.find(closing, start)
        if end < 0:
            raise DDGSearchError("Missing vqd token terminator")
        token_bytes = html[start:end]
        if not token_bytes or len(token_bytes) > _MAX_TOKEN_BYTES:
            raise DDGSearchError("Empty or overlong vqd token")
        try:
            token = token_bytes.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise DDGSearchError("Invalid UTF-8 vqd token") from exc
        if any(
            not char.isprintable() or char.isspace() or char in "\"'<>&"
            for char in token
        ):
            raise DDGSearchError("Malformed vqd token")
        return token
    raise DDGSearchError("Missing vqd token")


def _invalid_json_constant(value: str) -> None:
    raise DDGSearchError(f"Invalid JSON constant: {value}")


class FixedDDGSearch:
    """A two-GET, single-page adapter; the caller supplies and isolates I/O."""

    def __init__(self, get_ddg: Callable[..., bytes], max_results: int = 10) -> None:
        if not callable(get_ddg):
            raise TypeError("get_ddg must be callable")
        if type(max_results) is not int or not 1 <= max_results <= 25:
            raise DDGSearchError(
                "max_results must be an exact integer from 1 through 25"
            )
        self._get_ddg = get_ddg
        self._max_results = max_results

    def _response(self, url: str, params: dict, max_bytes: int) -> bytes:
        raw = self._get_ddg(
            url,
            params=params,
            max_bytes=max_bytes,
            timeout=_TIMEOUT_SECONDS,
            max_redirects=0,
        )
        if not isinstance(raw, bytes) or not raw or len(raw) > max_bytes:
            raise DDGSearchError(
                "DDG response must be nonempty bytes within the fixed byte cap"
            )
        return raw

    def __call__(self, query: str) -> list[dict]:
        if not isinstance(query, str) or not query.strip() or len(query) > 2000:
            raise DDGSearchError(
                "query must be a nonempty string of at most 2000 characters"
            )
        html = self._response(_ROOT_URL, {"q": query}, _HTML_MAX_BYTES)
        token = _extract_vqd(html)
        raw = self._response(
            _IMAGES_URL,
            {"o": "json", "q": query, "l": "br-pt", "vqd": token, "p": "1", "ct": "AT"},
            _JSON_MAX_BYTES,
        )
        if raw.lstrip().startswith(b"<"):
            _reject_challenge_html(raw)
            raise DDGSearchError("Expected DDG JSON, got HTML")
        try:
            payload = json.loads(
                raw.decode("utf-8"), parse_constant=_invalid_json_constant
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise DDGSearchError("Malformed DDG JSON") from exc
        if not isinstance(payload, dict):
            raise DDGSearchError("DDG JSON root must be an object")
        items = payload.get("results")
        if not isinstance(items, list):
            raise DDGSearchError("DDG results must be a list")
        # Validate the entire page before applying the output limit. Never hide
        # a malformed item by skipping it or truncating validation early.
        for item in items:
            if not isinstance(item, dict):
                raise DDGSearchError("Every DDG result must be an object")
            for field in ("image", "title", "url"):
                value = item.get(field)
                if not isinstance(value, str) or not value.strip():
                    raise DDGSearchError(
                        f"DDG result {field} must be a nonempty string"
                    )
            for field in ("height", "width"):
                value = item.get(field)
                if type(value) is not int or value <= 0:
                    raise DDGSearchError(
                        f"DDG result {field} must be a positive exact integer"
                    )
        # Preserve all original DDGS dictionary keys and values. In particular,
        # no response URL or 'next' value is ever used as a request endpoint.
        return items[: self._max_results]
