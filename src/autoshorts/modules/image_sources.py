"""Provenance-resolved web image sources.

Every candidate a provider returns carries its use-rights evidence (license +
attribution when the provider exposes it) and contextual-fit evidence (which
distinctive subject token matched). Candidates whose provenance or fit cannot
be established are rejected by the provider itself - no manual curation
downstream.

Providers, in the order configured by IMAGE_PROVIDERS:
- ``commons``: Wikimedia Commons API. License is verified against an
  allowlist of reusable licenses; attribution is collected. Candidates
  without an allowlisted license are rejected.
- ``ddg``: DuckDuckGo image search (no license metadata exists there).
  Stock-agency domains (which serve watermarked previews) are blocked, and
  the result title must prove contextual fit against the query.
"""

from __future__ import annotations

import html
import re
import unicodedata
from dataclasses import dataclass, field
from urllib.parse import urlsplit

import requests

from .logging_system import log

# Stock agencies serve watermarked previews; a watermark is proof the license
# was NOT resolved, so these hosts are never acceptable sources.
WATERMARKED_AGENCY_DOMAINS: set[str] = {
    "alamy",
    "gettyimages",
    "getty",
    "shutterstock",
    "dreamstime",
    "istockphoto",
    "istock",
    "123rf",
    "depositphotos",
    "vectorstock",
    "agefotostock",
    "stock.adobe",
    "adobe.stock",
    "canstockphoto",
    "pinterest",
    "ftcdn",  # Adobe Stock's CDN (as1.ftcdn.net etc.)
}

# Reusable licenses accepted from Commons (normalized, see _normalize_license).
_ALLOWED_LICENSES = {
    "cc0",
    "public domain",
    "cc by 1.0",
    "cc by 2.0",
    "cc by 2.5",
    "cc by 3.0",
    "cc by 4.0",
    "cc by-sa 1.0",
    "cc by-sa 2.0",
    "cc by-sa 2.5",
    "cc by-sa 3.0",
    "cc by-sa 4.0",
}

_FIT_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "from",
    "that",
    "this",
    "para",
    "com",
    "uma",
    "um",
    "das",
    "dos",
    "que",
    "por",
    "nos",
    "nas",
    "sobre",
    "contra",
    "entre",
    "apos",
    "até",
    "mais",
    "menos",
    "novo",
    "nova",
}

_TAG_RE = re.compile(r"<[^>]+>")
_WORD_RE = re.compile(r"[a-z0-9]+")

_COMMONS_API = "https://commons.wikimedia.org/w/api.php"


@dataclass
class SourceCandidate:
    url: str
    provider: str
    fit_evidence: str
    license: str | None = None
    attribution: str | None = None
    title: str = ""
    extra: dict = field(default_factory=dict)


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", (text or "").casefold())
    return "".join(ch for ch in text if not unicodedata.combining(ch))


def _normalize_license(raw: str) -> str:
    text = _TAG_RE.sub(" ", html.unescape(raw or "")).strip()
    text = re.sub(r"\s+", " ", text)
    # Commons variants: "CC BY-SA 4.0", "Cc-by-sa-4.0", "CC0", "Public domain"
    text = text.replace("Attribution", "CC BY").replace("ShareAlike", "SA")
    text = re.sub(r"[-_]", " ", text.casefold())
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\bcc\b", "cc", text)
    text = re.sub(r"\bby sa\b", "by-sa", text)
    text = re.sub(r"(\d) (\d)", r"\1.\2", text)  # "4 0" -> "4.0"
    return text.strip()


def _license_url(license_name: str) -> str:
    """Canonical license text URL for a normalized allowed license name."""
    m = re.fullmatch(r"cc (by|by-sa) (\d\.\d)", license_name)
    if m:
        return f"https://creativecommons.org/licenses/{m.group(1)}/{m.group(2)}/"
    if license_name == "cc0":
        return "https://creativecommons.org/publicdomain/zero/1.0/"
    return ""


def distinctive_tokens(text: str) -> set[str]:
    """Long, content-bearing tokens - the ones that prove contextual fit."""
    return {
        tok
        for tok in _WORD_RE.findall(_normalize(text))
        if len(tok) >= 4 and tok not in _FIT_STOPWORDS
    }


# Words marking data graphics, not photos: a standings table or infographic
# can share every query token yet contradict the subject (wrong season,
# wrong entity) - fail closed on them from every provider.
_DATA_GRAPHIC_WORDS = {
    "tabela",
    "classificacao",
    "standings",
    "infographic",
    "infografico",
    "chart",
    "diagram",
    "bracket",
    "mapa",
    "map",
    "logo",
    "crest",
    "escudo",
}


def fit_evidence(query: str, title: str, context: str = "") -> str | None:
    """Proof of contextual fit, or None when the candidate is ambiguous.

    Entity disambiguation: a single shared token proves nothing
    (Apple iPhone vs apple pie, Jaguar car vs animal, Corinthians
    football vs basketball). With several distinctive query tokens the
    candidate must cover ALL of them; a one-token query needs the token
    anchored in the candidate's TITLE. URL text is never evidence - a
    page path can carry query tokens under a completely unrelated image.
    """
    title_and_context = f"{title} {context}"
    if distinctive_tokens(title_and_context) & _DATA_GRAPHIC_WORDS:
        return None
    query_tokens = distinctive_tokens(query)
    if not query_tokens:
        return None
    candidate_tokens = distinctive_tokens(title_and_context)
    covered = query_tokens & candidate_tokens
    if len(query_tokens) == 1:
        token = next(iter(query_tokens))
        if token in distinctive_tokens(title):
            return token
        return None
    if len(query_tokens) == 2:
        # Two-token entities (Apple iPhone, Table Mountain, Giants vs Mets):
        # partial coverage is exactly the contradiction case - fail closed.
        if covered == query_tokens:
            return ",".join(sorted(covered))
        return None
    # Longer queries: require at least two distinctive anchors, so no
    # single shared token (a surname, a club name) can admit a
    # contradictory subject.
    if len(covered) >= 2:
        return ",".join(sorted(covered))
    return None


def is_agency_host(url: str) -> bool:
    host = (urlsplit(url).hostname or "").casefold()
    return any(agency in host for agency in WATERMARKED_AGENCY_DOMAINS)


def search_commons(
    query: str, *, limit: int = 5, min_width: int = 720, min_height: int = 720
) -> list[SourceCandidate]:
    """License-verified candidates from Wikimedia Commons."""
    params = {
        "action": "query",
        "format": "json",
        "generator": "search",
        "gsrsearch": f"filetype:bitmap {query}",
        "gsrnamespace": "6",
        "gsrlimit": str(limit * 2),
        "prop": "imageinfo",
        "iiprop": "url|size|mime|extmetadata",
    }
    try:
        resp = requests.get(
            _COMMONS_API,
            params=params,
            timeout=20,
            headers={"User-Agent": "autoshorts/1.0 (license-verified image search)"},
        )
        data = resp.json()
    except Exception as e:
        log(f"Commons search failed: {e}", "WARNING")
        return []
    candidates: list[SourceCandidate] = []
    for page in (data.get("query", {}).get("pages", {}) or {}).values():
        info = (page.get("imageinfo") or [{}])[0]
        url = info.get("url") or ""
        if info.get("mime") not in ("image/jpeg", "image/png"):
            continue
        if (info.get("width") or 0) < min_width or (
            info.get("height") or 0
        ) < min_height:
            continue
        meta = info.get("extmetadata") or {}
        license_raw = (meta.get("LicenseShortName") or {}).get("value") or ""
        license_name = _normalize_license(license_raw)
        if license_name not in _ALLOWED_LICENSES:
            continue
        title = page.get("title") or ""
        description = _TAG_RE.sub(
            " ", html.unescape((meta.get("ImageDescription") or {}).get("value") or "")
        )
        categories = _TAG_RE.sub(
            " ", html.unescape((meta.get("Categories") or {}).get("value") or "")
        )
        evidence = fit_evidence(query, title, f"{description} {categories}")
        if evidence is None:
            continue
        artist = _TAG_RE.sub(
            " ", html.unescape((meta.get("Artist") or {}).get("value") or "")
        ).strip()
        attribution = f"{artist}, {license_raw}" if artist else license_raw
        license_url = (meta.get("LicenseUrl") or {}).get("value") or _license_url(
            license_name
        )
        candidates.append(
            SourceCandidate(
                url=url,
                provider="commons",
                license=license_raw,
                attribution=attribution,
                fit_evidence=evidence,
                title=title,
                extra={
                    "source_page": info.get("descriptionurl") or "",
                    "creator": artist,
                    "license_url": license_url,
                },
            )
        )
        if len(candidates) >= limit:
            break
    log(f"Commons '{query[:50]}': {len(candidates)} license-verified candidates")
    return candidates


def filter_ddg_results(query: str, results: list[dict]) -> list[SourceCandidate]:
    """DDG results with agency domains and ambiguous fit removed."""
    candidates: list[SourceCandidate] = []
    for result in results:
        url = result.get("image") or ""
        page = result.get("url") or result.get("source") or ""
        title = result.get("title") or ""
        if not url:
            continue
        if is_agency_host(url) or is_agency_host(page):
            continue
        # Fit from the title only: URL/path tokens are not evidence (a page
        # about the query can host a completely unrelated image).
        evidence = fit_evidence(query, title)
        if evidence is None:
            continue
        candidates.append(
            SourceCandidate(
                url=url,
                provider="ddg",
                license=None,  # DDG exposes no license metadata
                attribution=None,
                fit_evidence=evidence,
                title=title,
            )
        )
    return candidates
