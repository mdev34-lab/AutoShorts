"""Conservative Commons portrait evidence extraction, not image admission.

A P180 statement is an authored source assertion, not independent visual proof.
This module never changes the searcher's fail-closed specific-image policy.
Callers must resolve the requested entity independently and inspect representative
pixels before proposing any production admission rule. Event/date/team requests
are intentionally outside this portrait-only evidence contract.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import urlsplit

from .image_sources import _ALLOWED_LICENSES, _normalize_license

_QID = re.compile(r"Q[1-9][0-9]*\Z")
_SCOPE = "commons-authored-depicts; pixels, event, date and team NOT verified"


@dataclass(frozen=True)
class PortraitEvidence:
    entity_id: str
    media_id: str
    source_page: str
    image_url: str
    license: str
    creator: str
    scope: str = _SCOPE


def portrait_evidence(
    page: dict,
    media_entity: dict,
    *,
    entity_id: str,
    identity_unambiguous: bool = False,
    context_constraints: tuple[str, ...] = (),
) -> PortraitEvidence | None:
    """Extract source evidence only, with explicit exact-identity preconditions.

    Inputs are raw query/imageinfo and MediaInfo entity records. Labels, titles,
    categories and generic person statements cannot substitute for the exact ID.
    An empty constraint list means portrait illustration only, not that a
    free-text query has been understood. The caller owns constraint extraction.
    """
    if not isinstance(page, dict) or not isinstance(media_entity, dict):
        return None
    if (
        not isinstance(entity_id, str)
        or not identity_unambiguous
        or not _QID.fullmatch(entity_id)
    ):
        return None
    if context_constraints:
        return None
    page_id = page.get("pageid")
    if not isinstance(page_id, int) or isinstance(page_id, bool) or page_id <= 0:
        return None
    media_id = f"M{page_id}"
    if media_entity.get("id") != media_id:
        return None
    if media_entity.get("missing") is not None:
        return None
    statements = media_entity.get("statements") or {}
    if not isinstance(statements, dict):
        return None
    claims = statements.get("P180", [])
    if not isinstance(claims, list):
        return None
    exact = False
    for claim in claims:
        if not isinstance(claim, dict) or claim.get("rank") == "deprecated":
            continue
        snak = claim.get("mainsnak") or {}
        if not isinstance(snak, dict):
            continue
        data = snak.get("datavalue") or {}
        if not isinstance(data, dict):
            continue
        value = data.get("value") or {}
        if (
            snak.get("snaktype") == "value"
            and snak.get("property") == "P180"
            and isinstance(value, dict)
            and value.get("id") == entity_id
        ):
            exact = True
    if not exact:
        return None
    infos = page.get("imageinfo") or []
    if not isinstance(infos, list) or len(infos) != 1 or not isinstance(infos[0], dict):
        return None
    info = infos[0]
    if info.get("mime") not in ("image/jpeg", "image/png"):
        return None
    source = info.get("descriptionurl") or ""
    image = info.get("url") or ""
    if not isinstance(source, str) or not isinstance(image, str):
        return None
    if (
        urlsplit(source).scheme != "https"
        or urlsplit(source).hostname != "commons.wikimedia.org"
    ):
        return None
    if not urlsplit(source).path.startswith("/wiki/File:"):
        return None
    if (
        urlsplit(image).scheme != "https"
        or urlsplit(image).hostname != "upload.wikimedia.org"
    ):
        return None
    metadata = info.get("extmetadata") or {}
    if not isinstance(metadata, dict):
        return None
    license_field = metadata.get("LicenseShortName") or {}
    artist_field = metadata.get("Artist") or {}
    if not isinstance(license_field, dict) or not isinstance(artist_field, dict):
        return None
    license_name = license_field.get("value") or ""
    creator = artist_field.get("value") or ""
    if not isinstance(license_name, str) or not isinstance(creator, str):
        return None
    if _normalize_license(license_name) not in _ALLOWED_LICENSES or not creator.strip():
        return None
    return PortraitEvidence(entity_id, media_id, source, image, license_name, creator)
