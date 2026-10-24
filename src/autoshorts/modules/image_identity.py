"""Conservative Commons portrait evidence extraction, not image admission.

A P180 statement is an authored source assertion, not independent visual proof.
This module never changes the searcher's fail-closed specific-image policy.
Callers must resolve the requested entity independently and inspect representative
pixels before proposing any production admission rule. Event/date/team requests
are intentionally outside this portrait-only evidence contract.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass, field
from urllib.parse import unquote, urlsplit

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
    scope: str = field(default=_SCOPE, init=False)


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
        or identity_unambiguous is not True
        or not _QID.fullmatch(entity_id)
    ):
        return None
    if not isinstance(context_constraints, tuple) or context_constraints:
        return None
    page_id = page.get("pageid")
    if not isinstance(page_id, int) or isinstance(page_id, bool) or page_id <= 0:
        return None
    media_id = f"M{page_id}"
    if media_entity.get("id") != media_id:
        return None
    if any(
        key in record
        for record in (page, media_entity)
        for key in ("missing", "invalid")
    ):
        return None
    for record in (page, media_entity):
        if "ns" in record and (type(record["ns"]) is not int or record["ns"] != 6):
            return None
        if "pageid" in record and (
            type(record["pageid"]) is not int or record["pageid"] != page_id
        ):
            return None
    statements = media_entity.get("statements") or {}
    if not isinstance(statements, dict):
        return None
    claims = statements.get("P180", [])
    if not isinstance(claims, list):
        return None
    if any(
        not isinstance(c, dict)
        or c.get("rank") not in ("normal", "preferred", "deprecated")
        for c in claims
    ):
        return None
    preferred = any(c.get("rank") == "preferred" for c in claims)
    exact = False
    active_ids = set()
    for claim in claims:
        if claim.get("rank") == "deprecated" or (
            preferred and claim.get("rank") != "preferred"
        ):
            continue
        if claim.get("type") != "statement" or claim.get("qualifiers"):
            return None
        snak = claim.get("mainsnak") or {}
        if not isinstance(snak, dict):
            return None
        data = snak.get("datavalue") or {}
        if not isinstance(data, dict):
            return None
        value = data.get("value") or {}
        if not isinstance(value, dict):
            return None
        claim_id = value.get("id")
        if not isinstance(claim_id, str) or not _QID.fullmatch(claim_id):
            return None
        if (
            value.get("entity-type") != "item"
            or type(value.get("numeric-id")) is not int
            or value["numeric-id"] != int(claim_id[1:])
        ):
            return None
        active_ids.add(claim_id)
        if (
            snak.get("snaktype") == "value"
            and snak.get("property") == "P180"
            and snak.get("datatype") == "wikibase-item"
            and data.get("type") == "wikibase-entityid"
            and isinstance(value, dict)
            and value.get("id") == entity_id
            and value.get("entity-type") == "item"
            and type(value.get("numeric-id")) is int
            and value.get("numeric-id") == int(entity_id[1:])
        ):
            exact = True
    if not exact or active_ids != {entity_id}:
        return None
    infos = page.get("imageinfo") or []
    if not isinstance(infos, list) or len(infos) != 1 or not isinstance(infos[0], dict):
        return None
    info = infos[0]
    if info.get("mime") not in ("image/jpeg", "image/png"):
        return None
    source = info.get("descriptionurl")
    image = info.get("url")
    if not isinstance(source, str) or not isinstance(image, str):
        return None
    try:
        source_parts, image_parts = urlsplit(source), urlsplit(image)
        if any(
            p.username or p.password or p.port or p.fragment
            for p in (source_parts, image_parts)
        ):
            return None
    except ValueError:
        return None
    if (
        source_parts.scheme != "https"
        or source_parts.hostname != "commons.wikimedia.org"
    ):
        return None
    if image_parts.scheme != "https" or image_parts.hostname != "upload.wikimedia.org":
        return None
    if source_parts.netloc != "commons.wikimedia.org" or source_parts.query:
        return None
    if image_parts.netloc != "upload.wikimedia.org":
        return None
    prefix = "/wiki/File:"
    if not source_parts.path.startswith(prefix):
        return None
    filename = unquote(source_parts.path[len(prefix) :]).replace(" ", "_")
    if not filename or "/" in filename or filename in (".", ".."):
        return None
    for record in (page, media_entity):
        if "title" in record and record["title"] != "File:" + filename.replace(
            "_", " "
        ):
            # MediaWiki titles commonly preserve underscores; normalize both.
            title = record["title"]
            if (
                not isinstance(title, str)
                or title.replace(" ", "_") != "File:" + filename
            ):
                return None
    image_path = unquote(image_parts.path)
    if not re.fullmatch(r"/wikipedia/commons/[0-9a-f]/[0-9a-f]{2}/[^/]+", image_path):
        return None
    if image_path.rsplit("/", 1)[-1].replace(" ", "_") != filename:
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
    # Rights metadata is not HTML prose: reject markup that can conceal a
    # conflicting license in attributes. Creator prose may be HTML but needs
    # visible nonempty text after decoding/stripping tags.
    for key in ("NonFreeRestrictions", "NonFree", "Restrictions"):
        restrictions = metadata.get(key)
        if restrictions:
            if not isinstance(restrictions, dict) or restrictions.get("value"):
                return None
    # LicenseShortName must be plain text, not encoded presentation markup.
    # Reject entity-bearing values at the first decode boundary rather than
    # recursively stripping them into a misleading allowlisted license.
    if html.unescape(license_name) != license_name:
        return None
    if "<" in license_name or ">" in license_name:
        return None
    creator = html.unescape(re.sub(r"<[^>]*>", " ", creator)).strip()
    if (
        _normalize_license(license_name) not in _ALLOWED_LICENSES
        or not creator
        or creator.casefold() in ("unknown", "unknown author", "anonymous")
    ):
        return None
    return PortraitEvidence(entity_id, media_id, source, image, license_name, creator)
