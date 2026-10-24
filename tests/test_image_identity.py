"""Exact authored identity evidence does not establish event or pixel truth."""

from copy import deepcopy

import pytest

from autoshorts.modules.image_identity import portrait_evidence


def fixture():
    page = {
        "pageid": 111425118,
        "imageinfo": [
            {
                "mime": "image/jpeg",
                "url": "https://upload.wikimedia.org/wikipedia/commons/c/ce/Clarence_Seedorf_2021.jpg",
                "descriptionurl": "https://commons.wikimedia.org/wiki/File:Clarence_Seedorf_2021.jpg",
                "extmetadata": {
                    "LicenseShortName": {"value": "CC BY-SA 3.0"},
                    "Artist": {"value": "Кирилл Венедиктов"},
                },
            }
        ],
    }
    media = {
        "id": "M111425118",
        "statements": {
            "P180": [
                {
                    "rank": "normal",
                    "mainsnak": {
                        "snaktype": "value",
                        "property": "P180",
                        "datavalue": {"value": {"id": "Q484909"}},
                    },
                }
            ]
        },
    }
    return page, media


def extract(page, media, **kwargs):
    return portrait_evidence(
        page, media, entity_id="Q484909", identity_unambiguous=True, **kwargs
    )


def test_exact_source_identity_returns_explicit_limited_evidence():
    page, media = fixture()
    e = extract(page, media)
    assert e and e.entity_id == "Q484909" and e.media_id == "M111425118"
    assert "authored" in e.scope and "NOT verified" in e.scope
    assert e.license == "CC BY-SA 3.0"


@pytest.mark.parametrize(
    "constraint", ["2012", "Corinthians", "match against Nautico", "launch event"]
)
def test_portrait_cannot_prove_date_team_or_event(constraint):
    assert extract(*fixture(), context_constraints=(constraint,)) is None


def test_identity_resolution_must_be_explicitly_unambiguous():
    assert portrait_evidence(*fixture(), entity_id="Q484909") is None


@pytest.mark.parametrize(
    "kind",
    [
        "other-person",
        "generic-person",
        "deprecated",
        "missing",
        "wrong-file",
        "unknown-snak",
        "no-claims",
        "no-license",
        "no-creator",
        "non-image",
        "foreign-source",
        "foreign-image",
    ],
)
def test_unproven_or_conflicting_source_records_fail_closed(kind):
    page, media = deepcopy(fixture())
    claim = media["statements"]["P180"][0]
    info = page["imageinfo"][0]
    if kind == "other-person":
        claim["mainsnak"]["datavalue"]["value"]["id"] = "Q11571"
    elif kind == "generic-person":
        claim["mainsnak"]["datavalue"]["value"]["id"] = "Q5"
    elif kind == "deprecated":
        claim["rank"] = "deprecated"
    elif kind == "missing":
        media["missing"] = ""
    elif kind == "wrong-file":
        media["id"] = "M42"
    elif kind == "unknown-snak":
        claim["mainsnak"]["snaktype"] = "somevalue"
    elif kind == "no-claims":
        media["statements"] = {}
    elif kind == "no-license":
        info["extmetadata"]["LicenseShortName"]["value"] = "All rights reserved"
    elif kind == "no-creator":
        info["extmetadata"]["Artist"]["value"] = ""
    elif kind == "non-image":
        info["mime"] = "image/svg+xml"
    elif kind == "foreign-source":
        info["descriptionurl"] = "https://example.org/wiki/File:portrait.jpg"
    elif kind == "foreign-image":
        info["url"] = "https://example.org/portrait.jpg"
    assert extract(page, media) is None


@pytest.mark.parametrize(
    "field,value",
    [
        ("statements", []),
        ("P180", "bad"),
        ("mainsnak", []),
        ("datavalue", "bad"),
        ("imageinfo", "bad"),
        ("extmetadata", []),
        ("Artist", "bad"),
        ("url", ["bad"]),
    ],
)
def test_malformed_provider_payload_is_not_evidence(field, value):
    page, media = fixture()
    if field == "statements":
        media[field] = value
    elif field == "P180":
        media["statements"][field] = value
    elif field == "mainsnak":
        media["statements"]["P180"][0][field] = value
    elif field == "datavalue":
        media["statements"]["P180"][0]["mainsnak"][field] = value
    elif field == "imageinfo":
        page[field] = value
    elif field == "extmetadata":
        page["imageinfo"][0][field] = value
    elif field == "Artist":
        page["imageinfo"][0]["extmetadata"][field] = value
    else:
        page["imageinfo"][0][field] = value
    assert extract(page, media) is None
