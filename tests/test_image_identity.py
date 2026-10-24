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
                    "type": "statement",
                    "rank": "normal",
                    "mainsnak": {
                        "snaktype": "value",
                        "property": "P180",
                        "datatype": "wikibase-item",
                        "datavalue": {
                            "type": "wikibase-entityid",
                            "value": {
                                "entity-type": "item",
                                "numeric-id": 484909,
                                "id": "Q484909",
                            },
                        },
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


def real_origin_chain_fixture():
    import json
    from pathlib import Path

    root = Path(__file__).parent / "fixtures"
    return (
        json.loads((root / "commons-file166913476.json").read_text()),
        json.loads((root / "commons-M166913476.json").read_text()),
    )


def test_documented_portrait_without_structured_identity_remains_unproven():
    page, media = real_origin_chain_fixture()
    assert page["pageid"] == 166913476
    assert (
        page["imageinfo"][0]["extmetadata"]["LicenseShortName"]["value"] == "CC BY 4.0"
    )
    assert extract(page, media) is None


@pytest.mark.xfail(
    strict=True,
    reason="Known #13 recall gap: source-faithful 2025 portrait has origin chain and reusable license but no P180; origin-chain evidence lookup not implemented",
)
def test_candidate_evidence_lookup_covers_documented_origin_chain():
    page, media = real_origin_chain_fixture()
    assert extract(page, media) is not None


@pytest.mark.parametrize("flag", ["false", "0", 2, [True], 1])
def test_identity_flag_requires_literal_boolean_true(flag):
    assert (
        portrait_evidence(*fixture(), entity_id="Q484909", identity_unambiguous=flag)
        is None
    )


@pytest.mark.parametrize("constraints", [None, [], "", {}, False])
def test_context_requires_explicit_typed_tuple(constraints):
    assert extract(*fixture(), context_constraints=constraints) is None


@pytest.mark.parametrize(
    "url",
    [
        "https://[bad/wiki/File:x.jpg",
        "https://upload.wikimedia.org/",
        "https://upload.wikimedia.org/wikipedia/en/a/ab/x.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/a/ab/x.jpg/200px-x.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/a/ab/Other.jpg",
    ],
)
def test_upload_must_bind_the_original_commons_file(url):
    page, media = fixture()
    page["imageinfo"][0]["url"] = url
    assert extract(page, media) is None


@pytest.mark.parametrize(
    "field,value",
    [
        ("Artist", "<span></span>"),
        ("LicenseShortName", '<span title="CC BY-NC 4.0">CC BY 4.0</span>'),
        ("NonFreeRestrictions", "non-commercial use only"),
    ],
)
def test_rights_fields_cannot_hide_restrictions_or_empty_creator(field, value):
    page, media = fixture()
    page["imageinfo"][0]["extmetadata"][field] = {"value": value}
    assert extract(page, media) is None


def test_actual_rich_origin_chain_without_depicts_is_not_identity():
    import json
    from pathlib import Path

    media = json.loads(
        (Path(__file__).parent / "fixtures/commons-M166913476.json").read_text()
    )
    assert sum(len(v) for v in media["statements"].values()) == 9
    assert "P180" not in media["statements"]
    assert "P7482" in media["statements"]  # rich source chain is not depicts
    page, _ = fixture()
    page["pageid"] = 166913476
    assert extract(page, media) is None


def test_preferred_other_subject_supersedes_stale_normal_target():
    page, media = fixture()
    other = deepcopy(media["statements"]["P180"][0])
    other["rank"] = "preferred"
    other["mainsnak"]["datavalue"]["value"] = {
        "id": "Q11571",
        "entity-type": "item",
        "numeric-id": 11571,
    }
    media["statements"]["P180"].append(other)
    assert extract(page, media) is None


@pytest.mark.parametrize(
    "mutation", ["invalid-rank", "numeric-mismatch", "wrong-type", "wrong-datatype"]
)
def test_corrupted_claim_is_not_exact_identity(mutation):
    page, media = fixture()
    claim = media["statements"]["P180"][0]
    if mutation == "invalid-rank":
        claim["rank"] = "garbage"
    elif mutation == "numeric-mismatch":
        claim["mainsnak"]["datavalue"]["value"]["numeric-id"] = 42
    elif mutation == "wrong-type":
        claim["type"] = "garbage"
    else:
        claim["mainsnak"]["datatype"] = "string"
    assert extract(page, media) is None


@pytest.mark.parametrize(
    "record,field,value",
    [
        ("page", "title", "File:Wrong.jpg"),
        ("page", "ns", 0),
        ("page", "missing", ""),
        ("media", "pageid", 999),
        ("media", "title", "File:Wrong.jpg"),
        ("media", "invalid", ""),
    ],
)
def test_optional_live_record_fields_must_bind(record, field, value):
    page, media = fixture()
    (page if record == "page" else media)[field] = value
    assert extract(page, media) is None


@pytest.mark.parametrize("key", ["NonFree", "Restrictions"])
def test_additional_nonfree_flags_reject(key):
    page, media = fixture()
    page["imageinfo"][0]["extmetadata"][key] = {"value": "true"}
    assert extract(page, media) is None


def test_corrupt_sibling_does_not_allow_valid_target():
    page, media = fixture()
    media["statements"]["P180"].insert(0, "bad")
    assert extract(page, media) is None


def test_co_depiction_is_not_single_person_portrait():
    page, media = fixture()
    other = deepcopy(media["statements"]["P180"][0])
    other["mainsnak"]["datavalue"]["value"] = {
        "id": "Q11571",
        "entity-type": "item",
        "numeric-id": 11571,
    }
    media["statements"]["P180"].append(other)
    assert extract(page, media) is None


@pytest.mark.parametrize(
    "license_text",
    [
        "CC BY 4.0 &lt;non-commercial&gt;",
        "CC BY 4.0 &amp;lt;non-commercial&amp;gt;",
        "CC BY 4.0 &amp;amp;lt;non-commercial&amp;amp;gt;",
    ],
)
def test_encoded_markup_cannot_hide_license_restrictions(license_text):
    page, media = fixture()
    page["imageinfo"][0]["extmetadata"]["LicenseShortName"]["value"] = license_text
    assert extract(page, media) is None


@pytest.mark.parametrize(
    "url",
    [
        "https://@commons.wikimedia.org/wiki/File:Clarence_Seedorf_2021.jpg",
        "https://commons.wikimedia.org:/wiki/File:Clarence_Seedorf_2021.jpg",
        "https://commons.wikimedia.org/wiki/File:Clarence_Seedorf_2021.jpg?title=File:Other.jpg",
    ],
)
def test_source_authority_and_query_are_canonical(url):
    page, media = fixture()
    page["imageinfo"][0]["descriptionurl"] = url
    assert extract(page, media) is None


@pytest.mark.parametrize("name", [".", ".."])
def test_dotsegment_file_is_not_a_portrait_filename(name):
    page, media = fixture()
    page["imageinfo"][0]["descriptionurl"] = (
        "https://commons.wikimedia.org/wiki/File:" + name
    )
    page["imageinfo"][0]["url"] = (
        "https://upload.wikimedia.org/wikipedia/commons/a/ab/" + name
    )
    assert extract(page, media) is None
