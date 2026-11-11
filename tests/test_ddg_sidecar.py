"""Offline source coverage for unknown-license DDG sidecar roundtrips."""

import json

import pytest

from autoshorts.modules import image_searcher as m
from autoshorts.modules.image_sources import SourceCandidate


@pytest.fixture
def sidecar(tmp_path, monkeypatch):
    monkeypatch.setattr(m, "ALLOW_UNVERIFIED_IMAGES", True)
    monkeypatch.setattr(m, "OUTPUT_DIR", tmp_path)
    image = tmp_path / "photo.jpg"
    image.write_bytes(b"local-placeholder-no-download")
    candidate = SourceCandidate(
        url="https://photos.example.invalid/a.jpg",
        provider="ddg",
        title="Flamengo Grêmio",
        fit_evidence="flamengo,gremio",
        license=None,
    )
    m.ImageSearcher._write_provenance_sidecar(image, candidate, "Flamengo Grêmio")
    written = json.loads(image.with_suffix(".credits.json").read_text("utf-8"))
    return image, written


def _write_and_read(image, record):
    image.with_suffix(".credits.json").write_text(json.dumps(record), encoding="utf-8")
    return m.ImageSearcher._read_sidecar(image)


def test_unknown_license_roundtrip_keeps_provenance_and_full_warning_credits(
    sidecar, tmp_path, monkeypatch
):
    image, written = sidecar
    assert written["kind"] == "web"
    assert written["provider"] == "ddg"
    assert written["license"] == ""
    assert written["license_verified"] is False
    assert written["unverified_opt_in"] is True
    assert m.ImageSearcher._read_sidecar(image) == written

    messages = []
    monkeypatch.setattr(
        m, "log", lambda message, level: messages.append((message, level))
    )
    m.ImageSearcher._write_credits_file([str(image)])
    credits = (tmp_path / "image_credits.txt").read_text("utf-8")
    status = "PUBLICATION REVIEW REQUIRED: 1 image(s) without verified reuse rights"
    assert credits.startswith(status + "\n")
    for line in (
        image.name,
        "Title: Flamengo Grêmio",
        "Creator: not provided by source",
        "Source: https://photos.example.invalid/a.jpg",
        "Image file: https://photos.example.invalid/a.jpg",
        "License: UNVERIFIED - reuse rights unknown",
        f"Changes: {written['modification']}",
    ):
        assert line in credits.splitlines()
    assert "no valid provenance record" not in credits
    assert (status, "WARNING") in messages
    assert json.loads(image.with_suffix(".credits.json").read_text("utf-8")) == written
    assert m.ImageSearcher._read_sidecar(image)["license"] == ""


def test_unknown_license_record_does_not_bypass_strict_cache_gate(sidecar, monkeypatch):
    image, written = sidecar
    assert m.ImageSearcher._usable_cache(image, "landscape")
    monkeypatch.setattr(m, "ALLOW_UNVERIFIED_IMAGES", False)
    assert m.ImageSearcher._read_sidecar(image) == written
    assert not m.ImageSearcher._usable_cache(image, "landscape")
    assert not m.ImageSearcher._usable_cache(image, "Flamengo Grêmio")


@pytest.mark.parametrize("key", ["license_verified", "unverified_opt_in"])
@pytest.mark.parametrize("value", [None, 0, 1, "false", "true", [], {}])
def test_unknown_license_flags_require_exact_boolean_values(sidecar, key, value):
    image, record = sidecar
    record[key] = value
    assert _write_and_read(image, record) is None


@pytest.mark.parametrize(
    "verified,opt_in", [(True, True), (True, False), (False, False)]
)
def test_empty_license_needs_unverified_rights_and_explicit_opt_in(
    sidecar, verified, opt_in
):
    image, record = sidecar
    record.update(license_verified=verified, unverified_opt_in=opt_in)
    assert _write_and_read(image, record) is None


@pytest.mark.parametrize(
    "key",
    [
        "schema",
        "kind",
        "title",
        "source_page",
        "image_url",
        "license",
        "provider",
        "license_verified",
        "unverified_opt_in",
    ],
)
def test_unknown_license_does_not_excuse_missing_required_fields(sidecar, key):
    image, record = sidecar
    del record[key]
    assert _write_and_read(image, record) is None


@pytest.mark.parametrize("key", ["title", "source_page", "image_url", "provider"])
@pytest.mark.parametrize("value", [None, False, True, 1, 1.0, [], {}, "", " \t\n"])
def test_other_required_fields_stay_nonempty_strings(sidecar, key, value):
    image, record = sidecar
    record[key] = value
    assert _write_and_read(image, record) is None


@pytest.mark.parametrize(
    "key,value",
    [
        ("schema", None),
        ("schema", 0),
        ("schema", 2),
        ("schema", "1"),
        ("kind", "ai"),
        ("kind", "stock"),
        ("kind", None),
        ("source_page", "file:///photo.jpg"),
        ("source_page", "//photos.example.invalid/page"),
        ("image_url", "data:image/jpeg;base64,AA"),
        ("image_url", "ftp://photos.example.invalid/a.jpg"),
    ],
)
def test_schema_kind_and_url_checks_still_apply(sidecar, key, value):
    image, record = sidecar
    record[key] = value
    assert _write_and_read(image, record) is None


@pytest.mark.parametrize("value", [None, False, True, 1, [], {}, " ", "\t\n"])
def test_unknown_license_must_be_the_empty_string_not_a_coercion(sidecar, value):
    image, record = sidecar
    record["license"] = value
    assert _write_and_read(image, record) is None


@pytest.mark.parametrize("value", ["", " ", "\t\n", None, False, 1, [], {}])
def test_verified_rights_still_require_a_nonempty_license(sidecar, value):
    image, record = sidecar
    record.update(license=value, license_verified=True, unverified_opt_in=False)
    assert _write_and_read(image, record) is None


@pytest.mark.parametrize("verified", [False, True])
def test_nonempty_license_preserves_existing_reader_rules(sidecar, verified):
    image, record = sidecar
    record.update(
        license="CC BY 4.0", license_verified=verified, unverified_opt_in=False
    )
    assert _write_and_read(image, record) == record


@pytest.mark.parametrize("raw", ["", "{", "null", "[]", "true", "1", '"text"'])
def test_malformed_or_nonobject_sidecars_still_fail_closed(sidecar, raw):
    image, _ = sidecar
    image.with_suffix(".credits.json").write_text(raw, encoding="utf-8")
    assert m.ImageSearcher._read_sidecar(image) is None
