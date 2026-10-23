"""Pictured-subject gate: real pixels, local model, fails closed."""

import pytest
from PIL import Image

from autoshorts.modules import image_subject


def test_missing_model_fails_closed(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOSHORTS_CLIP_DIR", str(tmp_path / "none"))
    image_subject._load.cache_clear()
    p = tmp_path / "x.jpg"
    Image.new("RGB", (300, 300), "gray").save(p)
    with pytest.raises(image_subject.SubjectUnavailableError):
        image_subject.check_subject(p, "Apple iPhone launch")


def test_low_probability_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(image_subject, "subject_probability", lambda p, s: 0.01)
    assert "does not look like" in image_subject.check_subject(tmp_path / "a", "x")
    monkeypatch.setattr(image_subject, "subject_probability", lambda p, s: 0.9)
    assert image_subject.check_subject(tmp_path / "a", "x") is None


@pytest.mark.skipif(
    not (image_subject.model_dir() / "tokenizer.json").exists(),
    reason="CLIP model files not installed (see image_subject.download_model)",
)
def test_real_model_rejects_a_plain_non_matching_image(tmp_path):
    p = tmp_path / "gray.jpg"
    Image.new("RGB", (400, 300), (120, 120, 120)).save(p)
    assert image_subject.check_subject(p, "Apple iPhone launch") is not None


def test_scope_is_recorded_and_never_claims_entity_verification():
    assert "NOT verified" in image_subject.VERIFICATION_SCOPE
    assert "entity" in image_subject.VERIFICATION_SCOPE


def test_sidecar_records_the_class_level_scope(tmp_path):
    import json

    from autoshorts.modules import image_sources
    from autoshorts.modules.image_searcher import ImageSearcher

    cand = image_sources.SourceCandidate(
        url="https://upload.wikimedia.org/p.jpg",
        provider="commons",
        license="CC BY-SA 4.0",
        fit_evidence="a,b",
        title="t",
    )
    img = tmp_path / "x.jpg"
    img.write_bytes(b"jpg")
    ImageSearcher._write_provenance_sidecar(img, cand, "q")
    record = json.loads(ImageSearcher._sidecar_path(img).read_text())
    assert "NOT verified" in record["pictured_subject_check"]


@pytest.mark.parametrize(
    "query,broad",
    [
        ("a bowl of fruit on a table", True),
        ("close up of a laptop keyboard", True),
        ("cidade à noite", True),
        ("Clarence Seedorf Corinthians 2012", False),
        ("seedorf corinthians", False),  # lowercase names are unknown words
        ("Apple iPhone launch", False),
        ("iphone launch keynote", False),
        ("football league table 2013", False),
        ("Kaká", False),
        ("", False),
    ],
)
def test_only_plainly_generic_requests_are_broad(query, broad):
    assert image_subject.is_broad_request(query) is broad
