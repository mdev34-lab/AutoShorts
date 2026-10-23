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
