"""The pixel gate looks at the real downloaded image, not its title."""

import shutil

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

from autoshorts.modules import image_content
from autoshorts.modules.image_searcher import ImageSearcher

pytest.importorskip("pytesseract")
if not shutil.which("tesseract"):  # pragma: no cover
    pytest.skip("tesseract not installed", allow_module_level=True)


def _font():
    for p in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
    ):
        try:
            return ImageFont.truetype(p, 56)
        except OSError:
            continue
    pytest.skip("no TrueType font to render the fixture")


def _table(path, lines):
    img = Image.new("RGB", (1000, 700), "white")
    d = ImageDraw.Draw(img)
    for i, line in enumerate(lines):
        d.text((40, 40 + i * 100), line, fill="black", font=_font())
    img.save(path)


def _photo(path):
    rng = np.random.default_rng(1)
    arr = (rng.random((400, 600, 3)) * 255).astype("uint8")
    Image.fromarray(arr).save(path)


def test_reviewer_graphic_with_wrong_season_is_rejected(tmp_path):
    p = tmp_path / "t.jpg"
    _table(p, ["CORINTHIANS 2013", "6th PLACE", "17 POINTS", "SEASON TABLE"])
    reason = image_content.check_image_content(
        p, "Seedorf Corinthians 2012 third 42 points"
    )
    assert reason is not None


def test_text_light_photo_passes_the_pixel_gate(tmp_path):
    p = tmp_path / "p.jpg"
    _photo(p)
    assert image_content.check_image_content(p, "Seedorf Corinthians 2012") is None


def test_searcher_fails_closed_when_ocr_is_unavailable(tmp_path, monkeypatch):
    def boom(path, query):
        raise image_content.ContentUnavailableError("no tesseract")

    monkeypatch.setattr(image_content, "check_image_content", boom)
    reason = ImageSearcher._content_rejection(tmp_path / "x.jpg", "q")
    assert "could not be checked" in reason
