"""Tests for overlay text detection + inpainting."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from autoshorts.modules.text_removal import (
    _dependencies_available,
    remove_overlay_text,
)

FONT = str(
    Path(__file__).parent.parent
    / "src"
    / "autoshorts"
    / "fonts"
    / "BebasNeue-Regular.ttf"
)


def _photo_with_text(
    tmp_path: Path,
    text: str = "50% OFF",
    fill: tuple = (255, 255, 255),
    xy: tuple = (60, 380),
    name: str = "with_text.jpg",
) -> Path:
    """Synthetic photo-like image with a big text overlay."""
    # Smooth photo-like background; OCR needs real-ish images to see text.
    # Realistic canvas: the image pipeline only accepts images >= 720px.
    y = np.linspace(30, 200, 900)[:, None, None]
    x = np.linspace(0, 120, 700)[None, :, None]
    base = np.broadcast_to(y + x, (900, 700, 3)).copy().astype(np.uint8)
    img = Image.fromarray(base, "RGB")
    draw = ImageDraw.Draw(img)
    draw.text(xy, text, fill=fill, font=ImageFont.truetype(FONT, 90))
    path = tmp_path / name
    img.save(path, "JPEG", quality=95)
    return path


def _assert_no_confident_text(image_path: Path) -> None:
    """OCR reads no confident text in either polarity."""
    import pytesseract

    cleaned = np.array(Image.open(image_path).convert("RGB"))
    for probe in (cleaned, 255 - cleaned):
        data = pytesseract.image_to_data(
            probe,
            lang="eng",
            config="--psm 11",
            output_type=pytesseract.Output.DICT,
        )
        assert not [
            t
            for t, c in zip(data["text"], data["conf"], strict=False)
            if str(t).strip() and float(c) >= 60
        ]


@pytest.mark.skipif(
    not _dependencies_available(), reason="cv2/pytesseract/tesseract missing"
)
def test_remove_overlay_text_erases_text(tmp_path):
    src = _photo_with_text(tmp_path)
    out = tmp_path / "cleaned.jpg"
    removed = remove_overlay_text(src, out)
    assert removed is True
    assert out.exists()
    _assert_no_confident_text(out)


@pytest.mark.skipif(
    not _dependencies_available(), reason="cv2/pytesseract/tesseract missing"
)
def test_remove_overlay_text_erases_dark_text_on_light(tmp_path):
    # Dark text on a bright region: the opposite polarity of the default fixture.
    src = _photo_with_text(
        tmp_path, fill=(20, 20, 20), xy=(150, 720), name="dark_on_light.jpg"
    )
    out = tmp_path / "cleaned.jpg"
    removed = remove_overlay_text(src, out)
    assert removed is True
    _assert_no_confident_text(out)


@pytest.mark.skipif(
    not _dependencies_available(), reason="cv2/pytesseract/tesseract missing"
)
def test_remove_overlay_text_passes_clean_image_through(tmp_path):
    rng = np.random.default_rng(3)
    # Smooth, photo-like texture: raw noise shifts too much under double JPEG
    # for a meaningful pixel comparison.
    img = Image.fromarray(
        (rng.normal(120, 40, (300, 300, 3))).clip(0, 255).astype(np.uint8), "RGB"
    ).filter(ImageFilter.GaussianBlur(8))
    src = tmp_path / "clean.jpg"
    img.save(src, "JPEG", quality=95)
    out = tmp_path / "out.jpg"
    removed = remove_overlay_text(src, out)
    assert removed is False
    assert out.exists()
    # Pass-through must not alter the pixels (beyond JPEG re-encoding noise).
    before = np.asarray(img, dtype=np.int16)
    after = np.asarray(Image.open(out).convert("RGB"), dtype=np.int16)
    assert np.abs(before - after).mean() < 5


def test_remove_overlay_text_survives_odd_input(tmp_path):
    src = tmp_path / "tiny.jpg"
    Image.new("RGB", (8, 8), (10, 20, 30)).save(src, "JPEG")
    out = tmp_path / "out.jpg"
    # Must never raise, whatever the detectors do with an 8x8 image.
    remove_overlay_text(src, out)
    assert out.exists()


@pytest.mark.skipif(
    not _dependencies_available(), reason="cv2/pytesseract/tesseract missing"
)
def test_oversized_text_blob_is_not_erased(tmp_path):
    # A "text" region taller/larger than any real caption (e.g. a merged
    # background blob over the main subject) must survive, even when OCR
    # reads words in it: inpainting it would destroy the photo.
    y = np.linspace(30, 200, 600)[:, None, None]
    x = np.linspace(0, 120, 800)[None, :, None]
    base = np.broadcast_to(y + x, (600, 800, 3)).copy().astype(np.uint8)
    img = Image.fromarray(base, "RGB")
    d = ImageDraw.Draw(img)
    # 200px-tall text: the merged candidate covers ~35% of the image
    # height, far beyond any real caption line.
    d.text((80, 180), "MEGA", fill=(255, 255, 255), font=ImageFont.truetype(FONT, 200))
    src = tmp_path / "blob.jpg"
    img.save(src, "JPEG", quality=95)
    out = tmp_path / "out.jpg"
    removed = remove_overlay_text(src, out)
    assert removed is False
    # Output pixels stay essentially identical to the input.
    before = np.asarray(img, dtype=np.int16)
    after = np.asarray(Image.open(out), dtype=np.int16)
    assert np.abs(before - after).mean() < 5
