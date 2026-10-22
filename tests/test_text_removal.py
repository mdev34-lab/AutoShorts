"""Tests for overlay text detection + inpainting."""
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

from autoshorts.modules.text_removal import (
    _dependencies_available,
    remove_overlay_text,
)

FONT = str(Path(__file__).parent.parent / "fonts" / "BebasNeue-Regular.ttf")


def _photo_with_text(tmp_path: Path, text: str = "50% OFF") -> Path:
    """Synthetic photo-like image with a big text overlay."""
    # Smooth photo-like background; OCR needs real-ish images to see text.
    y = np.linspace(30, 200, 400)[:, None, None]
    x = np.linspace(0, 120, 600)[None, :, None]
    base = np.broadcast_to(y + x, (400, 600, 3)).copy().astype(np.uint8)
    img = Image.fromarray(base, "RGB")
    draw = ImageDraw.Draw(img)
    draw.text((60, 160), text, fill=(255, 255, 255), font=ImageFont.truetype(FONT, 90))
    path = tmp_path / "with_text.jpg"
    img.save(path, "JPEG", quality=95)
    return path


@pytest.mark.skipif(not _dependencies_available(), reason="cv2/pytesseract/tesseract missing")
def test_remove_overlay_text_erases_text(tmp_path):
    src = _photo_with_text(tmp_path)
    out = tmp_path / "cleaned.jpg"
    removed = remove_overlay_text(src, out)
    assert removed is True
    assert out.exists()
    # The real goal: OCR no longer reads confident text, in either polarity.
    import pytesseract

    cleaned = np.array(Image.open(out).convert("RGB"))
    for probe in (cleaned, 255 - cleaned):
        data = pytesseract.image_to_data(
            probe, lang="eng", config="--psm 11",
            output_type=pytesseract.Output.DICT,
        )
        assert not [
            t for t, c in zip(data["text"], data["conf"])
            if str(t).strip() and float(c) >= 60
        ]


@pytest.mark.skipif(not _dependencies_available(), reason="cv2/pytesseract/tesseract missing")
def test_remove_overlay_text_passes_clean_image_through(tmp_path):
    rng = np.random.default_rng(3)
    img = Image.fromarray(
        (rng.normal(120, 40, (300, 300, 3))).clip(0, 255).astype(np.uint8), "RGB"
    )
    src = tmp_path / "clean.jpg"
    img.save(src, "JPEG", quality=95)
    out = tmp_path / "out.jpg"
    removed = remove_overlay_text(src, out)
    assert removed is False
    assert out.exists()


def test_remove_overlay_text_survives_odd_input(tmp_path):
    src = tmp_path / "tiny.jpg"
    Image.new("RGB", (8, 8), (10, 20, 30)).save(src, "JPEG")
    out = tmp_path / "out.jpg"
    # Must never raise, whatever the detectors do with an 8x8 image.
    remove_overlay_text(src, out)
    assert out.exists()
