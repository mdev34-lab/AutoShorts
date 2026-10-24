"""Pixel-content gate for web images, run on the DOWNLOADED bytes.

Title and license checks say nothing about what the picture shows (a table of
the wrong season can carry every query word). This gate OCRs the real pixels:

* an image carrying a lot of legible text is a data graphic / screenshot, not
  a photo of the subject, and is rejected;
* any number read from the image that the query does not contain (2013 vs
  2012, 17 vs 42 points) rejects it when the query has numbers.

It does NOT prove the subject (a fruit bowl for "Apple iPhone" has no text and
passes); subject identity needs a vision model or a human. When OCR is not
available the candidate is rejected (fail closed).
"""

from __future__ import annotations

import re
from pathlib import Path

from PIL import Image

from .image_sources import _normalize

MAX_TEXT_WORDS = 6  # confident words above which the image is a text graphic
MIN_CONF = 60


class ContentUnavailableError(RuntimeError):
    pass


def _ocr_words(path: Path) -> list[str]:
    try:
        import pytesseract
    except ImportError as exc:  # pragma: no cover - environment
        raise ContentUnavailableError("pytesseract is not installed") from exc
    try:
        with Image.open(path) as img:
            rgb = img.convert("RGB")
            rgb.thumbnail((1600, 1600))
            data = pytesseract.image_to_data(
                rgb, output_type=pytesseract.Output.DICT, config="--psm 11"
            )
    except pytesseract.TesseractNotFoundError as exc:  # pragma: no cover
        raise ContentUnavailableError("tesseract binary is not available") from exc
    words = []
    for text, conf in zip(data["text"], data["conf"], strict=False):
        try:
            ok = float(conf) >= MIN_CONF
        except ValueError:
            ok = False
        if ok and re.search(r"[A-Za-z0-9]{2,}", text or ""):
            words.append(text)
    return words


def check_image_content(path: Path, query: str) -> str | None:
    """None when the pixels pass; otherwise the rejection reason.

    Raises ContentUnavailableError when OCR cannot run (caller must fail closed).
    """
    words = _ocr_words(path)
    if len(words) > MAX_TEXT_WORDS:
        return f"image carries {len(words)} legible words (data graphic/screenshot)"
    query_numbers = set(re.findall(r"\d+", _normalize(query)))
    image_numbers = set(re.findall(r"\d+", _normalize(" ".join(words))))
    if query_numbers and image_numbers - query_numbers:
        return f"image shows numbers {sorted(image_numbers - query_numbers)} not in the query"
    return None
