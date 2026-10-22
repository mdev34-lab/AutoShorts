"""Detect and remove overlaid text from downloaded images.

Web images often carry captions, price tags or social-media overlays. This
module finds text regions and erases them with OpenCV inpainting so rendered
videos show clean pictures.

Pipeline (deliberately "tosco", no ML models to download):
1. Sobel-x gradient + Otsu threshold + horizontal morphological close finds
   candidate text-line regions (polarity-agnostic).
2. Each candidate is OCR-checked with Tesseract (crop, polarity picked from
   border brightness) so textured backgrounds (crowds, grass, fabric) are
   not mistaken for text.
3. Confirmed regions are dilated and inpainted (Telea).

Safety guards: at most MAX_CANDIDATES regions per image; if confirmed text
covers more than MAX_COVERAGE of the image it is a text graphic, not a photo
with an overlay, and the original is kept untouched.

Best-effort and dependency-tolerant: if OpenCV, pytesseract or the tesseract
binary are unavailable, images pass through untouched.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from .config import (
    TEXT_REMOVAL_INPAINT_RADIUS,
    TEXT_REMOVAL_LANG,
    TEXT_REMOVAL_MAX_COVERAGE,
    TEXT_REMOVAL_MIN_CONFIDENCE,
)
from .logging_system import log

_MAX_CANDIDATES = 25
# A real overlay text line is a thin strip. Candidates taller than this
# fraction of the image, or covering more than this fraction of its area,
# are merged background blobs (faces, bodies, scenery crossed by texture)
# and erasing them does more damage than the text ever did.
_MAX_BOX_HEIGHT_FRAC = 0.15
_MAX_BOX_AREA_FRAC = 0.10


def _dependencies_available() -> bool:
    try:
        import cv2  # noqa: F401
        import pytesseract  # noqa: F401
    except ImportError:
        return False
    return shutil.which("tesseract") is not None


def _candidate_boxes(img_rgb: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Morphological text-line candidates, largest first."""
    import cv2

    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    width = gray.shape[1]
    grad_x = np.absolute(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=-1))
    lo, hi = float(grad_x.min()), float(grad_x.max())
    if hi - lo < 1:
        return []
    grad_x = (255 * (grad_x - lo) / (hi - lo)).astype("uint8")
    grad_x = cv2.threshold(grad_x, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    rect = cv2.getStructuringElement(cv2.MORPH_RECT, (max(9, width // 40), 3))
    closed = cv2.morphologyEx(grad_x, cv2.MORPH_CLOSE, rect)
    kernel3 = np.ones((3, 3), np.uint8)
    closed = cv2.erode(closed, kernel3, iterations=2)
    closed = cv2.dilate(closed, kernel3, iterations=2)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_h, img_w = gray.shape[:2]
    boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if h < 8 or w < 12 or w / float(h) < 0.5:
            continue
        if h > _MAX_BOX_HEIGHT_FRAC * img_h or (w * h) > _MAX_BOX_AREA_FRAC * (
            img_w * img_h
        ):
            continue
        if cv2.contourArea(c) / float(w * h) < 0.2:
            continue
        boxes.append((x, y, w, h))
    boxes.sort(key=lambda b: b[2] * b[3], reverse=True)
    return boxes[:_MAX_CANDIDATES]


def _ocr_confirms(img_rgb: np.ndarray, box: tuple[int, int, int, int]) -> bool:
    """True when tesseract reads real text inside the candidate crop."""
    import cv2
    import pytesseract

    x, y, w, h = box
    # Generous padding: tesseract reads tight crops poorly.
    pad = max(10, h // 3)
    crop = img_rgb[max(0, y - pad) : y + h + pad, max(0, x - pad) : x + w + pad]
    if crop.size == 0:
        return False
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    border = np.concatenate(
        [
            gray[0, :].ravel(),
            gray[-1, :].ravel(),
            gray[:, 0].ravel(),
            gray[:, -1].ravel(),
        ]
    )
    # Dark border hints light-on-dark text; tesseract prefers dark-on-light.
    # When the hint polarity reads nothing, try the other one.
    probes = [255 - gray, gray] if border.mean() < 110 else [gray, 255 - gray]
    for probe in probes:
        data = pytesseract.image_to_data(
            probe,
            lang=TEXT_REMOVAL_LANG,
            config="--psm 6",
            output_type=pytesseract.Output.DICT,
        )
        for text, conf in zip(data["text"], data["conf"], strict=False):
            try:
                confidence = float(conf)
            except (TypeError, ValueError):
                continue
            word = str(text).strip()
            # >= 2 alnum chars: single glyphs are too easily hallucinated
            # from textures (circuit vias read as "O", "8", ...).
            if (
                confidence >= TEXT_REMOVAL_MIN_CONFIDENCE
                and sum(ch.isalnum() for ch in word) >= 2
            ):
                return True
    return False


def _has_line_neighbor(
    box: tuple[int, int, int, int],
    candidates: list[tuple[int, int, int, int]],
) -> bool:
    """True when another candidate sits on the same text line (nearby, same
    vertical band) - real text comes in lines, hallucinations come alone."""
    x, y, w, h = box
    cy = y + h / 2
    for ox, oy, ow, oh in candidates:
        if (ox, oy, ow, oh) == (x, y, w, h):
            continue
        ocy = oy + oh / 2
        same_band = abs(ocy - cy) <= max(h, oh) * 0.75
        gap_x = max(0, max(x, ox) - min(x + w, ox + ow))
        if same_band and gap_x <= max(30, h):
            return True
    return False


def _atomic_jpeg_write(img: Image.Image, output_path: Path) -> None:
    """Write via a unique sibling temp file + atomic replace (no torn sidecars)."""
    tmp = output_path.with_name(f"{output_path.name}.{os.getpid()}.part")
    try:
        img.save(tmp, "JPEG", quality=95)
        os.replace(tmp, output_path)
    finally:
        tmp.unlink(missing_ok=True)


def _passthrough(image_path: Path, output_path: Path) -> None:
    with Image.open(image_path) as img:
        _atomic_jpeg_write(img.convert("RGB"), output_path)


def remove_overlay_text(image_path: Path, output_path: Path) -> bool:
    """Erase detected text from image_path into output_path.

    Returns True when at least one text region was inpainted. On any failure
    the original image is copied through and False is returned, so the video
    pipeline never breaks because of text removal.
    """
    image_path = Path(image_path)
    output_path = Path(output_path)
    if not _dependencies_available():
        log(
            "Text removal unavailable (cv2/pytesseract/tesseract missing), skipping",
            "WARNING",
        )
        _passthrough(image_path, output_path)
        return False
    import cv2  # safe: _dependencies_available() just verified it

    try:
        with Image.open(image_path) as img:
            rgb = np.array(img.convert("RGB"))
        candidates = _candidate_boxes(rgb)
        confirmed = [b for b in candidates if _ocr_confirms(rgb, b)]
        # Real overlays form lines: a confirmed box with no other candidate
        # nearby is almost always an OCR hallucination on texture.
        confirmed = [b for b in confirmed if _has_line_neighbor(b, candidates)]
        if not confirmed:
            _passthrough(image_path, output_path)
            return False
        mask = np.zeros(rgb.shape[:2], dtype=np.uint8)
        for x, y, w, h in confirmed:
            cv2.rectangle(mask, (x, y), (x + w, y + h), 255, thickness=-1)
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(mask, kernel, iterations=3)
        # Guard AFTER dilation: it measures what would actually be erased.
        coverage = float(np.count_nonzero(dilated)) / float(dilated.size)
        if coverage > TEXT_REMOVAL_MAX_COVERAGE:
            log(
                f"  Text removal: {coverage:.0%} of the image is text (graphic, not a"
                " photo) - keeping original",
                "WARNING",
            )
            _passthrough(image_path, output_path)
            return False
        cleaned = cv2.inpaint(
            cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
            mask,
            TEXT_REMOVAL_INPAINT_RADIUS,
            cv2.INPAINT_TELEA,
        )
        _atomic_jpeg_write(
            Image.fromarray(cv2.cvtColor(cleaned, cv2.COLOR_BGR2RGB)), output_path
        )
        log(f"  Text removal: inpainted {len(confirmed)} text region(s)", "INFO")
        return True
    except Exception as e:  # never break the pipeline on cleanup
        log(f"  Text removal failed ({e}); keeping original image", "WARNING")
        _passthrough(image_path, output_path)
        return False
