"""Tests for subject-aware framing."""
import numpy as np
from PIL import Image

from autoshorts.modules.image_framing import (
    _saliency_center,
    frame_subject,
    frame_subject_file,
)

TW, TH = 360, 640  # 9:16


def _blob_image(w=1200, h=800, blob_x=0.15, blob_y=0.5) -> np.ndarray:
    """Photo-like image with one bright subject off-center."""
    img = np.full((h, w, 3), 40, np.uint8)
    yy, xx = np.mgrid[0:h, 0:w]
    cx, cy = int(blob_x * w), int(blob_y * h)
    blob = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * 60.0 ** 2))
    img = np.clip(img + (blob * 215)[:, :, None], 0, 255).astype(np.uint8)
    return img


def test_saliency_finds_off_center_blob():
    import cv2

    img = _blob_image()
    center = _saliency_center(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY))
    assert center is not None
    assert center[0] < 0.35  # blob is at x=0.15, far left


def test_cover_crop_follows_subject():
    img = _blob_image()
    framed = frame_subject(img, TW, TH)
    assert framed.shape == (TH, TW, 3)
    gray = framed.mean(axis=2)
    # The bright blob must survive in the left half of the framed image,
    # while a center crop would have kept the dark middle of the source.
    assert gray[:, : TW // 2].mean() > gray[:, TW // 2 :].mean()
    assert gray.max() > 200


def test_blur_fill_for_very_wide_image():
    img = _blob_image(w=2400, h=600)  # 4:1 into 9:16 -> fill
    framed = frame_subject(img, TW, TH)
    assert framed.shape == (TH, TW, 3)
    # Foreground is the whole sharp image contained: corners of the
    # foreground band stay sharp while the fill above/below is blurred.
    assert framed.std() > 5


def test_frame_subject_file_writes_target_size(tmp_path):
    src = tmp_path / "in.jpg"
    Image.fromarray(_blob_image()).save(src, "JPEG")
    out = tmp_path / "out.jpg"
    frame_subject_file(src, out, TW, TH)
    with Image.open(out) as img:
        assert img.size == (TW, TH)


def test_frame_subject_file_tiny_image(tmp_path):
    src = tmp_path / "tiny.jpg"
    Image.new("RGB", (12, 12), (200, 30, 30)).save(src, "JPEG")
    out = tmp_path / "out.jpg"
    frame_subject_file(src, out, TW, TH)  # must not raise
    with Image.open(out) as img:
        assert img.size == (TW, TH)
