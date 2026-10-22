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
    blob = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * 60.0**2))
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


def _noise_image(w=2400, h=600, seed=7) -> np.ndarray:
    """Sharp high-frequency texture: ideal for measuring blur."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)


def _row_sharpness(img: np.ndarray) -> np.ndarray:
    """Mean absolute horizontal gradient per row."""
    return np.abs(np.diff(img.astype(np.float32), axis=1)).mean(axis=(1, 2))


def test_blur_fill_keeps_foreground_sharp_and_blurs_fill():
    img = _noise_image()  # 4:1 into 9:16 -> blurred fill path
    framed = frame_subject(img, TW, TH)
    assert framed.shape == (TH, TW, 3)
    sharp = _row_sharpness(framed)
    # The sharp contain-scaled copy is placed around the vertical middle;
    # rows well above and below it are pure blurred fill.
    fg = sharp[TH // 2 - 30 : TH // 2 + 30].mean()
    fill = np.concatenate([sharp[:80], sharp[-80:]]).mean()
    assert fill < fg * 0.2  # fill rows lost the detail to the blur
    assert fg > 5  # foreground band retains real high-frequency energy


def test_face_region_drives_crop(monkeypatch):
    from autoshorts.modules import image_framing

    img = _blob_image()  # also has a salient blob at x=0.15
    # A face far to the right: detection must win over saliency/center.
    fx, fy, size = 0.85 * img.shape[1], 0.40 * img.shape[0], 120.0
    monkeypatch.setattr(image_framing, "_face_centers", lambda gray: [(fx, fy, size)])
    framed = frame_subject(img, TW, TH)
    assert framed.shape == (TH, TW, 3)
    # Cover-crop of 1200x800 to 9:16 keeps a 450px-wide window. A center
    # crop keeps x in [375, 825); the face at x=1020 is outside it, so the
    # crop must have shifted right to include the face region.
    gray = framed.mean(axis=2)
    # Draw nothing: instead verify via the crop window math - recompute the
    # window the same image yields with and without the detected face.
    import cv2

    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    monkeypatch.setattr(image_framing, "_face_centers", lambda gray: [])
    salient = image_framing._saliency_center(g)
    assert salient is not None and salient[0] < 0.35  # blob, far left
    # With the face patched out, the subject is the left blob; with the
    # face, subject_point must return the face location.
    monkeypatch.setattr(image_framing, "_face_centers", lambda gray: [(fx, fy, size)])
    px, py = image_framing.subject_point(img)
    assert abs(px - fx / img.shape[1]) < 0.01
    assert py < fy / img.shape[0]  # biased slightly above face center
    assert gray.std() > 0  # framed output is a real image


def test_face_crop_window_contains_face(monkeypatch):
    from autoshorts.modules import image_framing

    # Marker image: bright square where the face is, dark elsewhere.
    img = np.full((800, 1200, 3), 30, np.uint8)
    fx, fy, size = 1020.0, 320.0, 120.0
    img[int(fy) - 40 : int(fy) + 40, int(fx) - 40 : int(fx) + 40] = 255
    monkeypatch.setattr(image_framing, "_face_centers", lambda gray: [(fx, fy, size)])
    framed = frame_subject(img, TW, TH)
    assert framed.shape == (TH, TW, 3)
    # The bright marker must survive in the framed output; a center crop
    # (x window [375, 825)) would have excluded the marker at x=1020.
    assert framed.max() > 200
    bright = np.nonzero(framed.mean(axis=2) > 200)
    assert len(bright[0]) > 100  # a real region, not a stray pixel


def test_missing_cv2_falls_back_to_legacy_center_crop(tmp_path, monkeypatch):
    import builtins

    from autoshorts.modules import image_framing

    real_import = builtins.__import__

    def no_cv2(name, *args, **kwargs):
        if name == "cv2" or name.startswith("cv2."):
            raise ImportError("No module named 'cv2'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_cv2)
    src = tmp_path / "in.jpg"
    Image.fromarray(_blob_image()).save(src, "JPEG")
    out = tmp_path / "out.jpg"
    frame_subject_file(src, out, TW, TH)
    expected = tmp_path / "expected.jpg"
    image_framing._center_crop(src, expected, TW, TH)
    assert out.read_bytes() == expected.read_bytes()


def test_framing_error_falls_back_to_legacy_center_crop(tmp_path, monkeypatch):
    from autoshorts.modules import image_framing

    def boom(*args, **kwargs):
        raise RuntimeError("detector exploded")

    monkeypatch.setattr(image_framing, "frame_subject", boom)
    src = tmp_path / "in.jpg"
    Image.fromarray(_blob_image()).save(src, "JPEG")
    out = tmp_path / "out.jpg"
    frame_subject_file(src, out, TW, TH)  # must not raise
    expected = tmp_path / "expected.jpg"
    image_framing._center_crop(src, expected, TW, TH)
    assert out.read_bytes() == expected.read_bytes()
