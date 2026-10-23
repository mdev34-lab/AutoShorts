"""Subject-aware framing for arbitrary web images.

`ImageSearcher.resize_to_fill` used to center-crop every image to the video
aspect ratio, which chops heads, products and the interesting half of wide
photos. This module instead:

1. Locates the likely subject: faces first (Haar cascade bundled with
   OpenCV), otherwise the spectral-residual saliency centroid (plain FFT,
   no contrib packages or model downloads), otherwise the image center.
2. Cover-crops around that point when the needed zoom is moderate.
3. Falls back to a blurred-background fill ("fill what falls outside the
   image") when the aspect mismatch is large: a blurred cover-scaled copy
   fills the frame and the sharp contain-scaled image sits on top, so
   nothing of the subject is cut.

Everything degrades gracefully: no cv2 -> old center crop; no detection ->
geometric center.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from .config import (
    SUBJECT_FILL_BLUR_RADIUS,
    SUBJECT_FILL_ZOOM_THRESHOLD,
)
from .logging_system import log


def _face_centers(gray: "np.ndarray") -> list[tuple[float, float, float]]:
    """(cx, cy, size) of detected faces, in pixels."""
    import cv2

    haarcascades: str = cv2.data.haarcascades  # type: ignore[attr-defined]
    cascade_path = str(Path(haarcascades) / "haarcascade_frontalface_default.xml")
    cascade = cv2.CascadeClassifier(cascade_path)
    if cascade.empty():
        return []
    faces = cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30)
    )
    return [(x + w / 2, y + h / 2, float(w)) for (x, y, w, h) in faces]


def _saliency_center(gray: "np.ndarray") -> tuple[float, float] | None:
    """Centroid of the most salient pixels via spectral residual (FFT)."""
    import cv2

    small = cv2.resize(gray, (256, 256))
    f = np.fft.fft2(small.astype(np.float32))
    log_amplitude = np.log(np.abs(f) + 1e-8)
    phase = np.angle(f)
    residual = log_amplitude - cv2.blur(log_amplitude, (3, 3))
    saliency = np.abs(np.fft.ifft2(np.exp(residual + 1j * phase))) ** 2
    saliency = cv2.GaussianBlur(saliency, (9, 9), 0)
    if saliency.max() <= 0:
        return None
    threshold = float(np.percentile(saliency.astype(np.float64), 92))
    ys, xs = np.nonzero(saliency >= threshold)
    if len(xs) == 0:
        return None
    return float(xs.mean() / 256), float(ys.mean() / 256)


def subject_point(img_rgb: "np.ndarray") -> tuple[float, float]:
    """Subject position as (cx, cy) fractions of width/height."""
    import cv2

    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    faces = _face_centers(gray)
    if faces:
        # Largest face wins; slightly above its center keeps foreheads safe.
        fx, fy, size = max(faces, key=lambda f: f[2])
        return (min(max(fx / w, 0.0), 1.0), min(max((fy - size * 0.15) / h, 0.0), 1.0))
    salient = _saliency_center(gray)
    if salient is not None:
        return salient
    return (0.5, 0.5)


def frame_subject(
    img_rgb: "np.ndarray",
    target_width: int,
    target_height: int,
) -> "np.ndarray":
    """Crop/fill img_rgb to exactly target_width x target_height."""
    import cv2

    h, w = img_rgb.shape[:2]
    target_ratio = target_width / target_height
    img_ratio = w / h
    cx_frac, cy_frac = subject_point(img_rgb)
    zoom_mismatch = max(img_ratio / target_ratio, target_ratio / img_ratio)

    if zoom_mismatch > SUBJECT_FILL_ZOOM_THRESHOLD:
        # Blurred fill: cover-scaled blurred background + sharp contained copy.
        cover_scale = max(target_width / w, target_height / h)
        bg = cv2.resize(
            img_rgb,
            (int(w * cover_scale) + 1, int(h * cover_scale) + 1),
            interpolation=cv2.INTER_AREA,
        )
        bx = max(0, (bg.shape[1] - target_width) // 2)
        by = max(0, (bg.shape[0] - target_height) // 2)
        bg = bg[by : by + target_height, bx : bx + target_width]
        ksize = SUBJECT_FILL_BLUR_RADIUS * 2 + 1
        bg = cv2.GaussianBlur(bg, (ksize, ksize), 0)

        contain_scale = min(target_width / w, target_height / h)
        fg = cv2.resize(
            img_rgb,
            (int(w * contain_scale), int(h * contain_scale)),
            interpolation=cv2.INTER_AREA,
        )
        fh, fw = fg.shape[:2]
        # Place the sharp copy so the subject stays in frame.
        subj_x = int(cx_frac * fw)
        subj_y = int(cy_frac * fh)
        left = min(max(int(target_width * cx_frac) - subj_x, 0), target_width - fw)
        top = min(max(int(target_height * cy_frac) - subj_y, 0), target_height - fh)
        bg[top : top + fh, left : left + fw] = fg
        log("  Framing: aspect mismatch too large, blurred fill", "INFO")
        return bg

    # Cover crop centered on the subject, clamped to the image bounds.
    if img_ratio > target_ratio:
        crop_h = h
        crop_w = int(h * target_ratio)
    else:
        crop_w = w
        crop_h = int(w / target_ratio)
    center_x = int(cx_frac * w)
    center_y = int(cy_frac * h)
    left = min(max(center_x - crop_w // 2, 0), w - crop_w)
    top = min(max(center_y - crop_h // 2, 0), h - crop_h)
    crop = img_rgb[top : top + crop_h, left : left + crop_w]
    return cv2.resize(crop, (target_width, target_height), interpolation=cv2.INTER_AREA)


def frame_subject_file(
    image_path: Path, output_path: Path, target_width: int, target_height: int
) -> None:
    image_path = Path(image_path)
    try:
        import cv2  # noqa: F401
        import numpy  # noqa: F401
    except ImportError:
        # Old behaviour: plain center crop.
        _center_crop(image_path, output_path, target_width, target_height)
        return
    try:
        with Image.open(image_path) as img:
            rgb = np.array(img.convert("RGB"))
        framed = frame_subject(rgb, target_width, target_height)
        Image.fromarray(framed).save(output_path, "JPEG", quality=85)
    except Exception as e:
        log(f"  Subject framing failed ({e}); center-cropping", "WARNING")
        _center_crop(image_path, output_path, target_width, target_height)


def _center_crop(
    image_path: Path, output_path: Path, target_width: int, target_height: int
) -> None:
    img = Image.open(image_path).convert("RGB")
    target_ratio = target_width / target_height
    img_ratio = img.width / img.height
    if img_ratio > target_ratio:
        new_height = target_height
        new_width = int(new_height * img_ratio)
    else:
        new_width = target_width
        new_height = int(new_width / img_ratio)
    img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    left = (new_width - target_width) // 2
    top = (new_height - target_height) // 2
    img = img.crop((left, top, left + target_width, top + target_height))
    img.save(output_path, "JPEG", quality=85)
    img.close()
