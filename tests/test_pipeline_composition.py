"""Text removal must run BEFORE subject framing in resize_to_fill."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

from autoshorts.modules import image_searcher
from autoshorts.modules.image_searcher import ImageSearcher
from autoshorts.modules.text_removal import _dependencies_available


def test_text_removal_runs_before_subject_framing(tmp_path, monkeypatch):
    monkeypatch.setattr(image_searcher, "IMAGE_TEXT_REMOVAL", True)
    monkeypatch.setattr(image_searcher, "SUBJECT_FRAMING", True)
    calls = []

    def fake_remove(src, dst):
        calls.append(("remove", src.name, dst.name))
        dst.write_bytes(b"clean")

    def fake_frame(src, dst, w, h):
        calls.append(("frame", src.name, dst.name))
        dst.write_bytes(b"framed")

    monkeypatch.setattr(
        "autoshorts.modules.text_removal.remove_overlay_text", fake_remove
    )
    monkeypatch.setattr(
        "autoshorts.modules.image_framing.frame_subject_file", fake_frame
    )

    src = tmp_path / "photo.jpg"
    src.write_bytes(b"raw")
    out = tmp_path / "out.jpg"
    ImageSearcher().resize_to_fill(src, out)

    assert [c[0] for c in calls] == ["remove", "frame"]
    # framing reads the cleaned copy, not the original
    assert calls[1][1].endswith("_notext.jpg")
    # sidecar is cleaned up
    assert not (tmp_path / "photo_notext.jpg").exists()
    assert out.read_bytes() == b"framed"


def test_framing_alone_when_removal_disabled(tmp_path, monkeypatch):
    monkeypatch.setattr(image_searcher, "IMAGE_TEXT_REMOVAL", False)
    monkeypatch.setattr(image_searcher, "SUBJECT_FRAMING", True)
    calls = []
    monkeypatch.setattr(
        "autoshorts.modules.image_framing.frame_subject_file",
        lambda src, dst, w, h: (calls.append(src.name), dst.write_bytes(b"f")),
    )
    src = tmp_path / "photo.jpg"
    src.write_bytes(b"raw")
    ImageSearcher().resize_to_fill(src, tmp_path / "out.jpg")
    assert calls == ["photo.jpg"]


# Real-pipeline composition: synthetic photo + overlay text + off-center
# subject through the actual removal and framing code (no mocks).

FONT = str(
    Path(__file__).parent.parent
    / "src"
    / "autoshorts"
    / "fonts"
    / "BebasNeue-Regular.ttf"
)

SUBJECT_COLOR = (225, 25, 25)
SUBJECT_FRAC = (0.38, 0.72)  # off-center: left-of-center, lower half


def _photo_with_text_and_subject(tmp_path: Path, name: str = "combo.jpg") -> Path:
    """1000x1200 photo-like image: muted gradient, a saturated subject blob
    at SUBJECT_FRAC, and an OCR-readable white text strip near the top."""
    # Flat background: gradients put low-frequency structure into the FFT
    # saliency map and pull the centroid off the subject. The subject is a
    # striped disc: a flat disc is salient only at its edge, stripes give it
    # a strong interior signature so the centroid lands on its center.
    base = np.full((1200, 1000, 3), (125, 130, 135), dtype=np.uint8)
    img = Image.fromarray(base, "RGB")
    draw = ImageDraw.Draw(img)
    cx, cy = int(SUBJECT_FRAC[0] * 1000), int(SUBJECT_FRAC[1] * 1200)
    for i, rr in enumerate(range(100, 0, -12)):
        color = SUBJECT_COLOR if i % 2 == 0 else (120, 15, 15)
        draw.ellipse((cx - rr, cy - rr, cx + rr, cy + rr), fill=color)
    font = ImageFont.truetype(FONT, 100)
    # Two words with a 60px gap: separate same-line candidate boxes, so the
    # removal pipeline's line-neighbor guard accepts both (a single merged
    # box is discarded as an OCR hallucination).
    draw.text((70, 90), "50%", fill=(255, 255, 255), font=font)
    draw.text(
        (70 + draw.textlength("50%", font=font) + 60, 90),
        "OFF",
        fill=(255, 255, 255),
        font=font,
    )
    path = tmp_path / name
    img.save(path, "JPEG", quality=95)
    return path


def _assert_no_confident_text(image_path: Path) -> None:
    """No confident multi-char token in either polarity. Single characters
    are ignored: inpaint texture and JPEG edges make tesseract hallucinate
    isolated one-letter reads, while a surviving overlay word ('50%', 'OFF')
    is always multi-char. Verified to fail on the unremoved source."""
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
            if len(str(t).strip()) >= 2 and float(c) >= 60
        ]


def _subject_stats(image_path: Path):
    arr = np.array(Image.open(image_path).convert("RGB"))
    mask = (
        (arr[:, :, 0].astype(int) > 170)
        & (arr[:, :, 1].astype(int) < 110)
        & (arr[:, :, 2].astype(int) < 110)
    )
    ys, xs = np.nonzero(mask)
    return len(xs), (float(xs.mean()) / arr.shape[1] if len(xs) else None)


@pytest.mark.skipif(
    not _dependencies_available(), reason="cv2/pytesseract/tesseract missing"
)
def test_real_pipeline_removes_text_and_frames_subject(tmp_path, monkeypatch):
    """Combined hook, real code: text is erased AND the off-center subject
    is preserved and pulled toward the frame center."""
    monkeypatch.setattr(image_searcher, "IMAGE_TEXT_REMOVAL", True)
    monkeypatch.setattr(image_searcher, "SUBJECT_FRAMING", True)
    src = _photo_with_text_and_subject(tmp_path)
    out = tmp_path / "out.jpg"

    ImageSearcher(target_width=720, target_height=1280).resize_to_fill(src, out)

    result = Image.open(out)
    assert result.size == (720, 1280)
    _assert_no_confident_text(out)  # removal really happened
    count, cx_frac = _subject_stats(out)
    assert count > 0.01 * 720 * 1280  # subject preserved, not cropped away
    # Source subject sits at x=0.38; the subject-centered cover crop must
    # move it to the horizontal center of the output.
    assert cx_frac is not None and abs(cx_frac - 0.5) < 0.08, cx_frac
    # sidecar cleaned up
    assert not (tmp_path / "combo_notext.jpg").exists()


@pytest.mark.skipif(
    not _dependencies_available(), reason="cv2/pytesseract/tesseract missing"
)
def test_real_pipeline_removal_on_framing_off(tmp_path, monkeypatch):
    """Inverse toggle: removal ON + framing OFF uses the classic center-crop
    path but still erases the overlay text."""
    monkeypatch.setattr(image_searcher, "IMAGE_TEXT_REMOVAL", True)
    monkeypatch.setattr(image_searcher, "SUBJECT_FRAMING", False)
    src = _photo_with_text_and_subject(tmp_path)
    out = tmp_path / "out.jpg"

    ImageSearcher(target_width=720, target_height=1280).resize_to_fill(src, out)

    assert Image.open(out).size == (720, 1280)
    _assert_no_confident_text(out)
    # Classic center-crop path keeps the source's horizontal layout: the
    # subject (x=0.38 in the source) lands at ~0.32 (resize 1000->1066 px
    # then crop 173 px from the left), NOT reframed to ~0.5.
    count, cx_frac = _subject_stats(out)
    assert count > 0.01 * 720 * 1280  # subject preserved
    assert cx_frac is not None and abs(cx_frac - 0.323) < 0.05, cx_frac
    assert not (tmp_path / "combo_notext.jpg").exists()
