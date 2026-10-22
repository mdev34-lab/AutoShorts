"""Text removal must run BEFORE subject framing in resize_to_fill."""

from autoshorts.modules import image_searcher
from autoshorts.modules.image_searcher import ImageSearcher


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
