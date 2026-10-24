"""A missing configured font must still give a real, size-honoring font."""

from autoshorts.modules import subtitle_system as ss


def test_missing_font_still_honors_requested_size():
    small = ss._load_pil_font("/nonexistent/none.ttf", 20)
    big = ss._load_pil_font("/nonexistent/none.ttf", 80)
    w_small = small.getbbox("Test")[2]
    w_big = big.getbbox("Test")[2]
    assert w_big > 2 * w_small  # a 10px bitmap font would give equal widths


def test_text_measurement_tracks_font_size_when_font_missing():
    a = ss._get_text_size_pil("Subtitle", "/nonexistent/none.ttf", 30)
    b = ss._get_text_size_pil("Subtitle", "/nonexistent/none.ttf", 90)
    assert b[0] > 2 * a[0]


def test_bundled_font_is_package_data():
    from autoshorts.modules.config import DEFAULT_FONT, PROJECT_FONTS_DIR

    assert PROJECT_FONTS_DIR.is_dir()
    assert (PROJECT_FONTS_DIR / "BebasNeue-Regular.ttf").is_file()
    assert "BebasNeue" in DEFAULT_FONT


def test_no_font_anywhere_uses_scalable_builtin_not_bitmap(monkeypatch):
    monkeypatch.setattr("autoshorts.modules.config.FALLBACK_FONTS", [])
    small = ss._load_pil_font("/nonexistent/none.ttf", 20)
    big = ss._load_pil_font("/nonexistent/none.ttf", 80)
    assert big.getbbox("Test")[2] > 2 * small.getbbox("Test")[2]


def test_built_wheel_contains_the_bundled_font(tmp_path):
    """Build the real wheel and inspect it; a source-tree path check cannot
    prove the font ships."""
    import subprocess
    import sys
    import zipfile
    from pathlib import Path

    import pytest

    root = Path(__file__).resolve().parent.parent
    try:
        import setuptools

        if tuple(int(x) for x in setuptools.__version__.split(".")[:1]) < (61,):
            pytest.skip("setuptools < 61 cannot read pyproject metadata")
    except ImportError:
        pytest.skip("setuptools not installed")
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            str(root),
            "--no-deps",
            "--no-build-isolation",
            "-w",
            str(tmp_path),
            "-q",
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, proc.stderr[-800:]
    wheel = next(tmp_path.glob("autoshorts-*.whl"))
    names = zipfile.ZipFile(wheel).namelist()
    assert "autoshorts/fonts/BebasNeue-Regular.ttf" in names, names


def test_scalable_fallback_guard_for_old_pillow(monkeypatch):
    monkeypatch.setattr("autoshorts.modules.config.FALLBACK_FONTS", [])
    real = ss.ImageFont.load_default

    def old_pillow(size=None):
        if size is not None:
            raise TypeError("load_default() takes no arguments")
        return real()

    monkeypatch.setattr(ss.ImageFont, "load_default", old_pillow)
    monkeypatch.setattr(ss, "_WARNED_FONTS", set())
    assert ss._load_pil_font("/nonexistent/none.ttf", 40) is not None


def _ink_width(frame):
    import numpy as np

    mask = frame.max(axis=2) > 40
    cols = np.where(mask.any(axis=0))[0]
    return 0 if len(cols) == 0 else int(cols[-1] - cols[0] + 1)


def test_real_compositor_path_missing_font_scales_with_resolution(tmp_path):
    """Drive SubtitleSystem.generate_subtitles -> render_subtitles ->
    CompositeVideoClip frames with a missing configured font. A 10px bitmap
    fallback gives the same ink width at every resolution; a scalable font
    must grow with the frame."""
    import numpy as np
    from moviepy import ColorClip, CompositeVideoClip

    widths = {}
    for size in ((720, 1280), (1080, 1920)):
        (tmp_path / f"s{size[0]}").mkdir()
        system = ss.SubtitleSystem(font_path="/nonexistent/none.ttf")
        vtt = system.generate_subtitles(
            ["Um texto de legenda com tamanho razoavel para o teste."],
            4.0,
            str(tmp_path / f"s{size[0]}"),
        )
        clips = system.render_subtitles(vtt, size)
        assert clips
        bg = ColorClip(size, color=(0, 0, 0)).with_duration(4.0)
        frame = CompositeVideoClip([bg, *clips], size=size).get_frame(1.0)
        assert isinstance(frame, np.ndarray)
        widths[size[0]] = _ink_width(frame)
    assert widths[720] > 0.25 * 720, widths  # legible, not a 10px strip
    assert widths[1080] > 1.3 * widths[720], widths
    assert widths[1080] < 0.95 * 1080, widths  # still inside the frame
