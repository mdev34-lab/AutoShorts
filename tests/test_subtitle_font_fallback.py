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
