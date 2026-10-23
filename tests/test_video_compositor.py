"""
Tests for VideoCompositor — easing, overlay animation, and composition helpers.
"""

from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

from autoshorts.modules.video_compositor import VideoCompositor


class _CompletedMock:
    returncode = 0
    stdout = b""
    stderr = b""


class TestEaseInOutCubic:
    def setup_method(self):
        self.compositor = VideoCompositor()

    def test_boundaries(self):
        assert self.compositor._ease_in_out_cubic(0.0) == 0.0
        assert self.compositor._ease_in_out_cubic(1.0) == 1.0

    def test_midpoint(self):
        assert self.compositor._ease_in_out_cubic(0.5) == 0.5

    def test_symmetric(self):
        for t in [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9]:
            val = self.compositor._ease_in_out_cubic(t)
            inv = self.compositor._ease_in_out_cubic(1 - t)
            assert abs(val - (1 - inv)) < 1e-10

    def test_monotonic(self):
        prev = -1.0
        for i in range(101):
            val = self.compositor._ease_in_out_cubic(i / 100.0)
            assert val >= prev
            prev = val

    def test_clamps_negative(self):
        assert self.compositor._ease_in_out_cubic(-0.5) == 0.0

    def test_clamps_above_one(self):
        assert self.compositor._ease_in_out_cubic(1.5) == 1.0


class TestApplyOverlayAnimation:
    def setup_method(self):
        self.compositor = VideoCompositor()

    def test_returns_clip_with_transform_and_position(self):
        mock_clip = MagicMock()
        mock_clip.size = (1080, 1920)
        mock_clip.with_effects.return_value = mock_clip
        mock_clip.transform.return_value = mock_clip
        mock_clip.with_mask.return_value = mock_clip
        mock_clip.with_position.return_value = mock_clip

        result = self.compositor._apply_overlay_animation(mock_clip, 3.0)

        assert result is mock_clip
        mock_clip.with_effects.assert_not_called()
        mock_clip.transform.assert_called_once()
        # Fade is carried by an alpha mask, not by darkening pixels.
        assert mock_clip.with_mask.call_args[0][0].is_mask
        mock_clip.with_position.assert_called_once_with(("center", "center"))

    def test_zero_duration(self):
        mock_clip = MagicMock()
        mock_clip.size = (1080, 1920)
        mock_clip.with_effects.return_value = mock_clip
        mock_clip.transform.return_value = mock_clip
        mock_clip.with_mask.return_value = mock_clip
        mock_clip.with_position.return_value = mock_clip

        result = self.compositor._apply_overlay_animation(mock_clip, 0.0)
        assert result is mock_clip


class TestCreateBlurredBackground:
    def setup_method(self):
        self.compositor = VideoCompositor()

    @patch("autoshorts.modules.video_compositor.subprocess.run")
    def test_fast_blur_calls_ffmpeg(self, mock_run):
        mock_run.return_value = _CompletedMock()

        result = self.compositor._apply_fast_blur("in.mp4", "out.mp4", radius=4)
        assert result == "out.mp4"
        args = mock_run.call_args[0][0]
        assert "boxblur" in " ".join(args)

    @patch("autoshorts.modules.video_compositor.subprocess.run")
    def test_fast_blur_default_radius(self, mock_run):
        mock_run.return_value = _CompletedMock()

        result = self.compositor._apply_fast_blur("in.mp4", "out.mp4")
        assert result == "out.mp4"


class TestOverlayAnimationFrameTransform:
    def setup_method(self):
        self.compositor = VideoCompositor()

    def test_opacity_transform_midpoint_visible(self):
        mock_clip = Mock()
        mock_clip.size = (100, 100)
        mock_clip.with_effects = Mock(return_value=mock_clip)
        mock_clip.transform = Mock(return_value=mock_clip)
        mock_clip.with_mask = Mock(return_value=mock_clip)
        mock_clip.with_position = Mock(return_value=mock_clip)

        self.compositor._apply_overlay_animation(mock_clip, 1.0)

        transform_call = mock_clip.transform.call_args[0][0]

        def dummy_get_frame(t):
            return np.ones((100, 100, 3), dtype=np.uint8) * 255

        # The RGB transform never darkens for opacity; the fade lives in the mask.
        at_mid = transform_call(dummy_get_frame, 0.5)
        at_end = transform_call(dummy_get_frame, 1.0)

        assert np.all(at_mid == 255)
        assert np.all(at_end == 255)

        mask = mock_clip.with_mask.call_args[0][0]
        assert mask.get_frame(0.5).max() == 1.0
        assert mask.get_frame(1.0).max() == 0.0
        assert 0.0 < mask.get_frame(0.05).max() < 1.0


class TestVerticalCropBox:
    """Crop math stays inside source bounds and preserves the historical
    simple-mode contract (smear regression, #10)."""

    def test_wide_source_crops_sides_to_16_9(self):
        # 640x338 (wider than 16:9): the old code cropped height to 360,
        # beyond the 338px frame, yielding an 11-row strip that stretched
        # into vertical streaks.
        box = VideoCompositor._vertical_crop_box(640, 338)
        assert box is not None
        x1, y1, cw, ch = box
        assert ch == 338  # height untouched
        assert cw == round(338 * 16 / 9)  # cropped to 16:9, inside the frame
        assert x1 >= 0 and x1 + cw <= 640
        assert y1 == 0

    def test_tall_source_crops_height_to_9_16(self):
        box = VideoCompositor._vertical_crop_box(360, 900)
        assert box is not None
        x1, y1, cw, ch = box
        assert cw == 360
        assert ch == round(360 * 16 / 9)
        assert y1 >= 0 and y1 + ch <= 900
        assert x1 == 0

    def test_standard_sources_not_cropped(self):
        # Historical contract: sources between 9:16 and 16:9 (incl. 1920x1080
        # and 4:3) are resized full-frame, never cropped.
        assert VideoCompositor._vertical_crop_box(1920, 1080) is None
        assert VideoCompositor._vertical_crop_box(3840, 2160) is None
        assert VideoCompositor._vertical_crop_box(720, 1280) is None
        assert VideoCompositor._vertical_crop_box(1080, 1920) is None
        assert VideoCompositor._vertical_crop_box(1000, 1000) is None
        assert VideoCompositor._vertical_crop_box(1024, 768) is None

    def test_tiny_and_extreme_inputs_positive_in_bounds(self):
        for w, h in [
            (1, 1),
            (2, 2),
            (1, 100),
            (100, 1),
            (4000, 2),
            (2, 4000),
            (1, 4000),
            (4000, 1),
        ]:
            box = VideoCompositor._vertical_crop_box(w, h)
            if box is None:
                continue
            x1, y1, cw, ch = box
            assert cw > 0 and ch > 0, (w, h, box)
            assert x1 >= 0 and y1 >= 0
            assert x1 + cw <= w
            assert y1 + ch <= h

    def test_invalid_dimensions_return_none(self):
        assert VideoCompositor._vertical_crop_box(0, 100) is None
        assert VideoCompositor._vertical_crop_box(100, 0) is None
        assert VideoCompositor._vertical_crop_box(-5, 100) is None

    def test_box_always_within_bounds(self):
        for w, h in [
            (640, 338),
            (360, 900),
            (1919, 1079),
            (50, 4000),
            (4000, 50),
        ]:
            box = VideoCompositor._vertical_crop_box(w, h)
            if box is None:
                continue
            x1, y1, cw, ch = box
            assert x1 >= 0 and y1 >= 0
            assert x1 + cw <= w
            assert y1 + ch <= h


class TestFitFgSize:
    """Overlay foreground band: aspect preserved, fitted inside the frame,
    always positive and even."""

    def test_landscape_source_fills_width(self):
        # 640x338 -> 720x380 (the original smear-case source)
        assert VideoCompositor._fit_fg_size(640, 338) == (720, 380)

    def test_tall_source_fits_height_not_overflow(self):
        # 360x900 would round to 720x1800, overflowing the 1280px frame;
        # it must fit inside instead.
        fw, fh = VideoCompositor._fit_fg_size(360, 900)
        assert fh <= 1280 and fw <= 720
        assert fh == 1280
        assert abs(fw / fh - 360 / 900) < 0.01

    def test_extreme_aspect_never_zero(self):
        fw, fh = VideoCompositor._fit_fg_size(4000, 2)
        assert fw > 0 and fh > 0
        assert fw % 2 == 0 and fh % 2 == 0
        fw2, fh2 = VideoCompositor._fit_fg_size(2, 4000)
        assert fw2 > 0 and fh2 > 0
        assert fw2 <= 720 and fh2 <= 1280

    def test_always_even_and_within_frame(self):
        for w, h in [
            (640, 338),
            (1920, 1080),
            (360, 900),
            (1, 1),
            (4000, 2),
            (2, 4000),
            (1919, 1079),
        ]:
            fw, fh = VideoCompositor._fit_fg_size(w, h)
            assert 2 <= fw <= 720 and 2 <= fh <= 1280
            assert fw % 2 == 0 and fh % 2 == 0

    def test_invalid_dimensions_default(self):
        assert VideoCompositor._fit_fg_size(0, 0) == (720, 1280)


class _SentinelError(Exception):
    def __init__(self, size):
        super().__init__(size)
        self.size = size


def _make_two_zone_source(tmp_path):
    """640x338 @10fps, 1s: top half red, bottom half blue - the exact source
    shape of the original smear bug, with zones distinct enough that a
    stretched 11-row strip is detectable in the output."""
    import subprocess

    from PIL import Image

    frame = tmp_path / "frame.png"
    img = Image.new("RGB", (640, 338))
    img.paste((255, 0, 0), (0, 0, 640, 169))
    img.paste((0, 0, 255), (0, 169, 640, 338))
    img.save(frame)
    video = tmp_path / "source.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loop",
            "1",
            "-framerate",
            "10",
            "-i",
            str(frame),
            "-t",
            "1",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(video),
        ],
        check=True,
        capture_output=True,
    )
    audio = tmp_path / "audio.wav"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=1",
            str(audio),
        ],
        check=True,
        capture_output=True,
    )
    return str(video), str(audio)


def _frame_pixel(path, x, y, seek="0.5"):
    import subprocess

    from PIL import Image

    frame_path = path + f"_{x}_{y}.png"
    subprocess.run(
        ["ffmpeg", "-y", "-ss", seek, "-i", path, "-frames:v", "1", frame_path],
        check=True,
        capture_output=True,
    )
    return Image.open(frame_path).convert("RGB").getpixel((x, y))


class TestRealPathComposition:
    """Short real renders through create_output_video (smear regression #10)."""

    def test_overlay_render_not_smeared(self, tmp_path, monkeypatch):
        monkeypatch.setattr("autoshorts.modules.video_compositor.BG_MODE", "speed")
        video, audio = _make_two_zone_source(tmp_path)
        out = str(tmp_path / "out.mp4")
        comp = VideoCompositor()
        ok = comp.create_output_video(video, audio, None, out, 1.0, use_blurred_bg=True)
        assert ok
        # Output frame: band is 720x380 centered (y 450..830).
        top = _frame_pixel(out, 360, 470)  # band, upper zone
        bottom = _frame_pixel(out, 360, 810)  # band, lower zone
        bg = _frame_pixel(out, 360, 150)  # blurred + darkened background
        # A smeared render would stretch an 11-row strip of one zone across
        # the whole band; the fixed pipeline keeps both zones distinct.
        assert top[0] > 180 and top[2] < 90, f"band top not red: {top}"
        assert bottom[2] > 180 and bottom[0] < 90, f"band bottom not blue: {bottom}"
        # Background is the darkened blur, clearly dimmer than the band.
        assert bg[0] < top[0] - 40, f"bg not darkened: {bg} vs {top}"

    def test_simple_mode_crops_overlay_keeps_source(self, tmp_path, monkeypatch):
        video, audio = _make_two_zone_source(tmp_path)
        out = str(tmp_path / "out.mp4")

        def fake_simple(self, video, audio, vtt_path, target_duration):
            raise _SentinelError(tuple(video.size))

        def fake_overlay(
            self, video, audio, vtt_path, target_duration, image_paths=None
        ):
            raise _SentinelError(tuple(video.size))

        monkeypatch.setattr(VideoCompositor, "_create_simple_mode", fake_simple)
        monkeypatch.setattr(VideoCompositor, "_create_with_overlay_mode", fake_overlay)

        comp = VideoCompositor()
        with pytest.raises(_SentinelError) as simple_exc:
            comp.create_output_video(video, audio, None, out, 1.0, use_blurred_bg=False)
        # Simple mode normalizes full-frame: in-bounds crop, then resize.
        assert simple_exc.value.size == (720, 1280)

        with pytest.raises(_SentinelError) as overlay_exc:
            comp.create_output_video(video, audio, None, out, 1.0, use_blurred_bg=True)
        # Overlay mode must receive the untouched source (full source also
        # feeds the blurred-background path).
        assert overlay_exc.value.size == (640, 338)


def test_overlay_fade_uses_alpha_not_black():
    """The overlay fade must reveal the background, never dim the frame to black."""
    import numpy as np
    from moviepy import CompositeVideoClip, ImageClip

    from autoshorts.modules.video_compositor import VideoCompositor

    compositor = VideoCompositor()
    red_bg = ImageClip(np.full((36, 64, 3), (200, 0, 0), dtype="uint8")).with_duration(
        1.0
    )
    white_fg = ImageClip(np.full((36, 64, 3), 255, dtype="uint8")).with_duration(1.0)
    overlay = compositor._apply_overlay_animation(white_fg, 1.0)
    composite = CompositeVideoClip([red_bg, overlay], size=(64, 36))

    # t=0: fade-in start -> background fully visible, not a black frame
    px_start = composite.get_frame(0.0)[18, 32]
    assert px_start[0] > 150 and px_start[1] < 60, f"fade start went dark: {px_start}"
    # t=0.5: fully opaque overlay
    px_mid = composite.get_frame(0.5)[18, 32]
    assert all(c > 240 for c in px_mid), f"overlay not opaque mid-show: {px_mid}"
    # t=0.075 (mid fade-in, opacity ~0.5): a blend of bg and overlay, never black
    px_blend = composite.get_frame(0.075)[18, 32]
    assert px_blend[1] > 60, f"mid-fade crushed to black: {px_blend}"


def test_overlay_last_encoded_frame_fully_transparent():
    """The final ENCODED frame (t = duration - 1/fps), not t=duration, must
    show pure background - no ghost overlay from an incomplete fade."""
    import numpy as np
    from moviepy import CompositeVideoClip, ImageClip

    from autoshorts.modules.video_compositor import VideoCompositor

    fps, duration = 30, 1.0
    compositor = VideoCompositor()
    red_bg = ImageClip(np.full((36, 64, 3), (200, 0, 0), dtype="uint8")).with_duration(
        duration
    )
    white_fg = ImageClip(np.full((36, 64, 3), 255, dtype="uint8")).with_duration(
        duration
    )
    overlay = compositor._apply_overlay_animation(white_fg, duration)
    composite = CompositeVideoClip([red_bg, overlay], size=(64, 36))

    last_t = (int(duration * fps) - 1) / fps  # frame 29/30: what is actually encoded
    px_last = composite.get_frame(last_t)[18, 32]
    assert px_last[0] > 190 and px_last[1] < 20 and px_last[2] < 20, (
        f"ghost overlay on final encoded frame: {px_last}"
    )


def test_overlay_fade_combines_with_source_alpha():
    """A transparent PNG's own alpha must survive: the fade mask multiplies
    it, never replaces it (a replacement resurrects transparent pixels)."""
    import numpy as np
    from moviepy import CompositeVideoClip, ImageClip

    from autoshorts.modules.video_compositor import VideoCompositor

    rgba = np.full((36, 64, 4), (255, 255, 255, 255), dtype="uint8")
    rgba[:, :32, 3] = 0  # left half fully transparent
    compositor = VideoCompositor()
    red_bg = ImageClip(np.full((36, 64, 3), (200, 0, 0), dtype="uint8")).with_duration(
        1.0
    )
    fg = ImageClip(rgba).with_duration(1.0)
    assert fg.mask is not None, "RGBA source should carry its alpha as a mask"
    overlay = compositor._apply_overlay_animation(fg, 1.0)
    composite = CompositeVideoClip([red_bg, overlay], size=(64, 36))

    # Mid-show (fully faded in): left half still shows the background...
    px_left = composite.get_frame(0.5)[18, 8]
    assert px_left[0] > 150 and px_left[1] < 60, (
        f"source alpha was replaced, transparent pixels resurrected: {px_left}"
    )
    # ...right half shows the opaque overlay
    px_right = composite.get_frame(0.5)[18, 48]
    assert all(c > 240 for c in px_right), f"opaque region lost: {px_right}"


def test_overlay_animation_duration_clamped_to_clip():
    """An animation longer than the clip must not strand the clip at full
    opacity past its own end - the fade completes within the clip's life."""
    import numpy as np
    from moviepy import ImageClip

    from autoshorts.modules.video_compositor import VideoCompositor

    compositor = VideoCompositor()
    clip = ImageClip(np.full((36, 64, 3), 255, dtype="uint8")).with_duration(1.0)
    result = compositor._apply_overlay_animation(clip, 10.0)
    assert abs(result.mask.duration - 1.0) < 1e-6, (
        f"mask runs past the clip: {result.mask.duration}"
    )
    # final encoded frame of the CLIP's life is transparent
    last_t = 29 / 30
    fade = result.mask.get_frame(last_t).max()
    assert fade == 0.0, f"clip would end visible: alpha {fade}"


def test_overlay_source_alpha_is_zoomed_with_the_pixels():
    """The source alpha must be zoomed exactly like the RGB frame: with an
    off-center transparent boundary, no pixel from the transparent (green)
    region may show up and the opaque region must start where its pixels do."""
    from moviepy import CompositeVideoClip, ImageClip

    from autoshorts.modules import video_compositor as vc

    w, h, boundary = 400, 72, 60
    rgba = np.zeros((h, w, 4), dtype="uint8")
    rgba[:, :boundary] = (0, 255, 0, 0)  # transparent, green RGB
    rgba[:, boundary:] = (0, 0, 255, 255)  # opaque blue
    compositor = vc.VideoCompositor()
    red_bg = ImageClip(np.full((h, w, 3), (200, 0, 0), dtype="uint8")).with_duration(
        1.0
    )
    fg = ImageClip(rgba).with_duration(1.0)
    overlay = compositor._apply_overlay_animation(fg, 1.0)
    frame = CompositeVideoClip([red_bg, overlay], size=(w, h)).get_frame(0.5)

    scale = vc.MAX_ZOOM_FACTOR
    assert scale > 1.01
    edge = int(round(w / 2 + (boundary - w / 2) * scale))
    row = frame[h // 2]
    for x in range(0, w):
        if x < edge - 4:
            assert row[x][0] > 150 and row[x][1] < 60, f"x={x} not background: {row[x]}"
        elif x > edge + 4:
            assert row[x][2] > 200 and row[x][1] < 60, f"x={x} not overlay: {row[x]}"


def test_short_clip_last_encoded_frame_transparent_at_any_fps():
    from moviepy import CompositeVideoClip, ImageClip

    from autoshorts.modules import video_compositor as vc

    compositor = vc.VideoCompositor()
    for fps in (24, 30):
        for duration in (0.5, 0.8, 2.0):
            red_bg = ImageClip(
                np.full((36, 64, 3), (200, 0, 0), dtype="uint8")
            ).with_duration(duration)
            white = ImageClip(np.full((36, 64, 3), 255, dtype="uint8")).with_duration(
                duration
            )
            overlay = compositor._apply_overlay_animation(white, duration)
            comp = CompositeVideoClip([red_bg, overlay], size=(64, 36))
            last_t = (int(round(duration * fps)) - 1) / fps
            px = comp.get_frame(last_t)[18, 32]
            assert px[0] > 195 and px[1] < 6 and px[2] < 6, (
                f"ghost at fps={fps} dur={duration}: {px}"
            )
