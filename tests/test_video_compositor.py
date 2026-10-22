"""
Tests for VideoCompositor — easing, overlay animation, and composition helpers.
"""

from unittest.mock import MagicMock, Mock, patch

import numpy as np

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
        mock_clip.with_position.return_value = mock_clip

        result = self.compositor._apply_overlay_animation(mock_clip, 3.0)

        assert result is mock_clip
        mock_clip.with_effects.assert_not_called()
        mock_clip.transform.assert_called_once()
        mock_clip.with_position.assert_called_once_with(("center", "center"))

    def test_zero_duration(self):
        mock_clip = MagicMock()
        mock_clip.size = (1080, 1920)
        mock_clip.with_effects.return_value = mock_clip
        mock_clip.transform.return_value = mock_clip
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
        mock_clip.with_position = Mock(return_value=mock_clip)

        self.compositor._apply_overlay_animation(mock_clip, 1.0)

        transform_call = mock_clip.transform.call_args[0][0]

        def dummy_get_frame(t):
            return np.ones((100, 100, 3), dtype=np.uint8) * 255

        at_mid = transform_call(dummy_get_frame, 0.5)
        at_end = transform_call(dummy_get_frame, 1.0)

        assert np.any(at_mid > 0)
        assert np.all(at_end == 0)


class TestVerticalCropBox:
    """Crop math must stay inside the source bounds (smear regression)."""

    def test_wide_source_crops_sides_not_height(self):
        # 640x338 (wider than 9:16): the old code cropped height to 360,
        # beyond the 338px frame, yielding an 11-row strip that stretched
        # into vertical streaks.
        box = VideoCompositor._vertical_crop_box(640, 338)
        assert box is not None
        x1, y1, cw, ch = box
        assert ch == 338  # height untouched
        assert cw == int(338 * (9 / 16))
        assert x1 >= 0 and x1 + cw <= 640
        assert y1 == 0

    def test_tall_source_crops_height_within_bounds(self):
        box = VideoCompositor._vertical_crop_box(360, 900)
        assert box is not None
        x1, y1, cw, ch = box
        assert cw == 360
        assert ch == int(360 * (16 / 9))
        assert y1 >= 0 and y1 + ch <= 900
        assert x1 == 0

    def test_exact_vertical_returns_none(self):
        assert VideoCompositor._vertical_crop_box(720, 1280) is None
        assert VideoCompositor._vertical_crop_box(1080, 1920) is None

    def test_box_always_within_bounds(self):
        for w, h in [
            (640, 338),
            (3840, 2160),
            (360, 900),
            (1000, 1000),
            (1919, 1080),
            (50, 4000),
        ]:
            box = VideoCompositor._vertical_crop_box(w, h)
            if box is None:
                continue
            x1, y1, cw, ch = box
            assert x1 >= 0 and y1 >= 0
            assert x1 + cw <= w
            assert y1 + ch <= h
