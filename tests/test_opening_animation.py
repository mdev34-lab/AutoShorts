"""
Tests for the cold-open S-curve zoom-out with motion blur
(VideoCompositor._apply_opening_animation).
"""

from unittest.mock import MagicMock, Mock

import numpy as np

from autoshorts.modules.video_compositor import VideoCompositor


class TestEaseOut:
    def setup_method(self):
        self.compositor = VideoCompositor()

    def test_boundaries(self):
        assert self.compositor._ease_out(0.0) == 0.0
        assert self.compositor._ease_out(1.0) == 1.0

    def test_clamps_negative(self):
        assert self.compositor._ease_out(-0.5) == 0.0

    def test_clamps_above_one(self):
        assert self.compositor._ease_out(1.5) == 1.0

    def test_monotonic(self):
        prev = -1.0
        for i in range(101):
            val = self.compositor._ease_out(i / 100.0)
            assert val >= prev
            prev = val

    def test_default_power_is_cubic(self):
        assert abs(self.compositor._ease_out(0.5) - 0.875) < 1e-10

    def test_linear_power(self):
        assert self.compositor._ease_out(0.5, power=1.0) == 0.5

    def test_quadratic_power(self):
        assert abs(self.compositor._ease_out(0.5, power=2.0) - 0.75) < 1e-10

    def test_higher_power_decelerates_more(self):
        cubic = self.compositor._ease_out(0.1, power=3.0)
        quintic = self.compositor._ease_out(0.1, power=5.0)
        assert cubic < quintic


class TestApplyOpeningAnimation:
    def setup_method(self):
        self.compositor = VideoCompositor()

    def _make_mock_clip(self):
        mock_clip = MagicMock()
        mock_clip.size = (1080, 1920)
        mock_clip.with_effects.return_value = mock_clip
        mock_clip.transform.return_value = mock_clip
        mock_clip.with_position.return_value = mock_clip
        return mock_clip

    def test_returns_clip_with_effects_transform_and_position(self):
        mock_clip = self._make_mock_clip()

        result = self.compositor._apply_opening_animation(mock_clip, 1.0)

        assert result is mock_clip
        mock_clip.with_effects.assert_called_once()
        mock_clip.transform.assert_called_once()
        mock_clip.with_position.assert_called_once_with(("center", "center"))

    def test_zero_duration_is_noop(self):
        mock_clip = self._make_mock_clip()

        result = self.compositor._apply_opening_animation(mock_clip, 0.0)

        assert result is mock_clip
        mock_clip.with_effects.assert_not_called()
        mock_clip.transform.assert_not_called()
        mock_clip.with_position.assert_not_called()

    def test_negative_duration_is_noop(self):
        mock_clip = self._make_mock_clip()

        result = self.compositor._apply_opening_animation(mock_clip, -0.5)

        assert result is mock_clip
        mock_clip.with_effects.assert_not_called()


class TestOpeningAnimationMotionBlur:
    def setup_method(self):
        self.compositor = VideoCompositor()

    def _capture_transform_callback(self, duration=1.0):
        mock_clip = Mock()
        mock_clip.size = (100, 100)
        mock_clip.with_effects = Mock(return_value=mock_clip)
        mock_clip.transform = Mock(return_value=mock_clip)
        mock_clip.with_position = Mock(return_value=mock_clip)
        self.compositor._apply_opening_animation(mock_clip, duration)
        return mock_clip.transform.call_args[0][0]

    def test_constant_frame_passes_through(self):
        callback = self._capture_transform_callback()

        def constant_frame(t):
            return np.full((100, 100, 3), 128, dtype=np.uint8)

        out = callback(constant_frame, 0.5)
        assert out.shape == (100, 100, 3)
        assert np.all(out == 128)

    def test_blur_averages_subframes(self):
        """When get_frame varies inside the shutter window, output is the mean."""
        callback = self._capture_transform_callback(duration=2.0)

        sample_log = []

        def get_frame(t):
            sample_log.append(round(t, 6))
            return np.full((100, 100, 3), int(t * 100) % 256, dtype=np.uint8)

        callback(get_frame, 1.0)

        assert len(sample_log) >= 2
        assert sample_log[0] == 1.0
        for s in sample_log[1:]:
            assert s < 1.0
            assert s >= 0.0

    def test_blur_output_is_uint8(self):
        callback = self._capture_transform_callback()

        def get_frame(t):
            return np.full((100, 100, 3), 200, dtype=np.uint8)

        out = callback(get_frame, 0.5)
        assert out.dtype == np.uint8
        assert np.all(out == 200)
