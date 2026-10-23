"""Regression tests for jumpcut background coverage (issue #14).

The composed timeline follows the narration audio, so a background clip
shorter than the target duration leaves black frames at the end of the
video. These tests pin the invariant: the jumpcut background always
covers the full target duration (minus the small trim buffer).
"""

import math

import pytest
from moviepy import ColorClip

from autoshorts.modules import video_compositor as vc

TRIM_BUFFER = 0.05


def _source(duration: float) -> ColorClip:
    return ColorClip(size=(64, 64), color=(30, 120, 30), duration=duration)


@pytest.fixture
def compositor():
    return vc.VideoCompositor()


def test_short_source_loops_to_cover_target(compositor, monkeypatch):
    """Source shorter than target (after intro skip) must still cover it."""
    monkeypatch.setattr(vc, "BG_SKIP_INTRO", 25.0)
    monkeypatch.setattr(vc, "JUMPCUT_SEG_DUR", 4.0)
    target = 10.0
    # available = 12 - min(25, 3.6) = 8.4s < target -> loop path
    result = compositor._jumpcut_background(_source(12.0), target)
    assert result.duration >= target - TRIM_BUFFER
    assert result.audio is None


def test_fractional_target_is_ceiled_not_floored(compositor, monkeypatch):
    """int() flooring leaves up to seg_dur seconds uncovered; ceil must not."""
    monkeypatch.setattr(vc, "BG_SKIP_INTRO", 25.0)
    monkeypatch.setattr(vc, "JUMPCUT_SEG_DUR", 4.0)
    target = 74.6
    result = compositor._jumpcut_background(_source(167.4), target)
    # int(74.6 / 4.0) == 18 segments -> 72.0s (black tail); ceil -> 19 -> 76s
    assert result.duration >= target - TRIM_BUFFER
    assert result.duration <= target


def test_exact_multiple_target_stays_covered(compositor, monkeypatch):
    monkeypatch.setattr(vc, "BG_SKIP_INTRO", 25.0)
    monkeypatch.setattr(vc, "JUMPCUT_SEG_DUR", 4.0)
    target = 12.0
    result = compositor._jumpcut_background(_source(100.0), target)
    assert result.duration >= target - TRIM_BUFFER


def test_sampled_segments_cover_at_least_target(compositor, monkeypatch):
    """Coverage invariant across several fractional targets."""
    monkeypatch.setattr(vc, "BG_SKIP_INTRO", 25.0)
    monkeypatch.setattr(vc, "JUMPCUT_SEG_DUR", 4.0)
    for target in (10.1, 33.3, 74.6, 90.0):
        result = compositor._jumpcut_background(_source(167.4), target)
        expected_segs = max(1, math.ceil(target / 4.0))
        assert result.duration >= min(
            target - TRIM_BUFFER, expected_segs * 4.0 - TRIM_BUFFER
        )
