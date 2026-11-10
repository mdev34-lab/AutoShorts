"""Regression test for the images-only black-tail bug (issue #31).

``_create_flux_video`` used to give every scene a fixed duration
(``SCENE_DURATION_SECONDS``), so the composed clip -- N scenes overlapped by
``CROSSFADE_TIME`` -- always ended before the audio and the duration-mismatch
branch padded the remainder with black frames (17 scenes over 53.4 s of
audio left ~10.4 s of black tail).  The fix redistributes the audio duration
across the scenes so the composed clip ends exactly with the audio.

The test drives the real ``ExplainerGenerator._create_flux_video`` with real
moviepy composition (ImageClips, CrossFadeIn/FadeIn effects and
``concatenate_videoclips(..., method="compose", padding=-CROSSFADE_TIME)``).
Only the I/O boundaries are monkeypatched: image file loading (synthetic
solid-color frames at brightness 200), audio file loading (a silent
AudioClip), the subtitle system, the opening animation and the final render
(``write_videofile``), so no real files, network or ffmpeg rendering are
needed.

The per-scene zoom/opacity animation (``_apply_overlay_animation``) is
stubbed to identity on purpose: it fades every scene out over its final 15 %,
which darkens the last ~0.5 s of *any* composition regardless of the
duration fix and would mask the black-tail bug under test here.
"""

import numpy as np
import pytest
from moviepy import AudioClip, CompositeVideoClip, ImageClip

import autoshorts.generators.explainer as explainer
from autoshorts.generators.explainer import ExplainerGenerator

BRIGHTNESS = 200

# (num_images, audio_duration): the three real black-tail runs plus one
# over-covered case where num_images * SCENE_DURATION_SECONDS already
# exceeds the audio (pre-fix, the composed clip then ran past the audio and
# was subclipped by the mismatch branch instead of padded).
FLUX_CASES = [
    (17, 53.4),
    (14, 44.6),
    (17, 51.5),
    (19, 44.6),
]


class _SolidImageClip(ImageClip):
    """Real ImageClip over a solid-color array.

    ``close()`` is a no-op so the composition stays inspectable after
    ``_create_flux_video`` closes its clips during cleanup.
    """

    def close(self):
        pass


class _FakeSubtitleSystem:
    def generate_subtitles(self, paragraphs, duration, out_dir):
        return f"{out_dir}/subtitles.vtt"

    def render_subtitles(self, vtt_path, size):
        return []


class _FakeVideoCompositor:
    def _apply_opening_animation(self, clip, duration):
        return clip


def _make_silent_audio(duration):
    return AudioClip(
        frame_function=lambda t: np.zeros((1, 2)),
        duration=duration,
        fps=44100,
    )


def _make_solid_image_clip(path):
    return _SolidImageClip(np.full((480, 270, 3), BRIGHTNESS, dtype=np.uint8))


def _compose_flux_video(monkeypatch, num_images, audio_duration):
    """Run the real _create_flux_video and return the composed final clip."""
    recorded = []

    class _RecordingComposite(CompositeVideoClip):
        def write_videofile(self, filename, *args, **kwargs):
            recorded.append(self)

        def close(self):
            pass

    monkeypatch.setattr(explainer, "CompositeVideoClip", _RecordingComposite)
    monkeypatch.setattr(explainer, "ImageClip", _make_solid_image_clip)
    monkeypatch.setattr(
        explainer, "AudioFileClip", lambda path: _make_silent_audio(audio_duration)
    )
    monkeypatch.setattr(explainer, "SubtitleSystem", _FakeSubtitleSystem)
    monkeypatch.setattr(explainer, "VideoCompositor", _FakeVideoCompositor)
    monkeypatch.setattr(explainer, "OPENING_ANIMATION_ENABLED", True)

    generator = ExplainerGenerator.__new__(ExplainerGenerator)

    def _identity_overlay_animation(clip, duration):
        return clip

    generator._apply_overlay_animation = _identity_overlay_animation

    generator._create_flux_video(
        [f"scene_{i:02d}.png" for i in range(num_images)],
        f"audio_{audio_duration}s.wav",
        ["first paragraph", "second paragraph", "third paragraph"],
        "output.mp4",
        None,
    )

    assert len(recorded) == 1
    return recorded[0]


@pytest.mark.parametrize(("num_images", "audio_duration"), FLUX_CASES)
def test_flux_video_duration_matches_audio(monkeypatch, num_images, audio_duration):
    final = _compose_flux_video(monkeypatch, num_images, audio_duration)

    assert abs(final.duration - audio_duration) < 0.05, (
        f"composed {final.duration}s vs audio {audio_duration}s ({num_images} images)"
    )

    # Sample strictly inside the composed content, never at t == duration:
    # pre-fix the gap between the composed content and the audio was padded
    # with black frames, so this tail frame was black.
    tail = final.get_frame(audio_duration - 0.05)
    assert tail.mean() > 50, (
        f"tail frame at t={audio_duration - 0.05}s is near-black "
        f"(mean {tail.mean():.1f})"
    )


def test_flux_video_with_no_images_skips_render(monkeypatch):
    recorded = []

    class _RecordingComposite(CompositeVideoClip):
        def write_videofile(self, filename, *args, **kwargs):
            recorded.append(self)

    monkeypatch.setattr(explainer, "CompositeVideoClip", _RecordingComposite)

    generator = ExplainerGenerator.__new__(ExplainerGenerator)

    with pytest.raises(ValueError):
        generator._create_flux_video(
            [],
            "audio.wav",
            ["first paragraph", "second paragraph", "third paragraph"],
            "output.mp4",
            None,
        )

    assert recorded == []
