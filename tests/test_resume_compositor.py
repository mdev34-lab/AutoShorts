"""Offline unit tests for ExplainerGenerator.compose_saved_inputs compositor slice.

Tests cover:
1. Exact dispatch: arguments forwarded accurately to _compose_flux_body with class overlay.
2. Asset validation: fail-closed validation of nonempty regular image/audio files,
   paragraphs list of non-empty strings, and output path.
3. Constructor traps: generator constructor, ScriptGenerator, and TTSSystem
   are never invoked.
4. Legacy samebody and instance override: instance wrapper _create_flux_video
   forwards self._apply_overlay_animation to shared _compose_flux_body, preserving
   instance overrides without duplicating compositor body or formulas.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from autoshorts.generators import explainer
from autoshorts.generators.explainer import ExplainerGenerator
from autoshorts.modules import VideoMetadata


def _create_dummy_assets(tmp_path: Path):
    """Helper to create dummy regular nonempty image and audio files."""
    img1 = tmp_path / "img1.png"
    img1.write_bytes(b"\x89PNG\r\n\x1a\nfakeimagebytes1")

    img2 = tmp_path / "img2.jpg"
    img2.write_bytes(b"\xff\xd8\xfffakeimagebytes2")

    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"RIFF\x00\x00\x00\x00WAVEfmt fakeaudiobytes")

    out_file = tmp_path / "output.mp4"
    paragraphs = ["First paragraph of voiceover.", "Second paragraph of voiceover."]

    return [img1, img2], audio, paragraphs, out_file


# ─────────────────────────────────────────────────────────────────────────────
# 1. Exact Dispatch Tests
# ─────────────────────────────────────────────────────────────────────────────


def test_compose_saved_inputs_exact_dispatch_with_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Verify compose_saved_inputs dispatches exactly to cls._compose_flux_body with metadata."""
    images, audio, paragraphs, out_file = _create_dummy_assets(tmp_path)
    metadata = VideoMetadata(title="Test Short", description="Test Description")

    dispatched_args = {}

    def _mock_compose_flux_body(
        cls,
        img_paths,
        audio_path,
        script_paragraphs,
        output_path,
        metadata=None,
        overlay_animation=None,
    ):
        dispatched_args["cls"] = cls
        dispatched_args["img_paths"] = img_paths
        dispatched_args["audio_path"] = audio_path
        dispatched_args["paragraphs"] = script_paragraphs
        dispatched_args["output_path"] = output_path
        dispatched_args["metadata"] = metadata
        dispatched_args["overlay_animation"] = overlay_animation
        return "sentinel_flux_result"

    monkeypatch.setattr(
        ExplainerGenerator, "_compose_flux_body", classmethod(_mock_compose_flux_body)
    )

    result = ExplainerGenerator.compose_saved_inputs(
        images,
        audio,
        paragraphs,
        out_file,
        metadata=metadata,
    )

    assert result == "sentinel_flux_result"
    assert dispatched_args["cls"] is ExplainerGenerator
    assert dispatched_args["img_paths"] == [str(images[0]), str(images[1])]
    assert dispatched_args["audio_path"] == str(audio)
    assert dispatched_args["paragraphs"] == paragraphs
    assert dispatched_args["output_path"] == str(out_file)
    assert dispatched_args["metadata"] is metadata
    assert (
        dispatched_args["overlay_animation"].__func__
        is ExplainerGenerator._apply_overlay_animation.__func__
    )
    assert dispatched_args["overlay_animation"].__self__ is ExplainerGenerator


def test_compose_saved_inputs_exact_dispatch_without_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Verify compose_saved_inputs dispatches with metadata=None when omitted."""
    images, audio, paragraphs, out_file = _create_dummy_assets(tmp_path)

    dispatched = {}

    def _mock_compose_flux_body(
        cls,
        img_paths,
        audio_path,
        script_paragraphs,
        output_path,
        metadata=None,
        overlay_animation=None,
    ):
        dispatched["metadata"] = metadata
        dispatched["overlay_animation"] = overlay_animation
        return True

    monkeypatch.setattr(
        ExplainerGenerator, "_compose_flux_body", classmethod(_mock_compose_flux_body)
    )

    result = ExplainerGenerator.compose_saved_inputs(
        [str(images[0]), str(images[1])],
        str(audio),
        paragraphs,
        str(out_file),
    )

    assert result is True
    assert dispatched["metadata"] is None
    assert (
        dispatched["overlay_animation"].__func__
        is ExplainerGenerator._apply_overlay_animation.__func__
    )
    assert dispatched["overlay_animation"].__self__ is ExplainerGenerator


def test_compose_saved_inputs_accepts_tuple_and_path_objects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Verify compose_saved_inputs accepts a tuple of Path objects."""
    images, audio, paragraphs, out_file = _create_dummy_assets(tmp_path)

    captured_paths = []

    def _mock_compose_flux_body(
        cls,
        img_paths,
        audio_path,
        script_paragraphs,
        output_path,
        metadata=None,
        overlay_animation=None,
    ):
        captured_paths.extend(img_paths)
        return "ok"

    monkeypatch.setattr(
        ExplainerGenerator, "_compose_flux_body", classmethod(_mock_compose_flux_body)
    )

    ExplainerGenerator.compose_saved_inputs(
        tuple(images),
        audio,
        paragraphs,
        out_file,
    )

    assert captured_paths == [str(images[0]), str(images[1])]


# ─────────────────────────────────────────────────────────────────────────────
# 2. Asset Validation Tests
# ─────────────────────────────────────────────────────────────────────────────


def test_validation_empty_image_paths(tmp_path: Path):
    _, audio, paragraphs, out_file = _create_dummy_assets(tmp_path)
    with pytest.raises(
        ValueError, match="image_paths must be a non-empty list or tuple"
    ):
        ExplainerGenerator.compose_saved_inputs([], audio, paragraphs, out_file)


def test_validation_non_sequence_image_paths(tmp_path: Path):
    _, audio, paragraphs, out_file = _create_dummy_assets(tmp_path)
    with pytest.raises(
        ValueError, match="image_paths must be a non-empty list or tuple"
    ):
        ExplainerGenerator.compose_saved_inputs(
            "not_a_sequence", audio, paragraphs, out_file
        )  # type: ignore


def test_validation_missing_image_file(tmp_path: Path):
    _, audio, paragraphs, out_file = _create_dummy_assets(tmp_path)
    missing = tmp_path / "non_existent.png"
    with pytest.raises(FileNotFoundError, match="Image file does not exist"):
        ExplainerGenerator.compose_saved_inputs([missing], audio, paragraphs, out_file)


def test_validation_directory_as_image_path(tmp_path: Path):
    _, audio, paragraphs, out_file = _create_dummy_assets(tmp_path)
    img_dir = tmp_path / "img_dir"
    img_dir.mkdir()
    with pytest.raises(ValueError, match="Image path is not a regular file"):
        ExplainerGenerator.compose_saved_inputs([img_dir], audio, paragraphs, out_file)


def test_validation_zero_byte_image_file(tmp_path: Path):
    _, audio, paragraphs, out_file = _create_dummy_assets(tmp_path)
    empty_img = tmp_path / "empty.png"
    empty_img.touch()
    with pytest.raises(ValueError, match="Image file is empty"):
        ExplainerGenerator.compose_saved_inputs(
            [empty_img], audio, paragraphs, out_file
        )


def test_validation_invalid_image_path_item(tmp_path: Path):
    _, audio, paragraphs, out_file = _create_dummy_assets(tmp_path)
    with pytest.raises(ValueError, match="Image path at index 0 cannot be None"):
        ExplainerGenerator.compose_saved_inputs([None], audio, paragraphs, out_file)  # type: ignore

    with pytest.raises(
        ValueError, match="Image path at index 0 cannot be empty or whitespace"
    ):
        ExplainerGenerator.compose_saved_inputs(["   "], audio, paragraphs, out_file)


def test_validation_missing_audio_file(tmp_path: Path):
    images, _, paragraphs, out_file = _create_dummy_assets(tmp_path)
    missing_audio = tmp_path / "missing_audio.wav"
    with pytest.raises(FileNotFoundError, match="Audio file does not exist"):
        ExplainerGenerator.compose_saved_inputs(
            images, missing_audio, paragraphs, out_file
        )


def test_validation_directory_as_audio_path(tmp_path: Path):
    images, _, paragraphs, out_file = _create_dummy_assets(tmp_path)
    audio_dir = tmp_path / "audio_dir"
    audio_dir.mkdir()
    with pytest.raises(ValueError, match="Audio path is not a regular file"):
        ExplainerGenerator.compose_saved_inputs(images, audio_dir, paragraphs, out_file)


def test_validation_zero_byte_audio_file(tmp_path: Path):
    images, _, paragraphs, out_file = _create_dummy_assets(tmp_path)
    empty_audio = tmp_path / "empty.wav"
    empty_audio.touch()
    with pytest.raises(ValueError, match="Audio file is empty"):
        ExplainerGenerator.compose_saved_inputs(
            images, empty_audio, paragraphs, out_file
        )


def test_validation_empty_audio_path(tmp_path: Path):
    images, _, paragraphs, out_file = _create_dummy_assets(tmp_path)
    with pytest.raises(ValueError, match="audio_path must be specified"):
        ExplainerGenerator.compose_saved_inputs(images, "", paragraphs, out_file)

    with pytest.raises(ValueError, match="audio_path cannot be empty or whitespace"):
        ExplainerGenerator.compose_saved_inputs(images, "   ", paragraphs, out_file)


def test_validation_empty_paragraphs(tmp_path: Path):
    images, audio, _, out_file = _create_dummy_assets(tmp_path)
    with pytest.raises(
        ValueError, match="paragraphs must be a non-empty list of strings"
    ):
        ExplainerGenerator.compose_saved_inputs(images, audio, [], out_file)


def test_validation_non_list_paragraphs(tmp_path: Path):
    images, audio, _, out_file = _create_dummy_assets(tmp_path)
    with pytest.raises(
        ValueError, match="paragraphs must be a non-empty list of strings"
    ):
        ExplainerGenerator.compose_saved_inputs(
            images, audio, ("tuple", "para"), out_file
        )  # type: ignore

    with pytest.raises(
        ValueError, match="paragraphs must be a non-empty list of strings"
    ):
        ExplainerGenerator.compose_saved_inputs(
            images, audio, "single string", out_file
        )  # type: ignore


def test_validation_paragraphs_containing_empty_or_non_string(tmp_path: Path):
    images, audio, _, out_file = _create_dummy_assets(tmp_path)
    with pytest.raises(
        ValueError, match="Paragraph at index 1 must be a non-empty string"
    ):
        ExplainerGenerator.compose_saved_inputs(images, audio, ["valid", ""], out_file)

    with pytest.raises(
        ValueError, match="Paragraph at index 0 must be a non-empty string"
    ):
        ExplainerGenerator.compose_saved_inputs(images, audio, ["   "], out_file)

    with pytest.raises(
        ValueError, match="Paragraph at index 0 must be a non-empty string"
    ):
        ExplainerGenerator.compose_saved_inputs(images, audio, [123], out_file)  # type: ignore


def test_validation_empty_output_path(tmp_path: Path):
    images, audio, paragraphs, _ = _create_dummy_assets(tmp_path)
    with pytest.raises(ValueError, match="output_path must be specified"):
        ExplainerGenerator.compose_saved_inputs(images, audio, paragraphs, "")

    with pytest.raises(ValueError, match="output_path cannot be empty or whitespace"):
        ExplainerGenerator.compose_saved_inputs(images, audio, paragraphs, "   ")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Constructor Trap Tests
# ─────────────────────────────────────────────────────────────────────────────


def test_constructortraps_generator_constructor_never_called(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Ensure ExplainerGenerator.__init__ is never called during compose_saved_inputs."""
    images, audio, paragraphs, out_file = _create_dummy_assets(tmp_path)

    def _trap_init(self, *args, **kwargs):
        raise AssertionError(
            "ExplainerGenerator.__init__ was called, but must be bypassed!"
        )

    monkeypatch.setattr(ExplainerGenerator, "__init__", _trap_init)
    monkeypatch.setattr(
        ExplainerGenerator,
        "_compose_flux_body",
        classmethod(lambda cls, *args, **kwargs: "ok"),
    )

    result = ExplainerGenerator.compose_saved_inputs(
        images, audio, paragraphs, out_file
    )
    assert result == "ok"


def test_constructortraps_script_generator_never_called(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Ensure ScriptGenerator is never instantiated during compose_saved_inputs."""
    images, audio, paragraphs, out_file = _create_dummy_assets(tmp_path)

    def _trap_script_gen(*args, **kwargs):
        raise AssertionError("ScriptGenerator was instantiated, but must be bypassed!")

    monkeypatch.setattr(explainer, "ScriptGenerator", _trap_script_gen)
    monkeypatch.setattr(
        ExplainerGenerator,
        "_compose_flux_body",
        classmethod(lambda cls, *args, **kwargs: "ok"),
    )

    result = ExplainerGenerator.compose_saved_inputs(
        images, audio, paragraphs, out_file
    )
    assert result == "ok"


def test_constructortraps_tts_system_never_called(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Ensure TTSSystem is never instantiated during compose_saved_inputs."""
    images, audio, paragraphs, out_file = _create_dummy_assets(tmp_path)

    def _trap_tts_system(*args, **kwargs):
        raise AssertionError("TTSSystem was instantiated, but must be bypassed!")

    monkeypatch.setattr(explainer, "TTSSystem", _trap_tts_system)
    monkeypatch.setattr(
        ExplainerGenerator,
        "_compose_flux_body",
        classmethod(lambda cls, *args, **kwargs: "ok"),
    )

    result = ExplainerGenerator.compose_saved_inputs(
        images, audio, paragraphs, out_file
    )
    assert result == "ok"


# ─────────────────────────────────────────────────────────────────────────────
# 4. Legacy Samebody & Instance Override Tests
# ─────────────────────────────────────────────────────────────────────────────


def test_legacy_create_flux_video_forwards_to_shared_body(
    monkeypatch: pytest.MonkeyPatch,
):
    """Verify instance method _create_flux_video forwards self._apply_overlay_animation to _compose_flux_body."""
    instance = ExplainerGenerator.__new__(ExplainerGenerator)

    called_args = {}

    def _mock_shared_body(
        cls,
        img_paths,
        audio_path,
        paragraphs,
        output_path,
        metadata=None,
        overlay_animation=None,
    ):
        called_args["cls"] = cls
        called_args["img_paths"] = img_paths
        called_args["audio_path"] = audio_path
        called_args["paragraphs"] = paragraphs
        called_args["output_path"] = output_path
        called_args["metadata"] = metadata
        called_args["overlay_animation"] = overlay_animation
        return "shared_body_result"

    monkeypatch.setattr(
        ExplainerGenerator, "_compose_flux_body", classmethod(_mock_shared_body)
    )

    res = instance._create_flux_video(["img1.png"], "audio.wav", ["para"], "out.mp4")
    assert res == "shared_body_result"
    assert called_args["cls"] is ExplainerGenerator
    assert called_args["img_paths"] == ["img1.png"]
    assert called_args["audio_path"] == "audio.wav"
    assert called_args["paragraphs"] == ["para"]
    assert called_args["output_path"] == "out.mp4"
    assert called_args["overlay_animation"] == instance._apply_overlay_animation


def test_legacy_instance_override_overlay_animation_preserved(
    monkeypatch: pytest.MonkeyPatch,
):
    """Verify legacy instance overriding generator._apply_overlay_animation is preserved and forwarded."""
    instance = ExplainerGenerator.__new__(ExplainerGenerator)

    def _custom_identity_overlay(clip, duration):
        return clip

    instance._apply_overlay_animation = _custom_identity_overlay

    called_args = {}

    def _mock_shared_body(
        cls,
        img_paths,
        audio_path,
        paragraphs,
        output_path,
        metadata=None,
        overlay_animation=None,
    ):
        called_args["overlay_animation"] = overlay_animation
        return "custom_forwarded"

    monkeypatch.setattr(
        ExplainerGenerator, "_compose_flux_body", classmethod(_mock_shared_body)
    )

    res = instance._create_flux_video(["img1.png"], "audio.wav", ["para"], "out.mp4")
    assert res == "custom_forwarded"
    assert called_args["overlay_animation"] is _custom_identity_overlay


def test_legacy_easing_staticmethod_matches_formulas():
    """Verify _ease_in_out_cubic is a staticmethod and produces exact cubic easing curve."""
    # Boundary conditions
    assert ExplainerGenerator._ease_in_out_cubic(-0.5) == 0.0
    assert ExplainerGenerator._ease_in_out_cubic(0.0) == 0.0
    assert ExplainerGenerator._ease_in_out_cubic(1.0) == 1.0
    assert ExplainerGenerator._ease_in_out_cubic(1.5) == 1.0

    # Intermediate points:
    # for t < 0.5: 4 * t^3
    assert pytest.approx(ExplainerGenerator._ease_in_out_cubic(0.25)) == 4 * (
        0.25**3
    )  # 0.0625
    # for t = 0.5: 1 - pow(-2*0.5 + 2, 3) / 2 = 1 - 1/2 = 0.5
    assert pytest.approx(ExplainerGenerator._ease_in_out_cubic(0.5)) == 0.5
    # for t > 0.5: 1 - pow(-2*0.75 + 2, 3) / 2 = 1 - pow(0.5, 3) / 2 = 1 - 0.125 / 2 = 0.9375
    assert pytest.approx(ExplainerGenerator._ease_in_out_cubic(0.75)) == 0.9375

    # Instance call matches class call
    instance = ExplainerGenerator.__new__(ExplainerGenerator)
    assert instance._ease_in_out_cubic(0.25) == ExplainerGenerator._ease_in_out_cubic(
        0.25
    )
    assert instance._ease_in_out_cubic(0.75) == ExplainerGenerator._ease_in_out_cubic(
        0.75
    )


def test_legacy_apply_overlay_animation_is_classmethod():
    """Verify _apply_overlay_animation is a classmethod callable on both class and instance."""
    instance = ExplainerGenerator.__new__(ExplainerGenerator)
    assert (
        ExplainerGenerator._apply_overlay_animation.__func__
        is instance._apply_overlay_animation.__func__
    )
