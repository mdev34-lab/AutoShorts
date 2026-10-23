#!/usr/bin/env python3
"""
Video Compositor Module

Provides unified video compositing with:
- Blurred background (using FFmpeg boxblur)
- AI image overlays with smooth easing animation
- Subtitle integration
- Standardized rendering pipeline

DRY principle: reused by both no-images and images modes.
"""

import subprocess
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
from moviepy import (
    AudioFileClip,
    CompositeVideoClip,
    ImageClip,
    VideoFileClip,
    concatenate_videoclips,
)
from moviepy.video.fx import FadeIn, MultiplyColor
from moviepy.video.io.ffmpeg_reader import FFMPEG_VideoReader

from .config import (
    AUDIO_CODEC,
    BG_MODE,
    BG_SKIP_INTRO,
    BLUR_RADIUS,
    ENCODING_CRF,
    ENCODING_PRESET,
    ENCODING_THREADS,
    IMAGE_BOUNCE_INTERVAL,
    IMAGE_OVERLAY_DURATION,
    JUMPCUT_SEG_DUR,
    MAX_ZOOM_FACTOR,
    OPENING_ANIMATION_DURATION,
    OPENING_ANIMATION_ENABLED,
    OPENING_BLUR_SAMPLES,
    OPENING_EASING_POWER,
    OPENING_SHUTTER_FRACTION,
    OPENING_ZOOM_START,
    START_WITH_IMAGE,
    VIDEO_CODEC,
    VIDEO_FPS,
    VIDEO_HEIGHT,
    VIDEO_WIDTH,
)
from .metadata import VideoMetadata
from .subtitle_system import SubtitleSystem
from .utils import create_temp_dir, get_video_duration, log

# Monkey-patch FFMPEG_VideoReader.close() to suppress WinError 6 during GC
_orig_close = FFMPEG_VideoReader.close


def _patched_close(self, delete_lastread=True):
    try:
        _orig_close(self, delete_lastread)
    except OSError:
        warnings.warn(
            "FFMPEG_VideoReader: ignored OSError during close (handle already cleaned up)",
            ResourceWarning,
            stacklevel=2,
        )


FFMPEG_VideoReader.close = _patched_close


class VideoCompositor:
    """
    Unified video compositor for AutoShorts.

    Provides common compositing operations:
    - Blurred background (yt_summarizer style)
    - Image overlays with smooth animation (experimental style)
    - Combined pipeline with subtitles
    """

    def __init__(self):
        self.temp_dir = create_temp_dir()
        self.subtitle_system = SubtitleSystem()

    def _apply_fast_blur(
        self, input_path: str, output_path: str, radius: int = BLUR_RADIUS
    ) -> str:
        """Apply blur using FFmpeg boxblur - much faster than PIL per-frame."""
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            input_path,
            "-vf",
            f"boxblur={radius}:{radius}",
            "-c:v",
            "libx264",
            "-crf",
            str(ENCODING_CRF),
            "-c:a",
            "copy",
            output_path,
        ]
        subprocess.run(cmd, capture_output=True, check=True)
        return output_path

    def _ease_in_out_cubic(self, t: float) -> float:
        """Smooth cubic ease-in-out: slow start, fast middle, slow end."""
        if t <= 0:
            return 0.0
        if t >= 1:
            return 1.0
        if t < 0.5:
            return 4 * t * t * t
        return 1 - pow(-2 * t + 2, 3) / 2

    def _ease_out(self, t: float, power: float = OPENING_EASING_POWER) -> float:
        """Ease-out (decelerating) curve: fast start, slow end.

        Uses 1 - (1 - t) ** power. power=3 is cubic, power=5 is quintic.
        """
        if t <= 0:
            return 0.0
        if t >= 1:
            return 1.0
        return 1.0 - pow(1.0 - t, power)

    def _apply_overlay_animation(self, clip, duration: float):
        """Smooth zoom-in-then-zoom-out + opacity + motion blur on overlay clips.

        Zoom is center-crop+resize (constant frame size), blur is a per-frame
        PIL Gaussian whose sigma tracks the ease curve's slope (velocity).
        Opacity is applied as a multiply in the same pass.
        """

        def _ease_slope(p: float) -> float:
            """Normalized derivative [0,1] of ease_in_out_cubic at progress p ∈ [0,1]."""
            if p <= 0.0:
                return 0.0
            if p >= 1.0:
                return 0.0
            if p < 0.5:
                return 4.0 * p * p
            return 4.0 * (1.0 - p) * (1.0 - p)

        def scale_anim(t: float) -> float:
            if duration <= 0:
                return 1.0
            progress = t / duration
            fade_in_end = 0.15
            fade_out_start = 0.85

            if progress < fade_in_end:
                return 1.0 + (MAX_ZOOM_FACTOR - 1.0) * self._ease_in_out_cubic(
                    progress / fade_in_end
                )
            elif progress < fade_out_start:
                return MAX_ZOOM_FACTOR
            else:
                out_progress = (progress - fade_out_start) / (1.0 - fade_out_start)
                return MAX_ZOOM_FACTOR - (
                    MAX_ZOOM_FACTOR - 1.0
                ) * self._ease_in_out_cubic(out_progress)

        def opacity_anim(t: float) -> float:
            if duration <= 0:
                return 1.0
            progress = t / duration
            fade_in_end = 0.15
            fade_out_start = 0.85

            if progress < fade_in_end:
                return self._ease_in_out_cubic(progress / fade_in_end)
            elif progress < fade_out_start:
                return 1.0
            else:
                out_progress = (progress - fade_out_start) / (1.0 - fade_out_start)
                return max(1.0 - self._ease_in_out_cubic(out_progress), 0.0)

        try:
            from PIL import Image, ImageFilter

            _pil = True
        except ImportError:
            _pil = False

        samples = max(1, int(OPENING_BLUR_SAMPLES))
        shutter = float(OPENING_SHUTTER_FRACTION)
        max_sigma = max(0.0, float(samples) * shutter)

        def apply_overlay(get_frame, t):
            frame = get_frame(t)
            scale = scale_anim(t)
            opacity = opacity_anim(t)

            # Zoom: center-crop + resize (constant frame size)
            if scale > 1.001 and _pil:
                h, w = frame.shape[:2]
                crop_w = max(1, int(round(w / scale)))
                crop_h = max(1, int(round(h / scale)))
                x1 = (w - crop_w) // 2
                y1 = (h - crop_h) // 2
                cropped = frame[y1 : y1 + crop_h, x1 : x1 + crop_w]
                frame = np.array(
                    Image.fromarray(cropped).resize((w, h), Image.Resampling.LANCZOS)
                )

            # Motion blur: velocity-driven Gaussian
            if _pil and max_sigma > 0:
                progress = t / duration if duration > 0 else 0.0
                fade_in_end = 0.15
                fade_out_start = 0.85
                if progress < fade_in_end:
                    velocity = _ease_slope(progress / fade_in_end)
                elif progress < fade_out_start:
                    velocity = 0.0
                else:
                    out_p = (progress - fade_out_start) / (1.0 - fade_out_start)
                    velocity = _ease_slope(out_p)
                sigma = velocity * max_sigma
                if sigma >= 0.1:
                    frame = np.array(
                        Image.fromarray(frame).filter(
                            ImageFilter.GaussianBlur(radius=sigma)
                        )
                    )

            # Opacity
            if opacity < 1.0 and opacity >= 0.0:
                frame = np.minimum(255, frame * opacity).astype("uint8")

            return frame

        return clip.transform(apply_overlay).with_position(("center", "center"))

    def _apply_opening_animation(self, clip, duration: float = OPENING_ANIMATION_DURATION):
        """Cold-open: ease-out zoom-out + per-frame motion-blur-style blur.

        Starts at OPENING_ZOOM_START (e.g. 1.25) and decelerates to 1.0
        over ``duration`` seconds.

        The zoom is implemented as a center-crop + upscale in a single
        ``transform`` callback, NOT via ``Resize(scale_anim)``. Animated
        Resize produces frames of variable size which the writer then
        letterboxes back to canvas size — inverting the visual (you'd see
        zoom-out from 100% to <100% with black bezels). Keeping the
        transform's output at canvas size avoids that entirely.

        Blur is a per-frame PIL Gaussian whose sigma tracks the zoom
        velocity (high at start, ~0 at the end) so it reads as
        decelerating motion blur.

        The legacy ``OPENING_BLUR_SAMPLES`` and ``OPENING_SHUTTER_FRACTION``
        env vars are still honored as a combined strength multiplier.
        """
        if duration <= 0:
            return clip
        zoom_start = OPENING_ZOOM_START
        samples = max(1, int(OPENING_BLUR_SAMPLES))
        shutter = float(OPENING_SHUTTER_FRACTION)
        max_sigma = max(0.0, float(samples) * shutter)

        try:
            from PIL import Image, ImageFilter

            _pil_available = True
        except ImportError:
            _pil_available = False

        def apply_zoom_and_blur(get_frame, t: float):
            frame = get_frame(t)
            progress = max(0.0, min(1.0, t / duration))
            eased = self._ease_out(progress)
            scale = zoom_start - (zoom_start - 1.0) * eased

            # Zoom: center-crop a (1/scale) fraction, then resize back to
            # canvas size. Output frame stays at canvas size, so the
            # writer never letterboxes (the root cause of the bezels).
            if scale > 1.001 and _pil_available:
                h, w = frame.shape[:2]
                crop_w = max(1, int(round(w / scale)))
                crop_h = max(1, int(round(h / scale)))
                x1 = (w - crop_w) // 2
                y1 = (h - crop_h) // 2
                cropped = frame[y1 : y1 + crop_h, x1 : x1 + crop_w]
                frame = np.array(
                    Image.fromarray(cropped).resize((w, h), Image.Resampling.LANCZOS)
                )

            # Motion blur: velocity-driven Gaussian, decays to 0 at end.
            if _pil_available and max_sigma > 0:
                velocity = 1.0 - eased
                sigma = velocity * max_sigma
                if sigma >= 0.1:
                    frame = np.array(
                        Image.fromarray(frame).filter(
                            ImageFilter.GaussianBlur(radius=sigma)
                        )
                    )

            return frame

        return clip.transform(apply_zoom_and_blur).with_position(("center", "center"))

    def _ensure_duration(
        self, clip: VideoFileClip, min_duration: float
    ) -> VideoFileClip:
        if clip.duration <= 0 or clip.duration >= min_duration:
            return clip
        loops = int(min_duration / clip.duration) + 1
        return concatenate_videoclips([clip] * loops).subclipped(0, min_duration)

    def _jumpcut_background(self, clip, target_duration: float) -> VideoFileClip:
        """Create background by randomly sampling segments instead of speed scaling."""
        import math
        import random

        from moviepy import concatenate_videoclips

        jumpcut_start = time.time()
        seg_dur = JUMPCUT_SEG_DUR
        source_dur = clip.duration

        if source_dur <= 0:
            raise ValueError(
                "Jumpcut background requires a source clip with positive duration."
            )
        if target_duration <= 0:
            raise ValueError("Jumpcut background requires a positive target duration.")

        # Skip intro section (default 25s) to avoid logos/intros
        start_offset = min(BG_SKIP_INTRO, source_dur * 0.3)
        available_dur = source_dur - start_offset

        if available_dur <= 0:
            start_offset = 0
            available_dur = source_dur

        if available_dur <= target_duration:
            # Source too short: loop the available range to cover the target
            n_loops = math.ceil(target_duration / available_dur)
            segments = [
                clip.subclipped(start_offset, source_dur) for _ in range(n_loops)
            ]
            result = concatenate_videoclips(segments)
            result = result.subclipped(0, min(target_duration - 0.05, result.duration))
            # Strip audio to avoid MoviePy composite bugs
            result.audio = None
            return result

        # Calculate valid start positions (after intro, ensuring segment fits)
        max_seg_start = source_dur - seg_dur
        if max_seg_start <= start_offset:
            end_time = min(start_offset + target_duration, source_dur)
            return clip.subclipped(start_offset, end_time)

        num_segs = max(1, math.ceil(target_duration / seg_dur))
        available_starts = list(range(int(start_offset), int(max_seg_start)))

        # Ensure we don't request more samples than available
        num_samples = min(num_segs, len(available_starts))
        if num_samples <= 0:
            end_time = min(start_offset + target_duration, source_dur)
            return clip.subclipped(start_offset, end_time)

        starts = sorted(random.sample(available_starts, num_samples))
        segments = [clip.subclipped(s, min(s + seg_dur, source_dur)) for s in starts]
        result = concatenate_videoclips(segments)

        # Safety: trim to target duration with small buffer
        if result.duration > target_duration:
            result = result.subclipped(0, target_duration - 0.05)
        # Strip audio to avoid MoviePy composite bugs
        result.audio = None
        log(f"Jumpcut time: {time.time() - jumpcut_start:.2f}s")
        return result

    def create_blurred_background(self, video_path: str) -> VideoFileClip:
        """
        Create blurred background video (yt_summarizer style).

        Steps:
        1. Resize to small (120x214)
        2. Apply FFmpeg boxblur
        3. Resize back to 720x1280
        4. Reduce brightness by 50%
        """
        log("Creating blurred background...")

        video = VideoFileClip(video_path)

        temp_blur_path = str(self.temp_dir / "blurred_bg.mp4")

        bg = video.resized((120, 214))
        temp_small = str(self.temp_dir / "temp_small.mp4")
        bg.write_videofile(
            temp_small,
            codec="libx264",
            fps=10,
            threads=2,
            preset=ENCODING_PRESET,
            logger=None,
        )
        bg.close()

        self._apply_fast_blur(temp_small, temp_blur_path, radius=BLUR_RADIUS)

        blurred = VideoFileClip(temp_blur_path).resized((VIDEO_WIDTH, VIDEO_HEIGHT))
        return blurred.with_effects([MultiplyColor(0.5)])

    def create_video_with_image_overlays(
        self,
        background_video_path: str,
        image_paths: list,
        audio_path: str,
        duration: float,
    ) -> CompositeVideoClip:
        """
        Create video with AI image overlays (experimental style).

        Features:
        - Background video cropped to 9:16
        - Image overlays with smooth animation
        - Audio integration
        """
        log("Creating video with image overlays...")

        background_video = VideoFileClip(background_video_path)
        audio = AudioFileClip(audio_path)

        num_overlays = max(
            1, int((duration - IMAGE_BOUNCE_INTERVAL) / IMAGE_BOUNCE_INTERVAL) + 1
        )
        if duration <= IMAGE_BOUNCE_INTERVAL:
            num_overlays = 0

        bg_w, bg_h = background_video.size
        if bg_w / bg_h < 9 / 16:
            new_w = int(bg_h * (16 / 9))
            background_video = background_video.cropped(
                x1=(bg_w - new_w) // 2, width=new_w
            )
        elif bg_w / bg_h > 16 / 9:
            new_h = int(bg_w * (9 / 16))
            background_video = background_video.cropped(
                y1=(bg_h - new_h) // 2, height=new_h
            )

        background_video = background_video.resized((VIDEO_WIDTH, VIDEO_HEIGHT))

        if background_video.duration < duration:
            loops = int(duration / background_video.duration) + 1
            background_video = concatenate_videoclips([background_video] * loops)

        background_video = background_video.subclipped(0, duration)

        overlay_clips = []
        for i in range(num_overlays):
            start_time = (i + 1) * IMAGE_BOUNCE_INTERVAL
            if start_time >= duration:
                break

            end_time = min(start_time + IMAGE_OVERLAY_DURATION, duration)
            overlay_duration = end_time - start_time

            if overlay_duration <= 0 or not image_paths:
                continue

            img_path = image_paths[i % len(image_paths)]

            try:
                img_clip = ImageClip(img_path).with_duration(overlay_duration)
                img_clip = img_clip.resized((VIDEO_WIDTH, VIDEO_HEIGHT))
                img_clip = self._apply_overlay_animation(img_clip, overlay_duration)
                img_clip = img_clip.with_start(start_time)
                overlay_clips.append(img_clip)
            except Exception as e:
                log(f"Error creating overlay: {e}", "ERROR")
                continue

        if overlay_clips:
            final_video = CompositeVideoClip(
                [background_video] + overlay_clips, size=(VIDEO_WIDTH, VIDEO_HEIGHT)
            )
        else:
            final_video = background_video

        final_video = final_video.with_audio(audio)

        try:
            background_video.close()
        except (OSError, AttributeError):
            pass

        return final_video

    @staticmethod
    def _vertical_crop_box(w: int, h: int) -> tuple[int, int, int, int] | None:
        """Minimal, always in-bounds center-crop box for out-of-range sources.

        Contract (preserves the historical simple-mode behavior):
        - Sources between 9:16 and 16:9 (e.g. 1920x1080, 4:3) are NOT cropped;
          the full frame is resized to the output, as before.
        - Sources wider than 16:9 are center-cropped to 16:9.
        - Sources taller than 9:16 are center-cropped to 9:16.

        Every returned box lies fully inside the source and has positive
        dimensions. Cropping out of bounds wraps around and returns a strip
        a few pixels tall, which later stretches into vertical streaks
        (the "smeared foreground" bug, #10).
        """
        if w <= 0 or h <= 0:
            return None
        if w / h > 16 / 9:  # wider than 16:9: crop the sides
            new_w = min(w, max(1, round(h * 16 / 9)))
            return ((w - new_w) // 2, 0, new_w, h)
        if w / h < 9 / 16:  # taller than 9:16: crop top/bottom
            new_h = min(h, max(1, round(w * 16 / 9)))
            return (0, (h - new_h) // 2, w, new_h)
        return None

    @staticmethod
    def _fit_fg_size(w: int, h: int) -> tuple[int, int]:
        """Foreground band size: source aspect preserved, fitted inside the
        output frame, always positive and even (yuv420p encoders reject odd
        dimensions). Never overflows the frame for tall sources and never
        rounds down to zero for extreme aspects.
        """
        if w <= 0 or h <= 0:
            return VIDEO_WIDTH, VIDEO_HEIGHT
        scale = min(VIDEO_WIDTH / w, VIDEO_HEIGHT / h)
        fw = max(2, round(w * scale))
        fh = max(2, round(h * scale))
        fw -= fw % 2
        fh -= fh % 2
        return min(fw, VIDEO_WIDTH), min(fh, VIDEO_HEIGHT)

    def create_output_video(
        self,
        video_path: str,
        audio_path: str,
        vtt_path: str,
        output_path: str,
        target_duration: float,
        use_blurred_bg: bool = True,
        image_paths: list[Any] | None = None,
        metadata: VideoMetadata | None = None,
    ) -> bool:
        """
        Create final output video with optional subtitle integration.

        Args:
            video_path: Source video path
            audio_path: TTS audio path
            vtt_path: Subtitle VTT path
            output_path: Final output path
            target_duration: Audio duration
            use_blurred_bg: Use blurred background (yt_summarizer style)
            image_paths: Image overlays (experimental style, overrides blurred_bg)

        Returns:
            bool: Success status
        """
        log("Compiling final video...")
        total_start = time.time()

        log("Loading video and audio...")
        load_start = time.time()
        video = VideoFileClip(video_path)
        audio = AudioFileClip(audio_path)
        log(f"Load time: {time.time() - load_start:.2f}s")

        if video.duration <= 0:
            log("Video has zero duration", "ERROR")
            return False

        overlay_mode = bool(image_paths and len(image_paths) >= 3) or use_blurred_bg
        if not overlay_mode:
            # Simple mode plays the source full-frame: normalize it to the
            # output aspect first. Overlay mode keeps the source untouched so
            # the foreground band can preserve its original aspect.
            crop_start = time.time()
            box = self._vertical_crop_box(*video.size)
            if box is not None:
                x1, y1, cw, ch = box
                video = video.cropped(x1=x1, y1=y1, width=cw, height=ch)
            video = video.resized((VIDEO_WIDTH, VIDEO_HEIGHT))
            log(f"Crop/resize time: {time.time() - crop_start:.2f}s")

        mode_start = time.time()
        if image_paths and len(image_paths) >= 3:
            final_video = self._create_with_overlay_mode(
                video, audio, vtt_path, target_duration, image_paths
            )
        elif use_blurred_bg:
            final_video = self._create_with_overlay_mode(
                video, audio, vtt_path, target_duration
            )
        else:
            final_video = self._create_simple_mode(
                video, audio, vtt_path, target_duration
            )
        log(f"Mode processing time: {time.time() - mode_start:.2f}s")

        encode_start = time.time()
        # Write video without audio to avoid MoviePy composite bugs
        temp_video = output_path.replace(".mp4", "_temp.mp4")
        meta_args = metadata.to_ffmpeg_args() if metadata else []
        ffmpeg_params = ["-crf", str(ENCODING_CRF)] + meta_args
        final_video.write_videofile(
            temp_video,
            codec=VIDEO_CODEC,
            audio=False,
            fps=VIDEO_FPS,
            threads=ENCODING_THREADS,
            preset=ENCODING_PRESET,
            ffmpeg_params=ffmpeg_params,
            logger="bar",
        )

        # Add audio back using ffmpeg (more reliable than MoviePy)
        if audio_path and Path(audio_path).exists():
            import subprocess

            cmd = [
                "ffmpeg",
                "-y",
                "-i",
                temp_video,
                "-i",
                audio_path,
                "-c:v",
                "copy",
                "-c:a",
                AUDIO_CODEC,
                "-shortest",
            ]
            if metadata:
                cmd.extend(metadata.to_ffmpeg_args())
            cmd.append(output_path)
            subprocess.run(cmd, capture_output=True, check=True)
            Path(temp_video).unlink(missing_ok=True)
        else:
            Path(temp_video).rename(output_path)

        log(f"Encode time: {time.time() - encode_start:.2f}s")

        try:
            final_video.close()
            video.close()
        except (OSError, AttributeError):
            pass

        log(f"Video compilation total: {time.time() - total_start:.2f}s", "SUCCESS")
        return True

    def _create_with_overlay_mode(
        self,
        video: VideoFileClip,
        audio: AudioFileClip,
        vtt_path: str,
        target_duration: float,
        image_paths: list | None = None,
    ) -> CompositeVideoClip:
        """
        Create video with blurred background + AI image overlays (experimental style).
        Uses flattened structure to avoid MoviePy timing bugs with nested composites.
        """
        blurred = self._create_blurred_background_from_clip(video)
        blurred = self._ensure_duration(blurred, target_duration)

        # Foreground band keeps the source aspect, fitted inside the frame,
        # instead of a fixed height that distorts the image.
        fg_w, fg_h = VideoCompositor._fit_fg_size(*video.size)
        fg = video.resized((fg_w, fg_h)).with_position(("center", "center"))
        fg = self._ensure_duration(fg, target_duration)

        base_composite = CompositeVideoClip(
            [blurred, fg], size=(VIDEO_WIDTH, VIDEO_HEIGHT)
        )

        if BG_MODE == "jumpcut":
            # _jumpcut_background already handles duration trimming
            final_video = self._jumpcut_background(base_composite, target_duration)
        else:
            # Subclip to target for speed-scaling path only (keeps playback at normal speed)
            if video.duration > target_duration:
                video = video.subclipped(0, target_duration)
            speed_factor = video.duration / target_duration
            final_video = base_composite.with_speed_scaled(speed_factor).with_duration(
                target_duration
            )

        # Add overlays and subtitles as a single composite operation
        all_clips = [final_video]
        if image_paths:
            num_overlays = max(
                1,
                int((target_duration - IMAGE_BOUNCE_INTERVAL) / IMAGE_BOUNCE_INTERVAL)
                + 1,
            )
            for i in range(num_overlays):
                if START_WITH_IMAGE and i == 0:
                    start_time = 0.0
                else:
                    start_time = (
                        i + (0 if START_WITH_IMAGE else 1)
                    ) * IMAGE_BOUNCE_INTERVAL
                if start_time >= target_duration:
                    break

                end_time = min(start_time + IMAGE_OVERLAY_DURATION, target_duration)
                overlay_duration = end_time - start_time

                if overlay_duration <= 0:
                    continue

                img_path = image_paths[i % len(image_paths)]
                try:
                    img_clip = ImageClip(img_path).with_duration(overlay_duration)
                    img_clip = img_clip.resized((VIDEO_WIDTH, VIDEO_HEIGHT))
                    img_clip = self._apply_overlay_animation(img_clip, overlay_duration)
                    img_clip = img_clip.with_start(start_time)
                    all_clips.append(img_clip)
                except Exception as e:
                    log(f"Error creating overlay: {e}", "ERROR")
                    continue

        subtitle_composite = None
        if vtt_path and Path(vtt_path).exists():
            subs = self.subtitle_system.render_subtitles(
                vtt_path, (VIDEO_WIDTH, VIDEO_HEIGHT)
            )
            if subs:
                safe_subs = []
                for sub in subs:
                    if hasattr(sub, "start") and sub.start < target_duration:
                        safe_subs.append(sub)
                if safe_subs:
                    subtitle_composite = CompositeVideoClip(
                        safe_subs, size=(VIDEO_WIDTH, VIDEO_HEIGHT)
                    )
                    all_clips.append(subtitle_composite)

        final_result = CompositeVideoClip(all_clips, size=(VIDEO_WIDTH, VIDEO_HEIGHT))

        if (
            OPENING_ANIMATION_ENABLED
            and OPENING_ANIMATION_DURATION > 0
            and final_result.duration > OPENING_ANIMATION_DURATION
            and image_paths
        ):
            try:
                first_image_path = image_paths[0]
                opening_img = (
                    ImageClip(first_image_path)
                    .resized((VIDEO_WIDTH, VIDEO_HEIGHT))
                    .with_duration(OPENING_ANIMATION_DURATION)
                )
                # Include subtitles inside the opening animation so they
                # get zoomed and blurred along with the image.
                opening_layers = [opening_img]
                if subtitle_composite is not None:
                    opening_subs = subtitle_composite.subclipped(
                        0, OPENING_ANIMATION_DURATION
                    )
                    opening_layers.append(opening_subs)
                opening_clip = CompositeVideoClip(
                    opening_layers, size=(VIDEO_WIDTH, VIDEO_HEIGHT)
                )
                opening_clip = self._apply_opening_animation(
                    opening_clip, OPENING_ANIMATION_DURATION
                )
                rest_section = final_result.subclipped(
                    OPENING_ANIMATION_DURATION, final_result.duration
                )
                final_result = concatenate_videoclips(
                    [opening_clip, rest_section], method="compose"
                )
            except Exception as e:
                log(f"Opening animation skipped: {e}", "WARNING")

        final_result.audio = audio
        return final_result

    def _add_image_overlays(
        self,
        base_video: CompositeVideoClip,
        image_paths: list,
        duration: float,
    ) -> CompositeVideoClip:
        """Add AI image overlays on top of video."""
        overlay_clips = []
        num_overlays = max(
            1, int((duration - IMAGE_BOUNCE_INTERVAL) / IMAGE_BOUNCE_INTERVAL) + 1
        )

        for i in range(num_overlays):
            if START_WITH_IMAGE and i == 0:
                start_time = 0.0
            else:
                start_time = (
                    i + (0 if START_WITH_IMAGE else 1)
                ) * IMAGE_BOUNCE_INTERVAL
            if start_time >= duration:
                break

            end_time = min(start_time + IMAGE_OVERLAY_DURATION, duration)
            overlay_duration = end_time - start_time

            if overlay_duration <= 0:
                continue

            img_path = image_paths[i % len(image_paths)]
            try:
                img_clip = ImageClip(img_path).with_duration(overlay_duration)
                img_clip = img_clip.resized((VIDEO_WIDTH, VIDEO_HEIGHT))
                img_clip = self._apply_overlay_animation(img_clip, overlay_duration)
                img_clip = img_clip.with_start(start_time)
                overlay_clips.append(img_clip)
            except Exception as e:
                log(f"Error creating overlay: {e}", "ERROR")
                continue

        if not overlay_clips:
            return base_video

        return CompositeVideoClip(
            [base_video] + overlay_clips, size=(VIDEO_WIDTH, VIDEO_HEIGHT)
        )

    def _create_simple_mode(
        self,
        video: VideoFileClip,
        audio: AudioFileClip,
        vtt_path: str,
        target_duration: float,
    ) -> CompositeVideoClip:
        """Simple video without blur."""
        video = self._ensure_duration(video, target_duration)
        if BG_MODE == "jumpcut":
            # _jumpcut_background already handles duration trimming
            final_video = self._jumpcut_background(video, target_duration)
            final_video.audio = audio
        else:
            speed_factor = video.duration / target_duration
            final_video = (
                video.with_speed_scaled(speed_factor)
                .with_duration(target_duration)
                .with_audio(audio)
            )

        if vtt_path and Path(vtt_path).exists():
            subs = self.subtitle_system.render_subtitles(
                vtt_path, (VIDEO_WIDTH, VIDEO_HEIGHT)
            )
            if subs:
                subtitle_composite = CompositeVideoClip(
                    subs, size=(VIDEO_WIDTH, VIDEO_HEIGHT)
                ).with_effects([FadeIn(0.1)])
                final_video = CompositeVideoClip([final_video, subtitle_composite])

        final_video = final_video.with_effects([FadeIn(0.1)])
        return final_video

    def _create_blurred_background_from_clip(
        self, video: VideoFileClip
    ) -> VideoFileClip:
        """Create blurred background from loaded video clip."""
        return self.create_blurred_background(str(video.filename))

    def _get_video_duration(self, video_path: str) -> float:
        return get_video_duration(video_path)
