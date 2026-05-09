#!/usr/bin/env python3
"""
Video Compositor Module

Provides unified video compositing with:
- Blurred background (using FFmpeg boxblur)
- AI image overlays with smooth easing animation
- Subtitle integration
- Standardized rendering pipeline

DRY principle: reused by both yt_summarizer and experimental modes.
"""

import subprocess
import time
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
from moviepy.video.fx import FadeIn, MultiplyColor, Resize

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
    START_WITH_IMAGE,
    VIDEO_CODEC,
    VIDEO_FPS,
    VIDEO_HEIGHT,
    VIDEO_WIDTH,
)
from .subtitle_system import SubtitleSystem
from .utils import create_temp_dir, get_video_duration, log


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
            "23",
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

    def _apply_overlay_animation(self, clip, duration: float):
        """Smooth zoom-in-then-zoom-out + opacity on overlay clips."""

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

        # MultiplyColor doesn't accept functions in MoviePy 2.x, use transform instead
        def apply_opacity(get_frame, t):
            frame = get_frame(t)
            opacity = opacity_anim(t)
            return np.minimum(255, frame * opacity).astype("uint8")

        clip = clip.with_effects([Resize(scale_anim)])
        return clip.transform(apply_opacity)

    def _jumpcut_background(self, clip, target_duration: float) -> VideoFileClip:
        """Create background by randomly sampling segments instead of speed scaling."""
        import random

        from moviepy import concatenate_videoclips

        jumpcut_start = time.time()
        seg_dur = JUMPCUT_SEG_DUR
        source_dur = clip.duration

        # Skip intro section (default 25s) to avoid logos/intros
        start_offset = min(BG_SKIP_INTRO, source_dur * 0.3)
        available_dur = source_dur - start_offset

        if available_dur <= 0:
            start_offset = 0
            available_dur = source_dur

        if available_dur <= target_duration:
            end_time = min(start_offset + target_duration, source_dur)
            return clip.subclipped(start_offset, end_time)

        # Calculate valid start positions (after intro, ensuring segment fits)
        max_seg_start = source_dur - seg_dur
        if max_seg_start <= start_offset:
            end_time = min(start_offset + target_duration, source_dur)
            return clip.subclipped(start_offset, end_time)

        num_segs = max(1, int(target_duration / seg_dur))
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
            preset="ultrafast",
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

    def create_output_video(
        self,
        video_path: str,
        audio_path: str,
        vtt_path: str,
        output_path: str,
        target_duration: float,
        use_blurred_bg: bool = True,
        image_paths: list[Any] | None = None,
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

        w, h = video.size
        crop_start = time.time()
        if w / h < 9 / 16:
            new_w = int(h * (16 / 9))
            video = video.cropped(x1=(w - new_w) // 2, width=new_w)
        elif w / h > 16 / 9:
            new_h = int(w * (9 / 16))
            video = video.cropped(y1=(h - new_h) // 2, height=new_h)

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
        final_video.write_videofile(
            temp_video,
            codec=VIDEO_CODEC,
            audio=False,
            fps=VIDEO_FPS,
            threads=ENCODING_THREADS,
            preset=ENCODING_PRESET,
            ffmpeg_params=["-crf", str(ENCODING_CRF)],
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
                output_path,
            ]
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
        # Subclip to target duration BEFORE processing to avoid encoding the full video
        if video.duration > target_duration:
            video = video.subclipped(0, target_duration)
        blurred = self._create_blurred_background_from_clip(video)

        content_h = int(VIDEO_HEIGHT * 0.45)
        fg = video.resized((VIDEO_WIDTH, content_h)).with_position(("center", "center"))

        base_composite = CompositeVideoClip(
            [blurred, fg], size=(VIDEO_WIDTH, VIDEO_HEIGHT)
        )

        if BG_MODE == "jumpcut":
            # _jumpcut_background already handles duration trimming
            final_video = self._jumpcut_background(base_composite, target_duration)
        else:
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
                    start_time = (i + (0 if START_WITH_IMAGE else 1)) * IMAGE_BOUNCE_INTERVAL
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
                start_time = (i + (0 if START_WITH_IMAGE else 1)) * IMAGE_BOUNCE_INTERVAL
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
