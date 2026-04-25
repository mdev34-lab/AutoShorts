#!/usr/bin/env python3
"""
AutoShorts-Py: Automated Viral Video Generator (Optimized Version)

Performance improvements:
- Uses FFmpeg boxblur instead of per-frame PIL blur
- Simplified subtitle rendering (1 clip per word instead of 3)
- Cached text clip dimensions
- Faster encoding with better presets
- Reduced composite layers
"""

# General imports
import argparse
import os
import platform
import subprocess
import time
from pathlib import Path

from moviepy import AudioFileClip, CompositeVideoClip, VideoFileClip
from moviepy.video.fx import FadeIn, MultiplyColor

# Local imports
from .modules import (
    AUDIO_CODEC,
    BLUR_RADIUS,
    ENCODING_CRF,
    ENCODING_PRESET,
    ENCODING_THREADS,
    MAX_VIDEO_CUT_DURATION,
    VIDEO_CODEC,
    VIDEO_FPS,
    VIDEO_HEIGHT,
    VIDEO_WIDTH,
    ScriptGenerator,
    SubtitleSystem,
    TTSSystem,
    VideoBackgroundManager,
    clean_temp_files,
    create_temp_dir,
    get_video_duration,
    log,
    setup_directories,
)

# Create temporary directory
TEMP_DIR = create_temp_dir()


class VideoProcessor:
    """Handles video editing with MoviePy 2.0 - Optimized version."""

    def __init__(self):
        self.subtitle_system = SubtitleSystem()

    def _get_video_duration(self, video_path: str) -> float:
        return get_video_duration(video_path)

    def _apply_fast_blur(
        self, input_path: str, output_path: str, radius: int = BLUR_RADIUS
    ) -> str:
        """Apply blur using FFmpeg boxblur - MUCH faster than PIL per-frame."""
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

    def _create_blurred_background(self, video: VideoFileClip) -> VideoFileClip:
        """Create blurred background using FFmpeg - efficient approach."""
        temp_blur_path = str(TEMP_DIR / "blurred_bg.mp4")

        # Resize once and save
        bg = video.resized((120, 214))
        temp_small = str(TEMP_DIR / "temp_small.mp4")
        bg.write_videofile(
            temp_small,
            codec="libx264",
            fps=10,
            threads=2,
            preset="ultrafast",
            logger=None,
        )

        # Apply FFmpeg blur (very fast)
        self._apply_fast_blur(temp_small, temp_blur_path, radius=BLUR_RADIUS)

        # Reload and resize
        blurred = VideoFileClip(temp_blur_path).resized((720, 1280))
        return blurred.with_effects([MultiplyColor(0.5)])

    def cut_video(
        self, video_path: str, max_duration: int = MAX_VIDEO_CUT_DURATION
    ) -> str:
        start_time = time.time()
        duration = self._get_video_duration(video_path)
        if duration <= max_duration:
            return video_path

        log(f"Cutting video from {duration}s to {max_duration}s...")
        cut_path = str(TEMP_DIR / f"cut_{int(time.time())}.mp4")
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            video_path,
            "-t",
            str(max_duration),
            "-c",
            "copy",
            "-avoid_negative_ts",
            "1",
            cut_path,
        ]

        cmd_start = time.time()
        subprocess.run(cmd, capture_output=True, check=True)
        cmd_time = time.time() - cmd_start

        total_time = time.time() - start_time
        log(
            f"Video cutting completed in {cmd_time:.2f}s (total: {total_time:.2f}s)",
            "SUCCESS",
        )
        return cut_path

    def create_output_video(
        self,
        video_path: str,
        audio_path: str,
        vtt_path: str,
        output_path: str,
        target_duration: float,
    ):
        """Create final video with optimized rendering."""
        log("Compiling final video...")
        start_time = time.time()

        try:
            cut_start = time.time()
            video_path = self.cut_video(video_path)
            cut_time = time.time() - cut_start
            log(f"Video cutting phase: {cut_time:.2f}s", "INFO")

            # Load video once
            video = VideoFileClip(video_path)
            audio = AudioFileClip(audio_path)

            if video.duration <= 0:
                raise ValueError("Video has zero duration")

            source_duration = self._get_video_duration(video_path)
            speed_factor = source_duration / target_duration

            # 9:16 Crop
            w, h = video.size
            # Check if video is in wrong aspect ratio (should be portrait 9:16)
            if (
                w / h < 9 / 16
            ):  # Video is too tall/narrow (portrait when should be landscape)
                new_w = int(h * (16 / 9))  # Make it landscape 16:9
                video = video.cropped(x1=(w - new_w) // 2, width=new_w)
            elif (
                w / h > 16 / 9
            ):  # Video is too wide/short (landscape when should be portrait)
                new_h = int(w * (9 / 16))  # Make it portrait 9:16
                video = video.cropped(y1=(h - new_h) // 2, height=new_h)

            video = video.resized((VIDEO_WIDTH, VIDEO_HEIGHT))

            # Create background - use FFmpeg blur method
            content_h = int(VIDEO_HEIGHT * 0.45)
            fg = video.resized((VIDEO_WIDTH, content_h)).with_position(
                ("center", "center")
            )
            bg = self._create_blurred_background(video)

            # Composite base layers
            base_composite = CompositeVideoClip(
                [bg, fg], size=(VIDEO_WIDTH, VIDEO_HEIGHT)
            )

            # Speed-adjusted clip
            final_video = (
                base_composite.with_speed_scaled(speed_factor)
                .with_duration(target_duration)
                .with_audio(audio)
            )

            # Add optimized subtitles
            if vtt_path:
                subs = self.subtitle_system.render_subtitles(
                    vtt_path, (VIDEO_WIDTH, VIDEO_HEIGHT)
                )
                if subs:
                    # Create subtitle composite separately to reduce main composite complexity
                    subtitle_composite = CompositeVideoClip(
                        subs, size=(VIDEO_WIDTH, VIDEO_HEIGHT)
                    )
                    # Apply fade-in to subtitles
                    subtitle_composite = subtitle_composite.with_effects([FadeIn(0.1)])
                    final_video = CompositeVideoClip([final_video, subtitle_composite])

            # Apply fade-in to the final composite video
            final_video = final_video.with_effects([FadeIn(0.1)])

            # Optimized encoding settings
            encode_start = time.time()
            final_video.write_videofile(
                output_path,
                codec=VIDEO_CODEC,
                audio_codec=AUDIO_CODEC,
                fps=VIDEO_FPS,
                threads=ENCODING_THREADS,
                preset=ENCODING_PRESET,
                ffmpeg_params=["-crf", str(ENCODING_CRF)],
                logger="bar",
            )
            encode_time = time.time() - encode_start
            log(f"Video encoding phase: {encode_time:.2f}s", "INFO")

            # Cleanup clips (ignore close errors)
            try:
                final_video.close()
                base_composite.close()
            except (OSError, AttributeError):
                pass

            total_time = time.time() - start_time
            log(f"Total video compilation: {total_time:.2f}s", "SUCCESS")

            return True
        except Exception as e:
            log(f"Error: {e}", "ERROR")
            import traceback

            traceback.print_exc()
            return False


def shutdown_computer():
    """Shutdown the computer after processing is complete."""
    try:
        system = platform.system()
        if system == "Windows":
            os.system("shutdown /s /t 30")
            log("Computer will shutdown in 30 seconds...")
        elif system == "Linux" or system == "Darwin":
            os.system("shutdown -h +1")
            log("Computer will shutdown in 1 minute...")
        else:
            log("Unsupported OS for auto-shutdown", "WARNING")
    except Exception as e:
        log(f"Failed to shutdown computer: {e}", "ERROR")


async def generate_tts(paragraphs: list, output_dir: Path) -> tuple:
    """Generate TTS and VTT using the TTS system."""
    tts_system = TTSSystem()
    return await tts_system.generate_audio_and_subtitles(paragraphs, output_dir)


def main():
    parser = argparse.ArgumentParser(description="AutoShorts-Py Generator (Optimized)")
    parser.add_argument("subject", nargs="?", help="Video subject")
    parser.add_argument("-o", "--output", default="output", help="Output directory")
    parser.add_argument("-y", "--youtube-url", help="Direct YouTube URL")
    parser.add_argument(
        "--goodnight", action="store_true", help="Shutdown computer after processing"
    )
    parser.add_argument(
        "--batch", nargs="+", help="Batch processing: multiple subjects"
    )
    parser.add_argument(
        "--web-search",
        action="store_true",
        help="Use web search with gemini-fast model",
    )
    args = parser.parse_args()

    subjects = []

    if args.batch:
        subjects = args.batch
        log(f"Batch processing {len(subjects)} subjects: {subjects}")
    elif args.subject:
        subjects = [args.subject]
    elif args.youtube_url:
        subjects = [None]  # Will use URL metadata
    else:
        parser.error("Subject or URL required")

    output_path = Path(args.output)
    if output_path.suffix:
        output_dir = output_path.parent
        filename = output_path.name
    else:
        output_dir = output_path
        filename = f"short_{int(time.time())}.mp4"

    setup_directories()

    success_count = 0
    total_count = len(subjects)

    try:
        for i, subject in enumerate(subjects, 1):
            log(f"Processing {i}/{total_count}: {subject or 'URL video'}")

            try:
                video_bg = VideoBackgroundManager()

                if args.youtube_url:
                    video_path, title, description = video_bg.download_from_url(
                        args.youtube_url
                    )

                    # Use provided subject or generate from metadata
                    if subject:
                        current_subject = subject
                        script_generator = ScriptGenerator(web_search=args.web_search)
                        script = script_generator.generate_script(current_subject)
                    else:
                        # Generate script from video metadata
                        script_generator = ScriptGenerator(web_search=args.web_search)
                        script = script_generator.generate_script_from_metadata(
                            title, description
                        )
                        log("Generated script from YouTube video metadata", "SUCCESS")
                        current_subject = title or "Video"
                else:
                    # For batch mode, each subject gets its own video
                    video_path = video_bg.search_and_download(subject)
                    current_subject = subject
                    script_generator = ScriptGenerator(web_search=args.web_search)
                    script = script_generator.generate_script(current_subject)

                import asyncio

                audio_path, vtt_path, duration = asyncio.run(
                    generate_tts(script, TEMP_DIR)
                )

                # Generate unique filename for batch processing
                if args.batch:
                    filename = f"short_{current_subject.replace(' ', '_')[:20]}_{int(time.time())}.mp4"
                elif output_path.suffix:
                    filename = output_path.name
                else:
                    filename = f"short_{int(time.time())}.mp4"

                output_full_path = output_dir / filename
                proc = VideoProcessor()
                if proc.create_output_video(
                    video_path, audio_path, vtt_path, str(output_full_path), duration
                ):
                    log(f"SUCCESS! Video saved to: {output_full_path}", "SUCCESS")
                    success_count += 1
                else:
                    log("Video processing failed", "ERROR")

                # Clean temp files between batch items
                clean_temp_files(TEMP_DIR)

            except Exception as e:
                log(f"Error processing '{subject or 'URL video'}': {e}", "ERROR")
                continue

        log(
            f"Batch processing complete: {success_count}/{total_count} videos created successfully",
            "SUCCESS",
        )

        if args.goodnight and success_count > 0:
            shutdown_computer()

    except Exception as e:
        log(f"Fatal error: {e}", "ERROR")
        import traceback

        traceback.print_exc()
    finally:
        clean_temp_files(TEMP_DIR)


if __name__ == "__main__":
    main()
