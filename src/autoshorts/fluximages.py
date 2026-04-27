import argparse
import asyncio
import os
import platform
import random
import shutil
import sys
import tempfile
import time
from pathlib import Path
from urllib.parse import quote

import moviepy.video.fx as vfx
import requests

# MoviePy v2.x Imports
from moviepy import (
    AudioFileClip,
    CompositeVideoClip,
    ImageClip,
    concatenate_videoclips,
)

# Local imports
from .modules import (
    API_KEY,
    API_TIMEOUT_IMAGE,
    AUDIO_CODEC,
    CROSSFADE_TIME,
    ENCODING_CRF,
    ENCODING_PRESET,
    ENCODING_THREADS,
    IMG_URL,
    MAX_ZOOM_VALUE,
    MODEL_IMAGE,
    OUTPUT_DIR,
    SCENE_DURATION_SECONDS,
    START_FADE,
    VIDEO_CODEC,
    VIDEO_FPS,
    VIDEO_HEIGHT,
    VIDEO_WIDTH,
    ScriptGenerator,
    SubtitleSystem,
    TTSSystem,
    log,
)


class ScriptEngine:
    def __init__(self):
        self.script_generator = ScriptGenerator()

    def generate(self, subject):
        """Generate script and image prompts for Flux images."""
        return self.script_generator.generate_script_with_prompts(subject)


class AssetManager:
    def __init__(self, temp_dir):
        self.temp_dir = temp_dir
        self.tts_system = TTSSystem()

    def generate_ai_images(self, prompts):
        log(f"Generating {len(prompts)} images using Flux...")
        img_paths = []
        headers = {"Authorization": f"Bearer {API_KEY}"}

        for i, prompt in enumerate(prompts):
            try:
                # Encode prompt for URL
                safe_prompt = quote(
                    f"{prompt}, cinematic lighting, highly detailed, vertical composition"
                )
                # Add seed for variety
                seed = random.randint(0, 999999)
                url = f"{IMG_URL}{safe_prompt}?model={MODEL_IMAGE}&width=1080&height=1920&seed={seed}&nologo=true"

                log(f"Generating image {i + 1}...")
                resp = requests.get(url, headers=headers, timeout=API_TIMEOUT_IMAGE)

                if resp.status_code == 200:
                    path = self.temp_dir / f"ai_img_{i}.jpg"
                    with open(path, "wb") as f:
                        f.write(resp.content)
                    img_paths.append(str(path))
                else:
                    log(f"Failed to generate image {i}: {resp.status_code}", "ERROR")
                    raise Exception(f"Failed to generate image {i}: {resp.status_code}")

            except Exception as e:
                log(f"Image error: {e}", "ERROR")
                raise

        return img_paths

    async def generate_audio(self, paragraphs):
        """Generate audio using TTS system."""
        return await self.tts_system.generate_audio_only(paragraphs, self.temp_dir)


class VideoEngine:
    def __init__(self):
        self.subtitle_system = SubtitleSystem()

    def _u_curve_zoom(self, t, duration, max_zoom=MAX_ZOOM_VALUE):
        """Create a U-curve zoom effect (zoom in, then zoom out).

        Args:
            t: Current time (0 to duration)
            duration: Total duration of the clip
            max_zoom: Maximum zoom factor (default 1.15 = 15% zoom)

        Returns:
            Scale factor for the current time
        """
        if duration <= 0:
            return 1.0

        # Normalize time to 0-1 range
        normalized_t = t / duration

        # U-curve function: starts at 1, goes up to max_zoom at midpoint, returns to 1
        # Using a quadratic function that creates a smooth U-shape
        if normalized_t <= 0.5:
            # First half: zoom in (1 to max_zoom)
            # Quadratic easing: 4 * t^2 for smooth acceleration
            zoom_factor = 1 + (max_zoom - 1) * 4 * normalized_t * normalized_t
        else:
            # Second half: zoom out (max_zoom to 1)
            # Use symmetric quadratic function for smooth deceleration back to 1
            t_reversed = 1 - normalized_t  # Reverse time for second half
            zoom_factor = 1 + (max_zoom - 1) * 4 * t_reversed * t_reversed

        return zoom_factor

    def _apply_u_curve_zoom(self, clip, duration, max_zoom=MAX_ZOOM_VALUE):
        """Apply centered U-curve zoom effect to a clip.

        Args:
            clip: MoviePy clip to apply effect to
            duration: Duration of the clip
            max_zoom: Maximum zoom factor

        Returns:
            Clip with centered U-curve zoom effect applied
        """

        def zoom_and_center(t):
            """Apply zoom that stays centered on the clip"""
            zoom_factor = self._u_curve_zoom(t, duration, max_zoom)

            # For MoviePy, we need to return the scale factor
            # The centering will be handled by positioning
            return zoom_factor

        # Apply the zoom effect
        zoomed = clip.with_effects([vfx.Resize(zoom_and_center)])

        # Apply cropping to show only the center portion when zoomed
        def crop_center(t):
            """Crop to show center portion of zoomed clip"""
            zoom_factor = self._u_curve_zoom(t, duration, max_zoom)
            if zoom_factor <= 1.0:
                return 0, 0, 1080, 1920  # No cropping when not zoomed

            # Calculate crop dimensions to show center portion
            clip_w, clip_h = 1080, 1920  # Target dimensions
            zoomed_w = clip_w * zoom_factor
            zoomed_h = clip_h * zoom_factor

            # Calculate crop box to show center
            x1 = int((zoomed_w - clip_w) / 2)
            y1 = int((zoomed_h - clip_h) / 2)
            x2 = int(x1 + clip_w)
            y2 = int(y1 + clip_h)

            return x1, y1, x2, y2

        return zoomed.with_effects([vfx.Crop(crop_center)])

    def create_video(self, img_paths, audio_path, paragraphs, output_path):
        log("Composing Video with Fixed Scene Timing...")
        audio = AudioFileClip(audio_path)
        num_imgs = len(img_paths)

        # Use fixed scene duration (3 seconds) and calculate total video duration based on TTS
        duration_per_img = SCENE_DURATION_SECONDS
        total_video_duration = num_imgs * duration_per_img

        log(
            f"Variable timing: {num_imgs} scenes × {duration_per_img}s each = {total_video_duration:.1f}s total"
        )
        log(f"Audio duration: {audio.duration:.1f}s")

        # Adjust video to match audio duration (video follows TTS)
        if audio.duration > total_video_duration:
            # Audio is longer - add more scenes or extend existing ones
            log("Audio longer than video - adjusting video to match audio...")
            # For now, we'll just use the audio duration and let the video be shorter
            total_video_duration = audio.duration
        elif audio.duration < total_video_duration:
            log(
                f"Video longer than audio by {total_video_duration - audio.duration:.1f}s - will have silence at end"
            )

        clips = []
        for i, path in enumerate(img_paths):
            try:
                clip = (
                    ImageClip(path).with_duration(duration_per_img).resized(width=1080)
                )

                # Apply U-curve zoom effect instead of linear Ken Burns effect
                clip = self._apply_u_curve_zoom(
                    clip, duration_per_img, max_zoom=MAX_ZOOM_VALUE
                )

                # Crossfade / FadeIn Logic
                if i > 0:
                    clip = clip.with_effects([vfx.CrossFadeIn(CROSSFADE_TIME)])
                if i == 0:
                    clip = clip.with_effects([vfx.FadeIn(START_FADE)])

                clips.append(clip)
            except Exception as e:
                log(f"Error processing clip {path}: {e}", "ERROR")
                raise

        if not clips:
            raise ValueError("No clips were successfully created.")

        # Build video with variable timing based on TTS
        video = concatenate_videoclips(clips, method="compose", padding=-CROSSFADE_TIME)

        # Adjust video duration to match audio if needed
        if abs(video.duration - audio.duration) > 1.0:  # More than 1 second difference
            if video.duration < audio.duration:
                # Extend video by repeating scenes or adding padding
                log(
                    f"Extending video from {video.duration:.1f}s to match audio {audio.duration:.1f}s"
                )
                # Simple approach: add silence at the end
                video = video.with_duration(audio.duration)
            else:
                # Trim video to match audio
                log(
                    f"Trimming video from {video.duration:.1f}s to match audio {audio.duration:.1f}s"
                )
                video = video.subclipped(0, audio.duration)

        video = video.with_audio(audio)

        # Generate VTT file first
        import tempfile

        temp_vtt_dir = tempfile.mkdtemp()
        vtt_path = self.subtitle_system.generate_subtitles(
            paragraphs, video.duration, temp_vtt_dir
        )

        # Create subtitle clips from VTT using correct video dimensions
        subs = self.subtitle_system.render_subtitles(
            vtt_path, (VIDEO_WIDTH, VIDEO_HEIGHT)
        )

        final_video = CompositeVideoClip(
            [video] + subs, size=(VIDEO_WIDTH, VIDEO_HEIGHT)
        )

        # Render with optimized settings
        final_video.write_videofile(
            output_path,
            fps=VIDEO_FPS,
            codec=VIDEO_CODEC,
            audio_codec=AUDIO_CODEC,
            threads=ENCODING_THREADS,
            preset=ENCODING_PRESET,
            ffmpeg_params=["-crf", str(ENCODING_CRF)],
        )

        # Cleanup handles
        final_video.close()
        audio.close()
        for c in clips:
            c.close()


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


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("subject", nargs="?", help="Story topic")
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
    else:
        parser.error("Subject required")

    # Setup directories
    local_temp = Path("temp_assets")
    local_temp.mkdir(exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(dir=local_temp))
    OUTPUT_DIR.mkdir(exist_ok=True)

    success_count = 0
    total_count = len(subjects)

    try:
        for i, subject in enumerate(subjects, 1):
            log(f"Processing {i}/{total_count}: {subject}")

            try:
                script_engine = ScriptEngine()
                script_engine.script_generator = ScriptGenerator(
                    web_search=args.web_search
                )
                assets = AssetManager(temp_dir)
                video_engine = VideoEngine()

                # 1. AI Script Generation (paragraphs only)
                paragraphs, _ = script_engine.generate(subject)

                # 2. Generate TTS audio first
                audio_task = assets.generate_audio(paragraphs)
                audio_path = await audio_task

                # 3. Calculate image count based on TTS duration
                audio = AudioFileClip(audio_path)
                num_images = max(3, int(audio.duration / SCENE_DURATION_SECONDS))
                log(f"TTS duration: {audio.duration:.1f}s → {num_images} images needed")

                # 4. Generate image prompts based on script and calculated count
                image_prompts = (
                    script_engine.script_generator.generate_image_prompts_from_script(
                        paragraphs, num_images
                    )
                )

                # 5. Generate images
                img_paths = assets.generate_ai_images(image_prompts)

                if len(img_paths) < 3:
                    log(
                        f"Failed to generate enough images ({len(img_paths)}/required). Need at least 3 for video.",
                        "ERROR",
                    )
                    continue

                # 6. Render
                if args.batch:
                    out_file = (
                        OUTPUT_DIR
                        / f"flux_short_{subject.replace(' ', '_')[:20]}_{int(time.time())}.mp4"
                    )
                else:
                    out_file = OUTPUT_DIR / f"flux_short_{int(time.time())}.mp4"

                video_engine.create_video(
                    img_paths, audio_path, paragraphs, str(out_file)
                )

                log(f"SUCCESS! Video saved: {out_file}", "SUCCESS")
                success_count += 1

                # Clean temp files between batch items
                if temp_dir.exists():
                    for item in temp_dir.iterdir():
                        if item.is_file():
                            item.unlink()

            except Exception as e:
                log(f"Error processing '{subject}': {e}", "ERROR")
                continue

        log(
            f"Batch processing complete: {success_count}/{total_count} videos created successfully",
            "SUCCESS",
        )

        if args.goodnight and success_count > 0:
            shutdown_computer()

    except Exception as e:
        log(f"PROCESS FAILED: {e}", "ERROR")
        import traceback

        traceback.print_exc()
    finally:
        log("Cleaning up...")
        time.sleep(2)  # Wait for file release
        if temp_dir.exists():
            shutil.rmtree(temp_dir)


if __name__ == "__main__" and not sys.gettrace():
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nCancelled.")
        raise SystemExit(130) from None
