#!/usr/bin/env python3
"""
AutoShorts Experimental Mode: YouTube Summarizer with AI Image Cycling

This experimental mode combines the logic from yt_summarizer.py with AI-generated images
that cycle in and out every 5 seconds, creating a more diverse visual experience.
"""

# General imports
import argparse
import asyncio
import hashlib
import os
import platform
import random
import time
from pathlib import Path

import requests

from .modules import (
    API_KEY,
    API_TIMEOUT_IMAGE,
    IMAGE_BOUNCE_INTERVAL,
    IMAGE_CACHE_DIR,
    IMG_URL,
    MODEL_IMAGE,
    ScriptGenerator,
    TTSSystem,
    VideoBackgroundManager,
    VideoCompositor,
    clean_temp_files,
    create_temp_dir,
    log,
    setup_directories,
)


class ExperimentalYouTubeProcessor:
    """Experimental processor that combines YouTube content with cycling AI images."""

    def __init__(self):
        self.script_generator = ScriptGenerator()
        self.tts_system = TTSSystem()
        self.video_bg = VideoBackgroundManager()
        self.video_compositor = VideoCompositor()
        self.temp_dir = create_temp_dir()

    def _generate_search_query(self, subject: str) -> str:
        """Generate an AI-optimized YouTube search query."""
        return self.video_bg.generate_search_query(subject)

    def _download_youtube_video(self, subject: str) -> tuple:
        """Download YouTube video and return path and metadata."""
        video_path = self.video_bg.search_and_download(subject)
        return video_path, subject, ""

    def download_direct_url(self, url: str) -> tuple:
        """Download directly from URL and return video path and metadata."""
        return self.video_bg.download_from_url(url)

    def _generate_ai_images(
        self, subject: str, script_paragraphs: list, num_images: int
    ) -> list:
        """Generate AI images based on the subject and script with diverse visual styles."""
        log(f"Generating {num_images} AI images for experimental mode...")

        # Visual pattern interrupts - create "wait, what?" moments
        visual_styles = [
            # High-impact openings (use for first image - pattern interrupt)
            {
                "angle": "extreme close-up",
                "lighting": "dramatic sudden flash",
                "mood": "shocking glimpse",
                "shot": "detail shot",
                "effect": "motion blur",
            },
            {
                "angle": "low angle",
                "lighting": "intense spotlight",
                "mood": "jaw-dropping moment",
                "shot": "wide shot",
                "effect": "slow motion",
            },
            {
                "angle": "Dutch angle",
                "lighting": "neon pulse",
                "mood": "unexpected revelation",
                "shot": "medium shot",
                "effect": "glitch",
            },
            {
                "angle": "worm's eye view",
                "lighting": "backlit silhouette",
                "mood": "epic scale",
                "shot": "ground level",
                "effect": "dramatic sky",
            },
            # Dynamic mid-content styles
            {
                "angle": "bird's eye view",
                "lighting": "dramatic rim lighting",
                "mood": "epic and powerful",
                "shot": "aerial view",
                "effect": "cinematic",
            },
            {
                "angle": "close-up",
                "lighting": "soft diffused light",
                "mood": "intimate and detailed",
                "shot": "macro shot",
                "effect": "depth of field",
            },
            {
                "angle": "eye level",
                "lighting": "misty morning light",
                "mood": "mysterious and atmospheric",
                "shot": "full shot",
                "effect": "fog",
            },
            {
                "angle": "high angle",
                "lighting": "stark contrasted lighting",
                "mood": "dramatic and tense",
                "shot": "overhead shot",
                "effect": "shadow play",
            },
            # Escalation/momentum styles
            {
                "angle": "side profile",
                "lighting": "cold blue hour light",
                "mood": "melancholic and reflective",
                "shot": "profile shot",
                "effect": "blue hour",
            },
            {
                "angle": "high angle",
                "lighting": "stark daylight",
                "mood": "clarity and insight",
                "shot": "aerial",
                "effect": "bird's eye",
            },
            {
                "angle": "panoramic",
                "lighting": "vibrant sunset colors",
                "mood": "uplifting and colorful",
                "shot": "panoramic shot",
                "effect": "golden hour",
            },
            {
                "angle": "dynamic angle",
                "lighting": "action lighting",
                "mood": "energy and movement",
                "shot": "action shot",
                "effect": "motion",
            },
            # Landing/close styles for final frames
            {
                "angle": "symmetric",
                "lighting": "soft warm light",
                "mood": "satisfying conclusion",
                "shot": "centered shot",
                "effect": "clean composition",
            },
            {
                "angle": "fade out",
                "lighting": "end lighting",
                "mood": "resolution",
                "shot": "final shot",
                "effect": "cinematic end",
            },
        ]

        # Pattern interrupt color palettes - create curiosity/confusion
        color_palettes = [
            "high contrast dramatic",
            "unexpected color pop",
            "monochrome with accent",
            "desaturated with shock",
            "vibrant saturated energy",
            "moody dark tones",
            "warm cool contrast",
            "cinematic desaturated",
        ]

        # First image gets special "stop the scroll" treatment
        image_prompts = []
        for i in range(num_images):
            style = visual_styles[i % len(visual_styles)]
            color = color_palettes[i % len(color_palettes)]

            # First image = pattern interrupt (hook)
            if i == 0:
                # Use the most dramatic style for the hook
                hook_style = visual_styles[0]
                color = "high contrast dramatic, shocking visual"

                if script_paragraphs:
                    # Extract hook content from first paragraph
                    hook_content = script_paragraphs[0][:60]
                    prompt = (
                        f"VIRAL HOOK: {hook_content}, "
                        f"{hook_style['shot']}, {hook_style['angle']}, "
                        f"{hook_style['lighting']}, {hook_style['mood']}, "
                        f"{color}, highly detailed, 4K quality, "
                        f"vertical composition, STOP THE SCROLL effect, "
                        f"captivating frame, must-watch moment"
                    )
                else:
                    prompt = (
                        f"SHOCKING {subject} moment, {hook_style['shot']}, "
                        f"{hook_style['angle']}, {hook_style['lighting']}, "
                        f"{hook_style['mood']}, {color}, "
                        f"vertical 9:16, 4K, MUST STOP THE SCROLL, "
                        f"viral thumbnail quality, curiosity gap visual"
                    )
            else:
                # Subsequent images - story progression
                if script_paragraphs and i < len(script_paragraphs) * 2:
                    paragraph_idx = (i - 1) % len(script_paragraphs)
                    paragraph_text = script_paragraphs[paragraph_idx][:80]
                    content_focus = f"showing {paragraph_text}"
                else:
                    focuses = [
                        f"the story continues with {subject}",
                        f"revealing more about {subject}",
                        f"the aftermath of {subject}",
                        f"capturing {subject} in detail",
                        f"the impact visual of {subject}",
                    ]
                    content_focus = focuses[i % len(focuses)]

                prompt = (
                    f"{subject}, {content_focus}, "
                    f"{style['shot']}, {style['angle']}, "
                    f"{style['lighting']}, {style['mood']}, "
                    f"{color}, highly detailed, 4K, "
                    f"vertical composition, professional cinematography"
                )

            image_prompts.append(prompt)

        img_paths = []
        headers = {"Authorization": f"Bearer {API_KEY}"}

        # Ensure cache dir exists
        IMAGE_CACHE_DIR.mkdir(parents=True, exist_ok=True)

        for i, prompt in enumerate(image_prompts):
            # Generate cache key from prompt hash
            cache_key = hashlib.sha256(prompt.encode()).hexdigest()[:16]
            cached_path = IMAGE_CACHE_DIR / f"{cache_key}.jpg"

            if cached_path.exists():
                img_paths.append(str(cached_path))
                log(f"Using cached image {i + 1}: {cached_path.name}")
                continue

            log(f"Generating experimental image {i + 1}/{num_images}...")
            try:
                from urllib.parse import quote

                safe_prompt = quote(prompt)
                seed = random.randint(0, 999999)
                url = f"{IMG_URL}{safe_prompt}?model={MODEL_IMAGE}&width=1080&height=1920&seed={seed}&nologo=true"

                resp = requests.get(url, headers=headers, timeout=API_TIMEOUT_IMAGE)

                if resp.status_code == 200:
                    with open(cached_path, "wb") as f:
                        f.write(resp.content)
                    img_paths.append(str(cached_path))
                    log(f"Generated image {i + 1}: {cached_path.name}")
                else:
                    log(f"Failed to generate image {i}: {resp.status_code}", "ERROR")
                    raise Exception(f"Failed to generate image {i}: {resp.status_code}")

            except Exception as e:
                log(f"Image generation error: {e}", "ERROR")
                raise

        return img_paths

    async def process_experimental_video(
        self, subject: str, output_path: str, youtube_url: str = None
    ) -> bool:
        """Main processing function for experimental mode."""
        log("Starting experimental video processing...")
        start_time = time.time()

        try:
            # Step 1: Download YouTube video (from URL or search)
            log("Step 1: Downloading YouTube video...")
            if youtube_url:
                video_path, title, description = self.download_direct_url(youtube_url)
            else:
                video_path, title, description = self._download_youtube_video(subject)
            log(f"Downloaded video: {title[:50]}...")

            # Step 2: Generate script
            log("Step 2: Generating script...")
            if subject:
                # User provided subject - use it (ignore video metadata)
                script = self.script_generator.generate_script(subject)
            else:
                # URL only - use video metadata
                script = self.script_generator.generate_script_from_metadata(
                    title, description
                )
            log(f"Generated script with {len(script)} paragraphs")

            # Step 3: Generate TTS audio
            log("Step 3: Generating TTS audio...")
            (
                audio_path,
                vtt_path,
                duration,
            ) = await self.tts_system.generate_audio_and_subtitles(
                script, self.temp_dir
            )
            log(f"Generated TTS audio: {duration:.1f}s")

            # Step 4: Calculate number of images needed
            num_images = max(3, int(duration / IMAGE_BOUNCE_INTERVAL) + 1)
            log(
                f"Generating {num_images} AI images for {duration:.1f}s video (overlays every {IMAGE_BOUNCE_INTERVAL}s)"
            )

            # Step 5: Generate AI images
            log("Step 4: Generating AI images...")
            img_paths = self._generate_ai_images(subject, script, num_images)

            if len(img_paths) < 3:
                log(
                    f"Failed to generate enough images ({len(img_paths)}/required)",
                    "ERROR",
                )
                return False

            # Step 6: Create video with overlays via VideoCompositor
            log("Step 5: Creating video with AI image overlays...")
            if self.video_compositor.create_output_video(
                video_path,
                audio_path,
                vtt_path,
                output_path,
                duration,
                use_blurred_bg=True,
                image_paths=img_paths,
            ):
                total_time = time.time() - start_time
                log(f"Experimental video completed in {total_time:.2f}s", "SUCCESS")
                log(f"Video saved to: {output_path}", "SUCCESS")
                return True
            else:
                log("Video processing failed", "ERROR")
                return False

        except Exception as e:
            log(f"Error in experimental processing: {e}", "ERROR")
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


async def main():
    parser = argparse.ArgumentParser(
        description="AutoShorts Experimental Mode: YouTube + AI Image Cycling"
    )
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
    youtube_url = args.youtube_url

    if args.batch:
        subjects = args.batch
        log(f"Batch processing {len(subjects)} subjects: {subjects}")
    elif args.subject:
        subjects = [args.subject]
    elif youtube_url:
        subjects = [None]
    else:
        parser.error("Subject or URL required")

    output_path = Path(args.output)
    if output_path.suffix:
        output_dir = output_path.parent
        filename = output_path.name
    else:
        output_dir = output_path
        filename = f"experimental_{int(time.time())}.mp4"

    setup_directories()

    success_count = 0
    total_count = len(subjects)

    try:
        for i, subject in enumerate(subjects, 1):
            log(f"Processing experimental {i}/{total_count}: {subject}")

            try:
                processor = ExperimentalYouTubeProcessor()
                processor.script_generator = ScriptGenerator(web_search=args.web_search)

                if args.batch:
                    output_file = (
                        output_dir
                        / f"exp_{subject.replace(' ', '_')[:20]}_{int(time.time())}.mp4"
                    )
                elif output_path.suffix:
                    output_file = output_dir / filename
                else:
                    output_file = output_dir / f"experimental_{int(time.time())}.mp4"

                success = await processor.process_experimental_video(
                    subject if subject else "url video", str(output_file), youtube_url
                )

                if success:
                    success_count += 1
                    log(f"SUCCESS! Experimental video saved: {output_file}", "SUCCESS")
                else:
                    log("Experimental video processing failed", "ERROR")

                # Clean temp files between batch items
                clean_temp_files(processor.temp_dir)

            except Exception as e:
                log(
                    f"Error processing experimental video '{subject or 'url video'}': {e}",
                    "ERROR",
                )
                continue

        log(
            f"Experimental batch processing complete: {success_count}/{total_count} videos created successfully",
            "SUCCESS",
        )

        if args.goodnight and success_count > 0:
            shutdown_computer()

    except Exception as e:
        log(f"Fatal error in experimental mode: {e}", "ERROR")
        import traceback

        traceback.print_exc()


def cli_main():
    """Synchronous entry point for script execution."""
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nCancelled.")
        raise SystemExit(130) from None


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nCancelled.")
        raise SystemExit(130) from None
