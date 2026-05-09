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
import random
import time
import traceback
from pathlib import Path
from urllib.parse import quote

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
    shutdown_computer,
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
        """Generate AI images based on the subject with diverse concrete scene descriptions."""
        log(f"Generating {num_images} AI images for experimental mode...")

        # Concrete scene templates - tells the AI exactly what to draw
        scene_templates = [
            "wide shot showing the scene of {subject}, realistic photography",
            "close up of {subject}, detailed view, professional photo",
            "action shot of {subject}, dynamic moment, high quality photo",
            "aerial view of {subject}, landscape, panoramic photo",
            "detailed close up of {subject} event, texture, realistic photo",
            "crowd and atmosphere of {subject}, event photography",
            "key moment from {subject}, dramatic scene, photo quality",
            "behind the scenes of {subject}, documentary photography",
            "wide angle capturing {subject} in its environment, real photo",
            "portrait style shot related to {subject}, professional photography",
            "the aftermath scene of {subject}, still moment, documentary",
            "Preparations for {subject}, behind the scenes, candid photo",
        ]

        # Generate concrete visual prompts
        image_prompts = []
        for i in range(num_images):
            template = scene_templates[i % len(scene_templates)]
            prompt = template.format(subject=subject)

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
        self, subject: str, output_path: str, youtube_url: str | None = None
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
            traceback.print_exc()
            return False


async def run():
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


def main():
    """Synchronous entry point for script execution."""
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        print("\nCancelled.")
        raise SystemExit(130) from None


if __name__ == "__main__":
    main()
