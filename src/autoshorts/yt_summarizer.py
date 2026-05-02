#!/usr/bin/env python3
"""
AutoShorts-Py: Automated Viral Video Generator

Uses VideoCompositor module for unified video processing.
"""

import argparse
import asyncio
import time
import traceback
from pathlib import Path

from .modules import (
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

TEMP_DIR = create_temp_dir()


async def generate_tts(paragraphs, output_dir):
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

                if args.batch:
                    filename = f"short_{current_subject.replace(' ', '_')[:20]}_{int(time.time())}.mp4"
                elif output_path.suffix:
                    filename = output_path.name
                else:
                    filename = f"short_{int(time.time())}.mp4"

                output_full_path = output_dir / filename
                compositor = VideoCompositor()
                if compositor.create_output_video(
                    video_path,
                    audio_path,
                    vtt_path,
                    str(output_full_path),
                    duration,
                    use_blurred_bg=True,
                ):
                    log(f"SUCCESS! Video saved to: {output_full_path}", "SUCCESS")
                    success_count += 1
                else:
                    log("Video processing failed", "ERROR")

                clean_temp_files(TEMP_DIR)

        except Exception as e:
            log(f"Error processing '{subject or 'URL video'}': {e}", "ERROR")
            traceback.print_exc()
            continue

    log(
        f"Batch processing complete: {success_count}/{total_count} videos created successfully",
        "SUCCESS",
    )

    if args.goodnight and success_count > 0:
        shutdown_computer()

except Exception as e:
    log(f"Fatal error: {e}", "ERROR")
    traceback.print_exc()
finally:
        clean_temp_files(TEMP_DIR)


if __name__ == "__main__":
    import time

    try:
        main()
    except KeyboardInterrupt:
        print("\nCancelled.")
        raise SystemExit(130) from None
