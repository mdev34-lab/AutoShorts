import asyncio
import json
import re
from datetime import datetime
from pathlib import Path

import typer

from ...generators import VIDEO_TYPES
from ...modules import VideoMetadata, log, shutdown_computer
from ..new import new_app


@new_app.command(name="explainer")
def explainer_command(
    subject: str = typer.Argument(None, help="Video subject"),
    output: str = typer.Option(
        "output", "--output", "-o", help="Output directory"
    ),
    youtube_url: str = typer.Option(
        None, "--youtube-url", "-y", help="Direct YouTube URL"
    ),
    goodnight: bool = typer.Option(
        False, "--goodnight", help="Shutdown after processing"
    ),
    batch: str = typer.Option(None, "--batch", help="Batch: semicolon-separated subjects (e.g. 'topic1; topic2')"),
    no_web_search: bool = typer.Option(
        False, "--no-web-search", help="Disable web search (use model knowledge only)"
    ),
    no_images: bool = typer.Option(False, "--no-images", help="Skip image overlays"),
    images_only: bool = typer.Option(
        False, "--images-only", help="Images only (no YouTube bg)"
    ),
    images: str = typer.Option(
        "web",
        "--images",
        help="Image source: 'web' (DDGS search) or 'ai' (Pollinations)",
    ),
):
    if no_images and images_only:
        raise typer.BadParameter("--no-images and --images-only are mutually exclusive")
    if images_only and youtube_url:
        raise typer.BadParameter("--youtube-url cannot be used with --images-only")
    if not subject and not youtube_url and not batch:
        raise typer.BadParameter("subject, --youtube-url, or --batch is required")
    if images not in ("web", "ai"):
        raise typer.BadParameter("--images must be 'web' or 'ai'")

    subjects: list[str | None] = []
    if batch:
        subjects = [s.strip() for s in batch.split(";") if s.strip()]
    elif subject:
        subjects = [subject]
    elif youtube_url:
        subjects = [None]

    output_path = Path(output)
    success_count = 0
    total_count = len(subjects)
    def _sanitize(s):
        return re.sub(r'[\\/*?:"<>|]', "", s).replace(" ", "_").lower()[:20]

    for i, subj in enumerate(subjects, 1):
        log(f"Processing {i}/{total_count}: {subj or 'youtube-url'}")
        try:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            parts = ["autoshorts"]
            if subj:
                parts.append(_sanitize(subj))
            if images_only:
                parts.append("explainer")
            parts.append(ts)
            name = "_".join(parts) + ".mp4"
            out_file = output_path / name

            metadata = VideoMetadata(
                title=subj,
                comment=json.dumps({
                    "subject": subj,
                    "youtube_url": youtube_url,
                    "no_images": no_images,
                    "images_only": images_only,
                    "image_source": images,
                    "no_web_search": no_web_search,
                    "batch": batch,
                }, ensure_ascii=False),
            )

            gen = VIDEO_TYPES["explainer"](
                subject=subj,
                output=str(out_file),
                youtube_url=youtube_url,
                web_search=not no_web_search,
                no_images=no_images or images_only,
                images_only=images_only,
                image_source=images,
                metadata=metadata,
            )
            success = asyncio.run(gen.generate())
            if success:
                success_count += 1
            gen.cleanup()
        except Exception as e:
            log(f"Error processing '{subj}': {e}", "ERROR")
            continue

    log(
        f"Done: {success_count}/{total_count} videos created",
        "SUCCESS" if success_count == total_count else "WARNING",
    )

    if goodnight and success_count > 0:
        log("Shutting down in 30s...")
        shutdown_computer()
