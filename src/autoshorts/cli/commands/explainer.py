import asyncio
import time
from pathlib import Path

import typer

from ...generators import VIDEO_TYPES
from ...modules import log, shutdown_computer
from ..new import new_app


@new_app.command(name="explainer")
def explainer_command(
    subject: str = typer.Argument(None, help="Video subject"),
    output: str = typer.Option(
        "output", "--output", "-o", help="Output directory or file path"
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
    tone: str = typer.Option(
        "opinionated",
        "--tone",
        help="Script tone: 'corporate' (neutral, factual) or 'opinionated' (dramatic, viral)",
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
    if tone not in ("corporate", "opinionated"):
        raise typer.BadParameter("--tone must be 'corporate' or 'opinionated'")

    subjects: list[str | None] = []
    if batch:
        subjects = [s.strip() for s in batch.split(";") if s.strip()]
    elif subject:
        subjects = [subject]
    elif youtube_url:
        subjects = [None]

    is_batch = batch is not None
    output_path = Path(output)
    success_count = 0
    total_count = len(subjects)

    for i, subj in enumerate(subjects, 1):
        log(f"Processing {i}/{total_count}: {subj or 'youtube-url'}")
        try:
            if output_path.suffix:
                out_dir = output_path.parent
                if is_batch:
                    prefix = "explainer_" if images_only else "as_"
                    name = f"{prefix}{subj.replace(' ', '_')[:20] if subj else 'video'}_{int(time.time())}.mp4"
                else:
                    name = output_path.name
            else:
                out_dir = output_path
                if is_batch:
                    prefix = "explainer_" if images_only else "as_"
                    name = f"{prefix}{subj.replace(' ', '_')[:20] if subj else 'video'}_{int(time.time())}.mp4"
                else:
                    prefix = "explainer_" if images_only else "autoshorts_"
                    name = f"{prefix}{int(time.time())}.mp4"

            out_file = out_dir / name

            gen = VIDEO_TYPES["explainer"](
                subject=subj,
                output=str(out_file),
                youtube_url=youtube_url,
                web_search=not no_web_search,
                no_images=no_images or images_only,
                images_only=images_only,
                image_source=images,
                tone=tone,
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
