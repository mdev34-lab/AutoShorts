import asyncio
import json
import os
from datetime import datetime
from pathlib import Path

import typer

from ...generators.product import ProductGenerator
from ...modules import VideoMetadata, log
from ..new import new_app


@new_app.command(name="product")
def product_command(
    link: str = typer.Argument(..., help="Shopee product link (or shp.ee short link)"),
    output: str = typer.Option(
        "output", "--output", "-o", help="Output directory"
    ),
    pronunciation_file: str = typer.Option(
        None,
        "--pronunciation-file",
        help="JSON file with extra TTS respellings {spelling: respelling}",
    ),
):
    """Generate a product showcase video from a Shopee product link.

    Never fetches Shopee pages: the product name comes from the link slug
    and photos come from public sources that prove the exact model code.
    """
    output_path = Path(output)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = output_path / f"autoshorts_product_{ts}.mp4"

    metadata = VideoMetadata(
        title=None,
        comment=json.dumps({
            "product_link": link,
            "type": "product",
        }, ensure_ascii=False),
    )

    pronunciation_file = pronunciation_file or os.getenv("AUTOSHORTS_PRONUNCIATION_FILE")

    gen = ProductGenerator(
        product_url=link,
        output=str(out_file),
        pronunciation_file=pronunciation_file,
        metadata=metadata,
    )
    success = asyncio.run(gen.generate())
    gen.cleanup()
    if not success:
        raise typer.Exit(code=1)
    log("Done: 1/1 videos created", "SUCCESS")
