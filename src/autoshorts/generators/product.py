"""Product-link video generator (images-only, fail-closed sources).

Pipeline: parse the product link (slug only, never the Shopee page),
compose the showcase script deterministically from facts verified in the
product name (no free text generation, so narration cannot fabricate
specs), synthesize narration with TTS-only pronunciation respellings,
then collect product photos from public sources that prove the exact
model code, and compose the standard images-only video.
"""

from __future__ import annotations

import time
import traceback

from moviepy import AudioFileClip

from ..modules.config import OUTPUT_DIR, SCENE_DURATION_SECONDS
from ..modules.image_searcher import ImageSearcher
from ..modules.logging_system import log
from ..modules.product.discovery import search_image_candidates
from ..modules.product.errors import ProductPipelineError
from ..modules.product.narration import compose_showcase_script
from ..modules.product.network import safe_get
from ..modules.product.product import parse_product_link
from ..modules.product.pronunciation import load_dictionary, prepare_for_tts
from ..modules.product.provenance import verify_image_sources
from ..modules.product.script_guard import assert_script_within_verified_facts
from ..modules.utils import setup_directories
from .explainer import ExplainerGenerator

MIN_IMAGES = 3
MAX_CANDIDATES_TO_TRY = 8
MAX_IMAGE_BYTES = 20 * 1024 * 1024


class ProductGenerator(ExplainerGenerator):
    """`autoshorts new product <link>` generator."""

    def __init__(
        self, *, product_url: str, pronunciation_file: str | None = None, **kwargs
    ):
        kwargs.setdefault("web_search", False)
        kwargs.setdefault("images_only", True)
        kwargs.setdefault("no_images", True)
        super().__init__(**kwargs)
        self.product_url = product_url
        self.pronunciation_file = pronunciation_file

    async def generate(self) -> bool:
        log("Starting product video (images-only, verified sources)...")
        start_time = time.time()
        try:
            setup_directories()
            OUTPUT_DIR.mkdir(exist_ok=True)

            log("Step 1: Parsing product link (no Shopee fetch)...")
            product = parse_product_link(self.product_url)
            log(f"Product: {product.name}", "SUCCESS")

            log("Step 2: Composing showcase script from verified name facts...")
            paragraphs = compose_showcase_script(product.name)
            if not paragraphs or len(paragraphs) < 3:
                log(
                    f"Script composition failed: got {len(paragraphs)} paragraphs, need >= 3",
                    "ERROR",
                )
                return False
            try:
                assert_script_within_verified_facts(paragraphs, product.name)
            except ProductPipelineError as exc:
                log(f"Script rejected by the verified-facts guard: {exc}", "ERROR")
                return False
            self.metadata.description = " ".join(paragraphs)

            log("Step 3: Generating TTS audio (with pronunciation respellings)...")
            pron_dict = load_dictionary(self.pronunciation_file)
            audio_path = await self.tts_system.generate_audio_only(
                paragraphs,
                self.temp_dir,
                text_transform=lambda text: prepare_for_tts(text, pron_dict),
            )

            audio = AudioFileClip(audio_path)
            num_images = max(MIN_IMAGES, int(audio.duration / SCENE_DURATION_SECONDS))
            log(f"TTS: {audio.duration:.1f}s -> {num_images} images")
            audio.close()

            log("Step 4: Searching public image sources...")
            candidates = search_image_candidates(product.name, limit=12)
            if not candidates:
                log("No image candidates found", "ERROR")
                return False

            log("Step 5: Verifying sources (fail-closed model match)...")
            verified: list = []
            for candidate in candidates[:MAX_CANDIDATES_TO_TRY]:
                if len(verified) >= num_images:
                    break
                try:
                    evidence = verify_image_sources([candidate], product.name)
                except ProductPipelineError as e:
                    log(f"Rejected {candidate.image_url[:60]}...: {e}", "WARNING")
                    continue
                verified.append((candidate, evidence[0]))
            if len(verified) < MIN_IMAGES:
                log(
                    f"Only {len(verified)} verified images, need >= {MIN_IMAGES}; "
                    "aborting without unverified media",
                    "ERROR",
                )
                return False
            log(f"{len(verified)} verified images", "SUCCESS")

            log("Step 6: Downloading and framing images...")
            searcher = ImageSearcher()
            img_paths = []
            for i, (candidate, _evidence) in enumerate(verified):
                try:
                    data, _ = safe_get(
                        candidate.image_url,
                        operation="product_image_download",
                        max_bytes=MAX_IMAGE_BYTES,
                        expected_content_prefix="image/",
                    )
                except ProductPipelineError as e:
                    log(
                        f"Download failed for {candidate.image_url[:60]}...: {e}",
                        "WARNING",
                    )
                    continue
                raw = self.temp_dir / f"product_raw_{i}.img"
                raw.write_bytes(data)
                final = self.temp_dir / f"product_scene_{i}.jpg"
                try:
                    searcher.resize_to_fill(raw, final)
                except Exception as e:
                    log(f"Framing failed for image {i}: {e}", "WARNING")
                    raw.unlink(missing_ok=True)
                    continue
                raw.unlink(missing_ok=True)
                img_paths.append(str(final))
            if len(img_paths) < MIN_IMAGES:
                log(
                    f"Only {len(img_paths)} usable images, need >= {MIN_IMAGES}",
                    "ERROR",
                )
                return False

            log("Step 7: Composing video...")
            self._create_flux_video(
                img_paths, audio_path, paragraphs, self.output, self.metadata
            )

            elapsed = time.time() - start_time
            log(f"Video completed in {elapsed:.2f}s", "SUCCESS")
            log(f"Saved: {self.output}", "SUCCESS")
            return True
        except ProductPipelineError as e:
            log(f"Product pipeline aborted: {e}", "ERROR")
            return False
        except Exception as e:
            log(f"Error: {e}", "ERROR")
            traceback.print_exc()
            return False
