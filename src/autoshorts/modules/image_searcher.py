import hashlib
import random
import re
from pathlib import Path
from urllib.parse import quote

import requests
from PIL import Image

from . import image_sources
from .config import (
    API_KEY,
    API_TIMEOUT_IMAGE,
    IMAGE_CACHE_DIR,
    IMAGE_MIN_HEIGHT,
    IMAGE_MIN_WIDTH,
    IMAGE_PROVIDERS,
    IMAGE_SEARCH_PER_QUERY,
    IMAGE_SEARCH_SAFE,
    IMAGE_TEXT_REMOVAL,
    IMG_URL,
    MODEL_IMAGE,
    OUTPUT_DIR,
    SUBJECT_FRAMING,
    TARGET_IMAGE_HEIGHT,
    TARGET_IMAGE_WIDTH,
)
from .logging_system import log

BLOCKED_DOMAINS: set[str] = {
    "crossdresser",
    "sissy",
    "femboy",
    "hentai",
    "rule34",
    "xvideos",
    "xnxx",
    "xhamster",
    "pornhub",
    "onlyfans",
    "redtube",
    "youporn",
    "erotic",
    "nsfw",
}

BLOCKED_KEYWORDS: set[str] = {
    "crossdresser",
    "sissy",
    "femboy",
    "hentai",
    "rule34",
    "nsfw",
    "xxx",
    "18+",
    "erotic",
    "nude",
    "naked",
    "seductive",
}


class ImageSearcher:
    def __init__(
        self,
        target_width: int = TARGET_IMAGE_WIDTH,
        target_height: int = TARGET_IMAGE_HEIGHT,
        safesearch: str = IMAGE_SEARCH_SAFE,
        min_width: int = IMAGE_MIN_WIDTH,
        min_height: int = IMAGE_MIN_HEIGHT,
        max_per_query: int = IMAGE_SEARCH_PER_QUERY,
    ):
        self.target_width = target_width
        self.target_height = target_height
        self.safesearch = safesearch
        self.min_width = min_width
        self.min_height = min_height
        self.max_per_query = max_per_query
        IMAGE_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _is_nsfw(result: dict) -> bool:
        text = (
            f"{result.get('image') or ''} "
            f"{result.get('url') or result.get('source') or ''} "
            f"{result.get('title') or ''}"
        )
        for d in BLOCKED_DOMAINS:
            if re.search(rf"(?:^|[\W_]){re.escape(d)}(?:$|[\W_])", text, re.IGNORECASE):
                return True
        for kw in BLOCKED_KEYWORDS:
            if re.search(
                rf"(?:^|[\W_]){re.escape(kw)}(?:$|[\W_])", text, re.IGNORECASE
            ):
                return True
        return False

    def search_images(self, query: str) -> list[dict]:
        try:
            from ddgs import DDGS

            results = list(
                DDGS().images(
                    query,
                    safesearch=self.safesearch,
                    max_results=self.max_per_query,
                )
            )
            log(f"DDGS image search '{query[:60]}': {len(results)} results", "INFO")
            return results
        except ImportError:
            log("DDGS not installed, cannot search web images", "WARNING")
            return []
        except Exception as e:
            log(f"DDGS image search failed: {e}", "WARNING")
            return []

    def download_image(self, url: str, cache_path: Path) -> bool:
        try:
            resp = requests.get(url, timeout=API_TIMEOUT_IMAGE, stream=True)
            if resp.status_code != 200:
                return False
            content_type = resp.headers.get("Content-Type", "")
            if not content_type.startswith("image/"):
                return False
            with open(cache_path, "wb") as f:
                for chunk in resp.iter_content(8192):
                    f.write(chunk)
            if cache_path.stat().st_size == 0:
                cache_path.unlink(missing_ok=True)
                return False
            return True
        except Exception as e:
            log(f"Download failed: {url[:60]}... - {e}", "WARNING")
            return False

    def resize_to_fill(self, image_path: Path, output_path: Path):
        # Order matters: text removal runs FIRST (inpainting needs the
        # untouched source, and removed text must not pull the saliency
        # centroid), subject framing SECOND.
        sidecar = None
        if IMAGE_TEXT_REMOVAL:
            from .text_removal import remove_overlay_text

            sidecar = image_path.with_name(image_path.stem + "_notext.jpg")
            remove_overlay_text(image_path, sidecar)
            image_path = sidecar
        if SUBJECT_FRAMING:
            from .image_framing import frame_subject_file

            try:
                frame_subject_file(
                    image_path, output_path, self.target_width, self.target_height
                )
            finally:
                if sidecar is not None:
                    sidecar.unlink(missing_ok=True)
            return
        img = Image.open(image_path).convert("RGB")
        if sidecar is not None:
            # convert() already copied the pixels; the sidecar can go.
            sidecar.unlink(missing_ok=True)
        target_ratio = self.target_width / self.target_height
        img_ratio = img.width / img.height

        if img_ratio > target_ratio:
            new_height = self.target_height
            new_width = int(new_height * img_ratio)
        else:
            new_width = self.target_width
            new_height = int(new_width / img_ratio)

        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

        left = (new_width - self.target_width) // 2
        top = (new_height - self.target_height) // 2
        img = img.crop((left, top, left + self.target_width, top + self.target_height))

        img.save(output_path, "JPEG", quality=85)
        img.close()

    @staticmethod
    def _cache_key(prompt: str) -> str:
        # Versioned by provider config: legacy hashes predate provenance/fit
        # gating and must never be reused, and a provider change invalidates.
        scheme = f"img2|{','.join(IMAGE_PROVIDERS)}|{prompt}"
        return hashlib.sha256(scheme.encode()).hexdigest()[:16]

    @classmethod
    def _usable_cache(cls, cached_path: Path) -> bool:
        """A cached web image is reusable only with its provenance sidecar -
        anything else (legacy stock, interrupted runs) is refetched."""
        return cached_path.exists() and cached_path.with_suffix(".credits.txt").exists()

    def get_images(self, prompts: list[str]) -> list[str]:
        img_paths = []
        total = len(prompts)

        for i, prompt in enumerate(prompts):
            log(f"Searching web image {i + 1}/{total}: '{prompt[:60]}...'", "INFO")

            cache_key = self._cache_key(prompt)
            cached_path = IMAGE_CACHE_DIR / f"{cache_key}.jpg"
            if self._usable_cache(cached_path):
                img_paths.append(str(cached_path))
                log(f"  Using cached: {cached_path.name}", "INFO")
                continue

            candidates = self._candidates(prompt)
            downloaded = False
            for candidate in candidates:
                url = candidate.url
                if not url:
                    continue
                temp_path = (
                    IMAGE_CACHE_DIR
                    / f"dl_{cache_key}_{hashlib.md5(url.encode()).hexdigest()[:8]}.jpg"
                )
                if self.download_image(url, temp_path):
                    try:
                        with Image.open(temp_path) as check:
                            actual_w, actual_h = check.size
                        if actual_w < 100 or actual_h < 100:
                            log(
                                f"  Downloaded image too small ({actual_w}×{actual_h}), skipping",
                                "WARNING",
                            )
                            temp_path.unlink(missing_ok=True)
                            continue
                        self.resize_to_fill(temp_path, cached_path)
                        temp_path.unlink(missing_ok=True)
                        img_paths.append(str(cached_path))
                        self._write_provenance_sidecar(cached_path, candidate, prompt)
                        log(
                            f"  Downloaded & processed: {cached_path.name} "
                            f"({candidate.provider}, license: "
                            f"{candidate.license or 'unverified'}, "
                            f"fit: {candidate.fit_evidence})",
                            "INFO",
                        )
                        downloaded = True
                        break
                    except Exception as e:
                        log(f"  Resize failed: {e}", "WARNING")
                        temp_path.unlink(missing_ok=True)

            if not downloaded:
                log(
                    "  Web search failed for prompt, generating via AI instead",
                    "WARNING",
                )
                self._fallback_to_ai([prompt], img_paths)

        self._write_credits_file(img_paths)
        return img_paths

    @staticmethod
    def _write_provenance_sidecar(
        image_path: Path, candidate: "image_sources.SourceCandidate", prompt: str
    ) -> None:
        """Persist provenance next to the cached image so credits survive cache hits."""
        try:
            license_line = candidate.license or "UNVERIFIED - review before publishing"
            lines = [
                f"prompt: {prompt}",
                f"provider: {candidate.provider}",
                f"source: {candidate.url}",
                f"license: {license_line}",
            ]
            if candidate.attribution:
                lines.append(f"attribution: {candidate.attribution}")
            lines.append(f"fit: {candidate.fit_evidence}")
            image_path.with_suffix(".credits.txt").write_text(
                "\n".join(lines) + "\n", encoding="utf-8"
            )
        except Exception as e:
            log(f"  Could not write provenance sidecar: {e}", "WARNING")

    @staticmethod
    def _write_credits_file(img_paths: list[str]) -> None:
        """Deliver per-image credits next to the video output.

        Never presents an unlicensed image as cleared: entries without a
        verified license are marked UNVERIFIED for review before publishing.
        """
        try:
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            blocks = []
            for p in img_paths:
                sidecar = Path(p).with_suffix(".credits.txt")
                if sidecar.exists():
                    blocks.append(
                        f"{Path(p).name}\n"
                        + sidecar.read_text(encoding="utf-8").strip()
                    )
                else:
                    blocks.append(
                        f"{Path(p).name}\n"
                        "license: UNVERIFIED - no provenance record "
                        "(AI-generated or cached before credits tracking); "
                        "review before publishing"
                    )
            (OUTPUT_DIR / "image_credits.txt").write_text(
                "\n\n".join(blocks) + "\n", encoding="utf-8"
            )
        except Exception as e:
            log(f"Could not write image credits file: {e}", "WARNING")

    def _candidates(self, query: str) -> list[image_sources.SourceCandidate]:
        """Provider-ordered candidates with provenance and fit evidence."""
        candidates: list[image_sources.SourceCandidate] = []
        for provider in IMAGE_PROVIDERS:
            if provider == "commons":
                candidates.extend(self._commons_candidates(query))
            elif provider == "ddg":
                raw = [r for r in self.search_images(query) if not self._is_nsfw(r)]
                candidates.extend(image_sources.filter_ddg_results(query, raw))
        return candidates

    def _commons_candidates(self, query: str) -> list[image_sources.SourceCandidate]:
        return image_sources.search_commons(
            query,
            limit=self.max_per_query,
            min_width=self.min_width,
            min_height=self.min_height,
        )

    def _fallback_to_ai(
        self, remaining_prompts: list[str], collected_paths: list[str]
    ) -> list[str]:
        log(
            f"Falling back to Pollinations.ai for {len(remaining_prompts)} images",
            "INFO",
        )
        headers = {"Authorization": f"Bearer {API_KEY}"}

        for prompt in remaining_prompts:
            cache_key = self._cache_key(prompt)
            cached_path = IMAGE_CACHE_DIR / f"{cache_key}.jpg"
            if self._usable_cache(cached_path):
                collected_paths.append(str(cached_path))
                continue

            log(f"  AI generating: '{prompt[:60]}...'", "INFO")
            try:
                safe_prompt = quote(prompt)
                seed = random.randint(0, 999999)
                url = f"{IMG_URL}{safe_prompt}?model={MODEL_IMAGE}&width={self.target_width}&height={self.target_height}&seed={seed}&nologo=true"
                resp = requests.get(url, headers=headers, timeout=API_TIMEOUT_IMAGE)
                if resp.status_code == 200:
                    with open(cached_path, "wb") as f:
                        f.write(resp.content)
                    self._write_provenance_sidecar(
                        cached_path,
                        image_sources.SourceCandidate(
                            url=url.split("?")[0],
                            provider="ai",
                            license="AI-generated - no source license",
                            attribution=None,
                            fit_evidence="generated-for-prompt",
                        ),
                        prompt,
                    )
                    collected_paths.append(str(cached_path))
                else:
                    raise Exception(f"HTTP {resp.status_code}")
            except Exception as e:
                log(f"  AI image error: {e}", "ERROR")
                raise

        return collected_paths
