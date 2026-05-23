import hashlib
import random
from pathlib import Path
from urllib.parse import quote

import requests
from PIL import Image

from .config import (
    API_KEY,
    API_TIMEOUT_IMAGE,
    IMAGE_CACHE_DIR,
    IMAGE_MIN_HEIGHT,
    IMAGE_MIN_WIDTH,
    IMAGE_SEARCH_PER_QUERY,
    IMAGE_SEARCH_SAFE,
    IMG_URL,
    MODEL_IMAGE,
    TARGET_IMAGE_HEIGHT,
    TARGET_IMAGE_WIDTH,
)
from .logging_system import log


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
        img = Image.open(image_path).convert("RGB")
        target_ratio = self.target_width / self.target_height
        img_ratio = img.width / img.height

        if img_ratio > target_ratio:
            new_height = self.target_height
            new_width = int(new_height * img_ratio)
        else:
            new_width = self.target_width
            new_height = int(new_width / img_ratio)

        img = img.resize((new_width, new_height), Image.LANCZOS)

        left = (new_width - self.target_width) // 2
        top = (new_height - self.target_height) // 2
        img = img.crop((left, top, left + self.target_width, top + self.target_height))

        img.save(output_path, "JPEG", quality=85)
        img.close()

    def get_images(self, prompts: list[str]) -> list[str]:
        img_paths = []
        total = len(prompts)

        for i, prompt in enumerate(prompts):
            log(f"Searching web image {i + 1}/{total}: '{prompt[:60]}...'", "INFO")

            cache_key = hashlib.sha256(prompt.encode()).hexdigest()[:16]
            cached_path = IMAGE_CACHE_DIR / f"{cache_key}.jpg"
            if cached_path.exists():
                img_paths.append(str(cached_path))
                log(f"  Using cached: {cached_path.name}", "INFO")
                continue

            results = self.search_images(prompt)
            downloaded = False
            for r in results:
                url = r.get("image", "")
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
                        log(f"  Downloaded & processed: {cached_path.name}", "INFO")
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

        return img_paths

    def _fallback_to_ai(
        self, remaining_prompts: list[str], collected_paths: list[str]
    ) -> list[str]:
        log(
            f"Falling back to Pollinations.ai for {len(remaining_prompts)} images",
            "INFO",
        )
        headers = {"Authorization": f"Bearer {API_KEY}"}

        for prompt in remaining_prompts:
            cache_key = hashlib.sha256(prompt.encode()).hexdigest()[:16]
            cached_path = IMAGE_CACHE_DIR / f"{cache_key}.jpg"
            if cached_path.exists():
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
                    collected_paths.append(str(cached_path))
                else:
                    raise Exception(f"HTTP {resp.status_code}")
            except Exception as e:
                log(f"  AI image error: {e}", "ERROR")
                raise

        return collected_paths
