import hashlib
import json
import os
import random
import re
import tempfile
from pathlib import Path
from urllib.parse import quote

import requests
from PIL import Image

from . import image_content, image_sources, image_subject
from .config import (
    ALLOW_UNVERIFIED_IMAGES,
    API_KEY,
    API_TIMEOUT_IMAGE,
    IMAGE_AI_FALLBACK,
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


# Bump when acceptance rules change so older cache entries are refetched.
ACCEPTANCE_POLICY = "specific-requests-fall-back-v1"


class _ProvenancePublishError(RuntimeError):
    """A web candidate could not be committed with its exact provenance."""


class ImageSearcher:
    def __init__(
        self,
        target_width: int = TARGET_IMAGE_WIDTH,
        target_height: int = TARGET_IMAGE_HEIGHT,
        safesearch: str = IMAGE_SEARCH_SAFE,
        min_width: int = IMAGE_MIN_WIDTH,
        min_height: int = IMAGE_MIN_HEIGHT,
        max_per_query: int = IMAGE_SEARCH_PER_QUERY,
        search=None,
        download=None,
        providers=None,
        ai_fallback=None,
    ):
        if search is not None and not callable(search):
            raise TypeError("search must be callable or None")
        if download is not None and not callable(download):
            raise TypeError("download must be callable or None")
        if providers is not None:
            if isinstance(providers, str) or not isinstance(providers, (tuple, list)):
                raise TypeError("providers must be a tuple of str")
            if any(not isinstance(p, str) for p in providers):
                raise TypeError("providers must contain only str")
            providers = tuple(providers)
        if ai_fallback is not None and not isinstance(ai_fallback, bool):
            raise TypeError("ai_fallback must be bool or None")

        self.target_width = target_width
        self.target_height = target_height
        self.safesearch = safesearch
        self.min_width = min_width
        self.min_height = min_height
        self.max_per_query = max_per_query
        self._search = search
        self._download = download
        self._providers = providers
        self._ai_fallback = ai_fallback
        IMAGE_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    @property
    def providers(self) -> tuple[str, ...]:
        if getattr(self, "_providers", None) is None:
            return tuple(IMAGE_PROVIDERS)
        return tuple(self._providers)

    @providers.setter
    def providers(self, value):
        self._providers = None if value is None else tuple(value)

    @property
    def ai_fallback(self) -> bool:
        if getattr(self, "_ai_fallback", None) is None:
            return bool(IMAGE_AI_FALLBACK)
        return bool(self._ai_fallback)

    @ai_fallback.setter
    def ai_fallback(self, value):
        self._ai_fallback = value

    @property
    def _public_only(self) -> bool:
        """This instance was given injected saved/public search/download callables.

        The saved/public-only wrapper builds a searcher this way. An injected
        callback says where results come from, not that it is harmless: nothing
        here inspects or vouches for what an injected callable may do. Since
        such a run has no provider route of its own to refetch from, a local
        verification problem is fail-closed instead of a miss.
        """
        return (
            getattr(self, "_search", None) is not None
            or getattr(self, "_download", None) is not None
        )

    def _instance_cache_key(self, prompt: str) -> str:
        if (
            getattr(self, "_providers", None) is None
            and getattr(self, "_ai_fallback", None) is None
            and getattr(self, "_search", None) is None
            and getattr(self, "_download", None) is None
        ):
            return ImageSearcher._cache_key(prompt)
        policy = "optin" if ALLOW_UNVERIFIED_IMAGES else "strict"
        scheme = (
            f"img3|{ACCEPTANCE_POLICY}|{policy}|"
            f"{','.join(self.providers)}|ai={int(bool(self.ai_fallback))}|{prompt}"
        )
        return hashlib.sha256(scheme.encode()).hexdigest()[:16]

    def _instance_usable_cache(self, cached_path: Path, prompt: str) -> bool:
        record = ImageSearcher._read_sidecar(cached_path)
        if record is not None and record.get("kind") == "ai":
            return bool(cached_path.exists() and self.ai_fallback)
        return ImageSearcher._usable_cache(cached_path, prompt)

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
        search = getattr(self, "_search", None)
        if search is not None:
            return list(search(query))
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
        download = getattr(self, "_download", None)
        if download is not None:
            return bool(download(url, cache_path))
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
        # The acceptance policy is part of the key: entries made under the
        # ALLOW_UNVERIFIED_IMAGES opt-in, or under an older/laxer policy
        # version, are never served to a stricter run.
        policy = "optin" if ALLOW_UNVERIFIED_IMAGES else "strict"
        scheme = (
            f"img3|{ACCEPTANCE_POLICY}|{policy}|{','.join(IMAGE_PROVIDERS)}|{prompt}"
        )
        return hashlib.sha256(scheme.encode()).hexdigest()[:16]

    @staticmethod
    def _sidecar_path(image_path: Path) -> Path:
        return image_path.with_suffix(".credits.json")

    @classmethod
    def _read_sidecar(cls, image_path: Path) -> dict | None:
        """Parsed provenance record, or None unless it is complete and valid.

        An empty, unparsable or incomplete sidecar is NOT provenance: the
        image is treated as having none."""
        try:
            data = json.loads(cls._sidecar_path(image_path).read_text("utf-8"))
        except Exception:
            return None
        if not isinstance(data, dict) or data.get("schema") != 1:
            return None
        kind = data.get("kind")
        if kind == "ai":
            return data if data.get("license") and data.get("provider") else None
        if kind != "web":
            return None
        required = ("title", "source_page", "image_url", "license", "provider")
        unknown_license = (
            data.get("license") == ""
            and data.get("license_verified") is False
            and data.get("unverified_opt_in") is True
        )
        if not all(
            isinstance(data.get(k), str)
            and (data[k].strip() or (k == "license" and unknown_license))
            for k in required
        ):
            return None
        for key in ("source_page", "image_url"):
            if not data[key].startswith(("http://", "https://")):
                return None
        if not isinstance(data.get("license_verified"), bool):
            return None
        return data

    @staticmethod
    def _pixel_gate(path: Path, prompt: str) -> tuple[str | None, str | None]:
        """Pixel gate result as ``(rejection, unavailability)``.

        ``rejection`` is the same reason string ``_content_rejection``
        reports. ``unavailability`` is set only when the OCR gate itself could
        not run (``image_content.ContentUnavailableError``), which is NOT a
        verdict about the pixels: callers that cannot reacquire anything must
        fail closed on it instead of treating the bytes as checked.
        """
        try:
            reason = image_content.check_image_content(path, prompt)
        except image_content.ContentUnavailableError as exc:
            return None, f"pixel content could not be checked ({exc})"
        if reason or ALLOW_UNVERIFIED_IMAGES:
            return reason, None
        try:
            # Broad-class similarity only (see image_subject.VERIFICATION_SCOPE).
            return image_subject.check_subject(path, prompt), None
        except image_subject.SubjectUnavailableError as exc:
            return f"pictured subject could not be verified ({exc})", None

    @staticmethod
    def _content_rejection(path: Path, prompt: str) -> str | None:
        """Pixel gate on the downloaded bytes; fails closed if OCR is missing."""
        reason, unavailable = ImageSearcher._pixel_gate(path, prompt)
        return reason or unavailable

    def _cached_pixel_gate(
        self, cached_path: Path, prompt: str
    ) -> tuple[bool, str | None]:
        """Re-check the pixels of an already-allowed cache hit.

        Returns ``(usable, unavailability)``. A rejected entry is only a cache
        MISS: its bytes and sidecar are left exactly as they are so the normal
        candidate flow can republish them, and nothing is deleted here. AI
        cache entries are not OCR-gated - this gate is about web pixels.
        """
        record = ImageSearcher._read_sidecar(cached_path)
        if record is not None and record.get("kind") == "ai":
            return True, None
        reason, unavailable = ImageSearcher._pixel_gate(cached_path, prompt)
        return reason is None and unavailable is None, unavailable

    @classmethod
    def _usable_cache(cls, cached_path: Path, prompt: str) -> bool:
        """Reusable only with a valid provenance record that is still allowed
        (legacy stock, interrupted runs, empty/UNVERIFIED records are
        refetched; so are AI images while the AI fallback is disabled)."""
        if not cached_path.exists():
            return False
        record = cls._read_sidecar(cached_path)
        if record is None:
            return False
        if record["kind"] == "ai":
            # Cached AI bytes are reusable only while the AI fallback that
            # produced them is allowed: IMAGE_AI_FALLBACK=off fails closed
            # here instead of serving them through the web route. An
            # explicitly selected image_source=ai run caches under its own
            # scheme and never passes through this gate - a chosen source is
            # not a fallback.
            return IMAGE_AI_FALLBACK
        if record["kind"] == "web":
            if not ALLOW_UNVERIFIED_IMAGES:
                # A cache hit must pass the same specificity gate as a fresh
                # download, and an opt-in entry is never reused without opt-in.
                if record.get(
                    "unverified_opt_in"
                ) or not image_subject.is_broad_request(prompt):
                    return False
            if not record["license_verified"]:
                return ALLOW_UNVERIFIED_IMAGES
        return True

    def get_images(self, prompts: list[str]) -> list[str]:
        img_paths = []
        total = len(prompts)

        for i, prompt in enumerate(prompts):
            log(f"Searching web image {i + 1}/{total}: '{prompt[:60]}...'", "INFO")

            cache_key = self._instance_cache_key(prompt)
            cached_path = IMAGE_CACHE_DIR / f"{cache_key}.jpg"
            if self._instance_usable_cache(cached_path, prompt):
                usable, unavailable = self._cached_pixel_gate(cached_path, prompt)
                if unavailable is not None and self._public_only:
                    # Nothing can be reacquired here, so an unreadable gate is
                    # fail-closed - before any refetch/network, and with the
                    # seeded cache bytes and sidecar left untouched.
                    raise RuntimeError(
                        f"Prompt '{prompt}' cached image could not be verified: "
                        f"{unavailable}"
                    )
                if usable:
                    img_paths.append(str(cached_path))
                    log(f"  Using cached: {cached_path.name}", "INFO")
                    continue
                # A rejected cache entry is a MISS, never a reason to delete
                # the shared/published file: the candidate flow below
                # reacquires and republishes it.
                detail = f" ({unavailable})" if unavailable else ""
                log(
                    f"  Cached image failed the pixel content check{detail}; "
                    "refetching",
                    "WARNING",
                )

            candidates = self._candidates(prompt)
            if candidates and not ALLOW_UNVERIFIED_IMAGES:
                # No free local check can verify a specific person, team, model,
                # year or event in a picture; only broad classes are acceptable.
                if not image_subject.is_broad_request(prompt):
                    log(
                        "  Request names a specific entity/date/event; web images "
                        "cannot be verified, using the fallback instead",
                        "WARNING",
                    )
                    candidates = []
            downloaded = False
            for candidate in candidates:
                url = candidate.url
                if not url:
                    continue
                # Without the opt-in a web image needs verified reuse rights AND
                # a passing pictured-subject check on the downloaded pixels.
                if not candidate.license and not ALLOW_UNVERIFIED_IMAGES:
                    log(
                        f"  Skipping {candidate.provider} image without verified "
                        "reuse rights (set ALLOW_UNVERIFIED_IMAGES=1 to opt in)",
                        "WARNING",
                    )
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
                        reason = self._content_rejection(temp_path, prompt)
                        if reason:
                            log(
                                f"  Rejected by pixel content check: {reason}",
                                "WARNING",
                            )
                            temp_path.unlink(missing_ok=True)
                            continue
                        # Transformation (text removal, framing, resize) can
                        # change what the pixels say, so the FINAL bytes are
                        # gated too - before the path is accepted and before
                        # any provenance sidecar is published. The transform is
                        # staged in a fresh uniquely-named file of this
                        # invocation: the shared/published cache entry is
                        # replaced only after those exact bytes pass, so a
                        # rejection (or any transform/gate/replace failure)
                        # leaves seeded bytes and sidecar exactly as they were.
                        staged_fd, staged_name = tempfile.mkstemp(
                            dir=IMAGE_CACHE_DIR,
                            prefix=(
                                f"dl_{cache_key}_"
                                f"{hashlib.md5(url.encode()).hexdigest()[:8]}_"
                            ),
                            suffix="_final.jpg",
                        )
                        os.close(staged_fd)
                        staged_path = Path(staged_name)
                        staged_sidecar = ImageSearcher._sidecar_path(staged_path)
                        try:
                            self.resize_to_fill(temp_path, staged_path)
                            temp_path.unlink(missing_ok=True)
                            final_reason, final_unavailable = self._pixel_gate(
                                staged_path, prompt
                            )
                            problem = final_reason or final_unavailable
                            if problem is not None:
                                log(
                                    "  Final image rejected by pixel content "
                                    f"check: {problem}",
                                    "WARNING",
                                )
                                continue
                            self._publish_candidate_pair(
                                staged_path,
                                staged_sidecar,
                                cached_path,
                                candidate,
                                prompt,
                            )
                        finally:
                            # The image staging file is ours. The publisher owns
                            # cleanup of a sidecar it successfully creates, while
                            # its exclusive create leaves any collision untouched.
                            staged_path.unlink(missing_ok=True)
                        img_paths.append(str(cached_path))
                        log(
                            f"  Downloaded & processed: {cached_path.name} "
                            f"({candidate.provider}, license: "
                            f"{candidate.license or 'unverified'}, "
                            f"fit: {candidate.fit_evidence})",
                            "INFO",
                        )
                        downloaded = True
                        break
                    except _ProvenancePublishError:
                        temp_path.unlink(missing_ok=True)
                        raise
                    except Exception as e:
                        log(f"  Resize failed: {e}", "WARNING")
                        temp_path.unlink(missing_ok=True)

            if not downloaded:
                if not self.ai_fallback:
                    log(
                        f"  No image for '{prompt[:60]}': every web candidate "
                        "failed and the AI fallback is disabled "
                        "(IMAGE_AI_FALLBACK=off); aborting before render",
                        "ERROR",
                    )
                    raise RuntimeError(
                        f"Prompt '{prompt}' got no image: web candidates were "
                        "exhausted and IMAGE_AI_FALLBACK is disabled"
                    )
                log(
                    "  Web search failed for prompt, generating via AI instead",
                    "WARNING",
                )
                self._fallback_to_ai([prompt], img_paths)

        self._write_credits_file(img_paths)
        return img_paths

    @classmethod
    def _provenance_record(
        cls, candidate: "image_sources.SourceCandidate", prompt: str
    ) -> dict:
        """Build the exact existing sidecar schema for one source candidate."""
        extra = candidate.extra or {}
        is_ai = candidate.provider == "ai"
        record = {
            "schema": 1,
            "kind": "ai" if is_ai else "web",
            "provider": candidate.provider,
            "prompt": prompt,
            "title": candidate.title or ("AI-generated image" if is_ai else ""),
            "image_url": candidate.url,
            "source_page": extra.get("source_page") or "",
            "creator": extra.get("creator") or "",
            "license": candidate.license or "",
            "license_url": extra.get("license_url") or "",
            "license_verified": bool(candidate.license) and not is_ai,
            "fit": candidate.fit_evidence,
            # Say exactly what was (not) verified; never "subject verified".
            "pictured_subject_check": (
                "not-run"
                if is_ai
                else "not-run (ALLOW_UNVERIFIED_IMAGES opt-in)"
                if ALLOW_UNVERIFIED_IMAGES
                else image_subject.VERIFICATION_SCOPE
            ),
            "unverified_opt_in": bool(ALLOW_UNVERIFIED_IMAGES and not is_ai),
            "modification": (
                "Resized and cropped to the video frame; overlay text may "
                "have been removed and framing adjusted."
            ),
        }
        if not is_ai and not record["source_page"]:
            # The image URL is the only locator some providers expose.
            record["source_page"] = candidate.url
        return record

    @classmethod
    def _write_provenance_sidecar(
        cls, image_path: Path, candidate: "image_sources.SourceCandidate", prompt: str
    ) -> None:
        """Persist provenance next to the cached image so credits survive cache hits.

        Legacy/AI callers retain best-effort writes. New web publication uses a
        separate strict staged path and will not accept bytes if it fails.
        """
        try:
            record = cls._provenance_record(candidate, prompt)
            cls._sidecar_path(image_path).write_text(
                json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except Exception as e:
            log(f"  Could not write provenance sidecar: {e}", "WARNING")

    @classmethod
    def _stage_provenance_sidecar(
        cls, staged_image: Path, candidate: "image_sources.SourceCandidate", prompt: str
    ) -> dict:
        """Write and validate a private sidecar for a uniquely staged image."""
        staged_image = Path(staged_image)
        staged_sidecar = cls._sidecar_path(staged_image)
        record = cls._provenance_record(candidate, prompt)
        serialized = json.dumps(record, ensure_ascii=False, indent=2)
        descriptor = None
        created = False
        try:
            descriptor = os.open(
                staged_sidecar,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
            )
            created = True
            stream = os.fdopen(descriptor, "w", encoding="utf-8")
            descriptor = None
            with stream:
                stream.write(serialized)
                stream.flush()
                os.fsync(stream.fileno())
            if cls._read_sidecar(staged_image) != record:
                raise OSError(
                    "staged provenance did not validate against its candidate"
                )
            return record
        except BaseException:
            if descriptor is not None:
                try:
                    os.close(descriptor)
                except OSError:
                    pass
            if created:
                try:
                    staged_sidecar.unlink(missing_ok=True)
                except OSError:
                    pass
            raise

    @staticmethod
    def _reserve_pair_backup(path: Path) -> Path:
        # Reserve an unguessable private directory, leaving the backup target
        # absent. A successful rename can then be distinguished from a failed
        # one during rollback without mistaking an empty placeholder for data.
        directory = Path(
            tempfile.mkdtemp(dir=path.parent, prefix=f".{path.name}.rollback-")
        )
        return directory / path.name

    @classmethod
    def _publish_cache_pair(
        cls,
        staged_image: Path,
        staged_sidecar: Path,
        cached_path: Path,
        expected_record: dict,
    ) -> None:
        """Commit bytes and their exact sidecar, restoring the old pair on error.

        The old sidecar is moved away before the cached image is replaced, so
        readers can observe a miss during publication but never new bytes with
        stale valid provenance.
        """
        cached_path = Path(cached_path)
        sidecar_path = cls._sidecar_path(cached_path)
        had_image = os.path.lexists(cached_path)
        had_sidecar = os.path.lexists(sidecar_path)
        backup_image = backup_sidecar = None
        moved_image = moved_sidecar = False
        installed_image = installed_sidecar = False
        committed = False

        try:
            if had_sidecar:
                backup_sidecar = cls._reserve_pair_backup(sidecar_path)
            if had_image:
                backup_image = cls._reserve_pair_backup(cached_path)

            if had_sidecar and backup_sidecar is not None:
                os.replace(sidecar_path, backup_sidecar)
                moved_sidecar = True
            if had_image and backup_image is not None:
                os.replace(cached_path, backup_image)
                moved_image = True

            os.replace(staged_image, cached_path)
            installed_image = True
            os.replace(staged_sidecar, sidecar_path)
            installed_sidecar = True

            if cls._read_sidecar(cached_path) != expected_record:
                raise OSError("published provenance does not match the candidate")
            committed = True
        except BaseException:
            # Reconcile flags if an asynchronous exception arrived just after a
            # successful atomic rename; only restore the old sidecar after the
            # old image is back, so rollback cannot create a stale pair.
            if backup_sidecar is not None and not os.path.lexists(sidecar_path):
                moved_sidecar = moved_sidecar or os.path.lexists(backup_sidecar)
            if backup_image is not None and not os.path.lexists(cached_path):
                moved_image = moved_image or os.path.lexists(backup_image)
            if not os.path.lexists(staged_image) and os.path.lexists(cached_path):
                installed_image = True
            if not os.path.lexists(staged_sidecar) and os.path.lexists(sidecar_path):
                installed_sidecar = True

            sidecar_removed = not installed_sidecar
            if installed_sidecar:
                try:
                    sidecar_path.unlink(missing_ok=True)
                    sidecar_removed = True
                except OSError:
                    log("  Could not remove candidate sidecar during rollback", "ERROR")
                    sidecar_removed = False

            image_removed = not installed_image
            if installed_image:
                try:
                    cached_path.unlink(missing_ok=True)
                    image_removed = True
                except OSError:
                    log("  Could not remove candidate image during rollback", "ERROR")
                    image_removed = False

            if not had_image:
                restored_image = image_removed
            elif moved_image:
                restored_image = False
                if image_removed and sidecar_removed and backup_image is not None:
                    try:
                        os.replace(backup_image, cached_path)
                        moved_image = False
                        restored_image = True
                    except OSError:
                        log("  Could not restore cached image during rollback", "ERROR")
            else:
                restored_image = not installed_image and os.path.lexists(cached_path)

            if (
                moved_sidecar
                and sidecar_removed
                and restored_image
                and backup_sidecar is not None
            ):
                try:
                    os.replace(backup_sidecar, sidecar_path)
                    moved_sidecar = False
                except OSError:
                    log(
                        "  Could not restore cached provenance during rollback", "ERROR"
                    )
            raise
        finally:
            for backup, retained in (
                (backup_image, moved_image),
                (backup_sidecar, moved_sidecar),
            ):
                if backup is not None and (committed or not retained):
                    try:
                        backup.unlink(missing_ok=True)
                        backup.parent.rmdir()
                    except OSError:
                        log("  Could not clean a cache publication backup", "WARNING")

    def _publish_candidate_pair(
        self,
        staged_image: Path,
        staged_sidecar: Path,
        cached_path: Path,
        candidate: "image_sources.SourceCandidate",
        prompt: str,
    ) -> None:
        """Fail closed unless this candidate's validated sidecar commits with its bytes."""
        staged_sidecar_created = False
        try:
            record = self._stage_provenance_sidecar(staged_image, candidate, prompt)
            staged_sidecar_created = True
            self._publish_cache_pair(staged_image, staged_sidecar, cached_path, record)
        except _ProvenancePublishError:
            raise
        except Exception as exc:
            raise _ProvenancePublishError(
                "candidate provenance could not be published with the image"
            ) from exc
        finally:
            if staged_sidecar_created:
                try:
                    staged_sidecar.unlink(missing_ok=True)
                except OSError:
                    log(
                        "  Could not clean candidate provenance staging file", "WARNING"
                    )

    @classmethod
    def _write_credits_file(cls, img_paths: list[str]) -> None:
        """Deliver per-image credits next to the video output.

        Reusable-license obligations (title, creator, source page, license and
        its URL, modification notice) come from validated sidecars only. Any
        image without a valid record, or without verified rights, makes the
        file start with a PUBLICATION REVIEW REQUIRED status.
        """
        try:
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            blocks: list[str] = []
            needs_review = 0
            for p in img_paths:
                record = cls._read_sidecar(Path(p))
                name = Path(p).name
                if record is None:
                    needs_review += 1
                    blocks.append(
                        f"{name}\nUNVERIFIED: no valid provenance record; "
                        "review before publishing"
                    )
                    continue
                if record["kind"] == "ai":
                    blocks.append(
                        f"{name}\nAI-generated ({record['provider']}); no "
                        "third-party source or license claimed."
                    )
                    continue
                if not record["license_verified"]:
                    needs_review += 1
                lines = [
                    name,
                    f"Title: {record['title']}",
                    f"Creator: {record.get('creator') or 'not provided by source'}",
                    f"Source: {record['source_page']}",
                    f"Image file: {record['image_url']}",
                    "License: "
                    + (
                        record["license"]
                        if record["license_verified"]
                        else "UNVERIFIED - reuse rights unknown"
                    ),
                ]
                if record.get("license_url"):
                    lines.append(f"License URL: {record['license_url']}")
                lines.append(f"Changes: {record['modification']}")
                blocks.append("\n".join(lines))
            status = (
                f"PUBLICATION REVIEW REQUIRED: {needs_review} image(s) without "
                "verified reuse rights"
                if needs_review
                else "Rights status: all web images carry a verified reusable license"
            )
            (OUTPUT_DIR / "image_credits.txt").write_text(
                status + "\n\n" + "\n\n".join(blocks) + "\n", encoding="utf-8"
            )
            if needs_review:
                log(status, "WARNING")
        except Exception as e:
            log(f"Could not write image credits file: {e}", "WARNING")

    def _candidates(self, query: str) -> list[image_sources.SourceCandidate]:
        """Provider-ordered candidates with provenance and fit evidence."""
        candidates: list[image_sources.SourceCandidate] = []
        for provider in self.providers:
            if provider == "commons":
                candidates.extend(self._commons_candidates(query))
            elif provider == "ddg":
                candidates.extend(self._ddg_candidates(query))
        return candidates

    def _ddg_candidates(self, query: str) -> list[image_sources.SourceCandidate]:
        """DDG candidates, plus an opt-in short-subject re-query (issue #37).

        Real photo titles never repeat a long scene prompt, so the full-query
        fit filter can reject every result. Under the ALLOW_UNVERIFIED_IMAGES
        opt-in only, DDG is re-queried with the prompt's distinctive subject
        tokens and fit evidence is evaluated against that short query: the
        photo must still prove the subject, and the recorded fit evidence
        names the short query it was actually matched to. An explicit
        "vs"/"versus"/"x" pair keeps its opponent even when first-N
        truncation would otherwise drop it. A retried title stating a number
        the FULL prompt does not contain (wrong year, score or edition) is
        rejected. Without the opt-in the behavior is unchanged: one query,
        same wipes, no retry."""
        raw = [r for r in self.search_images(query) if not self._is_nsfw(r)]
        matched = image_sources.filter_ddg_results(query, raw)
        if matched or not ALLOW_UNVERIFIED_IMAGES:
            return matched
        short = image_sources.retry_subject_query(query)
        if not short or short == query:
            return []
        log(
            "  No DDG result fit the full prompt; retrying with short subject "
            f"query '{short}' (ALLOW_UNVERIFIED_IMAGES opt-in)",
            "INFO",
        )
        raw = [r for r in self.search_images(short) if not self._is_nsfw(r)]
        retried = image_sources.filter_ddg_results(short, raw)
        # The short query carries no numbers, so fit_evidence cannot see a
        # wrong year, score or edition: reject retried candidates whose
        # title states a number the FULL prompt does not contain.
        kept = [
            c
            for c in retried
            if not image_sources.has_contradictory_number(query, c.title)
        ]
        if len(kept) < len(retried):
            log(
                f"  {len(retried) - len(kept)} retried candidate(s) rejected: "
                "title states a number the full prompt does not contain",
                "INFO",
            )
        for candidate in kept:
            candidate.fit_evidence = (
                f"short-query:'{short}' via {candidate.fit_evidence}"
            )
        return kept

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
            cache_key = self._instance_cache_key(prompt)
            cached_path = IMAGE_CACHE_DIR / f"{cache_key}.jpg"
            if self._instance_usable_cache(cached_path, prompt):
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
