import hashlib
import random
import shutil
import tempfile
import time
import traceback
from pathlib import Path
from urllib.parse import quote

import moviepy.video.fx as vfx
import numpy as np
import requests
from moviepy import (
    AudioFileClip,
    CompositeVideoClip,
    ImageClip,
    VideoClip,
    concatenate_videoclips,
)

from ..modules import (
    API_KEY,
    API_TIMEOUT_IMAGE,
    AUDIO_CODEC,
    CROSSFADE_TIME,
    ENCODING_CRF,
    ENCODING_PRESET,
    ENCODING_THREADS,
    IMAGE_BOUNCE_INTERVAL,
    IMAGE_CACHE_DIR,
    IMG_URL,
    MAX_ZOOM_VALUE,
    MODEL_IMAGE,
    OPENING_ANIMATION_DURATION,
    OPENING_ANIMATION_ENABLED,
    OUTPUT_DIR,
    SCENE_DURATION_SECONDS,
    START_FADE,
    VIDEO_CODEC,
    VIDEO_FPS,
    VIDEO_HEIGHT,
    VIDEO_WIDTH,
    ImageSearcher,
    ScriptGenerator,
    SubtitleSystem,
    TTSSystem,
    VideoBackgroundManager,
    VideoCompositor,
    VideoMetadata,
    create_temp_dir,
    log,
    setup_directories,
)


class ExplainerGenerator:
    def __init__(
        self,
        *,
        subject: str | None = None,
        output: str = "output",
        youtube_url: str | None = None,
        web_search: bool = True,
        no_images: bool = False,
        images_only: bool = False,
        image_source: str = "web",
        custom_instructions: str | None = None,
        metadata: VideoMetadata | None = None,
    ):
        self.subject = subject
        self.output = output
        self.youtube_url = youtube_url
        self.web_search = web_search
        self.no_images = no_images
        self.images_only = images_only
        self.image_source = image_source
        self.custom_instructions = custom_instructions
        self.metadata = metadata or VideoMetadata()

        self.script_generator = ScriptGenerator(
            web_search=web_search,
            custom_instructions=custom_instructions,
        )
        self.tts_system = TTSSystem()
        self.temp_dir = create_temp_dir()

    async def generate(self) -> bool:
        if self.images_only:
            return await self._run_images_only_pipeline()
        return await self._run_normal_pipeline()

    # ── Normal Pipeline (YouTube background + optional AI overlays) ──────────

    async def _run_normal_pipeline(self) -> bool:
        log("Starting explainer video (normal mode)...")
        start_time = time.time()
        subject = self.subject

        try:
            setup_directories()

            log("Step 1: Downloading YouTube video...")
            video_bg = VideoBackgroundManager()
            if self.youtube_url:
                video_path, title, _ = video_bg.download_from_url(self.youtube_url)
                log(f"Downloaded video: {title[:50]}...")
            else:
                assert subject is not None
                video_path = video_bg.search_and_download(subject)
                title = Path(video_path).stem if video_path else "Unknown"
                log(f"Downloaded video: {title[:50]}...")

            log("Step 2: Generating script...")
            if subject:
                script = self.script_generator.generate_script(subject)
            else:
                script = self.script_generator.generate_script_from_metadata(title, "")
            if not script or len(script) < 3:
                log(
                    f"Script generation failed: got {len(script)} paragraphs, need >= 3",
                    "ERROR",
                )
                return False
            log(f"Generated script with {len(script)} paragraphs")
            self.metadata.description = " ".join(script)

            log("Step 3: Generating TTS audio...")
            (
                audio_path,
                vtt_path,
                duration,
            ) = await self.tts_system.generate_audio_and_subtitles(
                script, self.temp_dir
            )
            log(f"Generated TTS audio: {duration:.1f}s")

            image_paths = None
            if self.no_images:
                log("Skipping AI image generation (--no-images)")
            else:
                num_images = max(3, int(duration / IMAGE_BOUNCE_INTERVAL) + 1)
                log(f"Step 4: Generating {num_images} images...")
                image_paths = self._generate_ai_images(subject, script, num_images)
                if len(image_paths) < 3:
                    log("Failed to generate enough images", "ERROR")
                    return False

            log("Step 5: Creating video...")
            compositor = VideoCompositor()
            if compositor.create_output_video(
                video_path,
                audio_path,
                vtt_path,
                self.output,
                duration,
                use_blurred_bg=True,
                image_paths=image_paths,
                metadata=self.metadata,
            ):
                elapsed = time.time() - start_time
                title = getattr(self.script_generator, "generated_title", None)
                if title:
                    log(f"Title: {title}", "SUCCESS")
                log(f"Video completed in {elapsed:.2f}s", "SUCCESS")
                log(f"Saved: {self.output}", "SUCCESS")
                return True

            log("Video processing failed", "ERROR")
            return False

        except Exception as e:
            log(f"Error: {e}", "ERROR")
            traceback.print_exc()
            return False

    def _generate_ai_images(
        self,
        subject: str | None,
        script_paragraphs: list,
        num_images: int = 0,
        prompts: list | None = None,
    ) -> list:
        if prompts is not None and len(prompts) > 0:
            if isinstance(prompts[0], dict):
                paired = prompts
            else:
                if self.image_source == "web":
                    # Plain strings are already-chosen subject queries: in web
                    # mode they go through the DDGS searcher with its
                    # provenance/fit gates (and its IMAGE_AI_FALLBACK=off
                    # abort), never straight to the metered AI API. The AI
                    # path stays for an explicitly selected image_source=ai.
                    return ImageSearcher().get_images(prompts)
                return self._call_pollinations_api(prompts)
        else:
            paired = self.script_generator.generate_image_prompts_from_script(
                script_paragraphs or [subject], num_images
            )

        if self.image_source == "web":
            queries = [p["web_query"] for p in paired]
            searcher = ImageSearcher()
            return searcher.get_images(queries)

        ai_prompts = [p["ai_prompt"] for p in paired]
        return self._call_pollinations_api(ai_prompts)

    def _call_pollinations_api(self, prompts: list[str]) -> list:
        img_paths = []
        headers = {"Authorization": f"Bearer {API_KEY}"}
        IMAGE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        total = len(prompts)

        for i, prompt in enumerate(prompts):
            cache_key = hashlib.sha256(
                f"ai1|{MODEL_IMAGE}|{prompt}".encode()
            ).hexdigest()[:16]
            cached_path = IMAGE_CACHE_DIR / f"{cache_key}.jpg"
            if cached_path.exists():
                img_paths.append(str(cached_path))
                log(f"Using cached image {i + 1}: {cached_path.name}")
                continue

            log(f"Generating image {i + 1}/{total}...")
            try:
                safe_prompt = quote(prompt)
                seed = random.randint(0, 999999)
                url = f"{IMG_URL}{safe_prompt}?model={MODEL_IMAGE}&width=1080&height=1920&seed={seed}&nologo=true"
                resp = requests.get(url, headers=headers, timeout=API_TIMEOUT_IMAGE)
                if resp.status_code == 200:
                    with open(cached_path, "wb") as f:
                        f.write(resp.content)
                    img_paths.append(str(cached_path))
                else:
                    raise Exception(f"Failed: {resp.status_code}")
            except Exception as e:
                log(f"Image error: {e}", "ERROR")
                raise

        return img_paths

    # ── Images-Only Pipeline (Flux AI images + zoom/crossfade) ───────────────

    async def _run_images_only_pipeline(self) -> bool:
        log("Starting explainer video (images-only mode)...")
        start_time = time.time()

        try:
            setup_directories()
            OUTPUT_DIR.mkdir(exist_ok=True)

            log("Step 1: Generating script with image prompts...")
            assert self.subject is not None
            paragraphs, _ = self.script_generator.generate_script_with_prompts(
                self.subject
            )
            if not paragraphs or len(paragraphs) < 3:
                log(
                    f"Script generation failed: got {len(paragraphs)} paragraphs, need >= 3",
                    "ERROR",
                )
                return False
            self.metadata.description = " ".join(paragraphs)

            log("Step 2: Generating TTS audio...")
            audio_path = await self.tts_system.generate_audio_only(
                paragraphs, self.temp_dir
            )

            log("Step 3: Calculating image count from TTS duration...")
            audio = AudioFileClip(audio_path)
            num_images = max(3, int(audio.duration / SCENE_DURATION_SECONDS))
            log(f"TTS: {audio.duration:.1f}s -> {num_images} images")

            log("Step 4: Generating image prompts from script...")
            image_prompts = self.script_generator.generate_image_prompts_from_script(
                paragraphs, num_images
            )

            log("Step 5: Generating images...")
            assert self.subject is not None
            img_paths = self._generate_ai_images(
                self.subject, paragraphs, prompts=image_prompts
            )
            if len(img_paths) < 3:
                log(f"Only {len(img_paths)} images, need >= 3", "ERROR")
                return False

            log("Step 6: Composing video...")
            self._create_flux_video(
                img_paths, audio_path, paragraphs, self.output, self.metadata
            )

            elapsed = time.time() - start_time
            title = getattr(self.script_generator, "generated_title", None)
            if title:
                log(f"Title: {title}", "SUCCESS")
            log(f"Video completed in {elapsed:.2f}s", "SUCCESS")
            log(f"Saved: {self.output}", "SUCCESS")
            return True

        except Exception as e:
            log(f"Error: {e}", "ERROR")
            traceback.print_exc()
            return False

    @staticmethod
    def _ease_in_out_cubic(t: float) -> float:
        if t <= 0:
            return 0.0
        if t >= 1:
            return 1.0
        if t < 0.5:
            return 4 * t * t * t
        return 1 - pow(-2 * t + 2, 3) / 2

    @classmethod
    def _apply_overlay_animation(cls, clip, duration: float):
        def scale_anim(t: float) -> float:
            if duration <= 0:
                return 1.0
            progress = t / duration
            fade_in_end = 0.15
            fade_out_start = 0.85
            if progress < fade_in_end:
                return 1.0 + (MAX_ZOOM_VALUE - 1.0) * cls._ease_in_out_cubic(
                    progress / fade_in_end
                )
            elif progress < fade_out_start:
                return MAX_ZOOM_VALUE
            else:
                out_progress = (progress - fade_out_start) / (1.0 - fade_out_start)
                return MAX_ZOOM_VALUE - (MAX_ZOOM_VALUE - 1.0) * cls._ease_in_out_cubic(
                    out_progress
                )

        def opacity_anim(t: float) -> float:
            if duration <= 0:
                return 1.0
            progress = t / duration
            fade_in_end = 0.15
            fade_out_start = 0.85
            if progress < fade_in_end:
                return cls._ease_in_out_cubic(progress / fade_in_end)
            elif progress < fade_out_start:
                return 1.0
            else:
                out_progress = (progress - fade_out_start) / (1.0 - fade_out_start)
                return max(1.0 - cls._ease_in_out_cubic(out_progress), 0.0)

        # Fade through an alpha mask, never by multiplying RGB: that yields an
        # opaque near-black frame and the "faded" image covered the previous
        # scene with black (issue #12). The mask is attached BEFORE the zoom so
        # Resize scales mask and picture together.
        width, height = clip.size
        base_mask = clip.mask if isinstance(clip.mask, VideoClip) else None

        def _opacity_frame(t):
            fade = opacity_anim(t)
            if base_mask is not None:
                return np.asarray(base_mask.get_frame(t), dtype=float) * fade
            return np.full((height, width), fade, dtype=float)

        mask = VideoClip(
            frame_function=_opacity_frame,
            is_mask=True,
            duration=getattr(clip, "duration", None),
        )
        clip = clip.with_mask(mask)
        clip = clip.with_effects([vfx.Resize(scale_anim)])
        return clip.with_position(("center", "center"))

    @classmethod
    def compose_saved_inputs(
        cls,
        image_paths: list[str | Path] | tuple[str | Path, ...],
        audio_path: str | Path,
        paragraphs: list[str],
        output_path: str | Path,
        metadata: VideoMetadata | None = None,
    ):
        """Compose a video from pre-saved and validated local assets.

        Scope and Boundary:
            Offline video composition only. This classmethod bypasses the
            generator constructor, LLM script generation, and TTS synthesis.
            The caller owns the network namespace (e.g. sandbox, network
            isolation, zero network egress) and durable asset hashing/integrity
            checks.

        Args:
            image_paths: Non-empty list or tuple of paths to non-empty,
                regular local image files.
            audio_path: Path to a non-empty, regular local audio file.
            paragraphs: Non-empty list of non-empty string paragraphs.
            output_path: Path where the final video will be written.
            metadata: Optional VideoMetadata instance.

        Returns:
            The return value of cls._compose_flux_body.

        Raises:
            ValueError: If inputs fail validation (empty sequence, non-regular
                file, 0-byte file, non-string paragraph, etc.).
            FileNotFoundError: If an image or audio file does not exist.
        """
        if not isinstance(image_paths, (list, tuple)) or len(image_paths) == 0:
            raise ValueError(
                "image_paths must be a non-empty list or tuple of file paths"
            )

        resolved_images: list[str] = []
        for i, img in enumerate(image_paths):
            if img is None:
                raise ValueError(f"Image path at index {i} cannot be None")
            if isinstance(img, str) and not img.strip():
                raise ValueError(
                    f"Image path at index {i} cannot be empty or whitespace"
                )
            if not isinstance(img, (str, Path)):
                raise ValueError(
                    f"Image path at index {i} must be a str or Path, got {type(img).__name__}"
                )
            img_p = Path(img)
            if not img_p.exists():
                raise FileNotFoundError(f"Image file does not exist: {img}")
            if not img_p.is_file():
                raise ValueError(f"Image path is not a regular file: {img}")
            if img_p.stat().st_size == 0:
                raise ValueError(f"Image file is empty (0 bytes): {img}")
            resolved_images.append(str(img_p))

        if not audio_path:
            raise ValueError("audio_path must be specified")
        if not isinstance(audio_path, (str, Path)):
            raise ValueError(
                f"audio_path must be a str or Path, got {type(audio_path).__name__}"
            )
        if isinstance(audio_path, str) and not audio_path.strip():
            raise ValueError("audio_path cannot be empty or whitespace")
        audio_p = Path(audio_path)
        if not audio_p.exists():
            raise FileNotFoundError(f"Audio file does not exist: {audio_path}")
        if not audio_p.is_file():
            raise ValueError(f"Audio path is not a regular file: {audio_path}")
        if audio_p.stat().st_size == 0:
            raise ValueError(f"Audio file is empty (0 bytes): {audio_path}")
        resolved_audio = str(audio_p)

        if not isinstance(paragraphs, list) or len(paragraphs) == 0:
            raise ValueError("paragraphs must be a non-empty list of strings")
        for i, p in enumerate(paragraphs):
            if not isinstance(p, str) or not p.strip():
                raise ValueError(f"Paragraph at index {i} must be a non-empty string")

        if not output_path:
            raise ValueError("output_path must be specified")
        if not isinstance(output_path, (str, Path)):
            raise ValueError(
                f"output_path must be a str or Path, got {type(output_path).__name__}"
            )
        if isinstance(output_path, str) and not output_path.strip():
            raise ValueError("output_path cannot be empty or whitespace")
        resolved_output = str(Path(output_path))

        return cls._compose_flux_body(
            resolved_images,
            resolved_audio,
            paragraphs,
            resolved_output,
            metadata=metadata,
            overlay_animation=cls._apply_overlay_animation,
        )

    def _create_flux_video(
        self,
        img_paths: list,
        audio_path: str,
        paragraphs: list,
        output_path: str,
        metadata: VideoMetadata | None = None,
    ):
        return self._compose_flux_body(
            img_paths,
            audio_path,
            paragraphs,
            output_path,
            metadata=metadata,
            overlay_animation=self._apply_overlay_animation,
        )

    @classmethod
    def _compose_flux_body(
        cls,
        img_paths: list,
        audio_path: str,
        paragraphs: list,
        output_path: str,
        metadata: VideoMetadata | None = None,
        overlay_animation=None,
    ):
        if not img_paths:
            raise ValueError("Cannot compose video: no image paths provided")
        log("Composing video with U-curve zoom...")
        audio = AudioFileClip(audio_path)
        num_imgs = len(img_paths)
        # Redistribute the audio duration across the scenes so the composed
        # clip (N scenes overlapped by CROSSFADE_TIME each) ends exactly with
        # the audio; a fixed scene duration always fell short and the branch
        # below padded the remainder with black frames (issue #31).
        dur_per_img = (audio.duration - CROSSFADE_TIME) / num_imgs + CROSSFADE_TIME
        total_video_dur = num_imgs * dur_per_img
        log(f"{num_imgs} scenes x {dur_per_img}s = {total_video_dur:.1f}s")

        overlay_fn = (
            overlay_animation
            if overlay_animation is not None
            else cls._apply_overlay_animation
        )

        clips = []
        compositor = VideoCompositor()
        for i, path in enumerate(img_paths):
            clip = (
                ImageClip(path)
                .with_duration(dur_per_img)
                .resized(new_size=(1080, 1920))
            )
            if i == 0 and OPENING_ANIMATION_ENABLED:
                clip = compositor._apply_opening_animation(
                    clip, OPENING_ANIMATION_DURATION
                )
            else:
                clip = overlay_fn(clip, dur_per_img)
            if i > 0:
                clip = clip.with_effects([vfx.CrossFadeIn(CROSSFADE_TIME)])
            if i == 0:
                clip = clip.with_effects([vfx.FadeIn(START_FADE)])
            clips.append(clip)

        video = concatenate_videoclips(clips, method="compose", padding=-CROSSFADE_TIME)

        if abs(video.duration - audio.duration) > 1.0:
            if video.duration < audio.duration:
                video = video.with_duration(audio.duration)
            else:
                video = video.subclipped(0, audio.duration)

        video = video.with_audio(audio)

        subtitle_system = SubtitleSystem()
        vtt_dir = tempfile.mkdtemp()
        vtt_path = subtitle_system.generate_subtitles(
            paragraphs, video.duration, vtt_dir
        )
        subs = subtitle_system.render_subtitles(vtt_path, (VIDEO_WIDTH, VIDEO_HEIGHT))

        final = CompositeVideoClip([video] + subs, size=(VIDEO_WIDTH, VIDEO_HEIGHT))
        meta_args = metadata.to_ffmpeg_args() if metadata else []
        final.write_videofile(
            output_path,
            fps=VIDEO_FPS,
            codec=VIDEO_CODEC,
            audio_codec=AUDIO_CODEC,
            threads=ENCODING_THREADS,
            preset=ENCODING_PRESET,
            ffmpeg_params=["-crf", str(ENCODING_CRF)] + meta_args,
        )

        final.close()
        audio.close()
        for c in clips:
            c.close()

    def cleanup(self):
        if hasattr(self, "temp_dir") and self.temp_dir.exists():
            try:
                shutil.rmtree(self.temp_dir)
            except Exception:
                pass
