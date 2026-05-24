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
    ):
        self.subject = subject
        self.output = output
        self.youtube_url = youtube_url
        self.web_search = web_search
        self.no_images = no_images
        self.images_only = images_only
        self.image_source = image_source

        self.script_generator = ScriptGenerator(web_search=web_search)
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
            log(f"Generated script with {len(script)} paragraphs")

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
            cache_key = hashlib.sha256(prompt.encode()).hexdigest()[:16]
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
            self._create_flux_video(img_paths, audio_path, paragraphs, self.output)

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

    def _ease_in_out_cubic(self, t: float) -> float:
        if t <= 0:
            return 0.0
        if t >= 1:
            return 1.0
        if t < 0.5:
            return 4 * t * t * t
        return 1 - pow(-2 * t + 2, 3) / 2

    def _apply_overlay_animation(self, clip, duration: float):
        def scale_anim(t: float) -> float:
            if duration <= 0:
                return 1.0
            progress = t / duration
            fade_in_end = 0.15
            fade_out_start = 0.85
            if progress < fade_in_end:
                return 1.0 + (MAX_ZOOM_VALUE - 1.0) * self._ease_in_out_cubic(
                    progress / fade_in_end
                )
            elif progress < fade_out_start:
                return MAX_ZOOM_VALUE
            else:
                out_progress = (progress - fade_out_start) / (1.0 - fade_out_start)
                return MAX_ZOOM_VALUE - (
                    MAX_ZOOM_VALUE - 1.0
                ) * self._ease_in_out_cubic(out_progress)

        def opacity_anim(t: float) -> float:
            if duration <= 0:
                return 1.0
            progress = t / duration
            fade_in_end = 0.15
            fade_out_start = 0.85
            if progress < fade_in_end:
                return self._ease_in_out_cubic(progress / fade_in_end)
            elif progress < fade_out_start:
                return 1.0
            else:
                out_progress = (progress - fade_out_start) / (1.0 - fade_out_start)
                return max(1.0 - self._ease_in_out_cubic(out_progress), 0.0)

        def apply_opacity(get_frame, t):
            frame = get_frame(t)
            opacity = opacity_anim(t)
            return np.minimum(255, frame * opacity).astype("uint8")

        clip = clip.with_effects([vfx.Resize(scale_anim)])
        clip = clip.transform(apply_opacity)
        return clip.with_position(("center", "center"))

    def _create_flux_video(
        self, img_paths: list, audio_path: str, paragraphs: list, output_path: str
    ):
        log("Composing video with U-curve zoom...")
        audio = AudioFileClip(audio_path)
        num_imgs = len(img_paths)
        dur_per_img = SCENE_DURATION_SECONDS
        total_video_dur = num_imgs * dur_per_img
        log(f"{num_imgs} scenes x {dur_per_img}s = {total_video_dur:.1f}s")

        clips = []
        for i, path in enumerate(img_paths):
            clip = (
                ImageClip(path)
                .with_duration(dur_per_img)
                .resized(new_size=(1080, 1920))
            )
            clip = self._apply_overlay_animation(clip, dur_per_img)
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
        final.write_videofile(
            output_path,
            fps=VIDEO_FPS,
            codec=VIDEO_CODEC,
            audio_codec=AUDIO_CODEC,
            threads=ENCODING_THREADS,
            preset=ENCODING_PRESET,
            ffmpeg_params=["-crf", str(ENCODING_CRF)],
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
