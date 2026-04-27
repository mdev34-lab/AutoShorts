#!/usr/bin/env python3
"""
Subtitle System Module for AutoShorts

Handles subtitle generation and rendering for video processing.
Extracted from autoshorts_yt_summarizer.py for modular reuse.

Performance improvements:
- PIL-based text measurement (10-50x faster than TextClip)
- Cached text dimensions using functools.lru_cache
- Batch word positioning calculations
- Reduced clip creation overhead
"""

from functools import lru_cache
from pathlib import Path

import webvtt
from moviepy import TextClip
from PIL import Image, ImageDraw, ImageFont

from .config import (
    COLOR_HIGHLIGHT,
    COLOR_STROKE,
    COLOR_TEXT,
    FONT_SIZE,
    LINE_SPACING,
    MAX_CHARS_PER_LINE,
    STROKE_WIDTH,
    SUBTITLE_MODE,
    SUBTITLE_START_Y_RATIO,
)


def get_system_font():
    """Return a safe font for TextClip by searching Windows font registry."""
    from .config import DEFAULT_FONT

    if DEFAULT_FONT and DEFAULT_FONT.strip():
        font_name = DEFAULT_FONT.strip()

        try:
            import winreg

            font_registry_key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"SOFTWARE\Microsoft\Windows NT\CurrentVersion\Fonts",
            )

            for i in range(winreg.QueryInfoKey(font_registry_key)[1]):
                try:
                    registry_font_name, registry_font_file, _ = winreg.EnumValue(
                        font_registry_key, i
                    )

                    if font_name.lower() in registry_font_name.lower():
                        winreg.CloseKey(font_registry_key)
                        return registry_font_file
                except Exception:
                    continue

            winreg.CloseKey(font_registry_key)
        except Exception as e:
            print(f"Warning: Could not search Windows font registry: {e}")

        return font_name

    return "arial.ttf"


def _load_pil_font(font_path: str, size: int) -> ImageFont.FreeTypeFont:
    """Load PIL font with fallback to default."""
    try:
        return ImageFont.truetype(font_path, size)
    except Exception:
        try:
            return ImageFont.truetype("arial.ttf", size)
        except Exception:
            try:
                return ImageFont.load_default()
            except Exception:
                return ImageFont.load("arial.ttf")


@lru_cache(maxsize=512)
def _get_text_size_pil(text: str, font_path: str, font_size: int) -> tuple:
    """
    Get text size using PIL - cached for performance.

    This is 10-50x faster than creating a TextClip just for measurement.
    """
    try:
        font = _load_pil_font(font_path, font_size)
        dummy_img = Image.new("RGBA", (1, 1), (0, 0, 0, 0))
        draw = ImageDraw.Draw(dummy_img)
        bbox = draw.textbbox((0, 0), text, font=font)
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        return (max(width, font_size // 2), max(height, font_size // 2))
    except Exception:
        est_width = len(text) * (font_size // 2)
        est_height = font_size
        return (max(est_width, font_size // 2), max(est_height, font_size // 2))


@lru_cache(maxsize=256)
def _get_line_dimensions_cached(line: str, font_path: str, font_size: int) -> tuple:
    """Cached line dimension calculation."""
    return _get_text_size_pil(line, font_path, font_size)


class SubtitleGenerator:
    """Handles subtitle generation from text and timing."""

    @staticmethod
    def format_vtt_time(seconds: float) -> str:
        """Convert seconds to VTT time format."""
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        ms = int((seconds % 1) * 1000)
        return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"

    @staticmethod
    def generate_vtt_from_paragraphs(
        paragraphs: list, duration: float, output_path: str
    ) -> str:
        """Generate VTT subtitle file from paragraphs and audio duration."""
        combined_text = " ".join(paragraphs)
        cleaned_text = combined_text.replace('"', "").replace("\n", " ").strip()

        words = cleaned_text.split()
        chunks = []
        for i in range(0, len(words), 5):
            chunk = " ".join(words[i : i + 5])
            if chunk.strip():
                chunks.append(chunk.strip())

        total_chunks = len(chunks)
        vtt_lines = ["WEBVTT", ""]
        current_time = 0.0

        for i, chunk in enumerate(chunks):
            chunk_duration = duration / total_chunks
            start = SubtitleGenerator.format_vtt_time(current_time)
            end = SubtitleGenerator.format_vtt_time(current_time + chunk_duration)

            vtt_lines.extend([str(i + 1), f"{start} --> {end}", chunk, ""])
            current_time += chunk_duration

        with open(output_path, "w", encoding="utf-8-sig") as f:
            f.write("\n".join(vtt_lines))

        return output_path


class SubtitleRenderer:
    """Handles subtitle rendering and clip creation - optimized version."""

    def __init__(self, font_path: str = None):
        self.font = font_path or get_system_font()
        self._text_clip_cache = {}

    def _wrap_text_to_lines(self, text: str, max_chars_per_line: int = 22) -> list:
        """Wrap text to lines with character limit."""
        words = text.split()
        lines, current_line = [], []
        current_len = 0
        for word in words:
            if current_len + len(word) + 1 > max_chars_per_line and current_line:
                lines.append(current_line)
                current_line, current_len = [], 0
            current_line.append(word)
            current_len += len(word) + 1
        if current_line:
            lines.append(current_line)
        return lines

    def _get_text_dimensions(self, text: str, font_size: int) -> tuple:
        """
        Get text dimensions - now uses PIL-based cached function.

        Falls back to cached TextClip measurements if PIL fails.
        """
        try:
            w, h = _get_text_size_pil(text, self.font, font_size)
            return (max(w, 30), max(h, 30))
        except Exception:
            pass

        cache_key = f"{text}_{font_size}"
        if cache_key in self._text_clip_cache:
            return self._text_clip_cache[cache_key]

        try:
            temp_clip = TextClip(
                text=text,
                font_size=font_size,
                font=self.font,
                color="white",
                stroke_color="black",
                stroke_width=1,
                method="label",
                transparent=True,
            )
            width, height = temp_clip.size
            temp_clip.close()

            width = max(width, 30)
            height = max(height, 30)

            self._text_clip_cache[cache_key] = (width, height)
            return (width, height)
        except Exception:
            width = max(len(text) * (font_size // 2), 50)
            height = max(font_size, 50)
            self._text_clip_cache[cache_key] = (width, height)
            return (width, height)

    def _vtt_time_to_seconds(self, time_str: str) -> float:
        """Convert VTT time string to seconds."""
        parts = time_str.split(":")
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])

    def create_subtitle_clips_optimized(self, vtt_path: str, video_size: tuple) -> list:
        """Create subtitle clips - routing to appropriate mode."""
        if not vtt_path or not Path(vtt_path).exists():
            return []

        if SUBTITLE_MODE == "simple":
            return self._create_simple_subtitles(vtt_path, video_size)
        else:
            return self._create_highlight_subtitles(vtt_path, video_size)

    def _create_simple_subtitles(self, vtt_path: str, video_size: tuple) -> list:
        """
        Fast subtitle rendering - whole caption at once, no word highlighting.

        Uses PIL for pixel-perfect text rendering on full-canvas ImageClip
        to avoid TextClip bounding-box clipping issues with strokes.
        """
        from io import BytesIO

        from moviepy import ImageClip

        width, height = video_size
        clips = []

        base_y = int(height * SUBTITLE_START_Y_RATIO)
        font_size = FONT_SIZE
        stroke_width = max(STROKE_WIDTH, 1)
        canvas_pad = 60

        try:
            vtt = webvtt.read(vtt_path)
            for caption in vtt:
                text = caption.text.strip()
                if not text:
                    continue

                start_time = self._vtt_time_to_seconds(caption.start)
                end_time = self._vtt_time_to_seconds(caption.end)
                duration = end_time - start_time
                if duration <= 0:
                    continue

                lines = self._wrap_text_to_lines(
                    text, max_chars_per_line=MAX_CHARS_PER_LINE
                )

                line_heights = []
                line_widths = []
                for line_words in lines:
                    line_text = " ".join(line_words)
                    w, h = _get_line_dimensions_cached(line_text, self.font, font_size)
                    line_heights.append(max(h, font_size))
                    line_widths.append(w)

                total_text_w = max(line_widths) if line_widths else font_size
                total_text_h = sum(line_heights) + LINE_SPACING * max(0, len(lines) - 1)
                canvas_w = total_text_w + canvas_pad * 2
                canvas_h = total_text_h + canvas_pad * 2

                pil_font = _load_pil_font(self.font, font_size)

                img = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
                draw = ImageDraw.Draw(img)

                y_offset = canvas_pad
                for line_idx, line_words in enumerate(lines):
                    line_text = " ".join(line_words)
                    line_h = line_heights[line_idx]
                    line_w = line_widths[line_idx]
                    x_offset = (canvas_w - line_w) // 2

                    if stroke_width > 0:
                        draw.text(
                            (x_offset, y_offset),
                            line_text,
                            font=pil_font,
                            fill=tuple(
                                int(COLOR_STROKE.lstrip("#")[i : i + 2], 16)
                                for i in (0, 2, 4)
                            )
                            + (255,),
                            stroke_width=stroke_width,
                        )

                    draw.text(
                        (x_offset, y_offset),
                        line_text,
                        font=pil_font,
                        fill=tuple(
                            int(COLOR_TEXT.lstrip("#")[i : i + 2], 16)
                            for i in (0, 2, 4)
                        )
                        + (255,),
                    )

                    y_offset += line_h + LINE_SPACING

                buf = BytesIO()
                img.save(buf, format="PNG")
                buf.seek(0)

                clip = ImageClip(buf).with_duration(duration)
                clip = clip.with_position(((width - canvas_w) // 2, base_y))

                clips.append(clip.with_start(start_time))

        except Exception as e:
            print(f"Error creating simple subtitles: {e}")

        return clips

    def _create_highlight_subtitles(self, vtt_path: str, video_size: tuple) -> list:
        """
        Subtitle rendering with word-by-word highlighting.

        Optimized with:
        - Pre-calculated word positions
        - Shared base clip for non-highlighted words
        - Only highlighted word gets separate clip
        """
        width, height = video_size
        clips = []

        base_y = int(height * SUBTITLE_START_Y_RATIO)
        font_size = FONT_SIZE
        stroke_width = STROKE_WIDTH
        space_width = font_size * 0.25

        try:
            vtt = webvtt.read(vtt_path)
            for caption in vtt:
                text = caption.text.strip()
                if not text:
                    continue

                start_time = self._vtt_time_to_seconds(caption.start)
                end_time = self._vtt_time_to_seconds(caption.end)
                duration = end_time - start_time
                if duration <= 0:
                    continue

                lines = self._wrap_text_to_lines(
                    text, max_chars_per_line=MAX_CHARS_PER_LINE
                )
                words = text.split()
                word_count = max(len(words), 1)
                word_duration = duration / word_count

                current_y = max(base_y, int(height * 0.3))

                line_heights = []
                for line_words in lines:
                    line_text = " ".join(line_words)
                    _, h = _get_line_dimensions_cached(line_text, self.font, font_size)
                    line_heights.append(max(h, font_size))

                for line_idx, line_words in enumerate(lines):
                    line_clips_data = []
                    total_line_width = 0

                    for word in line_words:
                        w, h = _get_text_size_pil(word, self.font, font_size)
                        w = max(w, 30)
                        h = max(h, 30)
                        line_clips_data.append({"word": word, "w": w, "h": h})
                        total_line_width += w + space_width

                    if not line_clips_data:
                        continue
                    total_line_width -= space_width

                    current_x = (width - total_line_width) // 2

                    line_text = " ".join(line_words)
                    try:
                        base_clip = TextClip(
                            text=line_text,
                            font_size=font_size,
                            font=self.font,
                            color=COLOR_TEXT,
                            stroke_color=COLOR_STROKE,
                            stroke_width=stroke_width,
                            method="label",
                            transparent=True,
                        )
                        clips.append(
                            base_clip.with_position((current_x, current_y))
                            .with_start(start_time)
                            .with_duration(duration)
                        )
                    except Exception as e:
                        print(f"Error creating base line: {e}")
                        continue

                    line_height = (
                        line_heights[line_idx]
                        if line_idx < len(line_heights)
                        else font_size
                    )

                    for word_idx, data in enumerate(line_clips_data):
                        word = data["word"]

                        word_start = start_time + (word_idx * word_duration)
                        word_end = min(word_start + word_duration + 0.1, end_time)

                        if word_end <= word_start:
                            continue

                        try:
                            word_clip = TextClip(
                                text=word,
                                font_size=font_size,
                                font=self.font,
                                color=COLOR_HIGHLIGHT,
                                stroke_color=COLOR_STROKE,
                                stroke_width=stroke_width,
                                method="label",
                                transparent=True,
                            )
                            if word_clip.size[0] <= 0 or word_clip.size[1] <= 0:
                                word_clip.close()
                                continue

                            clips.append(
                                word_clip.with_position((current_x, current_y))
                                .with_start(word_start)
                                .with_duration(max(0.1, word_end - word_start))
                            )
                        except Exception as e:
                            print(f"Error creating word clip: {e}")
                            continue

                        current_x += data["w"] + space_width

                    current_y += line_height + LINE_SPACING

        except Exception as e:
            print(f"Error processing subtitles: {e}")
            raise

        return clips


class SubtitleSystem:
    """Main interface for subtitle generation and rendering."""

    def __init__(self, font_path: str = None):
        self.generator = SubtitleGenerator()
        self.renderer = SubtitleRenderer(font_path)

    def generate_subtitles(
        self, paragraphs: list, duration: float, output_dir: str
    ) -> str:
        """Generate VTT subtitle file from paragraphs."""
        vtt_path = Path(output_dir) / "subtitles.vtt"
        return self.generator.generate_vtt_from_paragraphs(
            paragraphs, duration, str(vtt_path)
        )

    def render_subtitles(self, vtt_path: str, video_size: tuple) -> list:
        """Render subtitle clips from VTT file."""
        return self.renderer.create_subtitle_clips_optimized(vtt_path, video_size)
