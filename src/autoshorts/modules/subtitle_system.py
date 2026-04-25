#!/usr/bin/env python3
"""
Subtitle System Module for AutoShorts

Handles subtitle generation and rendering for video processing.
Extracted from autoshorts_yt_summarizer.py for modular reuse.
"""

import os

import webvtt
from moviepy import TextClip

from .config import (
    COLOR_HIGHLIGHT,
    COLOR_STROKE,
    COLOR_TEXT,
    FONT_SIZE,
    LINE_SPACING,
    MAX_CHARS_PER_LINE,
    STROKE_WIDTH,
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
    """Handles subtitle rendering and clip creation."""

    def __init__(self, font_path: str = None):
        self.font = font_path or get_system_font()
        self._text_clip_cache = {}

    def _wrap_text_to_lines(self, text, max_chars_per_line=22):
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
        """Calculate text dimensions using actual TextClip for accurate measurements."""
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
        except Exception as e:
            print(f"Error calculating text dimensions for '{text}': {e}")
            try:
                temp_clip = TextClip(
                    text=text,
                    font_size=font_size,
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
            except Exception as e2:
                print(f"Fallback also failed for '{text}': {e2}")
                width = max(len(text) * (font_size // 2), 50)
                height = max(font_size, 50)
                self._text_clip_cache[cache_key] = (width, height)
                return (width, height)

    def _vtt_time_to_seconds(self, time_str: str) -> float:
        """Convert VTT time string to seconds."""
        parts = time_str.split(":")
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])

    def create_subtitle_clips_optimized(self, vtt_path: str, video_size: tuple) -> list:
        """Create subtitle clips - simple mode is faster (no word highlighting)."""
        if not vtt_path or not os.path.exists(vtt_path):
            return []

        from .config import SUBTITLE_MODE

        if SUBTITLE_MODE == "simple":
            return self._create_simple_subtitles(vtt_path, video_size)
        else:
            return self._create_highlight_subtitles(vtt_path, video_size)

    def _create_simple_subtitles(self, vtt_path: str, video_size: tuple) -> list:
        """Fast subtitle rendering - whole line at once, no word highlighting."""
        width, height = video_size
        clips = []

        base_y = int(height * SUBTITLE_START_Y_RATIO)

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
                for line_words in lines:
                    max_h = FONT_SIZE
                    for word in line_words:
                        _, h = self._get_text_dimensions(word, FONT_SIZE)
                        max_h = max(max_h, h)
                    line_heights.append(max_h)

                current_y = base_y
                min_y = int(height * 0.3)
                if current_y < min_y:
                    current_y = min_y

                for line_idx, line_words in enumerate(lines):
                    line_text = " ".join(line_words)
                    try:
                        clip = TextClip(
                            text=line_text,
                            font_size=FONT_SIZE,
                            font=self.font,
                            color=COLOR_TEXT,
                            stroke_color=COLOR_STROKE,
                            stroke_width=STROKE_WIDTH,
                            method="label",
                            transparent=True,
                        )
                        if clip.size[0] <= 0 or clip.size[1] <= 0:
                            clip.close()
                            continue

                        current_x = (width - clip.size[0]) // 2
                        clips.append(
                            clip.with_position((current_x, current_y))
                            .with_start(start_time)
                            .with_duration(duration)
                        )
                    except Exception as e:
                        print(f"Error creating simple subtitle: {e}")
                        continue

                    current_y += line_heights[line_idx] + LINE_SPACING

        except Exception as e:
            print(f"Error processing subtitles: {e}")

        return clips

    def _create_highlight_subtitles(self, vtt_path: str, video_size: tuple) -> list:
        """Subtitle rendering with word-by-word highlighting (slower, more clips)."""
        width, height = video_size
        clips = []

        base_y = int(height * SUBTITLE_START_Y_RATIO)

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
                word_duration = duration / max(len(words), 1)

                word_global_index = 0

                line_heights = []
                for line_words in lines:
                    max_h = FONT_SIZE
                    for word in line_words:
                        _, h = self._get_text_dimensions(word, FONT_SIZE)
                        max_h = max(max_h, h)
                    line_heights.append(max_h)

                current_y = base_y

                min_y = int(height * 0.3)
                if current_y < min_y:
                    current_y = min_y

                for line_idx, line_words in enumerate(lines):
                    space_width = FONT_SIZE * 0.25
                    line_clips_data = []
                    total_line_width = 0

                    for word in line_words:
                        w, h = self._get_text_dimensions(word, FONT_SIZE)
                        w = max(w, 30)
                        h = max(h, 30)
                        line_clips_data.append({"word": word, "w": w, "h": h})
                        total_line_width += w + space_width

                    if not line_clips_data:
                        continue
                    total_line_width -= space_width

                    current_x = (width - total_line_width) // 2

                    try:
                        line_text = " ".join(line_words)
                        line_width, line_height = self._get_text_dimensions(
                            line_text, FONT_SIZE
                        )

                        try:
                            base_clip = TextClip(
                                text=line_text,
                                font_size=FONT_SIZE,
                                font=self.font,
                                color=COLOR_TEXT,
                                stroke_color=COLOR_STROKE,
                                stroke_width=STROKE_WIDTH,
                                method="label",
                                transparent=True,
                            )
                        except Exception as font_error:
                            print(f"Font error for base line: {font_error}")
                            base_clip = TextClip(
                                text=line_text,
                                font_size=FONT_SIZE,
                                color=COLOR_TEXT,
                                stroke_color=COLOR_STROKE,
                                stroke_width=STROKE_WIDTH,
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

                    max_h = (
                        line_heights[line_idx]
                        if line_idx < len(line_heights)
                        else FONT_SIZE
                    )

                    for data in line_clips_data:
                        word, w, h = data["word"], data["w"], data["h"]
                        max_h = max(max_h, h)

                        word_start = start_time + (word_global_index * word_duration)
                        word_end = min(word_start + word_duration + 0.1, end_time)

                        try:
                            word_width, word_height = self._get_text_dimensions(
                                word, FONT_SIZE
                            )

                            try:
                                clip = TextClip(
                                    text=word,
                                    font_size=FONT_SIZE,
                                    font=self.font,
                                    color=COLOR_HIGHLIGHT,
                                    stroke_color=COLOR_STROKE,
                                    stroke_width=STROKE_WIDTH,
                                    method="label",
                                    transparent=True,
                                )
                            except Exception as font_error:
                                print(f"Font error for word '{word}': {font_error}")
                                clip = TextClip(
                                    text=word,
                                    font_size=FONT_SIZE,
                                    color=COLOR_HIGHLIGHT,
                                    stroke_color=COLOR_STROKE,
                                    stroke_width=STROKE_WIDTH,
                                    method="label",
                                    transparent=True,
                                )

                            if clip.size[0] <= 0 or clip.size[1] <= 0:
                                clip.close()
                                continue

                            clips.append(
                                clip.with_position((current_x, current_y))
                                .with_start(word_start)
                                .with_duration(max(0.1, word_end - word_start))
                            )

                        except Exception as e:
                            print(f"Error creating subtitle clip: {e}")
                            continue

                        current_x += w + space_width
                        word_global_index += 1

                    current_y += max_h + LINE_SPACING

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
        vtt_path = os.path.join(output_dir, "subtitles.vtt")
        return self.generator.generate_vtt_from_paragraphs(
            paragraphs, duration, vtt_path
        )

    def render_subtitles(self, vtt_path: str, video_size: tuple) -> list:
        """Render subtitle clips from VTT file."""
        return self.renderer.create_subtitle_clips_optimized(vtt_path, video_size)
