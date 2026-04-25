"""
TTS (Text-to-Speech) system module for AutoShorts

Unified TTS generation using Edge TTS.
"""

import os
import subprocess
import sys
from pathlib import Path

import edge_tts

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from .config import FFPROBE_TIMEOUT, TTS_RATE, VOICE
from .logging_system import log
from .subtitle_system import SubtitleSystem


class TTSSystem:
    """Unified TTS generation system."""

    def __init__(self):
        self.voice = VOICE
        self.rate = TTS_RATE
        self.subtitle_system = SubtitleSystem()

    async def generate_audio_and_subtitles(
        self, paragraphs: list, output_dir: Path
    ) -> tuple:
        """Generate both audio and subtitles from paragraphs."""
        combined_text = " ".join(paragraphs)
        audio_file = output_dir / "narration.mp3"

        cleaned_text = combined_text.replace('"', "").replace("\n", " ").strip()

        log(f"Generating TTS Audio ({len(cleaned_text)} chars)...")
        communicate = edge_tts.Communicate(cleaned_text, self.voice, rate=self.rate)
        await communicate.save(str(audio_file))

        # Get audio duration
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(audio_file),
        ]
        duration = float(
            subprocess.run(
                cmd, capture_output=True, text=True, timeout=FFPROBE_TIMEOUT
            ).stdout.strip()
        )

        # Generate subtitles using subtitle system
        vtt_file = self.subtitle_system.generate_subtitles(
            paragraphs, duration, str(output_dir)
        )

        return str(audio_file), vtt_file, duration

    async def generate_audio_only(self, paragraphs: list, temp_dir: Path) -> str:
        """Generate only audio without subtitles."""
        text = " ".join(paragraphs)
        path = temp_dir / "audio.mp3"
        log("Generating TTS audio...")

        communicate = edge_tts.Communicate(text, self.voice, rate=self.rate)
        await communicate.save(str(path))
        return str(path)
