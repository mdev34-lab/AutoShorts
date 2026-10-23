"""
TTS (Text-to-Speech) system module for AutoShorts

Unified TTS generation using Edge TTS.
"""

import os
import re
import subprocess
import sys
from pathlib import Path

import edge_tts

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from .config import FFPROBE_TIMEOUT, TTS_ENGINE, TTS_RATE, VOICE
from .logging_system import log
from .subtitle_system import SubtitleSystem
from .utils import strip_markdown

_VALID_ENGINES = ("edge", "piper")


def _validate_engine(engine: str) -> str:
    """Reject unknown TTS_ENGINE values instead of silently using the cloud."""
    if engine not in _VALID_ENGINES:
        raise ValueError(
            f"Unknown TTS_ENGINE '{engine}': expected one of "
            f"{', '.join(_VALID_ENGINES)}. Fix the typo or unset TTS_ENGINE "
            "to use the default Edge engine."
        )
    return engine


def _rate_to_length_scale(rate: str) -> float:
    """Map an Edge-style TTS_RATE (e.g. '+20%', '-10%') to a Piper length_scale."""
    match = re.fullmatch(r"([+-]?\d+(?:\.\d+)?)%", rate.strip())
    if not match:
        log(f"Unparseable TTS_RATE '{rate}', using neutral Piper speed.", "WARNING")
        return 1.0
    factor = 1.0 + float(match.group(1)) / 100.0
    if factor <= 0:
        log(f"Invalid TTS_RATE '{rate}', using neutral Piper speed.", "WARNING")
        return 1.0
    return min(2.0, max(0.5, 1.0 / factor))


class TTSSystem:
    """Unified TTS generation system."""

    def __init__(self):
        self.voice = VOICE
        self.rate = TTS_RATE
        self.engine = _validate_engine(TTS_ENGINE)
        self.subtitle_system = SubtitleSystem()

    async def generate_audio_and_subtitles(
        self, paragraphs: list, output_dir: Path
    ) -> tuple:
        """Generate both audio and subtitles from paragraphs."""
        combined_text = " ".join(paragraphs)
        audio_file = output_dir / "narration.mp3"

        cleaned_text = strip_markdown(combined_text.replace('"', "").replace("\n", " ")).strip()

        log(
            f"Generating TTS Audio ({len(cleaned_text)} chars, engine={self.engine})..."
        )
        timings = None
        if self.engine == "piper":
            timings = self._piper_to_mp3(cleaned_text, audio_file)
        else:
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
        try:
            duration = float(
                subprocess.run(
                    cmd, capture_output=True, text=True, timeout=FFPROBE_TIMEOUT
                ).stdout.strip()
            )
        except (ValueError, subprocess.TimeoutExpired, OSError) as e:
            log(f"Failed to get audio duration from ffprobe: {e}", "WARNING")
            duration = 0.0

        # Generate subtitles using subtitle system: real per-sentence
        # timings when Piper produced them, evenly spaced chunks otherwise.
        if timings:
            vtt_file = self.subtitle_system.generate_subtitles_from_timings(
                timings, duration, str(output_dir)
            )
        else:
            vtt_file = self.subtitle_system.generate_subtitles(
                paragraphs, duration, str(output_dir)
            )

        return str(audio_file), vtt_file, duration

    async def generate_audio_only(self, paragraphs: list, temp_dir: Path) -> str:
        """Generate only audio without subtitles."""
        text = " ".join(paragraphs)
        path = temp_dir / "audio.mp3"
        log(f"Generating TTS audio (engine={self.engine})...")

        if self.engine == "piper":
            self._piper_to_mp3(text, path)
        else:
            communicate = edge_tts.Communicate(text, self.voice, rate=self.rate)
            await communicate.save(str(path))
        return str(path)

    def _piper_to_mp3(self, text: str, output_mp3: Path) -> list:
        """Synthesize with local Piper (WAV) and convert to mp3 via ffmpeg.

        Returns per-sentence (text, start, end) audio timings so subtitles
        can follow the real narration instead of even chunks.
        """
        from .piper_tts import synthesize

        wav_path = output_mp3.with_suffix(".wav")
        _, timings = synthesize(
            text, wav_path, length_scale=_rate_to_length_scale(self.rate)
        )
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(wav_path),
            "-codec:a",
            "libmp3lame",
            "-q:a",
            "4",
            str(output_mp3),
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        finally:
            wav_path.unlink(missing_ok=True)
        if result.returncode != 0 or not output_mp3.exists():
            raise RuntimeError(
                f"ffmpeg failed converting Piper WAV to mp3: {result.stderr[-500:]}"
            )
        return timings
