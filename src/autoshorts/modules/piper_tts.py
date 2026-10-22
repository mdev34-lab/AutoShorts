"""Local neural TTS via Piper (no cloud account or API key).

Piper (https://github.com/OHF-Voice/piper1-gpl) runs small ONNX voices
fully offline. This module downloads a voice once into a cache directory
and synthesizes the narration to WAV; `TTSSystem` then converts to mp3
with ffmpeg so the rest of the pipeline sees exactly what Edge TTS would
have produced.

Enabled with TTS_ENGINE=piper. Requires the optional dependency:
`pip install "autoshorts[piper]"` (or `pip install piper-tts`).
"""
from __future__ import annotations

import re
import wave
from pathlib import Path

import numpy as np
import requests

from .config import PIPER_CACHE_DIR, PIPER_VOICE_ID, PIPER_VOICE_URL
from .logging_system import log

_HF_VOICES = "https://huggingface.co/rhasspy/piper-voices/resolve/main"
_MAX_MODEL_BYTES = 100 * 1024 * 1024


class PiperError(RuntimeError):
    """Raised when Piper cannot produce audio (missing dep, model, ...)."""


def voice_url(voice_id: str) -> str:
    """Hugging Face URL for a voice id like 'pt_BR-faber-medium'."""
    match = re.fullmatch(r"([a-z]{2})_([A-Z]{2})-([a-zA-Z0-9]+)-([a-z]+)", voice_id)
    if not match:
        raise PiperError(
            f"PIPER_VOICE_ID '{voice_id}' does not look like 'pt_BR-faber-medium'; "
            "set PIPER_VOICE_URL to the full .onnx URL instead."
        )
    lang, country, name, quality = match.groups()
    return f"{_HF_VOICES}/{lang}/{lang}_{country}/{name}/{quality}/{voice_id}.onnx"


def ensure_voice(cache_dir: Path, voice_id: str, base_url: str | None = None) -> tuple[Path, Path]:
    """Download the voice model + config once; return their paths."""
    base = (base_url or voice_url(voice_id)).rstrip("/")
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    model = cache_dir / f"{voice_id}.onnx"
    config = cache_dir / f"{voice_id}.onnx.json"
    for path, url in ((model, base), (config, base + ".json")):
        if path.exists() and path.stat().st_size > 100:
            continue
        log(f"Downloading Piper voice {voice_id} ({url.rsplit('/', 1)[-1]})...")
        try:
            with requests.get(url, stream=True, timeout=60) as resp:
                resp.raise_for_status()
                chunks, total = [], 0
                for chunk in resp.iter_content(chunk_size=1 << 20):
                    total += len(chunk)
                    if total > _MAX_MODEL_BYTES:
                        raise PiperError(f"Piper voice download exceeded {_MAX_MODEL_BYTES} bytes")
                    chunks.append(chunk)
        except PiperError:
            raise
        except Exception as e:
            raise PiperError(f"Could not download Piper voice {voice_id}: {e}") from e
        data = b"".join(chunks)
        if len(data) < 100:
            raise PiperError("Piper voice download came back empty/truncated.")
        tmp = path.with_suffix(path.suffix + ".part")
        tmp.write_bytes(data)
        tmp.replace(path)
    return model, config


def _sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        raise PiperError("Empty narration text; nothing to synthesize.")
    pieces = re.split(r"(?<=[.!?])\s+", text)
    return [p if p[-1] in ".!?…" else p + "." for p in (s.strip() for s in pieces) if p]


def synthesize(text: str, output_wav: Path, *, length_scale: float = 1.0) -> float:
    """Synthesize text to a WAV file; return its duration in seconds."""
    sentences = _sentences(text)  # validates before any heavy work
    try:
        from piper import PiperVoice
        from piper.config import SynthesisConfig
    except ImportError as e:
        raise PiperError(
            "piper-tts is not installed. Install with "
            '`pip install "autoshorts[piper]"` or set TTS_ENGINE=edge.'
        ) from e

    model_path, config_path = ensure_voice(PIPER_CACHE_DIR, PIPER_VOICE_ID, PIPER_VOICE_URL or None)
    try:
        voice = PiperVoice.load(str(model_path), str(config_path))
    except Exception as e:
        raise PiperError(f"Failed to load Piper voice {PIPER_VOICE_ID}: {e}") from e

    cfg = SynthesisConfig()
    cfg.length_scale = max(0.85, length_scale)

    pcm_parts: list[np.ndarray] = []
    sample_rate: int | None = None
    gap = 0.16  # natural pause between sentences
    for sentence in sentences:
        chunks = list(voice.synthesize(sentence, cfg))
        if not chunks:
            continue
        sr = int(chunks[0].sample_rate)
        if sample_rate is None:
            sample_rate = sr
        elif sr != sample_rate:
            raise PiperError("Piper returned inconsistent sample rates.")
        pcm_parts.append(np.concatenate([c.audio_int16_array for c in chunks]))
        pcm_parts.append(np.zeros(round(sr * gap), dtype=np.int16))
    if sample_rate is None or not pcm_parts:
        raise PiperError("Piper produced no usable audio.")
    pcm_parts.pop()  # no trailing gap
    audio = np.concatenate(pcm_parts)

    output_wav = Path(output_wav)
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output_wav), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(audio.tobytes())
    return len(audio) / sample_rate
