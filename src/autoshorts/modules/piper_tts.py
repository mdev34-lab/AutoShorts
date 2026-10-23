"""Local neural TTS via Piper (no cloud account or API key).

Piper (https://github.com/OHF-Voice/piper1-gpl) runs small ONNX voices
fully offline. This module downloads a voice once into a cache directory
and synthesizes the narration to WAV; `TTSSystem` then converts to mp3
with ffmpeg so the rest of the pipeline sees roughly what Edge TTS would
have produced.

Enabled with TTS_ENGINE=piper. Requires the optional dependency:
`pip install "autoshorts[piper]"` (or `pip install piper-tts`).
"""

from __future__ import annotations

import hashlib
import json
import re
import wave
from pathlib import Path

import numpy as np
import requests

from .config import PIPER_CACHE_DIR, PIPER_VOICE_ID, PIPER_VOICE_SHA256, PIPER_VOICE_URL
from .logging_system import log

_HF_VOICES = "https://huggingface.co/rhasspy/piper-voices/resolve/main"
_MAX_MODEL_BYTES = 100 * 1024 * 1024
# A real Piper voice model is tens of MB; anything below 1 MB is unusable.
_MIN_MODEL_BYTES = 1_000_000

SentenceTiming = tuple[str, float, float]


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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _valid_config(path: Path) -> bool:
    """A cached config must parse and look like a Piper voice config."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    audio = data.get("audio") if isinstance(data, dict) else None
    return isinstance(audio, dict) and isinstance(audio.get("sample_rate"), int)


def _valid_model(path: Path, sha256: str | None = None) -> bool:
    """A cached model must have a plausible size and match an optional pin."""
    try:
        size = path.stat().st_size
    except OSError:
        return False
    if not (_MIN_MODEL_BYTES <= size <= _MAX_MODEL_BYTES):
        return False
    if sha256 and _sha256(path) != sha256.lower():
        return False
    return True


def _download(url: str, path: Path, min_bytes: int = 100) -> None:
    log(f"Downloading Piper voice file ({url.rsplit('/', 1)[-1]})...")
    try:
        with requests.get(url, stream=True, timeout=60) as resp:
            resp.raise_for_status()
            chunks, total = [], 0
            for chunk in resp.iter_content(chunk_size=1 << 20):
                total += len(chunk)
                if total > _MAX_MODEL_BYTES:
                    raise PiperError(
                        f"Piper voice download exceeded {_MAX_MODEL_BYTES} bytes"
                    )
                chunks.append(chunk)
    except PiperError:
        raise
    except Exception as e:
        raise PiperError(f"Could not download Piper voice file {url}: {e}") from e
    data = b"".join(chunks)
    if len(data) < min_bytes:
        raise PiperError("Piper voice download came back empty/truncated.")
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_bytes(data)
    tmp.replace(path)


def ensure_voice(
    cache_dir: Path,
    voice_id: str,
    base_url: str | None = None,
    sha256: str | None = None,
) -> tuple[Path, Path]:
    """Download the voice model + config once; return their paths.

    Cached files are reused only after integrity checks: the config must
    parse as a Piper voice config and the model must have a plausible size
    and match the optional SHA-256 pin. A cache entry that fails validation
    is deleted and re-downloaded; a fresh download that still fails raises.
    """
    base = (base_url or voice_url(voice_id)).rstrip("/")
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    model = cache_dir / f"{voice_id}.onnx"
    config = cache_dir / f"{voice_id}.onnx.json"
    checks = (
        (model, base, lambda p: _valid_model(p, sha256), 100),
        (config, base + ".json", _valid_config, 10),
    )
    for path, url, is_valid, min_bytes in checks:
        if path.exists():
            if is_valid(path):
                continue
            log(
                f"Piper cache entry {path.name} failed integrity check; re-downloading.",
                "WARNING",
            )
            path.unlink()
        _download(url, path, min_bytes=min_bytes)
        if not is_valid(path):
            path.unlink(missing_ok=True)
            raise PiperError(
                f"Downloaded Piper voice file {path.name} failed integrity check."
            )
    return model, config


def _sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        raise PiperError("Empty narration text; nothing to synthesize.")
    pieces = re.split(r"(?<=[.!?])\s+", text)
    return [p if p[-1] in ".!?…" else p + "." for p in (s.strip() for s in pieces) if p]


def synthesize(
    text: str, output_wav: Path, *, length_scale: float = 1.0
) -> tuple[float, list[SentenceTiming]]:
    """Synthesize text to a WAV file.

    Returns (duration_seconds, timings) where timings is one
    (sentence, start_seconds, end_seconds) entry per synthesized sentence,
    measured from the actual audio so subtitles can follow real timing.
    """
    sentences = _sentences(text)  # validates before any heavy work
    try:
        from piper import PiperVoice  # type: ignore[import-not-found]
        from piper.config import SynthesisConfig  # type: ignore[import-not-found]
    except ImportError as e:
        raise PiperError(
            "piper-tts is not installed. Install with "
            '`pip install "autoshorts[piper]"` or set TTS_ENGINE=edge.'
        ) from e

    model_path, config_path = ensure_voice(
        PIPER_CACHE_DIR,
        PIPER_VOICE_ID,
        PIPER_VOICE_URL or None,
        sha256=PIPER_VOICE_SHA256 or None,
    )
    try:
        voice = PiperVoice.load(str(model_path), str(config_path))
    except Exception as first_error:
        # A cached model can pass the size/format checks and still be
        # corrupt: drop the cache entry, re-download once and retry once.
        log(
            f"Piper voice failed to load ({first_error}); "
            "refreshing the cached voice and retrying once.",
            "WARNING",
        )
        model_path.unlink(missing_ok=True)
        config_path.unlink(missing_ok=True)
        model_path, config_path = ensure_voice(
            PIPER_CACHE_DIR,
            PIPER_VOICE_ID,
            PIPER_VOICE_URL or None,
            sha256=PIPER_VOICE_SHA256 or None,
        )
        try:
            voice = PiperVoice.load(str(model_path), str(config_path))
        except Exception as e:
            raise PiperError(f"Failed to load Piper voice {PIPER_VOICE_ID}: {e}") from e

    cfg = SynthesisConfig()
    cfg.length_scale = min(2.0, max(0.5, length_scale))

    pcm_parts: list[np.ndarray] = []
    timings: list[SentenceTiming] = []
    sample_rate: int | None = None
    cursor = 0  # samples written so far
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
        pcm = np.concatenate([c.audio_int16_array for c in chunks])
        timings.append((sentence, cursor / sr, (cursor + len(pcm)) / sr))
        pcm_parts.append(pcm)
        cursor += len(pcm)
        gap_samples = round(sr * gap)
        pcm_parts.append(np.zeros(gap_samples, dtype=np.int16))
        cursor += gap_samples
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
    return len(audio) / sample_rate, timings
