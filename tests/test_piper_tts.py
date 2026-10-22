"""Tests for the Piper TTS engine (no network, no model downloads)."""
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from autoshorts.modules import piper_tts
from autoshorts.modules.piper_tts import PiperError, ensure_voice, voice_url


def test_voice_url_derives_huggingface_path():
    url = voice_url("pt_BR-faber-medium")
    assert url.endswith("/pt/pt_BR/faber/medium/pt_BR-faber-medium.onnx")
    assert url.startswith("https://huggingface.co/")


def test_voice_url_rejects_malformed_id():
    with pytest.raises(PiperError, match="PIPER_VOICE_URL"):
        voice_url("not-a-voice")


def _fake_response(data: bytes):
    resp = mock.MagicMock()
    resp.raise_for_status = lambda: None
    resp.iter_content = lambda chunk_size: [data]
    resp.__enter__ = lambda s: resp
    resp.__exit__ = lambda *a: False
    return resp


def test_ensure_voice_downloads_once_and_reuses_cache(tmp_path):
    payload = b"x" * 200
    with mock.patch.object(piper_tts.requests, "get",
                           return_value=_fake_response(payload)) as get:
        model, config = ensure_voice(tmp_path, "pt_BR-faber-medium")
        assert model.read_bytes() == payload and config.read_bytes() == payload
        assert get.call_count == 2
        # Second call: files exist and are non-trivial -> no new downloads.
        ensure_voice(tmp_path, "pt_BR-faber-medium")
        assert get.call_count == 2


def test_ensure_voice_rejects_truncated_download(tmp_path):
    with mock.patch.object(piper_tts.requests, "get",
                           return_value=_fake_response(b"tiny")):
        with pytest.raises(PiperError, match="empty/truncated"):
            ensure_voice(tmp_path, "pt_BR-faber-medium")


class _FakeChunk:
    def __init__(self, samples, sr=22050):
        self.audio_int16_array = samples
        self.sample_rate = sr


def _fake_voice(sr=22050):
    voice = mock.MagicMock()
    voice.synthesize = lambda text, cfg: [
        _FakeChunk(np.full(sr // 10, 100, dtype=np.int16), sr)]
    return voice


def test_synthesize_writes_valid_wav(tmp_path):
    import wave

    fake_piper = mock.MagicMock()
    fake_piper.PiperVoice.load.return_value = _fake_voice()
    fake_config = mock.MagicMock()
    with mock.patch.dict("sys.modules", {"piper": fake_piper,
                                         "piper.config": fake_config}), \
         mock.patch.object(piper_tts, "ensure_voice",
                           return_value=(tmp_path / "m.onnx", tmp_path / "c.json")):
        out = tmp_path / "out.wav"
        duration = piper_tts.synthesize("Um gol. Dois gols!", out)
    with wave.open(str(out), "rb") as wav:
        assert wav.getframerate() == 22050
        assert wav.getnchannels() == 1
        frames = wav.getnframes()
    assert frames / 22050 == pytest.approx(duration, abs=0.01)
    # two sentences of 0.1s + one 0.16s gap
    assert duration == pytest.approx(0.36, abs=0.01)


def test_synthesize_requires_piper_package(tmp_path):
    with mock.patch.dict("sys.modules", {"piper": None}):
        with pytest.raises(PiperError, match="piper-tts is not installed"):
            piper_tts.synthesize("texto", tmp_path / "x.wav")


def test_synthesize_rejects_empty_text(tmp_path):
    with pytest.raises(PiperError, match="Empty narration"):
        piper_tts.synthesize("   ", tmp_path / "x.wav")


@pytest.mark.asyncio
async def test_tts_system_piper_engine_writes_mp3(tmp_path, monkeypatch):
    from autoshorts.modules import tts_system

    monkeypatch.setattr(tts_system, "TTS_ENGINE", "piper")

    def fake_synthesize(text, wav_path, *, length_scale=1.0):
        Path(wav_path).write_bytes(b"RIFFfake")
        return 1.0

    def fake_run(cmd, **kwargs):
        Path(cmd[-1]).write_bytes(b"ID3fake")
        return mock.MagicMock(returncode=0, stderr="")

    monkeypatch.setattr("autoshorts.modules.piper_tts.synthesize", fake_synthesize)
    monkeypatch.setattr(tts_system.subprocess, "run", fake_run)

    tts = tts_system.TTSSystem()
    out = await tts.generate_audio_only(["Frase um.", "Frase dois."], tmp_path)
    assert out.endswith("audio.mp3")
    assert Path(out).read_bytes() == b"ID3fake"
    # intermediate wav is cleaned up
    assert not (tmp_path / "audio.wav").exists()
