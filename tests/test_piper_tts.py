"""Tests for the Piper TTS engine (no network, no model downloads)."""

from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from autoshorts.modules import piper_tts
from autoshorts.modules.piper_tts import PiperError, ensure_voice, voice_url

MODEL_PAYLOAD = b"x" * (piper_tts._MIN_MODEL_BYTES + 1)
CONFIG_PAYLOAD = b'{"audio": {"sample_rate": 22050}, "num_symbols": 256}'


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


def _fake_downloads(get_mock):
    """Route fake downloads by URL: .json -> config payload, else model."""

    def route(url, **kwargs):
        return _fake_response(
            CONFIG_PAYLOAD if url.endswith(".json") else MODEL_PAYLOAD
        )

    get_mock.side_effect = route


def test_ensure_voice_downloads_once_and_reuses_valid_cache(tmp_path):
    with mock.patch.object(piper_tts.requests, "get") as get:
        _fake_downloads(get)
        model, config = ensure_voice(tmp_path, "pt_BR-faber-medium")
        assert model.read_bytes() == MODEL_PAYLOAD
        assert config.read_bytes() == CONFIG_PAYLOAD
        assert get.call_count == 2
        # Second call: cached files pass integrity -> no new downloads.
        ensure_voice(tmp_path, "pt_BR-faber-medium")
        assert get.call_count == 2


def test_ensure_voice_redownloads_corrupt_cached_config(tmp_path):
    with mock.patch.object(piper_tts.requests, "get") as get:
        _fake_downloads(get)
        _, config = ensure_voice(tmp_path, "pt_BR-faber-medium")
        assert get.call_count == 2
        config.write_bytes(b"not json at all")
        ensure_voice(tmp_path, "pt_BR-faber-medium")
        # Only the corrupt config is re-downloaded; the model cache holds.
        assert get.call_count == 3
        assert config.read_bytes() == CONFIG_PAYLOAD


def test_ensure_voice_redownloads_undersized_model(tmp_path):
    with mock.patch.object(piper_tts.requests, "get") as get:
        _fake_downloads(get)
        model, config = ensure_voice(tmp_path, "pt_BR-faber-medium")
        assert get.call_count == 2
        model.write_bytes(b"x" * 200)  # truncated/garbage cache entry
        ensure_voice(tmp_path, "pt_BR-faber-medium")
        assert get.call_count == 3
        assert model.read_bytes() == MODEL_PAYLOAD
        assert config.read_bytes() == CONFIG_PAYLOAD


def test_ensure_voice_rejects_oversized_download(tmp_path, monkeypatch):
    monkeypatch.setattr(piper_tts, "_MAX_MODEL_BYTES", 300)
    payload = b"y" * 400
    with mock.patch.object(
        piper_tts.requests, "get", return_value=_fake_response(payload)
    ):
        with pytest.raises(PiperError, match="exceeded"):
            ensure_voice(tmp_path, "pt_BR-faber-medium")


def test_ensure_voice_rejects_truncated_download(tmp_path):
    with mock.patch.object(
        piper_tts.requests, "get", return_value=_fake_response(b"tiny")
    ):
        with pytest.raises(PiperError, match="empty/truncated"):
            ensure_voice(tmp_path, "pt_BR-faber-medium")


def test_ensure_voice_sha256_pin_mismatch_raises(tmp_path):
    with mock.patch.object(piper_tts.requests, "get") as get:
        _fake_downloads(get)
        with pytest.raises(PiperError, match="integrity"):
            ensure_voice(tmp_path, "pt_BR-faber-medium", sha256="0" * 64)
        # The failed download is not left behind as a valid-looking cache.
        assert not (tmp_path / "pt_BR-faber-medium.onnx").exists()


def test_ensure_voice_sha256_pin_match_reuses_cache(tmp_path):
    import hashlib

    pin = hashlib.sha256(MODEL_PAYLOAD).hexdigest()
    with mock.patch.object(piper_tts.requests, "get") as get:
        _fake_downloads(get)
        ensure_voice(tmp_path, "pt_BR-faber-medium", sha256=pin)
        assert get.call_count == 2
        ensure_voice(tmp_path, "pt_BR-faber-medium", sha256=pin)
        assert get.call_count == 2


class _FakeChunk:
    def __init__(self, samples, sr=22050):
        self.audio_int16_array = samples
        self.sample_rate = sr


def _fake_voice(sr=22050):
    voice = mock.MagicMock()
    voice.synthesize = lambda text, cfg: [
        _FakeChunk(np.full(sr // 10, 100, dtype=np.int16), sr)
    ]
    return voice


def _patched_piper():
    fake_piper = mock.MagicMock()
    fake_piper.PiperVoice.load.return_value = _fake_voice()
    return mock.patch.dict(
        "sys.modules", {"piper": fake_piper, "piper.config": mock.MagicMock()}
    )


def test_synthesize_writes_valid_wav_and_timings(tmp_path):
    import wave

    with (
        _patched_piper(),
        mock.patch.object(
            piper_tts,
            "ensure_voice",
            return_value=(tmp_path / "m.onnx", tmp_path / "c.json"),
        ),
    ):
        out = tmp_path / "out.wav"
        duration, timings = piper_tts.synthesize("Um gol. Dois gols!", out)
    with wave.open(str(out), "rb") as wav:
        assert wav.getframerate() == 22050
        assert wav.getnchannels() == 1
        frames = wav.getnframes()
    assert frames / 22050 == pytest.approx(duration, abs=0.01)
    # two sentences of 0.1s + one 0.16s gap
    assert duration == pytest.approx(0.36, abs=0.01)
    assert [t[0] for t in timings] == ["Um gol.", "Dois gols!"]
    assert timings[0][1:] == pytest.approx((0.0, 0.1), abs=0.01)
    assert timings[1][1:] == pytest.approx((0.26, 0.36), abs=0.01)


def test_synthesize_applies_length_scale_floor(tmp_path):
    captured = {}

    fake_piper = mock.MagicMock()
    voice = _fake_voice()

    def capture(text, cfg):
        captured["length_scale"] = cfg.length_scale
        return voice.synthesize(text, cfg)

    fake_piper.PiperVoice.load.return_value.synthesize = capture
    fake_config = mock.MagicMock()
    with (
        mock.patch.dict(
            "sys.modules", {"piper": fake_piper, "piper.config": fake_config}
        ),
        mock.patch.object(
            piper_tts,
            "ensure_voice",
            return_value=(tmp_path / "m.onnx", tmp_path / "c.json"),
        ),
    ):
        piper_tts.synthesize("Um gol.", tmp_path / "x.wav", length_scale=0.1)
    assert captured["length_scale"] == 0.5


def test_synthesize_requires_piper_package(tmp_path):
    with mock.patch.dict("sys.modules", {"piper": None}):
        with pytest.raises(PiperError, match="piper-tts is not installed"):
            piper_tts.synthesize("texto", tmp_path / "x.wav")


def test_synthesize_rejects_empty_text(tmp_path):
    with pytest.raises(PiperError, match="Empty narration"):
        piper_tts.synthesize("   ", tmp_path / "x.wav")


def test_unknown_tts_engine_raises_instead_of_silent_cloud(monkeypatch):
    from autoshorts.modules import tts_system

    monkeypatch.setattr(tts_system, "TTS_ENGINE", "pipre")
    with pytest.raises(ValueError, match="Unknown TTS_ENGINE 'pipre'"):
        tts_system.TTSSystem()


@pytest.mark.parametrize(
    "rate,expected",
    [
        ("+20%", 1 / 1.2),
        ("-10%", 1 / 0.9),
        ("+0%", 1.0),
        ("junk", 1.0),
        ("+500%", 0.5),  # clamped to the supported floor
        ("-90%", 2.0),  # clamped to the supported ceiling
    ],
)
def test_rate_to_length_scale(rate, expected):
    from autoshorts.modules.tts_system import _rate_to_length_scale

    assert _rate_to_length_scale(rate) == pytest.approx(expected, abs=0.001)


@pytest.mark.asyncio
async def test_tts_system_piper_engine_writes_mp3(tmp_path, monkeypatch):
    from autoshorts.modules import tts_system

    monkeypatch.setattr(tts_system, "TTS_ENGINE", "piper")
    captured = {}

    def fake_synthesize(text, wav_path, *, length_scale=1.0):
        captured["length_scale"] = length_scale
        Path(wav_path).write_bytes(b"RIFFfake")
        return 1.0, [("Frase um. Frase dois.", 0.0, 1.0)]

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
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
    # ffmpeg converts the wav with the expected args
    cmd = captured["cmd"]
    assert cmd[0] == "ffmpeg" and cmd[1] == "-y"
    assert cmd[cmd.index("-i") + 1].endswith("audio.wav")
    assert "libmp3lame" in cmd
    assert cmd[cmd.index("-q:a") + 1] == "4"
    assert cmd[cmd.index("-af") + 1] == "alimiter=limit=0.89:level=disabled"
    assert cmd[-1].endswith("audio.mp3")
    # the configured TTS_RATE (+20% default) reaches Piper as a length_scale
    assert captured["length_scale"] == pytest.approx(1 / 1.2, abs=0.001)


@pytest.mark.asyncio
async def test_tts_system_piper_subtitles_follow_sentence_timings(
    tmp_path, monkeypatch
):
    from autoshorts.modules import tts_system

    monkeypatch.setattr(tts_system, "TTS_ENGINE", "piper")
    timings = [
        ("Um gol bonito demais para esquecer agora.", 0.0, 2.0),
        ("Dois gols.", 2.16, 3.0),
    ]

    def fake_synthesize(text, wav_path, *, length_scale=1.0):
        Path(wav_path).write_bytes(b"RIFFfake")
        return 3.0, timings

    def fake_run(cmd, **kwargs):
        if "ffprobe" in cmd[0]:
            return mock.MagicMock(stdout="3.0\n", returncode=0)
        Path(cmd[-1]).write_bytes(b"ID3fake")
        return mock.MagicMock(returncode=0, stderr="")

    monkeypatch.setattr("autoshorts.modules.piper_tts.synthesize", fake_synthesize)
    monkeypatch.setattr(tts_system.subprocess, "run", fake_run)

    tts = tts_system.TTSSystem()
    _, vtt_file, duration = await tts.generate_audio_and_subtitles(
        ["Um gol bonito demais para esquecer agora.", "Dois gols."], tmp_path
    )
    assert duration == pytest.approx(3.0)
    cues = [
        line
        for line in Path(vtt_file).read_text(encoding="utf-8-sig").splitlines()
        if "-->" in line
    ]
    # sentence 1 (7 words -> 2 chunks) + sentence 2 (2 words -> 1 chunk)
    assert len(cues) == 3
    # first cue starts at 0; second-sentence cue never starts before its audio
    assert "00:00:00.000" in cues[0]
    assert cues[1].startswith("00:00:01.")  # ~1.43s into a 2s span, not 1.5s even-split
    assert "00:00:02.160" in cues[2]


def test_load_failure_invalidates_cache_and_retries_once(tmp_path):
    """A corrupt-but-plausibly-sized cached model is dropped and re-downloaded."""
    model = tmp_path / "m.onnx"
    config = tmp_path / "c.json"
    model.write_bytes(b"corrupt")
    config.write_bytes(b"{}")

    calls = {"load": 0, "ensure": 0}

    def fake_ensure(*args, **kwargs):
        calls["ensure"] += 1
        # Re-create the files as a fresh download would.
        model.write_bytes(b"corrupt" if calls["ensure"] == 1 else b"fresh")
        config.write_bytes(b"{}")
        return model, config

    fake_piper = mock.MagicMock()

    def fake_load(*args, **kwargs):
        calls["load"] += 1
        if calls["load"] == 1:
            raise RuntimeError("invalid protobuf")
        return _fake_voice()

    fake_piper.PiperVoice.load.side_effect = fake_load
    with (
        mock.patch.dict(
            "sys.modules", {"piper": fake_piper, "piper.config": mock.MagicMock()}
        ),
        mock.patch.object(piper_tts, "ensure_voice", side_effect=fake_ensure),
    ):
        out = tmp_path / "out.wav"
        duration, _ = piper_tts.synthesize("Um gol.", out)
    assert calls["ensure"] == 2
    assert calls["load"] == 2
    assert duration > 0


def test_load_failure_persistent_raises_after_single_retry(tmp_path):
    """Retry is bounded: one cache refresh, then PiperError."""
    calls = {"ensure": 0}

    def fake_ensure(*args, **kwargs):
        calls["ensure"] += 1
        return tmp_path / "m.onnx", tmp_path / "c.json"

    fake_piper = mock.MagicMock()
    fake_piper.PiperVoice.load.side_effect = RuntimeError("still broken")
    with (
        mock.patch.dict(
            "sys.modules", {"piper": fake_piper, "piper.config": mock.MagicMock()}
        ),
        mock.patch.object(piper_tts, "ensure_voice", side_effect=fake_ensure),
    ):
        with pytest.raises(PiperError, match="Failed to load Piper voice"):
            piper_tts.synthesize("Um gol.", tmp_path / "out.wav")
    assert calls["ensure"] == 2  # initial + one refresh, never more


def test_vtt_timings_starting_after_duration_are_dropped(tmp_path):
    from autoshorts.modules.subtitle_system import SubtitleGenerator

    timings = [
        ("Primeira frase do vídeo aqui.", 0.0, 2.0),
        ("Frase que começa depois do fim do áudio.", 5.0, 6.0),
    ]
    out = tmp_path / "subs.vtt"
    SubtitleGenerator.generate_vtt_from_sentence_timings(timings, 3.0, str(out))
    text = out.read_text(encoding="utf-8-sig")
    cues = [ln for ln in text.splitlines() if "-->" in ln]
    assert cues, "first sentence must produce cues"
    for cue in cues:
        start = cue.split(" --> ")[0]
        h, m, rest = start.split(":")
        seconds = int(h) * 3600 + int(m) * 60 + float(rest)
        assert seconds < 3.0
    assert "tardia" not in text


def test_vtt_timing_straddling_duration_is_clamped(tmp_path):
    from autoshorts.modules.subtitle_system import SubtitleGenerator

    timings = [("Uma frase longa o bastante para dividir em partes.", 1.0, 5.0)]
    out = tmp_path / "subs.vtt"
    SubtitleGenerator.generate_vtt_from_sentence_timings(timings, 2.0, str(out))
    text = out.read_text(encoding="utf-8-sig")
    cues = [ln for ln in text.splitlines() if "-->" in ln]
    assert len(cues) == 2  # 8 words -> two 5-word chunks

    def to_seconds(ts):
        h, m, rest = ts.split(":")
        return int(h) * 3600 + int(m) * 60 + float(rest)

    for cue in cues:
        start_ts, end_ts = cue.split(" --> ")
        start_s, end_s = to_seconds(start_ts), to_seconds(end_ts)
        assert start_s < end_s <= 2.0
    assert to_seconds(cues[-1].split(" --> ")[1]) == pytest.approx(2.0)


def test_vtt_has_no_one_word_trailing_cue(tmp_path):
    from autoshorts.modules.subtitle_system import SubtitleGenerator

    timings = [("Vem com estojo de carregamento e custa menos de 100 reais.", 0.0, 4.0)]
    out = tmp_path / "subs.vtt"
    SubtitleGenerator.generate_vtt_from_sentence_timings(timings, 4.0, str(out))
    lines = out.read_text(encoding="utf-8-sig").splitlines()
    texts = [lines[i + 1] for i, ln in enumerate(lines) if "-->" in ln]
    assert (
        " ".join(texts) == "Vem com estojo de carregamento e custa menos de 100 reais."
    )
    assert all(1 < len(t.split()) <= 5 for t in texts)
    assert len(texts) == 3
