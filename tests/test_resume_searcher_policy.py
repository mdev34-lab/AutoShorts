"""Per-instance public-only web policy for ImageSearcher."""

import json
from pathlib import Path
from unittest.mock import Mock

import pytest
from PIL import Image

from autoshorts.modules import image_searcher as mod
from autoshorts.modules.image_searcher import ImageSearcher


def test_public_only_injected_no_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr(mod, "IMAGE_CACHE_DIR", tmp_path)
    monkeypatch.setattr(mod, "ALLOW_UNVERIFIED_IMAGES", False)
    monkeypatch.setattr(mod, "IMAGE_AI_FALLBACK", True)
    monkeypatch.setattr(mod, "IMAGE_PROVIDERS", ["commons", "ddg"])
    monkeypatch.setattr(mod, "OUTPUT_DIR", tmp_path / "out")

    search = Mock(return_value=[])
    download = Mock(return_value=True)
    searcher = ImageSearcher(
        search=search, download=download, providers=("ddg",), ai_fallback=False
    )
    searcher._fallback_to_ai = Mock(side_effect=AssertionError("AI fallback called"))

    with pytest.raises(RuntimeError, match="IMAGE_AI_FALLBACK"):
        searcher.get_images(["a bowl of fruit"])
    search.assert_called()
    download.assert_not_called()
    searcher._fallback_to_ai.assert_not_called()


def test_public_only_pixel_rejection_aborts(tmp_path, monkeypatch):
    monkeypatch.setattr(mod, "IMAGE_CACHE_DIR", tmp_path)
    monkeypatch.setattr(mod, "ALLOW_UNVERIFIED_IMAGES", True)
    monkeypatch.setattr(mod, "IMAGE_AI_FALLBACK", True)
    monkeypatch.setattr(mod, "OUTPUT_DIR", tmp_path / "out")

    calls = []

    def fake_check(path, prompt):
        calls.append((path, prompt, path.exists()))
        return "bad pixels"

    monkeypatch.setattr(mod.image_content, "check_image_content", fake_check)

    url = "https://example.com/p.jpg"

    def fake_download(download_url, path):
        Image.new("RGB", (400, 300), "gray").save(path, "JPEG")
        return True

    search = Mock(return_value=[{"image": url, "title": "fruit bowl"}])
    searcher = ImageSearcher(
        search=search, download=fake_download, providers=("ddg",), ai_fallback=False
    )
    searcher._fallback_to_ai = Mock(side_effect=AssertionError("AI fallback called"))

    with pytest.raises(RuntimeError, match="IMAGE_AI_FALLBACK"):
        searcher.get_images(["fruit bowl"])
    assert calls and calls[0][1] == "fruit bowl"
    assert calls[0][2] is True
    search.assert_called()
    searcher._fallback_to_ai.assert_not_called()


def test_public_only_rejects_cached_ai(tmp_path, monkeypatch):
    monkeypatch.setattr(mod, "IMAGE_CACHE_DIR", tmp_path)
    monkeypatch.setattr(mod, "ALLOW_UNVERIFIED_IMAGES", False)
    monkeypatch.setattr(mod, "IMAGE_AI_FALLBACK", True)
    monkeypatch.setattr(mod, "OUTPUT_DIR", tmp_path / "out")

    search = Mock(return_value=[])
    searcher = ImageSearcher(
        search=search,
        download=Mock(return_value=True),
        providers=("ddg",),
        ai_fallback=False,
    )
    searcher._fallback_to_ai = Mock(side_effect=AssertionError("AI fallback called"))
    prompt = "a bowl of fruit"
    key = searcher._instance_cache_key(prompt)
    cached = tmp_path / f"{key}.jpg"
    cached.write_bytes(b"ai-bytes")
    ImageSearcher._sidecar_path(cached).write_text(
        json.dumps(
            {
                "schema": 1,
                "kind": "ai",
                "provider": "ai",
                "license": "AI-generated - no source license",
                "title": "AI-generated image",
            }
        )
    )

    with pytest.raises(RuntimeError, match="IMAGE_AI_FALLBACK"):
        searcher.get_images([prompt])
    search.assert_called()
    searcher._fallback_to_ai.assert_not_called()


def test_public_only_success_writes_web_provenance(tmp_path, monkeypatch):
    monkeypatch.setattr(mod, "IMAGE_CACHE_DIR", tmp_path)
    monkeypatch.setattr(mod, "ALLOW_UNVERIFIED_IMAGES", True)
    monkeypatch.setattr(mod, "IMAGE_AI_FALLBACK", True)
    monkeypatch.setattr(mod, "OUTPUT_DIR", tmp_path / "out")
    monkeypatch.setattr(
        mod.image_content, "check_image_content", lambda path, prompt: None
    )

    url = "https://example.com/p.jpg"

    def fake_download(download_url, path):
        Image.new("RGB", (400, 300), "gray").save(path, "JPEG")
        return True

    search = Mock(return_value=[{"image": url, "title": "fruit bowl"}])
    searcher = ImageSearcher(
        search=search, download=fake_download, providers=("ddg",), ai_fallback=False
    )
    searcher._fallback_to_ai = Mock(side_effect=AssertionError("AI fallback called"))

    result = searcher.get_images(["fruit bowl"])
    assert result and result[0].endswith(".jpg")
    sidecar = ImageSearcher._sidecar_path(Path(result[0]))
    record = json.loads(sidecar.read_text())
    assert record["kind"] == "web"
    search.assert_called()
    searcher._fallback_to_ai.assert_not_called()


def test_per_instance_independent_and_defaults(tmp_path, monkeypatch):
    monkeypatch.setattr(mod, "IMAGE_CACHE_DIR", tmp_path)
    monkeypatch.setattr(mod, "IMAGE_PROVIDERS", ["commons", "ddg"])
    monkeypatch.setattr(mod, "IMAGE_AI_FALLBACK", True)
    monkeypatch.setattr(mod, "ALLOW_UNVERIFIED_IMAGES", False)

    prompt = "a bowl of fruit"
    default = ImageSearcher()
    assert default._instance_cache_key(prompt) == ImageSearcher._cache_key(prompt)
    assert default.ai_fallback == mod.IMAGE_AI_FALLBACK
    assert default.providers == tuple(mod.IMAGE_PROVIDERS)

    s1 = ImageSearcher(
        search=Mock(return_value=[]),
        download=Mock(return_value=True),
        providers=("ddg",),
        ai_fallback=False,
    )
    s2 = ImageSearcher(
        search=Mock(return_value=[]),
        download=Mock(return_value=True),
        providers=("commons",),
        ai_fallback=True,
    )
    assert s1._instance_cache_key(prompt) != s2._instance_cache_key(prompt)
    assert s1.ai_fallback is False
    assert s2.ai_fallback is True
    assert mod.IMAGE_AI_FALLBACK is True
    assert mod.IMAGE_PROVIDERS == ["commons", "ddg"]


def test_constructor_validation():
    with pytest.raises(TypeError):
        ImageSearcher(search="not callable")
    with pytest.raises(TypeError):
        ImageSearcher(download="not callable")
    with pytest.raises(TypeError):
        ImageSearcher(providers="ddg")
    with pytest.raises(TypeError):
        ImageSearcher(providers=("ddg", 1))
    with pytest.raises(TypeError):
        ImageSearcher(ai_fallback=1)


# --------------------------------------------------------------------------
# W4: the FINAL transformed bytes are gated, and a rejected cache entry is a
# miss (never a deletion). OCR unavailability fails closed before any refetch.
# --------------------------------------------------------------------------

_FINAL_PROMPT = "fruit bowl"
_FIRST_URL = "https://example.com/first.jpg"
_SECOND_URL = "https://example.com/second.jpg"


def _synthetic_jpeg(path: Path, colour: str = "gray") -> None:
    Image.new("RGB", (400, 300), colour).save(path, "JPEG")


def _disk_layout(cache_dir: Path) -> dict:
    """Every cache file and sidecar with its exact bytes (mtime-independent)."""
    return {
        path.name: path.read_bytes()
        for path in sorted(cache_dir.iterdir())
        if path.is_file()
    }


def _is_staged_final(path: Path) -> bool:
    """A uniquely-named staging file the transform is written to.

    Matches on the stable prefix/suffix only; the middle is a random per-call
    name, so no test may depend on an exact staging filename.
    """
    return path.name.startswith("dl_") and path.name.endswith("_final.jpg")


def _final_gate_searcher(tmp_path, monkeypatch, gate, candidates, downloads):
    """Searcher whose conversion writes a distinct final image per source URL.

    ``gate(path, prompt)`` is called for the original download and again for
    the transformed bytes while they are still in the staging file; the test
    decides which of the two fails.
    """
    monkeypatch.setattr(mod, "IMAGE_CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(mod, "ALLOW_UNVERIFIED_IMAGES", True)
    monkeypatch.setattr(mod, "IMAGE_AI_FALLBACK", True)
    monkeypatch.setattr(mod, "OUTPUT_DIR", tmp_path / "out")
    monkeypatch.setattr(mod.image_content, "check_image_content", gate)

    searcher = ImageSearcher(
        search=Mock(return_value=[]),
        download=Mock(return_value=True),
        providers=("ddg",),
        ai_fallback=False,
    )
    searcher._candidates = lambda prompt: list(candidates)
    searcher._fallback_to_ai = Mock(side_effect=AssertionError("AI fallback called"))

    def fake_download(url, path):
        downloads.append(url)
        _synthetic_jpeg(Path(path), "red" if url == _FIRST_URL else "blue")
        return True

    def fake_resize(source, destination):
        _synthetic_jpeg(Path(destination), "green")

    searcher.download_image = fake_download
    searcher.resize_to_fill = fake_resize
    return searcher


def _candidate(url: str):
    from autoshorts.modules import image_sources

    return image_sources.SourceCandidate(
        url=url,
        provider="ddg",
        license=None,
        fit_evidence="fruit,bowl",
        title="fruit bowl",
    )


def test_final_transformed_rejection_selects_next_candidate(tmp_path, monkeypatch):
    rejected_prompts = []
    final_calls = []

    def gate(path, prompt):
        rejected_prompts.append(prompt)
        if not _is_staged_final(Path(path)):
            return None  # the original downloaded bytes pass
        final_calls.append(Path(path).name)
        # Final transformed bytes: reject the first candidate's, accept the next.
        if len(final_calls) == 1:
            return "final pixels contradict the prompt"
        return None

    downloads = []
    searcher = _final_gate_searcher(
        tmp_path,
        monkeypatch,
        gate,
        [_candidate(_FIRST_URL), _candidate(_SECOND_URL)],
        downloads,
    )
    result = searcher.get_images([_FINAL_PROMPT])

    assert downloads == [_FIRST_URL, _SECOND_URL]  # second candidate used
    assert len(result) == 1
    sidecar = ImageSearcher._sidecar_path(Path(result[0]))
    record = json.loads(sidecar.read_text())
    assert record["image_url"] == _SECOND_URL
    credits = (tmp_path / "out" / "image_credits.txt").read_text()
    assert _SECOND_URL in credits
    assert _FIRST_URL not in credits
    # the rejected first candidate left no accepted sidecar of its own
    leftovers = [
        path.name
        for path in (tmp_path / "cache").iterdir()
        if path.name.startswith("dl_") or path.name.endswith(".credits.json")
    ]
    assert leftovers == [sidecar.name]
    # two originals passed, the first final was rejected, the second accepted
    assert rejected_prompts.count(_FINAL_PROMPT) == 4
    assert len(final_calls) == 2
    searcher._fallback_to_ai.assert_not_called()


def test_every_final_rejection_keeps_noimage_failure(tmp_path, monkeypatch):
    def gate(path, prompt):
        # only the transformed bytes are rejected; originals pass
        return "final pixels rejected" if _is_staged_final(path) else None

    downloads = []
    searcher = _final_gate_searcher(
        tmp_path,
        monkeypatch,
        gate,
        [_candidate(_FIRST_URL), _candidate(_SECOND_URL)],
        downloads,
    )
    with pytest.raises(RuntimeError, match="IMAGE_AI_FALLBACK"):
        searcher.get_images([_FINAL_PROMPT])

    assert downloads == [_FIRST_URL, _SECOND_URL]
    searcher._fallback_to_ai.assert_not_called()
    # no candidate was accepted: no kept image and no credits block for it
    cache = tmp_path / "cache"
    assert [p.name for p in cache.iterdir() if p.suffix == ".jpg"] == []
    credits = tmp_path / "out" / "image_credits.txt"
    assert not credits.exists() or _FIRST_URL not in credits.read_text()


def test_original_rejection_never_reaches_transformation(tmp_path, monkeypatch):
    frames = []

    def gate(path, prompt):
        return "original pixels rejected"

    downloads = []
    searcher = _final_gate_searcher(
        tmp_path, monkeypatch, gate, [_candidate(_FIRST_URL)], downloads
    )
    searcher.resize_to_fill = Mock(side_effect=frames.append)
    with pytest.raises(RuntimeError, match="IMAGE_AI_FALLBACK"):
        searcher.get_images([_FINAL_PROMPT])

    assert frames == []  # framing/transformation never called
    assert downloads == [_FIRST_URL]
    searcher._fallback_to_ai.assert_not_called()


def test_rejected_cache_hit_is_a_miss_not_a_deletion(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    cache.mkdir()
    monkeypatch.setattr(mod, "IMAGE_CACHE_DIR", cache)
    monkeypatch.setattr(mod, "ALLOW_UNVERIFIED_IMAGES", True)
    monkeypatch.setattr(mod, "IMAGE_AI_FALLBACK", True)
    monkeypatch.setattr(mod, "OUTPUT_DIR", tmp_path / "out")

    searcher = ImageSearcher(
        search=Mock(return_value=[]),
        download=Mock(return_value=True),
        providers=("ddg",),
        ai_fallback=False,
    )
    prompt = _FINAL_PROMPT
    cached = cache / f"{searcher._instance_cache_key(prompt)}.jpg"
    _synthetic_jpeg(cached)
    ImageSearcher._write_provenance_sidecar(cached, _candidate(_FIRST_URL), prompt)
    assert searcher._instance_usable_cache(cached, prompt) is True
    seeded_bytes = cached.read_bytes()

    gate_calls = []

    def gate(path, prompt_arg):
        gate_calls.append(Path(path).name)
        # The seeded hit's own pixel re-check fails; nothing else does.
        if Path(path) == cached and len(gate_calls) == 1:
            return "cached pixels rejected"
        return None

    monkeypatch.setattr(mod.image_content, "check_image_content", gate)

    downloads = []

    def fake_download(url, path):
        downloads.append(url)
        _synthetic_jpeg(Path(path), "blue")
        return True

    searcher.download_image = fake_download
    searcher.resize_to_fill = lambda src, dst: Path(dst).write_bytes(
        b"fresh-final-bytes"
    )
    searcher._candidates = lambda q: [_candidate(_SECOND_URL)]
    searcher._fallback_to_ai = Mock(side_effect=AssertionError("AI fallback called"))

    result = searcher.get_images([prompt])

    # A miss re-runs the normal candidate flow with no manual cleanup step.
    assert gate_calls[0] == cached.name
    assert downloads == [_SECOND_URL]
    assert result == [str(cached)]
    record = json.loads(ImageSearcher._sidecar_path(cached).read_text())
    assert record["image_url"] == _SECOND_URL  # republished only after acceptance
    assert cached.read_bytes() != seeded_bytes
    assert (tmp_path / "out" / "image_credits.txt").exists()
    assert _SECOND_URL in (tmp_path / "out" / "image_credits.txt").read_text()
    searcher._fallback_to_ai.assert_not_called()


def test_final_rejection_after_cache_miss_leaves_seeded_bytes_and_sidecar(
    tmp_path, monkeypatch
):
    """The transform is staged, so a rejected replacement never overwrites the
    shared/published cache entry it was about to replace."""
    cache = tmp_path / "cache"
    cache.mkdir()
    monkeypatch.setattr(mod, "IMAGE_CACHE_DIR", cache)
    monkeypatch.setattr(mod, "ALLOW_UNVERIFIED_IMAGES", True)
    monkeypatch.setattr(mod, "IMAGE_AI_FALLBACK", True)
    monkeypatch.setattr(mod, "OUTPUT_DIR", tmp_path / "out")

    searcher = ImageSearcher(
        search=Mock(return_value=[]),
        download=Mock(return_value=True),
        providers=("ddg",),
        ai_fallback=False,
    )
    prompt = _FINAL_PROMPT
    cached = cache / f"{searcher._instance_cache_key(prompt)}.jpg"
    _synthetic_jpeg(cached)
    ImageSearcher._write_provenance_sidecar(cached, _candidate(_FIRST_URL), prompt)
    seeded = _disk_layout(cache)

    def gate(path, prompt_arg):
        # stale seeded hit is a miss; the fresh replacement fails its final gate
        return (
            "rejected" if _is_staged_final(Path(path)) or Path(path) == cached else None
        )

    monkeypatch.setattr(mod.image_content, "check_image_content", gate)

    def fake_download(url, path):
        _synthetic_jpeg(Path(path), "blue")
        return True

    replaced = []

    def fake_resize(source, destination):
        replaced.append(Path(destination))
        Path(destination).write_bytes(b"transformed-but-unwanted-bytes")

    searcher.download_image = fake_download
    searcher.resize_to_fill = fake_resize
    searcher._candidates = lambda q: [_candidate(_SECOND_URL)]
    searcher._fallback_to_ai = Mock(side_effect=AssertionError("AI fallback called"))

    with pytest.raises(RuntimeError, match="IMAGE_AI_FALLBACK"):
        searcher.get_images([prompt])

    # transformation never targeted the published entry, only a staging file
    assert replaced and all(path != cached for path in replaced)
    assert all(_is_staged_final(path) for path in replaced)
    # the seeded bytes and sidecar are untouched, and no staging file survived
    assert _disk_layout(cache) == seeded
    searcher._fallback_to_ai.assert_not_called()


def test_rejected_cache_hit_shared_file_survives_unrepublished_miss(
    tmp_path, monkeypatch
):
    """A rejected hit with no replacement candidate leaves the file untouched."""
    cache = tmp_path / "cache"
    cache.mkdir()
    monkeypatch.setattr(mod, "IMAGE_CACHE_DIR", cache)
    monkeypatch.setattr(mod, "ALLOW_UNVERIFIED_IMAGES", True)
    monkeypatch.setattr(mod, "IMAGE_AI_FALLBACK", True)
    monkeypatch.setattr(mod, "OUTPUT_DIR", tmp_path / "out")

    searcher = ImageSearcher(
        search=Mock(return_value=[]),
        download=Mock(return_value=True),
        providers=("ddg",),
        ai_fallback=False,
    )
    prompt = _FINAL_PROMPT
    key = searcher._instance_cache_key(prompt)
    cached = cache / f"{key}.jpg"
    _synthetic_jpeg(cached)
    ImageSearcher._write_provenance_sidecar(cached, _candidate(_FIRST_URL), prompt)
    seeded = _disk_layout(cache)

    monkeypatch.setattr(
        mod.image_content, "check_image_content", lambda path, prompt_arg: "rejected"
    )
    searcher._candidates = lambda q: []
    searcher._fallback_to_ai = Mock(side_effect=AssertionError("AI fallback called"))

    with pytest.raises(RuntimeError, match="IMAGE_AI_FALLBACK"):
        searcher.get_images([prompt])
    assert _disk_layout(cache) == seeded  # nothing deleted or rewritten
    searcher._fallback_to_ai.assert_not_called()


def test_public_only_cached_ocr_unavailable_aborts_before_refetch(
    tmp_path, monkeypatch
):
    cache = tmp_path / "cache"
    cache.mkdir()
    monkeypatch.setattr(mod, "IMAGE_CACHE_DIR", cache)
    monkeypatch.setattr(mod, "ALLOW_UNVERIFIED_IMAGES", True)
    monkeypatch.setattr(mod, "IMAGE_AI_FALLBACK", True)
    monkeypatch.setattr(mod, "OUTPUT_DIR", tmp_path / "out")

    search_calls = []

    def saved_search(query):
        search_calls.append(query)
        return [{"image": _SECOND_URL, "title": "fruit bowl"}]

    searcher = ImageSearcher(search=saved_search, providers=("ddg",), ai_fallback=True)
    prompt = _FINAL_PROMPT
    key = searcher._instance_cache_key(prompt)
    cached = cache / f"{key}.jpg"
    _synthetic_jpeg(cached)
    ImageSearcher._write_provenance_sidecar(cached, _candidate(_FIRST_URL), prompt)
    seeded = _disk_layout(cache)

    def unavailable(path, prompt_arg):
        raise mod.image_content.ContentUnavailableError("OCR unavailable")

    monkeypatch.setattr(mod.image_content, "check_image_content", unavailable)

    with pytest.raises(RuntimeError, match="could not be verified"):
        searcher.get_images([prompt])
    # aborted BEFORE any search/refetch, and the seeded cache is intact
    assert search_calls == []
    assert _disk_layout(cache) == seeded


def test_public_only_cached_ocr_unavailable_never_downloads(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    cache.mkdir()
    monkeypatch.setattr(mod, "IMAGE_CACHE_DIR", cache)
    monkeypatch.setattr(mod, "ALLOW_UNVERIFIED_IMAGES", True)
    monkeypatch.setattr(mod, "IMAGE_AI_FALLBACK", True)
    monkeypatch.setattr(mod, "OUTPUT_DIR", tmp_path / "out")

    downloads = []

    def saved_download(url, path):
        downloads.append(url)
        return True

    searcher = ImageSearcher(
        download=saved_download, providers=("ddg",), ai_fallback=True
    )
    prompt = _FINAL_PROMPT
    cached = cache / f"{searcher._instance_cache_key(prompt)}.jpg"
    _synthetic_jpeg(cached)
    ImageSearcher._write_provenance_sidecar(cached, _candidate(_FIRST_URL), prompt)
    seeded = _disk_layout(cache)

    monkeypatch.setattr(
        mod.image_content,
        "check_image_content",
        Mock(
            side_effect=mod.image_content.ContentUnavailableError("no tesseract"),
            name="check_image_content",
        ),
    )
    with pytest.raises(RuntimeError, match="could not be verified"):
        searcher.get_images([prompt])
    assert downloads == []
    assert _disk_layout(cache) == seeded


def test_valid_cached_image_is_reused_without_refetch(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    cache.mkdir()
    monkeypatch.setattr(mod, "IMAGE_CACHE_DIR", cache)
    monkeypatch.setattr(mod, "ALLOW_UNVERIFIED_IMAGES", True)
    monkeypatch.setattr(mod, "IMAGE_AI_FALLBACK", True)
    monkeypatch.setattr(mod, "OUTPUT_DIR", tmp_path / "out")

    searcher = ImageSearcher(
        search=Mock(return_value=[]),
        download=Mock(return_value=True),
        providers=("ddg",),
        ai_fallback=True,
    )
    prompt = _FINAL_PROMPT
    cached = cache / f"{searcher._instance_cache_key(prompt)}.jpg"
    _synthetic_jpeg(cached)
    ImageSearcher._write_provenance_sidecar(cached, _candidate(_FIRST_URL), prompt)
    seeded = _disk_layout(cache)

    monkeypatch.setattr(
        mod.image_content, "check_image_content", lambda path, prompt: None
    )
    searcher._candidates = Mock(side_effect=AssertionError("no refetch"))
    searcher._fallback_to_ai = Mock(side_effect=AssertionError("AI fallback called"))

    assert searcher.get_images([prompt]) == [str(cached)]
    searcher._candidates.assert_not_called()
    searcher._fallback_to_ai.assert_not_called()
    after = _disk_layout(cache)
    assert after[cached.name] == seeded[cached.name]
    assert (
        after[ImageSearcher._sidecar_path(cached).name]
        == seeded[ImageSearcher._sidecar_path(cached).name]
    )


def test_cached_pixel_gate_leaves_ai_entries_alone(tmp_path, monkeypatch):
    """The new cache re-check gates web pixels only; AI entries keep policy."""
    cache = tmp_path / "cache"
    cache.mkdir()
    monkeypatch.setattr(mod, "IMAGE_CACHE_DIR", cache)
    monkeypatch.setattr(mod, "ALLOW_UNVERIFIED_IMAGES", True)
    monkeypatch.setattr(mod, "IMAGE_AI_FALLBACK", True)
    monkeypatch.setattr(mod, "OUTPUT_DIR", tmp_path / "out")

    searcher = ImageSearcher(
        search=Mock(return_value=[]),
        download=Mock(return_value=True),
        providers=("ddg",),
        ai_fallback=True,
    )
    prompt = _FINAL_PROMPT
    cached = cache / f"{searcher._instance_cache_key(prompt)}.jpg"
    cached.write_bytes(b"ai-bytes")
    ImageSearcher._sidecar_path(cached).write_text(
        json.dumps(
            {
                "schema": 1,
                "kind": "ai",
                "provider": "ai",
                "license": "AI-generated - no source license",
                "title": "AI-generated image",
            }
        )
    )
    monkeypatch.setattr(
        mod.image_content,
        "check_image_content",
        Mock(side_effect=AssertionError("AI cache must not be OCR-gated")),
    )
    searcher._candidates = Mock(side_effect=AssertionError("no refetch"))
    assert searcher.get_images([prompt]) == [str(cached)]
    searcher._candidates.assert_not_called()


def test_final_gate_keeps_verified_rights_policy_active(tmp_path, monkeypatch):
    """The new final gate does not bypass the rights gate: an unlicensed
    candidate is skipped before any download even when pixels would pass."""
    monkeypatch.setattr(mod, "IMAGE_CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(mod, "ALLOW_UNVERIFIED_IMAGES", False)
    monkeypatch.setattr(mod, "IMAGE_AI_FALLBACK", True)
    monkeypatch.setattr(mod, "OUTPUT_DIR", tmp_path / "out")
    monkeypatch.setattr(mod, "IMAGE_PROVIDERS", ["commons"])

    downloads = []
    searcher = ImageSearcher(
        search=Mock(return_value=[]),
        download=Mock(return_value=True),
        providers=("commons",),
        ai_fallback=True,
    )

    def fake_download(url, path):
        downloads.append(url)
        _synthetic_jpeg(Path(path))
        return True

    searcher.download_image = fake_download
    searcher._candidates = lambda q: [_candidate(_FIRST_URL)]  # license=None
    searcher._fallback_to_ai = Mock(
        side_effect=lambda prompts, paths: paths.append("ai.jpg")
    )

    assert searcher.get_images([_FINAL_PROMPT]) == ["ai.jpg"]
    assert downloads == []  # rights gate still refuses the unlicensed web image
    searcher._fallback_to_ai.assert_called_once()


def _seeded_replacement_case(tmp_path, monkeypatch, *, ai_fallback=True):
    cache = tmp_path / "cache"
    monkeypatch.setattr(mod, "IMAGE_CACHE_DIR", cache)
    monkeypatch.setattr(mod, "ALLOW_UNVERIFIED_IMAGES", True)
    monkeypatch.setattr(mod, "IMAGE_AI_FALLBACK", True)
    monkeypatch.setattr(mod, "OUTPUT_DIR", tmp_path / "out")

    cached_ref = {}
    gate_calls = []

    def gate(path, prompt):
        path = Path(path)
        gate_calls.append(path)
        if path == cached_ref["path"]:
            return "seeded cache pixels rejected"
        return None

    monkeypatch.setattr(mod.image_content, "check_image_content", gate)
    searcher = ImageSearcher(
        search=Mock(return_value=[]),
        download=Mock(return_value=True),
        providers=("ddg",),
        ai_fallback=ai_fallback,
    )
    prompt = _FINAL_PROMPT
    cached = cache / f"{searcher._instance_cache_key(prompt)}.jpg"
    cached_ref["path"] = cached
    _synthetic_jpeg(cached, "red")
    ImageSearcher._write_provenance_sidecar(cached, _candidate(_FIRST_URL), prompt)
    seeded = _disk_layout(cache)

    downloads = []

    def fake_download(url, path):
        downloads.append(url)
        _synthetic_jpeg(Path(path), "blue")
        return True

    def fake_resize(source, destination):
        _synthetic_jpeg(Path(destination), "green")

    searcher.download_image = fake_download
    searcher.resize_to_fill = fake_resize
    searcher._candidates = lambda query: [_candidate(_SECOND_URL)]
    searcher._fallback_to_ai = Mock(side_effect=AssertionError("AI fallback called"))
    return searcher, prompt, cached, cache, seeded, downloads, gate_calls


def test_readonly_0444_before_sidecar_write_failure_preserves_seeded_pair_and_credits(
    tmp_path, monkeypatch
):
    searcher, prompt, cached, cache, _, downloads, gate_calls = (
        _seeded_replacement_case(tmp_path, monkeypatch, ai_fallback=True)
    )
    sidecar = ImageSearcher._sidecar_path(cached)
    cached.chmod(0o444)
    sidecar.chmod(0o444)
    seeded = _disk_layout(cache)
    credits = tmp_path / "out" / "image_credits.txt"
    credits.parent.mkdir(parents=True)
    old_credits = f"previous published record: {_FIRST_URL}"
    credits.write_text(old_credits)
    searcher._stage_provenance_sidecar = Mock(
        side_effect=PermissionError("synthetic read-only 0444 failure before write")
    )

    with pytest.raises(mod._ProvenancePublishError, match="provenance"):
        searcher.get_images([prompt])

    assert downloads == [_SECOND_URL]
    assert cached in gate_calls and any(_is_staged_final(path) for path in gate_calls)
    assert _disk_layout(cache) == seeded
    assert cached.stat().st_mode & 0o777 == 0o444
    assert sidecar.stat().st_mode & 0o777 == 0o444
    assert credits.read_text() == old_credits
    assert _SECOND_URL not in credits.read_text()
    searcher._fallback_to_ai.assert_not_called()


def test_sidecar_replace_failure_rolls_back_seeded_pair_without_acceptance_or_credits(
    tmp_path, monkeypatch
):
    searcher, prompt, cached, cache, seeded, downloads, _gate_calls = (
        _seeded_replacement_case(tmp_path, monkeypatch, ai_fallback=True)
    )
    sidecar = ImageSearcher._sidecar_path(cached)
    credits = tmp_path / "out" / "image_credits.txt"
    credits.parent.mkdir(parents=True)
    old_credits = f"previous published record: {_FIRST_URL}"
    credits.write_text(old_credits)
    real_replace = mod.os.replace
    failed = []

    def fail_candidate_sidecar_once(source, destination):
        source_path, destination_path = Path(source), Path(destination)
        if (
            destination_path == sidecar
            and source_path.name.startswith("dl_")
            and source_path.name.endswith("_final.credits.json")
            and not failed
        ):
            failed.append(source_path)
            raise OSError("synthetic sidecar replace failure")
        return real_replace(source, destination)

    monkeypatch.setattr(mod.os, "replace", fail_candidate_sidecar_once)
    with pytest.raises(mod._ProvenancePublishError, match="provenance"):
        searcher.get_images([prompt])

    assert len(failed) == 1
    assert downloads == [_SECOND_URL]
    assert _disk_layout(cache) == seeded
    assert credits.read_text() == old_credits
    assert _SECOND_URL not in credits.read_text()
    searcher._fallback_to_ai.assert_not_called()


def test_successful_replacement_publishes_matching_pair_and_cleans_staging(
    tmp_path, monkeypatch
):
    searcher, prompt, cached, cache, seeded, downloads, _gate_calls = (
        _seeded_replacement_case(tmp_path, monkeypatch, ai_fallback=False)
    )
    credits = tmp_path / "out" / "image_credits.txt"
    credits.parent.mkdir(parents=True)
    credits.write_text(f"previous published record: {_FIRST_URL}")

    result = searcher.get_images([prompt])

    sidecar = ImageSearcher._sidecar_path(cached)
    record = ImageSearcher._read_sidecar(cached)
    assert result == [str(cached)]
    assert downloads == [_SECOND_URL]
    assert cached.read_bytes() != seeded[cached.name]
    assert record == ImageSearcher._provenance_record(_candidate(_SECOND_URL), prompt)
    assert set(_disk_layout(cache)) == {cached.name, sidecar.name}
    published_credits = credits.read_text()
    assert _SECOND_URL in published_credits
    assert _FIRST_URL not in published_credits
    searcher._fallback_to_ai.assert_not_called()
