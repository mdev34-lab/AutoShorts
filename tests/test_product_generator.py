"""Orchestration tests for ProductGenerator with all externals mocked."""

import asyncio
from unittest import mock

from autoshorts.generators import VIDEO_TYPES
from autoshorts.generators.product import ProductGenerator
from autoshorts.modules.product.discovery import ImageCandidate
from autoshorts.modules.product.errors import ProductNotFoundError
from autoshorts.modules.product.product import Product


def _product():
    return Product(
        input_url="https://shopee.com.br/X-i.1.2",
        resolved_url="https://shopee.com.br/X-i.1.2",
        slug="Projetor-HY300-Pro",
        name="Projetor HY300 Pro",
        shop_id="1",
        item_id="2",
    )


def _candidate(i):
    return ImageCandidate(
        f"https://img{i}.example.org/p.jpg",
        f"https://page{i}.example.org/",
        f"p{i}.jpg",
        "example.org",
    )


def _make_generator(tmp_path):
    gen = ProductGenerator(
        product_url="https://shopee.com.br/X-i.1.2", output=str(tmp_path / "out.mp4")
    )
    gen.tts_system = mock.MagicMock()

    async def fake_audio(paragraphs, temp_dir, text_transform=None):
        assert text_transform is not None
        text_transform("teste")  # must be callable without blowing up
        audio = tmp_path / "audio.mp3"
        audio.write_bytes(b"ID3")
        return str(audio)

    gen.tts_system.generate_audio_only = fake_audio
    gen._create_flux_video = mock.MagicMock()
    return gen


def test_registered_in_video_types():
    assert VIDEO_TYPES["product"] is ProductGenerator


def test_defaults_are_offline_and_images_only():
    gen = ProductGenerator(product_url="https://shopee.com.br/X-i.1.2")
    assert gen.web_search is False
    assert gen.images_only is True


def test_pipeline_skips_rejected_candidates(tmp_path, monkeypatch):
    gen = _make_generator(tmp_path)
    monkeypatch.setattr(
        "autoshorts.generators.product.parse_product_link", lambda url: _product()
    )
    monkeypatch.setattr(
        "autoshorts.generators.product.search_image_candidates",
        lambda name, limit: [_candidate(i) for i in range(5)],
    )

    def fake_verify(cands, name, prefetched_page=None):
        if "img0" in cands[0].image_url or "img1" in cands[0].image_url:
            raise ProductNotFoundError("model mismatch")
        return [{"identity_match": True, "image_bound_to_page": True}]

    monkeypatch.setattr(
        "autoshorts.generators.product.verify_image_sources", fake_verify
    )

    # Evidence-before-narration: the first candidate page verifies the model.
    _evidence_mocks(monkeypatch)

    monkeypatch.setattr(
        "autoshorts.generators.product.safe_get",
        lambda url, **kw: (b"\xff\xd8\xff" + b"0" * 200, None),
    )
    monkeypatch.setattr(
        "autoshorts.generators.product.ImageSearcher",
        lambda: mock.MagicMock(resize_to_fill=lambda src, dst: dst.write_bytes(b"jpg")),
    )
    # audio duration probe
    fake_clip = mock.MagicMock()
    fake_clip.duration = 25.0
    monkeypatch.setattr(
        "autoshorts.generators.product.AudioFileClip", lambda p: fake_clip
    )

    assert asyncio.run(gen.generate()) is True
    # 5 candidates - 2 rejected = 3 images passed to composition
    img_paths = gen._create_flux_video.call_args[0][0]
    assert len(img_paths) == 3


def test_pipeline_aborts_when_too_few_verified(tmp_path, monkeypatch):
    gen = _make_generator(tmp_path)
    monkeypatch.setattr(
        "autoshorts.generators.product.parse_product_link", lambda url: _product()
    )
    monkeypatch.setattr(
        "autoshorts.generators.product.search_image_candidates",
        lambda name, limit: [_candidate(i) for i in range(3)],
    )
    _evidence_mocks(monkeypatch)
    verify = mock.MagicMock(side_effect=ProductNotFoundError("nope"))
    monkeypatch.setattr("autoshorts.generators.product.verify_image_sources", verify)
    fake_clip = mock.MagicMock()
    fake_clip.duration = 25.0
    monkeypatch.setattr(
        "autoshorts.generators.product.AudioFileClip", lambda p: fake_clip
    )

    assert asyncio.run(gen.generate()) is False
    # the image-source gate was actually reached (not exited earlier)
    assert verify.call_count >= 1
    gen._create_flux_video.assert_not_called()


def _page(url, title, body=""):
    from autoshorts.modules.product.provenance import PageEvidence, model_codes

    return PageEvidence(
        url=url,
        status=200,
        title=title,
        h1="",
        meta_title="",
        identity_codes=model_codes(title),
        html_text=f"<html><main><ul>{body}</ul></main></html>",
    )


_SPEC_BODY = "<li>Wi-Fi 6</li><li>Bluetooth 5.0</li><li>8GB RAM</li>"


def _evidence_mocks(monkeypatch, title="Projetor HY300 Pro 4K", body=_SPEC_BODY):
    def fake_fetch(url, **kw):
        return _page(url, title, body)

    monkeypatch.setattr("autoshorts.generators.product.fetch_page_evidence", fake_fetch)
    monkeypatch.setattr(
        "autoshorts.generators.product.verify_product_identity", lambda *a, **k: {}
    )


def _run_ok(gen, monkeypatch):
    monkeypatch.setattr(
        "autoshorts.generators.product.search_image_candidates",
        lambda name, limit: [_candidate(i) for i in range(3)],
    )
    monkeypatch.setattr(
        "autoshorts.generators.product.verify_image_sources",
        lambda cands, name, prefetched_page=None: [
            {"identity_match": True, "image_bound_to_page": True}
        ],
    )
    monkeypatch.setattr(
        "autoshorts.generators.product.safe_get",
        lambda url, **kw: (b"\xff\xd8\xff" + b"0" * 200, None),
    )
    monkeypatch.setattr(
        "autoshorts.generators.product.ImageSearcher",
        lambda: mock.MagicMock(resize_to_fill=lambda src, dst: dst.write_bytes(b"jpg")),
    )
    fake_clip = mock.MagicMock()
    fake_clip.duration = 25.0
    monkeypatch.setattr(
        "autoshorts.generators.product.AudioFileClip", lambda p: fake_clip
    )
    return asyncio.run(gen.generate())


def test_narration_uses_attributed_page_atoms_not_title_prose(tmp_path, monkeypatch):
    """Only structured atoms from the verified page are narrated, attributed to
    its host; title marketing prose (origin, certification, battery) never is."""
    gen = _make_generator(tmp_path)
    monkeypatch.setattr(
        "autoshorts.generators.product.parse_product_link", lambda url: _product()
    )
    _evidence_mocks(
        monkeypatch,
        title="Projetor HY300 Pro 4K Oficial Fabricado no Japão aluminio aeronautico",
    )
    assert _run_ok(gen, monkeypatch) is True
    description = gen.metadata.description
    assert "Segundo o site" in description and "example.org" in description
    assert "8GB de memória" in description and "Wi-Fi" in description
    for prose in ("Japão", "aeronautico", "Oficial"):
        assert prose not in description


def test_negated_and_ambiguous_page_clauses_are_never_narrated(tmp_path, monkeypatch):
    gen = _make_generator(tmp_path)
    monkeypatch.setattr(
        "autoshorts.generators.product.parse_product_link", lambda url: _product()
    )
    body = (
        "<li>Sem Bluetooth</li><li>Não tem suporte para HDMI</li>"
        "<li>Suporta 2kg peso 1kg</li><li>Wi-Fi 6</li><li>8GB RAM</li><li>4K</li>"
    )
    _evidence_mocks(monkeypatch, body=body)
    assert _run_ok(gen, monkeypatch) is True
    description = gen.metadata.description
    assert "Bluetooth" not in description and "HDMI" not in description
    assert "2kg" not in description  # ambiguous load-vs-weight is skipped
    assert "1kg de peso" in description


def test_too_few_verified_facts_aborts_without_a_video(tmp_path, monkeypatch):
    gen = _make_generator(tmp_path)
    monkeypatch.setattr(
        "autoshorts.generators.product.parse_product_link", lambda url: _product()
    )
    _evidence_mocks(monkeypatch, body="<li>Wi-Fi 6</li>")
    assert _run_ok(gen, monkeypatch) is False
    gen._create_flux_video.assert_not_called()


def test_no_verifiable_page_aborts_before_narration(tmp_path, monkeypatch):
    """No authoritative page, no narration: fail closed, never slug claims."""
    gen = _make_generator(tmp_path)
    monkeypatch.setattr(
        "autoshorts.generators.product.parse_product_link", lambda url: _product()
    )
    monkeypatch.setattr(
        "autoshorts.generators.product.search_image_candidates",
        lambda name, limit: [_candidate(0)],
    )

    def reject(url, **kw):
        raise ProductNotFoundError("model mismatch")

    monkeypatch.setattr("autoshorts.generators.product.fetch_page_evidence", reject)
    tts_mock = mock.MagicMock()
    gen.tts_system.generate_audio_only = tts_mock
    assert asyncio.run(gen.generate()) is False
    tts_mock.assert_not_called()


def test_too_short_narration_aborts_without_a_video(tmp_path, monkeypatch):
    """The approved showcase is ~25-30s: a short evidenced script must not be
    shipped as a finished video."""
    gen = _make_generator(tmp_path)
    monkeypatch.setattr(
        "autoshorts.generators.product.parse_product_link", lambda url: _product()
    )
    _evidence_mocks(monkeypatch)
    monkeypatch.setattr(
        "autoshorts.generators.product.search_image_candidates",
        lambda name, limit: [_candidate(i) for i in range(5)],
    )
    short = mock.MagicMock()
    short.duration = 8.0
    monkeypatch.setattr("autoshorts.generators.product.AudioFileClip", lambda p: short)
    assert asyncio.run(gen.generate()) is False
    gen._create_flux_video.assert_not_called()
