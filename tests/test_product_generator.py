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
    gen.script_generator = mock.MagicMock()
    gen.script_generator.generate_script.return_value = [
        "Uma frase.",
        "Outra frase.",
        "Mais uma.",
    ]
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

    def fake_verify(cands, name):
        if "img0" in cands[0].image_url or "img1" in cands[0].image_url:
            raise ProductNotFoundError("model mismatch")
        return [{"identity_match": True, "image_bound_to_page": True}]

    monkeypatch.setattr(
        "autoshorts.generators.product.verify_image_sources", fake_verify
    )

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
    fake_clip.duration = 12.0
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
    monkeypatch.setattr(
        "autoshorts.generators.product.verify_image_sources",
        mock.MagicMock(side_effect=ProductNotFoundError("nope")),
    )
    fake_clip = mock.MagicMock()
    fake_clip.duration = 12.0
    monkeypatch.setattr(
        "autoshorts.generators.product.AudioFileClip", lambda p: fake_clip
    )

    assert asyncio.run(gen.generate()) is False
    gen._create_flux_video.assert_not_called()
