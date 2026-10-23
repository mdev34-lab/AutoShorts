"""Provider provenance: license verification and contextual-fit rejection."""

import json
from unittest.mock import Mock, patch

import pytest

from autoshorts.modules import image_sources


def _commons_page(
    title,
    license_name,
    url="https://upload.wikimedia.org/x.jpg",
    width=1600,
    height=1200,
    mime="image/jpeg",
    description="",
):
    return {
        "title": title,
        "imageinfo": [
            {
                "url": url,
                "mime": mime,
                "width": width,
                "height": height,
                "extmetadata": {
                    "LicenseShortName": {"value": license_name},
                    "Artist": {"value": "Some Photographer"},
                    "ImageDescription": {"value": description},
                },
            }
        ],
    }


def _mock_commons(pages):
    resp = Mock()
    resp.json.return_value = {
        "query": {"pages": {str(i): p for i, p in enumerate(pages)}}
    }
    return resp


def test_commons_rejects_unverified_license():
    pages = [
        _commons_page(
            "File:Seedorf Corinthians 2012.jpg",
            "CC BY-SA 4.0",
            description="Seedorf no Corinthians",
        ),
        _commons_page("File:Seedorf Getty.jpg", "Getty Images licensing"),
        _commons_page("File:Seedorf gfdl.jpg", "GFDL"),
    ]
    with patch.object(image_sources.requests, "get", return_value=_mock_commons(pages)):
        out = image_sources.search_commons("Seedorf Corinthians")
    assert [c.url for c in out] == ["https://upload.wikimedia.org/x.jpg"]
    assert out[0].license == "CC BY-SA 4.0"
    assert "Some Photographer" in out[0].attribution
    assert out[0].fit_evidence == "corinthians,seedorf"


def test_commons_rejects_ambiguous_fit():
    pages = [
        _commons_page(
            "File:Tabela classificação.jpg",
            "CC0",
            description="Tabela de classificação do campeonato",
        ),
    ]
    with patch.object(image_sources.requests, "get", return_value=_mock_commons(pages)):
        out = image_sources.search_commons("Seedorf estreia no Corinthians")
    assert out == []


def test_commons_rejects_small_and_nonbitmap():
    pages = [
        _commons_page(
            "File:Seedorf tiny.jpg", "CC0", width=300, height=200, description="Seedorf"
        ),
        _commons_page(
            "File:Seedorf.svg", "CC0", mime="image/svg+xml", description="Seedorf"
        ),
    ]
    with patch.object(image_sources.requests, "get", return_value=_mock_commons(pages)):
        out = image_sources.search_commons("Seedorf Corinthians", min_width=720)
    assert out == []


def test_ddg_blocks_watermarked_agencies():
    results = [
        {"image": "https://c8.alamy.com/comp/seedorf.jpg", "title": "Seedorf photo"},
        {"image": "https://www.shutterstock.com/x.jpg", "title": "Seedorf"},
        {
            "image": "https://clubesite.com/seedorf.jpg",
            "title": "Seedorf no Corinthians",
        },
    ]
    out = image_sources.filter_ddg_results("Seedorf Corinthians", results)
    assert [c.url for c in out] == ["https://clubesite.com/seedorf.jpg"]
    assert out[0].license is None  # DDG has no license metadata


def test_ddg_rejects_ambiguous_fit():
    results = [{"image": "https://x.com/img.jpg", "title": "clique aqui"}]
    assert image_sources.filter_ddg_results("Seedorf Corinthians", results) == []


def test_searcher_candidates_provider_order():
    from autoshorts.modules.image_searcher import ImageSearcher

    searcher = ImageSearcher()
    commons_hit = image_sources.SourceCandidate(
        url="https://upload.wikimedia.org/a.jpg",
        provider="commons",
        license="CC0",
        fit_evidence="seedorf",
    )
    with (
        patch(
            "autoshorts.modules.image_searcher.IMAGE_PROVIDERS",
            ["commons", "ddg"],
        ),
        patch.object(searcher, "_commons_candidates", return_value=[commons_hit]),
        patch.object(
            searcher,
            "search_images",
            return_value=[
                {"image": "https://y.com/s.jpg", "title": "seedorf corinthians foto"}
            ],
        ),
    ):
        out = searcher._candidates("Seedorf Corinthians")
    assert out[0].provider == "commons"  # verified license comes first
    assert out[1].provider == "ddg"


def test_default_providers_fail_closed_commons_only():
    """Default config must be license-verified only; DDG is explicit opt-in."""
    import importlib
    import os

    os.environ.pop("IMAGE_PROVIDERS", None)
    import autoshorts.modules.config as cfg

    try:
        importlib.reload(cfg)
        assert cfg.IMAGE_PROVIDERS == ["commons"]
    finally:
        importlib.reload(cfg)


def test_ddg_is_opt_in_only():
    """With the default provider list, the DDG path is never consulted."""
    from autoshorts.modules.image_searcher import ImageSearcher

    searcher = ImageSearcher()
    with (
        patch("autoshorts.modules.image_searcher.IMAGE_PROVIDERS", ["commons"]),
        patch.object(searcher, "_commons_candidates", return_value=[]),
        patch.object(searcher, "search_images") as mock_ddg,
    ):
        assert searcher._candidates("Seedorf Corinthians") == []
        mock_ddg.assert_not_called()


def test_provenance_sidecar_and_credits_delivery(tmp_path):
    """The reusable-license obligations are delivered: title, creator, source
    page, license + URL and the modification notice. Entries without a valid
    record are marked UNVERIFIED and flip the publication status."""
    from autoshorts.modules import image_searcher as mod
    from autoshorts.modules.image_searcher import ImageSearcher

    img = tmp_path / "a.jpg"
    img.write_bytes(b"x")
    cand = image_sources.SourceCandidate(
        url="https://upload.wikimedia.org/a.jpg",
        provider="commons",
        license="CC BY-SA 4.0",
        attribution="Jane Doe, CC BY-SA 4.0, via Wikimedia Commons",
        fit_evidence="seedorf",
        title="File:Seedorf.jpg",
        extra={
            "source_page": "https://commons.wikimedia.org/wiki/File:Seedorf.jpg",
            "creator": "Jane Doe",
            "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
        },
    )
    ImageSearcher._write_provenance_sidecar(img, cand, "Seedorf Corinthians")

    orphan = tmp_path / "b.jpg"
    orphan.write_bytes(b"x")
    with patch.object(mod, "OUTPUT_DIR", tmp_path / "out"):
        ImageSearcher._write_credits_file([str(img)])
        clean = (tmp_path / "out" / "image_credits.txt").read_text()
        ImageSearcher._write_credits_file([str(img), str(orphan)])
        flagged = (tmp_path / "out" / "image_credits.txt").read_text()
    for needle in (
        "Title: File:Seedorf.jpg",
        "Creator: Jane Doe",
        "Source: https://commons.wikimedia.org/wiki/File:Seedorf.jpg",
        "License: CC BY-SA 4.0",
        "License URL: https://creativecommons.org/licenses/by-sa/4.0/",
        "Changes:",
    ):
        assert needle in clean, needle
    assert "PUBLICATION REVIEW REQUIRED" not in clean
    assert "UNVERIFIED" in flagged  # orphan never reported as licensed/safe
    assert flagged.startswith("PUBLICATION REVIEW REQUIRED: 1")


class TestEntityDisambiguation:
    """A single shared token never proves fit - the contradiction cases."""

    @pytest.mark.parametrize(
        "query,bad_title",
        [
            ("Apple iPhone lançamento", "Torta de maçã apple pie receita"),
            ("Jaguar car new model", "Jaguar animal no Pantanal"),
            ("Table Mountain hiking", "Dining table madeira rústica"),
            ("Giants Mets baseball", "New York Mets victory photo"),
            ("Corinthians football crise", "Corinthians basketball quarter"),
            ("Seedorf Corinthians estreia", "Clarence Seedorf Botafogo 2013"),
        ],
    )
    def test_single_token_contradictions_rejected(self, query, bad_title):
        assert image_sources.fit_evidence(query, bad_title) is None

    @pytest.mark.parametrize(
        "query,good_title,good_context",
        [
            ("Apple iPhone lançamento", "Apple iPhone lançamento event", ""),
            ("Giants Mets baseball", "Giants Mets baseball game at the stadium", ""),
            (
                "Seedorf estreia no Corinthians em crise",
                "Seedorf Corinthians",
                "estreia crise presentation at the club",
            ),
        ],
    )
    def test_multi_token_coverage_accepted(self, query, good_title, good_context):
        assert image_sources.fit_evidence(query, good_title, good_context) is not None

    def test_one_token_query_needs_title_anchor(self):
        # token only in description context, not the title: ambiguous
        assert image_sources.fit_evidence("Seedorf", "Match photo", "Seedorf") is None
        assert image_sources.fit_evidence("Seedorf", "Seedorf portrait") == "seedorf"

    def test_data_graphics_fail_closed_even_with_full_coverage(self):
        # a standings table can share every token yet contradict the subject
        assert (
            image_sources.fit_evidence(
                "Corinthians tabela", "Corinthians tabela classificação 2013"
            )
            is None
        )
        assert (
            image_sources.fit_evidence(
                "Brasileirão standings", "Brasileirão standings chart"
            )
            is None
        )

    def test_ddg_url_tokens_are_not_fit_evidence(self):
        # the page path carries query tokens under an unrelated image
        results = [
            {
                "image": "https://blog.com/seedorf/corinthians/img7.jpg",
                "url": "https://blog.com/seedorf-corinthians-post",
                "title": "cute kitten of the day",
            }
        ]
        assert image_sources.filter_ddg_results("Seedorf Corinthians", results) == []

    def test_ftcdn_adobe_cdn_blocked(self):
        results = [
            {"image": "https://as1.ftcdn.net/x.jpg", "title": "Seedorf Corinthians"}
        ]
        assert image_sources.filter_ddg_results("Seedorf Corinthians", results) == []


_VALID_RECORD = {
    "schema": 1,
    "kind": "web",
    "provider": "commons",
    "title": "File:X.jpg",
    "image_url": "https://upload.wikimedia.org/x.jpg",
    "source_page": "https://commons.wikimedia.org/wiki/File:X.jpg",
    "license": "CC BY 4.0",
    "license_verified": True,
}


class TestCacheProvenance:
    def test_legacy_cache_without_sidecar_is_not_reused(self, tmp_path, monkeypatch):
        from autoshorts.modules import image_searcher as mod
        from autoshorts.modules.image_searcher import ImageSearcher

        monkeypatch.setattr(mod, "IMAGE_CACHE_DIR", tmp_path)
        key = ImageSearcher._cache_key("qualquer prompt")
        legacy = tmp_path / f"{key}.jpg"
        legacy.write_bytes(b"old-stock")
        sidecar = legacy.with_suffix(".credits.json")
        assert ImageSearcher._usable_cache(legacy) is False
        # empty / unparsable / incomplete / unverified records are not provenance
        for bad in (
            "",
            "provider: commons\n",
            "{}",
            json.dumps({"schema": 1, "kind": "web", "provider": "commons"}),
            json.dumps({**_VALID_RECORD, "license": ""}),
            json.dumps({**_VALID_RECORD, "source_page": "not a url"}),
            json.dumps({**_VALID_RECORD, "license_verified": False}),
        ):
            sidecar.write_text(bad)
            assert ImageSearcher._usable_cache(legacy) is False, bad
        sidecar.write_text(json.dumps(_VALID_RECORD))
        assert ImageSearcher._usable_cache(legacy) is True
        # unverified rights are reusable only under the explicit opt-in
        sidecar.write_text(json.dumps({**_VALID_RECORD, "license_verified": False}))
        monkeypatch.setattr(mod, "ALLOW_UNVERIFIED_IMAGES", True)
        assert ImageSearcher._usable_cache(legacy) is True

    def test_unverified_candidates_skipped_without_opt_in(self, tmp_path, monkeypatch):
        from autoshorts.modules import image_searcher as mod
        from autoshorts.modules.image_searcher import ImageSearcher

        monkeypatch.setattr(mod, "IMAGE_CACHE_DIR", tmp_path)
        monkeypatch.setattr(mod, "ALLOW_UNVERIFIED_IMAGES", False)
        ddg = image_sources.SourceCandidate(
            url="https://x.example.org/p.jpg",
            provider="ddg",
            fit_evidence="a,b",
            title="t",
        )
        searcher = ImageSearcher.__new__(ImageSearcher)
        searcher._candidates = lambda q: [ddg]
        downloads = []
        searcher.download_image = lambda url, path: downloads.append(url) or True
        searcher._fallback_to_ai = lambda prompts, paths: paths.append("ai.jpg")
        with patch.object(mod, "OUTPUT_DIR", tmp_path / "out"):
            result = searcher.get_images(["some prompt"])
        assert downloads == []  # never even downloaded
        assert result == ["ai.jpg"]

    def test_cache_key_versioned_by_providers(self, monkeypatch):
        from autoshorts.modules import image_searcher as mod
        from autoshorts.modules.image_searcher import ImageSearcher

        monkeypatch.setattr(mod, "IMAGE_PROVIDERS", ["commons"])
        k1 = ImageSearcher._cache_key("prompt x")
        monkeypatch.setattr(mod, "IMAGE_PROVIDERS", ["commons", "ddg"])
        k2 = ImageSearcher._cache_key("prompt x")
        assert k1 != k2
        # and neither collides with the legacy raw-prompt hash
        import hashlib

        assert k1 != hashlib.sha256(b"prompt x").hexdigest()[:16]


@pytest.mark.parametrize(
    "query,title",
    [
        ("Jaguar electric car launch", "Jaguar and electric eel wildlife"),
        (
            "Seedorf Corinthians 2012 third 42 points",
            "Seedorf Corinthians table 2013 sixth 17 points",
        ),
        (
            "Seedorf Corinthians 2012 third 42 points",
            "Corinthians football 2013 sixth 17 points",
        ),
        ("Seedorf Corinthians 2012", "Seedorf Corinthians 2013 portrait"),
    ],
)
def test_reviewer_probes_are_rejected(query, title):
    assert image_sources.fit_evidence(query, title) is None


def test_full_entity_coverage_still_admits_a_real_match():
    assert image_sources.fit_evidence(
        "Seedorf Corinthians 2012", "Seedorf in Corinthians shirt, 2012"
    )
