"""Provider provenance: license verification and contextual-fit rejection."""

from unittest.mock import Mock, patch

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
        _commons_page("File:Seedorf 2012.jpg", "CC BY-SA 4.0", description="Seedorf"),
        _commons_page("File:Seedorf Getty.jpg", "Getty Images licensing"),
        _commons_page("File:Seedorf gfdl.jpg", "GFDL"),
    ]
    with patch.object(image_sources.requests, "get", return_value=_mock_commons(pages)):
        out = image_sources.search_commons("Seedorf Corinthians")
    assert [c.url for c in out] == ["https://upload.wikimedia.org/x.jpg"]
    assert out[0].license == "CC BY-SA 4.0"
    assert "Some Photographer" in out[0].attribution
    assert out[0].fit_evidence == "seedorf"


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
        {"image": "https://clubesite.com/seedorf.jpg", "title": "Seedorf no clube"},
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
        patch.object(searcher, "_commons_candidates", return_value=[commons_hit]),
        patch.object(
            searcher,
            "search_images",
            return_value=[{"image": "https://y.com/s.jpg", "title": "seedorf foto"}],
        ),
    ):
        out = searcher._candidates("Seedorf Corinthians")
    assert out[0].provider == "commons"  # verified license comes first
    assert out[1].provider == "ddg"
