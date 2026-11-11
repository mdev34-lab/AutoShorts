"""Provider provenance: license verification and contextual-fit rejection."""

import json
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from PIL import Image

from autoshorts.modules import image_sources, image_subject


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


PROMPT = "a bowl of fruit on a table"  # broad request: passes the specificity gate


class TestCacheProvenance:
    def test_legacy_cache_without_sidecar_is_not_reused(self, tmp_path, monkeypatch):
        from autoshorts.modules import image_searcher as mod
        from autoshorts.modules.image_searcher import ImageSearcher

        monkeypatch.setattr(mod, "IMAGE_CACHE_DIR", tmp_path)
        key = ImageSearcher._cache_key(PROMPT)
        legacy = tmp_path / f"{key}.jpg"
        legacy.write_bytes(b"old-stock")
        sidecar = legacy.with_suffix(".credits.json")
        assert ImageSearcher._usable_cache(legacy, PROMPT) is False
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
            assert ImageSearcher._usable_cache(legacy, PROMPT) is False, bad
        sidecar.write_text(json.dumps(_VALID_RECORD))
        assert ImageSearcher._usable_cache(legacy, PROMPT) is True
        # unverified rights are reusable only under the explicit opt-in
        sidecar.write_text(json.dumps({**_VALID_RECORD, "license_verified": False}))
        monkeypatch.setattr(mod, "ALLOW_UNVERIFIED_IMAGES", True)
        assert ImageSearcher._usable_cache(legacy, PROMPT) is True

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

    def _commons_searcher(self, tmp_path, monkeypatch, subject_result):
        from autoshorts.modules import image_searcher as mod
        from autoshorts.modules.image_searcher import ImageSearcher

        monkeypatch.setattr(mod, "IMAGE_CACHE_DIR", tmp_path)
        monkeypatch.setattr(mod, "ALLOW_UNVERIFIED_IMAGES", False)
        monkeypatch.setattr(
            mod.image_content, "check_image_content", lambda path, prompt: None
        )

        def fake_subject(path, prompt):
            if isinstance(subject_result, Exception):
                raise subject_result
            return subject_result

        monkeypatch.setattr(mod.image_subject, "check_subject", fake_subject)
        commons = image_sources.SourceCandidate(
            url="https://upload.wikimedia.org/p.jpg",
            provider="commons",
            license="CC BY-SA 4.0",
            fit_evidence="a,b",
            title="t",
        )
        searcher = ImageSearcher.__new__(ImageSearcher)
        searcher._candidates = lambda q: [commons]

        def fake_download(url, path):
            Image.new("RGB", (400, 300), "gray").save(path, "JPEG")
            return True

        searcher.download_image = fake_download
        searcher._fallback_to_ai = lambda prompts, paths: paths.append("ai.jpg")
        searcher._write_provenance_sidecar = lambda *a, **k: None
        searcher.resize_to_fill = lambda src, dst: Path(dst).write_bytes(b"jpg")
        return mod, searcher

    def test_license_alone_is_not_subject_evidence(self, tmp_path, monkeypatch):
        """Verified license but the pictured subject cannot be verified: reject."""
        mod, searcher = self._commons_searcher(
            tmp_path,
            monkeypatch,
            image_subject.SubjectUnavailableError("no model"),
        )
        with patch.object(mod, "OUTPUT_DIR", tmp_path / "out"):
            assert searcher.get_images(["a bowl of fruit"]) == ["ai.jpg"]

    def test_wrong_pictured_subject_is_rejected(self, tmp_path, monkeypatch):
        mod, searcher = self._commons_searcher(
            tmp_path, monkeypatch, "image does not look like 'x' (p=0.01)"
        )
        with patch.object(mod, "OUTPUT_DIR", tmp_path / "out"):
            assert searcher.get_images(["a bowl of fruit"]) == ["ai.jpg"]

    def test_verified_license_and_subject_are_used(self, tmp_path, monkeypatch):
        mod, searcher = self._commons_searcher(tmp_path, monkeypatch, None)
        with patch.object(mod, "OUTPUT_DIR", tmp_path / "out"):
            result = searcher.get_images(["a bowl of fruit"])
        assert result and result[0].endswith(".jpg") and result != ["ai.jpg"]

    def test_specific_entity_request_never_uses_web_images(self, tmp_path, monkeypatch):
        """Wrong footballers/models of the right class pass the class check, so
        specific person/team/model/year/event requests must fall back."""
        mod, searcher = self._commons_searcher(tmp_path, monkeypatch, None)
        downloads = []
        searcher.download_image = lambda url, path: downloads.append(url) or True
        with patch.object(mod, "OUTPUT_DIR", tmp_path / "out"):
            result = searcher.get_images(["Clarence Seedorf Corinthians 2012"])
        assert downloads == [] and result == ["ai.jpg"]

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

    def test_cached_ai_rejected_while_fallback_off(self, tmp_path, monkeypatch):
        """A supplied existing kind=ai cache entry is never served through
        the web route while IMAGE_AI_FALLBACK=off; with the knob on the
        cached AI image stays reusable (fallback behavior preserved)."""
        from autoshorts.modules import image_searcher as mod
        from autoshorts.modules.image_searcher import ImageSearcher

        monkeypatch.setattr(mod, "IMAGE_CACHE_DIR", tmp_path)
        key = ImageSearcher._cache_key(PROMPT)
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
        monkeypatch.setattr(mod, "IMAGE_AI_FALLBACK", False)
        assert ImageSearcher._usable_cache(cached, PROMPT) is False
        monkeypatch.setattr(mod, "IMAGE_AI_FALLBACK", True)
        assert ImageSearcher._usable_cache(cached, PROMPT) is True


class TestAIFallbackKnob:
    """IMAGE_AI_FALLBACK=off fails the image stage closed instead of spending
    credits on metered AI generation (issue #37)."""

    def _searcher(self, tmp_path, monkeypatch, ai_fallback, download):
        from autoshorts.modules import image_searcher as mod
        from autoshorts.modules.image_searcher import ImageSearcher

        monkeypatch.setattr(mod, "IMAGE_CACHE_DIR", tmp_path)
        monkeypatch.setattr(mod, "ALLOW_UNVERIFIED_IMAGES", False)
        monkeypatch.setattr(mod, "IMAGE_AI_FALLBACK", ai_fallback)
        monkeypatch.setattr(
            mod.image_content, "check_image_content", lambda path, prompt: None
        )
        monkeypatch.setattr(mod.image_subject, "check_subject", lambda path, prompt: None)
        licensed = image_sources.SourceCandidate(
            url="https://upload.wikimedia.org/x.jpg",
            provider="commons",
            license="CC BY-SA 4.0",
            fit_evidence="a,b",
            title="t",
        )
        searcher = ImageSearcher.__new__(ImageSearcher)
        searcher._candidates = lambda q: [licensed]
        searcher.download_image = download
        searcher.resize_to_fill = lambda src, dst: Path(dst).write_bytes(b"jpg")
        searcher._write_provenance_sidecar = lambda *a, **k: None
        searcher._fallback_to_ai = Mock(
            side_effect=lambda prompts, paths: paths.append("ai.jpg")
        )
        return mod, searcher

    def test_web_failure_still_falls_back_to_ai_when_enabled(
        self, tmp_path, monkeypatch
    ):
        mod, searcher = self._searcher(
            tmp_path, monkeypatch, True, Mock(return_value=False)
        )
        with patch.object(mod, "OUTPUT_DIR", tmp_path / "out"):
            assert searcher.get_images([PROMPT]) == ["ai.jpg"]
        searcher._fallback_to_ai.assert_called_once()

    def test_disabled_knob_raises_instead_of_paying_for_ai(
        self, tmp_path, monkeypatch, capsys
    ):
        mod, searcher = self._searcher(
            tmp_path, monkeypatch, False, Mock(return_value=False)
        )
        with patch.object(mod.requests, "get") as mock_get:
            with pytest.raises(RuntimeError, match="IMAGE_AI_FALLBACK"):
                searcher.get_images([PROMPT])
        mock_get.assert_not_called()
        searcher._fallback_to_ai.assert_not_called()
        err = capsys.readouterr().out
        assert "[ERROR]" in err and "AI fallback is disabled" in err

    def test_disabled_knob_is_invisible_when_web_candidate_succeeds(
        self, tmp_path, monkeypatch
    ):
        def fake_download(url, path):
            Image.new("RGB", (400, 300), "gray").save(path, "JPEG")
            return True

        mod, searcher = self._searcher(tmp_path, monkeypatch, False, fake_download)
        with patch.object(mod, "OUTPUT_DIR", tmp_path / "out"):
            result = searcher.get_images([PROMPT])
        assert result and result[0].endswith(".jpg") and result != ["ai.jpg"]
        searcher._fallback_to_ai.assert_not_called()

    def test_disabled_knob_aborts_without_any_http(self, tmp_path, monkeypatch):
        """Actual route (no mocked _fallback_to_ai): no candidates and
        IMAGE_AI_FALLBACK=off abort the image stage before a single real
        HTTP call - cached AI bytes are rejected too, so nothing is served."""
        from autoshorts.modules import image_searcher as mod
        from autoshorts.modules.image_searcher import ImageSearcher

        monkeypatch.setattr(mod, "IMAGE_CACHE_DIR", tmp_path)
        monkeypatch.setattr(mod, "ALLOW_UNVERIFIED_IMAGES", False)
        monkeypatch.setattr(mod, "IMAGE_AI_FALLBACK", False)
        searcher = ImageSearcher.__new__(ImageSearcher)
        searcher._candidates = lambda q: []
        with patch.object(mod.requests, "get") as spy_get:
            with pytest.raises(RuntimeError, match="IMAGE_AI_FALLBACK"):
                searcher.get_images([PROMPT])
        spy_get.assert_not_called()


LONG_PROMPT = "Flamengo vs Grêmio Maracanã 2026 Brasileirão 34ª rodada"
SHORT_QUERY = "Flamengo Grêmio Maracanã"
# A realistic photo title: on-subject, but it never repeats the whole prompt.
REALISTIC_TITLE = "Flamengo vence o Grêmio no Maracanã e amplia liderança"


class TestShortSubjectQuery:
    """The retry query keeps the tokens that identify the subject."""

    def test_keeps_first_distinctive_tokens(self):
        assert image_sources.short_subject_query(LONG_PROMPT) == SHORT_QUERY

    def test_drops_generic_words_and_connectors(self):
        assert (
            image_sources.short_subject_query("Palmeiras best photo official")
            == "Palmeiras"
        )
        assert image_sources.short_subject_query("Final entre os dois times") == (
            "Final dois times"
        )

    def test_all_generic_prompt_has_no_subject_query(self):
        assert image_sources.short_subject_query("best photo image") == ""

    def test_dedupes_on_normalized_form_keeps_original(self):
        assert (
            image_sources.short_subject_query("FLAMENGO flamengo Grêmio")
            == "FLAMENGO Grêmio"
        )


class TestShortSubjectRetry:
    """Issue #37: long prompts never match photo titles verbatim, so the DDG
    route is re-queried with the short subject query - under the
    ALLOW_UNVERIFIED_IMAGES rights opt-in only, and still fit-filtered."""

    def _ddg_searcher(self, monkeypatch, opt_in, fake_search):
        from autoshorts.modules import image_searcher as mod
        from autoshorts.modules.image_searcher import ImageSearcher

        monkeypatch.setattr(mod, "ALLOW_UNVERIFIED_IMAGES", opt_in)
        searcher = ImageSearcher()
        return (
            searcher,
            patch(
                "autoshorts.modules.image_searcher.IMAGE_PROVIDERS", ["ddg"]
            ),
            patch.object(searcher, "search_images", side_effect=fake_search),
        )

    def test_opt_in_retries_with_short_query_and_records_fit_evidence(
        self, monkeypatch
    ):
        calls = []

        def fake_search(query):
            calls.append(query)
            return [
                {"image": "https://ge.globo.com/f.jpg", "title": REALISTIC_TITLE}
            ]

        searcher, providers, search = self._ddg_searcher(
            monkeypatch, True, fake_search
        )
        with providers, search:
            out = searcher._candidates(LONG_PROMPT)
        assert calls == [LONG_PROMPT, SHORT_QUERY]  # the retry happened
        assert [c.url for c in out] == ["https://ge.globo.com/f.jpg"]
        assert out[0].provider == "ddg"
        # fit evidence names the short query the photo was actually matched to
        assert out[0].fit_evidence == (
            f"short-query:'{SHORT_QUERY}' via flamengo,gremio,maracana"
        )

    def test_opt_in_retry_still_drops_agency_hosts(self, monkeypatch):
        def fake_search(query):
            if query == LONG_PROMPT:
                return []
            return [
                {
                    "image": "https://c8.alamy.com/comp/flamengo.jpg",
                    "title": REALISTIC_TITLE,
                },
                {
                    "image": "https://www.shutterstock.com/x.jpg",
                    "title": "Flamengo Grêmio Maracanã",
                },
                {
                    "image": "https://ge.globo.com/g.jpg",
                    "title": "Flamengo x Grêmio no Maracanã",
                },
            ]

        searcher, providers, search = self._ddg_searcher(
            monkeypatch, True, fake_search
        )
        with providers, search:
            out = searcher._candidates(LONG_PROMPT)
        assert [c.url for c in out] == ["https://ge.globo.com/g.jpg"]

    def test_no_retry_without_opt_in(self, monkeypatch):
        calls = []

        def fake_search(query):
            calls.append(query)
            return [
                {"image": "https://ge.globo.com/f.jpg", "title": REALISTIC_TITLE}
            ]

        searcher, providers, search = self._ddg_searcher(
            monkeypatch, False, fake_search
        )
        with providers, search:
            assert searcher._candidates(LONG_PROMPT) == []
        assert calls == [LONG_PROMPT]  # one query, results wiped as before

    def test_no_retry_when_full_query_already_fits(self, monkeypatch):
        prompt = "Flamengo Grêmio Maracanã 2026"
        calls = []

        def fake_search(query):
            calls.append(query)
            return [
                {
                    "image": "https://ge.globo.com/f.jpg",
                    "title": "Flamengo vence o Grêmio no Maracanã 2026",
                }
            ]

        searcher, providers, search = self._ddg_searcher(
            monkeypatch, True, fake_search
        )
        with providers, search:
            out = searcher._candidates(prompt)
        assert calls == [prompt]  # no wasted request
        assert out[0].fit_evidence == "2026,flamengo,gremio,maracana"


class TestShortSubjectRetryGuards:
    """Issue #37 follow-up: the retry must not admit a photo whose title
    states a number the FULL prompt does not contain (wrong year, score or
    edition - the short query itself carries no numbers), the short query is
    the subject and never the prompt's aesthetic prefix, and a numberless
    on-subject photo is still accepted."""

    def _retry(self, monkeypatch, prompt, retry_results):
        from autoshorts.modules import image_searcher as mod
        from autoshorts.modules.image_searcher import ImageSearcher

        monkeypatch.setattr(mod, "ALLOW_UNVERIFIED_IMAGES", True)
        searcher = ImageSearcher()
        calls = []

        def fake_search(query):
            calls.append(query)
            return [] if query == prompt else retry_results

        with (
            patch("autoshorts.modules.image_searcher.IMAGE_PROVIDERS", ["ddg"]),
            patch.object(searcher, "search_images", side_effect=fake_search),
        ):
            out = searcher._candidates(prompt)
        return calls, out

    def test_retry_rejects_wrong_year_in_title(self, monkeypatch):
        calls, out = self._retry(
            monkeypatch,
            LONG_PROMPT,
            [
                {
                    "image": "https://ge.globo.com/old.jpg",
                    "title": "Flamengo 2x0 Grêmio 2019 no Maracanã",
                }
            ],
        )
        assert calls == [LONG_PROMPT, SHORT_QUERY]  # judged in the retry path
        assert out == []  # 2019 (and 2x0) is not in the 2026 prompt

    def test_retry_rejects_wrong_score_in_title(self, monkeypatch):
        prompt = "Palmeiras 6x1 Remo Nubank Parque 2026 Brasileirão"
        calls, out = self._retry(
            monkeypatch,
            prompt,
            [
                {
                    "image": "https://ge.globo.com/score.jpg",
                    "title": "Palmeiras 3x0 Remo Nubank Parque",
                }
            ],
        )
        assert calls == [prompt, "Palmeiras Remo Nubank"]
        assert out == []  # 3x0 contradicts the prompt's 6x1

    def test_retry_rejects_extra_edition_number_in_title(self, monkeypatch):
        prompt = "Flamengo Palmeiras Nubank Parque 21 novembro 2026 decisão"
        calls, out = self._retry(
            monkeypatch,
            prompt,
            [
                {
                    "image": "https://ge.globo.com/sub17.jpg",
                    "title": (
                        "Flamengo vence Palmeiras em estreia do Nubank Parque "
                        "pelo Brasileiro Sub-17"
                    ),
                }
            ],
        )
        assert calls == [prompt, "Flamengo Palmeiras Nubank"]
        assert out == []  # 17 is not among the prompt's {21, 2026}

    def test_retry_subject_is_never_the_aesthetic_prefix(self, monkeypatch):
        calls, out = self._retry(
            monkeypatch,
            "wide shot Flamengo celebration Maracanã 2026 Brasileirão final",
            [
                {
                    "image": "https://ge.globo.com/f.jpg",
                    "title": "Flamengo celebration at Maracanã",
                }
            ],
        )
        assert calls[1].startswith("Flamengo")  # not "wide shot Flamengo"
        assert [c.url for c in out] == ["https://ge.globo.com/f.jpg"]

        calls, out = self._retry(
            monkeypatch,
            "cinematic dramatic Palmeiras Remo Nubank Parque",
            [
                {
                    "image": "https://ge.globo.com/p.jpg",
                    "title": "Palmeiras x Remo no Nubank Parque",
                }
            ],
        )
        assert calls[1].startswith("Palmeiras")  # not "cinematic dramatic..."
        assert [c.url for c in out] == ["https://ge.globo.com/p.jpg"]

    def test_retry_accepts_numberless_subject_photo(self, monkeypatch):
        calls, out = self._retry(
            monkeypatch,
            LONG_PROMPT,
            [
                {
                    "image": "https://ge.globo.com/f.jpg",
                    "title": "No Maracanã, Grêmio sofre revés do Flamengo",
                }
            ],
        )
        assert calls == [LONG_PROMPT, SHORT_QUERY]
        assert [c.url for c in out] == ["https://ge.globo.com/f.jpg"]
        assert out[0].fit_evidence.startswith(f"short-query:'{SHORT_QUERY}'")

    def test_retry_accepts_title_number_stated_by_full_prompt(self, monkeypatch):
        calls, out = self._retry(
            monkeypatch,
            LONG_PROMPT,
            [
                {
                    "image": "https://ge.globo.com/f.jpg",
                    "title": "Flamengo e Grêmio se enfrentam no Maracanã em 2026",
                }
            ],
        )
        assert calls == [LONG_PROMPT, SHORT_QUERY]
        # the title's only number, 2026, is stated by the full prompt
        assert [c.url for c in out] == ["https://ge.globo.com/f.jpg"]

    def test_retry_keeps_short_query_opponent_requirement(self, monkeypatch):
        prompt = "Palmeiras 6x1 Remo Nubank Parque 2026 Brasileirão"
        calls, out = self._retry(
            monkeypatch,
            prompt,
            [
                {
                    "image": "https://ge.globo.com/p.jpg",
                    "title": "Palmeiras vence no Nubank Parque",
                }
            ],
        )
        assert calls == [prompt, "Palmeiras Remo Nubank"]
        assert out == []  # the short query still demands "Remo" in the title

    def test_retry_keeps_explicit_pair_opponent(self, monkeypatch):
        """First-N truncation must not drop the opponent after an explicit
        'vs': the Vasco photo has the right player and venue but the wrong
        opponent, so it is not a valid retry candidate."""
        prompt = "Flamengo Maracanã celebration vs Grêmio"
        calls, out = self._retry(
            monkeypatch,
            prompt,
            [
                {
                    "image": "https://ge.globo.com/vasco.jpg",
                    "title": (
                        "Flamengo celebration at Maracanã after beating Vasco"
                    ),
                }
            ],
        )
        assert calls == [prompt, "Flamengo Maracanã Grêmio"]
        assert out == []

    def test_retry_accepts_same_explicit_pair(self, monkeypatch):
        prompt = "Flamengo Maracanã celebration vs Grêmio"
        calls, out = self._retry(
            monkeypatch,
            prompt,
            [
                {
                    "image": "https://ge.globo.com/g.jpg",
                    "title": (
                        "Flamengo celebration at Maracanã after beating Grêmio"
                    ),
                }
            ],
        )
        assert calls == [prompt, "Flamengo Maracanã Grêmio"]
        assert [c.url for c in out] == ["https://ge.globo.com/g.jpg"]


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


def test_distinctive_fit_admits_real_photo_with_filler_query_word():
    # "home" is filler: the title proves the distinctive subject tokens.
    assert (
        image_sources.fit_evidence(
            "Allianz Parque Palmeiras home stadium",
            "Palmeiras Stadium - Allianz Parque (Sao Paulo)",
        )
        is not None
    )


def test_distinctive_fit_still_rejects_a_different_subject():
    assert (
        image_sources.fit_evidence(
            "Allianz Parque Palmeiras home stadium",
            "Maracana Stadium - Flamengo (Rio de Janeiro)",
        )
        is None
    )


def test_distinctive_fit_keeps_number_contradiction_rejection():
    assert (
        image_sources.fit_evidence(
            "Allianz Parque Palmeiras 2014",
            "Allianz Parque Palmeiras 2014 2016",
        )
        is None
    )


def test_distinctive_fit_requires_query_numbers_in_title():
    assert (
        image_sources.fit_evidence(
            "Palmeiras 6x1 Remo 2026",
            "Palmeiras Remo match at Allianz Parque",
        )
        is None
    )


def test_distinctive_fit_accepts_matching_query_numbers():
    assert (
        image_sources.fit_evidence(
            "Palmeiras 6x1 Remo 2026",
            "Palmeiras 6 x 1 Remo 2026 at Allianz Parque",
        )
        is not None
    )


def test_distinctive_fit_keeps_data_graphic_rejection():
    assert (
        image_sources.fit_evidence(
            "Allianz Parque Palmeiras standings",
            "Allianz Parque Palmeiras standings table",
        )
        is None
    )


def test_commons_title_only_keyword_match_is_not_subject_evidence():
    """A keyword-stuffed title with an unrelated description/categories is not
    positive subject evidence (title and description must agree)."""
    pages = [
        _commons_page(
            "File:Apple iPhone launch.jpg",
            "CC BY-SA 4.0",
            description="A bowl of fruit on a table",
        )
    ]
    with patch.object(image_sources.requests, "get", return_value=_mock_commons(pages)):
        assert image_sources.search_commons("Apple iPhone launch") == []
