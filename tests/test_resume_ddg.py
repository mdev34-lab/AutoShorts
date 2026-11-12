"""Source-only tests: every GET is an injected fake, never a real provider."""

from __future__ import annotations

import json

import pytest

from autoshorts.modules.resume_ddg import DDGSearchError, FixedDDGSearch

HTML = b'<html><script>vqd="4-123-456";</script></html>'


class FakeDDG:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(
        self, url, params=None, *, max_bytes=1_048_576, timeout=10.0, max_redirects=3
    ):
        self.calls.append(
            (
                url,
                params.copy() if isinstance(params, dict) else params,
                max_bytes,
                timeout,
                max_redirects,
            )
        )
        if not self.responses:
            raise AssertionError("Unexpected extra GET")
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


def _item(index=0):
    return {
        "image": f"https://images.example/{index}.jpg",
        "thumbnail": f"https://images.example/thumb-{index}.jpg",
        "title": f"  Flamengo Grêmio {index}  ",
        "url": f"https://pages.example/{index}",
        "height": 720,
        "width": 1280,
        "source": "example",
        "extra": {"preserve": [True, index]},
    }


def _body(results, **extra):
    return json.dumps({"results": results, **extra}, ensure_ascii=False).encode("utf-8")


def test_fixed_gets_caps_payload_default_limit_and_original_dict_keys():
    items = [_item(index) for index in range(12)]
    fake = FakeDDG(HTML, _body(items))
    search = FixedDDGSearch(fake)
    assert fake.calls == []
    query = "  Flamengo x Grêmio  "
    assert search(query) == items[:10]
    assert fake.calls == [
        ("https://duckduckgo.com/", {"q": query}, 262_144, 10.0, 0),
        (
            "https://duckduckgo.com/i.js",
            {
                "o": "json",
                "q": query,
                "l": "br-pt",
                "vqd": "4-123-456",
                "p": "1",
                "ct": "AT",
            },
            1_048_576,
            10.0,
            0,
        ),
    ]


class IntSubclass(int):
    pass


@pytest.mark.parametrize(
    "value", [False, True, 0, -1, 26, 1.0, "10", None, [], {}, IntSubclass(1)]
)
def test_max_results_is_exact_integer_one_through_25(value):
    fake = FakeDDG()
    with pytest.raises(DDGSearchError):
        FixedDDGSearch(fake, max_results=value)
    assert fake.calls == []


@pytest.mark.parametrize("value", [None, False, 1, "get", [], {}])
def test_get_ddg_must_be_callable(value):
    with pytest.raises(TypeError):
        FixedDDGSearch(value)


@pytest.mark.parametrize("limit", [1, 25])
@pytest.mark.parametrize(
    "next_value",
    ["https://evil.invalid/next", "/i.js?q=other&s=100", {"url": "elsewhere"}],
)
def test_limit_never_causes_pagination_or_remote_endpoint_selection(limit, next_value):
    items = [_item(index) for index in range(30)]
    fake = FakeDDG(
        HTML, _body(items, next=next_value, endpoint="https://evil.invalid/")
    )
    assert FixedDDGSearch(fake, limit)("query") == items[:limit]
    assert [call[0] for call in fake.calls] == [
        "https://duckduckgo.com/",
        "https://duckduckgo.com/i.js",
    ]


@pytest.mark.parametrize(
    "query", [None, False, True, 1, 1.0, b"query", [], {}, "", " \t\n", "x" * 2001]
)
def test_query_validation_happens_before_any_get(query):
    fake = FakeDDG()
    with pytest.raises(DDGSearchError):
        FixedDDGSearch(fake)(query)
    assert fake.calls == []


@pytest.mark.parametrize("query", ["x" * 2000, "ã" * 2000])
def test_query_character_limit_is_inclusive_and_not_coerced(query):
    fake = FakeDDG(HTML, _body([_item()]))
    assert FixedDDGSearch(fake)(query) == [_item()]
    assert all(call[1]["q"] == query for call in fake.calls)


@pytest.mark.parametrize(
    "html",
    [
        b'<script>vqd="4-123-456";</script>',
        b"/?q=query&vqd=4-123-456&other=ignored",
        b"<script>vqd='4-123-456';</script>",
        b"vqd='4-123-456'&other=ignored",
    ],
)
def test_installed_token_forms_are_parsed_purely(html):
    fake = FakeDDG(html, _body([_item()]))
    assert FixedDDGSearch(fake)("query") == [_item()]
    assert fake.calls[1][1]["vqd"] == "4-123-456"


@pytest.mark.parametrize(
    "opening,closing", [(b'vqd="', b'"'), (b"vqd=", b"&"), (b"vqd='", b"'")]
)
def test_token_length_limit_is_inclusive(opening, closing):
    fake = FakeDDG(opening + b"a" * 256 + closing, _body([]))
    assert FixedDDGSearch(fake)("query") == []
    assert fake.calls[1][1]["vqd"] == "a" * 256


@pytest.mark.parametrize(
    "opening,closing", [(b'vqd="', b'"'), (b"vqd=", b"&"), (b"vqd='", b"'")]
)
@pytest.mark.parametrize("length", [0, 257])
def test_empty_or_overlong_tokens_fail_before_second_get(opening, closing, length):
    fake = FakeDDG(opening + b"a" * length + closing)
    with pytest.raises(DDGSearchError):
        FixedDDGSearch(fake)("query")
    assert len(fake.calls) == 1


@pytest.mark.parametrize(
    "html",
    [
        b"<html>No token</html>",
        b"vqd=",
        b'vqd="missing',
        b"vqd='missing",
        b"vqd=missing",
        b'vqd="  "',
        b'vqd="bad token"',
        b'vqd="\xff"',
        b'vqd="bad<token"',
        b'vqd="bad\x00token"',
    ],
)
def test_missing_unterminated_or_malformed_tokens_fail_closed(html):
    fake = FakeDDG(html)
    with pytest.raises(DDGSearchError):
        FixedDDGSearch(fake)("query")
    assert len(fake.calls) == 1


@pytest.mark.parametrize(
    "marker",
    [
        b"CAPTCHA",
        b"challenge-form",
        b"anomaly.js",
        b"anomaly-modal",
        b"Verify you are human",
        b"verify that you are human",
        b"Bots use DuckDuckGo",
    ],
)
def test_challenge_html_with_an_apparent_valid_token_is_rejected(marker):
    fake = FakeDDG(b"<html>" + marker + HTML + b"</html>", _body([_item()]))
    with pytest.raises(DDGSearchError):
        FixedDDGSearch(fake)("query")
    assert len(fake.calls) == 1


@pytest.mark.parametrize(
    "raw", [b"<html>CAPTCHA</html>", b"<html>challenge</html>", HTML]
)
def test_html_instead_of_image_json_is_rejected_without_retry(raw):
    fake = FakeDDG(HTML, raw)
    with pytest.raises(DDGSearchError):
        FixedDDGSearch(fake)("query")
    assert len(fake.calls) == 2


@pytest.mark.parametrize("stage", [0, 1])
@pytest.mark.parametrize("raw", [None, "body", bytearray(HTML), {}, b""])
def test_responses_must_be_nonempty_bytes_without_coercion(stage, raw):
    fake = FakeDDG(raw) if stage == 0 else FakeDDG(HTML, raw)
    with pytest.raises(DDGSearchError):
        FixedDDGSearch(fake)("query")
    assert len(fake.calls) == stage + 1


@pytest.mark.parametrize("stage,cap", [(0, 262_144), (1, 1_048_576)])
def test_response_caps_are_enforced_even_if_injected_get_violates_them(stage, cap):
    raw = b" " * (cap + 1)
    fake = FakeDDG(raw) if stage == 0 else FakeDDG(HTML, raw)
    with pytest.raises(DDGSearchError):
        FixedDDGSearch(fake)("query")
    assert len(fake.calls) == stage + 1


def test_responses_exactly_at_fixed_byte_caps_are_accepted():
    raw = _body([_item()])
    fake = FakeDDG(
        HTML + b" " * (262_144 - len(HTML)), raw + b" " * (1_048_576 - len(raw))
    )
    assert FixedDDGSearch(fake)("query") == [_item()]


@pytest.mark.parametrize(
    "raw",
    [
        b"{",
        b'{"results": []} trailing',
        b"\xff",
        b'{"results": [], "unused": NaN}',
        b'{"results": [], "unused": Infinity}',
        b'{"results": [], "unused": -Infinity}',
    ],
)
def test_malformed_or_nonstandard_json_rejected(raw):
    fake = FakeDDG(HTML, raw)
    with pytest.raises(DDGSearchError):
        FixedDDGSearch(fake)("query")
    assert len(fake.calls) == 2


@pytest.mark.parametrize(
    "payload",
    [
        None,
        True,
        False,
        1,
        1.0,
        "json",
        [],
        [_item()],
        {},
        {"results": None},
        {"results": True},
        {"results": 1},
        {"results": "items"},
        {"results": {}},
    ],
)
def test_json_root_must_be_object_and_results_must_be_present_list(payload):
    fake = FakeDDG(HTML, json.dumps(payload).encode("utf-8"))
    with pytest.raises(DDGSearchError):
        FixedDDGSearch(fake)("query")


def test_empty_result_list_is_valid_and_next_is_ignored():
    fake = FakeDDG(HTML, _body([], next="https://evil.invalid/next"))
    assert FixedDDGSearch(fake)("query") == []
    assert len(fake.calls) == 2


@pytest.mark.parametrize("item", [None, True, False, 1, 1.0, "item", [], ["item"]])
def test_nonobject_items_are_rejected_even_beyond_output_limit(item):
    fake = FakeDDG(HTML, _body([_item(), item]))
    with pytest.raises(DDGSearchError):
        FixedDDGSearch(fake, 1)("query")


@pytest.mark.parametrize("field", ["image", "title", "url", "height", "width"])
def test_required_result_keys_must_be_present(field):
    item = _item()
    del item[field]
    fake = FakeDDG(HTML, _body([item]))
    with pytest.raises(DDGSearchError):
        FixedDDGSearch(fake)("query")


@pytest.mark.parametrize("field", ["image", "title", "url"])
@pytest.mark.parametrize("value", [None, False, True, 1, 1.0, [], {}, "", " \t\n"])
def test_result_strings_are_nonempty_without_coercion_or_silent_filtering(field, value):
    item = _item()
    item[field] = value
    fake = FakeDDG(HTML, _body([_item(), item]))
    with pytest.raises(DDGSearchError):
        FixedDDGSearch(fake, 1)("query")


@pytest.mark.parametrize("field", ["height", "width"])
@pytest.mark.parametrize("value", [None, False, True, 0, -1, 720.0, "720", [], {}])
def test_dimensions_are_positive_exact_integers(field, value):
    item = _item()
    item[field] = value
    fake = FakeDDG(HTML, _body([_item(), item]))
    with pytest.raises(DDGSearchError):
        FixedDDGSearch(fake, 1)("query")


def test_challenge_words_inside_valid_json_are_not_html():
    item = _item()
    item["title"] = "<html>CAPTCHA challenge</html>"
    fake = FakeDDG(HTML, _body([item]))
    assert FixedDDGSearch(fake)("query") == [item]


@pytest.mark.parametrize("stage", [0, 1])
def test_injected_get_errors_propagate_without_fallback_or_retry(stage):
    failure = RuntimeError("bounded GET refused")
    fake = FakeDDG(failure) if stage == 0 else FakeDDG(HTML, failure)
    with pytest.raises(RuntimeError) as caught:
        FixedDDGSearch(fake)("query")
    assert caught.value is failure
    assert len(fake.calls) == stage + 1
