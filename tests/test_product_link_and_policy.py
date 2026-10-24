import pytest

from autoshorts.modules.product.errors import (
    InvalidProductLinkError,
    SourceBlockedError,
)
from autoshorts.modules.product.network import (
    assert_safe_request_url,
    is_shopee_cdn_host,
)
from autoshorts.modules.product.product import (
    normalized_product_query,
    parse_product_link,
)


def test_parse_full_slug_with_ids():
    p = parse_product_link(
        "https://shopee.com.br/Projetor-HY300-Pro-4K-Android-11-Wifi6-i.123456.789012"
    )
    assert p.shop_id == "123456" and p.item_id == "789012"
    assert p.slug == "Projetor-HY300-Pro-4K-Android-11-Wifi6"
    assert "HY300" in p.name and "Projetor" in p.name


def test_parse_rejects_non_shopee():
    with pytest.raises(InvalidProductLinkError):
        parse_product_link("https://example.com/whatever-i.1.2")


def test_parse_rejects_category_path():
    with pytest.raises(InvalidProductLinkError):
        parse_product_link("https://shopee.com.br/search")


def test_parse_requires_letters_in_name():
    with pytest.raises(InvalidProductLinkError):
        parse_product_link("https://shopee.com.br/123-456-i.1.2")


def test_policy_blocks_shopee_and_cdn():
    assert is_shopee_cdn_host("cf.shopee.com.br")
    assert is_shopee_cdn_host("down-brc.img.susercontent.com")
    with pytest.raises(SourceBlockedError):
        assert_safe_request_url("https://shopee.com.br/x", operation="t")
    with pytest.raises(SourceBlockedError):
        assert_safe_request_url("https://img.susercontent.com/x.jpg", operation="t")
    with pytest.raises(SourceBlockedError):
        assert_safe_request_url("http://insecure.example.com/x", operation="t")
    assert assert_safe_request_url("https://images.example.com/x.jpg", operation="t")


def test_shp_ee_needs_resolver():
    with pytest.raises(SourceBlockedError):
        assert_safe_request_url("https://shp.ee/abc", operation="t")
    # allowed only for the resolver itself
    assert assert_safe_request_url(
        "https://shp.ee/abc", operation="t", allow_shp_ee=True
    )


def test_normalized_query_keeps_codes():
    q = normalized_product_query("Projetor HY300 Pro 4K Android 11 Wifi6")
    assert "HY300" in q and "4K" in q


def test_slug_markup_stripped_and_bounded():
    from autoshorts.modules.product.product import parse_product_link

    p = parse_product_link("https://shopee.com.br/Projetor-HY300-%3Cb%3E-i.1.2")
    assert "<" not in p.name and ">" not in p.name
    assert p.name == "Projetor HY300"

    long_slug = "Projetor-HY300-" + "x" * 400
    with pytest.raises(InvalidProductLinkError):
        parse_product_link(f"https://shopee.com.br/{long_slug}-i.1.2")
