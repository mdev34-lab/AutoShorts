from unittest.mock import patch

import pytest

from autoshorts.modules.product.discovery import ImageCandidate
from autoshorts.modules.product.errors import ProductNotFound
from autoshorts.modules.product.provenance import (
    PageEvidence,
    model_codes,
    primary_model_code,
    verify_image_sources,
    verify_product_identity,
)


def test_model_code_keeps_plus_variant_distinct():
    assert primary_model_code("MAGCUBIC HY300ProPlus Projector 290 ANSI") == "hy300proplus"
    assert primary_model_code("MAGCUBIC HY300 Pro Projector 200 ANSI") == "hy300pro"
    assert "hy300proplus" in model_codes("Magcubic HY300Pro+ Smart Mini Projector")
    assert "hy300proplus" not in model_codes("Magcubic HY300Pro Smart Mini Projector")


def test_wrong_model_page_is_rejected():
    page = PageEvidence(
        url="https://example.org/product",
        status=200,
        title="Replacement Pads MDR-ZX110",
        h1="Replacement Pads MDR-ZX110",
        meta_title="",
        identity_codes=model_codes("Replacement Pads MDR-ZX110"),
    )
    with pytest.raises(ProductNotFound):
        verify_product_identity("JBL Tune 510BT Ear Pads", page, label="photo")


def _magcubic_page(page_url, images):
    return PageEvidence(
        url=page_url,
        status=200,
        title="Magcubic HY300Pro+ Smart Mini Projector",
        h1="Magcubic HY300Pro+ Smart Mini Projector",
        meta_title="",
        identity_codes=model_codes("Magcubic HY300Pro+ Smart Mini Projector"),
        images=images,
    )


def test_photo_must_be_bound_to_model_labelled_gallery_image():
    product_name = "MAGCUBIC HY300ProPlus Projector 290 ANSI"
    page_url = "https://www.magcubic.com/products/miniprojector-hy300proplus"
    image_url = "https://www.magcubic.com/cdn/shop/files/hy300_pro.png?v=1&width=600"
    page = _magcubic_page(page_url, [{
        "alt": "Magcubic HY300Pro+ Smart Mini Projector",
        "sources": "//www.magcubic.com/cdn/shop/files/hy300_pro.png?v=1&width=600",
    }])
    candidate = ImageCandidate(image_url, page_url, "hy300_pro.png", "www.magcubic.com", False)
    with patch("autoshorts.modules.product.provenance.fetch_page_evidence", return_value=page):
        evidence = verify_image_sources([candidate], product_name)
    assert evidence[0]["identity_match"]
    assert evidence[0]["image_bound_to_page"]
    assert evidence[0]["expected_model_code"] == "hy300proplus"


def test_correct_model_page_does_not_authorize_unlisted_image():
    product_name = "MAGCUBIC HY300ProPlus Projector 290 ANSI"
    page_url = "https://www.magcubic.com/products/miniprojector-hy300proplus"
    page = _magcubic_page(page_url, [])
    candidate = ImageCandidate(
        "https://images.example.org/unrelated.jpg", page_url, "unrelated.jpg",
        "images.example.org", False,
    )
    with patch("autoshorts.modules.product.provenance.fetch_page_evidence", return_value=page):
        with pytest.raises(ProductNotFound):
            verify_image_sources([candidate], product_name)
