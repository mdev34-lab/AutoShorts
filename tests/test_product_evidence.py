"""Fact atoms: only bindable, non-negated, attributed claims are extracted."""

import pytest

from autoshorts.modules.product.errors import ProductPipelineError
from autoshorts.modules.product.evidence import atoms_from_text
from autoshorts.modules.product.narration import compose_showcase_script
from autoshorts.modules.product.script_guard import assert_script_within_atoms


def _surfaces(text):
    return sorted(a.surface for a in atoms_from_text(text, "loja.example.org"))


@pytest.mark.parametrize(
    "text",
    [
        "Projetor HY300 Pro sem Bluetooth",
        "Projetor HY300 Pro não tem suporte para Bluetooth",
        "Projetor HY300 Pro nem Bluetooth nem HDMI",
        "Fabricado no Japão, certificado CE, alumínio aeronáutico",
        "Bateria para o dia todo",
        "R$ 99 de desconto 20W",
    ],
)
def test_unsupported_or_negated_clauses_yield_no_atoms(text):
    assert _surfaces(text) == []


def test_resolution_never_takes_a_neighbouring_attribute():
    assert _surfaces("8GB RAM Wi-Fi 4K") == ["8GB de memória", "Wi-Fi", "resolução 4K"]


def test_multi_spec_binds_each_quantity_to_its_own_attribute():
    assert _surfaces("8GB de RAM e 256GB de armazenamento") == [
        "256GB de armazenamento",
        "8GB de memória",
    ]


def test_load_versus_weight_ambiguity_is_dropped_not_guessed():
    assert _surfaces("Suporta 2kg peso 1kg") == ["1kg de peso"]


def test_narration_is_attributed_and_deduplicated():
    atoms = atoms_from_text("4K Wi-Fi 8GB RAM", "a.example.org") + atoms_from_text(
        "4K Wi-Fi 8GB RAM", "b.example.org"
    )
    script = compose_showcase_script("Projetor HY300 Pro", atoms)
    text = " ".join(script)
    assert "Segundo o site a.example.org" in text
    assert "b.example.org" not in text
    assert_script_within_atoms(script, atoms)


def test_guard_rejects_spec_and_feature_claims_outside_the_atoms():
    """Finite classes only (specs, known features, certification/price terms):
    the real protection is that the script is composed from atoms alone."""
    atoms = atoms_from_text("8GB RAM Wi-Fi 4K", "a.example.org")
    for bad in (
        "O projetor traz 256GB de armazenamento.",
        "O projetor tem Bluetooth.",
        "O projetor traz 8GB de armazenamento.",
    ):
        with pytest.raises(ProductPipelineError):
            assert_script_within_atoms([bad], atoms)


class _Page:
    url = "https://www.loja.example.org/p/hy300"
    title = "HY300 Smart Projector"
    h1 = "HY300 Smart Projector"
    meta_title = ""

    def __init__(self, html_text):
        self.html_text = html_text


def test_storefront_page_without_main_uses_meta_description_and_skips_junk():
    from autoshorts.modules.product.evidence import page_atoms

    html = (
        '<html><head><meta name="description" content="HY300 delivers 720P HD, '
        '290 ANSI brightness, WiFi 6, Bluetooth 5.4"></head><body>'
        "<nav><li>Frete 99W</li></nav>"
        "<div><p>Gap 7mm entre as peças</p><p>Alto-falante 1*5W</p></div>"
        "</body></html>"
    )
    surfaces = sorted(a.surface for a in page_atoms(_Page(html)))
    assert "290 ANSI de brilho" in surfaces and "resolução 720P" in surfaces
    # bare, attribute-less quantities are never narratable atoms
    assert not any("7mm" in s or "5W" in s or "99W" in s for s in surfaces)


def test_conflicting_numeric_claims_are_omitted_not_chosen():
    from autoshorts.modules.product.evidence import reconcile_atoms

    a = atoms_from_text("290 ANSI brightness, 720P HD, WiFi 6", "a.example.org")
    b = atoms_from_text("170 ANSI brightness, 720P HD, Bluetooth", "b.example.org")
    out = {x.surface for x in reconcile_atoms(a + b)}
    assert not any("ANSI" in s for s in out)  # 290 vs 170: neither is chosen
    assert "resolução 720P" in out and "Wi-Fi" in out and "Bluetooth" in out
    # source COUNT is not truth: an undisputed single-source number is kept
    assert any("290 ANSI" in x.surface for x in reconcile_atoms(a))


def test_same_model_listed_with_another_value_elsewhere_on_the_page_is_disputed():
    from autoshorts.modules.product.evidence import page_atoms

    class P(_Page):
        identity_codes = ["hy300proplus"]

    html = (
        '<html><head><meta name="description" content="HY300Pro+ delivers 720P HD, '
        '290 ANSI brightness, WiFi 6"></head><body>'
        "<nav>HY320 260ANSI 4K HY300Pro+ HOT 720P 200ANSI 4K Wifi6 HY300Pro 720P "
        "150ANSI</nav><div><p>Pronto</p></div></body></html>"
    )
    surfaces = {a.surface for a in page_atoms(P(html))}
    assert not any("ANSI" in s for s in surfaces)  # 290 vs 200 for the same model
    assert "resolução 720P" in surfaces  # no dispute on resolution


def test_composer_output_passes_its_own_guard_for_multi_attribute_facts():
    from autoshorts.modules.product.evidence import atoms_from_text
    from autoshorts.modules.product.narration import compose_showcase_script
    from autoshorts.modules.product.script_guard import assert_script_within_atoms

    atoms = atoms_from_text(
        "8GB memória, 256GB armazenamento, 20W potência", "m.example"
    )
    assert {a.key[3] for a in atoms} == {"memory", "storage", "power"}
    paragraphs = compose_showcase_script("Magcubic HY300 Pro", atoms)
    assert_script_within_atoms(paragraphs, atoms)  # must not raise


def test_guard_still_rejects_swapped_attribute_across_commas():
    import pytest

    from autoshorts.modules.product.errors import ProductPipelineError
    from autoshorts.modules.product.evidence import atoms_from_text
    from autoshorts.modules.product.script_guard import assert_script_within_atoms

    atoms = atoms_from_text("8GB memória, 256GB armazenamento", "m.example")
    with pytest.raises(ProductPipelineError):
        assert_script_within_atoms(["O HY300 Pro traz 256GB de memória."], atoms)


def test_lines_about_another_model_never_feed_atoms():
    from autoshorts.modules.product.evidence import page_atoms
    from autoshorts.modules.product.provenance import _parse_page_html

    html_doc = (
        "<html><head><title>Magcubic HY300 Pro</title></head><body><main>"
        "<h1>Magcubic HY300 Pro</h1><p>HY300 Pro: Wi-Fi.</p>"
        "<p>HY320: Bluetooth HDMI laser.</p></main></body></html>"
    )
    page = _parse_page_html(html_doc, url="https://m.example/p", status=200)
    keys = {a.key for a in page_atoms(page, "hy300pro")}
    assert ("feature", "wi-fi") in keys
    assert not keys & {
        ("feature", "bluetooth"),
        ("feature", "hdmi"),
        ("feature", "laser"),
    }


def test_comparison_title_and_raw_title_conflicts_are_rejected():
    import pytest

    from autoshorts.modules.product.errors import ProductNotFoundError
    from autoshorts.modules.product.provenance import (
        _parse_page_html,
        verify_product_identity,
    )

    def page(title, h1, og=""):
        meta = f'<meta property="og:title" content="{og}">' if og else ""
        return _parse_page_html(
            f"<html><head><title>{title}</title>{meta}</head><body><h1>{h1}</h1></body></html>",
            url="https://m.example/p",
            status=200,
        )

    for p in (
        page("HY300 Pro vs HY300 ProPlus", "HY300 Pro vs HY300 ProPlus"),
        page("HY300 ProPlus", "HY300 Pro", og="HY300 Pro"),
    ):
        with pytest.raises(ProductNotFoundError):
            verify_product_identity("Magcubic HY300 Pro", p, label="t")
    ok = page("Magcubic HY300 Pro", "Magcubic HY300 Pro")
    assert verify_product_identity("Magcubic HY300 Pro", ok, label="t")[
        "identity_match"
    ]


def test_attribute_word_belongs_to_the_quantity_before_it():
    from autoshorts.modules.product.evidence import atoms_from_text

    keys = {a.key for a in atoms_from_text("8GB RAM 128GB de armazenamento", "x.y")}
    assert ("spec", "8", "gb", "memory") in keys
    assert ("spec", "128", "gb", "storage") in keys
    assert ("spec", "128", "gb", "memory") not in keys


def test_spec_words_before_numbers_are_not_other_models():
    from autoshorts.modules.product.evidence import page_atoms
    from autoshorts.modules.product.provenance import _parse_page_html

    desc = (
        "Portable mini projector Magcubic HY300Pro+ delivers 720P HD, 290 ANSI "
        "brightness, WiFi 6, Bluetooth 5.4, Android 14."
    )
    html_doc = (
        "<html><head><title>Magcubic HY300 Pro+</title>"
        f'<meta property="og:description" content="{desc}"></head>'
        "<body><h1>Magcubic HY300 Pro+</h1></body></html>"
    )
    page = _parse_page_html(html_doc, url="https://m.example/p", status=200)
    keys = {a.key for a in page_atoms(page, "hy300proplus")}
    assert ("feature", "bluetooth") in keys and ("feature", "android") in keys


class _ProPage(_Page):
    url = "https://manufacturer.example/p/hy300pro"
    title = "HY300 Pro Projector"
    h1 = "HY300 Pro Projector"


def _pro_surfaces(html_body, expected="hy300pro"):
    from autoshorts.modules.product.evidence import page_atoms

    html = f"<html><body><main>{html_body}</main></body></html>"
    return {a.surface for a in page_atoms(_ProPage(html), expected_code=expected)}


def test_features_under_another_models_heading_do_not_leak_to_this_model():
    out = _pro_surfaces(
        "<p>HY300 Pro: Wi-Fi.</p>"
        "<section><h2>HY320</h2><p>Bluetooth HDMI laser.</p></section>"
    )
    assert "Wi-Fi" in out
    assert not {"Bluetooth", "HDMI", "laser"} & out


def test_heading_scope_ends_at_the_next_heading_of_same_or_higher_level():
    out = _pro_surfaces(
        "<h2>HY320</h2><p>Bluetooth.</p><h2>HY300 Pro</h2><p>Android.</p>"
    )
    assert "Android" in out and "Bluetooth" not in out


def test_items_under_a_negative_section_are_not_positive_atoms():
    for label in ("<h2>Não possui:</h2>", "<p><strong>Sem:</strong></p>"):
        out = _pro_surfaces(
            f"<p>HY300 Pro: Android Wi-Fi.</p>{label}"
            "<ul><li>Bluetooth</li><li>HDMI</li></ul>"
        )
        assert {"Android", "Wi-Fi"} <= out
        assert not {"Bluetooth", "HDMI"} & out, label


def test_compound_variant_spelling_is_bound_like_the_spaced_one():
    from autoshorts.modules.product.errors import ProductNotFoundError
    from autoshorts.modules.product.provenance import (
        PageEvidence,
        model_codes,
        verify_product_identity,
    )

    assert "hy300proplus" in model_codes("HY300 ProPlus")
    for spelling in ("HY300 ProPlus", "HY300 Pro Plus"):
        page = PageEvidence(
            url="https://m.example/p",
            status=200,
            title=spelling,
            h1=spelling,
            meta_title="",
            identity_codes=model_codes(spelling),
        )
        with pytest.raises(ProductNotFoundError):
            verify_product_identity("HY300", page, label="x")


def test_section_boundaries_keep_own_model_and_positive_labels():
    # a heading that names THIS model is not "another model"
    assert "Android" in _pro_surfaces("<h2>HY300 Pro</h2><p>Android.</p>")
    # a positive label after a negative one replaces it
    out = _pro_surfaces(
        "<p>Não possui:</p><ul><li>HDMI</li></ul>"
        "<p>Possui:</p><ul><li>Bluetooth</li></ul>"
    )
    assert "Bluetooth" in out and "HDMI" not in out
    # a new heading ends a label's scope
    out = _pro_surfaces(
        "<p>Sem:</p><ul><li>HDMI</li></ul><h2>Recursos</h2><p>Wi-Fi.</p>"
    )
    assert "Wi-Fi" in out and "HDMI" not in out
    # deeper headings stay inside a negative or other-model section ...
    out = _pro_surfaces(
        "<h2>Not included</h2><h3>Recursos</h3><p>HDMI.</p>"
        "<h2>HY320</h2><h3>Recursos</h3><p>Laser.</p>"
    )
    assert not {"HDMI", "laser"} & out
    # ... and a sibling heading closes them
    assert "Android" in _pro_surfaces(
        "<h2>Not included</h2><p>HDMI.</p><h2>Recursos</h2><p>Android.</p>"
    )


def test_negative_vocabulary_variants_in_headings_and_labels():
    for label in (
        "<h3>Not included</h3>",
        "<h3>Without</h3>",
        "<h3>Sem</h3>",
        "<h3>Não inclui</h3>",
        "<p>Não vem com:</p>",
        "<p>Does not have:</p>",
    ):
        out = _pro_surfaces(f"<p>Android.</p>{label}<ul><li>HDMI</li></ul>")
        assert "HDMI" not in out, label
        assert "Android" in out, label


def test_other_model_heading_variants_and_unrelated_headings():
    for heading in ("HY320", "HY 320", "Magcubic HY320 Pro", "HY300 ProPlus"):
        out = _pro_surfaces(f"<h2>{heading}</h2><p>Laser.</p>")
        assert "laser" not in out, heading
    # generic spec words in a heading are not another model
    out = _pro_surfaces("<h2>Native 720P HD, 200 ANSI</h2><p>Android.</p>")
    assert "Android" in out


def test_variant_suffix_lookalikes_are_not_variants():
    from autoshorts.modules.product.provenance import model_codes

    for word in ("Pronto", "Maxim", "Proton", "Plusieurs", "Neon"):
        assert "hy300" in model_codes(f"HY300 {word}"), word
        assert not any(
            c.startswith("hy300")
            and c != "hy300"
            and len(c) > 5
            and c.endswith(word.lower())
            for c in model_codes(f"HY300 {word}")
        ), word
    for glued in ("proplus", "promax", "ultraplus", "proultramax"):
        assert f"hy300{glued}" in model_codes(f"HY300 {glued}"), glued


def test_hd_is_a_model_prefix_unless_the_number_is_a_spec_value():
    from autoshorts.modules.product.errors import ProductNotFoundError
    from autoshorts.modules.product.provenance import (
        PageEvidence,
        model_codes,
        verify_product_identity,
    )

    # spec fragments: resolution word + resolution or unit-bearing number
    for text in (
        "HY300 Pro, native 720P HD, 200 ANSI",
        "720P HD, 290 ANSI brightness",
        "Full HD 1080 native",
        "HD290 ANSI",
        "HD 200 lumens",
    ):
        assert not any(c.startswith(("hd", "fhd")) for c in model_codes(text)), text
    # real model family: HD146X is a product, and HD200 is a DIFFERENT one
    assert "hd146x" in model_codes("Optoma HD146X Projector")
    assert "hd200" in model_codes("Projector HD200")

    def page(title):
        return PageEvidence(
            url="https://m.example/p",
            status=200,
            title=title,
            h1=title,
            meta_title="",
            identity_codes=model_codes(title),
        )

    assert verify_product_identity(
        "HD146X", page("HD146X Projector 720P HD 290 ANSI"), label="x"
    )
    for wrong in ("HD200 Projector", "HD146X and HD200 compared", "Optoma HD 200"):
        with pytest.raises(ProductNotFoundError):
            verify_product_identity("HD146X", page(wrong), label="x")


def test_section_for_another_hd_model_does_not_leak():
    from autoshorts.modules.product.evidence import page_atoms

    class P(_Page):
        url = "https://manufacturer.example/p/hd146x"
        title = h1 = "Optoma HD146X"

    html = "<main><p>HD146X: Wi-Fi.</p><h2>HD200</h2><p>Bluetooth HDMI.</p></main>"
    out = {a.surface for a in page_atoms(P(html), expected_code="hd146x")}
    assert out == {"Wi-Fi"}


def test_bare_hd_models_stay_identifiable_next_to_legit_units():
    from autoshorts.modules.product.provenance import model_codes

    for text, code in (
        ("HD200 Projector, 290 ANSI", "hd200"),
        ("HD146X 3600 lumens", "hd146x"),
        ("Optoma HD146X 720P HD 3600 lumens", "hd146x"),
        ("BenQ HD 200 Projector 4K", "hd200"),
        ("HD200 and 290 ANSI brightness", "hd200"),
        ("720P HD, 290 ANSI. HD200 Projector", "hd200"),
    ):
        assert code in model_codes(text), text
