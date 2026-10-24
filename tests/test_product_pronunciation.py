from autoshorts.modules.product.pronunciation import prepare_for_tts


def test_brand_decimal_and_code():
    out = prepare_for_tts("Magcubic HY300 com Bluetooth 5.4 e Wi-Fi 6, 1,6 metro.")
    assert "Mag quiúbic" in out and "agá ípsilon trezentos" in out
    assert "cinco vírgula quatro" in out and "um vírgula seis" in out
    assert "uai fai" in out and "blu tuf" in out


def test_user_dictionary_overrides():
    assert prepare_for_tts("Foo", {"foo": "fú"}) == "fú"


def test_plain_integers_untouched():
    assert "135" in prepare_for_tts("até 135 polegadas")


def test_operator_file_is_trusted_local_config_shape_checked_only(tmp_path):
    """The file is honored verbatim (it may change spoken words by design);
    only its JSON shape is validated. No length/content heuristic."""
    import json

    import pytest

    from autoshorts.modules.product.pronunciation import load_dictionary

    f = tmp_path / "p.json"
    f.write_text(json.dumps({"HY300": "agá ípsilon trezentos"}), encoding="utf-8")
    assert load_dictionary(f)["HY300"] == "agá ípsilon trezentos"
    f.write_text(json.dumps(["not", "an", "object"]), encoding="utf-8")
    with pytest.raises(ValueError):
        load_dictionary(f)


def test_untrusted_text_cannot_reach_the_dictionary():
    """Text is only ever the INPUT of prepare_for_tts: respelling-like JSON in
    it defines nothing, so a later word is not rewritten by it."""
    out = prepare_for_tts('{"Zorb": "Feito no Japão"} Zorb', {})
    assert out.endswith("Zorb")
    assert out.count("Japão") == 1


def test_legitimate_respellings_still_load(tmp_path):
    import json

    from autoshorts.modules.product.pronunciation import load_dictionary

    f = tmp_path / "ok.json"
    f.write_text(
        json.dumps({"Xiaomi": "chaomi", "HDMI": "agá dê ême i", "JBL": "jota bê éle"}),
        encoding="utf-8",
    )
    assert load_dictionary(f)["Xiaomi"] == "chaomi"
