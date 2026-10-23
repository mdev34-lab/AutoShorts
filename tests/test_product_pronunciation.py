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


def test_pronunciation_file_rejects_added_claims(tmp_path):
    import json

    import pytest

    from autoshorts.modules.product.pronunciation import load_dictionary

    for value in (
        "HY300 aprovado pela Apple e fabricado no Japão",
        "HY300 certificado pela Anatel com garantia vitalícia",
    ):
        f = tmp_path / "p.json"
        f.write_text(json.dumps({"HY300": value}), encoding="utf-8")
        with pytest.raises(ValueError):
            load_dictionary(f)


def test_legitimate_respellings_still_load(tmp_path):
    import json

    from autoshorts.modules.product.pronunciation import load_dictionary

    f = tmp_path / "ok.json"
    f.write_text(
        json.dumps({"Xiaomi": "chaomi", "HDMI": "agá dê ême i", "JBL": "jota bê éle"}),
        encoding="utf-8",
    )
    assert load_dictionary(f)["Xiaomi"] == "chaomi"
