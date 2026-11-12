"""Offline validation and raw-counterexample coverage for resume inputs."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from autoshorts.modules.resume_inputs import (
    AISourceError,
    ManifestMissingError,
    ManifestSchemaError,
    ManifestTamperError,
    ResumeInputError,
    ResumeInputs,
    load_resume_inputs,
)


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _json_bytes(value) -> bytes:
    return json.dumps(value, separators=(",", ":")).encode("utf-8")


@pytest.fixture
def manifest(tmp_path: Path) -> dict:
    audio = b"raw-audio-bytes-123"
    transcript = b"Recorded sentence."
    (tmp_path / "audio.wav").write_bytes(audio)
    (tmp_path / "transcript.txt").write_bytes(transcript)
    return {
        "schema": 1,
        "audio_path": "audio.wav",
        "audio_sha256": _sha256(audio),
        "transcript_path": "transcript.txt",
        "transcript_sha256": _sha256(transcript),
        "transcript_provenance": "field-recording",
        "queries": ["valid query"],
        "queries_sha256": _sha256(_json_bytes(["valid query"])),
    }


def _write_transcript(
    tmp_path: Path, manifest: dict, raw: bytes, name="transcript.txt"
):
    (tmp_path / name).write_bytes(raw)
    manifest["transcript_path"] = name
    manifest["transcript_sha256"] = _sha256(raw)


def _file_list(tmp_path: Path, manifest: dict, field: str, raw: bytes):
    name = f"{field}.json"
    (tmp_path / name).write_bytes(raw)
    manifest.pop(field, None)
    manifest[f"{field}_path"] = name
    manifest[f"{field}_sha256"] = _sha256(raw)


def test_schema1_manifest_success_with_exact_transcript_and_tts_provenance(
    tmp_path, manifest
):
    exact_text = "  Flamengo vs Grêmio Maracanã 2026. \n\t"
    provenance = {"provider": "piper-tts", "model": "pt_BR-faber"}
    manifest.pop("transcript_provenance")
    _write_transcript(
        tmp_path,
        manifest,
        _json_bytes({"text": exact_text, "provenance": provenance}),
        "transcript.json",
    )
    manifest_file = tmp_path / "manifest.json"
    manifest_file.write_bytes(_json_bytes(manifest))

    inputs = load_resume_inputs(manifest_file)
    assert isinstance(inputs, ResumeInputs)
    assert inputs.audio_path == (tmp_path / "audio.wav").resolve()
    assert inputs.transcript_path == (tmp_path / "transcript.json").resolve()
    assert inputs.transcript_text == exact_text
    assert inputs.transcript_provenance == provenance
    assert inputs.paragraphs is None
    assert inputs.queries == ("valid query",)
    assert inputs.audio_sha256 == manifest["audio_sha256"]
    assert inputs.transcript_sha256 == manifest["transcript_sha256"]


def test_plain_transcript_whitespace_preserved(tmp_path, manifest, monkeypatch):
    exact_text = " \tRecorded sentence.\r\nSecond sentence.  "
    _write_transcript(tmp_path, manifest, exact_text.encode("utf-8"))
    monkeypatch.chdir(tmp_path)
    assert load_resume_inputs(manifest).transcript_text == exact_text


def test_bool_schema_counterexample_rejected(tmp_path, manifest):
    manifest["schema"] = True
    with pytest.raises(ManifestSchemaError):
        load_resume_inputs(manifest, base_dir=tmp_path)


@pytest.mark.parametrize("schema", [None, False, 0, 2, 1.0, "1", [], {}])
def test_schema_must_be_exact_integer_one(tmp_path, manifest, schema):
    manifest["schema"] = schema
    with pytest.raises(ManifestSchemaError):
        load_resume_inputs(manifest, base_dir=tmp_path)


def test_missing_schema_rejected(tmp_path, manifest):
    manifest.pop("schema")
    with pytest.raises(ManifestSchemaError):
        load_resume_inputs(manifest, base_dir=tmp_path)


@pytest.mark.parametrize("root", [None, [], ["object"], "text", True, False, 1, 1.0])
def test_manifest_json_root_must_be_object(tmp_path, root):
    path = tmp_path / "manifest.json"
    path.write_bytes(_json_bytes(root))
    with pytest.raises(ManifestSchemaError):
        load_resume_inputs(path)


@pytest.mark.parametrize("value", [None, False, 1, 1.0, [], (), "", " \t\n"])
def test_invalid_manifest_argument_rejected(value):
    with pytest.raises(ManifestSchemaError):
        load_resume_inputs(value)


@pytest.mark.parametrize("raw", [b"", b"{", b"\xff"])
def test_malformed_manifest_rejected(tmp_path, raw):
    path = tmp_path / "manifest.json"
    path.write_bytes(raw)
    with pytest.raises(ManifestSchemaError):
        load_resume_inputs(path)


def test_missing_manifest_file_rejected(tmp_path):
    with pytest.raises(ManifestMissingError):
        load_resume_inputs(tmp_path / "missing.json")


@pytest.mark.parametrize(
    "field", ["audio_path", "audio_sha256", "transcript_path", "transcript_sha256"]
)
def test_missing_required_fields_rejected(tmp_path, manifest, field):
    manifest.pop(field)
    with pytest.raises(ManifestMissingError):
        load_resume_inputs(manifest, base_dir=tmp_path)


@pytest.mark.parametrize(
    "field", ["audio_path", "transcript_path", "queries_path", "paragraphs_path"]
)
@pytest.mark.parametrize(
    "value", [None, False, 1, [], {}, Path("asset.json"), "", " \t\n"]
)
def test_asset_paths_must_be_nonempty_strings(tmp_path, manifest, field, value):
    if field in ("queries_path", "paragraphs_path"):
        asset = field.removesuffix("_path")
        manifest.pop(asset, None)
        manifest[f"{asset}_sha256"] = "0" * 64
    manifest[field] = value
    with pytest.raises(ManifestSchemaError):
        load_resume_inputs(manifest, base_dir=tmp_path)


@pytest.mark.parametrize(
    "field",
    ["audio_sha256", "transcript_sha256", "queries_sha256", "paragraphs_sha256"],
)
@pytest.mark.parametrize(
    "value",
    [
        None,
        False,
        True,
        1,
        [],
        {},
        "",
        "a" * 63,
        "a" * 65,
        "g" * 64,
        " " * 64,
        "a" * 64 + "\n",
    ],
)
def test_hashes_must_be_64_hex_strings(tmp_path, manifest, field, value):
    if field == "paragraphs_sha256":
        manifest["paragraphs"] = ["paragraph"]
    manifest[field] = value
    with pytest.raises(ManifestSchemaError):
        load_resume_inputs(manifest, base_dir=tmp_path)


@pytest.mark.parametrize("field", ["queries", "paragraphs"])
def test_file_list_hash_is_validated_without_coercion(tmp_path, manifest, field):
    _file_list(tmp_path, manifest, field, _json_bytes(["valid"]))
    manifest[f"{field}_sha256"] = True
    with pytest.raises(ManifestSchemaError):
        load_resume_inputs(manifest, base_dir=tmp_path)


@pytest.mark.parametrize("field", ["queries", "paragraphs"])
@pytest.mark.parametrize("storage", ["inline", "file"])
def test_saved_lists_require_hashes(tmp_path, manifest, field, storage):
    if storage == "file":
        _file_list(tmp_path, manifest, field, _json_bytes(["valid"]))
    else:
        manifest[field] = ["valid"]
    manifest.pop(f"{field}_sha256", None)
    with pytest.raises(ManifestMissingError):
        load_resume_inputs(manifest, base_dir=tmp_path)


@pytest.mark.parametrize(
    "field", ["audio_path", "transcript_path", "queries_path", "paragraphs_path"]
)
def test_missing_asset_files_rejected(tmp_path, manifest, field):
    if field in ("queries_path", "paragraphs_path"):
        asset = field.removesuffix("_path")
        manifest.pop(asset, None)
        manifest[f"{asset}_sha256"] = "0" * 64
    manifest[field] = "missing.asset"
    with pytest.raises(ManifestMissingError):
        load_resume_inputs(manifest, base_dir=tmp_path)


@pytest.mark.parametrize("asset", ["audio", "transcript"])
def test_empty_assets_rejected_even_with_matching_raw_hash(tmp_path, manifest, asset):
    (tmp_path / manifest[f"{asset}_path"]).write_bytes(b"")
    manifest[f"{asset}_sha256"] = _sha256(b"")
    with pytest.raises(ManifestMissingError):
        load_resume_inputs(manifest, base_dir=tmp_path)


@pytest.mark.parametrize("asset", ["audio", "transcript"])
def test_raw_asset_tampering_rejected(tmp_path, manifest, asset):
    path = tmp_path / manifest[f"{asset}_path"]
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ManifestTamperError):
        load_resume_inputs(manifest, base_dir=tmp_path)


def test_transcript_hash_checks_raw_json_whitespace(tmp_path, manifest):
    payload = {"text": "Recorded sentence."}
    _write_transcript(tmp_path, manifest, _json_bytes(payload), "transcript.json")
    (tmp_path / "transcript.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    with pytest.raises(ManifestTamperError):
        load_resume_inputs(manifest, base_dir=tmp_path)


@pytest.mark.parametrize("field,value", [("queries", "fruit"), ("paragraphs", "text")])
def test_string_list_raw_counterexamples_rejected(tmp_path, manifest, field, value):
    manifest[field] = value
    manifest[f"{field}_sha256"] = _sha256(_json_bytes(value))
    with pytest.raises(ManifestSchemaError):
        load_resume_inputs(manifest, base_dir=tmp_path)


INVALID_LISTS = [
    None,
    True,
    False,
    1,
    1.0,
    "",
    "fruit",
    {},
    {"query": "value"},
    [],
    [""],
    [" \t\n"],
    [None],
    [False],
    [1],
    [{}],
    [["nested"]],
    ["valid", ""],
]


@pytest.mark.parametrize("field", ["queries", "paragraphs"])
@pytest.mark.parametrize("value", INVALID_LISTS + [("valid",)])
def test_inline_lists_require_nonempty_lists_of_nonempty_strings(
    tmp_path, manifest, field, value
):
    manifest[field] = value
    manifest[f"{field}_sha256"] = _sha256(_json_bytes(value))
    with pytest.raises(ManifestSchemaError):
        load_resume_inputs(manifest, base_dir=tmp_path)


@pytest.mark.parametrize("field", ["queries", "paragraphs"])
@pytest.mark.parametrize("value", INVALID_LISTS)
def test_file_lists_require_nonempty_lists_of_nonempty_strings(
    tmp_path, manifest, field, value
):
    _file_list(tmp_path, manifest, field, _json_bytes(value))
    with pytest.raises(ManifestSchemaError):
        load_resume_inputs(manifest, base_dir=tmp_path)


@pytest.mark.parametrize("field", ["queries", "paragraphs"])
@pytest.mark.parametrize("storage", ["inline", "file"])
def test_saved_lists_preserve_strings_and_accept_uppercase_hashes(
    tmp_path, manifest, field, storage
):
    values = ["  Flamengo Grêmio  ", "Maracanã\n"]
    if storage == "file":
        _file_list(
            tmp_path,
            manifest,
            field,
            json.dumps(values, ensure_ascii=False, indent=2).encode("utf-8"),
        )
    else:
        manifest[field] = values
        manifest[f"{field}_sha256"] = _sha256(_json_bytes(values))
    for key in ("audio_sha256", "transcript_sha256", f"{field}_sha256"):
        manifest[key] = manifest[key].upper()
    inputs = load_resume_inputs(manifest, base_dir=tmp_path)
    assert getattr(inputs, field) == tuple(values)
    assert inputs.audio_sha256 == manifest["audio_sha256"].lower()
    assert inputs.transcript_sha256 == manifest["transcript_sha256"].lower()


@pytest.mark.parametrize("field", ["queries", "paragraphs"])
@pytest.mark.parametrize("storage", ["inline", "file"])
def test_list_hash_mismatch_rejected(tmp_path, manifest, field, storage):
    values = ["Flamengo Grêmio", "Maracanã"]
    if storage == "file":
        _file_list(
            tmp_path, manifest, field, json.dumps(values, indent=2).encode("utf-8")
        )
        # Canonical JSON is not the saved file's raw bytes.
        manifest[f"{field}_sha256"] = _sha256(_json_bytes(values))
    else:
        manifest[field] = values
        manifest[f"{field}_sha256"] = "0" * 64
    with pytest.raises(ManifestTamperError):
        load_resume_inputs(manifest, base_dir=tmp_path)


@pytest.mark.parametrize("field", ["queries", "paragraphs"])
@pytest.mark.parametrize(
    "raw,error",
    [
        (b"", ManifestMissingError),
        (b" ", ManifestSchemaError),
        (b"{", ManifestSchemaError),
        (b"\xff", ManifestSchemaError),
    ],
)
def test_malformed_list_files_rejected(tmp_path, manifest, field, raw, error):
    _file_list(tmp_path, manifest, field, raw)
    with pytest.raises(error):
        load_resume_inputs(manifest, base_dir=tmp_path)


@pytest.mark.parametrize("field", ["queries", "paragraphs"])
def test_conflicting_inline_and_file_lists_rejected(tmp_path, manifest, field):
    manifest[field] = ["valid"]
    manifest[f"{field}_sha256"] = _sha256(_json_bytes(["valid"]))
    manifest[f"{field}_path"] = "missing.json"
    with pytest.raises(ManifestSchemaError):
        load_resume_inputs(manifest, base_dir=tmp_path)


@pytest.mark.parametrize("field", ["queries", "paragraphs"])
def test_orphan_list_hash_rejected(tmp_path, manifest, field):
    manifest.pop(field, None)
    manifest[f"{field}_sha256"] = "0" * 64
    with pytest.raises(ManifestSchemaError):
        load_resume_inputs(manifest, base_dir=tmp_path)


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"text": None},
        {"text": False},
        {"text": True},
        {"text": 1},
        {"text": 1.0},
        {"text": []},
        {"text": {}},
        {"text": ""},
        {"text": " \t\n"},
    ],
)
def test_json_transcript_requires_nonempty_string_text(tmp_path, manifest, payload):
    _write_transcript(tmp_path, manifest, _json_bytes(payload), "transcript.json")
    with pytest.raises(ManifestSchemaError):
        load_resume_inputs(manifest, base_dir=tmp_path)


@pytest.mark.parametrize("payload", [None, True, False, 1, 1.0, [], ["text"], "text"])
@pytest.mark.parametrize("name", ["transcript.json", "transcript.txt"])
def test_json_transcript_nonobjects_rejected_not_used_as_raw_text(
    tmp_path, manifest, payload, name
):
    _write_transcript(tmp_path, manifest, _json_bytes(payload), name)
    with pytest.raises(ManifestSchemaError):
        load_resume_inputs(manifest, base_dir=tmp_path)


@pytest.mark.parametrize("raw", [b'{"text":', b"not JSON"])
def test_malformed_json_transcript_rejected(tmp_path, manifest, raw):
    _write_transcript(tmp_path, manifest, raw, "transcript.json")
    with pytest.raises(ManifestSchemaError):
        load_resume_inputs(manifest, base_dir=tmp_path)


@pytest.mark.parametrize("raw", [b" \t\n", b"\xff"])
def test_blank_or_non_utf8_transcript_rejected(tmp_path, manifest, raw):
    _write_transcript(tmp_path, manifest, raw)
    with pytest.raises(ManifestSchemaError):
        load_resume_inputs(manifest, base_dir=tmp_path)


def _put_provenance(tmp_path, manifest, location, field, value):
    manifest.pop("transcript_provenance")
    if location == "manifest":
        manifest[field] = value
    else:
        _write_transcript(
            tmp_path,
            manifest,
            _json_bytes({"text": "Recorded sentence.", field: value}),
            "transcript.json",
        )


@pytest.mark.parametrize("location", ["manifest", "transcript"])
@pytest.mark.parametrize("field", ["transcript_provenance", "provenance"])
@pytest.mark.parametrize(
    "value", ["manual-notes", "  manual-notes\n", {"provider": "piper-tts"}]
)
def test_explicit_provenance_accepted_without_rewriting(
    tmp_path, manifest, location, field, value
):
    _put_provenance(tmp_path, manifest, location, field, value)
    assert (
        load_resume_inputs(manifest, base_dir=tmp_path).transcript_provenance == value
    )


@pytest.mark.parametrize("location", ["manifest", "transcript"])
@pytest.mark.parametrize("field", ["transcript_provenance", "provenance"])
@pytest.mark.parametrize(
    "value", [None, "", " \t\n", {}, False, True, 0, 1, 1.0, [], ["recorded"]]
)
def test_invalid_provenance_rejected(tmp_path, manifest, location, field, value):
    _put_provenance(tmp_path, manifest, location, field, value)
    with pytest.raises(ManifestSchemaError):
        load_resume_inputs(manifest, base_dir=tmp_path)


@pytest.mark.parametrize("json_transcript", [False, True])
def test_missing_provenance_rejected(tmp_path, manifest, json_transcript):
    manifest.pop("transcript_provenance")
    if json_transcript:
        _write_transcript(
            tmp_path,
            manifest,
            _json_bytes({"text": "Recorded sentence.", "provider": "tts"}),
            "transcript.json",
        )
    with pytest.raises(ManifestMissingError):
        load_resume_inputs(manifest, base_dir=tmp_path)


@pytest.mark.parametrize("location", ["manifest", "transcript"])
def test_invalid_provenance_does_not_fall_back_to_valid_alias(
    tmp_path, manifest, location
):
    _put_provenance(tmp_path, manifest, location, "transcript_provenance", False)
    if location == "manifest":
        manifest["provenance"] = "manual-notes"
    else:
        _write_transcript(
            tmp_path,
            manifest,
            _json_bytes(
                {
                    "text": "Recorded sentence.",
                    "transcript_provenance": False,
                    "provenance": "manual-notes",
                }
            ),
            "transcript.json",
        )
    with pytest.raises(ManifestSchemaError):
        load_resume_inputs(manifest, base_dir=tmp_path)


def test_invalid_embedded_provenance_not_hidden_by_manifest_provenance(
    tmp_path, manifest
):
    _write_transcript(
        tmp_path,
        manifest,
        _json_bytes({"text": "Recorded sentence.", "provenance": []}),
        "transcript.json",
    )
    with pytest.raises(ManifestSchemaError):
        load_resume_inputs(manifest, base_dir=tmp_path)


def test_valid_provenance_precedence_preserved(tmp_path, manifest):
    manifest["provenance"] = "manifest-alias"
    _write_transcript(
        tmp_path,
        manifest,
        _json_bytes(
            {
                "text": "Recorded sentence.",
                "transcript_provenance": "embedded-primary",
                "provenance": "embedded-alias",
            }
        ),
        "transcript.json",
    )
    assert (
        load_resume_inputs(manifest, base_dir=tmp_path).transcript_provenance
        == "field-recording"
    )


def test_missing_queries_uses_retry_subject_query(tmp_path, manifest):
    manifest.pop("queries")
    manifest.pop("queries_sha256")
    _write_transcript(
        tmp_path,
        manifest,
        "Flamengo Maracanã x Grêmio.\nflamengo contra grêmio.".encode("utf-8"),
    )
    inputs = load_resume_inputs(manifest, base_dir=tmp_path)
    assert inputs.queries == ("Flamengo Maracanã Grêmio", "flamengo grêmio")


@pytest.mark.parametrize("retry_result", ["", None])
def test_empty_retry_raw_counterexample_uses_same_original_sentence(
    tmp_path, manifest, monkeypatch, retry_result
):
    manifest.pop("queries")
    manifest.pop("queries_sha256")
    seen = []

    def retry(sentence):
        seen.append(sentence)
        return retry_result

    monkeypatch.setattr(
        "autoshorts.modules.resume_inputs.image_sources.retry_subject_query", retry
    )
    inputs = load_resume_inputs(manifest, base_dir=tmp_path)
    assert seen == ["Recorded sentence."]
    assert inputs.queries == ("Recorded sentence.",)


def test_ambiguous_retry_keeps_original_venue_only_sentence(tmp_path, manifest):
    manifest.pop("queries")
    manifest.pop("queries_sha256")
    _write_transcript(tmp_path, manifest, "Maracanã x Flamengo".encode("utf-8"))
    assert load_resume_inputs(manifest, base_dir=tmp_path).queries == (
        "Maracanã x Flamengo",
    )


def test_every_derived_sentence_is_retry_or_unchanged_without_semantic_filtering(
    tmp_path, manifest, monkeypatch
):
    text = "  First sentence.\n\nMaracanã x Flamengo\n \nE então.\nMaracanã x Flamengo\nLast sentence  "
    manifest.pop("queries")
    manifest.pop("queries_sha256")
    _write_transcript(tmp_path, manifest, text.encode("utf-8"))
    seen = []

    def retry(sentence):
        seen.append(sentence)
        return "subject query" if sentence == "  First sentence." else ""

    monkeypatch.setattr(
        "autoshorts.modules.resume_inputs.image_sources.retry_subject_query", retry
    )
    inputs = load_resume_inputs(manifest, base_dir=tmp_path)
    assert seen == [
        "  First sentence.",
        "Maracanã x Flamengo",
        "E então.",
        "Maracanã x Flamengo",
        "Last sentence  ",
    ]
    assert inputs.queries == (
        "subject query",
        "Maracanã x Flamengo",
        "E então.",
        "Maracanã x Flamengo",
        "Last sentence  ",
    )
    assert inputs.transcript_text == text


@pytest.mark.parametrize("storage", ["inline", "file"])
def test_saved_queries_are_not_rederived_or_semantically_filtered(
    tmp_path, manifest, monkeypatch, storage
):
    values = ["Maracanã x Flamengo", "E então."]
    if storage == "file":
        _file_list(tmp_path, manifest, "queries", _json_bytes(values))
    else:
        manifest["queries"] = values
        manifest["queries_sha256"] = _sha256(_json_bytes(values))

    def forbidden_retry(sentence):
        raise AssertionError("Saved queries must not be rederived")

    monkeypatch.setattr(
        "autoshorts.modules.resume_inputs.image_sources.retry_subject_query",
        forbidden_retry,
    )
    assert load_resume_inputs(manifest, base_dir=tmp_path).queries == tuple(values)


def test_leading_year_keeps_number_and_prior_short_derivation(tmp_path, manifest):
    manifest.pop("queries")
    manifest.pop("queries_sha256")
    text = "  1994 Flamengo campeão  "
    _write_transcript(tmp_path, manifest, text.encode("utf-8"))
    inputs = load_resume_inputs(manifest, base_dir=tmp_path)
    # The short retry query is still derived exactly as before; it carries the
    # leading year, so no numeric token was traded away for brevity.
    assert inputs.queries == ("1994 Flamengo campeão",)
    assert inputs.transcript_text == text


@pytest.mark.parametrize(
    "sentence",
    [
        "Flamengo Grêmio Maracanã 3 a 1",  # short score past the three short tokens
        "Flamengo Grêmio Maracanã 1994",  # late year past the three short tokens
        "Flamengo 3 x 1 Grêmio",  # explicit pair with score numbers
    ],
)
def test_derived_query_never_drops_sentence_numbers(tmp_path, manifest, sentence):
    manifest.pop("queries")
    manifest.pop("queries_sha256")
    _write_transcript(tmp_path, manifest, sentence.encode("utf-8"))
    queries = load_resume_inputs(manifest, base_dir=tmp_path).queries
    # The original full sentence is kept verbatim, whitespace included.
    assert queries == (sentence,)
    assert set(re.findall(r"\d+", sentence)) <= set(re.findall(r"\d+", queries[0]))


@pytest.mark.parametrize(
    "sentence,retry_result,expected",
    [
        ("Flamengo campeão em 1994", "Flamengo campeão", "Flamengo campeão em 1994"),
        ("1994 Flamengo campeão", "1994 Flamengo", "1994 Flamengo"),
        ("Flamengo Grêmio Maracanã", "Flamengo Grêmio", "Flamengo Grêmio"),
    ],
)
def test_retry_dropping_a_number_keeps_the_original_sentence(
    tmp_path, manifest, monkeypatch, sentence, retry_result, expected
):
    manifest.pop("queries")
    manifest.pop("queries_sha256")
    _write_transcript(tmp_path, manifest, sentence.encode("utf-8"))
    monkeypatch.setattr(
        "autoshorts.modules.resume_inputs.image_sources.retry_subject_query",
        lambda _sentence: retry_result,
    )
    assert load_resume_inputs(manifest, base_dir=tmp_path).queries == (expected,)


@pytest.mark.parametrize("storage", ["inline", "file"])
def test_explicit_saved_queries_with_numbers_remain_byte_exact(
    tmp_path, manifest, monkeypatch, storage
):
    values = ["Flamengo 3 x 1 Grêmio", "  1994 Flamengo campeão  ", "no numbers here"]
    if storage == "file":
        _file_list(tmp_path, manifest, "queries", _json_bytes(values))
    else:
        manifest["queries"] = values
        manifest["queries_sha256"] = _sha256(_json_bytes(values))

    def forbidden_retry(sentence):
        raise AssertionError("Saved queries must not be rederived")

    monkeypatch.setattr(
        "autoshorts.modules.resume_inputs.image_sources.retry_subject_query",
        forbidden_retry,
    )
    transcript_raw = (tmp_path / manifest["transcript_path"]).read_bytes()
    audio_raw = (tmp_path / manifest["audio_path"]).read_bytes()
    inputs = load_resume_inputs(manifest, base_dir=tmp_path)
    assert inputs.queries == tuple(values)
    assert inputs.transcript_text == "Recorded sentence."
    assert inputs.audio_sha256 == manifest["audio_sha256"]
    assert inputs.transcript_sha256 == manifest["transcript_sha256"]
    assert (tmp_path / manifest["transcript_path"]).read_bytes() == transcript_raw
    assert (tmp_path / manifest["audio_path"]).read_bytes() == audio_raw


def test_pixel_gate_rejects_wrong_number_for_retained_numeric_query(
    tmp_path, manifest, monkeypatch
):
    from autoshorts.modules import image_content

    manifest.pop("queries")
    manifest.pop("queries_sha256")
    text = "  1994 Flamengo campeão  "
    _write_transcript(tmp_path, manifest, text.encode("utf-8"))
    transcript_raw = (tmp_path / "transcript.txt").read_bytes()
    audio_raw = (tmp_path / "audio.wav").read_bytes()
    inputs = load_resume_inputs(manifest, base_dir=tmp_path)
    query = inputs.queries[0]
    assert query == "1994 Flamengo campeão"

    candidate = tmp_path / "candidate.png"
    candidate.write_bytes(b"synthetic-candidate-bytes")
    monkeypatch.setattr(
        "autoshorts.modules.image_content._ocr_words", lambda path: ["1994", "Flamengo"]
    )
    assert image_content.check_image_content(candidate, query) is None
    monkeypatch.setattr(
        "autoshorts.modules.image_content._ocr_words", lambda path: ["2017", "Flamengo"]
    )
    reason = image_content.check_image_content(candidate, query)
    assert reason is not None
    assert "2017" in reason

    assert inputs.transcript_text == text
    assert inputs.audio_sha256 == manifest["audio_sha256"]
    assert inputs.transcript_sha256 == manifest["transcript_sha256"]
    assert (tmp_path / "transcript.txt").read_bytes() == transcript_raw
    assert (tmp_path / "audio.wav").read_bytes() == audio_raw


def test_retry_errors_remain_fail_closed(tmp_path, manifest, monkeypatch):
    manifest.pop("queries")
    manifest.pop("queries_sha256")

    def retry(sentence):
        raise ResumeInputError("retry failed")

    monkeypatch.setattr(
        "autoshorts.modules.resume_inputs.image_sources.retry_subject_query", retry
    )
    with pytest.raises(ResumeInputError, match="retry failed"):
        load_resume_inputs(manifest, base_dir=tmp_path)


@pytest.mark.parametrize("field", ["image_source", "images"])
def test_ai_image_source_gate_remains_closed(tmp_path, manifest, field):
    manifest[field] = "ai"
    with pytest.raises(AISourceError):
        load_resume_inputs(manifest, base_dir=tmp_path)


def test_resume_inputs_immutability(tmp_path, manifest):
    inputs = load_resume_inputs(manifest, base_dir=tmp_path)
    with pytest.raises(FrozenInstanceError):
        inputs.audio_path = Path("/new/path")
