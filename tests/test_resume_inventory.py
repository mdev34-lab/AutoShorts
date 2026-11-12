"""Offline fake-only coverage for durable resume image inventories."""

from __future__ import annotations

import hashlib
import json
import os
import socket
from pathlib import Path
from types import SimpleNamespace

import pytest
import requests

from autoshorts.modules import image_sources, resume_inventory
from autoshorts.modules.image_searcher import ImageSearcher
from autoshorts.modules.resume_inputs import ResumeInputError, load_resume_inputs
from autoshorts.modules.resume_inventory import (
    ResumeInventoryError,
    validate_inventory,
    write_inventory,
)

QUERIES = ["Flamengo Grêmio", "Maracanã Rio"]
SHORT_QUERY = "Flamengo Grêmio Maracanã 2026"
_DROP = object()


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _jb(value) -> bytes:
    return json.dumps(value, separators=(",", ":")).encode("utf-8")


def _write_manifest(
    root,
    queries=QUERIES,
    *,
    audio=b"audio-bytes",
    transcript=b"Recorded sentence.",
    provenance="field-recording",
    paragraphs=None,
):
    (root / "audio.wav").write_bytes(audio)
    (root / "transcript.txt").write_bytes(transcript)
    manifest = {
        "schema": 1,
        "audio_path": "audio.wav",
        "audio_sha256": _sha(audio),
        "transcript_path": "transcript.txt",
        "transcript_sha256": _sha(transcript),
        "transcript_provenance": provenance,
        "queries": list(queries),
        "queries_sha256": _sha(_jb(list(queries))),
    }
    if paragraphs is not None:
        manifest["paragraphs"] = list(paragraphs)
        manifest["paragraphs_sha256"] = _sha(_jb(list(paragraphs)))
    path = root / "manifest.json"
    path.write_bytes(_jb(manifest))
    return path


def _record(prompt, title=None, /, **changes):
    # Positional-only so tests can override the "prompt" and "title" fields.
    title = prompt if title is None else title
    record = {
        "schema": 1,
        "kind": "web",
        "provider": "ddg",
        "prompt": prompt,
        "title": title,
        "image_url": "https://photos.example.invalid/a.jpg",
        "source_page": "https://pages.example.invalid/a",
        "creator": "",
        "license": "",
        "license_url": "",
        "license_verified": False,
        "fit": image_sources.fit_evidence(prompt, title),
        "pictured_subject_check": "not-run (ALLOW_UNVERIFIED_IMAGES opt-in)",
        "unverified_opt_in": True,
        "modification": "Resized and cropped to the video frame.",
    }
    for key, value in changes.items():
        if value is _DROP:
            record.pop(key, None)
        else:
            record[key] = value
    return record


def _place(root, index, record, raw=None):
    image = root / f"img{index}.jpg"
    image.write_bytes(raw if raw is not None else f"image-{index}".encode())
    sidecar = image.with_suffix(".credits.json")
    sidecar.write_bytes(
        json.dumps(record, ensure_ascii=False, indent=2).encode("utf-8")
    )
    return image


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("network access attempted in offline inventory test")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(requests.Session, "request", refuse)


@pytest.fixture
def gate(monkeypatch):
    state = SimpleNamespace(calls=[], verdicts={}, hook=None)

    def fake(path, prompt):
        state.calls.append((Path(path).name, prompt))
        if state.hook is not None:
            state.hook(Path(path), prompt)
        return state.verdicts.get(Path(path).name)

    monkeypatch.setattr(ImageSearcher, "_content_rejection", staticmethod(fake))
    return state


@pytest.fixture
def project(tmp_path, gate):
    manifest = _write_manifest(tmp_path)
    for index, query in enumerate(QUERIES):
        _place(tmp_path, index, _record(query))
    return SimpleNamespace(
        root=tmp_path,
        manifest=manifest,
        images=["img0.jpg", "img1.jpg"],
        inventory=tmp_path / "inventory.json",
        gate=gate,
    )


def _write(p, images=None, *, allow_unverified=True):
    write_inventory(
        p.manifest,
        p.images if images is None else images,
        p.inventory,
        allow_unverified=allow_unverified,
    )


def _validate(p, *, allow_unverified=True):
    return validate_inventory(
        p.manifest, p.inventory, allow_unverified=allow_unverified
    )


def _no_inventory(p):
    assert not p.inventory.exists()
    assert list(p.root.glob(".inventory-*")) == []


def _mutate(p, change):
    data = json.loads(p.inventory.read_text("utf-8"))
    replaced = change(data)
    p.inventory.write_bytes(
        json.dumps(data if replaced is None else replaced).encode("utf-8")
    )


def _set(*path_and_value):
    *path, value = path_and_value

    def apply(data):
        target = data
        for key in path[:-1]:
            target = target[key]
        if value is _DROP:
            del target[path[-1]]
        else:
            target[path[-1]] = value

    return apply


def _image_path(value):
    def apply(data):
        data["items"][0]["image_path"] = value
        data["items"][0]["sidecar_path"] = (
            Path(value).with_suffix(".credits.json").as_posix()
        )

    return apply


def _swap(*fields):
    def apply(data):
        first, second = data["items"]
        for field in fields:
            first[field], second[field] = second[field], first[field]

    return apply


def _rewrite_manifest(p, **kwargs):
    p.manifest = _write_manifest(p.root, **kwargs)


# --- successful roundtrip -------------------------------------------------


def test_roundtrip_binds_fresh_inputs_prompts_images_and_sidecars(project):
    p = project
    sidecars = [(p.root / f"img{i}.credits.json").read_bytes() for i in range(2)]
    _write(p)

    raw = p.inventory.read_bytes()
    data = json.loads(raw)
    assert type(data["schema"]) is int and data["schema"] == 1
    assert data["inputs"] == {
        "audio_sha256": _sha(b"audio-bytes"),
        "transcript_sha256": _sha(b"Recorded sentence."),
        "queries": QUERIES,
        "paragraphs": None,
        "transcript_provenance": "field-recording",
    }
    assert data["items"] == [
        {
            "index": i,
            "prompt": QUERIES[i],
            "image_path": f"img{i}.jpg",
            "image_sha256": _sha(f"image-{i}".encode()),
            "sidecar_path": f"img{i}.credits.json",
            "sidecar_sha256": _sha(sidecars[i]),
        }
        for i in range(2)
    ]
    assert p.gate.calls == [("img0.jpg", QUERIES[0]), ("img1.jpg", QUERIES[1])]
    assert list(p.root.glob(".inventory-*")) == []

    p.gate.calls.clear()
    assert _validate(p) == [
        p.root.resolve() / "img0.jpg",
        p.root.resolve() / "img1.jpg",
    ]
    assert p.gate.calls == [("img0.jpg", QUERIES[0]), ("img1.jpg", QUERIES[1])]
    assert p.inventory.read_bytes() == raw
    assert [
        (p.root / f"img{i}.credits.json").read_bytes() for i in range(2)
    ] == sidecars


def test_unknown_empty_license_is_preserved_and_not_rewritten(project):
    p = project
    before = (p.root / "img0.credits.json").read_bytes()
    _write(p)
    _validate(p)
    assert (p.root / "img0.credits.json").read_bytes() == before
    record = json.loads(before)
    assert record["license"] == ""
    assert record["license_verified"] is False
    assert record["unverified_opt_in"] is True


def test_absolute_image_inside_parent_is_stored_relative(project):
    p = project
    _write(p, [str(p.root / "img0.jpg"), p.root / "img1.jpg"])
    items = json.loads(p.inventory.read_text("utf-8"))["items"]
    assert [item["image_path"] for item in items] == ["img0.jpg", "img1.jpg"]
    _validate(p)


def test_paragraphs_and_dict_provenance_are_bound(project):
    p = project
    _rewrite_manifest(
        p, paragraphs=["First paragraph."], provenance={"provider": "piper-tts"}
    )
    _write(p)
    inputs = json.loads(p.inventory.read_text("utf-8"))["inputs"]
    assert inputs["paragraphs"] == ["First paragraph."]
    assert inputs["transcript_provenance"] == {"provider": "piper-tts"}
    _validate(p)


# --- writer refusals ------------------------------------------------------


def test_writer_refuses_existing_destination_without_overwrite(project):
    p = project
    _write(p)
    raw = p.inventory.read_bytes()
    with pytest.raises(ResumeInventoryError, match="already exists"):
        _write(p)
    assert p.inventory.read_bytes() == raw

    other = SimpleNamespace(**{**vars(p), "inventory": p.root / "other.json"})
    other.inventory.write_bytes(b"unrelated")
    with pytest.raises(ResumeInventoryError, match="already exists"):
        _write(other)
    assert other.inventory.read_bytes() == b"unrelated"


def test_writer_refuses_dangling_symlink_destination(project):
    p = project
    try:
        os.symlink(p.root / "missing-target.json", p.inventory)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable")
    with pytest.raises(ResumeInventoryError):
        _write(p)
    assert not (p.root / "missing-target.json").exists()


@pytest.mark.parametrize(
    "images",
    [
        [],
        ["img0.jpg"],
        ["img0.jpg", "img1.jpg", "img1.jpg"],
        ("img0.jpg", "img1.jpg"),
        None,
        "img0.jpg",
        ["img0.jpg", 5],
        ["img0.jpg", ""],
        ["img0.jpg", "   "],
    ],
)
def test_writer_requires_list_matching_every_query(project, images):
    p = project
    with pytest.raises(ResumeInventoryError):
        write_inventory(p.manifest, images, p.inventory, allow_unverified=True)
    _no_inventory(p)


@pytest.mark.parametrize("images", [["img1.jpg", "img0.jpg"], ["img0.jpg", "img0.jpg"]])
def test_writer_rejects_swapped_or_reused_images_by_sidecar_prompt(project, images):
    p = project
    with pytest.raises(ResumeInventoryError, match="prompt"):
        _write(p, images)
    _no_inventory(p)


def test_writer_rejects_images_outside_inventory_parent(project):
    p = project
    nested = p.root / "nested"
    nested.mkdir()
    q = SimpleNamespace(**{**vars(p), "inventory": nested / "inventory.json"})
    for images in (
        ["../img0.jpg", "../img1.jpg"],
        [str(p.root / "img0.jpg"), str(p.root / "img1.jpg")],
    ):
        with pytest.raises(ResumeInventoryError):
            _write(q, images)
        assert not q.inventory.exists()
        assert list(nested.glob(".inventory-*")) == []


def test_writer_rejects_symlink_escape(project, tmp_path_factory):
    p = project
    outside = tmp_path_factory.mktemp("outside") / "img0.jpg"
    outside.write_bytes(b"image-0")
    (p.root / "img0.jpg").unlink()
    try:
        os.symlink(outside, p.root / "img0.jpg")
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable")
    with pytest.raises(ResumeInventoryError, match="Unsafe|inside"):
        _write(p)
    _no_inventory(p)


@pytest.mark.parametrize("damage", ["missing", "empty", "directory"])
def test_writer_requires_nonempty_regular_image(project, damage):
    p = project
    image = p.root / "img1.jpg"
    image.unlink()
    if damage == "empty":
        image.write_bytes(b"")
    elif damage == "directory":
        image.mkdir()
    with pytest.raises(ResumeInventoryError):
        _write(p)
    _no_inventory(p)


def test_caller_created_inputs_are_not_a_validation_bypass(project):
    p = project
    inputs = load_resume_inputs(p.manifest)
    with pytest.raises(ResumeInputError):
        write_inventory(inputs, p.images, p.inventory, allow_unverified=True)
    _no_inventory(p)
    _write(p)
    with pytest.raises(ResumeInputError):
        validate_inventory(inputs, p.inventory, allow_unverified=True)


def test_tampered_input_assets_fail_before_writing(project):
    p = project
    (p.root / "audio.wav").write_bytes(b"changed-audio")
    with pytest.raises(ResumeInputError):
        _write(p)
    _no_inventory(p)


# --- pixel gate -----------------------------------------------------------


def test_pixel_gate_rejection_leaves_no_inventory(project):
    p = project
    p.gate.verdicts["img1.jpg"] = "image is mostly text"
    with pytest.raises(ResumeInventoryError, match="mostly text"):
        _write(p)
    _no_inventory(p)


def test_pixel_gate_exception_fails_closed(project):
    p = project

    def unavailable(path, prompt):
        raise RuntimeError("checker unavailable")

    p.gate.hook = unavailable
    with pytest.raises(ResumeInventoryError, match="unavailable"):
        _write(p)
    _no_inventory(p)


def test_pixel_gate_is_rerun_on_validation(project):
    p = project
    _write(p)
    p.gate.verdicts["img0.jpg"] = "pictured subject could not be verified"
    with pytest.raises(ResumeInventoryError, match="subject"):
        _validate(p)


# --- rights opt-in --------------------------------------------------------


def test_unknown_rights_require_current_opt_in(project):
    p = project
    with pytest.raises(ResumeInventoryError, match="opt-in"):
        _write(p, allow_unverified=False)
    _no_inventory(p)
    _write(p)
    with pytest.raises(ResumeInventoryError, match="opt-in"):
        _validate(p, allow_unverified=False)


@pytest.mark.parametrize("flag", [1, "true", None, 1.0])
def test_allow_unverified_must_be_exact_bool(project, flag):
    p = project
    with pytest.raises(ResumeInventoryError):
        write_inventory(p.manifest, p.images, p.inventory, allow_unverified=flag)
    _no_inventory(p)
    _write(p)
    with pytest.raises(ResumeInventoryError):
        validate_inventory(p.manifest, p.inventory, allow_unverified=flag)


def test_unknown_rights_require_recorded_opt_in(project):
    p = project
    _place(p.root, 1, _record(QUERIES[1], unverified_opt_in=False))
    with pytest.raises(ResumeInventoryError):
        _write(p)
    _no_inventory(p)


def test_verified_rights_accept_without_opt_in(project):
    p = project
    for index, query in enumerate(QUERIES):
        _place(
            p.root,
            index,
            _record(
                query,
                license="CC BY 4.0",
                license_verified=True,
                unverified_opt_in=False,
            ),
        )
    _write(p, allow_unverified=False)
    assert len(_validate(p, allow_unverified=False)) == 2
    assert len(_validate(p, allow_unverified=True)) == 2


@pytest.mark.parametrize(
    "changes",
    [
        {"license": "CC BY 4.0", "license_verified": True, "unverified_opt_in": True},
        {"license": "", "license_verified": True, "unverified_opt_in": False},
        {"license": "   ", "license_verified": True, "unverified_opt_in": False},
    ],
)
def test_verified_rights_need_license_and_no_opt_in(project, changes):
    p = project
    _place(p.root, 0, _record(QUERIES[0], **changes))
    for allow in (False, True):
        with pytest.raises(ResumeInventoryError):
            _write(p, allow_unverified=allow)
        _no_inventory(p)


# --- sidecar provenance ---------------------------------------------------


@pytest.mark.parametrize(
    "changes",
    [
        {"schema": True},
        {"schema": "1"},
        {"schema": 1.0},
        {"schema": 2},
        {"schema": _DROP},
        {"kind": "ai"},
        {"kind": _DROP},
        {"provider": "commons"},
        {"provider": "ai"},
        {"provider": _DROP},
        {"prompt": "Maracanã Rio"},
        {"prompt": _DROP},
        {"fit": ""},
        {"fit": "flamengo"},
        {"fit": _DROP},
        {"title": "Unrelated photo"},
        {"title": "   "},
        {"license_verified": "false"},
        {"license_verified": None},
        {"license_verified": _DROP},
        {"unverified_opt_in": 1},
        {"unverified_opt_in": "true"},
        {"unverified_opt_in": _DROP},
        {"license": 0},
        {"license": "   "},
        {"license": _DROP},
        {"source_page": ""},
        {"source_page": "ftp://pages.example.invalid/a"},
        {"image_url": _DROP},
    ],
)
def test_sidecar_record_must_be_valid_web_ddg_provenance(project, changes):
    p = project
    _place(p.root, 0, _record(QUERIES[0], **changes))
    with pytest.raises(ResumeInventoryError):
        _write(p)
    _no_inventory(p)


@pytest.mark.parametrize(
    "raw", [None, b"", b"{not json", b'{"schema": NaN}', b"\xff\xfe", b"[]"]
)
def test_missing_or_malformed_sidecar_is_rejected(project, raw):
    p = project
    sidecar = p.root / "img0.credits.json"
    sidecar.unlink()
    if raw is not None:
        sidecar.write_bytes(raw)
    with pytest.raises(ResumeInventoryError):
        _write(p)
    _no_inventory(p)


def test_ai_sidecar_is_rejected_even_with_full_fields(project):
    p = project
    _place(
        p.root,
        0,
        _record(
            QUERIES[0],
            kind="ai",
            provider="ai",
            license="AI-generated - no source license",
        ),
    )
    with pytest.raises(ResumeInventoryError, match="web/ddg"):
        _write(p)
    _no_inventory(p)


# --- short-query metadata fit --------------------------------------------


@pytest.fixture
def short_project(tmp_path, gate):
    short = image_sources.retry_subject_query(SHORT_QUERY)
    assert short and short != SHORT_QUERY
    manifest = _write_manifest(tmp_path, [SHORT_QUERY])
    return SimpleNamespace(
        root=tmp_path,
        manifest=manifest,
        images=["img0.jpg"],
        inventory=tmp_path / "inventory.json",
        gate=gate,
        short=short,
    )


def _short_record(p, title=None, short=None, **changes):
    title = p.short if title is None else title
    short = p.short if short is None else short
    fit = f"short-query:'{short}' via {image_sources.fit_evidence(short, title)}"
    return _record(SHORT_QUERY, title, fit=fit, **changes)


def test_full_query_fit_alone_does_not_cover_short_query_case(short_project):
    p = short_project
    assert image_sources.fit_evidence(SHORT_QUERY, p.short) is None


def test_recorded_short_query_fit_is_rechecked_and_accepted(short_project):
    p = short_project
    _place(p.root, 0, _short_record(p))
    _write(p)
    assert len(_validate(p)) == 1


def test_short_query_rejects_contradictory_title_number(short_project):
    p = short_project
    title = f"{p.short} 2019"
    assert image_sources.has_contradictory_number(SHORT_QUERY, title)
    _place(p.root, 0, _short_record(p, title=title))
    with pytest.raises(ResumeInventoryError, match="short-query"):
        _write(p)
    _no_inventory(p)


def test_short_query_must_be_current_retry_subject_query(short_project):
    p = short_project
    _place(
        p.root, 0, _short_record(p, title="Flamengo Grêmio", short="Flamengo Grêmio")
    )
    with pytest.raises(ResumeInventoryError, match="fit"):
        _write(p)
    _no_inventory(p)


def test_short_query_fit_requires_unverified_opt_in(short_project):
    p = short_project
    _place(
        p.root,
        0,
        _short_record(
            p, license="CC BY 4.0", license_verified=True, unverified_opt_in=False
        ),
    )
    for allow in (False, True):
        with pytest.raises(ResumeInventoryError):
            _write(p, allow_unverified=allow)
        _no_inventory(p)


# --- validation of stored inventory shape ---------------------------------


def _swap_prompts(data):
    first, second = data["items"]
    first["prompt"], second["prompt"] = second["prompt"], first["prompt"]


@pytest.mark.parametrize(
    "change",
    [
        _set("schema", True),
        _set("schema", "1"),
        _set("schema", 1.0),
        _set("schema", 2),
        _set("schema", _DROP),
        _set("extra", 1),
        _set("items", _DROP),
        _set("items", []),
        _set("items", {}),
        lambda d: d.update(items=d["items"][:1]),
        lambda d: d["items"].append(dict(d["items"][0], index=2)),
        lambda d: d["items"].reverse(),
        _set("items", 0, "index", False),
        _set("items", 1, "index", True),
        _set("items", 0, "index", "0"),
        _set("items", 0, "index", 0.0),
        _set("items", 0, "index", 1),
        _swap_prompts,
        _swap("image_path", "image_sha256", "sidecar_path", "sidecar_sha256"),
        _swap("image_sha256"),
        _swap("sidecar_sha256"),
        _set("items", 0, "prompt", ""),
        _set("items", 0, "prompt", None),
        _set("items", 0, "image_sha256", "0" * 64),
        _set("items", 0, "image_sha256", "g" * 64),
        _set("items", 0, "image_sha256", "ab"),
        _set("items", 0, "image_sha256", 123),
        _set("items", 0, "sidecar_sha256", "0" * 64),
        _set("items", 0, "sidecar_sha256", None),
        _set("items", 0, "sidecar_path", "img1.credits.json"),
        _set("items", 0, "sidecar_path", ""),
        _set("items", 0, "image_path", None),
        _set("items", 0, "extra", 1),
        _set("items", 0, "sidecar_sha256", _DROP),
        _image_path("/img0.jpg"),
        _image_path("../img0.jpg"),
        _image_path("./img0.jpg"),
        _image_path("sub\\img0.jpg"),
        _image_path("C:/img0.jpg"),
        _image_path("img0.jpg/."),
        _set("inputs", _DROP),
        _set("inputs", "audio_sha256", "0" * 64),
        _set("inputs", "audio_sha256", "short"),
        _set("inputs", "transcript_sha256", "0" * 64),
        _set("inputs", "queries", list(reversed(QUERIES))),
        _set("inputs", "queries", []),
        _set("inputs", "paragraphs", ["Injected paragraph."]),
        _set("inputs", "paragraphs", []),
        _set("inputs", "transcript_provenance", "other-provenance"),
        _set("inputs", "transcript_provenance", ""),
        _set("inputs", "identity_sha256", "0" * 64),
        lambda d: [d],
        lambda d: {},
    ],
)
def test_malformed_or_rebound_inventory_is_rejected(project, change):
    p = project
    _write(p)
    _mutate(p, change)
    with pytest.raises(ResumeInventoryError):
        _validate(p)


@pytest.mark.parametrize(
    "raw", [b"", b"{not json", b'{"schema": NaN}', b"\xff\xfe", b"null"]
)
def test_unparseable_inventory_is_rejected(project, raw):
    p = project
    _write(p)
    p.inventory.write_bytes(raw)
    with pytest.raises(ResumeInventoryError):
        _validate(p)


def test_identity_digest_cannot_replace_bound_fields(project):
    p = project
    _write(p)

    def replace(data):
        digest = _sha(json.dumps(data["inputs"], sort_keys=True).encode("utf-8"))
        data["inputs"] = {"identity_sha256": digest}

    _mutate(p, replace)
    with pytest.raises(ResumeInventoryError):
        _validate(p)


def test_missing_or_symlinked_inventory_is_rejected(project):
    p = project
    with pytest.raises(ResumeInventoryError):
        _validate(p)
    _write(p)
    alias = p.root / "alias.json"
    try:
        os.symlink(p.inventory, alias)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable")
    with pytest.raises(ResumeInventoryError):
        validate_inventory(p.manifest, alias, allow_unverified=True)


# --- changed assets after writing -----------------------------------------


def test_changed_image_bytes_are_rejected(project):
    p = project
    _write(p)
    (p.root / "img1.jpg").write_bytes(b"different-image")
    with pytest.raises(ResumeInventoryError, match="SHA256"):
        _validate(p)


def test_reserialized_sidecar_is_rejected_by_raw_hash(project):
    p = project
    _write(p)
    sidecar = p.root / "img0.credits.json"
    sidecar.write_bytes(_jb(json.loads(sidecar.read_bytes())))
    with pytest.raises(ResumeInventoryError, match="SHA256"):
        _validate(p)


@pytest.mark.parametrize("name", ["img0.jpg", "img0.credits.json"])
def test_deleted_asset_is_rejected(project, name):
    p = project
    _write(p)
    (p.root / name).unlink()
    with pytest.raises(ResumeInventoryError):
        _validate(p)


def test_symlink_escape_with_identical_bytes_is_rejected(project, tmp_path_factory):
    p = project
    _write(p)
    outside = tmp_path_factory.mktemp("outside") / "img0.jpg"
    outside.write_bytes(b"image-0")
    (p.root / "img0.jpg").unlink()
    try:
        os.symlink(outside, p.root / "img0.jpg")
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable")
    with pytest.raises(ResumeInventoryError, match="Unsafe|inside"):
        _validate(p)


@pytest.mark.parametrize("name", ["audio.wav", "transcript.txt"])
def test_raw_input_tamper_is_rejected(project, name):
    p = project
    _write(p)
    (p.root / name).write_bytes(b"tampered")
    with pytest.raises(ResumeInputError):
        _validate(p)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"audio": b"replacement-audio"},
        {"transcript": b"Replacement sentence."},
        {"queries": list(reversed(QUERIES))},
        {"provenance": "other-provenance"},
        {"paragraphs": ["New paragraph."]},
    ],
)
def test_consistently_rehashed_new_inputs_do_not_match_inventory(project, kwargs):
    p = project
    _write(p)
    _rewrite_manifest(p, **kwargs)
    with pytest.raises(ResumeInventoryError, match="different inputs"):
        _validate(p)


# --- final re-read before returning ---------------------------------------


def test_writer_rechecks_images_before_publishing(project):
    p = project

    def mutate(path, prompt):
        if path.name == "img1.jpg":
            (p.root / "img0.jpg").write_bytes(b"swapped-after-hash")

    p.gate.hook = mutate
    with pytest.raises(ResumeInventoryError, match="changed"):
        _write(p)
    _no_inventory(p)


def test_writer_rechecks_inputs_before_publishing(project):
    p = project

    def mutate(path, prompt):
        if path.name == "img1.jpg":
            (p.root / "transcript.txt").write_bytes(b"changed transcript")

    p.gate.hook = mutate
    with pytest.raises(ResumeInputError):
        _write(p)
    _no_inventory(p)


def test_validator_rereads_assets_immediately_before_returning(project):
    p = project
    _write(p)

    def mutate(path, prompt):
        if path.name == "img1.jpg":
            (p.root / "img0.credits.json").write_bytes(b"{}")

    p.gate.hook = mutate
    with pytest.raises(ResumeInventoryError, match="changed"):
        _validate(p)


def test_validator_rereads_inputs_immediately_before_returning(project):
    p = project
    _write(p)

    def mutate(path, prompt):
        if path.name == "img1.jpg":
            (p.root / "audio.wav").write_bytes(b"changed-audio")

    p.gate.hook = mutate
    with pytest.raises(ResumeInputError):
        _validate(p)


def test_failed_post_publish_recheck_removes_inventory(project, monkeypatch):
    p = project
    calls = []
    original = resume_inventory._recheck

    def flaky(*args):
        calls.append(1)
        if len(calls) == 2:
            raise ResumeInventoryError(
                "Asset changed during inventory preparation/validation"
            )
        return original(*args)

    monkeypatch.setattr(resume_inventory, "_recheck", flaky)
    with pytest.raises(ResumeInventoryError, match="changed"):
        _write(p)
    assert len(calls) == 2
    _no_inventory(p)


# --- scope ----------------------------------------------------------------


def test_module_has_no_acquisition_render_or_process_paths():
    source = Path(resume_inventory.__file__).read_text("utf-8")
    for forbidden in (
        "import requests",
        "subprocess",
        "DDGS",
        "search_images",
        "download_image",
        "get_images",
        "_fallback_to_ai",
        "ScriptGenerator",
        "resume_ddg",
    ):
        assert forbidden not in source
