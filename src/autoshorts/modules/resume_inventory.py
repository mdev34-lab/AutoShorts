"""Durable, input-bound inventories of already acquired DDG images.

Hashes bind bytes, not photo truth. Existing pixel/subject and metadata-fit
checks are rerun; this module never acquires images or renders a video.
Remaining gap: fit is title metadata only, as in the searcher, so it cannot
prove what the pixels show beyond the unchanged ``_content_rejection`` gate.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path, PureWindowsPath

from . import image_sources
from .image_searcher import ImageSearcher
from .resume_inputs import ResumeInputError, load_resume_inputs

_INPUT_FIELDS = (
    "audio_sha256 transcript_sha256 queries paragraphs transcript_provenance"
)
_ITEM_FIELDS = "index prompt image_path image_sha256 sidecar_path sidecar_sha256"


class ResumeInventoryError(ValueError):
    """Incomplete, changed, disallowed or unsafe inventory assets."""


def _text(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ResumeInventoryError(f"{label} must be a nonempty string")
    return value


def _sha(value, label):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-fA-F]{64}", value) is None:
        raise ResumeInventoryError(f"{label} must be a 64-hex SHA256 string")
    return value.lower()


def _object(value, fields, label):
    if not isinstance(value, dict) or set(value) != set(fields.split()):
        raise ResumeInventoryError(f"Invalid {label} object fields")


def _strings(value, label):
    if not isinstance(value, list) or not value:
        raise ResumeInventoryError(f"{label} must be a nonempty list")
    for item in value:
        _text(item, label)


def _canonical(value):
    try:
        text = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        if json.loads(text) != value:
            raise ValueError("JSON serialization would coerce input values")
        return text.encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise ResumeInventoryError("Binding must contain lossless JSON values") from exc


def _invalid_constant(value):
    raise ResumeInventoryError(f"Invalid JSON constant: {value}")


def _json(raw, label):
    try:
        return json.loads(raw.decode("utf-8"), parse_constant=_invalid_constant)
    except (ValueError, UnicodeError) as exc:
        raise ResumeInventoryError(f"Malformed {label} JSON") from exc


def _binding(value):
    _object(value, _INPUT_FIELDS, "input binding")
    _strings(value["queries"], "queries")
    if value["paragraphs"] is not None:
        _strings(value["paragraphs"], "paragraphs")
    provenance = value["transcript_provenance"]
    if not (
        (isinstance(provenance, str) and provenance.strip())
        or (isinstance(provenance, dict) and provenance)
    ):
        raise ResumeInventoryError(
            "Explicit string/dict transcript provenance is required"
        )
    _canonical(provenance)
    return {
        **value,
        "audio_sha256": _sha(value["audio_sha256"], "audio_sha256"),
        "transcript_sha256": _sha(value["transcript_sha256"], "transcript_sha256"),
    }


def _fresh_binding(input_manifest):
    inputs = load_resume_inputs(input_manifest)
    return _binding(
        {
            "audio_sha256": inputs.audio_sha256,
            "transcript_sha256": inputs.transcript_sha256,
            "queries": list(inputs.queries),
            "paragraphs": None
            if inputs.paragraphs is None
            else list(inputs.paragraphs),
            "transcript_provenance": inputs.transcript_provenance,
        }
    )


def _location(value):
    if not isinstance(value, (str, Path)) or (
        isinstance(value, str) and not value.strip()
    ):
        raise ResumeInventoryError("Inventory path must be a nonempty path")
    path = Path(value)
    if path.name in ("", ".", ".."):
        raise ResumeInventoryError("Inventory must have a file name")
    try:
        parent = path.parent.resolve(strict=True)
        if not parent.is_dir() or (parent / path.name).is_symlink():
            raise ResumeInventoryError(
                "Inventory parent must be a directory; no inventory symlinks"
            )
        return parent / path.name
    except (OSError, ValueError, RuntimeError) as exc:
        raise ResumeInventoryError("Invalid inventory location") from exc


def _asset(root, name):
    _text(name, "asset path")
    relative = Path(name)
    if (
        relative.is_absolute()
        or PureWindowsPath(name).drive
        or "\\" in name
        or ".." in relative.parts
        or relative.as_posix() != name
    ):
        raise ResumeInventoryError("Asset path must be relative and without traversal")
    path = root / relative
    try:
        real = path.resolve(strict=True)
        if not real.is_relative_to(root) or not real.is_file():
            raise ResumeInventoryError(
                "Asset must be a regular file inside the inventory parent"
            )
        raw = path.read_bytes()
    except (OSError, ValueError, RuntimeError) as exc:
        raise ResumeInventoryError(f"Unsafe or missing asset: {name}") from exc
    if not raw:
        raise ResumeInventoryError(f"Empty asset: {name}")
    return path, raw, hashlib.sha256(raw).hexdigest()


def _image_name(root, value):
    if not isinstance(value, (str, Path)) or (
        isinstance(value, str) and not value.strip()
    ):
        raise ResumeInventoryError("Image path must be a nonempty path")
    path = Path(value)
    if not path.is_absolute():
        return path.as_posix()
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        pass
    try:
        # Resolve only the parent; a final-component symlink escape is caught by _asset.
        return (
            (path.parent.resolve(strict=True) / path.name).relative_to(root).as_posix()
        )
    except (OSError, ValueError, RuntimeError) as exc:
        raise ResumeInventoryError("Image is outside the inventory parent") from exc


def _sidecar_name(image_name):
    try:
        return ImageSearcher._sidecar_path(Path(image_name)).as_posix()
    except ValueError as exc:
        raise ResumeInventoryError(
            f"Image path has no sidecar location: {image_name}"
        ) from exc


def _record(image, raw, prompt, allow_unverified):
    record = _json(raw, "sidecar")
    if (
        not isinstance(record, dict)
        or type(record.get("schema")) is not int
        or record["schema"] != 1
    ):
        raise ResumeInventoryError("Sidecar schema must be exact integer 1")
    if record.get("kind") != "web" or record.get("provider") != "ddg":
        raise ResumeInventoryError("Only web/ddg provenance is allowed")
    accepted = ImageSearcher._read_sidecar(image)
    if accepted is None or _canonical(accepted) != _canonical(record):
        raise ResumeInventoryError("Missing, incomplete or changing sidecar provenance")
    if _text(record.get("prompt"), "sidecar prompt") != prompt:
        raise ResumeInventoryError("Sidecar prompt does not match its query")
    fit = _text(record.get("fit"), "sidecar fit")
    verified, opted_in = record.get("license_verified"), record.get("unverified_opt_in")
    if type(verified) is not bool or type(opted_in) is not bool:
        raise ResumeInventoryError("Rights flags must be exact booleans")
    if verified is True:
        _text(record.get("license"), "verified license")
        if opted_in is not False:
            raise ResumeInventoryError("Verified rights must not carry an opt-in")
    elif not (allow_unverified is True and opted_in is True):
        raise ResumeInventoryError(
            "Unknown rights require explicit current and recorded opt-in"
        )
    title = _text(record.get("title"), "sidecar title")
    if fit.startswith("short-query:"):
        short = image_sources.retry_subject_query(prompt)
        if (
            allow_unverified is not True
            or opted_in is not True
            or not short
            or short == prompt
            or image_sources.has_contradictory_number(prompt, title)
        ):
            raise ResumeInventoryError("Disallowed short-query fit")
        evidence = image_sources.fit_evidence(short, title)
        expected = f"short-query:'{short}' via {evidence}"
    else:
        evidence = image_sources.fit_evidence(prompt, title)
        expected = evidence
    if not isinstance(evidence, str) or not evidence.strip() or fit != expected:
        raise ResumeInventoryError(
            "Metadata fit no longer matches the recorded evidence"
        )


def _inspect(root, index, prompt, image_name, allow_unverified, expected=None):
    image, _, image_sha = _asset(root, image_name)
    sidecar_name = _sidecar_name(image_name)
    _, sidecar_raw, sidecar_sha = _asset(root, sidecar_name)
    if expected is not None and (
        image_sha != expected["image_sha256"]
        or sidecar_sha != expected["sidecar_sha256"]
    ):
        raise ResumeInventoryError("Image or sidecar raw SHA256 mismatch")
    _record(image, sidecar_raw, prompt, allow_unverified)
    try:
        rejection = ImageSearcher._content_rejection(image, prompt)
    except Exception as exc:
        raise ResumeInventoryError("Pixel/subject check unavailable") from exc
    if rejection is not None:
        raise ResumeInventoryError(f"Pixel/subject check rejected image: {rejection}")
    return {
        "index": index,
        "prompt": prompt,
        "image_path": image_name,
        "image_sha256": image_sha,
        "sidecar_path": sidecar_name,
        "sidecar_sha256": sidecar_sha,
    }


def _items(value, queries):
    if not isinstance(value, list) or not value or len(value) != len(queries):
        raise ResumeInventoryError("A nonempty item list must match every query")
    result = []
    for index, item in enumerate(value):
        _object(item, _ITEM_FIELDS, "inventory item")
        if type(item["index"]) is not int or item["index"] != index:
            raise ResumeInventoryError("Item indices must be exact ordered integers")
        if _text(item["prompt"], "item prompt") != queries[index]:
            raise ResumeInventoryError("Item prompt does not match its query")
        _text(item["image_path"], "image path")
        _text(item["sidecar_path"], "sidecar path")
        if item["sidecar_path"] != _sidecar_name(item["image_path"]):
            raise ResumeInventoryError(
                "Sidecar must use the image's .credits.json location"
            )
        result.append(
            {
                **item,
                "image_sha256": _sha(item["image_sha256"], "image_sha256"),
                "sidecar_sha256": _sha(item["sidecar_sha256"], "sidecar_sha256"),
            }
        )
    return result


def _recheck(input_manifest, root, binding, items):
    paths = []
    for item in items:
        for field in ("image", "sidecar"):
            path, _, digest = _asset(root, item[f"{field}_path"])
            if digest != item[f"{field}_sha256"]:
                raise ResumeInventoryError(
                    "Asset changed during inventory preparation/validation"
                )
            if field == "image":
                paths.append(path)
    if _canonical(_fresh_binding(input_manifest)) != _canonical(binding):
        raise ResumeInventoryError(
            "Inputs changed during inventory preparation/validation"
        )
    return paths


def _sync_directory(parent):
    descriptor = os.open(parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish(destination, raw, recheck):
    temporary, published = None, None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent, prefix=".inventory-", delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        recheck()
        published = temporary.stat()
        # Unlike replace/rename, link atomically refuses even a racing destination.
        os.link(temporary, destination)
        temporary.unlink()
        temporary = None
        _sync_directory(destination.parent)
        recheck()
    except Exception:
        if published is not None:
            try:
                current = destination.lstat()
                if (current.st_dev, current.st_ino) == (
                    published.st_dev,
                    published.st_ino,
                ):
                    destination.unlink()
                    _sync_directory(destination.parent)
            except FileNotFoundError:
                pass
        raise
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def write_inventory(
    input_manifest, image_paths, inventory_path, *, allow_unverified=False
):
    """Validate and exclusively publish an inventory; relative images use its parent."""
    binding = _fresh_binding(input_manifest)
    if type(allow_unverified) is not bool:
        raise ResumeInventoryError("allow_unverified must be an exact boolean")
    destination = _location(inventory_path)
    if destination.exists():
        raise ResumeInventoryError("Inventory destination already exists")
    if (
        not isinstance(image_paths, list)
        or not image_paths
        or len(image_paths) != len(binding["queries"])
    ):
        raise ResumeInventoryError(
            "Image paths must be a nonempty list matching every query"
        )
    items = [
        _inspect(
            destination.parent,
            index,
            prompt,
            _image_name(destination.parent, image),
            allow_unverified,
        )
        for index, (prompt, image) in enumerate(
            zip(binding["queries"], image_paths, strict=True)
        )
    ]
    raw = _canonical({"schema": 1, "inputs": binding, "items": items}) + b"\n"
    try:
        _publish(
            destination,
            raw,
            lambda: _recheck(input_manifest, destination.parent, binding, items),
        )
    except OSError as exc:
        if isinstance(exc, ResumeInputError):
            raise
        raise ResumeInventoryError(
            "Could not exclusively publish durable inventory"
        ) from exc


def validate_inventory(
    input_manifest, inventory_path, *, allow_unverified=False
) -> list[Path]:
    """Revalidate fresh input binding, provenance, fit, pixels and final raw bytes."""
    binding = _fresh_binding(input_manifest)
    if type(allow_unverified) is not bool:
        raise ResumeInventoryError("allow_unverified must be an exact boolean")
    inventory = _location(inventory_path)
    _, raw, _ = _asset(inventory.parent, inventory.name)
    data = _json(raw, "inventory")
    _object(data, "schema inputs items", "inventory")
    if type(data["schema"]) is not int or data["schema"] != 1:
        raise ResumeInventoryError("Inventory schema must be exact integer 1")
    if _canonical(_binding(data["inputs"])) != _canonical(binding):
        raise ResumeInventoryError("Inventory is bound to different inputs")
    items = _items(data["items"], binding["queries"])
    for item in items:
        _inspect(
            inventory.parent,
            item["index"],
            item["prompt"],
            item["image_path"],
            allow_unverified,
            expected=item,
        )
    return _recheck(input_manifest, inventory.parent, binding, items)
