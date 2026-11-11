"""Fail-closed loading of existing assets from schema 1 resume manifests."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeGuard

from . import image_sources


class ResumeInputError(ValueError):
    """Base error for all resume input failures."""


class ManifestSchemaError(ResumeInputError):
    """A manifest or asset has an unsupported or invalid shape."""


class ManifestMissingError(ResumeInputError, FileNotFoundError):
    """A required manifest field or nonempty asset is missing."""


class ManifestTamperError(ResumeInputError):
    """A raw-byte SHA256 checksum does not match."""


class AISourceError(ResumeInputError):
    """AI image sourcing is prohibited."""


@dataclass(frozen=True)
class ResumeInputs:
    """Validated inputs; transcript and saved strings are not rewritten."""

    audio_path: Path
    transcript_path: Path
    transcript_text: str
    transcript_provenance: Any
    paragraphs: tuple[str, ...] | None
    queries: tuple[str, ...]
    audio_sha256: str
    transcript_sha256: str


def _nonempty_string(value: Any) -> TypeGuard[str]:
    return isinstance(value, str) and bool(value.strip())


def _required_string(data: dict, field: str) -> str:
    if field not in data:
        raise ManifestMissingError(f"Missing required manifest field: {field}")
    value = data[field]
    if not _nonempty_string(value):
        raise ManifestSchemaError(f"{field} must be a nonempty string")
    return value


def _required_sha256(data: dict, field: str) -> str:
    value = _required_string(data, field)
    if re.fullmatch(r"[0-9a-fA-F]{64}", value) is None:
        raise ManifestSchemaError(
            f"{field} must be a 64-character hexadecimal SHA256 string"
        )
    return value.lower()


def _check_hash(raw: bytes, expected: str, label: str) -> None:
    actual = hashlib.sha256(raw).hexdigest()
    if actual != expected:
        raise ManifestTamperError(
            f"{label} SHA256 mismatch: expected {expected}, got {actual}"
        )


def _verified_bytes(path: Path, expected: str, label: str) -> bytes:
    if not path.is_file():
        raise ManifestMissingError(f"{label} file not found: {path}")
    raw = path.read_bytes()
    if not raw:
        raise ManifestMissingError(f"{label} file is empty: {path}")
    _check_hash(raw, expected, label)
    return raw


def _string_list(value: Any, field: str) -> tuple[str, ...]:
    if (
        not isinstance(value, list)
        or not value
        or any(not _nonempty_string(item) for item in value)
    ):
        raise ManifestSchemaError(
            f"{field} must be a nonempty list of nonempty strings"
        )
    return tuple(value)


def _saved_strings(
    data: dict, field: str, manifest_dir: Path
) -> tuple[str, ...] | None:
    path_field = f"{field}_path"
    hash_field = f"{field}_sha256"
    if field not in data and path_field not in data:
        if hash_field in data:
            raise ManifestSchemaError(f"{hash_field} requires {field} or {path_field}")
        return None
    if field in data and path_field in data:
        raise ManifestSchemaError(f"Specify either {field} or {path_field}, not both")
    expected = _required_sha256(data, hash_field)
    if field in data:
        values = _string_list(data[field], field)
        raw = json.dumps(data[field], separators=(",", ":")).encode("utf-8")
        _check_hash(raw, expected, field.capitalize())
        return values
    path = (manifest_dir / _required_string(data, path_field)).resolve()
    raw = _verified_bytes(path, expected, field.capitalize())
    try:
        parsed = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ManifestSchemaError(f"Malformed {field} JSON: {exc}") from exc
    return _string_list(parsed, field)


def _numeric_tokens(text: str) -> set[str]:
    """Numeric tokens under image_sources' exact normalization and regex."""
    return set(image_sources._NUM_RE.findall(image_sources._normalize(text)))


def _derived_query(sentence: str) -> str:
    """Short retry query only when it keeps every number of the sentence.

    ``retry_subject_query`` truncates to a few tokens, so a short score ("3 a
    1") or a late year ("Flamengo campeão em 1994") can fall out of the short
    query. The downloaded-image pixel gate rejects contradictory numbers only
    while the prompt still has numbers, so a numberless short prompt would
    silently disable it. Keep the original sentence - exact whitespace
    included - whenever any number would be lost. Numbers are never appended,
    rewritten or invented, and no provider is called here.
    """
    derived = image_sources.retry_subject_query(sentence)
    if not derived:
        return sentence
    if _numeric_tokens(derived) != _numeric_tokens(sentence):
        return sentence
    return derived


def _provenance(data: dict, transcript_data: dict | None) -> Any:
    provenance: Any = None
    for source in (data, transcript_data):
        if source is None:
            continue
        for field in ("transcript_provenance", "provenance"):
            if field not in source:
                continue
            value = source[field]
            if not (
                _nonempty_string(value) or (isinstance(value, dict) and bool(value))
            ):
                raise ManifestSchemaError(
                    "Transcript provenance must be a nonempty string or dict"
                )
            if provenance is None:
                provenance = value
    if provenance is None:
        raise ManifestMissingError("Explicit transcript provenance is required")
    return provenance


def load_resume_inputs(
    manifest: str | Path | dict,
    *,
    base_dir: str | Path | None = None,
) -> ResumeInputs:
    """Validate schema, asset shapes and raw hashes without sourcing new assets."""
    if isinstance(manifest, (str, Path)):
        if isinstance(manifest, str) and not manifest.strip():
            raise ManifestSchemaError("Manifest path must be a nonempty string")
        manifest_file = Path(manifest).resolve()
        if not manifest_file.is_file():
            raise ManifestMissingError(f"Manifest file not found: {manifest_file}")
        manifest_dir = manifest_file.parent
        try:
            data = json.loads(manifest_file.read_text("utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ManifestSchemaError(f"Malformed manifest JSON: {exc}") from exc
    elif isinstance(manifest, dict):
        data = manifest
        manifest_dir = (
            Path(base_dir).resolve() if base_dir is not None else Path.cwd().resolve()
        )
    else:
        raise ManifestSchemaError(
            f"Manifest must be path or dict, got {type(manifest).__name__}"
        )

    if not isinstance(data, dict):
        raise ManifestSchemaError("Manifest root must be an object")
    schema = data.get("schema")
    if type(schema) is not int or schema != 1:
        raise ManifestSchemaError(f"Schema 1 manifest required, got {schema!r}")

    if data.get("image_source") == "ai" or data.get("images") == "ai":
        raise AISourceError("AI image sourcing is prohibited")

    audio_path = (manifest_dir / _required_string(data, "audio_path")).resolve()
    transcript_path = (
        manifest_dir / _required_string(data, "transcript_path")
    ).resolve()
    audio_sha = _required_sha256(data, "audio_sha256")
    transcript_sha = _required_sha256(data, "transcript_sha256")
    _verified_bytes(audio_path, audio_sha, "Audio")
    transcript_bytes = _verified_bytes(transcript_path, transcript_sha, "Transcript")

    try:
        raw_transcript_text = transcript_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ManifestSchemaError("Transcript must be UTF-8") from exc
    transcript_data: dict | None = None
    transcript_text: object
    try:
        parsed = json.loads(raw_transcript_text)
    except json.JSONDecodeError as exc:
        if transcript_path.suffix.lower() == ".json":
            raise ManifestSchemaError(f"Malformed transcript JSON: {exc}") from exc
        transcript_text = raw_transcript_text
    else:
        if not isinstance(parsed, dict):
            raise ManifestSchemaError("JSON transcript must be an object")
        transcript_data = parsed
        transcript_text = parsed.get("text")
    if not _nonempty_string(transcript_text):
        raise ManifestSchemaError("Transcript text must be a nonempty string")

    provenance = _provenance(data, transcript_data)
    saved_queries = _saved_strings(data, "queries", manifest_dir)
    saved_paragraphs = _saved_strings(data, "paragraphs", manifest_dir)
    if saved_queries is not None:
        queries = saved_queries
    else:
        # Only blank split fragments are separators, not sentences. Keep every
        # actual sentence, including its whitespace, unless the short retry
        # query retains all of that sentence's numeric tokens.
        sentences = (
            sentence
            for sentence in re.split(r"(?<=[.!?])\s+|\n+", transcript_text)
            if sentence.strip()
        )
        queries = tuple(_derived_query(sentence) for sentence in sentences)

    return ResumeInputs(
        audio_path=audio_path,
        transcript_path=transcript_path,
        transcript_text=transcript_text,
        transcript_provenance=provenance,
        paragraphs=saved_paragraphs,
        queries=queries,
        audio_sha256=audio_sha,
        transcript_sha256=transcript_sha,
    )
