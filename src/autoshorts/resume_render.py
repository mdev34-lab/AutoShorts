"""One-shot saved-input renderer; direct-file launch inside resume_boundary only.

No acquisition, RPC, generator construction, script generation or TTS calls.
The unchanged compositor is imported in a package context after the namespace,
readiness marker and installed dotenv stub have been checked. Root/generator
package initializers are bypassed; the real modules package supplies exports.
Use a fresh worker process, as for acquisition. No networking fallback exists.

Publication uses exclusive hard links, not replacement. Each file is atomic,
not the video/credits pair. Credits are copied byte-for-byte only when needed;
identical existing credits may be reused. Rollback removes only matching inodes
created by this invocation. This is not a filesystem sandbox against another
process with the same UID. Uncatchable termination can leave staging or already
linked files; no Python cleanup can run after SIGKILL.
Hashes and existing gates do not establish image truth or legal reuse rights.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.machinery
import json
import os
import secrets
import stat
import sys
import tempfile
import types
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path

_HERE = Path(os.path.realpath(__file__)).parent
_APP = "_autoshorts_resume_render_app"
_METADATA_LIMIT = 1024 * 1024


class RenderError(ValueError):
    """A fixed, non-sensitive refusal diagnostic."""


def _path(value):
    if isinstance(value, bool) or not isinstance(value, (str, Path)):
        raise RenderError("paths must be canonical absolute paths")
    text = str(value)
    if (
        not text
        or "\x00" in text
        or not os.path.isabs(text)
        or os.path.realpath(text) != text
    ):
        raise RenderError("paths must be canonical absolute paths")
    return Path(text)


def _regular(value):
    path = _path(value)
    if not stat.S_ISREG(path.lstat().st_mode):
        raise RenderError("a required file is not regular")
    return path


def _directory(value, *, writable=False):
    path = _path(value)
    info = path.lstat()
    if not stat.S_ISDIR(info.st_mode):
        raise RenderError("a required directory is unsafe")
    if writable and (info.st_uid != os.getuid() or info.st_mode & 0o022):
        raise RenderError("the output parent must be owned and protected")
    return path


def _exists(path):
    try:
        path.lstat()
    except FileNotFoundError:
        return False
    return True


def _read(path, limit=_METADATA_LIMIT):
    descriptor = os.open(
        _regular(path), os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW | os.O_CLOEXEC
    )
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or not 0 < info.st_size <= limit:
            raise RenderError("metadata must be bounded nonempty regular files")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            raw = stream.read(limit + 1)
        if len(raw) != info.st_size or len(raw) > limit:
            raise RenderError("metadata changed while being read")
        return raw
    finally:
        os.close(descriptor)


def _digest(path):
    descriptor = os.open(
        _regular(path), os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW | os.O_CLOEXEC
    )
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_size <= 0:
            raise RenderError("assets must be nonempty regular files")
        digest, count = hashlib.sha256(), 0
        while chunk := os.read(descriptor, 65536):
            digest.update(chunk)
            count += len(chunk)
        after = os.fstat(descriptor)
        if count != info.st_size or (
            info.st_size,
            info.st_mtime_ns,
            info.st_ctime_ns,
        ) != (after.st_size, after.st_mtime_ns, after.st_ctime_ns):
            raise RenderError("an asset changed while being read")
        return digest.hexdigest()
    finally:
        os.close(descriptor)


def _require_boundary():
    try:
        if any(
            os.environ.get(key) != "1"
            for key in (
                "AUTOSHORTS_NETWORK_GUARD",
                "AUTOSHORTS_OFFLINE_BOUNDARY",
                "PYTHON_DOTENV_DISABLED",
            )
        ):
            raise RenderError("the verified offline boundary is required")
        current = os.readlink("/proc/self/ns/net")
        parent = os.environ.get("AUTOSHORTS_PARENT_NETNS", "")
        if (
            not current.startswith("net:[")
            or not parent.startswith("net:[")
            or current == parent
        ):
            raise RenderError("a separate offline network namespace is required")
        ready = _regular(os.environ.get("AUTOSHORTS_BOUNDARY_READY", ""))
        if _read(ready, 256).decode("ascii") != current:
            raise RenderError("the current namespace readiness marker is required")
        lines = (
            Path("/proc/net/dev")
            .read_text(encoding="ascii", errors="replace")
            .splitlines()[2:]
        )
        devices = {line.split(":", 1)[0].strip() for line in lines if ":" in line}
        for name, column, skip in (("route", 0, 1), ("ipv6_route", -1, 0)):
            try:
                routes = (Path("/proc/net") / name).read_text(
                    encoding="ascii", errors="replace"
                )
            except FileNotFoundError:
                continue
            devices.update(
                line.split()[column]
                for line in routes.splitlines()[skip:]
                if line.split()
            )
        if devices - {"lo"}:
            raise RenderError("external network interfaces or routes are prohibited")
        for name in ("dotenv", "dotenv.main"):
            module = sys.modules.get(name)
            if (
                not isinstance(module, types.ModuleType)
                or getattr(module, "AUTOSHORTS_OFFLINE_STUB", None) is not True
                or any(
                    not callable(getattr(module, method, None))
                    for method in ("load_dotenv", "dotenv_values", "find_dotenv")
                )
            ):
                raise RenderError("the installed offline dotenv stub is required")
    except RenderError:
        raise
    except (OSError, UnicodeError, ValueError, IndexError) as exc:
        raise RenderError("the verified offline boundary is unavailable") from exc


@contextmanager
def _quiet_application():
    with open(os.devnull, "w", encoding="utf-8") as sink:
        saved = []
        try:
            sys.stdout.flush()
            sys.stderr.flush()
            for descriptor in (1, 2):
                saved.append((descriptor, os.dup(descriptor)))
                os.dup2(sink.fileno(), descriptor)
            with redirect_stdout(sink), redirect_stderr(sink):
                yield
        finally:
            for descriptor, original in reversed(saved):
                try:
                    os.dup2(original, descriptor)
                finally:
                    os.close(original)


def _package(name, directory):
    package = types.ModuleType(name)
    package.__package__ = name
    package.__path__ = [str(_directory(directory))]
    package.__spec__ = importlib.machinery.ModuleSpec(
        name, loader=None, is_package=True
    )
    package.__spec__.submodule_search_locations = package.__path__
    sys.modules[name] = package


def _configure(parent, allow_unverified):
    os.environ.update(
        OUTPUT_DIR=str(parent),
        IMAGE_PROVIDERS="ddg",
        IMAGE_AI_FALLBACK="off",
        ALLOW_UNVERIFIED_IMAGES="1" if allow_unverified else "0",
    )
    tempfile.tempdir = None


def _load_data():
    _require_boundary()
    if any(name == _APP or name.startswith(_APP + ".") for name in sys.modules):
        raise RenderError("a fresh render worker process is required")
    _package(_APP, _HERE)
    inputs = __import__(
        _APP + ".modules.resume_inputs", fromlist=["load_resume_inputs"]
    )
    inventory = __import__(
        _APP + ".modules.resume_inventory", fromlist=["validate_inventory"]
    )
    config = __import__(_APP + ".modules.config", fromlist=["ALLOW_UNVERIFIED_IMAGES"])
    return types.SimpleNamespace(inputs=inputs, inventory=inventory, config=config)


def _load_compositor():
    _require_boundary()
    _package(_APP + ".generators", _HERE / "generators")
    module = __import__(_APP + ".generators.explainer", fromlist=["ExplainerGenerator"])
    return module.ExplainerGenerator


def _named(base, value, *, saved_input=False):
    if not isinstance(value, str) or not value.strip():
        raise RenderError("a saved asset path is missing")
    path = Path(value)
    path = path if path.is_absolute() else base / path
    # Normalize lexical saved-input traversal, not symlinks or inventory names.
    return _regular(os.path.abspath(path) if saved_input else path)


def _credits(raw, images, records):
    text, unknown = raw.decode("utf-8"), 0
    for image, record in zip(images, records, strict=True):
        if record.get("kind") != "web" or record.get("provider") != "ddg":
            raise RenderError("existing DDG provenance is required")
        verified = record.get("license_verified") is True
        unknown += not verified
        license_text = (
            record["license"] if verified else "UNVERIFIED - reuse rights unknown"
        )
        required = [
            image.name + "\n",
            "Title: " + record["title"],
            "Creator: " + (record.get("creator") or "not provided by source"),
            "Source: " + record["source_page"],
            "Image file: " + record["image_url"],
            "License: " + license_text,
            "Changes: " + record["modification"],
        ]
        if record.get("license_url"):
            required.append("License URL: " + record["license_url"])
        if any(part not in text for part in required):
            raise RenderError("credits do not match the existing image provenance")
    status = (
        f"PUBLICATION REVIEW REQUIRED: {unknown} image(s) without verified reuse rights"
        if unknown
        else "Rights status: all web images carry a verified reusable license"
    )
    if not text.startswith(status + "\n\n"):
        raise RenderError("the existing credits rights status is required")


def _fresh(app, manifest, inventory, allow_unverified):
    _require_boundary()
    raw_manifest, raw_inventory = _read(manifest), _read(inventory)
    credits = _regular(inventory.parent / "image_credits.txt")
    raw_credits = _read(credits)
    protected = {
        path: hashlib.sha256(raw).hexdigest()
        for path, raw in (
            (manifest, raw_manifest),
            (inventory, raw_inventory),
            (credits, raw_credits),
        )
    }
    inputs = app.inputs.load_resume_inputs(manifest)
    values = app.inventory.validate_inventory(
        manifest, inventory, allow_unverified=allow_unverified
    )
    if type(values) is not list or not values or len(values) != len(inputs.queries):
        raise RenderError("every saved query requires a validated image")
    data, listing = json.loads(raw_manifest), json.loads(raw_inventory)

    def bind(path, expected):
        path = _regular(path)
        if _digest(path) != expected.lower() or (
            path in protected and protected[path] != expected.lower()
        ):
            raise RenderError("saved asset bytes changed")
        protected[path] = expected.lower()
        return path

    for field in ("audio", "transcript"):
        path = _named(manifest.parent, data[field + "_path"], saved_input=True)
        if path != getattr(inputs, field + "_path"):
            raise RenderError("saved input paths changed")
        bind(path, getattr(inputs, field + "_sha256"))
    for field in ("queries", "paragraphs"):
        if field + "_path" in data:
            bind(
                _named(manifest.parent, data[field + "_path"], saved_input=True),
                data[field + "_sha256"],
            )
    images, records = [], []
    for value, item in zip(values, listing["items"], strict=True):
        image = _regular(value)
        if not image.is_relative_to(inventory.parent) or image != _named(
            inventory.parent, item["image_path"]
        ):
            raise RenderError("inventory images must remain contained and canonical")
        bind(image, item["image_sha256"])
        sidecar = _named(inventory.parent, item["sidecar_path"])
        if not sidecar.is_relative_to(inventory.parent) or sidecar != image.with_suffix(
            ".credits.json"
        ):
            raise RenderError("inventory sidecars must remain contained and canonical")
        bind(sidecar, item["sidecar_sha256"])
        images.append(image)
        records.append(json.loads(_read(sidecar)))
    _credits(raw_credits, images, records)
    if any(_digest(path) != expected for path, expected in protected.items()):
        raise RenderError("saved input, inventory, credits or image bytes changed")
    paragraphs = (
        list(inputs.paragraphs)
        if inputs.paragraphs is not None
        else [inputs.transcript_text]
    )
    return types.SimpleNamespace(
        inputs=inputs,
        images=images,
        paragraphs=paragraphs,
        protected=protected,
        credits=raw_credits,
    )


def _revalidate(app, manifest, inventory, allow_unverified, initial):
    fresh = _fresh(app, manifest, inventory, allow_unverified)
    if (
        fresh.protected != initial.protected
        or fresh.images != initial.images
        or fresh.paragraphs != initial.paragraphs
    ):
        raise RenderError("saved render bindings changed")


def _identity(info):
    return info.st_dev, info.st_ino


def _parent_current(parent, descriptor):
    if _identity(_directory(parent, writable=True).lstat()) != _identity(
        os.fstat(descriptor)
    ):
        raise RenderError("the output parent changed identity")


def _discard(descriptor, entries):
    if not entries:
        return
    error = None
    for name, identity in reversed(entries):
        try:
            info = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
            if _identity(info) == identity:
                os.unlink(name, dir_fd=descriptor)
        except FileNotFoundError:
            pass
        except OSError as exc:
            error = error or exc
    try:
        os.fsync(descriptor)
    except OSError as exc:
        error = error or exc
    if error is not None:
        raise error


def _stage(parent, descriptor, suffix, body=b""):
    name = ".resume-render-" + secrets.token_hex(16) + suffix
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC
    output = os.open(name, flags, 0o600, dir_fd=descriptor)
    identity = _identity(os.fstat(output))
    try:
        view = memoryview(body)
        while view:
            written = os.write(output, view)
            if written <= 0:
                raise RenderError("staging bytes could not be written")
            view = view[written:]
        os.fsync(output)
    except BaseException:
        _discard(descriptor, [(name, identity)])
        raise
    finally:
        os.close(output)
    return parent / name, identity


def _link(source, destination, descriptor, identity, published):
    if (
        _identity(os.stat(source.name, dir_fd=descriptor, follow_symlinks=False))
        != identity
    ):
        raise RenderError("a staging file changed identity")
    os.link(
        source.name,
        destination.name,
        src_dir_fd=descriptor,
        dst_dir_fd=descriptor,
        follow_symlinks=False,
    )
    published.append((destination.name, identity))
    os.fsync(descriptor)
    if (
        _identity(os.stat(destination.name, dir_fd=descriptor, follow_symlinks=False))
        != identity
    ):
        raise RenderError("a published file changed identity")


def render(manifest, inventory, output, *, allow_unverified=False):
    _require_boundary()
    if type(allow_unverified) is not bool:
        raise RenderError("unknown rights require an explicit boolean option")
    manifest, inventory, output = _regular(manifest), _regular(inventory), _path(output)
    parent = _directory(output.parent, writable=True)
    if output.suffix != ".mp4" or _exists(output):
        raise RenderError("output must be a new local .mp4 path")
    _configure(parent, allow_unverified)
    with _quiet_application():
        app = _load_data()
        if (
            app.config.ALLOW_UNVERIFIED_IMAGES is not allow_unverified
            or app.config.IMAGE_AI_FALLBACK is not False
            or tuple(app.config.IMAGE_PROVIDERS) != ("ddg",)
            or app.config.OUTPUT_DIR != parent
        ):
            raise RenderError("application configuration is stale or unsafe")
        initial = _fresh(app, manifest, inventory, allow_unverified)
        credits = parent / "image_credits.txt"
        copy_credits = not _exists(credits)
        if not copy_credits and _read(credits) != initial.credits:
            raise RenderError(
                "existing destination credits differ; no overwrite is allowed"
            )
        descriptor = os.open(
            parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
        )
        staging, published = [], []
        try:
            _parent_current(parent, descriptor)
            temporary, identity = _stage(parent, descriptor, ".mp4")
            staging.append((temporary.name, identity))
            compositor = _load_compositor()
            compositor.compose_saved_inputs(
                [str(path) for path in initial.images],
                str(initial.inputs.audio_path),
                list(initial.paragraphs),
                str(temporary),
            )
            info = _regular(temporary).lstat()
            if (
                _identity(info) != identity
                or info.st_uid != os.getuid()
                or info.st_nlink != 1
                or info.st_mode & 0o022
            ):
                raise RenderError("the compositor produced an unsafe staging file")
            video_sha = _digest(temporary)
            video = os.open(temporary, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
            try:
                os.fsync(video)
            finally:
                os.close(video)
            _revalidate(app, manifest, inventory, allow_unverified, initial)
            _parent_current(parent, descriptor)
            if copy_credits:
                credit_stage, credit_identity = _stage(
                    parent, descriptor, ".txt", initial.credits
                )
                staging.append((credit_stage.name, credit_identity))
                _link(credit_stage, credits, descriptor, credit_identity, published)
            elif _read(credits) != initial.credits:
                raise RenderError("destination credits changed")
            _link(temporary, output, descriptor, identity, published)
            _revalidate(app, manifest, inventory, allow_unverified, initial)
            _parent_current(parent, descriptor)
            if _digest(output) != video_sha or _read(credits) != initial.credits:
                raise RenderError("published output or credits changed")
            _discard(descriptor, staging)
            staging.clear()
        except BaseException:
            _discard(descriptor, published)
            raise
        finally:
            try:
                _discard(descriptor, staging)
            finally:
                os.close(descriptor)
    return {
        "output": str(output),
        "inventory": str(inventory),
        "credits": str(credits),
        "audio_sha256": initial.inputs.audio_sha256,
        "transcript_sha256": initial.inputs.transcript_sha256,
    }


class _Parser(argparse.ArgumentParser):
    def error(self, message):
        raise RenderError("invalid render arguments")


def main(argv=None):
    parser = _Parser(prog="resume_render.py", allow_abbrev=False)
    for flag in ("--manifest", "--inventory", "--output"):
        parser.add_argument(flag, required=True)
    parser.add_argument("--allow-unverified", action="store_true", default=False)
    try:
        args = parser.parse_args(argv)
        result = render(
            args.manifest,
            args.inventory,
            args.output,
            allow_unverified=args.allow_unverified,
        )
    except RenderError as exc:
        print(f"resume_render: {exc}", file=sys.stderr)
        return 1
    except Exception:
        print(
            "resume_render: rendering failed; output was not accepted", file=sys.stderr
        )
        return 1
    print(json.dumps(result, sort_keys=True, separators=(",", ":")), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
