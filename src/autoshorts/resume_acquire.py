"""One-shot, direct-file acquisition worker; no rendering or publication.

Run only through the existing offline supervisor. The boundary verifier lives
inside its generated bootstrap, not as an exported function: the assertions
below recheck its namespace/marker semantics, never install a replacement
bootstrap, dotenv stub, or weaker networking fallback. Application modules
load only after that check and process-local configuration, without executing
package __init__ files. Use one invocation per process.

The existing searcher and inventory APIs retain all acceptance checks. Hashes
bind saved bytes and metadata, not image truth or legal reuse rights. Unknown
rights remain unknown under explicit opt-in. Private directories and no-follow
writes are not a sandbox against another process with the same filesystem UID.
The supervisor owns the whole-run deadline and process-group cleanup.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.machinery
import importlib.util
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
_APP = "_autoshorts_resume_acquire_modules"
_METADATA_LIMIT = 1024 * 1024


class AcquireError(ValueError):
    """A fixed, non-sensitive acquisition refusal diagnostic."""


def _path(value):
    if isinstance(value, bool) or not isinstance(value, (str, Path)):
        raise AcquireError("paths must be canonical absolute paths")
    text = str(value)
    if (
        not text
        or "\x00" in text
        or not os.path.isabs(text)
        or os.path.realpath(text) != text
    ):
        raise AcquireError("paths must be canonical absolute paths")
    return Path(text)


def _directory(value, *, private=False):
    path = _path(value)
    info = path.lstat()
    if not stat.S_ISDIR(info.st_mode):
        raise AcquireError("a required directory is unsafe")
    if private and (info.st_uid != os.getuid() or info.st_mode & 0o077):
        raise AcquireError(
            "private directories must be owned and inaccessible to other users"
        )
    return path


def _regular(value, *, writable=False):
    path = _path(value)
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode):
        raise AcquireError("a required file is not regular")
    if writable and (
        info.st_uid != os.getuid() or info.st_nlink != 1 or info.st_mode & 0o022
    ):
        raise AcquireError(
            "writable files must be owned, protected and have no hard-link aliases"
        )
    return path


def _exists(path):
    try:
        path.lstat()
    except FileNotFoundError:
        return False
    return True


def _read(path, limit):
    descriptor = os.open(
        path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW | os.O_CLOEXEC
    )
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or not 0 < info.st_size <= limit:
            raise AcquireError("metadata must be a bounded nonempty regular file")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            raw = stream.read(limit + 1)
        if len(raw) != info.st_size or len(raw) > limit:
            raise AcquireError("metadata changed while being read")
        return raw
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
            raise AcquireError("the verified offline boundary is required")
        current = os.readlink("/proc/self/ns/net")
        parent = os.environ.get("AUTOSHORTS_PARENT_NETNS", "")
        if (
            not current.startswith("net:[")
            or not parent.startswith("net:[")
            or current == parent
        ):
            raise AcquireError("a separate offline network namespace is required")
        ready = _regular(os.environ.get("AUTOSHORTS_BOUNDARY_READY", ""))
        if _read(ready, 256).decode("ascii") != current:
            raise AcquireError("the current namespace readiness marker is required")
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
            raise AcquireError("external network interfaces or routes are prohibited")
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
                raise AcquireError("the installed offline dotenv stub is required")
    except AcquireError:
        raise
    except (OSError, UnicodeError, ValueError, IndexError) as exc:
        raise AcquireError("the verified offline boundary is unavailable") from exc


def _private_child(parent, name):
    path = parent / name
    if not _exists(path):
        path.mkdir(mode=0o700)
    return _directory(path, private=True)


def _tree(root):
    for entry in root.iterdir():
        info = entry.lstat()
        if stat.S_ISDIR(info.st_mode):
            _tree(_directory(entry, private=True))
        else:
            _regular(entry, writable=True)


def _digest(path):
    descriptor = os.open(
        _regular(path), os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC
    )
    try:
        digest = hashlib.sha256()
        while chunk := os.read(descriptor, 65536):
            digest.update(chunk)
        return digest.hexdigest()
    finally:
        os.close(descriptor)


def _protected(manifest, raw, inputs, tmp, inventory, credits):
    data = json.loads(raw)
    protected = [
        (manifest, hashlib.sha256(raw).hexdigest()),
        (_path(inputs.audio_path), inputs.audio_sha256),
        (_path(inputs.transcript_path), inputs.transcript_sha256),
    ]
    for field in ("queries", "paragraphs"):
        if f"{field}_path" in data:
            path = (manifest.parent / data[f"{field}_path"]).resolve(strict=True)
            protected.append((path, data[f"{field}_sha256"].lower()))
    for path, _ in protected:
        if path.is_relative_to(tmp) or path in (inventory, credits):
            raise AcquireError("saved inputs overlap writable acquisition artifacts")
    return protected


def _unchanged(protected):
    if any(_digest(path) != expected for path, expected in protected):
        raise AcquireError("saved input bytes changed")


@contextmanager
def _quiet_application():
    # Suppress application/native diagnostics that might contain saved queries,
    # URLs or tokens. Only main's fixed diagnostic or minimal JSON escapes.
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


def _new_package():
    _require_boundary()
    if any(name == _APP or name.startswith(_APP + ".") for name in sys.modules):
        raise AcquireError("a fresh acquisition worker process is required")
    directory = _directory(_HERE / "modules")
    package = types.ModuleType(_APP)
    package.__package__ = _APP
    package.__path__ = [str(directory)]
    package.__spec__ = importlib.machinery.ModuleSpec(
        _APP, loader=None, is_package=True
    )
    package.__spec__.submodule_search_locations = package.__path__
    sys.modules[_APP] = package


def _load_inputs(manifest):
    _new_package()
    module = __import__(_APP + ".resume_inputs", fromlist=["load_resume_inputs"])
    return module.load_resume_inputs(manifest)


def _load_components():
    _require_boundary()
    inventory = __import__(_APP + ".resume_inventory", fromlist=["write_inventory"])
    searcher = __import__(_APP + ".image_searcher", fromlist=["ImageSearcher"])
    ddg = __import__(_APP + ".resume_ddg", fromlist=["FixedDDGSearch"])
    config = __import__(_APP + ".config", fromlist=["IMAGE_CACHE_DIR"])
    return types.SimpleNamespace(
        inventory=inventory,
        ImageSearcher=searcher.ImageSearcher,
        FixedDDGSearch=ddg.FixedDDGSearch,
        config=config,
    )


def _load_rpc():
    path = _regular(_HERE / "resume_rpc.py")
    name = "_autoshorts_acquire_rpc"
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise AcquireError("the fixed sibling RPC module is unavailable")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(name, None)
    return module


@contextmanager
def _downloader(rpc, client, cache):
    descriptor = os.open(
        cache, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    )
    identity = os.fstat(descriptor)

    def check(target):
        target = _path(target)
        now = _directory(cache, private=True).lstat()
        if target.parent != cache or (now.st_dev, now.st_ino) != (
            identity.st_dev,
            identity.st_ino,
        ):
            raise AcquireError("download targets must stay in the private image cache")
        try:
            info = os.stat(target.name, dir_fd=descriptor, follow_symlinks=False)
        except FileNotFoundError:
            return target
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_nlink != 1
            or info.st_uid != os.getuid()
            or info.st_mode & 0o022
        ):
            raise AcquireError("unsafe download target")
        return target

    def download(url, target):
        target = check(target)
        try:
            body = client.get_photo(
                url,
                max_bytes=rpc.PHOTO_MAX_BYTES,
                timeout=rpc.PHOTO_MAX_TIMEOUT,
                max_redirects=rpc.PHOTO_MAX_REDIRECTS,
            )
        except (rpc.RPCRequestError, rpc.RPCRemoteError):
            return False
        if type(body) is not bytes or not 0 < len(body) <= rpc.PHOTO_MAX_BYTES:
            raise AcquireError("photo RPC returned invalid bounded bytes")
        target = check(target)
        temporary = ".resume-photo-" + secrets.token_hex(16)
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC
        output = os.open(temporary, flags, 0o600, dir_fd=descriptor)
        try:
            try:
                view = memoryview(body)
                while view:
                    written = os.write(output, view)
                    if written <= 0:
                        raise AcquireError("photo bytes could not be written")
                    view = view[written:]
                os.fsync(output)
            finally:
                os.close(output)
            check(target)
            os.replace(
                temporary, target.name, src_dir_fd=descriptor, dst_dir_fd=descriptor
            )
            os.fsync(descriptor)
        finally:
            try:
                os.unlink(temporary, dir_fd=descriptor)
            except FileNotFoundError:
                pass
        return True

    try:
        yield download
    finally:
        os.close(descriptor)


def _paths(values, count, root, searcher):
    if type(values) is not list or not values or len(values) != count:
        raise AcquireError("every saved query requires an image and sidecar")
    paths = []
    for value in values:
        image = _regular(value, writable=True)
        sidecar = _regular(searcher._sidecar_path(image), writable=True)
        if not image.is_relative_to(root) or not sidecar.is_relative_to(root):
            raise AcquireError(
                "inventory assets must remain inside the acquisition output"
            )
        paths.append(image)
    return paths


def _credits(paths, credits, searcher):
    text = _read(_regular(credits, writable=True), _METADATA_LIMIT).decode("utf-8")
    unknown = 0
    for image in paths:
        record = searcher._read_sidecar(image)
        if (
            record is None
            or record.get("kind") != "web"
            or record.get("provider") != "ddg"
        ):
            raise AcquireError(
                "DDG provenance and existing-writer credits are required"
            )
        verified = record.get("license_verified") is True
        unknown += not verified
        license_text = (
            record["license"] if verified else "UNVERIFIED - reuse rights unknown"
        )
        required = [
            image.name + "\n",
            "Title: " + record["title"],
            "Source: " + record["source_page"],
            "Image file: " + record["image_url"],
            "License: " + license_text,
            "Changes: " + record["modification"],
        ]
        if any(part not in text for part in required):
            raise AcquireError("credits are missing existing sidecar details")
    status = (
        f"PUBLICATION REVIEW REQUIRED: {unknown} image(s) without verified reuse rights"
        if unknown
        else "Rights status: all web images carry a verified reusable license"
    )
    if not text.startswith(status + "\n\n"):
        raise AcquireError("credits rights status or unknown-rights warning is missing")


def _remove_inventory(path, published):
    try:
        current = path.lstat()
        if (current.st_dev, current.st_ino) != (published.st_dev, published.st_ino):
            return
        path.unlink()
        descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except FileNotFoundError:
        pass


def acquire(manifest, output_dir, rpc_dir, *, allow_unverified=False):
    _require_boundary()
    # This worker is single-use; all new application artifacts must be private.
    os.umask(0o077)
    if type(allow_unverified) is not bool:
        raise AcquireError("unknown rights require an explicit boolean option")
    manifest, output, rpc_dir = (
        _regular(manifest),
        _directory(output_dir),
        _directory(rpc_dir, private=True),
    )
    info = output.lstat()
    if info.st_uid != os.getuid() or info.st_mode & 0o022:
        raise AcquireError(
            "output directory must be owned and not writable by other users"
        )
    inventory, credits = output / "inventory.json", output / "image_credits.txt"
    for path in (inventory, credits):
        if _exists(path):
            _regular(path, writable=True)
    tmp = _private_child(output, ".resume-tmp")
    cache = _private_child(tmp, "autoshorts_image_cache")
    _tree(tmp)
    raw_manifest = _read(manifest, _METADATA_LIMIT)
    os.environ.update(
        IMAGE_PROVIDERS="ddg",
        IMAGE_AI_FALLBACK="off",
        ALLOW_UNVERIFIED_IMAGES="1" if allow_unverified else "0",
        OUTPUT_DIR=str(output),
        TMPDIR=str(tmp),
        TMP=str(tmp),
        TEMP=str(tmp),
        TEMP_DIR_PREFIX="autoshorts_",
    )
    tempfile.tempdir = None
    with _quiet_application():
        inputs = _load_inputs(manifest)
        queries = list(inputs.queries)
        protected = _protected(manifest, raw_manifest, inputs, tmp, inventory, credits)
        _unchanged(protected)
        app = _load_components()
        config = app.config
        if (
            config.OUTPUT_DIR != output
            or config.IMAGE_CACHE_DIR != cache
            or tuple(config.IMAGE_PROVIDERS) != ("ddg",)
            or config.IMAGE_AI_FALLBACK is not False
            or config.ALLOW_UNVERIFIED_IMAGES is not allow_unverified
        ):
            raise AcquireError("application configuration is stale or unsafe")
        reused = _exists(inventory)
        if reused:
            values = app.inventory.validate_inventory(
                manifest, inventory, allow_unverified=allow_unverified
            )
            paths = _paths(values, len(queries), output, app.ImageSearcher)
            _credits(paths, credits, app.ImageSearcher)
            _unchanged(protected)
            _tree(tmp)
        else:
            rpc = _load_rpc()
            client = rpc.FileRPC(str(rpc_dir))
            with _downloader(rpc, client, cache) as download:
                searcher = app.ImageSearcher(
                    search=app.FixedDDGSearch(client.get_ddg),
                    download=download,
                    providers=("ddg",),
                    ai_fallback=False,
                )
                values = searcher.get_images(queries)
            _tree(tmp)
            paths = _paths(values, len(queries), cache, app.ImageSearcher)
            _credits(paths, credits, app.ImageSearcher)
            _unchanged(protected)
            app.inventory.write_inventory(
                manifest,
                [str(path) for path in paths],
                inventory,
                allow_unverified=allow_unverified,
            )
            published = inventory.lstat()
            try:
                _regular(inventory, writable=True)
                values = app.inventory.validate_inventory(
                    manifest, inventory, allow_unverified=allow_unverified
                )
                paths = _paths(values, len(queries), cache, app.ImageSearcher)
                _credits(paths, credits, app.ImageSearcher)
                _unchanged(protected)
                _tree(tmp)
            except BaseException:
                _remove_inventory(inventory, published)
                raise
    return {
        "inventory": str(inventory),
        "images": [str(path) for path in paths],
        "credits": str(credits),
        "reused": reused,
    }


class _Parser(argparse.ArgumentParser):
    def error(self, message):
        raise AcquireError("invalid acquisition arguments")


def main(argv=None):
    parser = _Parser(prog="resume_acquire.py", allow_abbrev=False)
    for flag in ("--manifest", "--output-dir", "--rpc-dir"):
        parser.add_argument(flag, required=True)
    parser.add_argument("--allow-unverified", action="store_true", default=False)
    try:
        args = parser.parse_args(argv)
        result = acquire(
            args.manifest,
            args.output_dir,
            args.rpc_dir,
            allow_unverified=args.allow_unverified,
        )
    except AcquireError as exc:
        print(f"resume_acquire: {exc}", file=sys.stderr)
        return 1
    except Exception:
        print(
            "resume_acquire: acquisition failed; inventory was not accepted",
            file=sys.stderr,
        )
        return 1
    print(json.dumps(result, sort_keys=True, separators=(",", ":")), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
