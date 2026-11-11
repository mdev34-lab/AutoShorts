"""Saved-input entry point: python src/autoshorts/resume_cli.py acquire|render.

Standard library and fixed stdlib-only siblings only. The parent never imports
application code. Acquisition always uses the reviewed public-fetch supervisor;
rendering always uses the complete offline boundary. No executable, helper,
transport, source root, RPC directory or environment is caller-selectable.
No normal generator, TTS, script generation, upload or automatic publication.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import stat
import sys
from pathlib import Path
from urllib.parse import urlsplit

_HERE = Path(os.path.realpath(__file__)).parent
_SIBLINGS = frozenset(
    {
        "resume_boundary.py",
        "resume_supervisor.py",
        "resume_acquire.py",
        "resume_render.py",
    }
)
_MAX_OUTPUT = 1024 * 1024


class CLIError(ValueError):
    """Invalid CLI input; never includes caller-provided text in diagnostics."""


def _path(value):
    if isinstance(value, bool) or not isinstance(value, (str, Path)):
        raise CLIError("invalid path")
    text = str(value)
    if (
        not text
        or "\x00" in text
        or not os.path.isabs(text)
        or os.path.realpath(text) != text
    ):
        raise CLIError("invalid path")
    return Path(text)


def _existing(value, *, directory=False):
    path = _path(value)
    try:
        mode = path.lstat().st_mode
    except OSError as exc:
        raise CLIError("missing path") from exc
    if not (stat.S_ISDIR(mode) if directory else stat.S_ISREG(mode)):
        raise CLIError("invalid path type")
    return path


def _sibling(name):
    if name not in _SIBLINGS:
        raise CLIError("unapproved sibling")
    return _existing(_HERE / name)


def _load_local(name):
    if name not in ("resume_boundary.py", "resume_supervisor.py"):
        raise CLIError("unapproved loader")
    module_name = "_autoshorts_cli_" + Path(name).stem
    spec = importlib.util.spec_from_file_location(module_name, str(_sibling(name)))
    if spec is None or spec.loader is None:
        raise CLIError("fixed launcher unavailable")
    module = importlib.util.module_from_spec(spec)
    previous = sys.modules.get(module_name)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        if previous is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = previous
    return module


def _seconds(value):
    try:
        seconds = float(value)
    except (ValueError, OverflowError) as exc:
        raise CLIError("invalid timeout") from exc
    if not math.isfinite(seconds) or not 0 < seconds <= 7200:
        raise CLIError("invalid timeout")
    return seconds


def _photo_prefix(value):
    # Early syntax check only; the unchanged supervisor/RPC revalidate authority.
    if (
        not 0 < len(value) <= 2048
        or "\\" in value
        or any(ord(char) <= 32 or ord(char) >= 127 for char in value)
    ):
        raise CLIError("invalid photo prefix")
    try:
        parts = urlsplit(value)
        port = parts.port
    except ValueError as exc:
        raise CLIError("invalid photo prefix") from exc
    host, path = parts.hostname or "", parts.path
    if (
        parts.scheme != "https"
        or parts.netloc != host
        or port is not None
        or parts.fragment
        or "#" in value
        or parts.query
        or "?" in value
        or "." not in host
        or host.replace(".", "").isdigit()
        or host.startswith(".")
        or host.endswith(".")
        or any(char not in "abcdefghijklmnopqrstuvwxyz0123456789.-" for char in host)
        or not path.startswith("/")
        or not path.endswith("/")
        or "//" in path
        or any(part in (".", "..") for part in path.split("/"))
        or any(code in path.lower() for code in ("%2e", "%2f", "%5c", "%00"))
    ):
        raise CLIError("invalid photo prefix")
    return value


def _status(code):
    if type(code) is int:
        if 1 <= code <= 255:
            return code
        if -127 <= code < 0:
            return 128 - code
    return 1


def _pairs(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate worker key")
        value[key] = item
    return value


def _constant(value):
    raise ValueError("invalid worker constant")


def _success(raw, args):
    if type(raw) is not str or not 0 < len(raw.encode("utf-8")) <= _MAX_OUTPUT:
        raise ValueError("invalid worker output")
    data = json.loads(raw, object_pairs_hook=_pairs, parse_constant=_constant)
    fields = (
        {"inventory", "images", "credits", "reused"}
        if args.command == "acquire"
        else {"output", "inventory", "credits", "audio_sha256", "transcript_sha256"}
    )
    if type(data) is not dict or set(data) != fields:
        raise ValueError("invalid worker output fields")
    if args.command == "acquire":
        root = _path(args.output_dir)
        if (
            _path(data["inventory"]) != root / "inventory.json"
            or _path(data["credits"]) != root / "image_credits.txt"
            or type(data["reused"]) is not bool
            or type(data["images"]) is not list
            or not data["images"]
        ):
            raise ValueError("invalid acquisition result")
        if any(not _path(path).is_relative_to(root) for path in data["images"]):
            raise ValueError("invalid acquisition image location")
    else:
        output = _path(args.output)
        if (
            _path(data["output"]) != output
            or _path(data["inventory"]) != _path(args.inventory)
            or _path(data["credits"]) != output.parent / "image_credits.txt"
        ):
            raise ValueError("invalid render result")
        for field in ("audio_sha256", "transcript_sha256"):
            value = data[field]
            if (
                type(value) is not str
                or len(value) != 64
                or any(char not in "0123456789abcdef" for char in value)
            ):
                raise ValueError("invalid saved-input hash")
    return data


class _Parser(argparse.ArgumentParser):
    def error(self, message):
        raise CLIError("invalid arguments")


def main(argv=None):
    parser = _Parser(prog="resume_cli.py", allow_abbrev=False)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("acquire", "render"):
        sub = commands.add_parser(name, allow_abbrev=False)
        sub.add_argument("--manifest", required=True)
        sub.add_argument("--allow-unverified", action="store_true", default=False)
        sub.add_argument("--timeout", type=_seconds, required=True)
        if name == "acquire":
            sub.add_argument("--output-dir", required=True)
            sub.add_argument(
                "--photo-allowlist", type=_photo_prefix, action="append", required=True
            )
        else:
            sub.add_argument("--inventory", required=True)
            sub.add_argument("--output", required=True)
    try:
        args = parser.parse_args(argv)
        manifest = _existing(args.manifest)
        source_root = _existing(_HERE.parent, directory=True)
        worker_args = ["--manifest", str(manifest)]
        if args.command == "acquire":
            output = _existing(args.output_dir, directory=True)
            if len(args.photo_allowlist) > 64 or len(set(args.photo_allowlist)) != len(
                args.photo_allowlist
            ):
                raise CLIError("invalid allowlist count")
            worker_args.extend(["--output-dir", str(output)])
            if args.allow_unverified:
                worker_args.append("--allow-unverified")
            worker = _sibling("resume_acquire.py")
            supervisor = _load_local("resume_supervisor.py")
            result = supervisor.run_with_public_fetch(
                str(worker),
                worker_args,
                timeout=args.timeout,
                source_root=str(source_root),
                photo_allowlist=list(args.photo_allowlist),
            )
        else:
            inventory, output = _existing(args.inventory), _path(args.output)
            _existing(output.parent, directory=True)
            if output.suffix != ".mp4" or os.path.lexists(output):
                raise CLIError("output must be a new .mp4 path")
            worker_args.extend(["--inventory", str(inventory), "--output", str(output)])
            if args.allow_unverified:
                worker_args.append("--allow-unverified")
            worker = _sibling("resume_render.py")
            boundary = _load_local("resume_boundary.py")
            result = boundary.run_offline(
                str(worker),
                worker_args,
                timeout=args.timeout,
                source_root=str(source_root),
            )
        if type(result.returncode) is not int or result.returncode != 0:
            print("resume_cli: worker failed", file=sys.stderr)
            return _status(result.returncode)
        payload = _success(result.stdout, args)
        if result.stderr != "":
            print("resume_cli: worker emitted diagnostics", file=sys.stderr)
    except CLIError:
        print("resume_cli: invalid arguments or paths", file=sys.stderr)
        return 2
    except Exception as exc:
        print("resume_cli: operation failed", file=sys.stderr)
        return _status(getattr(exc, "returncode", None))
    print(json.dumps(payload, sort_keys=True, separators=(",", ":")), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
