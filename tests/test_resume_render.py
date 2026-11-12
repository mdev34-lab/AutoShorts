"""SOURCE ONLY: author did not import, compile, lint or execute this suite.

The only application substitute is the compositor classmethod: its nonempty
bytes are NOT a real MP4 or a duration/codec regression. Input loading, inventory
writing/validation, Pillow, candidates, fit/pixel/subject policy and credits
writers are real. Filesystem/guard fault injection tests refusal and rollback,
not weaker successful gates. Test code calls no acquisition, RPC, TTS, model or
native rendering API; the synthetic compositor stands in for composition.

Application initializers/explainer were unavailable in the author's workspace;
installed/native dependencies were not probed. Missing real dependencies yield
explicit skips, never fake gates. Unqualified discovery exposes one aggregate
that launches all 14 original inner tests with the same standalone offline
launcher. Qualified discovery runs the original classes after the unchanged
real gate. Child reports retain their underlying counts; one outer item is
not evidence of 14 separately collected pytest tests.
"""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import os
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

_HERE = Path(__file__).resolve().parent
_PROJECT = _HERE.parent if _HERE.name == "tests" else _HERE
_PRODUCTION = _PROJECT / "src" / "autoshorts"
_INNER = "--inside-resume-render-tests"
# A routing hint only: inner execution must still pass the real gate below.
_INNER_RUN = _INNER in sys.argv[1:] or os.environ.get("AUTOSHORTS_NETWORK_GUARD") == "1"


def _direct(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError("reviewed source unavailable")
    module = importlib.util.module_from_spec(spec)
    previous = sys.modules.get(name)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        if previous is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous
    return module


def _require_boundary():
    _direct(_PRODUCTION / "resume_render.py", "_render_suite_guard")._require_boundary()


def setUpModule(module=None):
    # Pytest supplies the module; unittest calls without it. Only the outer
    # pytest aggregate defers qualification to its real offline child.
    if module is None or _INNER_RUN:
        _require_boundary()


setup_module = setUpModule


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _drop_package(worker):
    for name in list(sys.modules):
        if name == worker._APP or name.startswith(worker._APP + "."):
            sys.modules.pop(name, None)


class RenderGuardTests(unittest.TestCase):
    __test__ = _INNER_RUN

    def setUp(self):
        self.worker = _direct(_PRODUCTION / "resume_render.py", "_render_guard_tests")
        self.temporary = tempfile.TemporaryDirectory(prefix="synthetic-render-guard-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()

    def test_full_guard_precedes_every_application_import(self):
        with mock.patch.object(self.worker, "_load_data") as application:
            changes = [
                {key: "0"}
                for key in (
                    "AUTOSHORTS_NETWORK_GUARD",
                    "AUTOSHORTS_OFFLINE_BOUNDARY",
                    "PYTHON_DOTENV_DISABLED",
                )
            ]
            changes.extend(
                [
                    {"AUTOSHORTS_PARENT_NETNS": os.readlink("/proc/self/ns/net")},
                    {"AUTOSHORTS_BOUNDARY_READY": str(self.root / "missing-ready")},
                ]
            )
            wrong = self.root / "wrong-ready"
            wrong.write_text("net:[0]", encoding="ascii")
            changes.append({"AUTOSHORTS_BOUNDARY_READY": str(wrong)})
            for change in changes:
                with self.subTest(change=change), mock.patch.dict(os.environ, change):
                    with self.assertRaises(self.worker.RenderError):
                        self.worker.render(
                            "/missing-manifest", "/missing-inventory", "/missing.mp4"
                        )
            for name in ("dotenv", "dotenv.main"):
                with (
                    self.subTest(stub=name),
                    mock.patch.dict(sys.modules, {name: types.ModuleType(name)}),
                ):
                    with self.assertRaises(self.worker.RenderError):
                        self.worker.render(
                            "/missing-manifest", "/missing-inventory", "/missing.mp4"
                        )
            read_text = Path.read_text

            def external_interface(path, *args, **kwargs):
                if str(path) == "/proc/net/dev":
                    return "header\nheader\neth0: 0 0\n"
                return read_text(path, *args, **kwargs)

            with mock.patch.object(Path, "read_text", new=external_interface):
                with self.assertRaises(self.worker.RenderError):
                    self.worker.render(
                        "/missing-manifest", "/missing-inventory", "/missing.mp4"
                    )
            application.assert_not_called()

    def test_bad_cli_arguments_and_guard_diagnostics_never_echo_private_text(self):
        for args in (
            ["--unknown=SYNTHETIC_SECRET"],
            ["--manifest", "SYNTHETIC_SECRET"],
            ["--man", "SYNTHETIC_SECRET"],
            ["--allow-unverified=1"],
        ):
            output, error = io.StringIO(), io.StringIO()
            with (
                self.subTest(args=args),
                redirect_stdout(output),
                redirect_stderr(error),
            ):
                self.assertNotEqual(self.worker.main(args), 0)
            self.assertEqual(output.getvalue(), "")
            self.assertNotIn("SYNTHETIC_SECRET", error.getvalue())
        output, error = io.StringIO(), io.StringIO()
        args = [
            "--manifest",
            "/SYNTHETIC_SECRET",
            "--inventory",
            "/missing",
            "--output",
            "/new.mp4",
        ]
        with mock.patch.dict(os.environ, {"AUTOSHORTS_NETWORK_GUARD": "0"}):
            with redirect_stdout(output), redirect_stderr(error):
                self.assertNotEqual(self.worker.main(args), 0)
        self.assertEqual(output.getvalue(), "")
        self.assertNotIn("SYNTHETIC_SECRET", error.getvalue())


class RenderTests(unittest.TestCase):
    __test__ = _INNER_RUN

    def setUp(self):
        self.worker = _direct(_PRODUCTION / "resume_render.py", "_saved_render_tests")
        required = (
            "modules/__init__.py",
            "modules/resume_inputs.py",
            "modules/resume_inventory.py",
            "modules/image_sources.py",
            "modules/image_content.py",
            "modules/image_subject.py",
            "modules/image_searcher.py",
            "modules/config.py",
        )
        missing = [name for name in required if not (_PRODUCTION / name).is_file()]
        if missing:
            self.skipTest(
                "real render fixture source unavailable: " + ", ".join(missing)
            )
        try:
            from PIL import Image, ImageDraw
        except ImportError as exc:
            self.skipTest(
                "real Pillow unavailable: " + (exc.name or type(exc).__name__)
            )
        self.assertTrue(getattr(Image, "__file__", None), "real Pillow is required")
        environment = mock.patch.dict(os.environ, {}, clear=False)
        environment.start()
        self.addCleanup(environment.stop)
        old_tempdir = tempfile.tempdir
        self.addCleanup(setattr, tempfile, "tempdir", old_tempdir)
        self.temporary = tempfile.TemporaryDirectory(prefix="synthetic-saved-render-")
        self.addCleanup(self.temporary.cleanup)
        self.addCleanup(_drop_package, self.worker)
        self.root = Path(self.temporary.name).resolve()
        self.acquired, self.rendered = self.root / "acquired", self.root / "rendered"
        self.acquired.mkdir(mode=0o700)
        self.rendered.mkdir(mode=0o700)
        self.output = self.rendered / "saved.mp4"
        self.inventory = self.acquired / "inventory.json"
        self.credits = self.acquired / "image_credits.txt"
        self.image = self.acquired / "mountain.png"
        image = Image.new("RGB", (1200, 1600), (110, 180, 235))
        try:
            drawing = ImageDraw.Draw(image)
            drawing.polygon([(0, 1080), (430, 350), (980, 1080)], fill=(105, 120, 135))
            drawing.polygon(
                [(500, 1120), (980, 520), (1200, 1120)], fill=(85, 110, 125)
            )
            drawing.rectangle((0, 1120, 1200, 1600), fill=(65, 125, 65))
            image.save(self.image, format="PNG")
        finally:
            image.close()
        self.audio, self.transcript = (
            self.root / "saved.wav",
            self.root / "transcript.json",
        )
        self.audio.write_bytes(
            b"SYNTHETIC SAVED AUDIO - NEVER DECODED BY UNIT COMPOSITOR"
        )
        self.text = "  Saved synthetic transcript.\nUnchanged recovery.  "
        self.transcript.write_text(json.dumps({"text": self.text}), encoding="utf-8")
        self.paragraphs = [
            "  Exact first paragraph.  ",
            "Second\nparagraph — unchanged.",
        ]
        self.paragraph_file, self.query_file = (
            self.root / "paragraphs.json",
            self.root / "queries.json",
        )
        self.paragraph_file.write_text(json.dumps(self.paragraphs), encoding="utf-8")
        self.query_file.write_text(json.dumps(["mountain"]), encoding="utf-8")
        self.manifest = self.root / "manifest.json"
        self.data = {
            "schema": 1,
            "audio_path": self.audio.name,
            "transcript_path": self.transcript.name,
            "audio_sha256": _sha(self.audio.read_bytes()),
            "transcript_sha256": _sha(self.transcript.read_bytes()),
            "queries_path": self.query_file.name,
            "queries_sha256": _sha(self.query_file.read_bytes()),
            "paragraphs_path": self.paragraph_file.name,
            "paragraphs_sha256": _sha(self.paragraph_file.read_bytes()),
            "transcript_provenance": {
                "kind": "synthetic saved fixture",
                "note": "SYNTHETIC_PRIVATE_PROVENANCE",
            },
        }
        self._rebuild_inventory()
        self.originals = {
            path: path.read_bytes()
            for path in (
                self.manifest,
                self.inventory,
                self.credits,
                self.audio,
                self.transcript,
                self.paragraph_file,
                self.query_file,
                self.image,
                self.image.with_suffix(".credits.json"),
            )
        }
        self.calls, self.effect = [], None
        self.body = b"SYNTHETIC NONEMPTY COMPOSITOR OUTPUT - NOT A REAL MP4"
        test = self

        class SavedCompositor:
            def __init__(self, *args, **kwargs):
                raise AssertionError("generator constructor must never run")

            @classmethod
            def compose_saved_inputs(cls, images, audio, paragraphs, output):
                test.calls.append((list(images), audio, list(paragraphs), Path(output)))
                print("SYNTHETIC_PRIVATE_COMPOSITOR")
                print("SYNTHETIC_PRIVATE_COMPOSITOR", file=sys.stderr)
                Path(output).write_bytes(test.body)
                if test.effect is not None:
                    test.effect(Path(output))

        patch = mock.patch.object(
            self.worker, "_load_compositor", return_value=SavedCompositor
        )
        self.compositor_loader = patch.start()
        self.addCleanup(patch.stop)

    def _rebuild_inventory(self):
        self.manifest.write_text(json.dumps(self.data), encoding="utf-8")
        self.inventory.unlink(missing_ok=True)
        _drop_package(self.worker)
        self.worker._configure(self.acquired, True)
        try:
            with self.worker._quiet_application():
                app = self.worker._load_data()
                sources = sys.modules[self.worker._APP + ".modules.image_sources"]
                content = sys.modules[self.worker._APP + ".modules.image_content"]
                subject = sys.modules[self.worker._APP + ".modules.image_subject"]
                self.assertTrue(subject.is_broad_request("mountain"))
                candidates = sources.filter_ddg_results(
                    "mountain",
                    [
                        {
                            "image": "https://images.example.test/safe/mountain.png",
                            "title": "Mountain with green hills and blue sky",
                            "url": "https://photos.example.test/gallery/mountain",
                            "width": 1200,
                            "height": 1600,
                        }
                    ],
                )
                self.assertEqual(
                    len(candidates),
                    1,
                    "real fit/source gates must accept the synthetic fixture",
                )
                try:
                    reason = content.check_image_content(self.image, "mountain")
                except content.ContentUnavailableError as exc:
                    self.skipTest(
                        "real pixel dependency unavailable: " + str(exc)[:400]
                    )
                self.assertIsNone(
                    reason, "real pixel gate must accept the synthetic fixture"
                )
                searcher = app.inventory.ImageSearcher
                searcher._write_provenance_sidecar(
                    self.image, candidates[0], "mountain"
                )
                searcher._write_credits_file([str(self.image)])
                app.inventory.write_inventory(
                    self.manifest,
                    [str(self.image)],
                    self.inventory,
                    allow_unverified=True,
                )
        except ModuleNotFoundError as exc:
            self.skipTest(
                "real Python dependency missing: " + (exc.name or type(exc).__name__)
            )
        finally:
            _drop_package(self.worker)

    def _parent_relative_manifest(self):
        parent = self.root / "saved-manifests"
        parent.mkdir(mode=0o700)
        self.manifest = parent / "manifest.json"
        saved = {
            "audio": self.audio,
            "transcript": self.transcript,
            "queries": self.query_file,
            "paragraphs": self.paragraph_file,
        }
        for field, path in saved.items():
            self.data[field + "_path"] = "../" + path.name
        return saved

    def _invoke(self, *, allow=True, output=None):
        # Each call models a fresh one-shot worker, without replacing data/gates.
        _drop_package(self.worker)
        return self.worker.render(
            str(self.manifest),
            str(self.inventory),
            str(output or self.output),
            allow_unverified=allow,
        )

    def _no_staging(self):
        for parent in (self.acquired, self.rendered):
            self.assertEqual(list(parent.glob(".resume-render-*")), [])

    def _sources_unchanged(self):
        for path, raw in self.originals.items():
            self.assertEqual(path.read_bytes(), raw)

    def test_exact_saved_inputs_classmethod_and_atomic_credits_copy(self):
        result = self._invoke()
        self.assertEqual(len(self.calls), 1)
        images, audio, paragraphs, temporary = self.calls[0]
        self.assertEqual(
            (images, audio, paragraphs),
            ([str(self.image)], str(self.audio), self.paragraphs),
        )
        self.assertEqual(temporary.parent, self.output.parent)
        self.assertEqual(temporary.suffix, ".mp4")
        self.assertNotEqual(temporary, self.output)
        self.assertFalse(temporary.exists())
        self.assertEqual(self.output.read_bytes(), self.body)
        copied = self.rendered / "image_credits.txt"
        self.assertEqual(copied.read_bytes(), self.originals[self.credits])
        self.assertIn(b"PUBLICATION REVIEW REQUIRED:", copied.read_bytes())
        self.assertEqual(
            set(result),
            {"output", "inventory", "credits", "audio_sha256", "transcript_sha256"},
        )
        self.assertEqual(result["audio_sha256"], self.data["audio_sha256"])
        self.assertEqual(result["transcript_sha256"], self.data["transcript_sha256"])
        self.assertEqual(result["credits"], str(copied))
        self.assertNotIn("SYNTHETIC_PRIVATE_PROVENANCE", json.dumps(result))
        self._sources_unchanged()
        self._no_staging()

    def test_parent_relative_saved_inputs_resolve_with_unchanged_hashes_and_bytes(self):
        saved = self._parent_relative_manifest()
        hashes = {field: self.data[field + "_sha256"] for field in saved}
        self._rebuild_inventory()
        bound = {
            path: path.read_bytes()
            for path in (
                self.manifest,
                self.inventory,
                self.credits,
                self.image,
                self.image.with_suffix(".credits.json"),
            )
        }
        result = self._invoke()
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(
            self.calls[0][:3], ([str(self.image)], str(self.audio), self.paragraphs)
        )
        self.assertEqual(self.output.read_bytes(), self.body)
        self.assertEqual(result["audio_sha256"], hashes["audio"])
        self.assertEqual(result["transcript_sha256"], hashes["transcript"])
        for field, path in saved.items():
            self.assertEqual(self.data[field + "_path"], "../" + path.name)
            self.assertEqual(self.data[field + "_sha256"], hashes[field])
            self.assertEqual(_sha(path.read_bytes()), hashes[field])
            self.assertEqual(path.read_bytes(), self.originals[path])
        for path, raw in bound.items():
            self.assertEqual(path.read_bytes(), raw)
        self.assertEqual(
            (self.rendered / "image_credits.txt").read_bytes(), bound[self.credits]
        )
        self._no_staging()

    def test_same_byte_saved_input_symlink_aliases_are_refused_before_composition(self):
        saved = self._parent_relative_manifest()
        target_dir = self.root / "saved-audio-target"
        target_dir.mkdir(mode=0o700)
        replacement = target_dir / self.audio.name
        replacement.write_bytes(self.originals[self.audio])
        self.assertEqual(_sha(replacement.read_bytes()), self.data["audio_sha256"])
        alias = self.root / "saved-audio-link.wav"
        alias.symlink_to(replacement)
        parent_alias = self.root / "saved-audio-parent-link"
        parent_alias.symlink_to(target_dir, target_is_directory=True)
        for value in (
            "../" + alias.name,
            "../" + parent_alias.name + "/" + replacement.name,
        ):
            with self.subTest(audio_path=value):
                self.data["audio_path"] = value
                # Rebuild real bindings so rejection cannot rely on a stale inventory.
                self._rebuild_inventory()
                bound = {
                    path: path.read_bytes()
                    for path in (
                        self.manifest,
                        self.inventory,
                        self.credits,
                        self.image,
                        self.image.with_suffix(".credits.json"),
                    )
                }
                self.assertEqual(
                    (self.manifest.parent / value).read_bytes(),
                    self.originals[self.audio],
                )
                with self.assertRaises(self.worker.RenderError):
                    self._invoke()
                self.assertEqual(self.calls, [])
                self.compositor_loader.assert_not_called()
                self.assertFalse(self.output.exists())
                self.assertFalse((self.rendered / "image_credits.txt").exists())
                self.assertTrue(alias.is_symlink())
                self.assertTrue(parent_alias.is_symlink())
                self.assertEqual(replacement.read_bytes(), self.originals[self.audio])
                for path, raw in bound.items():
                    self.assertEqual(path.read_bytes(), raw)
                for path in saved.values():
                    self.assertEqual(path.read_bytes(), self.originals[path])
                self._no_staging()

    def test_parent_relative_saved_inputs_do_not_authorize_inventory_image_escape(self):
        self._parent_relative_manifest()
        self._rebuild_inventory()
        raw = self.inventory.read_bytes()
        listing = json.loads(raw)
        outside = self.root / "outside.png"
        outside_sidecar = outside.with_suffix(".credits.json")
        outside.write_bytes(self.originals[self.image])
        outside_sidecar.write_bytes(
            self.originals[self.image.with_suffix(".credits.json")]
        )
        listing["items"][0]["image_path"] = "../" + outside.name
        listing["items"][0]["sidecar_path"] = "../" + outside_sidecar.name
        self.inventory.write_text(json.dumps(listing), encoding="utf-8")
        try:
            with self.assertRaises((ValueError, OSError)):
                self._invoke()
            self.assertEqual(self.calls, [])
            self.compositor_loader.assert_not_called()
            self.assertFalse(self.output.exists())
            self.assertFalse((self.rendered / "image_credits.txt").exists())
            self.assertEqual(outside.read_bytes(), self.originals[self.image])
            self.assertEqual(
                outside_sidecar.read_bytes(),
                self.originals[self.image.with_suffix(".credits.json")],
            )
        finally:
            self.inventory.write_bytes(raw)
        self._no_staging()

    def test_no_saved_paragraphs_uses_one_exact_unchanged_transcript(self):
        del self.data["paragraphs_path"]
        del self.data["paragraphs_sha256"]
        self._rebuild_inventory()
        saved = self.transcript.read_bytes()
        self._invoke()
        self.assertEqual(self.calls[0][2], [self.text])
        self.assertEqual(self.transcript.read_bytes(), saved)
        self._no_staging()

    def test_same_directory_or_identical_credits_are_reused_without_overwrite(self):
        source_identity = self.credits.stat().st_ino
        same_directory = self.acquired / "same-directory.mp4"
        result = self._invoke(output=same_directory)
        self.assertEqual(result["credits"], str(self.credits))
        self.assertEqual(self.credits.stat().st_ino, source_identity)
        destination = self.rendered / "image_credits.txt"
        destination.write_bytes(self.originals[self.credits])
        destination_identity = destination.stat().st_ino
        self._invoke()
        self.assertEqual(destination.stat().st_ino, destination_identity)
        self._sources_unchanged()
        self._no_staging()

    def test_unknown_rights_require_explicit_current_opt_in_before_composition(self):
        os.environ["ALLOW_UNVERIFIED_IMAGES"] = "1"
        with self.assertRaises(ValueError):
            self._invoke(allow=False)
        self.assertEqual(os.environ["ALLOW_UNVERIFIED_IMAGES"], "0")
        self.assertEqual(self.calls, [])
        self.compositor_loader.assert_not_called()
        self.assertFalse(self.output.exists())
        self._sources_unchanged()

    def test_corrupt_initial_assets_and_credits_are_refused_before_composition(self):
        for path in (
            self.audio,
            self.transcript,
            self.paragraph_file,
            self.query_file,
            self.inventory,
            self.image,
            self.image.with_suffix(".credits.json"),
            self.credits,
        ):
            raw = path.read_bytes()
            with self.subTest(path=path.name):
                path.write_bytes(b"INVALID SYNTHETIC ASSET")
                try:
                    with self.assertRaises((ValueError, OSError)):
                        self._invoke()
                    self.assertEqual(self.calls, [])
                finally:
                    path.write_bytes(raw)
        self.compositor_loader.assert_not_called()
        self.assertFalse(self.output.exists())
        self._no_staging()

    def test_raw_bindings_are_rechecked_after_composition_before_publication(self):
        for path, raw in self.originals.items():
            with self.subTest(changed=path.name):
                self.effect = lambda temporary, path=path, raw=raw: path.write_bytes(
                    raw + b" "
                )
                try:
                    with self.assertRaises((ValueError, OSError)):
                        self._invoke()
                    self.assertFalse(self.output.exists())
                    self.assertFalse((self.rendered / "image_credits.txt").exists())
                    self._no_staging()
                finally:
                    path.write_bytes(raw)
        self.effect = None
        self._sources_unchanged()

    def test_noncanonical_symlink_and_existing_output_paths_are_refused(self):
        existing = self.rendered / "existing.mp4"
        existing.write_bytes(b"DO NOT REPLACE")
        link = self.rendered / "linked.mp4"
        link.symlink_to(existing)
        dangling = self.rendered / "dangling.mp4"
        dangling.symlink_to(self.root / "missing")
        directory = self.rendered / "directory.mp4"
        directory.mkdir()
        paths = [
            existing,
            link,
            dangling,
            directory,
            self.rendered / "bad.mkv",
            "relative.mp4",
            str(self.rendered) + "/../rendered/escape.mp4",
        ]
        for path in paths:
            with self.subTest(path=str(path)), self.assertRaises((ValueError, OSError)):
                self._invoke(output=path)
        self.assertEqual(existing.read_bytes(), b"DO NOT REPLACE")
        self.assertEqual(self.calls, [])
        self.assertTrue(link.is_symlink())
        self.assertTrue(dangling.is_symlink())
        self._no_staging()

    def test_input_aliases_and_inventory_escapes_are_refused(self):
        for path in (self.audio, self.image, self.image.with_suffix(".credits.json")):
            saved = path.with_name(path.name + ".saved")
            path.rename(saved)
            path.symlink_to(saved)
            try:
                with (
                    self.subTest(alias=path.name),
                    self.assertRaises((ValueError, OSError)),
                ):
                    self._invoke()
            finally:
                path.unlink()
                saved.rename(path)
        listing = json.loads(self.originals[self.inventory])
        listing["items"][0]["image_path"] = "../outside.png"
        outside = self.root / "outside.png"
        outside.write_bytes(self.originals[self.image])
        self.inventory.write_text(json.dumps(listing), encoding="utf-8")
        try:
            with self.assertRaises((ValueError, OSError)):
                self._invoke()
        finally:
            self.inventory.write_bytes(self.originals[self.inventory])
        self.assertEqual(outside.read_bytes(), self.originals[self.image])
        self.assertEqual(self.calls, [])
        self.assertFalse(self.output.exists())
        self._sources_unchanged()
        self._no_staging()

    def test_compositor_exception_empty_or_unsafe_staging_never_publishes(self):
        def fail(temporary):
            raise RuntimeError("SYNTHETIC_PRIVATE_FAILURE")

        for body, effect in (
            (self.body, fail),
            (b"", None),
            (self.body, lambda temporary: temporary.chmod(0o666)),
        ):
            with self.subTest(empty=not body, effect=effect):
                self.body, self.effect = body, effect
                with self.assertRaises((ValueError, RuntimeError, OSError)):
                    self._invoke()
                self.assertFalse(self.output.exists())
                self.assertFalse((self.rendered / "image_credits.txt").exists())
                self._no_staging()
        self._sources_unchanged()

    def test_racing_video_or_credits_are_never_overwritten_or_deleted(self):
        destination_credits = self.rendered / "image_credits.txt"
        for target in (self.output, destination_credits):
            with self.subTest(target=target.name):
                self.effect = lambda temporary, target=target: target.write_bytes(
                    b"RACING OWNER BYTES"
                )
                with self.assertRaises((ValueError, OSError)):
                    self._invoke()
                self.assertEqual(target.read_bytes(), b"RACING OWNER BYTES")
                other = destination_credits if target == self.output else self.output
                self.assertFalse(other.exists())
                target.unlink()
                self._no_staging()
        self._sources_unchanged()

    def test_postpublication_failure_rolls_back_only_matching_owned_inodes(self):
        real_link = os.link
        raw_manifest = self.manifest.read_bytes()
        for replace_video in (False, True):

            def link_then_change(
                source, destination, replace_video=replace_video, **kwargs
            ):
                real_link(source, destination, **kwargs)
                if destination == self.output.name:
                    if replace_video:
                        os.unlink(destination, dir_fd=kwargs["dst_dir_fd"])
                        descriptor = os.open(
                            destination,
                            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                            0o600,
                            dir_fd=kwargs["dst_dir_fd"],
                        )
                        try:
                            os.write(descriptor, b"FOREIGN REPLACEMENT")
                        finally:
                            os.close(descriptor)
                    self.manifest.write_bytes(raw_manifest + b" ")

            # Real link first, then a filesystem fault; no validator is replaced.
            with (
                self.subTest(replace_video=replace_video),
                mock.patch.object(self.worker.os, "link", side_effect=link_then_change),
            ):
                try:
                    with self.assertRaises((ValueError, OSError)):
                        self._invoke()
                    if replace_video:
                        self.assertEqual(
                            self.output.read_bytes(), b"FOREIGN REPLACEMENT"
                        )
                    else:
                        self.assertFalse(self.output.exists())
                    self.assertFalse((self.rendered / "image_credits.txt").exists())
                    self._no_staging()
                finally:
                    self.manifest.write_bytes(raw_manifest)
                    self.output.unlink(missing_ok=True)
        self._sources_unchanged()

    def test_main_prints_only_minimal_json_not_compositor_text_or_provenance(self):
        _drop_package(self.worker)
        output, error = io.StringIO(), io.StringIO()
        args = [
            "--manifest",
            str(self.manifest),
            "--inventory",
            str(self.inventory),
            "--output",
            str(self.output),
            "--allow-unverified",
        ]
        with redirect_stdout(output), redirect_stderr(error):
            code = self.worker.main(args)
        self.assertEqual((code, error.getvalue()), (0, ""))
        result = json.loads(output.getvalue())
        self.assertEqual(result["output"], str(self.output))
        self.assertNotIn("SYNTHETIC_PRIVATE", output.getvalue())
        self.assertNotIn(self.text, output.getvalue())
        self._sources_unchanged()
        self._no_staging()


def _run_outer_suite():
    """Run the full original 14-test standalone suite through its real launcher."""
    boundary = _direct(_PRODUCTION / "resume_boundary.py", "_render_outer_boundary")
    try:
        result = boundary.run_offline(
            str(Path(__file__).resolve()),
            [_INNER],
            timeout=600,
            source_root=str(_PROJECT),
        )
    except boundary.BoundaryError as exc:
        # Preserve all diagnostics the unchanged boundary supplies, then fail.
        sys.stdout.write(exc.stdout)
        sys.stderr.write(exc.stderr)
        raise
    sys.stdout.write(result.stdout)
    sys.stderr.write(result.stderr)
    if result.returncode != 0:
        raise AssertionError(f"Guarded render suite exited with {result.returncode}")
    return result.returncode


def test_guarded_render_suite(capsys):
    """One outer aggregate runs all original render guard and saved-input tests."""
    # This disables only output capture, never a network or qualification guard.
    with capsys.disabled():
        _run_outer_suite()


test_guarded_render_suite.__test__ = not _INNER_RUN


def load_tests(_loader, tests, _pattern):
    if _INNER_RUN:
        _require_boundary()
        return tests
    # Public FunctionTestCase keeps outer dispatch separate from inner fixtures.
    return unittest.TestSuite([unittest.FunctionTestCase(_run_outer_suite)])


def _main():
    if sys.argv[1:] == [_INNER]:
        unittest.main(argv=[sys.argv[0]], verbosity=2)
        return 0
    if sys.argv[1:]:
        raise SystemExit(
            "Run without arguments or discover inside the qualified outer boundary"
        )
    return _run_outer_suite()


if __name__ == "__main__":
    raise SystemExit(_main())
