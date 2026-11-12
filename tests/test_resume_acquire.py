"""SOURCE ONLY; author did not import, compile, lint or execute this suite.

Actual checkout layout: tests/ and src/autoshorts/. Explicit execution and
unqualified discovery use the same reviewed outer offline launcher. Discovery
outside the boundary exposes one aggregate that launches all 13 original inner
tests, including the separate integration case; it is not 13 outer test items.
Qualified discovery runs the original classes after the unchanged real gate.
Child reports are forwarded, not replaced by an aggregate-count claim.
No test uses live HTTP, providers or private assets.
Unit cases use synthetic dependencies and establish wiring/refusal behavior
only. The separate integration case uses real modules/PIL/pixel+fit gates and
real RPC/supervisor/fetch, replacing only the local public_http byte transport.
Its explicit unknown-rights opt-in does not exercise learned subject checking.
Source-only authoring: image_sources.py, image_content.py and image_subject.py
were unavailable in this workspace; installed Pillow/OCR dependencies were not
probed. Explicit missing dependencies skip integration, never substitute gates.
No test establishes photo truth, source honesty or legal reuse rights.
"""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import os
import shutil
import stat
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
if not (_PRODUCTION / "resume_acquire.py").is_file():
    _PRODUCTION = _PROJECT
_INNER = "--inside-acquire-tests"
# A routing hint only: inner execution must still pass the real gate below.
_INNER_RUN = _INNER in sys.argv[1:] or os.environ.get("AUTOSHORTS_NETWORK_GUARD") == "1"


def _direct(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError("reviewed source unavailable")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(name, None)
    return module


def _require_boundary():
    # This import is stdlib-only; no application imports precede its guard.
    _direct(
        _PRODUCTION / "resume_acquire.py", "_acquire_gate_probe"
    )._require_boundary()


def setUpModule(module=None):
    # Pytest supplies the module; unittest calls without it. Only the outer
    # pytest aggregate defers qualification to its real offline child.
    if module is None or _INNER_RUN:
        _require_boundary()


setup_module = setUpModule


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


class AcquireTests(unittest.TestCase):
    __test__ = _INNER_RUN

    def setUp(self):
        self.worker = _direct(
            _PRODUCTION / "resume_acquire.py", "_synthetic_acquire_worker"
        )
        # acquire() is a one-shot worker, but these unit calls share a process.
        old_umask = os.umask(0o077)
        os.umask(old_umask)
        self.addCleanup(os.umask, old_umask)
        environment = mock.patch.dict(os.environ, {}, clear=False)
        environment.start()
        self.addCleanup(environment.stop)
        old_tempdir = tempfile.tempdir
        self.addCleanup(setattr, tempfile, "tempdir", old_tempdir)
        self.temporary = tempfile.TemporaryDirectory(prefix="synthetic-acquire-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.output, self.rpc_dir = self.root / "output", self.root / "rpc"
        self.output.mkdir(mode=0o700)
        self.rpc_dir.mkdir(mode=0o700)
        self.queries = ["  broad observatory scene  ", "Second scene — unchanged"]
        self.audio = self.root / "saved.wav"
        self.transcript = self.root / "saved-transcript.json"
        self.audio.write_bytes(b"SYNTHETIC AUDIO ONLY")
        self.transcript.write_bytes(
            json.dumps(
                {
                    "text": "  Saved transcript.\nNo regeneration.  ",
                    "provenance": "synthetic saved input",
                }
            ).encode()
        )
        self.manifest = self.root / "manifest.json"
        self.data = {
            "schema": 1,
            "audio_path": self.audio.name,
            "transcript_path": self.transcript.name,
            "audio_sha256": _sha(self.audio.read_bytes()),
            "transcript_sha256": _sha(self.transcript.read_bytes()),
            "queries": self.queries,
            "queries_sha256": _sha(
                json.dumps(self.queries, separators=(",", ":")).encode()
            ),
            "transcript_provenance": "synthetic saved input",
        }
        self._save_manifest()
        self.events, self.ddg_calls, self.photo_calls, self.clients = [], [], [], []
        self.constructor_calls, self.prompt_lists, self.forbidden = [], [], []
        self.license = "CC BY 4.0"
        self.photo_body = b"SYNTHETIC IMAGE BYTES"
        self.denied, self.fail_query = set(), None
        self.pixel_reject = self.subject_reject = self.fit_reject = False
        self.broad = True
        self.drop_return = self.fail_validation = False
        self.omit_credits = self.omit_sidecar = False
        self.candidate_urls = ["https://images.example.test/safe/a.jpg"]
        self.rpc = self.app = None
        self.addCleanup(self._drop_package)
        self._install_global_traps()
        new_package = self.worker._new_package

        def dependencies_after_guard():
            new_package()
            self._install_dependencies()

        patch = mock.patch.object(
            self.worker, "_new_package", side_effect=dependencies_after_guard
        )
        patch.start()
        self.addCleanup(patch.stop)
        load_rpc = self.worker._load_rpc
        test = self

        class Client:
            def __init__(self, directory):
                test.rpc._private_directory(directory)
                test.clients.append(directory)

            def get_ddg(self, url, params=None, **limits):
                test.ddg_calls.append((url, dict(params), dict(limits)))
                test.rpc._ddg(
                    url,
                    params,
                    limits["max_bytes"],
                    limits["timeout"],
                    limits["max_redirects"],
                )
                if url == test.rpc.DDG_ROOT_URL:
                    return b'<html>vqd="synthetic-token"</html>'
                results = (
                    []
                    if params["q"] == test.fail_query
                    else [
                        {
                            "image": candidate,
                            "title": params["q"],
                            "url": "https://page.example.test/source",
                            "width": 1200,
                            "height": 1200,
                            "license": test.license,
                        }
                        for candidate in test.candidate_urls
                    ]
                )
                return json.dumps({"results": results}).encode()

            def get_photo(self, url, **limits):
                test.photo_calls.append((url, dict(limits)))
                test.rpc._photo(
                    url, limits["max_bytes"], limits["timeout"], limits["max_redirects"]
                )
                if url in test.denied:
                    raise test.rpc.RPCRemoteError("synthetic authoritative denial")
                return test.photo_body

        def rpc_with_fake_client():
            self.rpc = load_rpc()
            self.rpc.FileRPC = Client
            return self.rpc

        patch = mock.patch.object(
            self.worker, "_load_rpc", side_effect=rpc_with_fake_client
        )
        self.rpc_loader = patch.start()
        self.addCleanup(patch.stop)
        load_components = self.worker._load_components

        def instrument_components():
            self.app = load_components()
            actual = self.app.ImageSearcher

            class Factory:
                def __getattr__(self, name):
                    return getattr(actual, name)

                def __call__(self, *args, **kwargs):
                    test.constructor_calls.append(kwargs)
                    instance = actual(*args, **kwargs)

                    class Acquisition:
                        def get_images(self, prompts):
                            test.prompt_lists.append(list(prompts))
                            values = instance.get_images(prompts)
                            if test.omit_credits:
                                (test.output / "image_credits.txt").unlink(
                                    missing_ok=True
                                )
                            if test.omit_sidecar and values:
                                actual._sidecar_path(Path(values[0])).unlink()
                            return values[:-1] if test.drop_return else values

                    return Acquisition()

            self.app.ImageSearcher = Factory()
            validate = self.app.inventory.validate_inventory

            def checked_inventory(*args, **kwargs):
                self.events.append("validate")
                if self.fail_validation:
                    raise self.app.inventory.ResumeInventoryError(
                        "synthetic final validation failure"
                    )
                return validate(*args, **kwargs)

            self.app.inventory.validate_inventory = checked_inventory
            return self.app

        patch = mock.patch.object(
            self.worker, "_load_components", side_effect=instrument_components
        )
        patch.start()
        self.addCleanup(patch.stop)

    def _save_manifest(self):
        self.manifest.write_text(json.dumps(self.data), encoding="utf-8")

    def _drop_package(self):
        prefix = self.worker._APP
        for name in list(sys.modules):
            if name == prefix or name.startswith(prefix + "."):
                sys.modules.pop(name, None)

    def _trap(self, *args, **kwargs):
        self.forbidden.append("forbidden constructor or default transport")
        raise AssertionError("forbidden constructor or default transport")

    def _install_global_traps(self):
        requests = types.ModuleType("requests")
        requests.get = requests.post = requests.Session = self._trap
        ddgs = types.ModuleType("ddgs")
        ddgs.DDGS = self._trap
        image = types.ModuleType("PIL.Image")
        test = self

        class SyntheticImage:
            size = (1200, 1200)

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        def open_image(path):
            if not Path(path).read_bytes():
                raise ValueError("synthetic empty image")
            test.events.append("dimensions")
            return SyntheticImage()

        image.open = open_image
        pil = types.ModuleType("PIL")
        pil.Image = image
        patch = mock.patch.dict(
            sys.modules,
            {"requests": requests, "ddgs": ddgs, "PIL": pil, "PIL.Image": image},
        )
        patch.start()
        self.addCleanup(patch.stop)

    def _install_dependencies(self):
        prefix, test = self.worker._APP, self

        def install(name, **attributes):
            module = types.ModuleType(prefix + "." + name)
            module.__dict__.update(attributes)
            sys.modules[module.__name__] = module

        class Candidate:
            def __init__(self, result):
                self.url, self.title = result["image"], result["title"]
                self.provider, self.license = "ddg", result["license"]
                self.fit_evidence = "synthetic metadata fit"
                self.extra = {
                    "source_page": result["url"],
                    "creator": "synthetic fixture",
                    "license_url": "https://page.example.test/license",
                }

        def fit(prompt, title):
            test.events.append("fit")
            return (
                "synthetic metadata fit"
                if prompt == title and not test.fit_reject
                else None
            )

        def filter_results(prompt, results):
            return [
                Candidate(result) for result in results if fit(prompt, result["title"])
            ]

        def retry(prompt):
            test.events.append("existing retry hook")
            return None

        install(
            "image_sources",
            SourceCandidate=Candidate,
            filter_ddg_results=filter_results,
            fit_evidence=fit,
            retry_subject_query=retry,
            search_commons=self._trap,
            has_contradictory_number=lambda *args: False,
        )

        class ContentUnavailableError(RuntimeError):
            pass

        class SubjectUnavailableError(RuntimeError):
            pass

        def pixels(path, prompt):
            test.events.append("pixels")
            return "synthetic pixel rejection" if test.pixel_reject else None

        def subject(path, prompt):
            test.events.append("subject")
            return "synthetic subject rejection" if test.subject_reject else None

        install(
            "image_content",
            ContentUnavailableError=ContentUnavailableError,
            check_image_content=pixels,
        )
        install(
            "image_subject",
            SubjectUnavailableError=SubjectUnavailableError,
            check_subject=subject,
            is_broad_request=lambda prompt: test.broad,
            VERIFICATION_SCOPE="synthetic broad-class only",
        )

        def copy_processing(source, destination, *args):
            test.events.append("existing image processing")
            shutil.copyfile(source, destination)

        install("text_removal", remove_overlay_text=copy_processing)
        install("image_framing", frame_subject_file=copy_processing)

        def log(message, level):
            test.events.append("log")
            print("SYNTHETIC_PRIVATE_LOG " + message)
            print("SYNTHETIC_PRIVATE_LOG " + message, file=sys.stderr)

        install("logging_system", log=log)
        for name in (
            "script_generator",
            "tts",
            "model",
            "image_generator",
            "video_renderer",
        ):
            install(
                name,
                ScriptGenerator=self._trap,
                TTS=self._trap,
                Model=self._trap,
                ImageGenerator=self._trap,
                render=self._trap,
            )

    def _invoke(self, *, allow=False, manifest=None, output=None):
        # A new private module namespace models a fresh one-shot worker process.
        self._drop_package()
        result = self.worker.acquire(
            str(manifest or self.manifest),
            str(output or self.output),
            str(self.rpc_dir),
            allow_unverified=allow,
        )
        self.assertEqual(self.forbidden, [])
        return result

    def test_exact_queries_transcript_rpc_injection_and_existing_gates(self):
        before = {
            path: path.read_bytes()
            for path in (self.manifest, self.audio, self.transcript)
        }
        os.environ.update(
            IMAGE_PROVIDERS="commons",
            IMAGE_AI_FALLBACK="on",
            ALLOW_UNVERIFIED_IMAGES="1",
        )
        result = self._invoke()
        self.assertEqual(self.prompt_lists, [self.queries])
        self.assertEqual(
            [params["q"] for _, params, _ in self.ddg_calls],
            [query for query in self.queries for _ in range(2)],
        )
        self.assertEqual(
            [url for url, _, _ in self.ddg_calls],
            [self.rpc.DDG_ROOT_URL, self.rpc.DDG_IMAGES_URL] * len(self.queries),
        )
        self.assertEqual(len(self.constructor_calls), 1)
        constructor = self.constructor_calls[0]
        self.assertEqual(constructor["providers"], ("ddg",))
        self.assertIs(constructor["ai_fallback"], False)
        self.assertTrue(callable(constructor["download"]))
        self.assertIsInstance(constructor["search"], self.app.FixedDDGSearch)
        self.assertFalse(self.app.config.ALLOW_UNVERIFIED_IMAGES)
        self.assertTrue(self.app.config.IMAGE_TEXT_REMOVAL)
        self.assertTrue(self.app.config.SUBJECT_FRAMING)
        self.assertIn("pixels", self.events)
        self.assertIn("subject", self.events)
        self.assertIn("existing image processing", self.events)
        self.assertIn("validate", self.events)
        for path, raw in before.items():
            self.assertEqual(path.read_bytes(), raw)
        for _, limits in self.photo_calls:
            self.assertEqual(
                limits,
                {
                    "max_bytes": self.rpc.PHOTO_MAX_BYTES,
                    "timeout": self.rpc.PHOTO_MAX_TIMEOUT,
                    "max_redirects": self.rpc.PHOTO_MAX_REDIRECTS,
                },
            )
        self.assertEqual(set(result), {"inventory", "images", "credits", "reused"})
        self.assertFalse(result["reused"])
        self.assertEqual(len(result["images"]), len(self.queries))
        self.assertEqual(Path(result["inventory"]), self.output / "inventory.json")

    def test_unknown_rights_opt_in_is_explicit_and_credits_remain_unknown(self):
        self.license = ""
        os.environ["ALLOW_UNVERIFIED_IMAGES"] = "1"
        with self.assertRaises(RuntimeError):
            self._invoke()
        self.assertEqual(self.photo_calls, [])
        self.assertFalse((self.output / "inventory.json").exists())
        result = self._invoke(allow=True)
        credits = Path(result["credits"]).read_text()
        self.assertIn("PUBLICATION REVIEW REQUIRED:", credits)
        self.assertIn("UNVERIFIED - reuse rights unknown", credits)
        self.assertTrue(self.app.config.ALLOW_UNVERIFIED_IMAGES)
        for image in result["images"]:
            sidecar = json.loads(Path(image).with_suffix(".credits.json").read_text())
            self.assertEqual(sidecar["license"], "")
            self.assertIs(sidecar["license_verified"], False)
            self.assertIs(sidecar["unverified_opt_in"], True)
        calls = len(self.photo_calls), len(self.ddg_calls), len(self.clients)
        with self.assertRaises(ValueError):
            self._invoke(allow=False)
        self.assertEqual(
            (len(self.photo_calls), len(self.ddg_calls), len(self.clients)), calls
        )

    def test_existing_inventory_reuse_performs_zero_acquisition_calls(self):
        first = self._invoke()
        raw = Path(first["inventory"]).read_bytes()
        counts = (
            len(self.clients),
            len(self.ddg_calls),
            len(self.photo_calls),
            len(self.constructor_calls),
        )
        result = self._invoke()
        self.assertTrue(result["reused"])
        self.assertEqual(result["images"], first["images"])
        self.assertEqual(Path(result["inventory"]).read_bytes(), raw)
        self.assertEqual(
            (
                len(self.clients),
                len(self.ddg_calls),
                len(self.photo_calls),
                len(self.constructor_calls),
            ),
            counts,
        )

    def test_changed_manifest_image_sidecar_or_inputs_refused_without_fetch(self):
        result = self._invoke()
        image = Path(result["images"][0])
        sidecar = image.with_suffix(".credits.json")
        originals = {
            path: path.read_bytes()
            for path in (self.manifest, self.audio, self.transcript, image, sidecar)
        }
        counts = len(self.clients), len(self.ddg_calls), len(self.photo_calls)
        for changed in originals:
            with self.subTest(changed=changed.name):
                if changed == self.manifest:
                    data = json.loads(originals[changed])
                    data["queries"] = ["different saved query"]
                    data["queries_sha256"] = _sha(
                        json.dumps(data["queries"], separators=(",", ":")).encode()
                    )
                    changed.write_text(json.dumps(data))
                else:
                    changed.write_bytes(originals[changed] + b" changed")
                try:
                    with self.assertRaises(ValueError):
                        self._invoke()
                    self.assertEqual(
                        (len(self.clients), len(self.ddg_calls), len(self.photo_calls)),
                        counts,
                    )
                finally:
                    changed.write_bytes(originals[changed])

    def test_pixel_subject_fit_and_specificity_rejections_are_not_bypassed(self):
        for attribute in ("pixel_reject", "subject_reject", "fit_reject"):
            with self.subTest(gate=attribute):
                setattr(self, attribute, True)
                try:
                    with self.assertRaises(RuntimeError):
                        self._invoke()
                    self.assertFalse((self.output / "inventory.json").exists())
                finally:
                    setattr(self, attribute, False)
        self.broad = False
        with self.assertRaises(RuntimeError):
            self._invoke()
        self.assertEqual(self.forbidden, [])

    def test_missing_image_sidecar_credits_or_full_count_never_publishes_inventory(
        self,
    ):
        self.fail_query = self.queries[-1]
        with self.assertRaises(RuntimeError):
            self._invoke()
        self.assertFalse((self.output / "inventory.json").exists())
        self.fail_query = None
        for attribute in ("drop_return", "omit_sidecar", "omit_credits"):
            with self.subTest(fault=attribute):
                setattr(self, attribute, True)
                try:
                    with self.assertRaises((ValueError, OSError)):
                        self._invoke()
                    self.assertFalse((self.output / "inventory.json").exists())
                finally:
                    setattr(self, attribute, False)

    def test_final_validation_failure_removes_newly_published_inventory(self):
        self.fail_validation = True
        with self.assertRaises(ValueError):
            self._invoke()
        self.assertIn("validate", self.events)
        self.assertFalse((self.output / "inventory.json").exists())

    def test_denied_photo_advances_candidates_without_allowlist_expansion(self):
        denied = "https://denied.example.test/untrusted/a.jpg"
        self.candidate_urls = [denied, "https://images.example.test/safe/a.jpg"]
        self.denied.add(denied)
        self._invoke()
        self.assertEqual(
            [url for url, _ in self.photo_calls],
            self.candidate_urls * len(self.queries),
        )
        self.assertTrue(
            all("allowlist" not in limits for _, limits in self.photo_calls)
        )

    def test_symlink_traversal_and_unsafe_cache_paths_refused(self):
        alias = self.root / "output-link"
        alias.symlink_to(self.output, target_is_directory=True)
        for output in (alias, self.output / ".." / "output"):
            with (
                self.subTest(output=str(output)),
                self.assertRaises(self.worker.AcquireError),
            ):
                self._invoke(output=output)
        outside = self.root / "untouched"
        outside.write_bytes(b"DO NOT WRITE")
        credits = self.output / "image_credits.txt"
        credits.symlink_to(outside)
        with self.assertRaises(self.worker.AcquireError):
            self._invoke()
        credits.unlink()
        (self.output / ".resume-tmp").symlink_to(self.root, target_is_directory=True)
        with self.assertRaises(self.worker.AcquireError):
            self._invoke()
        self.assertEqual(outside.read_bytes(), b"DO NOT WRITE")
        self.assertEqual(self.clients, [])

    def test_download_is_exact_bounded_and_rejects_unsafe_targets(self):
        cache = self.output / "private-cache"
        cache.mkdir(mode=0o700)
        rpc = _direct(_PRODUCTION / "resume_rpc.py", "_download_contract_rpc")
        client = mock.Mock()
        client.get_photo.return_value = b"EXACT SYNTHETIC BYTES"
        target = cache / "download.jpg"
        with self.worker._downloader(rpc, client, cache) as download:
            self.assertTrue(download("https://images.example.test/safe/a.jpg", target))
            self.assertEqual(target.read_bytes(), b"EXACT SYNTHETIC BYTES")
            outside = self.root / "outside.jpg"
            outside.write_bytes(b"UNCHANGED")
            link = cache / "link.jpg"
            link.symlink_to(outside)
            fifo = cache / "fifo.jpg"
            os.mkfifo(fifo, 0o600)
            hardlink = cache / "hardlink.jpg"
            os.link(outside, hardlink)
            calls = client.get_photo.call_count
            for unsafe in (outside, link, fifo, hardlink, cache / ".." / "escape.jpg"):
                with (
                    self.subTest(unsafe=str(unsafe)),
                    self.assertRaises(self.worker.AcquireError),
                ):
                    download("https://images.example.test/safe/a.jpg", unsafe)
            self.assertEqual(client.get_photo.call_count, calls)
            self.assertEqual(outside.read_bytes(), b"UNCHANGED")
            for body in (
                b"",
                bytearray(b"not exact bytes"),
                b"x" * (rpc.PHOTO_MAX_BYTES + 1),
            ):
                client.get_photo.return_value = body
                with self.assertRaises(self.worker.AcquireError):
                    download("https://images.example.test/safe/a.jpg", target)
                self.assertEqual(target.read_bytes(), b"EXACT SYNTHETIC BYTES")
            for error in (rpc.RPCRequestError, rpc.RPCRemoteError):
                client.get_photo.side_effect = error("synthetic denial")
                self.assertFalse(
                    download("https://images.example.test/safe/a.jpg", target)
                )
            client.get_photo.side_effect = rpc.RPCProtocolError(
                "synthetic fatal protocol failure"
            )
            with self.assertRaises(rpc.RPCProtocolError):
                download("https://images.example.test/safe/a.jpg", target)
            self.assertEqual(target.read_bytes(), b"EXACT SYNTHETIC BYTES")

    def test_guard_precedes_all_application_imports(self):
        with mock.patch.object(self.worker, "_load_inputs") as inputs:
            with mock.patch.dict(os.environ, {"AUTOSHORTS_NETWORK_GUARD": "0"}):
                with self.assertRaises(self.worker.AcquireError):
                    self._invoke()
            with mock.patch.dict(sys.modules, {"dotenv": types.ModuleType("dotenv")}):
                with self.assertRaises(self.worker.AcquireError):
                    self._invoke()
            with mock.patch.dict(
                os.environ,
                {"AUTOSHORTS_PARENT_NETNS": os.readlink("/proc/self/ns/net")},
            ):
                with self.assertRaises(self.worker.AcquireError):
                    self._invoke()
            with mock.patch.dict(
                os.environ,
                {"AUTOSHORTS_BOUNDARY_READY": str(self.root / "missing-ready")},
            ):
                with self.assertRaises(self.worker.AcquireError):
                    self._invoke()
            read_text = Path.read_text

            def external_interface(path, *args, **kwargs):
                if str(path) == "/proc/net/dev":
                    return "header\nheader\neth0: 0 0\n"
                return read_text(path, *args, **kwargs)

            with mock.patch.object(Path, "read_text", new=external_interface):
                with self.assertRaises(self.worker.AcquireError):
                    self._invoke()
            inputs.assert_not_called()
        self.rpc_loader.assert_not_called()
        self.assertEqual(self.constructor_calls, [])

    def test_boundary_refusal_does_not_change_ambient_umask(self):
        original = os.umask(0o002)
        try:
            with mock.patch.dict(os.environ, {"AUTOSHORTS_NETWORK_GUARD": "0"}):
                with mock.patch.object(
                    self.worker.os, "umask", wraps=os.umask
                ) as change_umask:
                    with self.assertRaises(self.worker.AcquireError):
                        self._invoke()
                    change_umask.assert_not_called()
            self.assertEqual(os.umask(0o002), 0o002)
            self.assertFalse((self.output / ".resume-tmp").exists())
        finally:
            os.umask(original)

    def test_main_returns_only_minimal_json_and_non_sensitive_errors(self):
        output, error = io.StringIO(), io.StringIO()
        args = [
            "--manifest",
            str(self.manifest),
            "--output-dir",
            str(self.output),
            "--rpc-dir",
            str(self.rpc_dir),
        ]
        with redirect_stdout(output), redirect_stderr(error):
            status = self.worker.main(args)
        self.assertEqual(status, 0)
        self.assertEqual(error.getvalue(), "")
        self.assertEqual(
            set(json.loads(output.getvalue())),
            {"inventory", "images", "credits", "reused"},
        )
        self.assertNotIn("SYNTHETIC_PRIVATE_LOG", output.getvalue())
        self.assertNotIn("Saved transcript", output.getvalue())
        for invalid in (
            args + ["--allow-unverified=1"],
            args + ["--unknown=SYNTHETIC_SECRET"],
        ):
            output, error = io.StringIO(), io.StringIO()
            with redirect_stdout(output), redirect_stderr(error):
                self.assertNotEqual(self.worker.main(invalid), 0)
            self.assertEqual(output.getvalue(), "")
            self.assertNotIn("SYNTHETIC_SECRET", error.getvalue())


_LOCAL_PUBLIC_HTTP = r"""
import json
import os
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]


def _record(op, **fields):
    with (_ROOT / "transport.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"op": op, "pid": os.getpid(), **fields}) + "\n")


def validate_photo_allowlist(entries):
    assert entries == [("images.example.test", "/safe/")]
    return entries


class BoundedPublicClient:
    def get_ddg(self, url, params, **limits):
        assert params["q"] == json.loads((_ROOT / "manifest.json").read_bytes())["queries"][0]
        _record("ddg", url=url, params=params)
        if url == "https://duckduckgo.com/":
            return b'<html>vqd="synthetic-integration-token"</html>'
        assert url == "https://duckduckgo.com/i.js"
        return (_ROOT / "ddg.json").read_bytes()

    def get_photo(self, url, entries, **limits):
        validate_photo_allowlist(entries)
        assert url == "https://images.example.test/safe/mountain-landscape.png"
        _record("photo", url=url)
        return (_ROOT / "photo.png").read_bytes()
"""

_REAL_PROBE = r"""
import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location(
    "_real_acquire_probe", _ROOT / "src" / "autoshorts" / "resume_acquire.py")
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)
worker._require_boundary()
os.environ.update(OUTPUT_DIR=str(_ROOT / "output"), TMPDIR=str(_ROOT / "probe-tmp"),
                  TMP=str(_ROOT / "probe-tmp"), TEMP=str(_ROOT / "probe-tmp"),
                  TEMP_DIR_PREFIX="autoshorts_", IMAGE_PROVIDERS="ddg",
                  IMAGE_AI_FALLBACK="off", ALLOW_UNVERIFIED_IMAGES="1")
tempfile.tempdir = None
try:
    with worker._quiet_application():
        inputs = worker._load_inputs(_ROOT / "manifest.json")
        app = worker._load_components()
        sources = sys.modules[worker._APP + ".image_sources"]
        content = sys.modules[worker._APP + ".image_content"]
        subject = sys.modules[worker._APP + ".image_subject"]
        query = inputs.queries[0]
        results = json.loads((_ROOT / "ddg.json").read_bytes())["results"]
        assert subject.is_broad_request(query), "fixture is not an allowed real broad query"
        fit = sources.fit_evidence(query, results[0]["title"])
        assert fit and sources.filter_ddg_results(query, results), "real metadata gates rejected fixture"
        try:
            reason = content.check_image_content(_ROOT / "photo.png", query)
        except content.ContentUnavailableError as exc:
            outcome = {"unavailable": "real pixel dependency unavailable: " + str(exc)[:400]}
        else:
            assert reason is None, "real pixel gate rejected synthetic image"
            assert app.ImageSearcher._content_rejection(_ROOT / "photo.png", query) is None
            if app.config.IMAGE_TEXT_REMOVAL:
                __import__(worker._APP + ".text_removal", fromlist=["remove_overlay_text"])
            if app.config.SUBJECT_FRAMING:
                __import__(worker._APP + ".image_framing", fromlist=["frame_subject_file"])
            outcome = {"fit": fit}
except ModuleNotFoundError as exc:
    outcome = {"unavailable": "real Python dependency missing: " + (exc.name or type(exc).__name__)}
print(json.dumps(outcome))
"""


_REAL_UMASK_WORKER = r"""
import argparse
import importlib.util
import json
import os
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location(
    "_real_umask_acquire", _ROOT / "src" / "autoshorts" / "resume_acquire.py")
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)
worker._require_boundary()
parser = argparse.ArgumentParser(allow_abbrev=False)
parser.add_argument("--initial-umask", required=True)
parser.add_argument("--output-dir", required=True)
parser.add_argument("--rpc-dir", required=True)
args = parser.parse_args()
original = os.umask(int(args.initial_umask, 8))
try:
    result = worker.acquire(str(_ROOT / "manifest.json"), args.output_dir, args.rpc_dir,
                            allow_unverified=True)
    private_umask = os.umask(0o077)
finally:
    os.umask(original)
print(json.dumps({"result": result, "umask": private_umask}))
"""


class AcquireIntegrationTests(unittest.TestCase):
    __test__ = _INNER_RUN

    def test_real_acquisition_artifacts_are_private_from_permissive_umasks(self):
        required = (
            "resume_acquire.py",
            "resume_supervisor.py",
            "resume_boundary.py",
            "resume_rpc.py",
            "resume_fetch.py",
            "modules/resume_inputs.py",
            "modules/resume_inventory.py",
            "modules/resume_ddg.py",
            "modules/image_searcher.py",
            "modules/config.py",
            "modules/image_sources.py",
            "modules/image_content.py",
            "modules/image_subject.py",
        )
        missing = [name for name in required if not (_PRODUCTION / name).is_file()]
        if missing:
            self.skipTest("real integration source unavailable: " + ", ".join(missing))
        try:
            from PIL import Image, ImageDraw
        except ImportError as exc:
            self.skipTest(
                "real Pillow unavailable: " + (exc.name or type(exc).__name__)
            )
        self.assertTrue(getattr(Image, "__file__", None), "real Pillow is required")
        with tempfile.TemporaryDirectory(prefix="real-acquire-umask-") as temporary:
            root = Path(temporary).resolve()
            source_root = root / "src"
            package = source_root / "autoshorts"
            package.mkdir(parents=True)
            for source in sorted(_PRODUCTION.rglob("*.py")):
                self.assertFalse(
                    source.is_symlink(), "integration sources must not be symlinks"
                )
                relative = source.relative_to(_PRODUCTION)
                if relative.as_posix() == "modules/public_http.py":
                    continue
                destination = package / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, destination)
            (package / "modules" / "public_http.py").write_text(
                _LOCAL_PUBLIC_HTTP, encoding="utf-8"
            )
            (root / "output").mkdir(mode=0o700)
            (root / "probe-tmp").mkdir(mode=0o700)
            image = Image.new("RGB", (1200, 1600), (110, 180, 235))
            try:
                drawing = ImageDraw.Draw(image)
                drawing.polygon(
                    [(0, 1080), (430, 350), (980, 1080)], fill=(105, 120, 135)
                )
                drawing.polygon(
                    [(500, 1120), (980, 520), (1200, 1120)], fill=(85, 110, 125)
                )
                drawing.rectangle((0, 1120, 1200, 1600), fill=(65, 125, 65))
                image.save(root / "photo.png", format="PNG")
            finally:
                image.close()
            query = "mountain"
            audio, transcript = root / "saved.wav", root / "transcript.txt"
            audio.write_bytes(b"SYNTHETIC SAVED AUDIO - NEVER DECODED OR REGENERATED")
            transcript.write_text(
                "Synthetic saved landscape transcript.\n", encoding="utf-8"
            )
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "schema": 1,
                        "audio_path": audio.name,
                        "transcript_path": transcript.name,
                        "audio_sha256": _sha(audio.read_bytes()),
                        "transcript_sha256": _sha(transcript.read_bytes()),
                        "queries": [query],
                        "queries_sha256": _sha(
                            json.dumps([query], separators=(",", ":")).encode()
                        ),
                        "transcript_provenance": "synthetic saved integration fixture",
                    }
                ),
                encoding="utf-8",
            )
            (root / "ddg.json").write_text(
                json.dumps(
                    {
                        "results": [
                            {
                                "image": "https://images.example.test/safe/mountain-landscape.png",
                                "title": "Mountain with green hills and blue sky",
                                "url": "https://photos.example.test/gallery/mountain-landscape",
                                "width": 1200,
                                "height": 1600,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            saved = {
                path: (path.read_bytes(), path.lstat().st_mode)
                for path in (manifest, audio, transcript)
            }
            probe = source_root / "probe.py"
            probe.write_text(_REAL_PROBE, encoding="utf-8")
            supervisor = _direct(
                package / "resume_supervisor.py", "_real_umask_supervisor"
            )
            checked = supervisor.boundary.run_offline(
                str(probe), [], timeout=45, source_root=str(source_root)
            )
            availability = json.loads(checked.stdout)
            if "unavailable" in availability:
                self.skipTest(availability["unavailable"])
            worker = source_root / "umask-worker.py"
            worker.write_text(_REAL_UMASK_WORKER, encoding="utf-8")
            for initial in ("0002", "0000"):
                with self.subTest(initial_umask=initial):
                    output = root / ("output-" + initial)
                    output.mkdir(mode=0o700)
                    completed = supervisor.run_with_public_fetch(
                        str(worker),
                        ["--initial-umask", initial, "--output-dir", str(output)],
                        timeout=90,
                        source_root=str(source_root),
                        photo_allowlist=["https://images.example.test/safe/"],
                    )
                    state = json.loads(completed.stdout)
                    self.assertEqual((completed.returncode, completed.stderr), (0, ""))
                    self.assertEqual(state["umask"], 0o077)
                    acquired = state["result"]
                    self.assertIs(acquired["reused"], False)
                    self.assertEqual(len(acquired["images"]), 1)
                    photo = Path(acquired["images"][0])
                    expected = [
                        Path(acquired["inventory"]),
                        Path(acquired["credits"]),
                        photo,
                        photo.with_suffix(".credits.json"),
                    ]
                    for path in expected:
                        self.assertTrue(path.is_relative_to(output))
                        self.assertTrue(stat.S_ISREG(path.lstat().st_mode))
                    with Image.open(photo) as decoded:
                        decoded.verify()
                    self.assertIn(
                        b"PUBLICATION REVIEW REQUIRED:",
                        Path(acquired["credits"]).read_bytes(),
                    )
                    for path in output.rglob("*"):
                        info = path.lstat()
                        self.assertTrue(
                            stat.S_ISREG(info.st_mode) or stat.S_ISDIR(info.st_mode)
                        )
                        self.assertEqual(info.st_uid, os.getuid())
                        self.assertEqual(info.st_mode & 0o077, 0, str(path))
                        if stat.S_ISREG(info.st_mode):
                            self.assertEqual(info.st_nlink, 1)
                    for path, (raw, mode) in saved.items():
                        self.assertEqual(
                            (path.read_bytes(), path.lstat().st_mode), (raw, mode)
                        )
            entries = [
                json.loads(line)
                for line in (root / "transport.jsonl").read_bytes().splitlines()
            ]
            self.assertEqual(
                [entry["op"] for entry in entries], ["ddg", "ddg", "photo"] * 2
            )

    def test_real_gates_rpc_supervisor_fetch_inventory_and_zero_helper_reuse(self):
        """No AcquireTests fixtures, fake app modules, gate patches or source edits."""
        required = (
            "resume_acquire.py",
            "resume_supervisor.py",
            "resume_boundary.py",
            "resume_rpc.py",
            "resume_fetch.py",
            "modules/resume_inputs.py",
            "modules/resume_inventory.py",
            "modules/resume_ddg.py",
            "modules/image_searcher.py",
            "modules/config.py",
            "modules/image_sources.py",
            "modules/image_content.py",
            "modules/image_subject.py",
        )
        missing = [name for name in required if not (_PRODUCTION / name).is_file()]
        if missing:
            self.skipTest("real integration source unavailable: " + ", ".join(missing))
        try:
            from PIL import Image, ImageDraw
        except ImportError as exc:
            self.skipTest(
                "real Pillow unavailable: " + (exc.name or type(exc).__name__)
            )
        self.assertTrue(
            getattr(Image, "__file__", None),
            "real Pillow, not a unit substitute, is required",
        )
        originals = {name: (_PRODUCTION / name).read_bytes() for name in required}
        with tempfile.TemporaryDirectory(
            prefix="real-acquire-integration-"
        ) as temporary:
            root = Path(temporary).resolve()
            source_root = root / "src"
            package = source_root / "autoshorts"
            package.mkdir(parents=True)
            # Copy public Python sources only: no .env, credentials, models or private assets.
            for source in sorted(_PRODUCTION.rglob("*.py")):
                self.assertFalse(
                    source.is_symlink(), "integration sources must not be symlinks"
                )
                relative = source.relative_to(_PRODUCTION)
                if relative.as_posix() == "modules/public_http.py":
                    continue
                destination = package / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, destination)
            (package / "modules" / "public_http.py").write_text(
                _LOCAL_PUBLIC_HTTP, encoding="utf-8"
            )
            output = root / "output"
            output.mkdir(mode=0o700)
            (root / "probe-tmp").mkdir(mode=0o700)
            image = Image.new("RGB", (1200, 1600), (110, 180, 235))
            try:
                drawing = ImageDraw.Draw(image)
                drawing.polygon(
                    [(0, 1080), (430, 350), (980, 1080)], fill=(105, 120, 135)
                )
                drawing.polygon(
                    [(500, 1120), (980, 520), (1200, 1120)], fill=(85, 110, 125)
                )
                drawing.rectangle((0, 1120, 1200, 1600), fill=(65, 125, 65))
                image.save(root / "photo.png", format="PNG")
            finally:
                image.close()
            query = "mountain"
            title = "Mountain with green hills and blue sky"
            audio, transcript = root / "saved.wav", root / "transcript.txt"
            audio.write_bytes(b"SYNTHETIC SAVED AUDIO - NEVER DECODED OR REGENERATED")
            transcript.write_text(
                "Synthetic saved landscape transcript.\n", encoding="utf-8"
            )
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "schema": 1,
                        "audio_path": audio.name,
                        "transcript_path": transcript.name,
                        "audio_sha256": _sha(audio.read_bytes()),
                        "transcript_sha256": _sha(transcript.read_bytes()),
                        "queries": [query],
                        "queries_sha256": _sha(
                            json.dumps([query], separators=(",", ":")).encode()
                        ),
                        "transcript_provenance": "synthetic saved integration fixture",
                    }
                ),
                encoding="utf-8",
            )
            (root / "ddg.json").write_text(
                json.dumps(
                    {
                        "results": [
                            {
                                "image": "https://images.example.test/safe/mountain-landscape.png",
                                "title": title,
                                "url": "https://photos.example.test/gallery/mountain-landscape",
                                "width": 1200,
                                "height": 1600,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            saved_inputs = {
                path: path.read_bytes() for path in (manifest, audio, transcript)
            }
            probe = source_root / "probe.py"
            probe.write_text(_REAL_PROBE, encoding="utf-8")
            supervisor = _direct(
                package / "resume_supervisor.py", "_real_acquire_supervisor"
            )
            checked = supervisor.boundary.run_offline(
                str(probe), [], timeout=45, source_root=str(source_root)
            )
            availability = json.loads(checked.stdout)
            if "unavailable" in availability:
                self.skipTest(availability["unavailable"])
            helper = str(package / "resume_fetch.py")
            launches, active = [], [True]

            def observe(event, arguments):
                if active[0] and event == "subprocess.Popen":
                    command = arguments[1]
                    if isinstance(command, (list, tuple)) and helper in command:
                        launches.append(tuple(command))

            # Observation only: supervisor, RPC, fetch and every acceptance gate stay unchanged.
            sys.addaudithook(observe)
            try:
                args = [
                    "--manifest",
                    str(manifest),
                    "--output-dir",
                    str(output),
                    "--allow-unverified",
                ]
                first = supervisor.run_with_public_fetch(
                    str(package / "resume_acquire.py"),
                    args,
                    timeout=90,
                    source_root=str(source_root),
                    photo_allowlist=["https://images.example.test/safe/"],
                )
                acquired = json.loads(first.stdout)
                self.assertEqual(first.returncode, 0)
                self.assertEqual(first.stderr, "")
                self.assertEqual(
                    set(acquired), {"inventory", "images", "credits", "reused"}
                )
                self.assertIs(acquired["reused"], False)
                self.assertEqual(len(acquired["images"]), 1)
                inventory = Path(acquired["inventory"])
                inventory_bytes = inventory.read_bytes()
                data = json.loads(inventory_bytes)
                self.assertEqual(data["schema"], 1)
                self.assertEqual(data["inputs"]["queries"], [query])
                self.assertEqual(len(data["items"]), 1)
                item = data["items"][0]
                photo, sidecar = (
                    output / item["image_path"],
                    output / item["sidecar_path"],
                )
                self.assertEqual(str(photo), acquired["images"][0])
                self.assertEqual(_sha(photo.read_bytes()), item["image_sha256"])
                self.assertEqual(_sha(sidecar.read_bytes()), item["sidecar_sha256"])
                with Image.open(photo) as decoded:
                    decoded.verify()
                record = json.loads(sidecar.read_bytes())
                self.assertEqual((record["kind"], record["provider"]), ("web", "ddg"))
                self.assertEqual(record["fit"], availability["fit"])
                self.assertIs(record["license_verified"], False)
                self.assertIs(record["unverified_opt_in"], True)
                self.assertEqual(
                    record["pictured_subject_check"],
                    "not-run (ALLOW_UNVERIFIED_IMAGES opt-in)",
                )
                credits_path = Path(acquired["credits"])
                credits_bytes = credits_path.read_bytes()
                credits = credits_bytes.decode("utf-8")
                self.assertIn("PUBLICATION REVIEW REQUIRED:", credits)
                self.assertIn("UNVERIFIED - reuse rights unknown", credits)
                self.assertIn(photo.name, credits)
                log = root / "transport.jsonl"
                transport_bytes = log.read_bytes()
                entries = [json.loads(line) for line in transport_bytes.splitlines()]
                self.assertEqual(
                    [entry["op"] for entry in entries], ["ddg", "ddg", "photo"]
                )
                self.assertEqual(len({entry["pid"] for entry in entries}), 3)
                self.assertEqual(len(launches), 3)
                reused = supervisor.run_with_public_fetch(
                    str(package / "resume_acquire.py"),
                    args,
                    timeout=45,
                    source_root=str(source_root),
                    photo_allowlist=["https://images.example.test/safe/"],
                )
                result = json.loads(reused.stdout)
                self.assertEqual((reused.returncode, reused.stderr), (0, ""))
                self.assertEqual(result, {**acquired, "reused": True})
                self.assertEqual(inventory.read_bytes(), inventory_bytes)
                self.assertEqual(credits_path.read_bytes(), credits_bytes)
                self.assertEqual(log.read_bytes(), transport_bytes)
                self.assertEqual(
                    len(launches), 3, "reuse must launch zero additional helpers"
                )
                for path, raw in saved_inputs.items():
                    self.assertEqual(path.read_bytes(), raw)
            finally:
                active[0] = False
        for name, raw in originals.items():
            self.assertEqual((_PRODUCTION / name).read_bytes(), raw)


def _run_outer_suite():
    """Run the full original 13-test standalone suite through its real launcher."""
    boundary = _direct(_PRODUCTION / "resume_boundary.py", "_acquire_outer_boundary")
    try:
        result = boundary.run_offline(
            str(Path(__file__).resolve()),
            [_INNER],
            timeout=240,
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
        raise AssertionError(f"Guarded acquire suite exited with {result.returncode}")
    return result.returncode


def test_guarded_acquire_suite(capsys):
    """One outer aggregate runs all original acquire units and integration."""
    # This disables only output capture, never a network or qualification guard.
    with capsys.disabled():
        _run_outer_suite()


test_guarded_acquire_suite.__test__ = not _INNER_RUN


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
            "Run without arguments or discover inside the existing qualified outer boundary"
        )
    return _run_outer_suite()


if __name__ == "__main__":
    raise SystemExit(_main())
