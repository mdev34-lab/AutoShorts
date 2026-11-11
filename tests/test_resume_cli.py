"""SOURCE ONLY: not imported, compiled, linted or executed by the author.

Dispatch unit tests replace only the launch functions/loader, not any boundary
implementation used to run a worker. They prove argument and result handling,
not namespace isolation or successful acquisition/rendering. All original test
execution requires the same qualified outer boundary as the earlier suites.
Unqualified discovery exposes one aggregate that launches all 9 original inner
tests with the same standalone offline launcher and forwards its report.
Qualified discovery runs the original class after the unchanged real gate;
one outer item is not evidence of 9 separately collected pytest tests.
No live HTTP, providers, private assets, generator, TTS or native renderer is
called here.
"""

from __future__ import annotations

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
_INNER = "--inside-resume-cli-tests"
_ALLOWLIST = ["https://images.example.test/safe/"]
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
    _direct(_PRODUCTION / "resume_render.py", "_cli_suite_guard")._require_boundary()


def setUpModule(module=None):
    # Pytest supplies the module; unittest calls without it. Only the outer
    # pytest aggregate defers qualification to its real offline child.
    if module is None or _INNER_RUN:
        _require_boundary()


setup_module = setUpModule


class CLITests(unittest.TestCase):
    __test__ = _INNER_RUN

    def setUp(self):
        self.cli = _direct(_PRODUCTION / "resume_cli.py", "_cli_dispatch_tests")
        self.temporary = tempfile.TemporaryDirectory(prefix="synthetic-resume-cli-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.acquired, self.rendered = self.root / "acquired", self.root / "rendered"
        self.acquired.mkdir(mode=0o700)
        self.rendered.mkdir(mode=0o700)
        self.manifest = self.root / "manifest.json"
        self.manifest.write_text(
            '{"queries":["https://untrusted.example.test/not-authority/"]}',
            encoding="utf-8",
        )
        self.inventory = self.acquired / "inventory.json"
        self.inventory.write_bytes(
            b"SYNTHETIC DISPATCH INVENTORY - NOT PARSED IN PARENT"
        )
        self.output = self.rendered / "saved.mp4"
        self.acquire_payload = {
            "inventory": str(self.inventory),
            "images": [str(self.acquired / "cache" / "mountain.jpg")],
            "credits": str(self.acquired / "image_credits.txt"),
            "reused": False,
        }
        self.render_payload = {
            "output": str(self.output),
            "inventory": str(self.inventory),
            "credits": str(self.rendered / "image_credits.txt"),
            "audio_sha256": "a" * 64,
            "transcript_sha256": "b" * 64,
        }
        self.online = mock.Mock(return_value=self._reply(self.acquire_payload))
        self.offline = mock.Mock(return_value=self._reply(self.render_payload))

        def launcher(name):
            if name == "resume_supervisor.py":
                return types.SimpleNamespace(run_with_public_fetch=self.online)
            if name == "resume_boundary.py":
                return types.SimpleNamespace(run_offline=self.offline)
            raise AssertionError("unapproved launcher requested")

        self.loader_patch = mock.patch.object(
            self.cli, "_load_local", side_effect=launcher
        )
        self.loader = self.loader_patch.start()
        self.addCleanup(self.loader_patch.stop)

    @staticmethod
    def _reply(payload, *, code=0, stderr=""):
        return types.SimpleNamespace(
            returncode=code, stdout=json.dumps(payload), stderr=stderr
        )

    def _args(self, command="acquire"):
        args = [command, "--manifest", str(self.manifest), "--timeout", "10.5"]
        if command == "acquire":
            return args + [
                "--output-dir",
                str(self.acquired),
                "--photo-allowlist",
                _ALLOWLIST[0],
            ]
        return args + ["--inventory", str(self.inventory), "--output", str(self.output)]

    @staticmethod
    def _replace(args, flag, value):
        values = list(args)
        values[values.index(flag) + 1] = str(value)
        return values

    def _invoke(self, args):
        output, error = io.StringIO(), io.StringIO()
        with redirect_stdout(output), redirect_stderr(error):
            code = self.cli.main(args)
        return code, output.getvalue(), error.getvalue()

    def test_acquire_dispatch_uses_only_fixed_supervisor_and_explicit_authority(self):
        second = "https://other.example.test/photos/"
        args = self._args() + ["--photo-allowlist", second, "--allow-unverified"]
        saved = self.manifest.read_bytes()
        code, output, error = self._invoke(args)
        self.assertEqual((code, error), (0, ""))
        self.assertEqual(json.loads(output), self.acquire_payload)
        self.loader.assert_called_once_with("resume_supervisor.py")
        self.online.assert_called_once_with(
            str(_PRODUCTION / "resume_acquire.py"),
            [
                "--manifest",
                str(self.manifest),
                "--output-dir",
                str(self.acquired),
                "--allow-unverified",
            ],
            timeout=10.5,
            source_root=str(_PRODUCTION.parent),
            photo_allowlist=[_ALLOWLIST[0], second],
        )
        self.offline.assert_not_called()
        self.assertEqual(self.manifest.read_bytes(), saved)

    def test_render_dispatch_always_uses_full_fixed_offline_boundary(self):
        code, output, error = self._invoke(
            self._args("render") + ["--allow-unverified"]
        )
        self.assertEqual((code, error), (0, ""))
        self.assertEqual(json.loads(output), self.render_payload)
        self.loader.assert_called_once_with("resume_boundary.py")
        self.offline.assert_called_once_with(
            str(_PRODUCTION / "resume_render.py"),
            [
                "--manifest",
                str(self.manifest),
                "--inventory",
                str(self.inventory),
                "--output",
                str(self.output),
                "--allow-unverified",
            ],
            timeout=10.5,
            source_root=str(_PRODUCTION.parent),
        )
        self.online.assert_not_called()

    def test_unknown_rights_opt_in_is_not_inherited_or_forwarded_by_default(self):
        with mock.patch.dict(
            os.environ,
            {"ALLOW_UNVERIFIED_IMAGES": "1", "SYNTHETIC_PARENT_ONLY": "SECRET"},
        ):
            for command, runner in (("acquire", self.online), ("render", self.offline)):
                with self.subTest(command=command):
                    self.assertEqual(self._invoke(self._args(command))[0], 0)
                    self.assertNotIn("--allow-unverified", runner.call_args.args[1])
                    self.assertNotIn("env", runner.call_args.kwargs)
                    self.assertNotIn("env_overrides", runner.call_args.kwargs)
                    self.assertNotIn("rpc_dir", runner.call_args.kwargs)

    def test_unselectable_controls_bad_arguments_and_timeouts_never_dispatch(self):
        cases = [[], ["unknown"], self._args() + ["--allow-unverified=1"]]
        for flag in (
            "--worker",
            "--helper",
            "--transport",
            "--source-root",
            "--env",
            "--rpc-dir",
            "--man",
        ):
            cases.append(self._args() + [flag, "SYNTHETIC_SECRET"])
        cases.append(self._args("render") + ["--no-boundary"])
        for timeout in (
            "0",
            "-1",
            "nan",
            "inf",
            "7200.01",
            "1e9999",
            "SYNTHETIC_SECRET",
        ):
            cases.append(self._replace(self._args(), "--timeout", timeout))
        for args in cases:
            with self.subTest(args=args):
                code, output, error = self._invoke(args)
                self.assertEqual(code, 2)
                self.assertEqual(output, "")
                self.assertNotIn("SYNTHETIC_SECRET", error)
        self.loader.assert_not_called()
        self.online.assert_not_called()
        self.offline.assert_not_called()

    def test_allowlist_is_explicit_bounded_https_prefixes_not_remote_metadata(self):
        invalid = [
            "http://images.example.test/safe/",
            "https://127.0.0.1/safe/",
            "https://USER@images.example.test/safe/",
            "https://images.example.test:443/safe/",
            "https://IMAGES.example.test/safe/",
            "https://images.example.test/safe/?x=1",
            "https://images.example.test/safe/#fragment",
            "https://images.example.test/safe/../",
            "https://images.example.test/safe/%2f/",
            "https://*.example.test/safe/",
            "https://images.example.test/no-trailing-slash",
            "https://images.example.test/\\safe/",
        ]
        cases = [
            self._replace(self._args(), "--photo-allowlist", value) for value in invalid
        ]
        cases.append(self._args() + ["--photo-allowlist", _ALLOWLIST[0]])
        cases.append(
            self._args()
            + [
                part
                for i in range(64)
                for part in ("--photo-allowlist", f"https://images.example.test/p{i}/")
            ]
        )
        for args in cases:
            with self.subTest(args=args):
                self.assertEqual(self._invoke(args)[0:2], (2, ""))
        self.loader.assert_not_called()

    def test_noncanonical_nonregular_and_existing_paths_are_refused(self):
        link = self.root / "manifest-link.json"
        link.symlink_to(self.manifest)
        fifo = self.root / "manifest-fifo.json"
        os.mkfifo(fifo, 0o600)
        dangling = self.rendered / "dangling.mp4"
        dangling.symlink_to(self.root / "missing")
        existing = self.rendered / "existing.mp4"
        existing.write_bytes(b"DO NOT REPLACE")
        cases = [
            self._replace(self._args(), "--manifest", value)
            for value in (
                link,
                fifo,
                self.root,
                self.root / "missing.json",
                "relative.json",
            )
        ]
        cases.extend(
            self._replace(self._args("render"), "--output", value)
            for value in (
                dangling,
                existing,
                self.root / "missing-parent" / "out.mp4",
                self.rendered / "out.mkv",
            )
        )
        cases.append(
            self._replace(
                self._args(),
                "--manifest",
                str(self.root) + "/../" + self.root.name + "/manifest.json",
            )
        )
        for args in cases:
            with self.subTest(args=args):
                self.assertEqual(self._invoke(args)[0:2], (2, ""))
        self.assertEqual(existing.read_bytes(), b"DO NOT REPLACE")
        self.assertTrue(dangling.is_symlink())
        self.loader.assert_not_called()

    def test_real_nonzero_and_signal_statuses_survive_without_private_diagnostics(self):
        for code, expected in ((7, 7), (-9, 137), (86, 86), (None, 1), (False, 1)):
            with self.subTest(code=code):
                self.online.return_value = types.SimpleNamespace(
                    returncode=code,
                    stdout="SYNTHETIC_SECRET_STDOUT",
                    stderr="SYNTHETIC_SECRET_STDERR",
                )
                status, output, error = self._invoke(self._args())
                self.assertEqual((status, output), (expected, ""))
                self.assertNotIn("SYNTHETIC_SECRET", error)
        for code, expected in ((11, 11), (-15, 143), (None, 1), (0, 1)):
            failure = RuntimeError("SYNTHETIC_SECRET_EXCEPTION")
            failure.returncode = code
            self.online.side_effect = failure
            with self.subTest(exception_code=code):
                status, output, error = self._invoke(self._args())
                self.assertEqual((status, output), (expected, ""))
                self.assertNotIn("SYNTHETIC_SECRET", error)

    def test_only_typed_minimal_worker_json_is_printed_on_success(self):
        good = json.dumps(self.acquire_payload)
        cases = [
            "SYNTHETIC_SECRET",
            "SYNTHETIC_SECRET\n" + good,
            good + good,
            json.dumps(
                {**self.acquire_payload, "private_transcript": "SYNTHETIC_SECRET"}
            ),
            good.replace('"reused": false', '"reused": false, "reused": true'),
            json.dumps({**self.acquire_payload, "images": ["relative.jpg"]}),
            json.dumps({**self.acquire_payload, "reused": 0}),
            "NaN",
            "x" * (self.cli._MAX_OUTPUT + 1),
        ]
        for raw in cases:
            self.online.return_value = types.SimpleNamespace(
                returncode=0, stdout=raw, stderr=""
            )
            with self.subTest(raw_prefix=raw[:60]):
                code, output, error = self._invoke(self._args())
                self.assertNotEqual(code, 0)
                self.assertEqual(output, "")
                self.assertNotIn("SYNTHETIC_SECRET", error)
        self.offline.return_value = self._reply(
            {**self.render_payload, "audio_sha256": "not-a-hash"}
        )
        self.assertNotEqual(self._invoke(self._args("render"))[0], 0)

        self.online.return_value = self._reply(
            self.acquire_payload, stderr="SYNTHETIC_SECRET_ACQUIRE_STDERR"
        )
        code, output, error = self._invoke(self._args())
        expected = (
            json.dumps(self.acquire_payload, sort_keys=True, separators=(",", ":"))
            + "\n"
        )
        self.assertEqual(code, 0)
        self.assertEqual(output, expected)
        self.assertEqual(error, "resume_cli: worker emitted diagnostics\n")
        self.assertNotIn("SYNTHETIC_SECRET", output + error)

        self.online.return_value = types.SimpleNamespace(
            returncode=0,
            stdout="SYNTHETIC_SECRET_MALFORMED",
            stderr="SYNTHETIC_SECRET_MALFORMED_STDERR",
        )
        code, output, error = self._invoke(self._args())
        self.assertNotEqual(code, 0)
        self.assertEqual(output, "")
        self.assertEqual(error, "resume_cli: operation failed\n")
        self.assertNotIn("SYNTHETIC_SECRET", output + error)

    def test_render_success_with_diagnostics_is_validated_and_sanitized(self):
        self.offline.return_value = self._reply(
            self.render_payload, stderr="SYNTHETIC_SECRET_RENDER_STDERR"
        )
        code, output, error = self._invoke(self._args("render"))
        expected = (
            json.dumps(self.render_payload, sort_keys=True, separators=(",", ":"))
            + "\n"
        )
        self.assertEqual(code, 0)
        self.assertEqual(output, expected)
        self.assertEqual(error, "resume_cli: worker emitted diagnostics\n")
        self.assertNotIn("SYNTHETIC_SECRET", output + error)

        self.offline.return_value = self._reply(
            self.render_payload, code=17, stderr="SYNTHETIC_SECRET_NONZERO_STDERR"
        )
        code, output, error = self._invoke(self._args("render"))
        self.assertEqual((code, output), (17, ""))
        self.assertEqual(error, "resume_cli: worker failed\n")
        self.assertNotIn("SYNTHETIC_SECRET", output + error)

    def test_direct_loader_accepts_only_canonical_fixed_stdlib_siblings(self):
        self.loader_patch.stop()
        package = self.root / "src" / "autoshorts"
        package.mkdir(parents=True)
        boundary = package / "resume_boundary.py"
        boundary.write_text("VALUE = 'synthetic fixed sibling'\n", encoding="utf-8")
        with mock.patch.object(self.cli, "_HERE", package):
            module = self.cli._load_local("resume_boundary.py")
            self.assertEqual(module.VALUE, "synthetic fixed sibling")
            for name in ("resume_acquire.py", "../outside.py", "arbitrary.py"):
                with self.subTest(name=name), self.assertRaises(self.cli.CLIError):
                    self.cli._load_local(name)
            saved = self.root / "saved-boundary.py"
            boundary.rename(saved)
            boundary.symlink_to(saved)
            with self.assertRaises(self.cli.CLIError):
                self.cli._load_local("resume_boundary.py")


def _run_outer_suite():
    """Run the full original 9-test standalone suite through its real launcher."""
    boundary = _direct(_PRODUCTION / "resume_boundary.py", "_cli_outer_boundary")
    try:
        result = boundary.run_offline(
            str(Path(__file__).resolve()),
            [_INNER],
            timeout=120,
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
        raise AssertionError(f"Guarded CLI suite exited with {result.returncode}")
    return result.returncode


def test_guarded_cli_suite(capsys):
    """One outer aggregate runs all original CLI dispatch tests."""
    # This disables only output capture, never a network or qualification guard.
    with capsys.disabled():
        _run_outer_suite()


test_guarded_cli_suite.__test__ = not _INNER_RUN


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
