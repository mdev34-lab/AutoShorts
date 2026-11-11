"""Synthetic tests as SOURCE ONLY: not imported, compiled or executed by author.

Run this file explicitly, with the three reviewed siblings installed. Its
launcher first uses the qualified boundary to put the ENTIRE suite in an
outer offline namespace. Tests then exercise real nested offline workers and
real one-request fetch helpers with a local fake public_http.py. Namespace
failure is an error, not a skip or a weaker fallback. No live HTTP is used.
Repository siblings live in src/autoshorts/; standalone sibling layouts also
work. Unqualified discovery exposes one aggregate that launches all 20 original
inner tests through that same standalone launcher and forwards its report.
Qualified discovery runs the original class after the unchanged real gate;
one outer item is not evidence of 20 separately collected pytest tests.
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
from pathlib import Path
from unittest import mock

_HERE = Path(__file__).resolve().parent
_PROJECT = _HERE if (_HERE / "resume_supervisor.py").is_file() else _HERE.parent
# The outer launch root must contain both tests/ and src/, not just the helpers.
_PRODUCTION = _PROJECT / "src" / "autoshorts"
if not (_PRODUCTION / "resume_supervisor.py").is_file():
    _PRODUCTION = _PROJECT
_ALLOWLIST = ["https://images.example.test/safe/"]
_INNER = "--inside-supervisor-tests"
# A routing hint only: inner execution must still pass the real gate below.
_INNER_RUN = _INNER in sys.argv[1:] or os.environ.get("AUTOSHORTS_NETWORK_GUARD") == "1"


def _direct(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(name, None)
    return module


def _require_boundary():
    # Original inner-suite gate: no original test executes outside this boundary.
    current = os.readlink("/proc/self/ns/net")
    marker = os.environ.get("AUTOSHORTS_BOUNDARY_READY", "")
    if (
        os.environ.get("AUTOSHORTS_NETWORK_GUARD") != "1"
        or not marker
        or current == os.environ.get("AUTOSHORTS_PARENT_NETNS")
        or Path(marker).read_text(encoding="ascii") != current
    ):
        raise RuntimeError(
            "Tests require an established qualified outer offline boundary"
        )
    devices = {
        line.split(":", 1)[0].strip()
        for line in Path("/proc/net/dev").read_text().splitlines()[2:]
        if ":" in line
    }
    for filename, column, skip in (("route", 0, 1), ("ipv6_route", -1, 0)):
        path = Path("/proc/net") / filename
        if path.exists():
            devices.update(
                line.split()[column]
                for line in path.read_text().splitlines()[skip:]
                if line.split()
            )
    if devices - {"lo"}:
        raise RuntimeError(
            "All tests require an outer namespace with no external interface/route"
        )


def setUpModule(module=None):
    # Pytest supplies the module; unittest calls without it. Only the outer
    # pytest aggregate defers qualification to its real offline child.
    if module is None or _INNER_RUN:
        _require_boundary()


setup_module = setUpModule


_FAKE_TRANSPORT = r"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

_LOG = Path(__file__).resolve().parents[2] / "calls.jsonl"


def _record(op, **fields):
    home = Path(os.environ["HOME"])
    descriptors = []
    for number in os.listdir("/proc/self/fd"):
        try:
            descriptors.append(os.readlink("/proc/self/fd/" + number))
        except FileNotFoundError:
            pass
    entry = dict(op=op, pid=os.getpid(), pgid=os.getpgrp(), env=dict(os.environ),
                 namespace=os.readlink("/proc/self/ns/net"), isolated=sys.flags.isolated,
                 paths=sys.path, app_imported="autoshorts" in sys.modules,
                 dotenv_imported="dotenv" in sys.modules, descriptors=descriptors,
                 home_mode=home.stat().st_mode & 0o777,
                 tmp_mode=Path(os.environ["TMPDIR"]).stat().st_mode & 0o777,
                 home_files=sorted(p.name for p in home.iterdir()),
                 cwd_files=sorted(p.name for p in Path.cwd().iterdir()), **fields)
    with _LOG.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(entry) + "\n")


def _child():
    process = subprocess.Popen([sys.executable, "-I", "-c", "import time; time.sleep(60)"],
                               stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL, close_fds=True)
    _record("child", child=process.pid)


def validate_photo_allowlist(entries):
    if entries != [("images.example.test", "/safe/")]:
        raise ValueError("synthetic allowlist mismatch")
    _record("validate")
    return entries


class BoundedPublicClient:
    def get_ddg(self, url, params, **limits):
        _record("ddg", url=url, params=params, limits=limits)
        query = params["q"]
        if query == "error":
            raise ValueError("synthetic public failure")
        if query in ("hang", "orphan"):
            _child()
            if query == "hang":
                time.sleep(60)
        if query == "overflow":
            sys.stdout.buffer.write(b"x" * (1024 * 1024 + 1))
            sys.stdout.buffer.flush()
        if query in ("drop-request", "change-request"):
            path = Path(sys.argv[sys.argv.index("--request") + 1])
            if query == "drop-request":
                path.unlink()
            else:
                data = json.loads(path.read_bytes())
                data["params"]["q"] = "changed"
                path.write_text(json.dumps(data), encoding="utf-8")
        return b"ddg:" + query.encode("ascii")

    def get_photo(self, url, entries, **limits):
        if entries != [("images.example.test", "/safe/")]:
            raise ValueError("synthetic allowlist mismatch")
        _record("photo", url=url, entries=entries, limits=limits)
        return b"photo"
"""

_WORKER_PREFIX = r"""
import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(allow_abbrev=False)
parser.add_argument("--rpc-dir", required=True)
options = parser.parse_args()
spec = importlib.util.spec_from_file_location("_worker_rpc", str(ROOT / "resume_rpc.py"))
rpc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rpc)
client = rpc.FileRPC(options.rpc_dir)
"""

_SPAWN = r"""
child = subprocess.Popen([sys.executable, "-I", "-c", "import time; time.sleep(60)"],
                         stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL, close_fds=True)
(ROOT / "worker-pids.json").write_text(json.dumps([os.getpid(), child.pid]))
"""


def _payload(**changes):
    value = {
        "version": 1,
        "id": 1,
        "op": "ddg",
        "url": "https://duckduckgo.com/",
        "params": {"q": "one"},
        "max_bytes": 100,
        "timeout": 1,
        "max_redirects": 0,
    }
    value.update(changes)
    return value


def _publish_source(payload):
    # A complete local file is published atomically, just like FileRPC.
    return (
        f"raw = {json.dumps(payload).encode('utf-8')!r}\n"
        "slot = Path(options.rpc_dir)\n"
        "(slot / 'manual.tmp').write_bytes(raw)\n"
        "os.replace(slot / 'manual.tmp', slot / 'request.json')\n"
    )


class SupervisorTests(unittest.TestCase):
    __test__ = _INNER_RUN

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="synthetic-supervisor-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        for leaf in (
            "resume_supervisor.py",
            "resume_boundary.py",
            "resume_rpc.py",
            "resume_fetch.py",
        ):
            shutil.copyfile(_PRODUCTION / leaf, self.root / leaf)
        modules = self.root / "autoshorts" / "modules"
        modules.mkdir(parents=True)
        for package in (modules.parent, modules):
            (package / "__init__.py").write_text(
                "raise AssertionError('application import forbidden')\n"
            )
        (modules / "public_http.py").write_text(
            textwrap.dedent(_FAKE_TRANSPORT), encoding="utf-8"
        )
        self.supervisor = _direct(
            self.root / "resume_supervisor.py", "_synthetic_supervisor"
        )
        self.boundary = self.supervisor.boundary
        self.outer_namespace = os.readlink("/proc/self/ns/net")

    def _worker(self, body):
        path = self.root / "worker.py"
        path.write_text(
            textwrap.dedent(_WORKER_PREFIX) + "\n" + textwrap.dedent(body),
            encoding="utf-8",
        )
        return str(path)

    def _run(self, body, *, timeout=10):
        return self.supervisor.run_with_public_fetch(
            self._worker(body),
            [],
            timeout=timeout,
            source_root=str(self.root),
            photo_allowlist=list(_ALLOWLIST),
        )

    def _events(self):
        path = self.root / "calls.jsonl"
        return (
            []
            if not path.exists()
            else [json.loads(line) for line in path.read_text().splitlines()]
        )

    def _dead(self, pid):
        deadline = time.monotonic() + 1
        while True:
            try:
                raw = Path(f"/proc/{pid}/stat").read_text()
            except FileNotFoundError:
                return
            state = raw[raw.rfind(")") + 2 :].split()[0]
            if state == "Z":  # Killed grandchildren may await the namespace's reaper.
                return
            if time.monotonic() >= deadline:
                self.fail(f"Process {pid} is still live")
            time.sleep(0.01)

    def _worker_group_dead(self):
        for pid in json.loads((self.root / "worker-pids.json").read_text()):
            self._dead(pid)

    def _helper_group_dead(self):
        for event in self._events():
            self._dead(event["pid"])
            if event["op"] == "child":
                self._dead(event["child"])

    def test_real_nested_filerpc_two_ddg_and_photo(self):
        result = self._run("""
            import dotenv
            assert dotenv.AUTOSHORTS_OFFLINE_STUB
            assert os.environ["AUTOSHORTS_NETWORK_GUARD"] == "1"
            assert os.getpid() == os.getpgrp()
            first = client.get_ddg(rpc.DDG_ROOT_URL, {"q": "one"})
            second = client.get_ddg(rpc.DDG_IMAGES_URL,
                {"o": "json", "q": "two", "l": "br-pt", "vqd": "token1", "p": "1", "ct": "AT"})
            photo = client.get_photo("https://images.example.test/safe/photo.jpg")
            print(json.dumps([first.decode(), second.decode(), photo.decode(),
                              os.readlink("/proc/self/ns/net")]))
        """)
        values = json.loads(result.stdout)
        self.assertEqual(values[:3], ["ddg:one", "ddg:two", "photo"])
        self.assertNotEqual(values[3], self.outer_namespace)
        self.assertIsInstance(result, subprocess.CompletedProcess)
        self.assertEqual((result.returncode, result.stderr), (0, ""))
        events = self._events()
        calls = [event for event in events if event["op"] in ("ddg", "photo")]
        self.assertEqual([event["op"] for event in calls], ["ddg", "ddg", "photo"])
        self.assertEqual(len({event["pid"] for event in calls}), 3)
        self.assertEqual([event["op"] for event in events].count("validate"), 1)
        for event in events:
            self.assertEqual(event["namespace"], self.outer_namespace)
            self.assertEqual(event["pid"], event["pgid"])
            self.assertEqual(event["isolated"], 1)
            self.assertFalse(event["app_imported"])
            self.assertFalse(event["dotenv_imported"])
            self.assertNotIn(str(self.root), event["paths"])

    def test_helper_environment_is_independent_and_no_fds_inherited(self):
        original_home = self.root / "synthetic-original-home"
        original_home.mkdir()
        for leaf in (".env", ".netrc"):
            (original_home / leaf).write_text("SYNTHETIC_ONLY=not-a-credential\n")
        sentinel = original_home / "inherited-fd"
        with sentinel.open("wb") as descriptor:
            os.set_inheritable(descriptor.fileno(), True)
            synthetic_env = {
                "HOME": str(original_home),
                "API_KEY": "synthetic-only",
                "OPENAI_API_KEY": "synthetic-only",
                "AWS_ACCESS_KEY_ID": "synthetic-only",
                "HTTP_PROXY": "http://synthetic.invalid",
                "HTTPS_PROXY": "http://synthetic.invalid",
                "ALL_PROXY": "http://synthetic.invalid",
                "http_proxy": "http://synthetic.invalid",
                "PYTHONPATH": str(original_home),
                "DOTENV_CONFIG_PATH": str(original_home / ".env"),
                "UNRELATED_PARENT_ONLY": "synthetic-only",
            }
            with mock.patch.dict(os.environ, synthetic_env):
                self._run('client.get_ddg(rpc.DDG_ROOT_URL, {"q": "env"})')
        event = self._events()[0]
        env = event["env"]
        for key in synthetic_env:
            if key != "HOME":
                self.assertNotIn(key, env)
        self.assertNotEqual(env["HOME"], str(original_home))
        self.assertEqual(env["PATH"], "/usr/bin:/bin")
        self.assertEqual(env["PYTHON_DOTENV_DISABLED"], "1")
        self.assertEqual(env["TMPDIR"], env["TMP"])
        self.assertEqual(env["TMPDIR"], env["TEMP"])
        self.assertEqual(event["home_mode"] & 0o077, 0)
        self.assertEqual(event["tmp_mode"] & 0o077, 0)
        self.assertEqual((event["home_files"], event["cwd_files"]), ([], []))
        self.assertNotIn(str(sentinel), event["descriptors"])
        self.assertNotIn("AUTOSHORTS_NETWORK_GUARD", env)

    def test_invalid_requests_are_rejected_before_helper_or_executor(self):
        malformed = [
            _payload(op="exec"),
            _payload(path="/synthetic/elsewhere"),
            _payload(env={"API_KEY": "synthetic-only"}),
            _payload(allowlist=_ALLOWLIST),
            _payload(executable="/bin/false"),
            _payload(id=0),
            _payload(id=True),
            _payload(id=2),
            _payload(id=257),
            _payload(id=1.0),
            _payload(version=True),
            _payload(url="https://duckduckgo.com/../elsewhere"),
            {
                "version": 1,
                "id": 1,
                "op": "photo",
                "url": "https://images.example.test/safe/../secret",
                "max_bytes": 100,
                "timeout": 1,
                "max_redirects": 0,
            },
        ]
        for payload in malformed:
            with (
                self.subTest(payload=payload),
                mock.patch.object(self.boundary, "_launch") as launch,
            ):
                with self.assertRaises(self.supervisor.SupervisorProtocolError):
                    self._run(_publish_source(payload) + "time.sleep(60)\n")
                launch.assert_not_called()
        self.assertEqual(self._events(), [])

    def test_nonallowlisted_photo_denial_precedes_transport_import_and_next_photo_works(
        self,
    ):
        transport = self.root / "autoshorts" / "modules" / "public_http.py"
        transport.write_text(
            "from pathlib import Path\n"
            "with (Path(__file__).resolve().parents[2] / 'transport-imports.jsonl').open('a') as stream:\n"
            "    stream.write('import\\n')\n" + textwrap.dedent(_FAKE_TRANSPORT),
            encoding="utf-8",
        )
        real_launch = self.boundary._launch
        launches = []

        def observe_launch(command, *args, **kwargs):
            request = json.loads(
                Path(command[command.index("--request") + 1]).read_bytes()
            )
            outcome = real_launch(command, *args, **kwargs)
            launches.append((request["id"], request["op"], outcome))
            return outcome

        with mock.patch.object(self.boundary, "_launch", side_effect=observe_launch):
            result = self._run("""
                try:
                    client.get_photo("https://other.example.test/safe/a.jpg")
                except rpc.RPCRemoteError as exc:
                    assert "outside the supervisor photo_allowlist" in str(exc)
                else:
                    raise AssertionError("nonallowlisted candidate was not denied")
                assert client._next_id == 2 and not client._broken
                assert not (ROOT / "transport-imports.jsonl").exists()
                assert not (ROOT / "calls.jsonl").exists()
                assert list(Path(options.rpc_dir).iterdir()) == []
                print(client.get_photo("https://images.example.test/safe/a.jpg").decode())
                assert client._next_id == 3 and not client._broken
                assert list(Path(options.rpc_dir).iterdir()) == []
            """)
        self.assertEqual(
            (result.returncode, result.stdout, result.stderr), (0, "photo\n", "")
        )
        self.assertEqual(launches, [(1, "photo", (False, 1)), (2, "photo", (False, 0))])
        self.assertEqual(
            (self.root / "transport-imports.jsonl").read_text(), "import\n"
        )
        events = self._events()
        self.assertEqual([event["op"] for event in events], ["validate", "photo"])
        self.assertEqual(events[-1]["url"], "https://images.example.test/safe/a.jpg")
        self.assertEqual(events[-1]["entries"], [["images.example.test", "/safe/"]])

    def test_nonallowlisted_photo_with_bad_schema_or_id_is_still_fatal(self):
        photo = {
            "version": 1,
            "id": 1,
            "op": "photo",
            "url": "https://other.example.test/safe/a.jpg",
            "max_bytes": 100,
            "timeout": 1,
            "max_redirects": 0,
        }
        for changes in (
            {"id": 0},
            {"id": True},
            {"id": 2},
            {"params": {}},
            {"allowlist": _ALLOWLIST},
        ):
            payload = {**photo, **changes}
            with (
                self.subTest(payload=payload),
                mock.patch.object(self.boundary, "_launch") as launch,
            ):
                with self.assertRaises(self.supervisor.SupervisorProtocolError):
                    self._run(_publish_source(payload) + "time.sleep(60)\n")
                launch.assert_not_called()
        self.assertEqual(self._events(), [])

    def test_nonallowlisted_photo_helper_exit_is_not_bypassed(self):
        real_launch = self.boundary._launch
        for fake_code in (0, 2):

            def wrong_exit(*args, code=fake_code, **kwargs):
                timed_out, actual_code = real_launch(*args, **kwargs)
                self.assertFalse(timed_out)
                self.assertEqual(actual_code, 1)
                return timed_out, code

            with (
                self.subTest(code=fake_code),
                mock.patch.object(self.boundary, "_launch", side_effect=wrong_exit),
            ):
                with self.assertRaises(self.supervisor.SupervisorHelperError):
                    self._run(
                        _SPAWN
                        + 'client.get_photo("https://other.example.test/safe/a.jpg")\n'
                    )
                self._worker_group_dead()
        self.assertEqual(self._events(), [])

    def test_helper_typed_error_reaches_worker_and_next_id_works(self):
        result = self._run("""
            try:
                client.get_ddg(rpc.DDG_ROOT_URL, {"q": "error"})
            except rpc.RPCRemoteError as exc:
                assert "synthetic public failure" in str(exc)
                print("remote-error")
            else:
                raise AssertionError("typed error was not delivered")
            print(client.get_ddg(rpc.DDG_ROOT_URL, {"q": "again"}).decode())
        """)
        self.assertEqual(result.stdout, "remote-error\nddg:again\n")
        self.assertEqual(
            [event["params"]["q"] for event in self._events()], ["error", "again"]
        )

    def test_unchanged_response_pending_is_not_dispatched_again(self):
        result = self._run(
            _publish_source(_payload())
            + """
while not (slot / "response.json").exists():
    time.sleep(0.01)
time.sleep(0.3)
(slot / "response.json").unlink()
(slot / "request.json").unlink()
print("drained")
"""
        )
        self.assertEqual(result.stdout, "drained\n")
        self.assertEqual(len(self._events()), 1)

    def test_replayed_and_changed_same_ids_abort_without_second_fetch(self):
        for query in ("one", "changed"):
            with self.subTest(query=query):
                before = len(self._events())
                body = 'client.get_ddg(rpc.DDG_ROOT_URL, {"q": "one"})\n'
                body += (
                    _publish_source(_payload(params={"q": query})) + "time.sleep(60)\n"
                )
                with self.assertRaises(self.supervisor.SupervisorProtocolError):
                    self._run(body)
                self.assertEqual(len(self._events()), before + 1)

    def test_request_missing_or_changed_before_response_acceptance_aborts(self):
        for query in ("drop-request", "change-request"):
            with self.subTest(query=query):
                with self.assertRaises(self.supervisor.SupervisorProtocolError):
                    self._run(f'client.get_ddg(rpc.DDG_ROOT_URL, {{"q": {query!r}}})')

    def test_fifo_request_and_symlink_never_block_or_reach_helper(self):
        bodies = [
            'os.mkfifo(Path(options.rpc_dir) / "request.json", 0o600)\ntime.sleep(60)\n',
            '(ROOT / "synthetic-target").write_text("{}")\n'
            'os.symlink(ROOT / "synthetic-target", Path(options.rpc_dir) / "request.json")\n'
            "time.sleep(60)\n",
        ]
        for body in bodies:
            with (
                self.subTest(body=body),
                mock.patch.object(self.boundary, "_launch") as launch,
            ):
                started = time.monotonic()
                with self.assertRaises(self.supervisor.SupervisorProtocolError):
                    self._run(body, timeout=5)
                self.assertLess(time.monotonic() - started, 4)
                launch.assert_not_called()

    def test_per_helper_timeout_kills_both_process_groups(self):
        with mock.patch.object(self.supervisor, "_HELPER_SECONDS", 1.0):
            with self.assertRaises(self.supervisor.SupervisorHelperTimeoutError):
                self._run(_SPAWN + 'client.get_ddg(rpc.DDG_ROOT_URL, {"q": "hang"})\n')
        self.assertTrue(any(event["op"] == "child" for event in self._events()))
        self._helper_group_dead()
        self._worker_group_dead()

    def test_whole_deadline_caps_a_helper_and_kills_both_groups(self):
        started = time.monotonic()
        with self.assertRaises(self.boundary.BoundaryTimeoutError):
            self._run(
                _SPAWN + 'client.get_ddg(rpc.DDG_ROOT_URL, {"q": "hang"})\n', timeout=2
            )
        self.assertLess(time.monotonic() - started, 4)
        self.assertTrue(any(event["op"] == "child" for event in self._events()))
        self._helper_group_dead()
        self._worker_group_dead()

    def test_successful_helper_also_kills_its_leftover_group(self):
        self.assertEqual(
            self._run(
                'print(client.get_ddg(rpc.DDG_ROOT_URL, {"q": "orphan"}).decode())'
            ).stdout,
            "ddg:orphan\n",
        )
        self.assertTrue(any(event["op"] == "child" for event in self._events()))
        self._helper_group_dead()

    def test_offline_timeout_kills_group(self):
        with self.assertRaises(self.boundary.BoundaryTimeoutError):
            self._run(_SPAWN + "time.sleep(60)\n", timeout=1.5)
        self._worker_group_dead()

    def test_offline_failure_kills_group_before_reaping_leader(self):
        with self.assertRaises(self.boundary.BoundaryFailureError) as caught:
            self._run(_SPAWN + "raise SystemExit(7)\n")
        self.assertEqual(caught.exception.returncode, 7)
        self._worker_group_dead()

    def test_unqualified_namespace_fails_closed_before_worker(self):
        def no_unshare(bootstrap, worker, arguments):
            return [sys.executable, "-I", str(bootstrap), str(worker), *arguments]

        with mock.patch.object(self.boundary, "_command", side_effect=no_unshare):
            with self.assertRaises(self.boundary.BoundaryUnavailableError) as caught:
                self._run('(ROOT / "worker-started").write_text("must not happen")')
        self.assertEqual(caught.exception.returncode, self.boundary.GUARD_EXIT_CODE)
        self.assertIn(
            "not running in a separate network namespace", caught.exception.stderr
        )
        self.assertFalse((self.root / "worker-started").exists())
        self.assertEqual(self._events(), [])

    def test_ready_marker_mandatory_and_must_differ_from_current_namespace(self):
        for action in (
            'Path(os.environ["AUTOSHORTS_BOUNDARY_READY"]).unlink()',
            'Path(os.environ["AUTOSHORTS_BOUNDARY_READY"]).write_text('
            'os.environ["AUTOSHORTS_PARENT_NETNS"])',
        ):
            with self.subTest(action=action):
                with self.assertRaises(self.boundary.BoundaryUnavailableError):
                    self._run(action + '\nprint("pretend-ready")\n')
        self.assertEqual(self._events(), [])

    def test_output_overflow_rejected_and_diagnostics_bounded(self):
        for body in (
            'sys.stdout.write("x" * (1024 * 1024 + 1))',
            'sys.stderr.write("x" * (1024 * 1024 + 1))',
            'client.get_ddg(rpc.DDG_ROOT_URL, {"q": "overflow"})',
        ):
            with self.subTest(body=body):
                with self.assertRaises(self.boundary.BoundaryOutputError) as caught:
                    self._run(body)
                self.assertLessEqual(
                    len(caught.exception.stdout), self.boundary.DIAGNOSTIC_CHARS
                )
                self.assertLessEqual(
                    len(caught.exception.stderr), self.boundary.DIAGNOSTIC_CHARS
                )

    def test_helper_bad_exit_or_mismatched_error_code_aborts_and_cleans_worker(self):
        real_launch = self.boundary._launch
        for fake_code in (1, 2, 3, -9):

            def wrong_exit(*args, code=fake_code, **kwargs):
                timed_out, _ = real_launch(*args, **kwargs)
                return timed_out, code

            with (
                self.subTest(code=fake_code),
                mock.patch.object(self.boundary, "_launch", side_effect=wrong_exit),
            ):
                with self.assertRaises(self.supervisor.SupervisorHelperError):
                    self._run(
                        _SPAWN + 'client.get_ddg(rpc.DDG_ROOT_URL, {"q": "one"})\n'
                    )
                self._worker_group_dead()

    def test_exception_with_helper_alive_cleans_both_groups(self):
        real_launch, real_popen = self.boundary._launch, subprocess.Popen
        helpers = []

        def record_helper(*args, **kwargs):
            process = real_popen(*args, **kwargs)
            helpers.append(process.pid)
            return process

        def faulty_launch(*args, **kwargs):
            with mock.patch.object(
                self.boundary.subprocess, "Popen", side_effect=record_helper
            ):
                with mock.patch.object(
                    self.boundary.os,
                    "waitid",
                    side_effect=RuntimeError("synthetic launcher fault"),
                ):
                    return real_launch(*args, **kwargs)

        with mock.patch.object(self.boundary, "_launch", side_effect=faulty_launch):
            with self.assertRaisesRegex(RuntimeError, "synthetic launcher fault"):
                self._run(_SPAWN + 'client.get_ddg(rpc.DDG_ROOT_URL, {"q": "one"})\n')
        self.assertEqual(len(helpers), 1)
        self._dead(helpers[0])
        self._worker_group_dead()

    def test_input_types_limits_and_reserved_arguments_rejected_without_launch(self):
        worker = self._worker("pass\n")
        valid = dict(timeout=5, source_root=str(self.root), photo_allowlist=_ALLOWLIST)
        cases = [
            {"timeout": value}
            for value in (True, 0, -1, 7200.01, float("nan"), float("inf"), 10**1000)
        ] + [
            {"photo_allowlist": value}
            for value in (
                None,
                [],
                ["https://images.example.test/safe/?q=bad"],
                _ALLOWLIST * 2,
                ["https://images.example.test/safe/../"],
                ["https://127.0.0.1/safe/"],
            )
        ]
        with mock.patch.object(self.supervisor.subprocess, "Popen") as launch:
            for changes in cases:
                with (
                    self.subTest(changes=changes),
                    self.assertRaises(self.boundary.BoundaryInputError),
                ):
                    self.supervisor.run_with_public_fetch(
                        worker, [], **{**valid, **changes}
                    )
            for args in (
                (),
                [True],
                ["x\x00"],
                ["x" * 4097],
                ["x"] * 63,
                ["--"],
                ["--rpc-dir", "/synthetic"],
                ["--rpc-dir=/synthetic"],
                ["--rpc"],
            ):
                with (
                    self.subTest(args=args),
                    self.assertRaises(self.boundary.BoundaryInputError),
                ):
                    self.supervisor.run_with_public_fetch(worker, args, **valid)
            launch.assert_not_called()

    def test_helper_and_transport_must_be_canonical_fixed_files(self):
        helper = self.root / "resume_fetch.py"
        helper.rename(self.root / "saved-fetch.py")
        helper.symlink_to(self.root / "saved-fetch.py")
        with self.assertRaises(self.boundary.BoundaryInputError):
            self._run("pass\n")
        helper.unlink()
        shutil.copyfile(_PRODUCTION / "resume_fetch.py", helper)
        transport = self.root / "autoshorts" / "modules" / "public_http.py"
        transport.rename(self.root / "elsewhere.py")
        transport.symlink_to(self.root / "elsewhere.py")
        with self.assertRaises(self.boundary.BoundaryInputError):
            self._run("pass\n")
        self.assertEqual(self._events(), [])


def _run_outer_suite():
    """Run the full original 20-test standalone suite through its real launcher."""
    qualified_boundary = _direct(
        _PRODUCTION / "resume_boundary.py", "_outer_qualified_boundary"
    )
    try:
        result = qualified_boundary.run_offline(
            str(Path(__file__).resolve()),
            [_INNER],
            timeout=240,
            source_root=str(_PROJECT),
        )
    except qualified_boundary.BoundaryError as exc:
        # Preserve all diagnostics the unchanged boundary supplies, then fail.
        sys.stdout.write(exc.stdout)
        sys.stderr.write(exc.stderr)
        raise
    sys.stdout.write(result.stdout)
    sys.stderr.write(result.stderr)
    if result.returncode != 0:
        raise AssertionError(
            f"Guarded supervisor suite exited with {result.returncode}"
        )
    return result.returncode


def test_guarded_supervisor_suite(capsys):
    """One outer aggregate runs all original supervisor and nested-worker tests."""
    # This disables only output capture, never a network or qualification guard.
    with capsys.disabled():
        _run_outer_suite()


test_guarded_supervisor_suite.__test__ = not _INNER_RUN


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
            "Run without arguments; the suite requires its outer offline launcher"
        )
    return _run_outer_suite()


if __name__ == "__main__":
    raise SystemExit(_main())
