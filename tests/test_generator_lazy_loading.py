"""The generator registry stays lazy and now exposes only the explainer."""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent


def _run(code: str) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT / "src")
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        env=env,
        cwd=REPO_ROOT,
    )


class TestLazyGeneratorRegistry:
    def test_registry_exposes_only_explainer(self):
        proc = _run(
            """
            from autoshorts.generators import VIDEO_TYPES
            assert list(VIDEO_TYPES) == ["explainer"]
            assert len(VIDEO_TYPES) == 1
            gen = VIDEO_TYPES["explainer"]
            assert gen.__name__ == "ExplainerGenerator"
            try:
                VIDEO_TYPES["product"]
            except KeyError:
                pass
            else:
                raise AssertionError("removed product generator must not resolve")
            print("OK")
            """
        )
        assert proc.returncode == 0, proc.stderr
        assert "OK" in proc.stdout

    def test_backward_compatible_class_import(self):
        proc = _run(
            """
            from autoshorts.generators import ExplainerGenerator
            assert ExplainerGenerator.__name__ == "ExplainerGenerator"
            try:
                from autoshorts.generators import ProductGenerator
            except (ImportError, AttributeError):
                pass
            else:
                raise AssertionError("removed product generator must not import")
            print("OK")
            """
        )
        assert proc.returncode == 0, proc.stderr
        assert "OK" in proc.stdout
