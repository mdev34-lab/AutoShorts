"""Shopee/product pipeline must not load when only the explainer runs."""

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
    def test_explainer_access_does_not_import_product(self):
        proc = _run(
            """
            import sys
            from autoshorts.generators import VIDEO_TYPES
            assert "autoshorts.generators.product" not in sys.modules
            assert not any(m.startswith("autoshorts.modules.product") for m in sys.modules)
            gen = VIDEO_TYPES["explainer"]
            assert gen.__name__ == "ExplainerGenerator"
            assert "autoshorts.generators.product" not in sys.modules
            assert not any(m.startswith("autoshorts.modules.product") for m in sys.modules)
            print("OK")
            """
        )
        assert proc.returncode == 0, proc.stderr
        assert "OK" in proc.stdout

    def test_product_access_still_works(self):
        proc = _run(
            """
            import sys
            from autoshorts.generators import VIDEO_TYPES
            gen = VIDEO_TYPES["product"]
            assert gen.__name__ == "ProductGenerator"
            assert "autoshorts.generators.product" in sys.modules
            assert "product" in VIDEO_TYPES and len(VIDEO_TYPES) == 2
            print("OK")
            """
        )
        assert proc.returncode == 0, proc.stderr
        assert "OK" in proc.stdout

    def test_backward_compatible_class_import(self):
        proc = _run(
            """
            from autoshorts.generators import ExplainerGenerator, ProductGenerator
            assert ExplainerGenerator.__name__ == "ExplainerGenerator"
            assert ProductGenerator.__name__ == "ProductGenerator"
            print("OK")
            """
        )
        assert proc.returncode == 0, proc.stderr
        assert "OK" in proc.stdout
