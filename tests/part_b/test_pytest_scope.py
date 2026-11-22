"""Plain `pytest` from the repo root must collect only tests/ (Steven's re-check of ec34f61).

Part A's checkout has third_party/pytorch3d with its own test suite; without testpaths, plain pytest at
the repo root would collect it (and anything else named test_*.py outside tests/).
"""

import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CONFIGS = ("pytest.ini", "pyproject.toml", "setup.cfg", "tox.ini")


def test_plain_pytest_collects_only_tests_dir(tmp_path):
    for name in CONFIGS:
        if (REPO / name).exists():
            shutil.copy(REPO / name, tmp_path / name)
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_inside.py").write_text("def test_inside():\n    pass\n")
    stray = tmp_path / "third_party" / "pytorch3d" / "tests"
    stray.mkdir(parents=True)
    (stray / "test_stray.py").write_text("def test_stray():\n    pass\n")
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "test_script_like.py").write_text("def test_script_like():\n    pass\n")

    r = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider"],
                       cwd=tmp_path, capture_output=True, text=True)
    assert "test_inside" in r.stdout, r.stdout + r.stderr
    assert "test_stray" not in r.stdout and "test_script_like" not in r.stdout, r.stdout
