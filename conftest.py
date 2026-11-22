"""pytest-only configuration (Part B). Not used by `python -m unittest`, which is Part A's runner.

Part A's test_encoder_provider.py imports upstream tracker code, which needs the real pytorch3d. Part B's
Mac environment does not have it (scripts/pytorch3d_shim.py is a stand-in for crop placement only), so
under pytest that one test is skipped when the real package cannot be found. Part A's test file is not
edited. The search ignores sys.modules, so the stand-in never counts as the real package.
"""

import importlib.machinery

import pytest

NEEDS_REAL_PYTORCH3D = {
    "tests/test_encoder_provider.py::ProviderChecks::test_no_hand_clears_history_without_encoder_call",
}


def _real_pytorch3d_available() -> bool:
    return importlib.machinery.PathFinder.find_spec("pytorch3d") is not None


def pytest_collection_modifyitems(config, items):
    if _real_pytorch3d_available():
        return
    skip = pytest.mark.skip(reason="needs the real pytorch3d package (not installed in this environment)")
    for item in items:
        if item.nodeid in NEEDS_REAL_PYTORCH3D:
            item.add_marker(skip)
