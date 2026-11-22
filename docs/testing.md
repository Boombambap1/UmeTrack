# Testing in the two environments

Part A (Steven) and Part B (Ali) test on different Python environments. Each side's tests are written for its
own runner, and each runner only picks up its own tests, so neither side needs the other's dependencies.

| | Part A — Mac reference environment | Part B — export/conversion environment |
| --- | --- | --- |
| Purpose | Accuracy baseline, capture/replay, raw exchange, timing | ONNX export, parity, SNPE I/O, calibration, INT8 preview |
| Python | 3.12.15 (macOS arm64) | 3.12.15 (macOS arm64) |
| Packages | `requirements-mac-reference.txt` (torch 2.4.1, numpy 1.26.4, av 13.1.0, opencv-python 4.10.0.84, Pillow 10.4.0) plus PyTorch3D 0.7.8 built from source | `requirements-mac.txt` (torch 2.15.0, numpy 2.5.5, onnx 1.23.2, onnxruntime 1.32.0, onnxscript 0.7.2, av 19.0.2, opencv-python-headless 5.1.0.95, Pillow 13.0.0, PyYAML, pytest) |
| pytorch3d | Real package | Not installed; `scripts/pytorch3d_shim.py` stands in for `so3_exp_map` during crop placement only |
| Runner | `unittest` | `pytest` |
| Tests | `tests/test_*.py` (TestCase classes) | `tests/part_b/test_*.py` |

Both READMEs create the environment at `.venv/`. They are normally separate machines; if both live in one
checkout, give one of them another directory (e.g. `.venv-b/`) and use that path in the commands below.

Crops generated in the two environments should not be treated as interchangeable (different package versions,
and Part B uses the pytorch3d stand-in); formal comparisons use Part A's frozen crop bundle.

## Part A

Create (from README.md; the PyTorch3D build flags are explained in `docs/pytorch3d-installation.md`):

```sh
python3.12 scripts/bootstrap_dependencies.py
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-mac-reference.txt
PATH="$PWD/.venv/bin:$PATH" MAX_JOBS=2 MACOSX_DEPLOYMENT_TARGET=14.0 CC=clang CXX=clang++ CFLAGS=-Wno-invalid-specialization PYTORCH3D_FORCE_NO_CUDA=1 .venv/bin/python -m pip install --no-build-isolation ./third_party/pytorch3d
```

Run:

```sh
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=2 .venv/bin/python -m unittest discover -s tests -v
```

`unittest discover` does not descend into `tests/part_b/` (it has no `__init__.py`), so Part B's pytest files
are never imported here. The no-hand/reappearance test also expects the smoke recording under
`artifacts/data/smoke/` (see README.md).

## Part B

Create:

```sh
python3.12 scripts/bootstrap_dependencies.py
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-mac.txt
```

Run:

```sh
.venv/bin/python -m pytest -q -rs
```

This collects `tests/part_b/` and also runs Part A's TestCase classes under pytest. Behaviour:

- Each Part B test file uses `pytest.importorskip` for its third-party dependencies, so a missing package
  skips that file instead of failing collection.
- `conftest.py` skips Part A's `test_no_hand_clears_history_without_encoder_call` when the real pytorch3d
  package is not installed (the stand-in does not count). Part A's file is unchanged.
- Tests that need ignored local files skip when those are absent: the pinned checkpoint, `artifacts/encoder.onnx`
  and `artifacts/crops/crops_uint8.npy`.

Run only Part B's tests:

```sh
.venv/bin/python -m pytest -q -rs tests/part_b
```

## If Part A wants the pytorch3d skip under unittest too

Not applied (Part A's file). The change would be one decorator on the test in `tests/test_encoder_provider.py`,
plus `import importlib.util` at the top:

```python
@unittest.skipUnless(importlib.util.find_spec("pytorch3d"), "needs real pytorch3d")
def test_no_hand_clears_history_without_encoder_call(self):
```
