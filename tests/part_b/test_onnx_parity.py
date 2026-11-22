"""PyTorch vs ONNX Runtime parity on a few real crops, within the declared tolerances."""

import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
yaml = pytest.importorskip("yaml")
pytest.importorskip("onnxruntime")

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
from check_onnx_parity import compare, preprocess, run_features  # noqa: E402
from common import load_config  # noqa: E402

ONNX = REPO / "artifacts" / "encoder.onnx"
CROPS = REPO / "artifacts" / "crops" / "crops_uint8.npy"
N_CROPS = 6


def test_preprocess_contract():
    crops = np.array([[[0, 255], [128, 1]]], dtype=np.uint8)
    x = preprocess(crops)
    assert x.dtype == np.float32 and x.shape == (1, 1, 2, 2)
    np.testing.assert_array_equal(x[0, 0], crops[0].astype(np.float32) / 255.0)


@pytest.mark.skipif(
    not (ONNX.exists() and CROPS.exists()),
    reason="run scripts/export_encoder.py and scripts/make_crops.py first",
)
def test_onnx_matches_torch_on_real_crops():
    cfg = load_config(REPO / "configs" / "experiment.yaml")
    x = preprocess(np.load(CROPS)[:N_CROPS])
    t, o = run_features(cfg, ONNX, x)
    assert t.shape == o.shape == (N_CROPS, 72, 6, 6)
    result = compare(t, o, cfg["tolerances"]["onnx_vs_torch"])
    assert result["pass"], result["checks"]
