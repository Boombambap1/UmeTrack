"""so3_exp_map stand-in (scripts/pytorch3d_shim.py) vs scipy Rotation.from_rotvec, within 1e-6.

100 axis-angle vectors: 80 with random axes and angles in (0, pi], 20 near zero
(including exactly zero and angles below pytorch3d's eps clamp of sqrt(1e-4) = 0.01 rad).
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from pytorch3d_shim import so3_exp_map  # noqa: E402

ATOL = 1e-6
NEAR_ZERO_ANGLES = [0.0, 1e-12, 1e-9, 1e-7, 1e-6, 1e-5, 1e-4, 3e-4, 1e-3, 2e-3,
                    3e-3, 5e-3, 7e-3, 9e-3, 9.9e-3, 1e-2, 1.01e-2, 1.5e-2, 2e-2, 5e-2]


def _vectors() -> np.ndarray:
    rng = np.random.default_rng(1234)
    axes = rng.normal(size=(100, 3))
    axes /= np.linalg.norm(axes, axis=1, keepdims=True)
    angles = np.concatenate([rng.uniform(0.0, np.pi, size=80), NEAR_ZERO_ANGLES])
    return axes * angles[:, None]


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_standin_matches_scipy(dtype):
    w = _vectors()
    assert w.shape == (100, 3)
    got = so3_exp_map(torch.from_numpy(w).to(dtype)).double().numpy()
    want = Rotation.from_rotvec(w).as_matrix()
    err = np.abs(got - want).max(axis=(1, 2))
    assert err.max() <= ATOL, f"worst {err.max():.3e} at angle {np.linalg.norm(w[err.argmax()]):.3e}"
