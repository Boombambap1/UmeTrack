"""Check the so3_exp_map stand-in against scipy's rotation-vector conversion."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from pytorch3d_shim import so3_exp_map  # noqa: E402


def _ref(w: np.ndarray) -> np.ndarray:
    return Rotation.from_rotvec(w).as_matrix()


def test_random_rotations_match_scipy():
    # Angles >= 0.02 rad, above the eps clamp (sqrt(1e-4) = 0.01); small angles are tested below.
    rng = np.random.default_rng(0)
    w = rng.normal(size=(1000, 3))
    w = w / np.linalg.norm(w, axis=1, keepdims=True) * rng.uniform(0.02, np.pi, size=(1000, 1))
    got = so3_exp_map(torch.from_numpy(w)).numpy()
    np.testing.assert_allclose(got, _ref(w), atol=1e-12)


def test_float32_like_upstream():
    rng = np.random.default_rng(1)
    w = (rng.normal(size=(500, 3)) * 0.8).astype(np.float32)
    got = so3_exp_map(torch.from_numpy(w)).numpy()
    np.testing.assert_allclose(got, _ref(w.astype(np.float64)), atol=2e-6)


@pytest.mark.parametrize("scale", [0.0, 1e-8, 1e-4, 1e-2])
def test_small_angles(scale):
    # Below sqrt(eps)=1e-2 pytorch3d (and this shim) clamp theta, giving an O(theta^2) error.
    rng = np.random.default_rng(2)
    w = rng.normal(size=(100, 3))
    w = w / np.linalg.norm(w, axis=1, keepdims=True) * scale
    got = so3_exp_map(torch.from_numpy(w)).numpy()
    np.testing.assert_allclose(got, _ref(w), atol=1e-4)


def test_outputs_are_rotations():
    w = torch.randn(200, 3, dtype=torch.float64)
    R = so3_exp_map(w)
    eye = torch.eye(3, dtype=torch.float64).expand_as(R)
    assert torch.allclose(R @ R.transpose(1, 2), eye, atol=1e-12)
    assert torch.allclose(torch.linalg.det(R), torch.ones(200, dtype=torch.float64), atol=1e-12)
