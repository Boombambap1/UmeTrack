"""Minimal stand-in for `pytorch3d.transforms.so3_exp_map`, the only pytorch3d symbol upstream uses.

Upstream lib/common/hand_skinning.py imports `so3_exp_map` to turn axis-angle vectors into
rotation matrices. pytorch3d has no macOS arm64 wheels, so `install()` registers this module
as `pytorch3d.transforms` when the real package is not importable. Upstream is not modified.

The formula and eps clamp follow pytorch3d's implementation (Rodrigues' formula):
    theta = sqrt(max(|w|^2, eps)),  R = I + sin(theta)/theta * K + (1 - cos(theta))/theta^2 * K^2
where K is the skew-symmetric (hat) matrix of w.
"""

import sys
import types

import torch

USING_SHIM = False


def hat(v: torch.Tensor) -> torch.Tensor:
    if v.dim() != 2 or v.shape[1] != 3:
        raise ValueError("Input vectors have to be 3-dimensional.")
    x, y, z = v.unbind(1)
    zero = torch.zeros_like(x)
    return torch.stack(
        [zero, -z, y, z, zero, -x, -y, x, zero], dim=1
    ).view(-1, 3, 3)


def so3_exp_map(log_rot: torch.Tensor, eps: float = 0.0001) -> torch.Tensor:
    if log_rot.dim() != 2 or log_rot.shape[1] != 3:
        raise ValueError("Input tensor shape has to be Nx3.")
    nrms = (log_rot * log_rot).sum(1)
    rot_angles = torch.clamp(nrms, eps).sqrt()
    rot_angles_inv = 1.0 / rot_angles
    fac1 = rot_angles_inv * rot_angles.sin()
    fac2 = rot_angles_inv * rot_angles_inv * (1.0 - rot_angles.cos())
    skews = hat(log_rot)
    skews_square = torch.bmm(skews, skews)
    eye = torch.eye(3, dtype=log_rot.dtype, device=log_rot.device)[None]
    return fac1[:, None, None] * skews + fac2[:, None, None] * skews_square + eye


def install() -> bool:
    """Register the shim if real pytorch3d is unavailable. Returns True if the shim is in use."""
    global USING_SHIM
    try:
        import pytorch3d.transforms  # noqa: F401

        return USING_SHIM
    except ImportError:
        pass
    pkg = types.ModuleType("pytorch3d")
    transforms = types.ModuleType("pytorch3d.transforms")
    transforms.so3_exp_map = so3_exp_map
    transforms.hat = hat
    transforms.__doc__ = "Part B shim: see scripts/pytorch3d_shim.py"
    pkg.transforms = transforms
    sys.modules["pytorch3d"] = pkg
    sys.modules["pytorch3d.transforms"] = transforms
    USING_SHIM = True
    return True
