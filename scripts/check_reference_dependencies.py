import importlib
import math
from pathlib import Path
import sys

import torch
import pytorch3d
from pytorch3d import _C
from pytorch3d.ops import knn_points
from pytorch3d.transforms import so3_exp_map

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / 'third_party' / 'UmeTrack'))


def main():
    angles = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, math.pi / 2]])
    rotations = so3_exp_map(angles)
    torch.testing.assert_close(rotations[0], torch.eye(3), atol=1e-6, rtol=0)
    rotated_x = rotations[1] @ torch.tensor([1.0, 0.0, 0.0])
    torch.testing.assert_close(
        rotated_x, torch.tensor([0.0, 1.0, 0.0]), atol=1e-6, rtol=0
    )
    print('PASS: SO(3) zero and quarter-turn rotations')

    points = torch.tensor([[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]])
    result = knn_points(torch.tensor([[[0.5, 0.0, 0.0]]]), points, K=1)
    torch.testing.assert_close(result.dists, torch.tensor([[[0.25]]]))
    if result.idx.item() != 0:
        raise RuntimeError('Unexpected nearest-neighbor index')
    print('PASS: compiled PyTorch3D CPU nearest-neighbor operation')

    from lib.common.hand_skinning import _joint_local_transform
    rest = torch.tensor([[[1.0, 2.0, 3.0], [2.0, 1.0, 0.0]]])
    axes = torch.tensor([[[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]])
    transforms = _joint_local_transform(axes, rest, torch.tensor([[0.5, -0.3]]))
    transformed_rest = (transforms[..., :3, :3] @ rest.unsqueeze(-1)).squeeze(-1)
    transformed_rest += transforms[..., :3, 3]
    torch.testing.assert_close(transformed_rest, rest)
    print('PASS: UmeTrack joint rotation preserves its pivot')

    for name in ('lib.tracker.tracker', 'run_eval_known_skeleton', 'load_eval'):
        importlib.import_module(name)
        print(f'PASS: import {name}')
    print(f'PyTorch3D {pytorch3d.__version__}; PyTorch {torch.__version__}; CPU')
    print('Dependency checks only; no recorded hand-pose accuracy measured.')


if __name__ == '__main__':
    main()
