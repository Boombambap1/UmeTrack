import hashlib
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_ROOT = PROJECT_ROOT / 'third_party' / 'UmeTrack'
sys.path.insert(0, str(UPSTREAM_ROOT))

import torch
from lib.models.model_loader import load_pretrained_model


def main():
    checkpoint = UPSTREAM_ROOT / 'pretrained_models' / 'pretrained_weights.torch'
    model = load_pretrained_model(str(checkpoint)).cpu().eval()
    with torch.no_grad():
        features = model._feature_extractor._image_backbone(
            torch.zeros((1, 1, 96, 96), dtype=torch.float32)
        )
    if tuple(features.shape) != (1, 72, 6, 6):
        raise RuntimeError(f'Unexpected feature shape: {tuple(features.shape)}')
    if not torch.isfinite(features).all():
        raise RuntimeError('Encoder produced nonfinite features')
    print(f'Python: {sys.version.split()[0]}')
    print(f'PyTorch: {torch.__version__}; device: {features.device}')
    print(f'Checkpoint SHA256: {hashlib.sha256(checkpoint.read_bytes()).hexdigest()}')
    print(f'PASS: synthetic zero crop -> finite features {tuple(features.shape)}')
    print('Setup check only; no pose accuracy or board performance measured.')


if __name__ == '__main__':
    main()
