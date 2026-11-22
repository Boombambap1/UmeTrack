import argparse
import hashlib
import json
from pathlib import Path
import pickle
import sys
import time

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_ROOT = PROJECT_ROOT / 'third_party' / 'UmeTrack'
sys.path.insert(0, str(UPSTREAM_ROOT))
from run_eval_known_skeleton import _track_sequence


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, default=PROJECT_ROOT / 'artifacts/data/smoke/recording_00.mp4')
    parser.add_argument('--output', type=Path, default=PROJECT_ROOT / 'artifacts/runs/sequence-smoke')
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f'Refusing to overwrite run directory: {args.output}')
    if torch.cuda.is_available():
        raise RuntimeError('This smoke runner is intended for the Mac CPU reference')
    checkpoint = UPSTREAM_ROOT / 'pretrained_models/pretrained_weights.torch'
    torch.set_num_threads(args.threads)
    args.output.mkdir(parents=True)
    output = args.output / 'upstream-results.npy'
    start = time.perf_counter()
    errors = _track_sequence((str(args.input), str(output)), str(checkpoint))
    elapsed = time.perf_counter() - start
    if errors is None or not len(errors) or not np.isfinite(errors).all():
        raise RuntimeError('No finite scored hand-frame errors returned')
    with output.open('rb') as handle:
        result = pickle.load(handle)
    valid = result['valid_tracking']
    if not np.isfinite(result['tracked_keypoints'][valid]).all():
        raise RuntimeError('Nonfinite predicted keypoints')
    summary = {
        'purpose': 'one-sequence smoke; not full published benchmark',
        'protocol': 'known_skeleton/separate_hand',
        'device': 'cpu', 'threads': args.threads,
        'frames': int(valid.shape[1]),
        'total_hand_frame_slots': int(valid.size),
        'tracked_hand_frames': int(valid.sum()),
        'tracked_hand_frames_by_hand': valid.sum(axis=1).tolist(),
        'coverage_all_slots': float(valid.mean()),
        'mean_joint_error_mm': float(errors.mean()),
        'wall_seconds_including_load_decode_and_io': elapsed,
        'input_sha256': hashlib.sha256(args.input.read_bytes()).hexdigest(),
        'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        'crop_selection': 'upstream ground-truth-assisted',
    }
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
