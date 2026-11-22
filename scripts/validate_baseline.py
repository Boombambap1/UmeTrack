import argparse
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import pickle
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'third_party/UmeTrack'))
from load_eval import _compute_metrics, aggregate_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    summary = json.loads((args.run / 'summary.json').read_text())
    manifest = json.loads((ROOT / 'manifests/separate-hand-test.json').read_text())
    rows = json.loads((args.run / 'per-sequence.json').read_text())
    expected = {r['source_path'] for r in manifest['recordings']}
    if not summary['complete'] or len(rows) != len(expected) or {r['source_path'] for r in rows} != expected:
        raise RuntimeError('Run does not cover the complete input manifest')
    errors, masks, accelerations = [], [], []
    for row in rows:
        source = Path(row['source_path'])
        path = args.run / 'sequences' / source.parent.name / source.with_suffix('.npy').name
        with path.open('rb') as handle:
            result = pickle.load(handle)
        metric = _compute_metrics(result['gt_keypoints'], result['tracked_keypoints'], result['valid_tracking'])
        if not np.isfinite(metric.keypoint_errors).all():
            raise RuntimeError(f'Nonfinite error: {source}')
        np.testing.assert_allclose(metric.keypoint_errors.mean(), row['mean_joint_error_mm'], atol=1e-12, rtol=0)
        errors.append(metric.keypoint_errors)
        masks.append(result['valid_tracking'])
        accelerations.append(metric.keypoint_accelerations)
    flat_errors = np.concatenate(errors)
    flat_masks = np.concatenate(masks, axis=1)
    np.testing.assert_allclose(flat_errors.mean(), summary['mean_joint_error_mm'], atol=1e-12, rtol=0)
    if int(flat_masks.sum()) != summary['tracked_hand_frames'] or flat_masks.size != summary['hand_frame_slots']:
        raise RuntimeError('Coverage/count mismatch')
    text = io.StringIO()
    with redirect_stdout(text):
        aggregate_metrics(str(args.run / 'sequences'))
    print(text.getvalue())
    print('PASS: all recording identities, saved predictions, weighted error, and coverage verified')
    print(f'Full-test MPJPE: {float(flat_errors.mean()):.9f} mm')
    print(f'Within task reproduction band 9.3–9.5 mm: {9.3 <= flat_errors.mean() <= 9.5}')
    validation_path = args.run / 'validation.txt'
    if validation_path.exists():
        raise FileExistsError(validation_path)
    validation_path.write_text(text.getvalue() + '\nPASS: complete manifest and saved-prediction cross-checks\n')


if __name__ == '__main__':
    main()
