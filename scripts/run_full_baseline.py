import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import pickle
import shutil
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / 'third_party/UmeTrack'
DATA_REPO = UPSTREAM / 'UmeTrack_data'
sys.path.insert(0, str(UPSTREAM))


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        while chunk := stream.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def prepare(item, revision):
    relative = Path(item['source_path'])
    dest = ROOT / 'artifacts/data/separate-hand-test' / relative.parent.name / relative.name
    dest.parent.mkdir(parents=True, exist_ok=True)
    smoke = ROOT / 'artifacts/data/smoke/recording_00.mp4'
    if not dest.exists() and item['sha256'] == '8df9d21e1843902a6bceb1750e177bd272aaec9a9cf8968363e6cfac7dd1bc3c' and smoke.exists():
        if digest(smoke) == item['sha256']:
            os.link(smoke, dest)
    if dest.exists():
        if dest.stat().st_size != item['size_bytes'] or digest(dest) != item['sha256']:
            raise RuntimeError(f'Existing video checksum mismatch: {dest}')
    else:
        url = f'https://media.githubusercontent.com/media/facebookresearch/UmeTrack_data/{revision}/{item["source_path"]}'
        partial = dest.with_suffix('.mp4.part')
        for attempt in range(3):
            try:
                with urllib.request.urlopen(url, timeout=90) as response, partial.open('wb') as output:
                    shutil.copyfileobj(response, output, length=1024 * 1024)
                if partial.stat().st_size != item['size_bytes'] or digest(partial) != item['sha256']:
                    raise RuntimeError(f'Download checksum mismatch: {relative}')
                partial.rename(dest)
                break
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(attempt + 1)
    label_path = item['source_path'][:-4] + '.json'
    labels = subprocess.check_output(['git', '-C', str(DATA_REPO), 'show', f'{revision}:{label_path}'])
    target = dest.with_suffix('.json')
    if target.exists() and target.read_bytes() != labels:
        raise RuntimeError(f'Existing labels mismatch: {target}')
    if not target.exists():
        target.write_bytes(labels)
    return {**item, 'local_path': str(dest.relative_to(ROOT)), 'labels_sha256': hashlib.sha256(labels).hexdigest()}


def evaluate(item, output_dir, threads):
    import numpy as np
    import torch
    from load_eval import _compute_metrics
    from run_eval_known_skeleton import _track_sequence

    if torch.cuda.is_available():
        raise RuntimeError('This run is specifically the Mac CPU baseline')
    torch.set_num_threads(threads)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    relative = Path(item['source_path'])
    result_path = Path(output_dir) / relative.parent.name / relative.with_suffix('.npy').name
    if result_path.exists():
        raise FileExistsError(result_path)
    checkpoint = UPSTREAM / 'pretrained_models/pretrained_weights.torch'
    start = time.perf_counter()
    errors = _track_sequence((str(ROOT / item['local_path']), str(result_path)), str(checkpoint))
    if errors is None or not len(errors) or not np.isfinite(errors).all():
        raise RuntimeError(f'No finite errors for {relative}')
    with result_path.open('rb') as stream:
        result = pickle.load(stream)
    mask = result['valid_tracking']
    if not np.isfinite(result['tracked_keypoints'][mask]).all():
        raise RuntimeError(f'Nonfinite predictions for {relative}')
    metrics = _compute_metrics(result['gt_keypoints'], result['tracked_keypoints'], mask)
    np.testing.assert_allclose(errors, metrics.keypoint_errors, rtol=0, atol=0)
    row = {
        **item, 'frames': int(mask.shape[1]), 'hand_frame_slots': int(mask.size),
        'tracked_hand_frames': int(mask.sum()), 'tracked_by_hand': mask.sum(axis=1).tolist(),
        'joint_error_sum_mm': float(errors.sum()), 'mean_joint_error_mm': float(errors.mean()),
        'acceleration_sum': float(metrics.keypoint_accelerations.sum()),
        'acceleration_count': int(metrics.keypoint_accelerations.size),
        'wall_seconds_including_load_decode_and_io': time.perf_counter() - start,
    }
    result_path.with_suffix('.json').write_text(json.dumps(row, indent=2) + '\n')
    print(f'EVAL {relative.parent.name}/{relative.name}: {row["mean_joint_error_mm"]:.6f} mm; {row["tracked_hand_frames"]}/{row["hand_frame_slots"]} tracked', flush=True)
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--downloads', type=int, default=3)
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/runs/separate-hand-baseline')
    args = parser.parse_args()
    if min(args.workers, args.threads, args.downloads) < 1:
        parser.error('Worker and thread counts must be positive')
    if args.output.exists():
        raise FileExistsError(f'Refusing existing run directory: {args.output}')
    manifest_path = ROOT / 'manifests/separate-hand-test.json'
    manifest = json.loads(manifest_path.read_text())
    revision = manifest['dataset_commit']
    checkpoint = UPSTREAM / 'pretrained_models/pretrained_weights.torch'
    config = {
        'dataset_commit': revision,
        'upstream_commit': subprocess.check_output(['git', '-C', str(UPSTREAM), 'rev-parse', 'HEAD'], text=True).strip(),
        'checkpoint_sha256': digest(checkpoint), 'input_manifest_sha256': digest(manifest_path),
        'protocol': 'known_skeleton/real/separate_hand/testing', 'device': 'Mac CPU',
        'workers': args.workers, 'threads_per_worker': args.threads,
        'native_thread_environment': {name: os.environ.get(name) for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'MKL_NUM_THREADS')},
        'torch_interop_threads': 1,
        'crop_selection': 'upstream ground-truth-assisted; known hand geometry',
        'purpose': 'full-split accuracy reference; wall times are not encoder benchmarks',
        'expected_recordings': len(manifest['recordings']),
    }
    args.output.mkdir(parents=True)
    (args.output / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    start = time.perf_counter()
    rows, failures, eval_futures = [], [], {}
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn')) as processors:
        with ThreadPoolExecutor(max_workers=args.downloads) as downloads:
            pending = {downloads.submit(prepare, item, revision): item for item in manifest['recordings']}
            for n, future in enumerate(as_completed(pending), 1):
                item = pending[future]
                try:
                    prepared = future.result()
                    eval_futures[processors.submit(evaluate, prepared, str(args.output / 'sequences'), args.threads)] = item
                    print(f'READY {n}/{len(pending)} {item["source_path"]}', flush=True)
                except Exception as exc:
                    failures.append({'source_path': item['source_path'], 'stage': 'download', 'error': repr(exc)})
                    print(f'DOWNLOAD FAILED {item["source_path"]}: {exc}', flush=True)
        for n, future in enumerate(as_completed(eval_futures), 1):
            item = eval_futures[future]
            try:
                rows.append(future.result())
            except Exception as exc:
                failures.append({'source_path': item['source_path'], 'stage': 'evaluation', 'error': repr(exc)})
            print(f'COLLECTED {n}/{len(eval_futures)}; failures={len(failures)}', flush=True)
    rows.sort(key=lambda row: row['source_path'])
    tracked = sum(r['tracked_hand_frames'] for r in rows)
    slots = sum(r['hand_frame_slots'] for r in rows)
    accel_count = sum(r['acceleration_count'] for r in rows)
    summary = {
        **config, 'completed_recordings': len(rows), 'complete': not failures and len(rows) == len(manifest['recordings']),
        'source_frames': sum(r['frames'] for r in rows), 'hand_frame_slots': slots,
        'tracked_hand_frames': tracked, 'coverage_all_slots': tracked / slots if slots else None,
        'mean_joint_error_mm': sum(r['joint_error_sum_mm'] for r in rows) / tracked if tracked else None,
        'mean_joint_acceleration': sum(r['acceleration_sum'] for r in rows) / accel_count if accel_count else None,
        'total_wall_seconds_including_downloads': time.perf_counter() - start,
        'failures': failures,
    }
    (args.output / 'per-sequence.json').write_text(json.dumps(rows, indent=2) + '\n')
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print('FINAL ' + json.dumps(summary), flush=True)
    if not summary['complete']:
        raise RuntimeError('Incomplete baseline; inspect failures')


if __name__ == '__main__':
    main()
