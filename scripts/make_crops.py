"""Make 96x96 uint8 encoder crops from one UmeTrack recording using upstream tracker code.

Crop placement follows upstream run_eval_known_skeleton.py: ground-truth poses from the
recording's json -> HandTracker.gen_crop_cameras(min_num_crops=1) -> _warp_image per
(hand, camera), in the same order as HandTracker._make_inputs. Every frame is cross-checked
against upstream _make_inputs: its left_images must equal crops / 255 exactly.

DEVELOPMENT ONLY. Crops made here are not the canonical crops: formal comparisons use Part A's frozen
crop files (scripts/canonical_crops.py). See PURPOSE_NOTE.
"""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

import pytorch3d_shim
import upstream_compat
from common import REPO, load_config, load_model, rel, sha256_file, update_hashes

PURPOSE = "development"
PURPOSE_NOTE = (
    "Development only - not for formal comparisons. " + 'Part A audit of a21a692: "Crop IDs match, but the decoding stacks produce different pixels from the same video. The first decoded frame differed by up to two intensity levels. The geometry shim also introduced small additional crop differences. Formal hardware comparisons must use the frozen canonical crop files, rather than independently regenerating crops on each machine."'
)


def extract_crops(model, recording_mp4: Path, recording: str, n_frames=None, frame_stride=1, max_crops=None):
    """Crops from one recording via upstream crop placement, cross-checked against _make_inputs.

    Considers frames 0, stride, 2*stride, ... and keeps those with at least one crop. Stops after
    n_frames kept frames, or before a frame would push the total past max_crops (whole frames only).
    Returns (crops uint8 [N,96,96], rows [(row, recording, frame, hand_idx, cam_idx)],
    per_frame_counts, frames_scanned).
    """
    from lib.tracker.tracker import HandTracker, HandTrackerOpts, _warp_image
    from lib.tracker.video_pose_data import SyncedImagePoseStream

    stream = SyncedImagePoseStream(str(recording_mp4))
    labels = stream._hand_pose_labels
    tracker = HandTracker(model, HandTrackerOpts())

    crops, rows = [], []
    frames_scanned = 0
    per_frame_counts = []
    with torch.no_grad():
        for frame_idx, (input_frame, gt_tracking) in enumerate(stream):
            if frame_idx % frame_stride:
                continue
            frames_scanned += 1
            crop_cameras = tracker.gen_crop_cameras(
                [view.camera for view in input_frame.views],
                labels.camera_angles,
                labels.hand_model,
                gt_tracking,
                min_num_crops=1,
            )
            if not crop_cameras:
                continue

            frame_crops, frame_rows = [], []
            for hand_idx, per_hand in crop_cameras.items():
                for cam_idx, crop_camera in per_hand.items():
                    view = input_frame.views[cam_idx]
                    crop = _warp_image(view.camera, crop_camera, view.image)
                    assert crop.dtype == np.uint8 and crop.shape == (96, 96), (crop.dtype, crop.shape)
                    frame_rows.append((len(crops) + len(frame_crops), recording, frame_idx, hand_idx, cam_idx))
                    frame_crops.append(crop)
            if max_crops is not None and len(crops) + len(frame_crops) > max_crops:
                break

            # Cross-check against upstream's own input builder.
            frame_data, _, _ = tracker._make_inputs(input_frame, labels.hand_model, crop_cameras)
            upstream = frame_data.left_images.numpy()
            ours = np.stack(frame_crops).astype(np.float32) / 255.0
            if upstream.shape != ours.shape or not np.array_equal(upstream, ours):
                raise SystemExit(f"{recording} frame {frame_idx}: crops differ from upstream _make_inputs")

            crops.extend(frame_crops)
            rows.extend(frame_rows)
            per_frame_counts.append(len(frame_crops))
            if n_frames is not None and len(per_frame_counts) == n_frames:
                break
    return np.stack(crops) if crops else np.zeros((0, 96, 96), np.uint8), rows, per_frame_counts, frames_scanned


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/experiment.yaml")
    ap.add_argument("--recording", default=None, help="relative to recordings_dir; default data.first_recording")
    ap.add_argument("--n-frames", type=int, default=20, help="number of frames with at least one crop")
    ap.add_argument("--out", default="artifacts/crops/crops_uint8.npy")
    ap.add_argument("--manifest", default="manifests/crops_d5.csv")
    ap.add_argument("--meta", default="results/crops_d5/crops_meta.json")
    ap.add_argument("--hashes", default="manifests/hashes.txt")
    args = ap.parse_args()

    cfg = load_config(REPO / args.config)
    recording = args.recording or cfg["data"]["first_recording"]
    mp4 = REPO / cfg["data"]["recordings_dir"] / f"{recording}.mp4"
    label_json = mp4.with_suffix(".json")

    using_shim = pytorch3d_shim.install()
    model, _, _ = load_model(cfg)
    patches = upstream_compat.apply()

    crops, rows, per_frame_counts, frames_scanned = extract_crops(model, mp4, recording, n_frames=args.n_frames)

    if len(per_frame_counts) < args.n_frames:
        raise SystemExit(f"only {len(per_frame_counts)} frames with crops in {frames_scanned} scanned")
    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, crops)

    manifest = REPO / args.manifest
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["row", "recording", "frame", "hand_idx", "cam_idx", "purpose"])
        w.writerows([(*r, PURPOSE) for r in rows])

    hashes = {
        rel(mp4): sha256_file(mp4),
        rel(label_json): sha256_file(label_json),
        rel(out): sha256_file(out),
        rel(manifest): sha256_file(manifest),
    }
    update_hashes(REPO / args.hashes, hashes)

    frames = sorted({r[2] for r in rows})
    meta = {
        "purpose": PURPOSE,
        "purpose_note": PURPOSE_NOTE,
        "recording": recording,
        "data_commit": "355df534a1c761ac416427925485b311f22ce50e",
        "n_crops": int(crops.shape[0]),
        "n_frames": len(frames),
        "frames": frames,
        "frames_scanned": frames_scanned,
        "crops_per_frame": dict(sorted(Counter(per_frame_counts).items())),
        "crops_per_hand": dict(sorted(Counter(r[3] for r in rows).items())),
        "crops_per_camera": dict(sorted(Counter(r[4] for r in rows).items())),
        "pixel_range": [int(crops.min()), int(crops.max())],
        "pixel_mean": float(crops.mean()),
        "crop_placement": "ground-truth poses, HandTracker.gen_crop_cameras(min_num_crops=1), HandTrackerOpts defaults",
        "upstream_make_inputs_crosscheck": "exact match on every frame",
        "pytorch3d": "scripts/pytorch3d_shim.py (so3_exp_map)" if using_shim else "real package",
        "upstream_runtime_patches": patches,
        "notes": (
            [
                "These crops were placed using the so3_exp_map stand-in in scripts/pytorch3d_shim.py "
                "(real pytorch3d is not installed on the Mac); it matches scipy Rotation.from_rotvec "
                "within 1e-6 (tests/test_so3_standin.py).",
                "Part A's full evaluation should use real pytorch3d, not this stand-in.",
            ]
            if using_shim
            else []
        ),
        "sha256": hashes,
    }
    meta_path = REPO / args.meta
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")

    print(f"[crops] N={crops.shape[0]} crops from {len(frames)} frames (scanned {frames_scanned}); frames {frames[0]}..{frames[-1]}")
    print(f"[crops] crops/frame {meta['crops_per_frame']}, per hand {meta['crops_per_hand']}, per camera {meta['crops_per_camera']}")
    print(f"[crops] pixel range {meta['pixel_range']}, mean {meta['pixel_mean']:.1f}; pytorch3d: {meta['pytorch3d']}")
    print("[crops] upstream _make_inputs cross-check: exact match on every frame")
    for p, d in hashes.items():
        print(f"[crops] {p}  sha256 {d}")
    print(f"[crops] wrote {rel(meta_path)}  sha256 {sha256_file(meta_path)}")


if __name__ == "__main__":
    main()
