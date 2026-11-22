import argparse
from collections import Counter
import json
from pathlib import Path
import pickle
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from umetrack_rb5.encoder_provider import save_records, load_records
from umetrack_rb5.reference import configure_cpu, file_digest, run_clip


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path, default=ROOT / "artifacts/reference-bundle"
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest = json.loads((ROOT / "manifests/reference-clips.json").read_text())
    configure_cpu()
    provenance = {
        "clips": manifest,
        "upstream_commit": subprocess.check_output(
            ["git", "-C", str(ROOT / "third_party/UmeTrack"), "rev-parse", "HEAD"],
            text=True,
        ).strip(),
        "checkpoint_sha256": file_digest(
            ROOT / "third_party/UmeTrack/pretrained_models/pretrained_weights.torch"
        ),
    }
    for clip in manifest["clips"]:
        assert file_digest(ROOT / clip["local_path"]) == clip["video_sha256"]
        assert (
            file_digest((ROOT / clip["local_path"]).with_suffix(".json"))
            == clip["labels_sha256"]
        )
    args.output.mkdir(parents=True)
    all_records, references, all_frames = {}, {}, []
    for clip in manifest["clips"]:
        normal, _, frames = run_clip(clip, "normal")
        captured, records, captured_frames = run_clip(clip, "capture")
        for key in normal:
            np.testing.assert_array_equal(normal[key], captured[key])
        assert frames == captured_frames
        baseline = (
            ROOT
            / "artifacts/runs/separate-hand-bounded/sequences"
            / clip["user"]
            / f'{clip["recording"]}.npy'
        )
        with baseline.open("rb") as handle:
            prior = pickle.load(handle)
        for key, source in [
            ("predicted", "tracked_keypoints"),
            ("truth", "gt_keypoints"),
            ("valid", "valid_tracking"),
        ]:
            np.testing.assert_array_equal(
                normal[key], prior[source][:, : clip["frames"]]
            )
        for key in records:
            if key in all_records:
                raise RuntimeError("Duplicate frame identity")
        all_records.update(records)
        references[clip["id"]] = normal
        all_frames.extend(frames)
        np.savez(args.output / f'{clip["id"]}-poses.npz', **normal)
        print(
            f'CAPTURE PASS {clip["id"]}: matches normal run and full-baseline prefix',
            flush=True,
        )
    save_records(all_records, args.output / "features", provenance)
    loaded = load_records(args.output / "features", provenance)
    errors = []
    for clip in manifest["clips"]:
        records = {k: v for k, v in loaded.items() if k.startswith(clip["id"] + ":")}
        replayed, _, frames = run_clip(clip, "replay", records)
        normal = references[clip["id"]]
        for key in normal:
            np.testing.assert_array_equal(normal[key], replayed[key])
        assert frames == [
            f for f in all_frames if f["key"].startswith(clip["id"] + ":")
        ]
        errors.extend(
            np.linalg.norm(
                (normal["predicted"] - normal["truth"])[normal["valid"]], axis=-1
            )
            .mean(axis=-1)
            .tolist()
        )
        print(
            f'REPLAY PASS {clip["id"]}: bit-identical predictions, ground truth and masks',
            flush=True,
        )
    summary = {
        "frames": len(all_frames),
        "crops": sum(f["crop_count"] for f in all_frames),
        "crop_count_histogram": dict(Counter(f["crop_count"] for f in all_frames)),
        "tracked_hand_frames": len(errors),
        "mean_joint_error_mm": float(np.mean(errors)),
        "capture_equals_normal": True,
        "normal_equals_saved_full_baseline_prefix": True,
        "pc_replay_equals_capture_exactly": True,
        "maximum_pose_difference_mm": 0.0,
        "scored_frames": 100,
        "context_frames": 0,
        "model_state": "fresh model/tracker per clip per mode",
        "purpose": "initial conversion-equivalence bundle; not representative headset validation",
        "provenance": provenance,
    }
    (args.output / "frames.json").write_text(json.dumps(all_frames, indent=2) + "\n")
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "provenance"}, indent=2))


if __name__ == "__main__":
    main()
