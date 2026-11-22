import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from umetrack_rb5.reference import configure_cpu, file_digest, profile_clip


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path, default=ROOT / "artifacts/runs/serial-profile"
    )
    parser.add_argument("--passes", type=int, default=3)
    args = parser.parse_args()
    if args.passes < 1:
        parser.error("--passes must be positive")
    if args.output.exists():
        raise FileExistsError(args.output)
    configure_cpu()
    manifest = json.loads((ROOT / "manifests/reference-clips.json").read_text())
    for clip in manifest["clips"]:
        if file_digest(ROOT / clip["local_path"]) != clip["video_sha256"]:
            raise ValueError("Input mismatch")
    args.output.mkdir(parents=True)
    rows = []
    for trial in range(args.passes):
        for clip in manifest["clips"]:
            batch = profile_clip(clip)
            rows.extend(dict(r, trial=trial) for r in batch)
        print(f"Profile pass {trial+1}/{args.passes} complete", flush=True)
    measured = [r for r in rows if not r["warmup"]]
    stages = [
        "decode_s",
        "crop_selection_s",
        "prepare_s",
        "encoder_s",
        "remaining_model_s",
        "postprocess_s",
        "landmarks_s",
        "total_s",
    ]
    stats = {
        k: {
            "median_ms": float(np.median([r[k] for r in measured]) * 1000),
            "p95_ms": float(np.percentile([r[k] for r in measured], 95) * 1000),
            "sum_s": sum(r[k] for r in measured),
        }
        for k in stages
    }
    total = stats["total_s"]["sum_s"]
    encoder = stats["encoder_s"]["sum_s"]
    remaining = stats["remaining_model_s"]["sum_s"]
    summary = {
        "device": "Mac CPU",
        "threads": 2,
        "passes": args.passes,
        "warmup_frames_per_clip_per_pass": 10,
        "measured_frame_instances": len(measured),
        "unique_measured_frames": len(measured) // args.passes,
        "crop_count_histogram": dict(Counter(r["crops"] for r in measured)),
        "stages": stats,
        "encoder_fraction_of_pipeline": encoder / total,
        "encoder_fraction_of_model": encoder / (encoder + remaining),
        "mean_encoder_ms_per_crop": 1000 * encoder / sum(r["crops"] for r in measured),
        "precision": "float32 host tensors",
        "serial": True,
        "capture_io_included": False,
        "scope": "two short clips; startup/label load excluded; decode included; timing hooks add overhead; not RB5 or headset latency",
    }
    (args.output / "frames.json").write_text(json.dumps(rows, indent=2) + "\n")
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
