import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "third_party/UmeTrack"))
from lib.models.model_loader import load_pretrained_model
from umetrack_rb5.raw_exchange import validate_package


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def power_state():
    if platform.system() != "Darwin":
        return "not collected"
    result = subprocess.run(
        ["/usr/bin/pmset", "-g", "batt"], capture_output=True, text=True, timeout=5
    )
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def statistics(rows):
    values = np.array([row["milliseconds"] for row in rows], dtype=np.float64)
    return {
        "median_ms_per_crop": float(np.median(values)),
        "p95_ms_per_crop": float(np.percentile(values, 95)),
        "mean_ms_per_crop": float(values.mean()),
        "min_ms_per_crop": float(values.min()),
        "max_ms_per_crop": float(values.max()),
    }


def main():
    parser = argparse.ArgumentParser(description="Isolated CPU batch-one encoder timing on fixed real crops.")
    parser.add_argument("--reference", type=Path, default=ROOT / "artifacts/reference-bundle")
    parser.add_argument("--package", type=Path, default=ROOT / "artifacts/qualcomm-inputs")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/runs/mac-encoder-batch-one")
    parser.add_argument("--passes", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if min(args.passes, args.warmup, args.threads) < 1:
        parser.error("passes, warmup and threads must be positive")
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    manifest, records = validate_package(args.reference, args.package)
    checkpoint = ROOT / "third_party/UmeTrack/pretrained_models/pretrained_weights.torch"
    provenance = manifest["reference_provenance"]
    checkpoint_hash = digest(checkpoint)
    if checkpoint_hash != provenance["checkpoint_sha256"]:
        raise ValueError("Checkpoint differs from reference features")
    upstream_commit = subprocess.check_output(
        ["git", "-C", str(ROOT / "third_party/UmeTrack"), "rev-parse", "HEAD"], text=True
    ).strip()
    if upstream_commit != provenance["upstream_commit"]:
        raise ValueError("Upstream revision differs from reference features")
    crops = []
    for row in manifest["crops"]:
        record = records[row["frame_key"]]
        index = row["crop_index"]
        crops.append((row["crop_id"], torch.from_numpy(record["inputs"][index:index + 1].copy()),
                      record["features"][index:index + 1]))
    if not crops:
        raise ValueError("No input crops")
    model = load_pretrained_model(str(checkpoint)).cpu().eval()
    encoder = model._feature_extractor._image_backbone
    differences = []
    # Check tolerance is fixed before timing; batch-one and frame-batch kernels may round differently.
    atol, rtol = 1e-5, 1e-4
    with torch.no_grad():
        for crop_id, value, expected in crops:
            actual = encoder(value).numpy()
            if actual.shape != (1, 72, 6, 6) or actual.dtype != np.float32 or not np.isfinite(actual).all():
                raise ValueError(f"Invalid output: {crop_id}")
            np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol, err_msg=crop_id)
            differences.append((actual.astype(np.float64) - expected).ravel())
        for index in range(args.warmup):
            encoder(crops[index % len(crops)][1])
        args.output.mkdir(parents=True, exist_ok=False)
        start_power = power_state()
        rows = []
        for trial in range(args.passes):
            for index, (crop_id, value, _) in enumerate(crops):
                start = time.perf_counter_ns()
                features = encoder(value)
                elapsed = time.perf_counter_ns() - start
                rows.append({"pass": trial + 1, "crop_index": index,
                             "crop_id": crop_id, "milliseconds": elapsed / 1e6})
            print(f"Batch-one pass {trial + 1}/{args.passes} complete", flush=True)
        end_power = power_state()
    with (args.output / "measurements.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    differences = np.concatenate(differences)
    summary = {
        "execution_kind": "mac-cpu-encoder-benchmark",
        "device": "Mac CPU" if platform.system() == "Darwin" else "CPU",
        "batch_size": 1, "input_shape": [1, 1, 96, 96], "output_shape": [1, 72, 6, 6],
        "dtype": "float32", "grad_mode": "no_grad", "model_mode": "eval",
        "unique_crops": len(crops), "passes": args.passes, "measured_inferences": len(rows),
        "warmup_inferences": args.warmup,
        "warmup_note": "All crops were also validated before the additional warmup; this is a warm compute benchmark.",
        "threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads(),
        **statistics(rows),
        "per_pass": [{"pass": trial + 1, **statistics([r for r in rows if r["pass"] == trial + 1])}
                     for trial in range(args.passes)],
        "batch_one_vs_saved_frame_batch": {
            "atol": atol, "rtol": rtol, "all_crops_within_tolerance": True,
            "feature_mae": float(np.abs(differences).mean()),
            "feature_rmse": float(np.sqrt(np.square(differences).mean())),
            "feature_max_absolute_error": float(np.abs(differences).max()),
        },
        "provenance": {"upstream_commit": upstream_commit, "checkpoint_sha256": checkpoint_hash,
                       "package_sha256": digest(args.package / "package.json"),
                       "input_list_sha256": digest(args.package / "input_list.txt"),
                       "reference_manifest_sha256": digest(args.reference / "features/manifest.json"),
                       "benchmark_script_sha256": digest(Path(__file__)),
                       "measurements_sha256": digest(args.output / "measurements.csv")},
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "numpy": np.__version__, "platform": platform.platform(),
                        "native_threads": {name: os.environ.get(name) for name in
                                           ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS")},
                        "power_start": start_power, "power_end": end_power,
                        "thermal_state": "not measured", "competing_workloads": "not measured"},
        "scope": "Warm synchronous CPU encoder calls only; preloaded batch-one crops. Excludes decode, preprocessing, file IO, transfer, model load and downstream tracking. Not RB5 or headset latency.",
        "command": [".venv/bin/python", "scripts/benchmark_encoder.py", *sys.argv[1:]],
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
