import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from umetrack_rb5.encoder_provider import load_records
from umetrack_rb5.raw_exchange import read_reference
from umetrack_rb5.reference import configure_cpu, run_clip


def evaluate_import(reference, imported):
    provenance, _ = read_reference(reference)
    manifest = json.loads((imported / "features/manifest.json").read_text())
    if manifest["provenance"]["reference"] != provenance:
        raise ValueError("Wrong imported reference provenance")
    records = load_records(imported / "features", manifest["provenance"])
    expected_frames = {
        entry["key"]
        for entry in json.loads((reference / "features/manifest.json").read_text())[
            "records"
        ]
    }
    if set(records) != expected_frames:
        raise ValueError("Missing/extra imported frame records")
    configure_cpu()
    errors = []
    reference_errors = []
    displacements = []
    count = 0
    for clip in provenance["clips"]["clips"]:
        subset = {k: v for k, v in records.items() if k.startswith(clip["id"] + ":")}
        replayed, _, _ = run_clip(clip, "replay", subset)
        with np.load(
            reference / f'{clip["id"]}-poses.npz', allow_pickle=False
        ) as expected:
            np.testing.assert_array_equal(replayed["truth"], expected["truth"])
            np.testing.assert_array_equal(replayed["valid"], expected["valid"])
            mask = replayed["valid"]
            count += int(mask.sum())
            errors.extend(
                np.linalg.norm(
                    (replayed["predicted"] - replayed["truth"])[mask], axis=-1
                )
                .mean(axis=-1)
                .tolist()
            )
            reference_errors.extend(
                np.linalg.norm(
                    (expected["predicted"] - expected["truth"])[mask], axis=-1
                )
                .mean(axis=-1)
                .tolist()
            )
            displacements.extend(
                np.linalg.norm(
                    (replayed["predicted"] - expected["predicted"])[mask], axis=-1
                )
                .ravel()
                .tolist()
            )
    result = {
        "execution_kind": manifest["provenance"]["execution"]["execution_kind"],
        "tracked_hand_frames": count,
        "reference_mpjpe_mm": float(np.mean(reference_errors)),
        "imported_mpjpe_mm": float(np.mean(errors)),
        "absolute_mpjpe_change_mm": float(
            abs(np.mean(errors) - np.mean(reference_errors))
        ),
        "mean_joint_displacement_mm": float(np.mean(displacements)),
        "p95_joint_displacement_mm": float(np.percentile(displacements, 95)),
        "max_joint_displacement_mm": float(np.max(displacements)),
        "masks_and_ground_truth_match": True,
        "backend_verified": False,
        "note": "Fixture validates file plumbing only; real runtime/fallback evidence requires review.",
    }
    path = imported / "replay-summary.json"
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reference", type=Path, default=ROOT / "artifacts/reference-bundle"
    )
    parser.add_argument("--imported", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(evaluate_import(args.reference, args.imported), indent=2))
