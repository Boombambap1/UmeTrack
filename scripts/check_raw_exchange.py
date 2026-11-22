import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))
from umetrack_rb5.raw_exchange import export_inputs, import_outputs
from raw_fixtures import write_fixture
from replay_imported_features import evaluate_import

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reference", type=Path, default=ROOT / "artifacts/reference-bundle"
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "artifacts/raw-exchange-check"
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    results = []
    for layout in ("NCHW", "NHWC"):
        directory = args.output / layout
        directory.mkdir()
        package = directory / "package"
        returned = directory / "fixture-return"
        imported = directory / "imported"
        export_inputs(args.reference, package, layout)
        write_fixture(args.reference, package, returned, layout)
        feature_result = import_outputs(args.reference, package, returned, imported)
        pose_result = evaluate_import(args.reference, imported)
        if (
            feature_result["feature_max_absolute_error"] != 0
            or pose_result["max_joint_displacement_mm"] != 0
        ):
            raise RuntimeError("Reference round-trip changed outputs")
        results.append(
            {"layout": layout, "features": feature_result, "poses": pose_result}
        )
        print(
            f"PASS {layout}: 335 reference crops -> raw files -> imported features -> identical poses; no board execution",
            flush=True,
        )
    (args.output / "summary.json").write_text(
        json.dumps(
            {"execution_kind": "mac-reference-fixture", "layouts": results}, indent=2
        )
        + "\n"
    )
