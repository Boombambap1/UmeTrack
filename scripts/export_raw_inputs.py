import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from umetrack_rb5.raw_exchange import export_inputs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reference", type=Path, default=ROOT / "artifacts/reference-bundle"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--input-layout", choices=["NCHW", "NHWC"], required=True)
    args = parser.parse_args()
    result = export_inputs(args.reference, args.output, args.input_layout)
    print(
        json.dumps(
            {
                "crops": len(result["crops"]),
                "input_shape": result["input_shape"],
                "bytes": sum(r["input_bytes"] for r in result["crops"]),
                "note": result["purpose"],
            },
            indent=2,
        )
    )
