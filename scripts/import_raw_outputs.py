import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from umetrack_rb5.raw_exchange import import_outputs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reference", type=Path, default=ROOT / "artifacts/reference-bundle"
    )
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--returned", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            import_outputs(args.reference, args.package, args.returned, args.output),
            indent=2,
        )
    )
