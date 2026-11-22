import json
from pathlib import Path

import numpy as np

from umetrack_rb5.raw_exchange import read_reference, sha256, to_layout


def write_fixture(reference, package, returned, layout):
    returned = Path(returned)
    returned.mkdir(parents=True, exist_ok=False)
    provenance, records = read_reference(reference)
    manifest = json.loads((Path(package) / "package.json").read_text())
    for row in manifest["crops"]:
        features = records[row["frame_key"]]["features"][
            row["crop_index"] : row["crop_index"] + 1
        ]
        path = returned / f'Result_{row["index"]}' / "features.raw"
        path.parent.mkdir()
        np.asarray(to_layout(features, layout), dtype="<f4").tofile(path)
    (returned / "fixture.txt").write_text(
        "Mac reference feature file round-trip fixture. No Qualcomm software or board execution.\n"
    )
    execution = {
        "execution_kind": "mac-reference-fixture",
        "device": "Mac; copying saved features only",
        "sdk_version": "not-applicable-fixture",
        "model_sha256": provenance.get("checkpoint_sha256", "0" * 64),
        "requested_backend": "mac-reference",
        "reported_backend": "mac-reference",
        "output_layout": layout,
        "output_dtype": "<f4",
        "output_file": "features.raw",
        "package_sha256": sha256(Path(package) / "package.json"),
        "input_list_sha256": manifest["input_list_sha256"],
        "command": "write_fixture: copy canonical reference features to explicit-layout raw files",
        "evidence": [
            {"file": "fixture.txt", "sha256": sha256(returned / "fixture.txt")}
        ],
    }
    (returned / "execution.json").write_text(json.dumps(execution, indent=2) + "\n")
