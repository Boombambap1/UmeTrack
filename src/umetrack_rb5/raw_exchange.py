import copy
import hashlib
import json
from pathlib import Path

import numpy as np

from .encoder_provider import load_records, save_records, tensor_digest


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_reference(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "features/manifest.json").read_text())
    records = load_records(directory / "features", manifest["provenance"])
    return manifest["provenance"], records


def to_layout(array, layout):
    if layout == "NCHW":
        return array
    if layout == "NHWC":
        return array.transpose(0, 2, 3, 1)
    raise ValueError("Layout must be NCHW or NHWC")


def crop_rows(records):
    rows = []
    for key, record in sorted(records.items()):
        for index, crop_id in enumerate(record["crop_ids"]):
            rows.append(
                {
                    "index": len(rows),
                    "frame_key": key,
                    "crop_index": index,
                    "crop_id": crop_id,
                }
            )
    if len({r["crop_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate crop identities")
    return rows


def export_inputs(reference, output, input_layout):
    reference, output = Path(reference), Path(output)
    provenance, records = read_reference(reference)
    if input_layout not in ("NCHW", "NHWC"):
        raise ValueError("Explicit input layout required")
    output.mkdir(parents=True, exist_ok=False)
    (output / "inputs").mkdir()
    rows = crop_rows(records)
    lines = []
    for row in rows:
        crop = records[row["frame_key"]]["inputs"][
            row["crop_index"] : row["crop_index"] + 1
        ]
        array = np.asarray(to_layout(crop, input_layout), dtype="<f4")
        path = f'inputs/crop-{row["index"]:05d}.raw'
        array.tofile(output / path)
        row.update(
            {
                "input_file": path,
                "input_sha256": sha256(output / path),
                "input_bytes": array.nbytes,
            }
        )
        lines.append(path)
    (output / "input_list.txt").write_text("\n".join(lines) + "\n")
    manifest = {
        "schema_version": 1,
        "purpose": "batch-one candidate SNPE inputs; DLC metadata must be confirmed before execution",
        "reference_manifest_sha256": sha256(reference / "features/manifest.json"),
        "reference_provenance": provenance,
        "input_layout": input_layout,
        "input_dtype": "<f4",
        "input_shape": [1, 1, 96, 96] if input_layout == "NCHW" else [1, 96, 96, 1],
        "input_list_sha256": sha256(output / "input_list.txt"),
        "crops": rows,
    }
    (output / "package.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def validate_package(reference, package):
    reference, package = Path(reference), Path(package)
    provenance, records = read_reference(reference)
    manifest = json.loads((package / "package.json").read_text())
    if (
        manifest["schema_version"] != 1
        or manifest["reference_manifest_sha256"]
        != sha256(reference / "features/manifest.json")
        or manifest["reference_provenance"] != provenance
    ):
        raise ValueError("Wrong reference package/provenance")
    layout = manifest["input_layout"]
    if layout not in ("NCHW", "NHWC") or manifest["input_dtype"] != "<f4":
        raise ValueError("Unsupported input layout/dtype")
    shape = [1, 1, 96, 96] if layout == "NCHW" else [1, 96, 96, 1]
    if manifest["input_shape"] != shape:
        raise ValueError("Wrong input shape")
    expected = crop_rows(records)
    if len(expected) != len(manifest["crops"]):
        raise ValueError("Missing/extra crop rows")
    for wanted, row in zip(expected, manifest["crops"]):
        if any(row.get(k) != value for k, value in wanted.items()):
            raise ValueError("Crop identity/order mismatch")
        name = f'inputs/crop-{wanted["index"]:05d}.raw'
        if row["input_file"] != name:
            raise ValueError("Unexpected input path")
        path = package / name
        crop = records[row["frame_key"]]["inputs"][
            row["crop_index"] : row["crop_index"] + 1
        ]
        expected_bytes = np.asarray(to_layout(crop, layout), dtype="<f4").tobytes(
            order="C"
        )
        if (
            path.read_bytes() != expected_bytes
            or row["input_bytes"] != len(expected_bytes)
            or sha256(path) != row["input_sha256"]
        ):
            raise ValueError("Raw input content/size/hash mismatch")
    if {
        p.relative_to(package).as_posix() for p in (package / "inputs").glob("*.raw")
    } != {r["input_file"] for r in manifest["crops"]}:
        raise ValueError("Unexpected raw input files")
    text = "".join(r["input_file"] + "\n" for r in manifest["crops"])
    if (package / "input_list.txt").read_text() != text or sha256(
        package / "input_list.txt"
    ) != manifest["input_list_sha256"]:
        raise ValueError("Input list mismatch")
    return manifest, records


def import_outputs(reference, package, returned, output):
    reference, package, returned, output = map(
        Path, (reference, package, returned, output)
    )
    manifest, records = validate_package(reference, package)
    execution = json.loads((returned / "execution.json").read_text())
    required = (
        "execution_kind",
        "device",
        "sdk_version",
        "model_sha256",
        "requested_backend",
        "reported_backend",
        "output_layout",
        "output_dtype",
        "output_file",
        "package_sha256",
        "input_list_sha256",
        "command",
        "evidence",
    )
    if any(not execution.get(k) for k in required):
        raise ValueError("Incomplete execution provenance")
    if execution["execution_kind"] not in ("board", "mac-reference-fixture"):
        raise ValueError("Unknown execution kind")
    if (
        execution["package_sha256"] != sha256(package / "package.json")
        or execution["input_list_sha256"] != manifest["input_list_sha256"]
    ):
        raise ValueError("Returned outputs belong to a different package/input list")
    if execution["execution_kind"] == "board":
        if execution.get("model_format") != "dlc":
            raise ValueError("Board run must identify the DLC model")
        if execution.get("conversion_sdk_version") != execution["sdk_version"]:
            raise ValueError("Conversion/runtime SDK versions must match")
        if (
            execution.get("source_checkpoint_sha256")
            != manifest["reference_provenance"]["checkpoint_sha256"]
        ):
            raise ValueError("Wrong or missing source checkpoint provenance")
    model_hash = execution["model_sha256"]
    if len(model_hash) != 64 or any(c not in "0123456789abcdef" for c in model_hash):
        raise ValueError("Invalid model hash")
    if execution["output_dtype"] != "<f4" or execution["output_layout"] not in (
        "NCHW",
        "NHWC",
    ):
        raise ValueError(
            "Only explicit little-endian float32 NCHW/NHWC outputs supported"
        )
    filename = execution["output_file"]
    if Path(filename).name != filename or not filename.endswith(".raw"):
        raise ValueError("Expected explicit flat output tensor filename ending .raw")
    for evidence in execution["evidence"]:
        path = (returned / evidence["file"]).resolve()
        if (
            not path.is_relative_to(returned.resolve())
            or sha256(path) != evidence["sha256"]
        ):
            raise ValueError("Invalid/missing execution evidence")
    expected_paths = {f'Result_{r["index"]}/{filename}' for r in manifest["crops"]}
    actual = {p.relative_to(returned).as_posix() for p in returned.rglob("*.raw")}
    if actual != expected_paths:
        raise ValueError("Missing/extra raw result files")
    imported = copy.deepcopy(records)
    errors = []
    source_hashes = {}
    for row in manifest["crops"]:
        path = returned / f'Result_{row["index"]}' / filename
        if path.stat().st_size != 72 * 6 * 6 * 4:
            raise ValueError("Wrong output byte count")
        array = np.fromfile(path, dtype="<f4")
        if not np.isfinite(array).all():
            raise ValueError("Nonfinite output features")
        if execution["output_layout"] == "NHWC":
            array = array.reshape(1, 6, 6, 72).transpose(0, 3, 1, 2)
        else:
            array = array.reshape(1, 72, 6, 6)
        key, index = row["frame_key"], row["crop_index"]
        errors.append(
            (array - records[key]["features"][index : index + 1])
            .astype(np.float64)
            .ravel()
        )
        imported[key]["features"][index] = array[0]
        source_hashes[row["crop_id"]] = sha256(path)
    for record in imported.values():
        record["feature_digest"] = tensor_digest(record["features"])
    provenance = {
        "reference": manifest["reference_provenance"],
        "execution": execution,
        "raw_output_sha256_by_crop": source_hashes,
    }
    output.mkdir(parents=True, exist_ok=False)
    save_records(imported, output / "features", provenance)
    errors = np.concatenate(errors)
    summary = {
        "execution_kind": execution["execution_kind"],
        "crops": len(manifest["crops"]),
        "feature_mae": float(np.abs(errors).mean()),
        "feature_rmse": float(np.sqrt(np.square(errors).mean())),
        "feature_max_absolute_error": float(np.abs(errors).max()),
        "backend_status": "reported in execution.json; runtime logs still require review for fallback",
        "package_sha256": sha256(package / "package.json"),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary
