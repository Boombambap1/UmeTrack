import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn


def tensor_digest(array):
    array = np.ascontiguousarray(array)
    h = hashlib.sha256()
    h.update(str(array.dtype).encode())
    h.update(json.dumps(list(array.shape)).encode())
    h.update(array.tobytes())
    return h.hexdigest()


def metadata_digest(metadata):
    return hashlib.sha256(
        json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


class EncoderProvider(nn.Module):
    def __init__(self, backbone, mode, records=None):
        super().__init__()
        if mode not in ("normal", "capture", "replay"):
            raise ValueError(mode)
        self.backbone = backbone
        self.mode = mode
        self.records = {} if records is None else records
        self.seen = set()
        self.context = None

    def set_context(self, key, crop_ids, metadata):
        self.context = (key, list(crop_ids), metadata)

    def forward(self, crops):
        if self.context is None:
            raise RuntimeError("Missing sequence/frame context")
        key, crop_ids, metadata = self.context
        self.context = None
        if key in self.seen:
            raise RuntimeError(f"Duplicate encoder call: {key}")
        self.seen.add(key)
        array = crops.detach().cpu().numpy()
        if array.dtype != np.float32 or array.shape[1:] != (1, 96, 96):
            raise ValueError("Expected float32 NCHW monochrome crops")
        if len(crop_ids) != len(array) or len(set(crop_ids)) != len(crop_ids):
            raise ValueError("Crop IDs must be unique and match batch length")
        if not np.isfinite(array).all():
            raise ValueError("Nonfinite input crop")
        identity = {
            "crop_ids": crop_ids,
            "input_digest": tensor_digest(array),
            "metadata_digest": metadata_digest(metadata),
        }
        if self.mode == "replay":
            if key not in self.records:
                raise KeyError(f"Missing features: {key}")
            record = self.records[key]
            if any(record[k] != v for k, v in identity.items()):
                raise ValueError(f"Input/order/metadata mismatch: {key}")
            features = record["features"]
            if features.dtype != np.float32 or features.shape != (len(array), 72, 6, 6):
                raise ValueError(f"Invalid feature shape/dtype: {key}")
            if (
                not np.isfinite(features).all()
                or tensor_digest(features) != record["feature_digest"]
            ):
                raise ValueError(f"Corrupt/nonfinite features: {key}")
            return torch.from_numpy(features.copy()).to(
                device=crops.device, dtype=crops.dtype
            )
        features = self.backbone(crops)
        if self.mode == "capture":
            value = features.detach().cpu().numpy().copy()
            self.records[key] = {
                **identity,
                "metadata": metadata,
                "inputs": array.copy(),
                "features": value,
                "feature_digest": tensor_digest(value),
            }
        return features

    def assert_consumed(self):
        if self.mode == "replay" and self.seen != set(self.records):
            raise RuntimeError("Unused or missing replay records")


def save_records(records, directory, provenance):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    manifest = {"schema_version": 1, "provenance": provenance, "records": []}
    for index, (key, record) in enumerate(sorted(records.items())):
        filename = f"features-{index:04d}.npz"
        path = directory / filename
        np.savez(path, inputs=record["inputs"], features=record["features"])
        manifest["records"].append(
            {k: v for k, v in record.items() if k not in ("inputs", "features")}
            | {
                "key": key,
                "file": filename,
                "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def load_records(directory, expected_provenance):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["schema_version"] != 1 or manifest["provenance"] != expected_provenance:
        raise ValueError("Bundle schema/provenance mismatch")
    records = {}
    filenames = []
    for entry in manifest["records"]:
        if entry["key"] in records or Path(entry["file"]).name != entry["file"]:
            raise ValueError("Duplicate record ID or invalid file path")
        path = directory / entry["file"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry["file_sha256"]:
            raise ValueError(f"File checksum mismatch: {path.name}")
        with np.load(path, allow_pickle=False) as arrays:
            value = dict(
                entry,
                inputs=arrays["inputs"].copy(),
                features=arrays["features"].copy(),
            )
        if (
            tensor_digest(value["inputs"]) != entry["input_digest"]
            or tensor_digest(value["features"]) != entry["feature_digest"]
        ):
            raise ValueError("Tensor checksum mismatch")
        records[entry["key"]] = value
        filenames.append(entry["file"])
    if len(filenames) != len(set(filenames)) or set(filenames) != {
        p.name for p in directory.glob("*.npz")
    }:
        raise ValueError("Extra or duplicate feature files")
    return records
