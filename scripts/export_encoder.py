"""Export UmeTrack's image encoder (_feature_extractor._image_backbone) to ONNX.

Tries the dynamo exporter first and falls back to the legacy TorchScript exporter,
recording which one was used and why. Checks the result with onnx.checker and a
single ONNX Runtime run, then writes manifests/hashes.txt and results/export/export_meta.json.
Never deletes an existing model: exports to a temp file, verifies it, then renames it into place;
an existing output is refused unless --force, which keeps it as a timestamped backup.
"""

import argparse
import datetime
import json
import os
import traceback
import warnings
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import onnxscript
import torch

from common import REPO, load_backbone, load_config, rel, sha256_file, update_hashes


def _export(backbone, x, out, opset, in_name, out_name, dynamo):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        kwargs = dict(
            input_names=[in_name],
            output_names=[out_name],
            opset_version=opset,
            dynamo=dynamo,
        )
        if dynamo:
            kwargs["external_data"] = False
        torch.onnx.export(backbone, (x,), str(out), **kwargs)
    return sorted({f"{w.category.__name__}: {w.message}" for w in caught})


def write_atomically(out: Path, produce, force: bool = False):
    """Create `out` via produce(tmp_path) without ever deleting an existing file.

    produce() writes and verifies the candidate at a temp path in the same folder and returns any value.
    If `out` exists: refuse unless force; with force, the old file is kept as `<name>.bak-<UTC time>`.
    The temp file is only renamed over `out` (os.replace, atomic on one filesystem) after produce()
    returns; if produce() raises, only this run's temp file is removed and `out` is untouched.
    Returns (produce_result, backup_path_or_None).
    """
    out = Path(out)
    if out.exists() and not force:
        raise SystemExit(f"{out} exists; refusing to overwrite (use --force to keep it as a timestamped backup)")
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f".{out.name}.tmp-{os.getpid()}")
    if tmp.exists():
        raise SystemExit(f"leftover temp file {tmp}; inspect it first")
    try:
        result = produce(tmp)
    except BaseException:
        tmp.unlink(missing_ok=True)
        tmp.with_name(tmp.name + ".data").unlink(missing_ok=True)
        raise
    backup = None
    if out.exists():
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        backup = out.with_name(f"{out.name}.bak-{stamp}")
        if backup.exists():
            raise SystemExit(f"backup {backup} already exists; not overwriting it")
        os.replace(out, backup)
    os.replace(tmp, out)
    return result, backup


def _io_info(values):
    info = []
    for v in values:
        t = v.type.tensor_type
        dims = [d.dim_value if d.HasField("dim_value") else d.dim_param for d in t.shape.dim]
        info.append({"name": v.name, "shape": dims, "dtype": onnx.TensorProto.DataType.Name(t.elem_type)})
    return info


def _validate(path, cfg, backbone, x):
    """Return (model_info, problems). problems is empty if the export meets the contract."""
    enc = cfg["encoder"]
    model = onnx.load(str(path))
    onnx.checker.check_model(model, full_check=True)
    problems = []

    opset = {o.domain: o.version for o in model.opset_import}.get("", None)
    if opset != enc["onnx_opset"]:
        problems.append(f"default-domain opset is {opset}, expected {enc['onnx_opset']}")

    init_names = {i.name for i in model.graph.initializer}
    inputs = _io_info([i for i in model.graph.input if i.name not in init_names])
    outputs = _io_info(model.graph.output)
    want_in = {"name": enc["input"]["name"], "shape": enc["input"]["shape"], "dtype": "FLOAT"}
    want_out = {"name": enc["output"]["name"], "shape": enc["output"]["shape"], "dtype": "FLOAT"}
    if inputs != [want_in]:
        problems.append(f"inputs {inputs} != {[want_in]}")
    if outputs != [want_out]:
        problems.append(f"outputs {outputs} != {[want_out]}")

    ops = {n.op_type for n in model.graph.node}
    for fn in model.functions:
        ops |= {n.op_type for n in fn.node}

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    (y_onnx,) = sess.run([enc["output"]["name"]], {enc["input"]["name"]: x.numpy()})
    with torch.no_grad():
        y_torch = backbone(x).numpy()
    if list(y_onnx.shape) != enc["output"]["shape"]:
        problems.append(f"ORT output shape {list(y_onnx.shape)}")

    info = {
        "opset": opset,
        "ir_version": model.ir_version,
        "inputs": inputs,
        "outputs": outputs,
        "op_types": sorted(ops),
        "n_nodes": len(model.graph.node),
        "n_functions": len(model.functions),
        "smoke_zeros_max_abs_vs_torch": float(np.max(np.abs(y_onnx - y_torch))),
    }
    return info, problems


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/experiment.yaml")
    ap.add_argument("--out", default="artifacts/encoder.onnx")
    ap.add_argument("--hashes", default="manifests/hashes.txt")
    ap.add_argument("--meta", default="results/export/export_meta.json")
    ap.add_argument("--exporter", choices=["auto", "dynamo", "legacy"], default="auto")
    ap.add_argument("--force", action="store_true", help="replace an existing output, keeping it as <name>.bak-<UTC time>")
    args = ap.parse_args()

    cfg = load_config(REPO / args.config)
    enc = cfg["encoder"]
    out = (REPO / args.out).resolve()
    if out.exists() and not args.force:
        raise SystemExit(f"{rel(out)} exists; refusing to overwrite (use --force to keep it as a timestamped backup)")

    torch.manual_seed(0)
    backbone, ckpt, ckpt_sha = load_backbone(cfg)
    x = torch.zeros(enc["input"]["shape"], dtype=torch.float32)

    order = {"auto": [True, False], "dynamo": [True], "legacy": [False]}[args.exporter]
    attempts = []

    def produce(tmp):
        """Export to tmp (only this run's temp file is ever removed), check, and return the chosen attempt."""
        for dynamo in order:
            name = "dynamo" if dynamo else "legacy"
            attempt = {"exporter": name}
            try:
                for own in (tmp, tmp.with_name(tmp.name + ".data")):
                    own.unlink(missing_ok=True)
                attempt["warnings"] = _export(
                    backbone, x, tmp, enc["onnx_opset"], enc["input"]["name"], enc["output"]["name"], dynamo
                )
                if tmp.with_name(tmp.name + ".data").exists():
                    raise RuntimeError("unexpected external data file next to the ONNX model")
                info, problems = _validate(tmp, cfg, backbone, x)  # includes onnx.checker full_check
                attempt["problems"] = problems
                if not problems:
                    attempts.append(attempt)
                    return name, info, attempt, sha256_file(tmp)
            except Exception as e:  # record and fall back
                attempt["error"] = f"{type(e).__name__}: {e}"
                attempt["traceback_tail"] = traceback.format_exc().splitlines()[-6:]
            attempts.append(attempt)
            print(f"[export] {name} exporter rejected: {attempt.get('error') or attempt.get('problems')}")
        raise SystemExit(f"all exporters failed: {json.dumps(attempts, indent=2)}")

    (name, info, attempt, onnx_sha), backup = write_atomically(out, produce, force=args.force)
    if sha256_file(out) != onnx_sha:
        raise SystemExit("sha256 changed between verification and rename")

    update_hashes(REPO / args.hashes, {rel(ckpt): ckpt_sha, rel(out): onnx_sha})

    meta = {
        "upstream_commit": cfg["upstream"]["commit"],
        "checkpoint": rel(ckpt),
        "checkpoint_sha256": ckpt_sha,
        "module": enc["module"],
        "onnx_path": rel(out),
        "onnx_sha256": onnx_sha,
        "onnx_bytes": out.stat().st_size,
        "exporter": name,
        "exporter_attempts": attempts,
        "notes": (
            [
                "dynamo exporter builds at opset 18 (its minimum) and down-converts to the "
                f"requested opset {info['opset']} with the onnx C API version converter; "
                "eval-mode BatchNorm is folded into Conv by the exporter's optimizer."
            ]
            if name == "dynamo" and info["opset"] < 18
            else []
        ),
        "versions": {
            "torch": torch.__version__,
            "onnx": onnx.__version__,
            "onnxscript": onnxscript.__version__,
            "onnxruntime": ort.__version__,
        },
        **info,
    }
    meta_path = REPO / args.meta
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")

    print(f"[export] exporter: {name}")
    for w in attempt.get("warnings", []):
        print(f"[export]   warning: {w}")
    print(f"[export] opset {info['opset']}, ir_version {info['ir_version']}, {info['n_nodes']} nodes")
    print(f"[export] inputs:  {info['inputs']}")
    print(f"[export] outputs: {info['outputs']}")
    print(f"[export] op types: {', '.join(info['op_types'])}")
    print(f"[export] zeros smoke test max_abs vs torch: {info['smoke_zeros_max_abs_vs_torch']:.3e}")
    if backup:
        print(f"[export] previous file kept as {rel(backup)}  sha256 {sha256_file(backup)}")
    print(f"[export] wrote {rel(out)}  sha256 {onnx_sha}")
    print(f"[export] wrote {rel(REPO / args.hashes)}  sha256 {sha256_file(REPO / args.hashes)}")
    print(f"[export] wrote {rel(meta_path)}  sha256 {sha256_file(meta_path)}")


if __name__ == "__main__":
    main()
