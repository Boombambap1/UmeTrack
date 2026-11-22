"""ORT INT8 preview — not SNPE/DSP, no pass/fail.

Statically quantizes artifacts/encoder.onnx with onnxruntime.quantization.quantize_static (QDQ format,
int8 activations and weights, per-channel weights, MinMax calibration) using ONLY the calibration
crops from make_calibration_set.py, then compares float ONNX vs INT8 ONNX features on the 60 D5
crops at batch 1 on the ORT CPU provider. This is a characterisation of how much INT8 rounding moves
the features, to set expectations before SNPE/DSP quantization (D8). It is not a DSP result and
has no tolerance or pass/fail; the SNPE quantizer and DSP arithmetic will differ.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType, quantize_static

from check_onnx_parity import preprocess
from common import REPO, load_config, rel, sha256_file

LABEL = "ORT INT8 preview — not SNPE/DSP, no pass/fail"


class CropReader(CalibrationDataReader):
    def __init__(self, x: np.ndarray, input_name: str):
        self._it = iter([{input_name: np.ascontiguousarray(x[i : i + 1])} for i in range(x.shape[0])])

    def get_next(self):
        return next(self._it, None)


def run(model_path: Path, x: np.ndarray, input_name: str, output_name: str) -> np.ndarray:
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    sess = ort.InferenceSession(str(model_path), so, providers=["CPUExecutionProvider"])
    assert sess.get_providers() == ["CPUExecutionProvider"], sess.get_providers()
    return np.concatenate(
        [sess.run([output_name], {input_name: np.ascontiguousarray(x[i : i + 1])})[0] for i in range(x.shape[0])]
    )


def metrics(ref: np.ndarray, cand: np.ndarray) -> dict:
    d = cand.astype(np.float64) - ref.astype(np.float64)
    return {
        "max_abs": float(np.abs(d).max()),
        "mean_abs": float(np.abs(d).mean()),
        "rmse": float(np.sqrt((d**2).mean())),
        "rel_l2": float(np.linalg.norm(d) / np.linalg.norm(ref.astype(np.float64))),
    }


def markdown(s: dict) -> str:
    o, fr = s["overall"], s["float_feature_range"]
    lines = [
        f"# {LABEL}",
        "",
        "Float ONNX vs ONNX Runtime static INT8 (QDQ) encoder features on the 60 D5 crops. Characterisation only: "
        "no tolerance, no pass/fail. The SNPE quantizer and Hexagon DSP arithmetic are different, so these numbers "
        "do not predict the D8 DSP result exactly.",
        "",
        "## Setup",
        "",
        f"- Float model: `{s['float_model']}` sha256 `{s['float_model_sha256']}`",
        f"- INT8 model (not in git): `{s['int8_model']}` sha256 `{s['int8_model_sha256']}`",
        f"- Quantization: onnxruntime {s['onnxruntime']} `quantize_static`, {s['quant']['format']}, activations "
        f"{s['quant']['activation_type']}, weights {s['quant']['weight_type']}, per_channel={s['quant']['per_channel']}, "
        f"calibration {s['quant']['calibrate_method']}",
        f"- Calibration: {s['calibration']['n_crops']} crops from `{s['calibration']['manifest']}` (training users "
        f"{', '.join(s['calibration']['users'])}; no overlap with evaluation), crops sha256 `{s['calibration']['crops_sha256']}`",
        f"- Evaluation: {s['n_eval_crops']} D5 crops `{s['eval_crops']}` sha256 `{s['eval_crops_sha256']}`, batch 1, "
        "ORT CPU provider, 1 thread",
        f"- Quantized graph: {s['int8_ops']}",
        "",
        "## Overall",
        "",
        "| max_abs | mean_abs (MAE) | RMSE | rel_l2 |",
        "| --- | --- | --- | --- |",
        f"| {o['max_abs']:.4e} | {o['mean_abs']:.4e} | {o['rmse']:.4e} | {o['rel_l2']:.4e} |",
        "",
        f"Float feature range: min {fr['min']:.4f}, max {fr['max']:.4f}, mean {fr['mean']:.4f}, std {fr['std']:.4f} "
        f"(span {fr['max'] - fr['min']:.4f}). Worst max_abs is {o['max_abs'] / (fr['max'] - fr['min']):.2%} of the span.",
        "",
        "Per-crop spread: "
        + ", ".join(f"{k} median {v['median']:.3e} / worst {v['worst']:.3e}" for k, v in s["per_crop_stats"].items()),
        "",
        "## Per crop",
        "",
        "| row | crop_id | max_abs | mean_abs (MAE) | RMSE | rel_l2 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for r in s["per_crop"]:
        lines.append(f"| {r['row']} | `{r['crop_id']}` | {r['max_abs']:.3e} | {r['mean_abs']:.3e} | {r['rmse']:.3e} | {r['rel_l2']:.3e} |")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/experiment.yaml")
    ap.add_argument("--onnx", default="artifacts/encoder.onnx")
    ap.add_argument("--calib-crops", default="artifacts/calibration/crops_uint8.npy")
    ap.add_argument("--calib-manifest", default="manifests/calibration.csv")
    ap.add_argument("--eval-crops", default="artifacts/crops/crops_uint8.npy")
    ap.add_argument("--eval-manifest", default="manifests/crops_d5.csv")
    ap.add_argument("--int8-out", default="artifacts/int8_preview/encoder_int8_qdq.onnx")
    ap.add_argument("--out-dir", default="results/int8_preview")
    args = ap.parse_args()

    from common import crop_id

    cfg = load_config(REPO / args.config)
    enc = cfg["encoder"]
    in_name, out_name = enc["input"]["name"], enc["output"]["name"]
    float_model = REPO / args.onnx

    with (REPO / args.calib_manifest).open() as f:
        calib_rows = list(csv.DictReader(f))
    calib = np.load(REPO / args.calib_crops)
    if calib.shape[0] != len(calib_rows):
        raise SystemExit("calibration crops and manifest disagree on N")
    with (REPO / args.eval_manifest).open() as f:
        eval_rows = list(csv.DictReader(f))
    eval_crops = np.load(REPO / args.eval_crops)
    eval_ids = [crop_id(r["recording"], r["frame"], r["hand_idx"], r["cam_idx"]) for r in eval_rows]
    calib_frames = {r["crop_id"].rsplit(":", 2)[0] for r in calib_rows}
    if calib_frames & {i.rsplit(":", 2)[0] for i in eval_ids}:
        raise SystemExit("calibration crops overlap evaluation crops")

    int8_model = REPO / args.int8_out
    int8_model.parent.mkdir(parents=True, exist_ok=True)
    quant = {
        "format": "QDQ",
        "activation_type": "QInt8",
        "weight_type": "QInt8",
        "per_channel": True,
        "calibrate_method": "MinMax",
    }
    quantize_static(
        str(float_model),
        str(int8_model),
        CropReader(preprocess(calib), in_name),
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
        calibrate_method=CalibrationMethod.MinMax,
    )
    ops = {}
    for n in onnx.load(str(int8_model)).graph.node:
        ops[n.op_type] = ops.get(n.op_type, 0) + 1

    x = preprocess(eval_crops)
    f32 = run(float_model, x, in_name, out_name)
    i8 = run(int8_model, x, in_name, out_name)
    if f32.shape != i8.shape or f32.shape[1:] != tuple(enc["output"]["shape"][1:]):
        raise SystemExit(f"shape mismatch {f32.shape} vs {i8.shape}")

    per_crop = [{"row": i, "crop_id": eval_ids[i], **metrics(f32[i], i8[i])} for i in range(x.shape[0])]
    summary = {
        "label": LABEL,
        "pass_fail": None,
        "onnxruntime": ort.__version__,
        "float_model": rel(float_model),
        "float_model_sha256": sha256_file(float_model),
        "int8_model": rel(int8_model),
        "int8_model_sha256": sha256_file(int8_model),
        "int8_ops": ", ".join(f"{k} {v}" for k, v in sorted(ops.items())),
        "quant": quant,
        "calibration": {
            "manifest": args.calib_manifest,
            "n_crops": int(calib.shape[0]),
            "users": sorted({Path(r["recording"]).parts[-2] for r in calib_rows}),
            "crops_sha256": sha256_file(REPO / args.calib_crops),
        },
        "eval_crops": args.eval_crops,
        "eval_crops_sha256": sha256_file(REPO / args.eval_crops),
        "n_eval_crops": int(x.shape[0]),
        "overall": metrics(f32, i8),
        "float_feature_range": {
            "min": float(f32.min()), "max": float(f32.max()), "mean": float(f32.mean()), "std": float(f32.std()),
        },
        "per_crop_stats": {
            k: {"median": float(np.median([r[k] for r in per_crop])), "worst": float(max(r[k] for r in per_crop))}
            for k in ("max_abs", "mean_abs", "rmse", "rel_l2")
        },
        "per_crop": per_crop,
    }
    out_dir = REPO / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    sm = out_dir / "summary.md"
    sm.write_text(markdown(summary))
    sj = REPO / "artifacts/int8_preview/summary.json"
    sj.write_text(json.dumps(summary, indent=2) + "\n")

    o = summary["overall"]
    print(f"[int8] {LABEL}")
    print(f"[int8] overall max_abs {o['max_abs']:.4e}, mean_abs {o['mean_abs']:.4e}, rmse {o['rmse']:.4e}, rel_l2 {o['rel_l2']:.4e}")
    print(f"[int8] float range [{summary['float_feature_range']['min']:.4f}, {summary['float_feature_range']['max']:.4f}]")
    print(f"[int8] per crop: {summary['per_crop_stats']}")
    print(f"[int8] quantized ops: {summary['int8_ops']}")
    for p in (int8_model, sm, sj):
        print(f"[int8] wrote {rel(p)}  sha256 {sha256_file(p)}")


if __name__ == "__main__":
    main()
