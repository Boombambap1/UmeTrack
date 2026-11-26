"""Compare PyTorch and ONNX Runtime (CPU) encoder features on real uint8 crops.

Preprocessing follows the encoder contract: uint8 / 255 -> float32 [N, 1, 96, 96] (NCHW).
Each crop runs at batch size 1. Per-crop and overall max_abs, mean_abs and
rel_l2 = ||onnx - torch|| / ||torch|| are compared against tolerances.onnx_vs_torch
in the config, which this script never changes.
"""

import argparse
import json
import sys

import numpy as np
import onnxruntime as ort
import torch

from common import REPO, load_backbone, load_config, rel, sha256_file, update_hashes


def preprocess(crops_uint8: np.ndarray) -> np.ndarray:
    """uint8 [N, 96, 96] -> float32 [N, 1, 96, 96] NCHW, scaled by 1/255."""
    if crops_uint8.dtype != np.uint8 or crops_uint8.ndim != 3:
        raise ValueError(f"expected uint8 [N,H,W], got {crops_uint8.dtype} {crops_uint8.shape}")
    return (crops_uint8.astype(np.float32) / 255.0)[:, None, :, :]


def run_features(cfg, onnx_path, x: np.ndarray, threads: int = 1):
    """Run each crop through PyTorch and ORT at batch size 1; return (torch_feats, onnx_feats)."""
    enc = cfg["encoder"]
    torch.set_num_threads(threads)
    backbone, _, _ = load_backbone(cfg)
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    sess = ort.InferenceSession(str(onnx_path), so, providers=["CPUExecutionProvider"])
    assert sess.get_providers() == ["CPUExecutionProvider"], sess.get_providers()

    t_out, o_out = [], []
    with torch.no_grad():
        for i in range(x.shape[0]):
            xi = np.ascontiguousarray(x[i : i + 1])
            t_out.append(backbone(torch.from_numpy(xi)).numpy())
            (oi,) = sess.run([enc["output"]["name"]], {enc["input"]["name"]: xi})
            o_out.append(oi)
    return np.concatenate(t_out), np.concatenate(o_out)


def compare(t: np.ndarray, o: np.ndarray, tol: dict) -> dict:
    t64, o64 = t.astype(np.float64), o.astype(np.float64)
    d = np.abs(o64 - t64)
    n = t.shape[0]
    per_crop = []
    for i in range(n):
        di = d[i]
        per_crop.append(
            {
                "row": i,
                "max_abs": float(di.max()),
                "mean_abs": float(di.mean()),
                "rel_l2": float(np.linalg.norm(o64[i] - t64[i]) / np.linalg.norm(t64[i])),
            }
        )
    worst = {k: max(per_crop, key=lambda r: r[k]) for k in ("max_abs", "mean_abs", "rel_l2")}
    overall = {
        "max_abs": float(d.max()),
        "mean_abs": float(d.mean()),
        "rmse": float(np.sqrt(np.mean((o64 - t64) ** 2))),
        "rel_l2": float(np.linalg.norm(o64 - t64) / np.linalg.norm(t64)),
        "worst_crop_max_abs": {"row": worst["max_abs"]["row"], "value": worst["max_abs"]["max_abs"]},
        "worst_crop_rel_l2": {"row": worst["rel_l2"]["row"], "value": worst["rel_l2"]["rel_l2"]},
    }
    checks = {
        "max_abs": {"limit": tol["max_abs"], "worst": overall["max_abs"], "pass": overall["max_abs"] <= tol["max_abs"]},
        "rel_l2_per_crop": {
            "limit": tol["rel_l2"],
            "worst": overall["worst_crop_rel_l2"]["value"],
            "pass": overall["worst_crop_rel_l2"]["value"] <= tol["rel_l2"],
            "n_crops_over": sum(r["rel_l2"] > tol["rel_l2"] for r in per_crop),
        },
    }
    for r in per_crop:
        r["pass"] = r["max_abs"] <= tol["max_abs"] and r["rel_l2"] <= tol["rel_l2"]
    feature_range = {
        "min": float(t.min()),
        "max": float(t.max()),
        "mean": float(t.mean()),
        "std": float(t.std()),
        "frac_zero": float((t == 0).mean()),
    }
    return {
        "pass": all(c["pass"] for c in checks.values()),
        "checks": checks,
        "overall": overall,
        "torch_feature_range": feature_range,
        "per_crop": per_crop,
    }


def _markdown(s: dict) -> str:
    c, o, fr = s["checks"], s["overall"], s["torch_feature_range"]
    lines = [
        "# D5 — PyTorch vs ONNX Runtime encoder parity",
        "",
        f"**Result: {'PASS' if s['pass'] else 'FAIL'}** on {s['n_crops']} real crops "
        f"(`{s['crops']}`, frames from `{s['recording']}`), batch size 1, ORT CPUExecutionProvider, "
        f"{s['threads']} thread(s).",
        "",
        "Tolerances (`tolerances.onnx_vs_torch`, declared before this run, unchanged):",
        "",
        "| Check | Limit | Measured (worst) | Pass |",
        "| --- | --- | --- | --- |",
        f"| max_abs over all crops | {c['max_abs']['limit']:.1e} | {c['max_abs']['worst']:.3e} | {c['max_abs']['pass']} |",
        f"| rel_l2 per crop | {c['rel_l2_per_crop']['limit']:.1e} | {c['rel_l2_per_crop']['worst']:.3e} "
        f"(row {o['worst_crop_rel_l2']['row']}) | {c['rel_l2_per_crop']['pass']} |",
        "",
        f"Overall: max_abs {o['max_abs']:.3e}, mean_abs {o['mean_abs']:.3e}, rel_l2 {o['rel_l2']:.3e}.",
        "",
        f"PyTorch feature range: min {fr['min']:.4f}, max {fr['max']:.4f}, mean {fr['mean']:.4f}, "
        f"std {fr['std']:.4f}, fraction exactly zero {fr['frac_zero']:.3f}.",
        "",
        "Preprocessing: `uint8 / 255 -> float32 [N,1,96,96]` NCHW.",
        f"ONNX: `{s['onnx']}` sha256 `{s['onnx_sha256']}`.",
        f"Crops: sha256 `{s['crops_sha256']}`; manifest `manifests/crops_d5.csv`.",
        f"Versions: {', '.join(f'{k} {v}' for k, v in s['versions'].items())}.",
        "",
        "## Per crop",
        "",
        "| row | max_abs | mean_abs | rel_l2 | pass |",
        "| --- | --- | --- | --- | --- |",
    ]
    for r in s["per_crop"]:
        lines.append(f"| {r['row']} | {r['max_abs']:.3e} | {r['mean_abs']:.3e} | {r['rel_l2']:.3e} | {r['pass']} |")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/experiment.yaml")
    ap.add_argument("--onnx", default="artifacts/encoder.onnx")
    ap.add_argument("--crops", default="artifacts/crops/crops_uint8.npy")
    ap.add_argument("--limit", type=int, default=None, help="only the first N crops")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--out-dir", default="results/onnx_parity")
    ap.add_argument("--ref-dir", default="artifacts/reference")
    ap.add_argument("--hashes", default="manifests/hashes.txt")
    args = ap.parse_args()

    cfg = load_config(REPO / args.config)
    onnx_path, crops_path = REPO / args.onnx, REPO / args.crops
    crops = np.load(crops_path)
    if args.limit:
        crops = crops[: args.limit]
    x = preprocess(crops)

    t, o = run_features(cfg, onnx_path, x, threads=args.threads)
    want = [x.shape[0], *cfg["encoder"]["output"]["shape"][1:]]
    assert list(t.shape) == want and list(o.shape) == want, (t.shape, o.shape)
    assert t.dtype == np.float32 and o.dtype == np.float32

    result = compare(t, o, cfg["tolerances"]["onnx_vs_torch"])
    meta = json.loads((REPO / "results/crops_d5/crops_meta.json").read_text())
    summary = {
        "pass": result["pass"],
        "n_crops": int(x.shape[0]),
        "recording": meta["recording"],
        "onnx": rel(onnx_path),
        "onnx_sha256": sha256_file(onnx_path),
        "crops": rel(crops_path),
        "crops_sha256": sha256_file(crops_path),
        "preprocessing": "uint8 / 255 -> float32 [N,1,96,96] NCHW",
        "batch_size": 1,
        "threads": args.threads,
        "provider": "CPUExecutionProvider",
        "tolerances": cfg["tolerances"]["onnx_vs_torch"],
        "versions": {"torch": torch.__version__, "onnxruntime": ort.__version__, "numpy": np.__version__},
        **{k: v for k, v in result.items() if k != "pass"},
    }

    ref_dir, out_dir = REPO / args.ref_dir, REPO / args.out_dir
    ref_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    tp, op = ref_dir / "torch_features.npy", ref_dir / "onnx_features.npy"
    np.save(tp, t)
    np.save(op, o)
    sj, sm = out_dir / "summary.json", out_dir / "summary.md"
    sj.write_text(json.dumps(summary, indent=2) + "\n")
    sm.write_text(_markdown(summary))
    written = {rel(p): sha256_file(p) for p in (tp, op, sj, sm)}
    update_hashes(REPO / args.hashes, {k: v for k, v in written.items() if k.startswith("artifacts/")})

    c, ov = result["checks"], result["overall"]
    print(f"[parity] {'PASS' if result['pass'] else 'FAIL'}: {x.shape[0]} crops, batch 1, ORT CPU, {args.threads} thread(s)")
    print(f"[parity] max_abs  worst {c['max_abs']['worst']:.3e} (limit {c['max_abs']['limit']:.1e})")
    print(f"[parity] rel_l2   worst crop {c['rel_l2_per_crop']['worst']:.3e} (limit {c['rel_l2_per_crop']['limit']:.1e}), "
          f"crops over limit: {c['rel_l2_per_crop']['n_crops_over']}")
    print(f"[parity] overall  max_abs {ov['max_abs']:.3e}, mean_abs {ov['mean_abs']:.3e}, rel_l2 {ov['rel_l2']:.3e}")
    fr = result["torch_feature_range"]
    print(f"[parity] torch features: min {fr['min']:.4f} max {fr['max']:.4f} mean {fr['mean']:.4f} std {fr['std']:.4f}")
    for p, dgst in written.items():
        print(f"[parity] wrote {p}  sha256 {dgst}")
    sys.exit(0 if result["pass"] else 1)


if __name__ == "__main__":
    main()
