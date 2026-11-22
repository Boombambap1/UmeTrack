"""Read snpe-net-run outputs, convert to NCHW explicitly, and compare with ONNX reference features.

Output dims and layout are NOT assumed: pass --output-dims and --output-layout exactly as
snpe-dlc-info reports them. Rows come from the round-trip manifest written by write_snpe_inputs.py.
Compares against tolerances.board_cpu_vs_onnx (never changed here). Also records the requested and
actual runtime; a silent CPU fallback on an accelerator run must be reported, not hidden.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

from check_onnx_parity import compare
from common import REPO, load_config, rel, sha256_file
from write_snpe_inputs import LAYOUTS, parse_dims


def to_nchw(arr: np.ndarray, layout: str) -> np.ndarray:
    if layout == "NCHW":
        return np.ascontiguousarray(arr)
    if layout == "NHWC":
        return np.ascontiguousarray(arr.transpose(0, 3, 1, 2))
    raise ValueError(f"unknown layout {layout}")


def read_outputs(outputs_dir: Path, n: int, output_file: str, dims, layout: str) -> np.ndarray:
    """Read Result_<i>/<output_file> for i in [0, n); return float32 [n, C, H, W]."""
    expected = int(np.prod(dims))
    feats = []
    for i in range(n):
        path = outputs_dir / f"Result_{i}" / output_file
        if not path.exists():
            have = sorted(p.name for p in (outputs_dir / f"Result_{i}").glob("*")) if (outputs_dir / f"Result_{i}").exists() else []
            raise SystemExit(f"missing {path}; Result_{i} contains {have}")
        raw = np.fromfile(path, dtype="<f4")
        if raw.size != expected:
            raise SystemExit(f"{path}: {raw.size} floats, expected {expected} for dims {dims}")
        feats.append(to_nchw(raw.reshape(dims), layout))
    return np.concatenate(feats).astype(np.float32)


def _markdown(s: dict) -> str:
    c, o = s["checks"], s["overall"]
    lines = [
        "# D6 — RB5 CPU vs ONNX Runtime encoder features",
        "",
        f"**Result: {'PASS' if s['pass'] else 'FAIL'}** on {s['n_crops']} crops "
        f"(rows {s['rows']}), runtime requested `{s['requested_runtime']}`, actual `{s['actual_runtime']}`.",
        "",
        "Tolerances (`tolerances.board_cpu_vs_onnx`, declared before this run, unchanged):",
        "",
        "| Check | Limit | Measured (worst) | Pass |",
        "| --- | --- | --- | --- |",
        f"| max_abs over all crops | {c['max_abs']['limit']:.1e} | {c['max_abs']['worst']:.3e} | {c['max_abs']['pass']} |",
        f"| rel_l2 per crop | {c['rel_l2_per_crop']['limit']:.1e} | {c['rel_l2_per_crop']['worst']:.3e} "
        f"(idx {o['worst_crop_rel_l2']['row']}) | {c['rel_l2_per_crop']['pass']} |",
        "",
        f"Overall: max_abs {o['max_abs']:.3e}, mean_abs {o['mean_abs']:.3e}, rel_l2 {o['rel_l2']:.3e}.",
        f"Diagnostic, board vs PyTorch: max_abs {s['vs_torch']['max_abs']:.3e}, rel_l2 {s['vs_torch']['rel_l2']:.3e}.",
        "",
        f"Board output: `{s['output_file']}` dims {s['output_dims']} layout {s['output_layout']} "
        "-> explicit transpose to NCHW [1,72,6,6].",
        f"Reference: `{s['reference']}` sha256 `{s['reference_sha256']}`.",
        f"Run log: `{s['run_log']}` sha256 `{s['run_log_sha256']}`.",
        "",
        "## Per crop",
        "",
        "| idx | manifest row | crop_id | max_abs | mean_abs | rel_l2 | pass |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in s["per_crop"]:
        lines.append(
            f"| {r['row']} | {s['rows'][r['row']]} | `{r.get('crop_id', '')}` | {r['max_abs']:.3e} | {r['mean_abs']:.3e} | {r['rel_l2']:.3e} | {r['pass']} |"
        )
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/experiment.yaml")
    ap.add_argument("--outputs-dir", default="results/raw/board_cpu_d6/output")
    ap.add_argument("--output-file", default=None, help="raw file name inside Result_<i>/ (default <output name>.raw)")
    ap.add_argument("--output-dims", required=True, help="as reported by snpe-dlc-info, e.g. 1,6,6,72")
    ap.add_argument("--output-layout", required=True, choices=LAYOUTS)
    ap.add_argument("--roundtrip-manifest", default="manifests/crops_d6_roundtrip.csv")
    ap.add_argument("--reference", default="artifacts/reference/onnx_features.npy")
    ap.add_argument("--torch-reference", default="artifacts/reference/torch_features.npy")
    ap.add_argument("--requested-runtime", default="cpu")
    ap.add_argument("--actual-runtime", required=True, help="runtime confirmed from the snpe-net-run logs")
    ap.add_argument("--run-log", default="results/raw/board_cpu_d6/snpe-net-run.log")
    ap.add_argument("--out-dir", default="results/board_cpu_d6")
    args = ap.parse_args()

    cfg = load_config(REPO / args.config)
    output_file = args.output_file or f"{cfg['encoder']['output']['name']}.raw"
    dims = parse_dims(args.output_dims)
    with (REPO / args.roundtrip_manifest).open() as f:
        rt_rows = list(csv.DictReader(f))
    rows = [int(r["row"]) for r in rt_rows]
    crop_ids = [r.get("crop_id", "") for r in rt_rows]

    board = read_outputs(REPO / args.outputs_dir, len(rows), output_file, dims, args.output_layout)
    ref = np.load(REPO / args.reference)[rows]
    want = [len(rows), *cfg["encoder"]["output"]["shape"][1:]]
    if list(board.shape) != want or list(ref.shape) != want:
        raise SystemExit(f"shape mismatch: board {list(board.shape)}, reference {list(ref.shape)}, want {want}")

    result = compare(ref, board, cfg["tolerances"]["board_cpu_vs_onnx"])
    torch_ref = np.load(REPO / args.torch_reference)[rows].astype(np.float64)
    vs_torch = {
        "max_abs": float(np.abs(board - torch_ref).max()),
        "rel_l2": float(np.linalg.norm(board - torch_ref) / np.linalg.norm(torch_ref)),
    }
    run_log = REPO / args.run_log
    summary = {
        "pass": result["pass"] and args.actual_runtime.lower() == args.requested_runtime.lower(),
        "n_crops": len(rows),
        "rows": rows,
        "requested_runtime": args.requested_runtime,
        "actual_runtime": args.actual_runtime,
        "output_file": output_file,
        "output_dims": dims,
        "output_layout": args.output_layout,
        "reference": rel(REPO / args.reference),
        "reference_sha256": sha256_file(REPO / args.reference),
        "run_log": rel(run_log),
        "run_log_sha256": sha256_file(run_log) if run_log.exists() else None,
        "tolerances": cfg["tolerances"]["board_cpu_vs_onnx"],
        "vs_torch": vs_torch,
        **{k: v for k, v in result.items() if k != "pass"},
    }
    for r in summary["per_crop"]:
        r["manifest_row"] = rows[r["row"]]
        r["crop_id"] = crop_ids[r["row"]]
    if args.actual_runtime.lower() != args.requested_runtime.lower():
        summary["runtime_mismatch"] = f"requested {args.requested_runtime}, actual {args.actual_runtime}: failed run"

    out_dir = REPO / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    sj, sm = out_dir / "summary.json", out_dir / "summary.md"
    sj.write_text(json.dumps(summary, indent=2) + "\n")
    sm.write_text(_markdown(summary))
    np.save(REPO / "artifacts/reference/board_cpu_d6_features.npy", board)

    c, o = result["checks"], result["overall"]
    print(f"[board] {'PASS' if summary['pass'] else 'FAIL'}: {len(rows)} crops, runtime requested {args.requested_runtime} / actual {args.actual_runtime}")
    print(f"[board] max_abs worst {c['max_abs']['worst']:.3e} (limit {c['max_abs']['limit']:.1e}); "
          f"rel_l2 worst crop {c['rel_l2_per_crop']['worst']:.3e} (limit {c['rel_l2_per_crop']['limit']:.1e})")
    print(f"[board] overall max_abs {o['max_abs']:.3e}, mean_abs {o['mean_abs']:.3e}, rel_l2 {o['rel_l2']:.3e}")
    for p in (sj, sm):
        print(f"[board] wrote {rel(p)}  sha256 {sha256_file(p)}")
    sys.exit(0 if summary["pass"] else 1)


if __name__ == "__main__":
    main()
