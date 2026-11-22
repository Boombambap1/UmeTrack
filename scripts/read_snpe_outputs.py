"""Read snpe-net-run outputs, convert to NCHW explicitly, and compare with ONNX reference features.

Output dims and layout are NOT assumed: pass --output-dims and --output-layout exactly as
snpe-dlc-info reports them. Result_<i> is matched to line i of input_list.txt, whose order must equal
the round-trip manifest written by write_snpe_inputs.py; reference features are then looked up by
crop_id, not by position. Compares against tolerances.board_cpu_vs_onnx (never changed here).

Two separate verdicts:
  numeric_match    - every check within tolerance
  backend_verified - the ACTUAL runtime parsed from the snpe-net-run log equals the requested one
Overall PASS only if both are true. A missing log, no runtime line, or conflicting runtime lines
means backend_verified = false (a silent fallback must never be reported as the requested backend).
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np

from check_onnx_parity import compare
from common import REPO, crop_id, load_config, rel, sha256_file
from write_snpe_inputs import LAYOUTS, parse_dims

RUNTIMES = ("CPU", "GPU", "GPU_FLOAT16", "DSP", "AIP", "HTP")
# Conservative patterns for lines that explicitly name the runtime in use. NOT yet confirmed against
# QAIRT 2.33.2 snpe-net-run output: confirm (and tighten) on the first real board log. They can only
# fail toward backend_verified = false; they never infer a runtime from the command line or file name.
_RT = "|".join(sorted(RUNTIMES, key=len, reverse=True))
RUNTIME_PATTERNS = [
    re.compile(rf"\bruntime\s*(?:selected|used|in use)?\s*[:=]\s*({_RT})\b", re.I),
    re.compile(rf"\busing\s+(?:the\s+)?({_RT})\s+runtime\b", re.I),
    re.compile(rf"\b({_RT})\s+runtime\s+(?:selected|is used|in use)\b", re.I),
]


def parse_runtime(log_text):
    """Return (runtime, evidence_lines). runtime is None unless every matching line names the same one."""
    found, evidence = set(), []
    for line in (log_text or "").splitlines():
        for pat in RUNTIME_PATTERNS:
            m = pat.search(line)
            if m:
                found.add(m.group(1).upper())
                evidence.append(line.strip())
                break
    return (found.pop() if len(found) == 1 else None), evidence


def backend_check(run_log: Path, requested: str):
    if run_log is None or not run_log.exists():
        return {"backend_verified": False, "actual_runtime": None, "reason": "no snpe-net-run log", "evidence": []}
    runtime, evidence = parse_runtime(run_log.read_text(errors="replace"))
    if runtime is None:
        reason = "conflicting runtime lines in log" if evidence else "no runtime line found in log"
        return {"backend_verified": False, "actual_runtime": None, "reason": reason, "evidence": evidence}
    ok = runtime == requested.upper()
    return {
        "backend_verified": ok,
        "actual_runtime": runtime,
        "reason": "log runtime matches request" if ok else f"requested {requested.upper()}, log says {runtime}",
        "evidence": evidence,
    }


def to_nchw(arr: np.ndarray, layout: str) -> np.ndarray:
    if layout == "NCHW":
        return np.ascontiguousarray(arr)
    if layout == "NHWC":
        return np.ascontiguousarray(arr.transpose(0, 3, 1, 2))
    raise ValueError(f"unknown layout {layout}")


def read_outputs(outputs_dir: Path, n: int, output_file: str, dims, layout: str) -> np.ndarray:
    """Read Result_<i>/<output_file> for i in [0, n); return float32 [n, C, H, W].

    Rejects: a missing or extra Result_* directory, any other entry in the outputs dir or a Result dir,
    and any output whose byte size is not exactly prod(dims) * 4.
    """
    expected_bytes = int(np.prod(dims)) * 4
    want_dirs = {f"Result_{i}" for i in range(n)}
    have = {p.name for p in outputs_dir.iterdir()} if outputs_dir.is_dir() else set()
    if have != want_dirs:
        raise SystemExit(
            f"{outputs_dir}: expected exactly {sorted(want_dirs)}; missing {sorted(want_dirs - have)}, "
            f"unexpected {sorted(have - want_dirs)}"
        )
    feats = []
    for i in range(n):
        d = outputs_dir / f"Result_{i}"
        entries = sorted(p.name for p in d.iterdir())
        if entries != [output_file]:
            raise SystemExit(f"{d}: expected only {output_file}, found {entries}")
        path = d / output_file
        size = path.stat().st_size
        if size != expected_bytes:
            raise SystemExit(f"{path}: {size} bytes, expected exactly {expected_bytes} for float32 dims {dims}")
        raw = np.fromfile(path, dtype="<f4")
        feats.append(to_nchw(raw.reshape(dims), layout))
    return np.concatenate(feats).astype(np.float32)


def match_inputs(rt_rows, input_list: Path):
    """Check input_list.txt order equals the round-trip manifest; return crop_ids in Result_<i> order."""
    lines = [l for l in input_list.read_text().splitlines() if l.strip()]
    names = [Path(l).name for l in lines]
    want = [r["raw_file"] for r in sorted(rt_rows, key=lambda r: int(r["idx"]))]
    if [int(r["idx"]) for r in rt_rows] != list(range(len(rt_rows))):
        raise SystemExit("round-trip manifest idx column is not 0..n-1 in order")
    if names != want:
        raise SystemExit(f"{input_list}: order/content differs from the round-trip manifest")
    ids = [r["crop_id"] for r in rt_rows]
    if len(set(ids)) != len(ids) or not all(ids):
        raise SystemExit("round-trip manifest crop_ids are missing or not unique")
    return ids


def reference_rows(ref_manifest: Path, ids):
    """Map crop_ids to rows of the reference feature array via its crop manifest."""
    with ref_manifest.open() as f:
        by_id = {crop_id(r["recording"], r["frame"], r["hand_idx"], r["cam_idx"]): int(r["row"]) for r in csv.DictReader(f)}
    missing = [i for i in ids if i not in by_id]
    if missing:
        raise SystemExit(f"crop_ids not in {ref_manifest}: {missing[:3]}")
    return [by_id[i] for i in ids]


def verdict_line(s: dict) -> str:
    if s["pass"]:
        return "PASS"
    return f"NOT PASS (numeric_match={s['numeric_match']}, backend_verified={s['backend_verified']})"


def _markdown(s: dict) -> str:
    c, o, b = s["checks"], s["overall"], s["backend"]
    lines = [
        "# D6 — RB5 vs ONNX Runtime encoder features",
        "",
        f"**Result: {verdict_line(s)}** on {s['n_crops']} crops.",
        "",
        f"- numeric_match: **{s['numeric_match']}** (tolerances `tolerances.board_cpu_vs_onnx`, declared before this run, unchanged)",
        f"- backend_verified: **{s['backend_verified']}** — requested `{s['requested_runtime']}`, "
        f"actual from log `{b['actual_runtime']}` ({b['reason']})",
        "",
        "| Check | Limit | Measured (worst) | Within |",
        "| --- | --- | --- | --- |",
        f"| max_abs over all crops | {c['max_abs']['limit']:.1e} | {c['max_abs']['worst']:.3e} | {c['max_abs']['pass']} |",
        f"| rel_l2 per crop | {c['rel_l2_per_crop']['limit']:.1e} | {c['rel_l2_per_crop']['worst']:.3e} "
        f"(idx {o['worst_crop_rel_l2']['row']}) | {c['rel_l2_per_crop']['pass']} |",
        "",
        f"Overall: max_abs {o['max_abs']:.3e}, mean_abs {o['mean_abs']:.3e}, rel_l2 {o['rel_l2']:.3e}.",
        f"Diagnostic, board vs PyTorch: max_abs {s['vs_torch']['max_abs']:.3e}, rel_l2 {s['vs_torch']['rel_l2']:.3e}.",
        "",
        f"Board output: `{s['output_file']}` dims {s['output_dims']} layout {s['output_layout']} "
        "-> explicit transpose to NCHW [1,72,6,6]. Result_<i> matched to input_list line i, reference by crop_id.",
        f"Reference: `{s['reference']}` sha256 `{s['reference_sha256']}`.",
        f"Run log: `{s['run_log']}` sha256 `{s['run_log_sha256']}`.",
        "Runtime evidence lines: " + ("; ".join(f"`{e}`" for e in b["evidence"]) if b["evidence"] else "none"),
        "",
        "## Per crop",
        "",
        "| idx | crop_id | reference row | max_abs | mean_abs | rel_l2 | within |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in s["per_crop"]:
        lines.append(
            f"| {r['row']} | `{r['crop_id']}` | {r['reference_row']} | {r['max_abs']:.3e} | {r['mean_abs']:.3e} "
            f"| {r['rel_l2']:.3e} | {r['pass']} |"
        )
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/experiment.yaml")
    ap.add_argument("--outputs-dir", default="results/raw/board_cpu_d6/output")
    ap.add_argument("--output-file", default=None, help="raw file name inside Result_<i>/ (default <output name>.raw)")
    ap.add_argument("--output-dims", required=True, help="as reported by snpe-dlc-info, e.g. 1,6,6,72")
    ap.add_argument("--output-layout", required=True, choices=LAYOUTS)
    ap.add_argument("--input-list", default="artifacts/snpe_inputs/d6/input_list.txt", help="the list given to snpe-net-run")
    ap.add_argument("--roundtrip-manifest", default="manifests/crops_d6_roundtrip.csv")
    ap.add_argument("--reference-manifest", default="manifests/crops_d5.csv")
    ap.add_argument("--reference", default="artifacts/reference/onnx_features.npy")
    ap.add_argument("--torch-reference", default="artifacts/reference/torch_features.npy")
    ap.add_argument("--requested-runtime", default="cpu")
    ap.add_argument("--run-log", required=True, help="snpe-net-run log; the actual runtime is parsed from it")
    ap.add_argument("--out-dir", default="results/board_cpu_d6")
    ap.add_argument("--features-out", default="artifacts/reference/board_cpu_d6_features.npy")
    args = ap.parse_args()

    cfg = load_config(REPO / args.config)
    output_file = args.output_file or f"{cfg['encoder']['output']['name']}.raw"
    dims = parse_dims(args.output_dims)
    with (REPO / args.roundtrip_manifest).open() as f:
        rt_rows = list(csv.DictReader(f))
    ids = match_inputs(rt_rows, REPO / args.input_list)
    ref_rows = reference_rows(REPO / args.reference_manifest, ids)

    board = read_outputs(REPO / args.outputs_dir, len(ids), output_file, dims, args.output_layout)
    ref = np.load(REPO / args.reference)[ref_rows]
    want = [len(ids), *cfg["encoder"]["output"]["shape"][1:]]
    if list(board.shape) != want or list(ref.shape) != want:
        raise SystemExit(f"shape mismatch: board {list(board.shape)}, reference {list(ref.shape)}, want {want}")

    result = compare(ref, board, cfg["tolerances"]["board_cpu_vs_onnx"])
    torch_ref = np.load(REPO / args.torch_reference)[ref_rows].astype(np.float64)
    vs_torch = {
        "max_abs": float(np.abs(board - torch_ref).max()),
        "rel_l2": float(np.linalg.norm(board - torch_ref) / np.linalg.norm(torch_ref)),
    }
    run_log = REPO / args.run_log
    backend = backend_check(run_log, args.requested_runtime)
    numeric_match = bool(result["pass"])
    summary = {
        "pass": numeric_match and backend["backend_verified"],
        "numeric_match": numeric_match,
        "backend_verified": backend["backend_verified"],
        "backend": backend,
        "n_crops": len(ids),
        "requested_runtime": args.requested_runtime.upper(),
        "output_file": output_file,
        "output_dims": dims,
        "output_layout": args.output_layout,
        "input_list": rel(REPO / args.input_list),
        "input_list_sha256": sha256_file(REPO / args.input_list),
        "reference": rel(REPO / args.reference),
        "reference_sha256": sha256_file(REPO / args.reference),
        "run_log": rel(run_log),
        "run_log_sha256": sha256_file(run_log) if run_log.exists() else None,
        "tolerances": cfg["tolerances"]["board_cpu_vs_onnx"],
        "vs_torch": vs_torch,
        **{k: v for k, v in result.items() if k != "pass"},
    }
    for r in summary["per_crop"]:
        r["crop_id"] = ids[r["row"]]
        r["reference_row"] = ref_rows[r["row"]]

    out_dir = REPO / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    sj, sm = out_dir / "summary.json", out_dir / "summary.md"
    sj.write_text(json.dumps(summary, indent=2) + "\n")
    sm.write_text(_markdown(summary))
    feats = REPO / args.features_out
    feats.parent.mkdir(parents=True, exist_ok=True)
    np.save(feats, board)

    c, o = result["checks"], result["overall"]
    print(f"[board] {verdict_line(summary)}: {len(ids)} crops")
    print(f"[board] numeric_match {numeric_match}: max_abs worst {c['max_abs']['worst']:.3e} (limit {c['max_abs']['limit']:.1e}); "
          f"rel_l2 worst crop {c['rel_l2_per_crop']['worst']:.3e} (limit {c['rel_l2_per_crop']['limit']:.1e})")
    print(f"[board] backend_verified {backend['backend_verified']}: requested {summary['requested_runtime']}, "
          f"log {backend['actual_runtime']} ({backend['reason']})")
    print(f"[board] overall max_abs {o['max_abs']:.3e}, mean_abs {o['mean_abs']:.3e}, rel_l2 {o['rel_l2']:.3e}")
    for p in (sj, sm, feats):
        print(f"[board] wrote {rel(p)}  sha256 {sha256_file(p)}")
    sys.exit(0 if summary["pass"] else 1)


if __name__ == "__main__":
    main()
