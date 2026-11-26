"""Read snpe-net-run outputs, convert to NCHW explicitly, and compare with ONNX reference features.

Output dims and layout are NOT assumed: pass --output-dims and --output-layout exactly as
snpe-dlc-info reports them. Result_<i> is matched to line i of input_list.txt, whose order must equal
the round-trip manifest written by write_snpe_inputs.py; reference features are then looked up by
crop_id, not by position. Compares against tolerances.board_cpu_vs_onnx (never changed here).

--purpose dev (default) uses Part B's regenerated development crops and is labelled "development only".
--purpose formal|smoke requires --canonical-inputs: a directory made by scripts/canonical_crops.py whose
raw inputs, input list and package manifest are re-verified against Part A's pinned sha256 values; crop
IDs then come from that verified package, and anything that does not verify is refused.

Two separate verdicts:
  numeric_match    - every check within tolerance
  backend_verified - the ACTUAL runtime parsed from the snpe-net-run log equals the requested one
Overall PASS only if both are true. backend_verified needs POSITIVE evidence of the runtime that
executed: per-layer runtimes from --diag-log (snpe-diagview output) when given, else explicit runtime lines
in the snpe-net-run log. Request-only text (flags such as --use_gpu, echoed commands, "requested runtime")
is never evidence, and any fallback / unavailable / not supported / failed-to-create / error line makes it
false whatever else appears. The snpe-diagview per-layer format ("0: <layer> : <t> us : <RUNTIME>") was
confirmed on the conversion host's x86_64 log; it and the stdout patterns must be re-confirmed on the first RB5 log.
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np

from check_onnx_parity import compare
import canonical_crops
from common import REPO, crop_id, load_config, rel, sha256_file
from write_snpe_inputs import LAYOUTS, parse_dims

RUNTIMES = ("CPU", "GPU", "GPU_FLOAT16", "DSP", "AIP", "HTP")
_RT = "|".join(sorted(RUNTIMES, key=len, reverse=True))
# Positive evidence that a runtime EXECUTED. NOT yet confirmed against QAIRT 2.33.2 output: confirm and
# tighten on the first real board log. Every rule below can only fail toward backend_verified = false.
RUNTIME_PATTERNS = [
    re.compile(rf"\bruntime\s*(?:selected|used|in use)?\s*[:=]\s*({_RT})\b", re.I),
    re.compile(rf"\busing\s+(?:the\s+)?({_RT})\s+runtime\b", re.I),
    re.compile(rf"\b({_RT})\s+runtime\s+(?:selected|is used|in use)\b", re.I),
]
# Per-layer runtime in snpe-diagview text output of SNPEDiag.log (preferred evidence when given).
# Real format (QAIRT 2.33.2 snpe-diagview, under "Per-Layer Execution Times:"), confirmed on the conversion
# host's x86_64 log results/raw/conversion/host_dev/diag.txt; re-confirm on the first RB5 log:
#     "    0: _node_Conv_102 : 1919 us : CPU"
# The runtime must be the whole last field of the line. The earlier generic form is kept as a fallback.
DIAG_LAYER_PATTERNS = [
    re.compile(rf"^\s*\d+:\s*\S+\s*:\s*\d+(?:\.\d+)?\s*us\s*:\s*({_RT})\s*$", re.I),
    re.compile(rf"\blayer\b.*?\bruntime\s*[:=]?\s*({_RT})\b", re.I),
]
# Any of these anywhere in the evidence -> not verified, whatever else appears (deliberately broad).
NEGATIVE_PATTERN = re.compile(
    r"fall(?:ing|s)?[\s_-]*back|unavailable|not available|not supported|unsupported|"
    r"(?:failed|unable|could not|cannot)\s+(?:to\s+)?(?:create|initiali[sz]e|init|load|open|set\s*up|build)|"
    r"\berror\b",
    re.I,
)


def _request_only(line: str) -> bool:
    """Text that only states what was requested (flags, echoed commands, 'requested ...') is never evidence."""
    l = line.strip()
    return (
        "request" in l.lower()
        or "snpe-net-run" in l
        or l.startswith(("$", "+ ", ">", "#"))
        or re.search(r"(?:^|\s)--?[A-Za-z_]", l) is not None
    )


def parse_runtime(log_text):
    """Return (runtime, evidence_lines) from stdout text; runtime is None unless all evidence agrees."""
    found, evidence = set(), []
    for line in (log_text or "").splitlines():
        if _request_only(line):
            continue
        for pat in RUNTIME_PATTERNS:
            m = pat.search(line)
            if m:
                found.add(m.group(1).upper())
                evidence.append(line.strip())
                break
    return (found.pop() if len(found) == 1 else None), evidence


def parse_diag(diag_text):
    """Return (runtime, layer_lines) from per-layer diag output; None unless every layer names the same runtime."""
    found, evidence = set(), []
    for line in (diag_text or "").splitlines():
        if _request_only(line):
            continue
        for pat in DIAG_LAYER_PATTERNS:
            m = pat.search(line)
            if m:
                found.add(m.group(1).upper())
                evidence.append(line.strip())
                break
    return (found.pop() if len(found) == 1 else None), evidence


def negative_lines(*texts):
    return [l.strip() for t in texts for l in (t or "").splitlines() if NEGATIVE_PATTERN.search(l)]


def backend_check(run_log: Path, requested: str, diag_log: Path = None):
    """backend_verified only with positive evidence of the runtime that executed and no negative line."""
    out = {"backend_verified": False, "actual_runtime": None, "source": None, "evidence": [], "negative": []}
    log_text = run_log.read_text(errors="replace") if run_log is not None and run_log.exists() else None
    diag_text = diag_log.read_text(errors="replace") if diag_log is not None and diag_log.exists() else None
    if diag_log is not None and diag_text is None:
        return {**out, "reason": f"diag log {diag_log} not found"}
    if log_text is None and diag_text is None:
        return {**out, "reason": "no snpe-net-run log"}
    out["negative"] = negative_lines(log_text, diag_text)
    if diag_text is not None:
        runtime, evidence = parse_diag(diag_text)
        out.update(source="diag", evidence=evidence)
        if runtime is None:
            reason = "diag layers name different runtimes" if evidence else "no per-layer runtime in diag log"
            return {**out, "reason": reason}
    else:
        runtime, evidence = parse_runtime(log_text)
        out.update(source="stdout", evidence=evidence)
        if runtime is None:
            reason = "conflicting runtime lines in log" if evidence else "no execution-runtime line in log"
            return {**out, "reason": reason}
    out["actual_runtime"] = runtime
    if out["negative"]:
        return {**out, "reason": f"log reports fallback/unavailable/error: {out['negative'][0]!r}"}
    if runtime != requested.upper():
        return {**out, "reason": f"requested {requested.upper()}, {out['source']} says {runtime}"}
    return {**out, "backend_verified": True, "reason": f"{out['source']} runtime matches request"}


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
    """Map crop_ids to rows of the reference feature array via its crop manifest (crop_id column or derived)."""
    with ref_manifest.open() as f:
        by_id = {
            (r.get("crop_id") or crop_id(r["recording"], r["frame"], r["hand_idx"], r["cam_idx"])): int(r["row"])
            for r in csv.DictReader(f)
        }
    missing = [i for i in ids if i not in by_id]
    if missing:
        raise SystemExit(f"crop_ids not in {ref_manifest}: {missing[:3]}")
    return [by_id[i] for i in ids]


def verdict_line(s: dict) -> str:
    if s["pass"]:
        return "PASS"
    return f"NOT PASS (numeric_match={s['numeric_match']}, backend_verified={s['backend_verified']})"


TARGETS = {
    "rb5": {"title": "D6 — RB5 vs ONNX Runtime encoder features", "who": "Board", "note": None},
    "conversion-host": {
        "title": "Development check on the conversion host — x86_64 snpe-net-run vs ONNX Runtime encoder features",
        "who": "Host",
        "note": "Development check on the conversion host, not the RB5 experiment: x86_64 host runtime, "
                "Part B development crops. Says nothing about RB5 CPU accuracy or timing.",
    },
}


def _markdown(s: dict) -> str:
    c, o, b = s["checks"], s["overall"], s["backend"]
    t = TARGETS[s["target"]]
    lines = [
        f"# {t['title']}",
        "",
        *([f"**{t['note']}**", ""] if t["note"] else []),
        f"**Purpose: {s['purpose_label']}**",
        "",
        f"**Result: {verdict_line(s)}** on {s['n_crops']} crops.",
        "",
        f"- numeric_match: **{s['numeric_match']}** (tolerances `tolerances.{s['tolerance_key']}`, declared before this run, unchanged)",
        f"- backend_verified: **{s['backend_verified']}** — requested `{s['requested_runtime']}`, "
        f"actual from log `{b['actual_runtime']}` ({b['reason']})",
        "",
        "| Check | Limit | Measured (worst) | Within |",
        "| --- | --- | --- | --- |",
        f"| max_abs over all crops | {c['max_abs']['limit']:.1e} | {c['max_abs']['worst']:.3e} | {c['max_abs']['pass']} |",
        f"| rel_l2 per crop | {c['rel_l2_per_crop']['limit']:.1e} | {c['rel_l2_per_crop']['worst']:.3e} "
        f"(idx {o['worst_crop_rel_l2']['row']}) | {c['rel_l2_per_crop']['pass']} |",
        "",
        f"Overall: max_abs {o['max_abs']:.3e}, mean_abs (MAE) {o['mean_abs']:.3e}, RMSE {o['rmse']:.3e}, rel_l2 {o['rel_l2']:.3e}.",
        f"Diagnostic, {t['who'].lower()} vs PyTorch: max_abs {s['vs_torch']['max_abs']:.3e}, rel_l2 {s['vs_torch']['rel_l2']:.3e}.",
        "",
        f"{t['who']} output: `{s['output_file']}` dims {s['output_dims']} layout {s['output_layout']} "
        "-> explicit transpose to NCHW [1,72,6,6]. Result_<i> matched to input_list line i, reference by crop_id.",
        f"Reference: `{s['reference']}` sha256 `{s['reference_sha256']}`.",
        f"Run log: `{s['run_log']}` sha256 `{s['run_log_sha256']}`.",
        f"Runtime evidence (source: {b['source']}): " + ("; ".join(f"`{e}`" for e in b["evidence"][:5]) if b["evidence"] else "none"),
        "Negative lines: " + ("; ".join(f"`{e}`" for e in b["negative"][:5]) if b["negative"] else "none"),
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
    ap.add_argument("--diag-log", default=None, help="snpe-diagview text output of SNPEDiag.log (preferred runtime evidence)")
    ap.add_argument("--out-dir", default="results/board_cpu_d6")
    ap.add_argument("--features-out", default="artifacts/reference/board_cpu_d6_features.npy")
    ap.add_argument("--purpose", choices=["dev", "formal", "smoke"], default="dev")
    ap.add_argument("--target", choices=sorted(TARGETS), default="rb5",
                    help="where snpe-net-run ran; conversion-host labels the run a development check, not the RB5 experiment")
    ap.add_argument("--tolerance-key", default="board_cpu_vs_onnx", help="key under tolerances: in the config")
    ap.add_argument("--canonical-inputs", default=None, help="for formal/smoke: dir made by scripts/canonical_crops.py")
    args = ap.parse_args()

    cfg = load_config(REPO / args.config)
    output_file = args.output_file or f"{cfg['encoder']['output']['name']}.raw"
    dims = parse_dims(args.output_dims)
    canonical = None
    if args.purpose == "dev":
        with (REPO / args.roundtrip_manifest).open() as f:
            rt_rows = list(csv.DictReader(f))
        ids = match_inputs(rt_rows, REPO / args.input_list)
        input_list = REPO / args.input_list
        purpose_label = "development only - Part B regenerated crops, not a formal comparison"
    else:
        if not args.canonical_inputs:
            raise SystemExit(f"--purpose {args.purpose} requires --canonical-inputs (Part A's verified frozen crops)")
        pkg = REPO / args.canonical_inputs
        try:
            canonical = canonical_crops.verify_extracted(pkg, args.purpose)
        except canonical_crops.CanonicalError as e:
            raise SystemExit(f"refusing --purpose {args.purpose}: {e}")
        input_list = pkg / "input_list.txt"
        ids = [r["crop_id"] for r in canonical["inputs"]]
        purpose_label = f"{args.purpose} - Part A frozen crops verified against {canonical['pinned']['manifest']}"
    ref_rows = reference_rows(REPO / args.reference_manifest, ids)

    board = read_outputs(REPO / args.outputs_dir, len(ids), output_file, dims, args.output_layout)
    ref = np.load(REPO / args.reference)[ref_rows]
    want = [len(ids), *cfg["encoder"]["output"]["shape"][1:]]
    if list(board.shape) != want or list(ref.shape) != want:
        raise SystemExit(f"shape mismatch: board {list(board.shape)}, reference {list(ref.shape)}, want {want}")

    if args.tolerance_key not in cfg["tolerances"]:
        raise SystemExit(f"tolerances.{args.tolerance_key} is not declared in {args.config}; declare it before comparing")
    tol = cfg["tolerances"][args.tolerance_key]
    result = compare(ref, board, tol)
    torch_ref = np.load(REPO / args.torch_reference)[ref_rows].astype(np.float64)
    vs_torch = {
        "max_abs": float(np.abs(board - torch_ref).max()),
        "rel_l2": float(np.linalg.norm(board - torch_ref) / np.linalg.norm(torch_ref)),
    }
    run_log = REPO / args.run_log
    diag_log = REPO / args.diag_log if args.diag_log else None
    backend = backend_check(run_log, args.requested_runtime, diag_log)
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
        "purpose": args.purpose,
        "purpose_label": purpose_label,
        "canonical": None if canonical is None else canonical["pinned"],
        "input_list": rel(input_list),
        "input_list_sha256": sha256_file(input_list),
        "reference": rel(REPO / args.reference),
        "reference_sha256": sha256_file(REPO / args.reference),
        "run_log": rel(run_log),
        "run_log_sha256": sha256_file(run_log) if run_log.exists() else None,
        "diag_log": rel(diag_log) if diag_log else None,
        "diag_log_sha256": sha256_file(diag_log) if diag_log and diag_log.exists() else None,
        "target": args.target,
        "target_note": TARGETS[args.target]["note"],
        "tolerance_key": args.tolerance_key,
        "tolerances": tol,
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
    if TARGETS[args.target]["note"]:
        print(f"[board] {TARGETS[args.target]['note']}")
    print(f"[board] purpose: {purpose_label}")
    print(f"[board] {verdict_line(summary)}: {len(ids)} crops")
    print(f"[board] numeric_match {numeric_match}: max_abs worst {c['max_abs']['worst']:.3e} (limit {c['max_abs']['limit']:.1e}); "
          f"rel_l2 worst crop {c['rel_l2_per_crop']['worst']:.3e} (limit {c['rel_l2_per_crop']['limit']:.1e})")
    print(f"[board] backend_verified {backend['backend_verified']}: requested {summary['requested_runtime']}, "
          f"log {backend['actual_runtime']} ({backend['reason']})")
    print(f"[board] overall max_abs {o['max_abs']:.3e}, mean_abs {o['mean_abs']:.3e}, rmse {o['rmse']:.3e}, rel_l2 {o['rel_l2']:.3e}")
    for p in (sj, sm, feats):
        print(f"[board] wrote {rel(p)}  sha256 {sha256_file(p)}")
    sys.exit(0 if summary["pass"] else 1)


if __name__ == "__main__":
    main()
