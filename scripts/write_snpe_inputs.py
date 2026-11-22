"""Write float32 .raw input files + input_list.txt for snpe-net-run from selected D5 crops.

Preprocessing follows the encoder contract (uint8 / 255 -> float32 [1,1,96,96] NCHW). The raw
layout is NOT assumed: pass --input-layout and --input-dims exactly as snpe-dlc-info reports them.
For C=1, NHWC [1,96,96,1] and NCHW [1,1,96,96] have identical bytes, but the transpose is still
done explicitly so the code stays correct if the reported layout ever differs.

DEVELOPMENT ONLY. These inputs are regenerated from Part B's own crops. Formal and smoke comparisons
must use Part A's frozen raw files as-is: verify and extract them with scripts/canonical_crops.py.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from check_onnx_parity import preprocess
from common import REPO, crop_id, load_config, rel, sha256_file, update_hashes

LAYOUTS = ("NCHW", "NHWC")


def parse_dims(s: str):
    return [int(d) for d in s.replace("x", ",").split(",") if d.strip()]


def to_layout(x_nchw: np.ndarray, layout: str) -> np.ndarray:
    """[N,C,H,W] float32 -> the requested layout, C-contiguous."""
    if layout == "NCHW":
        return np.ascontiguousarray(x_nchw)
    if layout == "NHWC":
        return np.ascontiguousarray(x_nchw.transpose(0, 2, 3, 1))
    raise ValueError(f"unknown layout {layout}")


def select_rows(n_total: int, n: int):
    """Deterministic, evenly spaced rows across the crop set (covers both hands and several frames)."""
    return sorted({int(round(v)) for v in np.linspace(0, n_total - 1, n)})


def write_inputs(crops_uint8, rows, layout, dims, out_dir: Path, board_prefix: str):
    """Write one raw per row and input_list.txt. Returns list of (raw_path, row)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    lines = []
    for i, row in enumerate(rows):
        x = preprocess(crops_uint8[row : row + 1])  # [1,1,96,96] NCHW float32
        arr = to_layout(x, layout)
        if list(arr.shape) != list(dims):
            raise SystemExit(f"row {row}: {layout} shape {list(arr.shape)} != --input-dims {dims}")
        name = f"crop_{i:03d}_row{row:04d}.raw"
        path = out_dir / name
        arr.astype("<f4", copy=False).tofile(path)
        assert path.stat().st_size == arr.size * 4
        written.append((path, row))
        lines.append(f"{board_prefix.rstrip('/')}/{name}" if board_prefix else name)
    (out_dir / "input_list.txt").write_text("\n".join(lines) + "\n")
    return written


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/experiment.yaml")
    ap.add_argument("--crops", default="artifacts/crops/crops_uint8.npy")
    ap.add_argument("--crop-manifest", default="manifests/crops_d5.csv")
    ap.add_argument("--input-layout", required=True, choices=LAYOUTS, help="as reported by snpe-dlc-info")
    ap.add_argument("--input-dims", required=True, help="as reported by snpe-dlc-info, e.g. 1,96,96,1")
    ap.add_argument("--n", type=int, default=8, help="number of crops (evenly spaced rows)")
    ap.add_argument("--rows", default=None, help="explicit comma-separated manifest rows (overrides --n)")
    ap.add_argument("--out-dir", default="artifacts/snpe_inputs/d6")
    ap.add_argument("--board-prefix", default="inputs", help="path prefix used in input_list.txt on the board")
    ap.add_argument("--roundtrip-manifest", default="manifests/crops_d6_roundtrip.csv")
    ap.add_argument("--hashes", default="manifests/hashes.txt")
    ap.add_argument("--purpose", choices=["dev", "formal", "smoke"], default="dev")
    args = ap.parse_args()
    if args.purpose != "dev":
        raise SystemExit(
            f"refusing --purpose {args.purpose}: formal/smoke inputs are Part A's frozen raw files, never "
            "regenerated. Verify and extract them with scripts/canonical_crops.py --set "
            f"{args.purpose} --archive <zip> --out-dir <new dir>"
        )

    cfg = load_config(REPO / args.config)
    crops = np.load(REPO / args.crops)
    with (REPO / args.crop_manifest).open() as f:
        manifest = {int(r["row"]): r for r in csv.DictReader(f)}
    if len(manifest) != crops.shape[0]:
        raise SystemExit("crop manifest and crops array disagree on N")

    rows = [int(r) for r in args.rows.split(",")] if args.rows else select_rows(crops.shape[0], args.n)
    dims = parse_dims(args.input_dims)
    out_dir = REPO / args.out_dir
    written = write_inputs(crops, rows, args.input_layout, dims, out_dir, args.board_prefix)

    rt = REPO / args.roundtrip_manifest
    rt.parent.mkdir(parents=True, exist_ok=True)
    with rt.open("w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["idx", "raw_file", "row", "recording", "frame", "hand_idx", "cam_idx", "crop_id", "raw_sha256", "purpose"])
        for i, (path, row) in enumerate(written):
            m = manifest[row]
            cid = crop_id(m["recording"], m["frame"], m["hand_idx"], m["cam_idx"])
            w.writerow([i, path.name, row, m["recording"], m["frame"], m["hand_idx"], m["cam_idx"], cid, sha256_file(path), "development"])

    input_list = out_dir / "input_list.txt"
    meta = {
        "purpose": "development only - not a formal comparison (formal/smoke use Part A's frozen crops)",
        "input_name": cfg["encoder"]["input"]["name"],
        "input_layout": args.input_layout,
        "input_dims": dims,
        "dtype": "float32 little-endian",
        "preprocessing": "uint8 / 255 -> float32, NCHW [1,1,96,96], then explicit transpose to input_layout",
        "rows": rows,
        "board_prefix": args.board_prefix,
        "crops_sha256": sha256_file(REPO / args.crops),
    }
    (out_dir / "inputs_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    update_hashes(REPO / args.hashes, {rel(rt): sha256_file(rt)})

    print(f"[inputs] {len(written)} crops, rows {rows}, layout {args.input_layout} dims {dims}")
    for path, row in written:
        print(f"[inputs] wrote {rel(path)}  (row {row})  sha256 {sha256_file(path)}")
    print(f"[inputs] wrote {rel(input_list)}  sha256 {sha256_file(input_list)}")
    print(f"[inputs] wrote {rel(rt)}  sha256 {sha256_file(rt)}")


if __name__ == "__main__":
    main()
