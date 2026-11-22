"""Make the INT8 calibration crop set (D8 prep) from TRAINING recordings of users in no testing split.

Recordings: real/separate_hand/training user_15 and user_16, recording_00, at the pinned UmeTrack_data
commit. At that commit the testing splits (real and synthetic, hand_hand and separate_hand) contain
users 05, 07, 09, 12, 19, 23, 29, 34, 39, and no training user appears in any testing split.

--download fetches each mp4 from its Git LFS media URL and each label json from the pinned git tree,
verifying the LFS sha256/size (mp4) or git blob hash (json) before keeping it. Crops come from
make_crops.extract_crops (same placement and upstream cross-check as D5), every --frame-stride frames,
whole frames only, up to --crops-per-recording each. The script then asserts that no crop overlaps
the D5 crops or Part A's paired subset (manifests/reference-clips.json) and that no calibration user
is a testing user (also checked against Part A's manifests/separate-hand-test.json).
"""

import argparse
import csv
import hashlib
import json
import urllib.request
from pathlib import Path

import numpy as np

import pytorch3d_shim
import upstream_compat
from common import REPO, crop_id, load_config, load_model, rel, sha256_file, update_hashes
from make_crops import extract_crops

PURPOSE = "calibration"
PURPOSE_NOTE = (
    "INT8 calibration only - never evaluation or formal comparison. " + 'Part A audit of a21a692: "Crop IDs match, but the decoding stacks produce different pixels from the same video. The first decoded frame differed by up to two intensity levels. The geometry shim also introduced small additional crop differences. Formal hardware comparisons must use the frozen canonical crop files, rather than independently regenerating crops on each machine."'
)

DATA_REPO = "facebookresearch/UmeTrack_data"
DATA_COMMIT = "355df534a1c761ac416427925485b311f22ce50e"
TEST_USERS = {"user_05", "user_07", "user_09", "user_12", "user_19", "user_23", "user_29", "user_34", "user_39"}

# Expected identities from the pinned tree: LFS pointer oid/size for mp4, git blob sha1/size for json.
RECORDINGS = {
    "raw_data/real/separate_hand/training/user_15/recording_00": {
        "mp4": {"sha256": "5a75165e3dd98d515b0960be4a30bd24f21150348bbace02b46c38f03923c2d4", "size": 74122290},
        "json": {"git_blob": "e5cae0007bfdf6588d563d80ead1fb09a989dfd9", "size": 3196955},
    },
    "raw_data/real/separate_hand/training/user_16/recording_00": {
        "mp4": {"sha256": "20423449fc49e2d7e5f2299d042f49f7cdfc29a4f62d0c6027c19fa84bd1c5cd", "size": 81460979},
        "json": {"git_blob": "8892576062fd478f819e3f16cc8883b0f56d660d", "size": 3192617},
    },
}


def git_blob_sha1(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def verify(path: Path, kind: str, want: dict) -> bool:
    if not path.exists() or path.stat().st_size != want["size"]:
        return False
    if kind == "mp4":
        return sha256_file(path) == want["sha256"]
    return git_blob_sha1(path) == want["git_blob"]


def fetch(url: str, dest: Path, kind: str, want: dict) -> None:
    """Download to <dest>.part, verify, then rename. Never overwrites a file that fails verification."""
    if dest.exists():
        if verify(dest, kind, want):
            print(f"[calib] already present and verified: {rel(dest)}")
            return
        raise SystemExit(f"{rel(dest)} exists but does not match the pinned identity; inspect it, not overwriting")
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    if part.exists():
        raise SystemExit(f"leftover partial download {rel(part)}; inspect/remove it first")
    print(f"[calib] GET {url}")
    with urllib.request.urlopen(url) as r, part.open("wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    if not verify(part, kind, want):
        raise SystemExit(f"{rel(part)} failed verification against the pinned identity (kept for inspection)")
    part.rename(dest)
    print(f"[calib] downloaded and verified {rel(dest)}")


def download(data_dir: Path, recording: str, ident: dict) -> None:
    mp4 = data_dir / f"{recording}.mp4"
    fetch(f"https://media.githubusercontent.com/media/{DATA_REPO}/{DATA_COMMIT}/{recording}.mp4", mp4, "mp4", ident["mp4"])
    pointer = data_dir / "lfs_pointers" / f"{recording}.mp4"  # same layout as the D5 pointer
    if not pointer.exists():
        pointer.parent.mkdir(parents=True, exist_ok=True)
        pointer.write_text(
            f"version https://git-lfs.github.com/spec/v1\noid sha256:{ident['mp4']['sha256']}\nsize {ident['mp4']['size']}\n"
        )
    fetch(f"https://raw.githubusercontent.com/{DATA_REPO}/{DATA_COMMIT}/{recording}.json",
          data_dir / f"{recording}.json", "json", ident["json"])


def excluded_frames():
    """(clip_id, frame) pairs that calibration must not touch: D5 crops and Part A's paired subset."""
    out = set()
    with (REPO / "manifests/crops_d5.csv").open() as f:
        for r in csv.DictReader(f):
            out.add(crop_id(r["recording"], r["frame"], 0, 0).rsplit(":", 2)[0])
    clips = json.loads((REPO / "manifests/reference-clips.json").read_text())["clips"]
    for c in clips:
        for fr in range(c["start_frame"], c["start_frame"] + c["frames"]):
            out.add(f"{c['id']}:{fr:06d}")
    return out


def part_a_test_users():
    m = json.loads((REPO / "manifests/separate-hand-test.json").read_text())
    return {Path(r["source_path"]).parts[-2] for r in m["recordings"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/experiment.yaml")
    ap.add_argument("--download", action="store_true", help="fetch missing recordings (verified) before cropping")
    ap.add_argument("--frame-stride", type=int, default=10)
    ap.add_argument("--crops-per-recording", type=int, default=100)
    ap.add_argument("--out", default="artifacts/calibration/crops_uint8.npy")
    ap.add_argument("--manifest", default="manifests/calibration.csv")
    ap.add_argument("--meta", default="artifacts/calibration/calibration_meta.json")
    ap.add_argument("--hashes", default="manifests/hashes.txt")
    args = ap.parse_args()

    cfg = load_config(REPO / args.config)
    data_root = REPO / cfg["data"]["recordings_dir"]

    users = {Path(r).parts[-2] for r in RECORDINGS}
    if users & TEST_USERS or users & part_a_test_users():
        raise SystemExit(f"calibration users {sorted(users)} overlap testing users")
    if any("/training/" not in r for r in RECORDINGS):
        raise SystemExit("calibration recordings must come from a training split")

    for recording, ident in RECORDINGS.items():
        if args.download:
            download(data_root, recording, ident)
        for kind in ("mp4", "json"):
            p = data_root / f"{recording}.{kind}"
            if not verify(p, kind, ident[kind]):
                raise SystemExit(f"{rel(p)} missing or not the pinned file; run with --download")

    using_shim = pytorch3d_shim.install()
    model, _, _ = load_model(cfg)
    patches = upstream_compat.apply()

    all_crops, all_rows, per_rec = [], [], {}
    for recording in RECORDINGS:
        crops, rows, counts, scanned = extract_crops(
            model, data_root / f"{recording}.mp4", recording,
            frame_stride=args.frame_stride, max_crops=args.crops_per_recording,
        )
        base = len(all_rows)
        all_rows.extend((base + r[0], *r[1:]) for r in rows)
        all_crops.append(crops)
        per_rec[recording] = {"crops": int(crops.shape[0]), "frames": len(counts), "frames_considered": scanned,
                              "frame_range": [rows[0][2], rows[-1][2]] if rows else None}
    crops = np.concatenate(all_crops)

    ids = [crop_id(rec, fr, h, c) for _, rec, fr, h, c in all_rows]
    if len(set(ids)) != len(ids):
        raise SystemExit("duplicate calibration crop ids")
    clash = {i.rsplit(":", 2)[0] for i in ids} & excluded_frames()
    if clash:
        raise SystemExit(f"calibration overlaps evaluation/D5 frames: {sorted(clash)[:5]}")

    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, crops)
    manifest = REPO / args.manifest
    with manifest.open("w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["row", "recording", "frame", "hand_idx", "cam_idx", "crop_id", "purpose"])
        for row, cid in zip(all_rows, ids):
            w.writerow([*row, cid, PURPOSE])

    hashes = {rel(out): sha256_file(out), rel(manifest): sha256_file(manifest)}
    for recording in RECORDINGS:
        for kind in ("mp4", "json"):
            p = data_root / f"{recording}.{kind}"
            hashes[rel(p)] = sha256_file(p)
    update_hashes(REPO / args.hashes, hashes)

    meta = {
        "purpose": PURPOSE,
        "purpose_note": PURPOSE_NOTE,
        "data_commit": DATA_COMMIT,
        "recordings": per_rec,
        "users": sorted(users),
        "testing_users_at_data_commit": sorted(TEST_USERS),
        "excluded": ["manifests/crops_d5.csv frames", "manifests/reference-clips.json frames"],
        "overlap_check": "none: no shared users, no shared (clip, frame)",
        "frame_stride": args.frame_stride,
        "crops_per_recording_max": args.crops_per_recording,
        "n_crops": int(crops.shape[0]),
        "crops_per_hand": {str(h): sum(1 for r in all_rows if r[3] == h) for h in sorted({r[3] for r in all_rows})},
        "crops_per_camera": {str(c): sum(1 for r in all_rows if r[4] == c) for c in sorted({r[4] for r in all_rows})},
        "pixel_range": [int(crops.min()), int(crops.max())],
        "pixel_mean": float(crops.mean()),
        "pytorch3d": "scripts/pytorch3d_shim.py (so3_exp_map stand-in)" if using_shim else "real package",
        "upstream_runtime_patches": patches,
        "sha256": hashes,
    }
    meta_path = REPO / args.meta
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")

    print(f"[calib] users {sorted(users)}; testing users {sorted(TEST_USERS)}: no overlap")
    for r, d in per_rec.items():
        print(f"[calib] {r}: {d['crops']} crops from {d['frames']} frames (range {d['frame_range']}, stride {args.frame_stride})")
    print(f"[calib] N={crops.shape[0]}; per hand {meta['crops_per_hand']}, per camera {meta['crops_per_camera']}")
    print("[calib] overlap with D5 crops / Part A paired subset: none")
    for p, d in hashes.items():
        print(f"[calib] {p}  sha256 {d}")
    print(f"[calib] wrote {rel(meta_path)}  sha256 {sha256_file(meta_path)}")


if __name__ == "__main__":
    main()
