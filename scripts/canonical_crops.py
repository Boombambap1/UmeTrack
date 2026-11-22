"""Verify and unpack Part A's frozen canonical crop packages for formal/smoke board comparisons.

Formal board comparisons (board vs reference, pose replay) must use Part A's frozen crops, never crops
regenerated on another machine. From Part A's audit of a21a692: "Crop IDs match, but the decoding stacks
produce different pixels from the same video. The first decoded frame differed by up to two intensity
levels. The geometry shim also introduced small additional crop differences. Formal hardware comparisons
must use the frozen canonical crop files, rather than independently regenerating crops on each machine."

Pinned identities come only from Part A's committed manifests:
  formal (335 crops): manifests/qualcomm-input-package.json  archive, package.json and input_list sha256
  smoke  (7 crops):   manifests/smoke-exchange.json          archive, internal manifest and input_list sha256
Every raw input listed is then checked against the per-crop sha256 inside the hash-pinned package
manifest. Expected hashes, crop IDs and order come ONLY from that pinned chain, re-verified each time the
crops are used; canonical_inputs.json is an informational copy that is refused if it disagrees. Raw files
are used as-is (never re-encoded). Anything that does not verify is refused.

CLI: verify an archive and extract it into a NEW directory, writing canonical_inputs.json (the ordered
input_file -> crop_id list that read_snpe_outputs.py uses for --purpose formal/smoke).
"""

import argparse
import hashlib
import json
import zipfile
from pathlib import Path, PurePosixPath

from common import REPO, rel, sha256_file

SETS = {
    "formal": {"manifest": "manifests/qualcomm-input-package.json", "crops": 335},
    "smoke": {"manifest": "manifests/smoke-exchange.json", "crops": 7},
}
RECORD = "canonical_inputs.json"


class CanonicalError(Exception):
    pass


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def pinned(set_name: str) -> dict:
    """Hashes Part A committed for this set."""
    spec = SETS[set_name]
    m = json.loads((REPO / spec["manifest"]).read_text())
    if int(m["crops"]) != spec["crops"]:
        raise CanonicalError(f"{spec['manifest']} lists {m['crops']} crops, expected {spec['crops']}")
    p = {
        "set": set_name,
        "manifest": spec["manifest"],
        "manifest_file_sha256": sha256_file(REPO / spec["manifest"]),
        "crops": spec["crops"],
        "archive_sha256": m["archive_sha256"],
        "archive_bytes": m["archive_bytes"],
        "input_list_sha256": m["input_list_sha256"],
    }
    if set_name == "formal":
        p["package_manifest_sha256"] = m["package_sha256"]  # sha256 of package.json (raw_exchange.py)
    else:
        p["package_manifest_sha256"] = m["manifest_sha256"]
        p["crop_ids"] = list(m["crop_ids"])
    return p


def _records(manifest: dict):
    """Ordered (input_file, input_sha256, crop_id) from a package manifest in Part A's package.json schema."""
    crops = manifest.get("crops") if isinstance(manifest, dict) else None
    if not isinstance(crops, list) or not all(
        isinstance(r, dict) and {"input_file", "input_sha256", "crop_id"} <= r.keys() for r in crops
    ):
        raise CanonicalError(
            "package manifest has no crops[] with input_file/input_sha256/crop_id; format not recognised, "
            "ask Part A for the schema instead of guessing"
        )
    if "index" in crops[0] and [int(r["index"]) for r in crops] != list(range(len(crops))):
        raise CanonicalError("package manifest crops[] are not in index order 0..n-1")
    return [(r["input_file"], r["input_sha256"], r["crop_id"]) for r in crops]


def verify_archive(set_name: str, archive: Path):
    """Check the archive and its contents against the pinned hashes. Returns (pinned, prefix, members)."""
    p = pinned(set_name)
    archive = Path(archive)
    if archive.stat().st_size != p["archive_bytes"] or sha256_file(archive) != p["archive_sha256"]:
        raise CanonicalError(f"{archive}: size/sha256 differ from {p['manifest']}")
    with zipfile.ZipFile(archive) as z:
        files = {i.filename: i for i in z.infolist() if not i.is_dir()}
        for name in files:
            pp = PurePosixPath(name)
            if pp.is_absolute() or ".." in pp.parts:
                raise CanonicalError(f"unsafe path in archive: {name}")
        lists = [n for n in files if PurePosixPath(n).name == "input_list.txt"]
        if len(lists) != 1:
            raise CanonicalError(f"expected exactly one input_list.txt in archive, found {lists}")
        prefix = str(PurePosixPath(lists[0]).parent)
        prefix = "" if prefix == "." else prefix + "/"
        input_list = z.read(lists[0])
        if _sha(input_list) != p["input_list_sha256"]:
            raise CanonicalError("input_list.txt sha256 differs from the pinned value")
        pkg = [n for n in files if n.endswith(".json") and _sha(z.read(n)) == p["package_manifest_sha256"]]
        if len(pkg) != 1:
            raise CanonicalError("no JSON file in the archive matches the pinned package-manifest sha256")
        records = _records(json.loads(z.read(pkg[0])))
        lines = [l for l in input_list.decode().splitlines() if l.strip()]
        if [r[0] for r in records] != lines:
            raise CanonicalError("package manifest order differs from input_list.txt")
        if len(records) != p["crops"] or len({r[2] for r in records}) != len(records):
            raise CanonicalError(f"expected {p['crops']} unique crop ids, got {len(records)}")
        if set_name == "smoke" and sorted(r[2] for r in records) != sorted(p["crop_ids"]):
            raise CanonicalError("smoke crop ids differ from manifests/smoke-exchange.json")
        for input_file, digest, _ in records:
            name = prefix + input_file
            if name not in files or _sha(z.read(name)) != digest:
                raise CanonicalError(f"{input_file}: missing or sha256 differs from the package manifest")
    return p, prefix, records, pkg[0]


def extract(set_name: str, archive: Path, out_dir: Path) -> dict:
    """Verify, then extract the verified inputs into a NEW directory and write canonical_inputs.json."""
    p, prefix, records, pkg_name = verify_archive(set_name, archive)
    out_dir = Path(out_dir)
    if out_dir.exists():
        raise CanonicalError(f"{out_dir} exists; extract into a new directory")
    out_dir.mkdir(parents=True)
    with zipfile.ZipFile(archive) as z:
        for name in [prefix + "input_list.txt", pkg_name] + [prefix + r[0] for r in records]:
            dest = out_dir / (name[len(prefix):] if name.startswith(prefix) else PurePosixPath(name).name)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(z.read(name))
    record = {
        "purpose": set_name,
        "pinned": p,
        "archive": str(archive),
        "package_manifest": pkg_name[len(prefix):] if pkg_name.startswith(prefix) else PurePosixPath(pkg_name).name,
        "inputs": [{"idx": i, "input_file": f, "input_sha256": d, "crop_id": c} for i, (f, d, c) in enumerate(records)],
    }
    (out_dir / RECORD).write_text(json.dumps(record, indent=2) + "\n")
    verify_extracted(out_dir, set_name)
    return record


def _pinned_package_manifest(package_dir: Path, p: dict) -> Path:
    """Find the package manifest by its pinned sha256 alone (never by a name written during unpacking)."""
    hits = [f for f in sorted(package_dir.glob("*.json")) if f.name != RECORD and sha256_file(f) == p["package_manifest_sha256"]]
    if len(hits) != 1:
        raise CanonicalError("no unique JSON file in the package matches the pinned package-manifest sha256")
    return hits[0]


def verify_extracted(package_dir: Path, set_name: str) -> dict:
    """Re-verify an extracted package from the pinned chain only, every time it is used.

    Chain: repo manifest (Part A's committed manifests/*.json) -> sha256 of the package manifest and of
    input_list.txt -> per-crop input_sha256, crop_id and ORDER from that hash-pinned package manifest ->
    sha256 of each raw file. canonical_inputs.json (written during unpacking) is never a source of
    expected values; if it disagrees with the chain in any way the package is refused.
    Returns the verified record (expected values all taken from the chain).
    """
    package_dir = Path(package_dir)
    p = pinned(set_name)
    input_list = package_dir / "input_list.txt"
    if not input_list.exists() or sha256_file(input_list) != p["input_list_sha256"]:
        raise CanonicalError("input_list.txt missing or sha256 differs from the pinned value")
    pkg_path = _pinned_package_manifest(package_dir, p)
    records = _records(json.loads(pkg_path.read_text()))
    lines = [l for l in input_list.read_text().splitlines() if l.strip()]
    if [r[0] for r in records] != lines:
        raise CanonicalError("pinned package manifest order differs from the pinned input_list.txt")
    if len(records) != p["crops"] or len({r[2] for r in records}) != len(records):
        raise CanonicalError(f"expected {p['crops']} unique crop ids, got {len(records)}")
    if set_name == "smoke" and sorted(r[2] for r in records) != sorted(p["crop_ids"]):
        raise CanonicalError("smoke crop ids differ from manifests/smoke-exchange.json")
    for input_file, digest, _ in records:
        f = package_dir / input_file
        if not f.exists() or sha256_file(f) != digest:
            raise CanonicalError(f"{input_file}: missing or sha256 differs from the pinned package manifest")
    verified = {
        "purpose": set_name,
        "pinned": p,
        "package_manifest": pkg_path.name,
        "inputs": [{"idx": i, "input_file": f, "input_sha256": d, "crop_id": c} for i, (f, d, c) in enumerate(records)],
    }
    rec_path = package_dir / RECORD
    if rec_path.exists():
        written = json.loads(rec_path.read_text())
        if any(written.get(k) != verified[k] for k in ("purpose", "pinned", "package_manifest", "inputs")):
            raise CanonicalError(f"{RECORD} disagrees with the pinned chain (modified after unpacking?)")
    return verified


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", required=True, choices=sorted(SETS))
    ap.add_argument("--archive", required=True, help="Part A's zip (qualcomm-inputs.zip or qualcomm-smoke-*.zip)")
    ap.add_argument("--out-dir", required=True, help="new directory to extract the verified inputs into")
    args = ap.parse_args()
    try:
        record = extract(args.set, Path(args.archive), REPO / args.out_dir)
    except CanonicalError as e:
        raise SystemExit(f"[canonical] REFUSED: {e}")
    out = REPO / args.out_dir
    print(f"[canonical] {args.set}: {len(record['inputs'])} crops verified against {record['pinned']['manifest']}")
    print(f"[canonical] archive sha256 {record['pinned']['archive_sha256']}")
    for name in ("input_list.txt", record["package_manifest"], RECORD):
        print(f"[canonical] wrote {rel(out / name)}  sha256 {sha256_file(out / name)}")
    print(f"[canonical] wrote {len(record['inputs'])} raw inputs under {rel(out)} (bytes unchanged from the archive)")


if __name__ == "__main__":
    main()
