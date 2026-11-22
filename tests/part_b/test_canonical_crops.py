"""Formal/smoke comparisons accept only Part A's frozen crops whose sha256 values match his manifests."""

import csv
import hashlib
import json
import sys
import zipfile
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("torch")
pytest.importorskip("yaml")
pytest.importorskip("onnxruntime")

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
import canonical_crops  # noqa: E402
from canonical_crops import CanonicalError, extract, pinned, verify_archive, verify_extracted  # noqa: E402

REC = "raw_data/real/separate_hand/testing/user_19/recording_00"


def sha(b):
    return hashlib.sha256(b).hexdigest()


def make_package(tmp, n=3, tamper_raw=False, schema_ok=True, prefix="pkg/"):
    """Synthetic archive in Part A's package.json schema, plus a Part A-style manifest pinning it."""
    rng = np.random.default_rng(0)
    raws, crops = {}, []
    for i in range(n):
        b = rng.random((1, 96, 96, 1), dtype=np.float32).astype("<f4").tobytes()
        name = f"inputs/crop-{i:05d}.raw"
        raws[name] = b
        crops.append({"index": i, "crop_id": f"user_19-recording_00:{i:06d}:hand0:camera1",
                      "input_file": name, "input_sha256": sha(b), "input_bytes": len(b)})
    input_list = "".join(c["input_file"] + "\n" for c in crops).encode()
    pkg = {"schema_version": 1, "input_list_sha256": sha(input_list), "crops": crops} if schema_ok else {"files": list(raws)}
    pkg_bytes = (json.dumps(pkg, indent=2) + "\n").encode()
    archive = tmp / "pkg.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr(prefix + "input_list.txt", input_list)
        z.writestr(prefix + "package.json", pkg_bytes)
        for name, b in raws.items():
            z.writestr(prefix + name, (b[:-1] + b"\x00") if tamper_raw and name.endswith("00001.raw") else b)
    manifest = tmp / "manifest.json"
    manifest.write_text(json.dumps({
        "crops": n, "archive_sha256": sha(archive.read_bytes()), "archive_bytes": archive.stat().st_size,
        "package_sha256": sha(pkg_bytes), "input_list_sha256": sha(input_list),
    }))
    return archive, manifest, crops


@pytest.fixture
def formal_set(monkeypatch, tmp_path):
    def setup(**kw):
        archive, manifest, crops = make_package(tmp_path, **kw)
        monkeypatch.setitem(canonical_crops.SETS, "formal", {"manifest": str(manifest), "crops": len(crops)})
        return archive, crops
    return setup


def test_real_part_a_manifests_are_pinned():
    f, s = pinned("formal"), pinned("smoke")
    assert f["crops"] == 335 and f["archive_sha256"].startswith("927b545c") and f["package_manifest_sha256"].startswith("7eab946c")
    assert s["crops"] == 7 and s["archive_sha256"].startswith("fcbcad10") and len(s["crop_ids"]) == 7


def test_good_package_extracts_and_reverifies(formal_set, tmp_path):
    archive, crops = formal_set()
    rec = extract("formal", archive, tmp_path / "out")
    assert [r["crop_id"] for r in rec["inputs"]] == [c["crop_id"] for c in crops]
    assert verify_extracted(tmp_path / "out", "formal")["purpose"] == "formal"


def test_archive_hash_mismatch_refused(formal_set, tmp_path):
    archive, _ = formal_set()
    with archive.open("ab") as f:
        f.write(b"\x00")
    with pytest.raises(CanonicalError, match="size/sha256 differ"):
        verify_archive("formal", archive)


def test_raw_not_matching_package_manifest_refused(formal_set):
    archive, _ = formal_set(tamper_raw=True)  # archive itself is pinned, but one raw differs from package.json
    with pytest.raises(CanonicalError, match="crop-00001.raw"):
        verify_archive("formal", archive)


def test_unrecognised_manifest_format_refused(formal_set):
    archive, _ = formal_set(schema_ok=False)
    with pytest.raises(CanonicalError, match="format not recognised"):
        verify_archive("formal", archive)


def test_modified_after_extraction_refused(formal_set, tmp_path):
    archive, _ = formal_set()
    extract("formal", archive, tmp_path / "out")
    raw = tmp_path / "out" / "inputs" / "crop-00002.raw"
    raw.write_bytes(raw.read_bytes()[:-1] + b"\x01")
    with pytest.raises(CanonicalError, match="crop-00002.raw"):
        verify_extracted(tmp_path / "out", "formal")


def test_extract_refuses_existing_dir(formal_set, tmp_path):
    archive, _ = formal_set()
    (tmp_path / "out").mkdir()
    with pytest.raises(CanonicalError, match="exists"):
        extract("formal", archive, tmp_path / "out")


def test_write_snpe_inputs_refuses_formal(monkeypatch):
    import write_snpe_inputs

    for purpose in ("formal", "smoke"):
        monkeypatch.setattr(sys, "argv", ["w", "--input-layout", "NHWC", "--input-dims", "1,96,96,1", "--purpose", purpose])
        with pytest.raises(SystemExit, match="never regenerated"):
            write_snpe_inputs.main()


def _write_results(d, feats):
    for i in range(feats.shape[0]):
        r = d / f"Result_{i}"
        r.mkdir(parents=True)
        np.ascontiguousarray(feats[i : i + 1].transpose(0, 2, 3, 1)).astype("<f4").tofile(r / "features.raw")


def _run_reader(monkeypatch, tmp_path, extra):
    import read_snpe_outputs

    feats = np.random.default_rng(1).normal(size=(3, 72, 6, 6)).astype(np.float32)
    np.save(tmp_path / "ref.npy", feats)
    with (tmp_path / "ref.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["row", "recording", "frame", "hand_idx", "cam_idx", "crop_id"])
        for i in range(3):
            w.writerow([i, REC, i, 0, 1, f"user_19-recording_00:{i:06d}:hand0:camera1"])
    _write_results(tmp_path / "res", feats)
    (tmp_path / "log.txt").write_text("Runtime: CPU\n")
    monkeypatch.setattr(sys, "argv", [
        "r", "--outputs-dir", str(tmp_path / "res"), "--output-dims", "1,6,6,72", "--output-layout", "NHWC",
        "--reference-manifest", str(tmp_path / "ref.csv"), "--reference", str(tmp_path / "ref.npy"),
        "--torch-reference", str(tmp_path / "ref.npy"), "--run-log", str(tmp_path / "log.txt"),
        "--out-dir", str(tmp_path / "summary"), "--features-out", str(tmp_path / "board.npy"), *extra,
    ])
    with pytest.raises(SystemExit) as e:
        read_snpe_outputs.main()
    return e.value


def test_reader_formal_requires_canonical_inputs(monkeypatch, tmp_path):
    err = _run_reader(monkeypatch, tmp_path, ["--purpose", "formal"])
    assert "requires --canonical-inputs" in str(err.code)


def test_reader_formal_refuses_tampered_package(formal_set, monkeypatch, tmp_path):
    archive, _ = formal_set()
    extract("formal", archive, tmp_path / "pkg_out")
    raw = tmp_path / "pkg_out" / "inputs" / "crop-00000.raw"
    raw.write_bytes(b"\x00" + raw.read_bytes()[1:])
    err = _run_reader(monkeypatch, tmp_path, ["--purpose", "formal", "--canonical-inputs", str(tmp_path / "pkg_out")])
    assert "refusing --purpose formal" in str(err.code)


def test_reader_formal_with_verified_package(formal_set, monkeypatch, tmp_path):
    archive, _ = formal_set()
    extract("formal", archive, tmp_path / "pkg_out")
    err = _run_reader(monkeypatch, tmp_path, ["--purpose", "formal", "--canonical-inputs", str(tmp_path / "pkg_out")])
    s = json.loads((tmp_path / "summary" / "summary.json").read_text())
    assert err.code == 0 and s["pass"] and s["purpose"] == "formal" and s["canonical"]["set"] == "formal"


def test_reader_dev_is_labelled(monkeypatch, tmp_path):
    with (tmp_path / "rt.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["idx", "raw_file", "row", "recording", "frame", "hand_idx", "cam_idx", "crop_id", "raw_sha256", "purpose"])
        for i in range(3):
            w.writerow([i, f"c{i}.raw", i, REC, i, 0, 1, f"user_19-recording_00:{i:06d}:hand0:camera1", "", "development"])
    (tmp_path / "list.txt").write_text("".join(f"inputs/c{i}.raw\n" for i in range(3)))
    _run_reader(monkeypatch, tmp_path, ["--roundtrip-manifest", str(tmp_path / "rt.csv"), "--input-list", str(tmp_path / "list.txt")])
    md = (tmp_path / "summary" / "summary.md").read_text()
    assert "development only" in md
