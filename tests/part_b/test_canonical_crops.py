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


# --- Steven's re-check of ec34f61: expected values must come only from the pinned chain ----------

def _extract_good(formal_set, tmp_path):
    archive, crops = formal_set()
    out = tmp_path / "out"
    extract("formal", archive, out)
    return out, crops


def _rewrite_record(out, fn):
    rec_path = out / canonical_crops.RECORD
    rec = json.loads(rec_path.read_text())
    fn(rec)
    rec_path.write_text(json.dumps(rec, indent=2) + "\n")


def test_bypass_a_edited_raw_with_updated_hash_in_record_rejected(formal_set, tmp_path):
    out, _ = _extract_good(formal_set, tmp_path)
    raw = out / "inputs" / "crop-00001.raw"
    raw.write_bytes(b"\x00" * raw.stat().st_size)  # different pixels, same size
    new_sha = sha(raw.read_bytes())

    def fix_hash(rec):
        rec["inputs"][1]["input_sha256"] = new_sha

    _rewrite_record(out, fix_hash)
    with pytest.raises(CanonicalError):
        verify_extracted(out, "formal")


def test_bypass_b_swapped_crop_ids_rejected(formal_set, tmp_path):
    out, _ = _extract_good(formal_set, tmp_path)

    def swap(rec):
        a, b = rec["inputs"][0], rec["inputs"][2]
        a["crop_id"], b["crop_id"] = b["crop_id"], a["crop_id"]

    _rewrite_record(out, swap)
    with pytest.raises(CanonicalError):
        verify_extracted(out, "formal")


def test_bypass_c_reordered_input_list_rejected(formal_set, tmp_path):
    out, _ = _extract_good(formal_set, tmp_path)
    lines = (out / "input_list.txt").read_text().splitlines()
    (out / "input_list.txt").write_text("\n".join([lines[1], lines[0], lines[2]]) + "\n")

    def reorder(rec):
        rec["inputs"][0], rec["inputs"][1] = rec["inputs"][1], rec["inputs"][0]

    _rewrite_record(out, reorder)
    with pytest.raises(CanonicalError):
        verify_extracted(out, "formal")


def test_bypass_record_cannot_choose_package_manifest(formal_set, tmp_path):
    """A rewritten record pointing at a forged package manifest must not be trusted."""
    out, crops = _extract_good(formal_set, tmp_path)
    forged = json.loads((out / "package.json").read_text())
    forged["crops"][0]["crop_id"], forged["crops"][1]["crop_id"] = forged["crops"][1]["crop_id"], forged["crops"][0]["crop_id"]
    (out / "forged.json").write_text(json.dumps(forged))
    _rewrite_record(out, lambda rec: rec.update(package_manifest="forged.json"))
    with pytest.raises(CanonicalError):
        verify_extracted(out, "formal")


def test_verified_ids_come_from_pinned_package_order(formal_set, tmp_path):
    out, crops = _extract_good(formal_set, tmp_path)
    got = verify_extracted(out, "formal")
    assert [r["crop_id"] for r in got["inputs"]] == [c["crop_id"] for c in crops]
    assert [r["input_file"] for r in got["inputs"]] == [c["input_file"] for c in crops]


# --- Part A's reference features (--reference-raw-dir): pinned by the same chain as the inputs ---------------

REF_BYTES = 72 * 6 * 6 * 4


def make_smoke_package(tmp, n=3):
    """Synthetic archive in Part A's smoke schema (inputs + expected-nchw), plus a smoke-exchange-style pin."""
    rng = np.random.default_rng(3)
    files, crops, feats = {}, [], []
    for i in range(n):
        x = rng.random((1, 96, 96, 1), dtype=np.float32).astype("<f4").tobytes()
        y = rng.normal(size=(1, 72, 6, 6)).astype("<f4")
        feats.append(y)
        cid = f"user_19-recording_00:000000:hand{i % 2}:camera{i}"
        inp, exp = f"inputs/crop-{i:05d}.raw", f"expected-nchw/crop-{i:05d}.raw"
        files[inp], files[exp] = x, y.tobytes()
        crops.append({"index": i, "crop_id": cid, "input_file": inp, "input_sha256": sha(x),
                      "expected_file": exp, "expected_sha256": sha(y.tobytes())})
    input_list = "".join(c["input_file"] + "\n" for c in crops).encode()
    pkg = {"schema_version": 1, "expected_layout": "NCHW", "expected_dtype": "<f4",
           "input_list_sha256": sha(input_list), "crops": crops}
    pkg_bytes = (json.dumps(pkg, indent=2) + "\n").encode()
    archive = tmp / "smoke.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("input_list.txt", input_list)
        z.writestr("manifest.json", pkg_bytes)
        for name, b in files.items():
            z.writestr(name, b)
    pin = tmp / "smoke-exchange.json"
    pin.write_text(json.dumps({
        "crops": n, "archive_sha256": sha(archive.read_bytes()), "archive_bytes": archive.stat().st_size,
        "manifest_sha256": sha(pkg_bytes), "input_list_sha256": sha(input_list),
        "crop_ids": [c["crop_id"] for c in crops],
    }))
    return archive, pin, crops, np.concatenate(feats)


@pytest.fixture
def smoke_set(monkeypatch, tmp_path):
    archive, pin, crops, feats = make_smoke_package(tmp_path)
    monkeypatch.setitem(canonical_crops.SETS, "smoke", {"manifest": str(pin), "crops": len(crops)})
    extract("smoke", archive, tmp_path / "pkg")
    canonical_crops.extract_references("smoke", archive, tmp_path / "refs")
    return tmp_path / "pkg", tmp_path / "refs", crops, feats


def test_references_verify_in_pinned_order(smoke_set):
    pkg, refs, crops, _ = smoke_set
    got = canonical_crops.verify_references(pkg, "smoke", refs, REF_BYTES)
    assert [r["crop_id"] for r in got] == [c["crop_id"] for c in crops]
    assert [r["sha256"] for r in got] == [c["expected_sha256"] for c in crops]
    assert not (refs / "inputs").exists() and not (pkg / "expected-nchw").exists()  # kept apart


def test_altered_reference_refused(smoke_set):
    pkg, refs, _, _ = smoke_set
    f = refs / "expected-nchw" / "crop-00001.raw"
    b = bytearray(f.read_bytes())
    b[100] ^= 0x01  # one bit, same size
    f.write_bytes(bytes(b))
    with pytest.raises(CanonicalError, match="crop-00001.raw.*sha256 differs"):
        canonical_crops.verify_references(pkg, "smoke", refs, REF_BYTES)


def test_missing_reference_refused(smoke_set):
    pkg, refs, _, _ = smoke_set
    (refs / "expected-nchw" / "crop-00002.raw").unlink()
    with pytest.raises(CanonicalError, match="crop-00002.raw.*missing"):
        canonical_crops.verify_references(pkg, "smoke", refs, REF_BYTES)


def test_wrong_size_reference_refused(smoke_set):
    pkg, refs, _, _ = smoke_set
    f = refs / "expected-nchw" / "crop-00000.raw"
    f.write_bytes(f.read_bytes() + b"\x00\x00\x00\x00")
    with pytest.raises(CanonicalError, match="crop-00000.raw.*10372 bytes, expected exactly 10368"):
        canonical_crops.verify_references(pkg, "smoke", refs, REF_BYTES)


def test_extra_reference_file_refused(smoke_set):
    pkg, refs, _, _ = smoke_set
    (refs / "expected-nchw" / "crop-00099.raw").write_bytes(b"\x00" * REF_BYTES)
    with pytest.raises(CanonicalError, match="unexpected files"):
        canonical_crops.verify_references(pkg, "smoke", refs, REF_BYTES)


def test_formal_schema_without_expected_files_refused(formal_set, tmp_path):
    archive, _ = formal_set()
    with pytest.raises(CanonicalError, match="no pinned reference features"):
        canonical_crops.extract_references("formal", archive, tmp_path / "refs")


def _smoke_reader(monkeypatch, tmp_path, pkg, refs, feats, extra=()):
    import read_snpe_outputs

    _write_results(tmp_path / "res", feats)
    (tmp_path / "diag.txt").write_text("".join(f"    {i}: _node_Conv_{i} : 10 us : CPU\n" for i in range(3)))
    (tmp_path / "log.txt").write_text("Successfully executed graph encoder_float\n")
    monkeypatch.setattr(sys, "argv", [
        "r", "--outputs-dir", str(tmp_path / "res"), "--output-dims", "1,6,6,72", "--output-layout", "NHWC",
        "--purpose", "smoke", "--canonical-inputs", str(pkg), "--reference-raw-dir", str(refs),
        "--target", "conversion-host", "--tolerance-key", "host_cpu_vs_reference_smoke",
        "--run-log", str(tmp_path / "log.txt"), "--diag-log", str(tmp_path / "diag.txt"),
        "--out-dir", str(tmp_path / "summary"), "--features-out", str(tmp_path / "runtime.npy"), *extra,
    ])
    with pytest.raises(SystemExit) as e:
        read_snpe_outputs.main()
    return e.value


def test_reader_smoke_against_part_a_references(smoke_set, monkeypatch, tmp_path):
    pkg, refs, crops, feats = smoke_set
    err = _smoke_reader(monkeypatch, tmp_path, pkg, refs, feats)
    s = json.loads((tmp_path / "summary" / "summary.json").read_text())
    md = (tmp_path / "summary" / "summary.md").read_text()
    assert err.code == 0 and s["pass"] and s["overall"]["max_abs"] == 0.0 and s["vs_torch"] is None
    assert [r["crop_id"] for r in s["per_crop"]] == [c["crop_id"] for c in crops]
    assert s["tolerances"]["status"].startswith("proposed")
    assert md.splitlines()[0].startswith("# Smoke check on the conversion host (x86_64 CPU), not the RB5 experiment")
    assert "Development check" not in md and "RB5 vs ONNX" not in md


def test_reader_refuses_altered_reference(smoke_set, monkeypatch, tmp_path):
    pkg, refs, _, feats = smoke_set
    f = refs / "expected-nchw" / "crop-00001.raw"
    f.write_bytes(b"\x00" * REF_BYTES)  # same size, different values
    err = _smoke_reader(monkeypatch, tmp_path, pkg, refs, feats)
    assert "refusing reference features" in str(err.code) and "crop-00001.raw" in str(err.code)
    assert not (tmp_path / "summary").exists()


def test_reader_reference_raw_dir_refused_for_dev(monkeypatch, tmp_path):
    err = _run_reader(monkeypatch, tmp_path, ["--reference-raw-dir", str(tmp_path)])
    assert "requires --purpose formal|smoke" in str(err.code)


REAL_SMOKE = REPO / "artifacts" / "qualcomm-smoke-2026-11-21.zip"


@pytest.mark.skipif(not REAL_SMOKE.exists(), reason="Part A's smoke archive is not on this machine")
def test_real_smoke_archive_references_pinned(tmp_path):
    refs = canonical_crops.extract_references("smoke", REAL_SMOKE, tmp_path / "refs")
    assert [r[0] for r in refs] == pinned("smoke")["crop_ids"]
    assert all((tmp_path / "refs" / f).stat().st_size == REF_BYTES for _, f, _ in refs)
