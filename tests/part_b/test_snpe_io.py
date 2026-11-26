"""Layout handling for SNPE raw inputs/outputs, using synthetic data in a temp dir (no board needed)."""

import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
yaml = pytest.importorskip("yaml")
pytest.importorskip("onnxruntime")

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from read_snpe_outputs import read_outputs, to_nchw  # noqa: E402
from write_snpe_inputs import parse_dims, select_rows, to_layout, write_inputs  # noqa: E402


def test_layout_round_trip():
    x = np.random.default_rng(0).normal(size=(2, 72, 6, 6)).astype(np.float32)
    for layout in ("NCHW", "NHWC"):
        np.testing.assert_array_equal(to_nchw(to_layout(x, layout), layout), x)


def test_nhwc_input_bytes_equal_nchw_when_c_is_1():
    x = np.random.default_rng(1).random((1, 1, 96, 96), dtype=np.float32)
    assert to_layout(x, "NHWC").tobytes() == x.tobytes()


def test_write_inputs(tmp_path):
    crops = np.random.default_rng(2).integers(0, 256, size=(10, 96, 96), dtype=np.uint8)
    rows = [0, 3, 9]
    written = write_inputs(crops, rows, "NHWC", [1, 96, 96, 1], tmp_path, "inputs")
    assert [r for _, r in written] == rows
    lst = (tmp_path / "input_list.txt").read_text().splitlines()
    assert lst == [f"inputs/{p.name}" for p, _ in written]
    for path, row in written:
        raw = np.fromfile(path, dtype="<f4").reshape(1, 96, 96, 1)
        np.testing.assert_array_equal(raw[0, :, :, 0], crops[row].astype(np.float32) / 255.0)


def test_write_inputs_rejects_wrong_dims(tmp_path):
    crops = np.zeros((2, 96, 96), dtype=np.uint8)
    with pytest.raises(SystemExit):
        write_inputs(crops, [0], "NHWC", [1, 1, 96, 96], tmp_path, "inputs")


def test_read_outputs_nhwc(tmp_path):
    feats = np.random.default_rng(3).normal(size=(3, 72, 6, 6)).astype(np.float32)
    for i in range(3):
        d = tmp_path / f"Result_{i}"
        d.mkdir()
        to_layout(feats[i : i + 1], "NHWC").astype("<f4").tofile(d / "features.raw")
    got = read_outputs(tmp_path, 3, "features.raw", [1, 6, 6, 72], "NHWC")
    np.testing.assert_array_equal(got, feats)
    # Reading NHWC bytes as if NCHW must NOT silently match.
    wrong = read_outputs(tmp_path, 3, "features.raw", [1, 72, 6, 6], "NCHW")
    assert not np.array_equal(wrong, feats)


def test_read_outputs_wrong_size(tmp_path):
    (tmp_path / "Result_0").mkdir()
    np.zeros(10, dtype="<f4").tofile(tmp_path / "Result_0" / "features.raw")
    with pytest.raises(SystemExit):
        read_outputs(tmp_path, 1, "features.raw", [1, 6, 6, 72], "NHWC")


def test_helpers():
    assert parse_dims("1,96,96,1") == [1, 96, 96, 1]
    assert parse_dims("1x6x6x72") == [1, 6, 6, 72]
    assert select_rows(60, 8) == [0, 8, 17, 25, 34, 42, 51, 59]


REPO = Path(__file__).resolve().parents[2]
D5_CROPS = REPO / "artifacts/crops/crops_uint8.npy"


@pytest.mark.skipif(not D5_CROPS.exists(), reason="D5 crops not generated (artifacts/ is ignored)")
def test_real_crops_nhwc_raw_round_trip(tmp_path):
    """Real D5 crops -> NHWC .raw -> read back -> NCHW must equal uint8/255 exactly."""
    from check_onnx_parity import preprocess

    crops = np.load(D5_CROPS)
    rows = list(range(crops.shape[0]))
    written = write_inputs(crops, rows, "NHWC", [1, 96, 96, 1], tmp_path, "inputs")
    for path, row in written:
        back = to_nchw(np.fromfile(path, dtype="<f4").reshape(1, 96, 96, 1), "NHWC")
        want = preprocess(crops[row : row + 1])
        assert back.shape == (1, 1, 96, 96) and back.dtype == np.float32
        np.testing.assert_array_equal(back, want)
        np.testing.assert_array_equal(back[0, 0], crops[row].astype(np.float32) / 255.0)


def test_crop_id_matches_part_a_format():
    from common import crop_id

    rec = "raw_data/real/separate_hand/testing/user_19/recording_00"
    assert crop_id(rec, 0, 0, 1) == "user_19-recording_00:000000:hand0:camera1"
    assert crop_id(rec, "123", "1", "2") == "user_19-recording_00:000123:hand1:camera2"
    with pytest.raises(ValueError):
        crop_id("recording_00", 0, 0, 0)


def test_crop_id_agrees_with_steven_smoke_set():
    """Our first D5 rows (user_19 frame 0) must give exactly Part A's smoke-set IDs for that frame."""
    import csv
    import json

    from common import crop_id

    smoke = json.loads((REPO / "manifests/smoke-exchange.json").read_text())["crop_ids"]
    with (REPO / "manifests/crops_d5.csv").open() as f:
        ours = [
            crop_id(r["recording"], r["frame"], r["hand_idx"], r["cam_idx"])
            for r in csv.DictReader(f)
            if r["frame"] == "0"
        ]
    assert ours == [s for s in smoke if s.startswith("user_19-recording_00:000000:")]
    assert len(ours) == 3


# --- Phase 1 audit fixes: exact sizes, exact Result dirs, backend evidence, two verdicts -----------

def _write_results(d, feats, layout="NHWC"):
    for i in range(feats.shape[0]):
        r = d / f"Result_{i}"
        r.mkdir(parents=True)
        to_layout(feats[i : i + 1], layout).astype("<f4").tofile(r / "features.raw")


def test_read_outputs_rejects_trailing_byte(tmp_path):
    feats = np.zeros((2, 72, 6, 6), np.float32)
    _write_results(tmp_path, feats)
    with (tmp_path / "Result_1" / "features.raw").open("ab") as f:
        f.write(b"\x00")
    with pytest.raises(SystemExit, match="bytes, expected exactly"):
        read_outputs(tmp_path, 2, "features.raw", [1, 6, 6, 72], "NHWC")


def test_read_outputs_rejects_extra_and_missing_result_dirs(tmp_path):
    feats = np.zeros((2, 72, 6, 6), np.float32)
    _write_results(tmp_path, feats)
    (tmp_path / "Result_2").mkdir()
    with pytest.raises(SystemExit, match="unexpected"):
        read_outputs(tmp_path, 2, "features.raw", [1, 6, 6, 72], "NHWC")
    with pytest.raises(SystemExit, match="missing"):
        read_outputs(tmp_path, 4, "features.raw", [1, 6, 6, 72], "NHWC")


def test_read_outputs_rejects_extra_file_in_result_dir(tmp_path):
    _write_results(tmp_path, np.zeros((1, 72, 6, 6), np.float32))
    (tmp_path / "Result_0" / "other.raw").write_bytes(b"")
    with pytest.raises(SystemExit, match="expected only"):
        read_outputs(tmp_path, 1, "features.raw", [1, 6, 6, 72], "NHWC")


def test_parse_runtime():
    from read_snpe_outputs import backend_check, parse_runtime

    assert parse_runtime("") == (None, [])
    assert parse_runtime("loading model\nRuntime: CPU\ndone")[0] == "CPU"
    assert parse_runtime("Using DSP runtime")[0] == "DSP"
    assert parse_runtime("Runtime: CPU\nRuntime: DSP")[0] is None  # conflicting -> unverified
    assert parse_runtime("--runtime_order cpu")[0] is None  # command-line flags are not evidence
    assert backend_check(None, "cpu")["backend_verified"] is False


def _synthetic_run(tmp_path, log_text=None, requested="cpu", extra=()):
    """Full read_snpe_outputs.main() on synthetic data that matches the reference exactly."""
    import csv as _csv

    import read_snpe_outputs
    from common import crop_id

    rec = "raw_data/real/separate_hand/testing/user_19/recording_00"
    feats = np.random.default_rng(7).normal(size=(3, 72, 6, 6)).astype(np.float32)
    np.save(tmp_path / "ref.npy", feats)
    with (tmp_path / "ref.csv").open("w", newline="") as f:
        w = _csv.writer(f)
        w.writerow(["row", "recording", "frame", "hand_idx", "cam_idx"])
        for i in range(3):
            w.writerow([i, rec, i, 0, 1])
    order = [2, 0, 1]  # board order differs from reference order: must match by crop_id
    with (tmp_path / "rt.csv").open("w", newline="") as f:
        w = _csv.writer(f)
        w.writerow(["idx", "raw_file", "row", "recording", "frame", "hand_idx", "cam_idx", "crop_id", "raw_sha256"])
        for i, row in enumerate(order):
            w.writerow([i, f"crop_{i:03d}.raw", row, rec, row, 0, 1, crop_id(rec, row, 0, 1), ""])
    (tmp_path / "input_list.txt").write_text("".join(f"inputs/crop_{i:03d}.raw\n" for i in range(3)))
    _write_results(tmp_path / "out", feats[order])
    log = tmp_path / "net-run.log"
    if log_text is not None:
        log.write_text(log_text)
    argv = [
        "read_snpe_outputs.py", "--outputs-dir", str(tmp_path / "out"), "--output-dims", "1,6,6,72",
        "--output-layout", "NHWC", "--input-list", str(tmp_path / "input_list.txt"),
        "--roundtrip-manifest", str(tmp_path / "rt.csv"), "--reference-manifest", str(tmp_path / "ref.csv"),
        "--reference", str(tmp_path / "ref.npy"), "--torch-reference", str(tmp_path / "ref.npy"),
        "--requested-runtime", requested, "--run-log", str(log), "--out-dir", str(tmp_path / "summary"),
        "--features-out", str(tmp_path / "board.npy"), *extra,
    ]
    old = sys.argv
    sys.argv = argv
    try:
        with pytest.raises(SystemExit) as e:
            read_snpe_outputs.main()
    finally:
        sys.argv = old
    import json as _json

    summary = tmp_path / "summary" / "summary.json"
    return e.value.code, (_json.loads(summary.read_text()) if summary.exists() else None)


def test_main_no_log_is_not_verified(tmp_path):
    code, s = _synthetic_run(tmp_path, log_text=None)
    assert s["numeric_match"] is True and s["backend_verified"] is False and s["pass"] is False
    assert s["backend"]["reason"] == "no snpe-net-run log" and code == 1


def test_main_unparseable_log_is_not_verified(tmp_path):
    _, s = _synthetic_run(tmp_path, log_text="snpe-net-run --runtime_order cpu\nfinished\n")
    assert s["numeric_match"] is True and s["backend_verified"] is False and s["pass"] is False


def test_main_wrong_runtime_is_not_verified(tmp_path):
    _, s = _synthetic_run(tmp_path, log_text="Runtime: DSP\n", requested="cpu")
    assert s["backend_verified"] is False and s["pass"] is False and s["backend"]["actual_runtime"] == "DSP"


def test_main_both_verdicts_true_passes(tmp_path):
    code, s = _synthetic_run(tmp_path, log_text="Runtime: CPU\n")
    assert s["numeric_match"] and s["backend_verified"] and s["pass"] and code == 0
    assert [r["crop_id"].split(":")[1] for r in s["per_crop"]] == ["000002", "000000", "000001"]


def test_main_default_target_is_rb5_and_reports_rmse(tmp_path):
    _, s = _synthetic_run(tmp_path, log_text="Runtime: CPU\n")
    md = (tmp_path / "summary" / "summary.md").read_text()
    assert s["target"] == "rb5" and s["tolerance_key"] == "board_cpu_vs_onnx" and s["overall"]["rmse"] == 0.0
    assert md.startswith("# D6 — RB5 vs ONNX") and "RMSE" in md


def test_main_conversion_host_target_is_labelled_development_check(tmp_path):
    _, s = _synthetic_run(tmp_path, log_text="Runtime: CPU\n",
                          extra=["--target", "conversion-host", "--tolerance-key", "host_cpu_vs_onnx_dev"])
    md = (tmp_path / "summary" / "summary.md").read_text()
    assert s["target"] == "conversion-host" and s["tolerance_key"] == "host_cpu_vs_onnx_dev"
    assert "Development check on the conversion host" in md.splitlines()[0]
    assert "not the RB5 experiment" in md and "RB5 vs ONNX" not in md and "development only" in md


def test_main_undeclared_tolerance_key_refused(tmp_path):
    code, s = _synthetic_run(tmp_path, log_text="Runtime: CPU\n", extra=["--tolerance-key", "made_up_key"])
    assert s is None and "tolerances.made_up_key is not declared" in str(code)


# --- Steven's re-check of ec34f61: runtime evidence must show what EXECUTED -----------------------

def _verified(text, requested, diag=None, tmp_path=None):
    from read_snpe_outputs import backend_check

    log = tmp_path / "net-run.log"
    log.write_text(text)
    diag_path = None
    if diag is not None:
        diag_path = tmp_path / "diag.txt"
        diag_path.write_text(diag)
    return backend_check(log, requested, diag_path) if diag is not None else backend_check(log, requested)


def test_bypass_selected_then_fallback_is_not_verified(tmp_path):
    r = _verified("Runtime selected: GPU\nGPU unavailable; falling back to CPU\n", "gpu", tmp_path=tmp_path)
    assert r["backend_verified"] is False


def test_bypass_command_line_only_is_not_verified(tmp_path):
    r = _verified("snpe-net-run --container encoder.dlc --input_list input_list.txt --use_gpu\n", "gpu", tmp_path=tmp_path)
    assert r["backend_verified"] is False


def test_bypass_request_only_text_is_not_verified(tmp_path):
    for text in ("Requested runtime: GPU\n", "requested runtime = DSP\n", "$ snpe-net-run --runtime_order dsp\n",
                 "+ snpe-net-run --container m.dlc --use_dsp   # runtime: DSP\n"):
        requested = "dsp" if "DSP" in text.upper() else "gpu"
        r = _verified(text, requested, tmp_path=tmp_path)
        assert r["backend_verified"] is False, text


def test_negative_lines_override_positive_evidence(tmp_path):
    for bad in ("DSP runtime not supported", "Failed to create DSP runtime", "error: could not initialize GPU",
                "Runtime DSP is not available", "fallback to CPU enabled"):
        r = _verified(f"Runtime: DSP\n{bad}\n", "dsp", tmp_path=tmp_path)
        assert r["backend_verified"] is False, bad


def test_positive_stdout_evidence_still_verifies(tmp_path):
    r = _verified("Loading model\nRuntime: CPU\nFinished executing 8 inputs\n", "cpu", tmp_path=tmp_path)
    assert r["backend_verified"] is True and r["source"] == "stdout"


def test_diag_runtime_preferred_over_stdout(tmp_path):
    diag = "Layer 0 (conv1) Runtime: CPU\nLayer 1 (relu1) Runtime: CPU\n"
    r = _verified("Runtime: GPU\n", "gpu", diag=diag, tmp_path=tmp_path)
    assert r["backend_verified"] is False and r["actual_runtime"] == "CPU" and r["source"] == "diag"
    r = _verified("", "cpu", diag=diag, tmp_path=tmp_path)
    assert r["backend_verified"] is True and r["source"] == "diag"


def test_diag_mixed_layers_not_verified(tmp_path):
    diag = "Layer 0 (conv1) Runtime: DSP\nLayer 1 (relu1) Runtime: CPU\n"
    r = _verified("Runtime: DSP\n", "dsp", diag=diag, tmp_path=tmp_path)
    assert r["backend_verified"] is False
