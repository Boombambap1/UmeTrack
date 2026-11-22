"""Layout handling for SNPE raw inputs/outputs, using synthetic data in a temp dir (no board needed)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
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


REPO = Path(__file__).resolve().parents[1]
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
