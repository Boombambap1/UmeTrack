"""export_encoder never deletes an existing model: temp file + verify + atomic rename, --force keeps a backup."""

import sys
from pathlib import Path

import pytest

pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")
pytest.importorskip("onnxscript")
pytest.importorskip("torch")
pytest.importorskip("yaml")

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
import export_encoder  # noqa: E402
from export_encoder import write_atomically  # noqa: E402

OLD = b"old model bytes"


def _fail(tmp):
    tmp.write_bytes(b"half-written")
    raise RuntimeError("export blew up")


def test_existing_file_survives_failed_export(tmp_path):
    out = tmp_path / "encoder.onnx"
    out.write_bytes(OLD)
    with pytest.raises(RuntimeError):
        write_atomically(out, _fail, force=True)
    assert out.read_bytes() == OLD
    assert sorted(p.name for p in tmp_path.iterdir()) == ["encoder.onnx"]  # no temp, no backup left


def test_refuses_existing_without_force(tmp_path):
    out = tmp_path / "encoder.onnx"
    out.write_bytes(OLD)
    called = []
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        write_atomically(out, lambda tmp: called.append(tmp), force=False)
    assert not called and out.read_bytes() == OLD


def test_force_keeps_timestamped_backup(tmp_path):
    out = tmp_path / "encoder.onnx"
    out.write_bytes(OLD)
    result, backup = write_atomically(out, lambda tmp: (tmp.write_bytes(b"new"), "ok")[1], force=True)
    assert result == "ok" and out.read_bytes() == b"new"
    assert backup is not None and backup.name.startswith("encoder.onnx.bak-") and backup.read_bytes() == OLD
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(["encoder.onnx", backup.name])


def test_new_file_without_existing(tmp_path):
    out = tmp_path / "sub" / "encoder.onnx"
    _, backup = write_atomically(out, lambda tmp: tmp.write_bytes(b"new"))
    assert out.read_bytes() == b"new" and backup is None


CKPT = REPO / "third_party/UmeTrack/pretrained_models/pretrained_weights.torch"


@pytest.mark.skipif(not CKPT.exists(), reason="pinned checkpoint not present")
def test_main_existing_model_survives_failed_export(tmp_path, monkeypatch):
    """End to end: every exporter fails -> the existing model, hashes and meta are untouched."""
    out = tmp_path / "encoder.onnx"
    out.write_bytes(OLD)

    def broken_export(backbone, x, path, *a, **k):
        Path(path).write_bytes(b"truncated")
        raise RuntimeError("simulated exporter failure")

    monkeypatch.setattr(export_encoder, "_export", broken_export)
    monkeypatch.setattr(sys, "argv", [
        "export_encoder.py", "--out", str(out), "--force",
        "--hashes", str(tmp_path / "hashes.txt"), "--meta", str(tmp_path / "meta.json"),
    ])
    with pytest.raises(SystemExit, match="all exporters failed"):
        export_encoder.main()
    assert out.read_bytes() == OLD
    assert sorted(p.name for p in tmp_path.iterdir()) == ["encoder.onnx"]
