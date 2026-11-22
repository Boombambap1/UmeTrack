"""Setup sanity checks: experiment config, checkpoint hash, encoder output shape."""

import hashlib
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
yaml = pytest.importorskip("yaml")

REPO = Path(__file__).resolve().parents[2]
UPSTREAM = REPO / "third_party" / "UmeTrack"
CONFIG = REPO / "configs" / "experiment.yaml"
PINNED_COMMIT = "e263ec0844abf73457468a050a75853444d33d69"


@pytest.fixture(scope="module")
def cfg():
    with CONFIG.open() as f:
        return yaml.safe_load(f)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def test_config_loads(cfg):
    for section in ("upstream", "encoder", "sdk", "data", "tolerances", "benchmark"):
        assert section in cfg, f"missing section: {section}"
    assert cfg["upstream"]["commit"] == PINNED_COMMIT


def test_checkpoint_sha256(cfg):
    ckpt = REPO / cfg["upstream"]["checkpoint"]
    assert ckpt.is_file(), f"checkpoint not found: {ckpt}"
    assert _sha256(ckpt) == cfg["upstream"]["checkpoint_sha256"]


def test_backbone_output_shape(cfg):
    # Import upstream without modifying it; don't leave __pycache__ inside the submodule.
    sys.dont_write_bytecode = True
    if str(UPSTREAM) not in sys.path:
        sys.path.insert(0, str(UPSTREAM))
    from lib.models.model_loader import load_pretrained_model

    model = load_pretrained_model(str(REPO / cfg["upstream"]["checkpoint"])).eval()
    backbone = model
    for attr in cfg["encoder"]["module"].split("."):
        backbone = getattr(backbone, attr)

    x = torch.zeros(cfg["encoder"]["input"]["shape"], dtype=torch.float32)
    with torch.no_grad():
        y = backbone(x)

    assert y.dtype == torch.float32
    assert list(y.shape) == cfg["encoder"]["output"]["shape"]
