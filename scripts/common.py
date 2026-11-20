"""Shared helpers for Part B scripts: repo paths, config, hashing, upstream import."""

import hashlib
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
UPSTREAM = REPO / "third_party" / "UmeTrack"


def load_config(path):
    with Path(path).open() as f:
        return yaml.safe_load(f)


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rel(path) -> str:
    """Path relative to the repo root when possible (for manifests and logs)."""
    p = Path(path).resolve()
    try:
        return str(p.relative_to(REPO))
    except ValueError:
        return str(p)


def add_upstream_to_path() -> None:
    """Make upstream `lib` importable without modifying it or writing __pycache__ into it."""
    sys.dont_write_bytecode = True
    if str(UPSTREAM) not in sys.path:
        sys.path.insert(0, str(UPSTREAM))


def update_hashes(manifest_path, entries) -> None:
    """Upsert `<sha256>  <path>` lines (shasum format), keyed by path, sorted by path."""
    manifest_path = Path(manifest_path)
    lines = {}
    if manifest_path.exists():
        for line in manifest_path.read_text().splitlines():
            if line.strip():
                digest, p = line.split(None, 1)
                lines[p.strip()] = digest
    for p, digest in entries.items():
        lines[p] = digest
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text("".join(f"{d}  {p}\n" for p, d in sorted(lines.items())))


def load_model(cfg):
    """Load the pretrained UmeTrack model via upstream code; return (model, ckpt_path, ckpt_sha)."""
    ckpt = REPO / cfg["upstream"]["checkpoint"]
    ckpt_sha = sha256_file(ckpt)
    if ckpt_sha != cfg["upstream"]["checkpoint_sha256"]:
        raise SystemExit(f"checkpoint sha256 mismatch: {ckpt_sha} != config")
    add_upstream_to_path()
    from lib.models.model_loader import load_pretrained_model

    model = load_pretrained_model(str(ckpt)).eval().to("cpu").float()
    return model, ckpt, ckpt_sha


def load_backbone(cfg):
    """Return (image backbone at cfg encoder.module, ckpt_path, ckpt_sha)."""
    model, ckpt, ckpt_sha = load_model(cfg)
    backbone = model
    for attr in cfg["encoder"]["module"].split("."):
        backbone = getattr(backbone, attr)
    return backbone.eval(), ckpt, ckpt_sha
