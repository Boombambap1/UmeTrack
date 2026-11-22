import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
DATA_URL = "https://github.com/facebookresearch/UmeTrack_data.git"


def git(path, *args):
    environment = dict(os.environ, GIT_LFS_SKIP_SMUDGE="1", GIT_TERMINAL_PROMPT="0")
    return subprocess.check_output(
        ["git", "-C", str(path), *args], text=True, env=environment
    ).strip()


def ensure_repository(path, url, revision, metadata_only=False, check_only=False):
    path = Path(path)
    if not path.exists() or (path.is_dir() and not any(path.iterdir())):
        if check_only:
            raise FileNotFoundError(f"Missing dependency: {path}")
        path.mkdir(parents=True, exist_ok=True)
        git(path, "init", "--bare" if metadata_only else "--quiet")
        git(path, "remote", "add", "origin", url)
        if metadata_only:
            git(path, "config", "remote.origin.promisor", "true")
            git(path, "config", "remote.origin.partialclonefilter", "blob:none")
        options = ["fetch", "--depth=1", "--no-recurse-submodules"]
        if metadata_only:
            options.append("--filter=blob:none")
        git(path, *options, "origin", revision)
        if metadata_only:
            git(path, "update-ref", "--no-deref", "HEAD", revision)
        else:
            git(path, "checkout", "--detach", revision)
    bare = git(path, "rev-parse", "--is-bare-repository") == "true"
    repository_root = git(
        path, "rev-parse", "--absolute-git-dir" if bare else "--show-toplevel"
    )
    if Path(repository_root).resolve() != path.resolve():
        raise ValueError(f"Dependency is not its own repository: {path}")
    if bare and not metadata_only:
        raise ValueError(f"Source dependency needs a worktree: {path}")
    if git(path, "remote", "get-url", "origin") != url:
        raise ValueError(f"Dependency origin differs from the pinned source: {path}")
    if git(path, "rev-parse", "HEAD") != revision:
        raise ValueError(f"Dependency revision mismatch; existing files preserved: {path}")
    if not bare and not metadata_only:
        if git(path, "status", "--porcelain", "--untracked-files=no", "--ignore-submodules=all"):
            raise ValueError(f"Tracked source changes found; existing files preserved: {path}")
    return {"revision": revision, "metadata_only": metadata_only, "bare": bare}


def verify_checkpoint(path, expected_hash, expected_bytes):
    path = Path(path)
    if path.stat().st_size != expected_bytes:
        raise ValueError("Checkpoint byte size differs from the pinned reference")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != expected_hash:
        raise ValueError("Checkpoint checksum differs from the pinned reference")
    return digest


def main():
    parser = argparse.ArgumentParser(
        description="Restore pinned upstream sources without recursive dataset checkout."
    )
    parser.add_argument(
        "--check", action="store_true", help="Verify existing dependencies; do not initialize or fetch repositories."
    )
    args = parser.parse_args()
    upstream = json.loads((ROOT / "manifests/upstream.json").read_text())
    pytorch3d = json.loads((ROOT / "manifests/pytorch3d.json").read_text())
    source = ROOT / "third_party/UmeTrack"
    report = {
        "UmeTrack": ensure_repository(
            source, upstream["upstream_url"], upstream["upstream_commit"], check_only=args.check
        ),
        "pytorch3d": ensure_repository(
            ROOT / "third_party/pytorch3d", pytorch3d["upstream_url"],
            pytorch3d["commit"], check_only=args.check
        ),
    }
    expected = upstream["data_submodule_commit"]
    tree = git(source, "ls-tree", "HEAD", "UmeTrack_data").split()
    if tree[:3] != ["160000", "commit", expected]:
        raise ValueError("Pinned dataset revision does not match upstream gitlink")
    report["dataset_metadata"] = ensure_repository(
        source / "UmeTrack_data", DATA_URL, expected,
        metadata_only=True, check_only=args.check
    )
    report["checkpoint_sha256"] = verify_checkpoint(
        source / "pretrained_models/pretrained_weights.torch",
        upstream["checkpoint_sha256"], upstream["checkpoint_bytes"]
    )
    report["scope"] = "Sources/checkpoint only; no pip installation, video download, conversion or board execution."
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
