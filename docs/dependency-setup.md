# Pinned dependency setup

Revision: November 21, 2026. Dependencies are independent, ignored Git repositories restored by `scripts/bootstrap_dependencies.py`, not submodules of this experiment repository. This avoids copying upstream histories, build products, datasets, or weights into commits. The source manifests remain eligible for version control.

## Sources and checkpoint

| Dependency | Destination | Pin |
| --- | --- | --- |
| Meta UmeTrack | `third_party/UmeTrack` | `manifests/upstream.json`: upstream_commit |
| Meta PyTorch3D | `third_party/pytorch3d` | `manifests/pytorch3d.json`: commit |
| UmeTrack_data Git metadata | `third_party/UmeTrack/UmeTrack_data` | upstream manifest data_submodule_commit; verified against UmeTrack's gitlink |
| Pretrained checkpoint | UmeTrack's `pretrained_models/pretrained_weights.torch` | Manifest SHA256 and byte count |

Run `python3.12 scripts/bootstrap_dependencies.py` to restore missing dependencies. It fetches exact revisions with Git LFS smudging disabled and does not recurse into submodules. A new dataset repository is bare and requests blob filtering; no dataset worktree or bulk LFS checkout is created. The full evaluator fetches selected videos separately and extracts labels using `git show`; missing promisor blobs can require network access. A server ignoring filters may transfer more metadata. Do not equate partial clone with a fixed small transfer size.

Run `.venv/bin/python scripts/bootstrap_dependencies.py --check` to verify existing sources, the upstream dataset gitlink, and checkpoint. Check mode does not initialize, fetch, reset, or checkout repositories. It checks source tracked changes but ignores build/untracked files and submodule worktree state. It accepts existing pinned dataset metadata worktrees because labels are read from pinned Git objects. It does not certify dataset worktree cleanliness or hash-check all local videos; the evaluation/bundle scripts perform input checks separately.

## Preservation and recovery

The script refuses existing source repositories with another origin, revision, or tracked changes. It does not repair them automatically. Keep research artifacts and local edits, inspect the mismatch, and choose a separate recovery location before replacing anything. An interrupted first fetch may leave a partial repository; a subsequent attempt fails on its missing revision instead of silently deleting it.

The earlier dataset metadata checkout is preserved with its documented sparse-checkout caveats. A fresh clone uses the new bare-metadata route instead. No bulk dataset operation is needed merely to verify the source/checkpoint installation.

## Environment and experiment prerequisites

Follow README.md for Python dependencies and the separately compiled PyTorch3D CPU extension. Source restoration does not install Python packages or provide Qualcomm tools. The historical environment freeze is in `docs/mac-reference-environment.txt`; it is an audit record, while `requirements-mac-reference.txt` pins the declared installation dependencies.

A complete accuracy reproduction downloads the selected 143 recordings (about 10.49 GiB video) and saves baseline predictions. Bundle rebuilding requires the source clips and saved baseline prefixes at the documented paths. The no-hand/reappearance test additionally uses the smoke recording; README.md shows how to materialize it after full evaluation. A supplied verified bundle can be used for raw-file exchange without rerunning the full baseline, but downstream pose replay still needs its source videos, labels, checkpoint, and runtime dependencies.

## Checks and limits

Offline Git-fixture tests exercise source restoration, bare metadata reads, missing dependencies, revision/origin mismatch, modified tracked files, enclosing-repository detection, and checkpoint corruption. The current real dependency trees are separately checked against their manifests. This validates local setup behavior; a fresh public-network installation and clean-machine PyTorch3D build have not been repeated as part of this documentation update.
