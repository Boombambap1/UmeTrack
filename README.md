# UmeTrack encoder evaluation on RB5

Evaluate whether the pretrained UmeTrack image encoder can execute on the Project Standalone RB5 system while preserving downstream hand-pose accuracy. The Mac runs the reference and remaining tracker; the board will run only the converted encoder. Tailscale/SSH provide access and transfer, separately from conversion and inference.

Task: [Hand tracking research — Steven and Ali](https://app.notion.com/p/3e6bc072402f80f984a1f73bc57c50dc). Steven owns Part A: Mac reference/replay and RB5 CPU/GPU experiments. Part B covers export/conversion, SDK setup, and DSP quantization/experiments.

## Evidence and status

Mac baseline/replay/frame-batch experiments completed November 19; batch-one timing, collaboration setup, and board diagnostics completed November 21, 2026. This repository versions the shared implementation, documentation, manifests, and small result summaries. Large experimental artifacts are managed separately.

| Experiment | Recorded result | Meaning |
| --- | --- | --- |
| Full known-skeleton/separate-hand baseline | 143 recordings, 63,798 frames; 9.425145 mm MPJPE | Reproduces the task's 9.3–9.5 mm reference band |
| Fixed paired subset | 100 frames, 335 crops, 200 valid hand-frame instances; 8.961398 mm | Reference denominator for future board comparisons |
| Original/capture/replay comparison | Exact equality; maximum pose difference 0 mm | Local adapter preserves selected reference outputs |
| Mac batch-one encoder | 3.944 ms/crop median; 4.147 ms/crop p95; 1,005 calls | Warm CPU compute on 335 fixed crops; features exactly match reference |
| Mac serial profile | Encoder median 12.232 ms/frame batch; total 29.960 ms/frame | Sampled offline CPU workload, not batch-one or headset latency |
| Raw export/import/replay | Exact equality for NCHW and NHWC using Mac fixtures | File exchange checks; no Qualcomm inference |
| RB5 backend diagnostics | GPU validator crashed; DSP validator failed with missing RPC library | Current staged runtime; see board-readiness report |
| RB5 encoder CPU/GPU/DSP | Pending | No UmeTrack board accuracy or latency established |

The evaluation uses known hand geometry and ground-truth-assisted crops. It does not test live hand detection, camera integration, or headset readiness. CPU/GPU will use the float DLC under the selected protocol; DSP will use a separately calibrated INT8 model. Float32 output files do not establish internal FP32 arithmetic.

## Repository map

- `src/umetrack_rb5/`: encoder capture/replay, sequence runner/profiler, strict raw-file exchange.
- `scripts/`: experiment entry points and pinned dependency setup.
- `tests/`: identity/state/layout/provenance checks and dependency-setup checks.
- `manifests/`: exact source revisions, dataset/crop identities, hashes, and candidate package metadata.
- `results/`: small saved experiment summaries; see their paired documents for interpretation.
- `docs/`: [architecture](docs/architecture.md), [tensor contract](docs/tensor-contract.md), [full baseline](docs/full-baseline.md), [replay/profile](docs/replay-and-profile.md), [Qualcomm handoff](docs/qualcomm-handoff.md), and [setup status](docs/setup-status.md).
- [Board readiness](docs/board-readiness.md): verified access/runtime identity and GPU/DSP diagnostic failures.
- [Part A / Part B handoff](docs/part-a-part-b-handoff.md): parallel work, conversion delivery checklist, staged inputs, and CPU acceptance.
- `examples/conversion-handoff.template.json`: Part B float conversion delivery checklist.
- `examples/board-execution.template.json`: fields required for a returned board run.
- Ignored `third_party/`, `.venv/`, and `artifacts/`: restored dependencies, Python environment, and large local research files. AI plans/guidance are also ignored.

## Restore dependencies on a fresh clone

Run commands from this repository's root. This reference was tested on macOS arm64 with Python 3.12.15, PyTorch 2.4.1, and PyTorch3D 0.7.8. Install Python 3.12, Git, and Apple command-line build tools first. Other machines need their own dependency/build validation; this setup does not certify a Qualcomm conversion host.

```sh
python3.12 scripts/bootstrap_dependencies.py
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-mac-reference.txt
PATH="$PWD/.venv/bin:$PATH" MAX_JOBS=2 MACOSX_DEPLOYMENT_TARGET=14.0 CC=clang CXX=clang++ CFLAGS=-Wno-invalid-specialization PYTORCH3D_FORCE_NO_CUDA=1 .venv/bin/python -m pip install --no-build-isolation ./third_party/pytorch3d
.venv/bin/python scripts/bootstrap_dependencies.py --check
.venv/bin/python -m pip check
.venv/bin/python scripts/check_reference_dependencies.py
.venv/bin/python scripts/check_encoder.py
```

The bootstrap script uses the existing manifests to pin UmeTrack and PyTorch3D, verifies the checkpoint, and prepares dataset Git metadata without a recursive checkout or bulk LFS/video download. New dataset metadata uses a bare partial repository at `third_party/UmeTrack/UmeTrack_data`; the evaluator reads labels through `git show`. Existing correctly pinned metadata worktrees also work. Missing label blobs may be fetched on demand. Metadata transfer size depends on server/filter support.

Existing repositories with another origin/revision or tracked source edits fail rather than being reset. `--check` verifies existing repositories and the checkpoint without initialization/fetch commands; it does not validate every dataset label, install dependencies, or test board readiness. Failed initial downloads are left for inspection rather than deleted. See [dependency setup](docs/dependency-setup.md) and [PyTorch3D build notes](docs/pytorch3d-installation.md).

## Reproduce accuracy and replay

The full evaluation downloads about 10.49 GiB of selected videos. Allow additional space for Git metadata, builds, predictions, and temporary files. Every run directory must be new. The exact paths below are for a fresh clone; preserve existing runs on an established checkout.

```sh
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=2 .venv/bin/python scripts/run_full_baseline.py --output artifacts/runs/separate-hand-bounded
.venv/bin/python scripts/validate_baseline.py artifacts/runs/separate-hand-bounded
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=2 .venv/bin/python scripts/build_reference_bundle.py --output artifacts/reference-bundle
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=2 .venv/bin/python scripts/profile_reference.py --output artifacts/runs/serial-profile
```

`build_reference_bundle.py` currently expects saved baseline predictions under `artifacts/runs/separate-hand-bounded`; changing the baseline directory alone does not redirect that dependency. Full-split evaluation is unnecessary when consuming an already verified reference bundle; bundle regeneration currently requires those baseline prefixes.

The state/reappearance test additionally expects the initial smoke recording at the following path. After a fresh full evaluation, create that local fixture from its already verified video/labels, then run the checks:

```sh
mkdir -p artifacts/data/smoke
cp -n artifacts/data/separate-hand-test/user_19/recording_00.mp4 artifacts/data/smoke/recording_00.mp4
cp -n artifacts/data/separate-hand-test/user_19/recording_00.json artifacts/data/smoke/recording_00.json
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=2 .venv/bin/python -m unittest discover -s tests -v
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=2 .venv/bin/python scripts/check_raw_exchange.py --output artifacts/raw-exchange-new
```

The raw exchange check deliberately uses saved Mac features; it is not an RB5 simulation. Existing result JSON files describe recorded runs and are not automatically rewritten by these commands.

## Next hardware gate

1. Confirm Part B's actual DLC I/O metadata, SDK build, model hash, and conversion evidence.
2. Access and staged SNPE v2.33.2.250410134701_117956 are verified. Stage the matched runtime for the benchmark account. CPU testing can begin once the float DLC is verified; investigate GPU/DSP diagnostics in parallel.
3. Exchange a few real crops and reference features, then execute all 335 crops on RB5 CPU.
4. Import outputs and replay the same 100 frames on the Mac. Planned CPU criterion: absolute paired MPJPE change below 0.1 mm versus 8.961398 mm, with direct joint-displacement diagnostics.
5. Measure warm inference separately from transfer, startup, and output I/O; repeat GPU float and DSP INT8 comparisons with backend evidence.

The candidate input archive is available locally at `artifacts/qualcomm-inputs.zip` and staged on the RB5 as recorded in `manifests/board-artifact-handoff.json`; its hashes/specification are in `manifests/qualcomm-input-package.json`. It contains no DLC or SDK. Its NHWC declaration needs confirmation against the actual model. A separate early exchange archive, `artifacts/qualcomm-smoke-2026-11-21.zip`, now includes first-frame crops from both clips and their expected NCHW features; its manifest is `manifests/smoke-exchange.json`. It is not a full pose-replay package. See [the handoff runbook](docs/qualcomm-handoff.md) for import/replay commands and output limitations.

## Upstream attribution

This project uses Meta's [UmeTrack](https://github.com/facebookresearch/UmeTrack) and [PyTorch3D](https://github.com/facebookresearch/pytorch3d), restored at the pinned revisions in `manifests/`. Their source/license notices remain in their own trees. UmeTrack's pinned license is CC BY-NC 4.0; PyTorch3D's is BSD. Dataset and model terms must be retained/reviewed for the intended use and artifact sharing. No commercial-use clearance is recorded. No upstream source or pretrained weights were modified for the reference experiments.

## Isolated batch-one timing

The separate warm CPU microbenchmark preloads all 335 frozen crops, validates each output against saved frame-batch features, performs 30 additional warm-up calls, then measures three ordered passes. Startup, file I/O, preprocessing, transfer, and downstream tracking are excluded. The November 21 run used battery power; thermal state and competing workloads were not measured. This is a Mac reference, not a board speed prediction. Full details: `results/mac-encoder-batch-one.json`.

```sh
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=2 .venv/bin/python scripts/benchmark_encoder.py --output artifacts/runs/mac-encoder-batch-one-new
```

The shared handoff directory and exact Part B delivery requirements are in [the reconciliation runbook](docs/part-a-part-b-handoff.md). Part B's conversion delivery is not available yet, as Steven confirmed November 21. Further board work awaits that collaboration.
