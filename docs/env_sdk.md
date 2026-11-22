# QAIRT SDK — version and contents

Inventory of the QAIRT SDK used for Part B. Read-only: none of the SDK's binaries were run (they are Linux-only)
and nothing from the SDK is copied into this repo. Values are also recorded in `configs/experiment.yaml` (`sdk`).

## Package

| Item | Value |
| --- | --- |
| Zip | `v2.33.2.250410.zip`, 1,184,072,995 bytes |
| Zip sha256 | `37487de4d9dd770f15826224bc915d138417eb44445ebe6bd8718b6ca172d617` |
| Unpacks to | `qairt/2.33.2.250410/` (8,338 files; the unpacked copy has all of them) |
| Location on the Mac | `~/Downloads/v2.33.2.250410.zip`, unpacked at `~/Downloads/qairt/2.33.2.250410` (not in git) |

Paths below are relative to the SDK root (`qairt/2.33.2.250410/`), since its location differs per machine.

## Version evidence

| Source | Says |
| --- | --- |
| Folder name | `2.33.2.250410` |
| `sdk.yaml` | `product: QAIRT`, `version: 2.33.2`, `build_id: 250410134701_117956`, `qnn_backend_api_version: 2.18.0` |
| `QNN_ReleaseNotes.txt` | Latest entry `2.33.2`, dated 4/15/2025 |
| `ReleaseNotes.txt` (SNPE) | Header and latest entry say `SNPE 2.33.0` |

The SNPE release notes have no 2.33.2 entry. The folder, `sdk.yaml` and the QNN notes all agree on 2.33.2, so this
looks like the SNPE notes not being updated for the patch release rather than a different build.

## Targets in the SDK

`bin/` and `lib/`: `x86_64-linux-clang`, `aarch64-oe-linux-gcc11.2`, `aarch64-oe-linux-gcc9.3`,
`aarch64-oe-linux-gcc8.2`, `aarch64-ubuntu-gcc9.4`, `aarch64-android`, `aarch64-windows-msvc`,
`arm64x-windows-msvc`, `x86_64-windows-msvc`. `lib/` also has `hexagon-v66`, `-v68`, `-v69`, `-v73`, `-v75`,
`-v79`, `android` and `python`.

## The three pieces Part B uses

| Piece | Path | Key files |
| --- | --- | --- |
| Host tools (x86_64 Ubuntu conversion host) | `bin/x86_64-linux-clang/` | `snpe-onnx-to-dlc`, `snpe-dlc-info`, `snpe-dlc-quantize` (also `snpe-net-run`) |
| Board runtime (RB5, aarch64 Linux) | `bin/aarch64-oe-linux-gcc11.2/`, `lib/aarch64-oe-linux-gcc11.2/` | `snpe-net-run`, `snpe-platform-validator`, `snpe-throughput-net-run`; `libSNPE.so`, `libSnpeDspV66Stub.so` |
| DSP libraries (QRB5165 = Hexagon v66) | `lib/hexagon-v66/unsigned/` | `libSnpeDspV66Skel.so`, `libCalculator_skel.so` (also `libQnnDspV66Skel.so`) |

The gcc11.2 OpenEmbedded build is the one the board setup used. `libSnpeDspV66Stub.so` (ARM side) and
`libSnpeDspV66Skel.so` (DSP side) must come from this same release.

## Host requirements (from the SDK's docs)

- `ReleaseNotes.txt`: converter tools need **Ubuntu 22.04** and **Python 3.10**; `sdk.yaml` also lists Ubuntu 22.04.
- `bin/check-python-dependency`: "Currently supported: Python 3.8 and 3.10".
- Framework versions the SDK targets (`sdk.yaml`): onnx 1.16.1, onnxruntime 1.17.1, onnxsimplifier 0.4.36,
  torch 1.13.1, tensorflow 2.10.1, tflite 2.3.0.
- Linux Embedded (board) platform: needs `libatomic.so.1`.

Whether convhost and the RB5 meet these is checked in D6 Phase 0/1 (`docs/env_conversion_host.md`).
