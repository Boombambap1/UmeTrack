# Conversion host environment (Part B, D6)

Recorded 2026-11-26. Connection details (hostname, account, keys) are kept in the local SSH config as
`convhost` and are intentionally not in this repository.

## Host

| Item | Observed |
| --- | --- |
| Machine | Shared university Ubuntu server (use approved by Vincent); no sudo |
| OS / arch | Ubuntu 22.04.5 LTS, x86_64 |
| System Python | 3.10.12 (`python3.10-venv` installed, `ensurepip` available) |
| Work directory | `~/umetrack_b/` (mode 700; SDK and model stay private) |
| Storage | Network home filesystem with ample free space; per-user quota could not be determined (`diskquota` absent, `quota -s -v` printed nothing, exit 1). No quota error during copy/unpack. |

## SDK

| Item | Value |
| --- | --- |
| Release | QAIRT 2.33.2.250410 (same zip as `configs/experiment.yaml` `sdk.zip_sha256`) |
| Copy check | rsync from the Mac; host sha256 `37487de4d9dd770f15826224bc915d138417eb44445ebe6bd8718b6ca172d617`, 1,184,072,995 bytes (match) |
| Unpack | `~/umetrack_b/qairt/qairt/2.33.2.250410/`, 8,338 files (equal to the zip listing), `unzip -t` no errors, 2.8 GB. Host zip deleted afterwards; the Mac copy is the original. |
| Env setup | `source bin/envsetup.sh` inside the venv: sets `SNPE_ROOT`, `QNN_SDK_ROOT`, `PYTHONPATH=lib/python`, `PATH=bin/x86_64-linux-clang`, `LD_LIBRARY_PATH=lib/x86_64-linux-clang` |

## Python environment (`~/umetrack_b/venv`, Python 3.10.12, 861 MB)

- `pip` upgraded 22.0.2 -> 26.3.
- `bin/check-python-dependency` (the release's own script) installed its full pinned list; all 34 packages
  report Installed == Recommended (e.g. numpy 1.26.4, protobuf 3.19.6, pyyaml 5.3, scipy 1.10.1). Exit 0.
- Then the documented framework versions (SNPE docs, `general/setup.html`): `onnx==1.16.1`, `onnxruntime==1.18.0`.
  pip replaced protobuf 3.19.6 -> 7.36.2 to satisfy onnx 1.16.1 (the SDK's pin 3.19.6 is below onnx's
  minimum of 3.20.2).
- **protobuf pinned to 3.20.3** (`pip install protobuf==3.20.3`). Reason: 7.36.2 is far outside what the SDK
  was tested with; 3.20.3 is the closest release to the SDK's 3.19.6 pin that satisfies onnx 1.16.1
  (`protobuf>=3.20.2`). This remains a deliberate deviation from the SDK's tested list (3.19.6). Final:
  numpy 1.26.4, onnx 1.16.1, onnxruntime 1.18.0, protobuf 3.20.3.
- `pip check`: `alembic` and `cryptography` (pulled in by optuna/paramiko) want typing-extensions >= 4.12 / 4.13.2,
  SDK pins 4.10.0. Not converter dependencies; left as pinned.
- `bin/envcheck -a`: clang++ found (`/usr/bin/clang++`), ONNX OK; TensorFlow, TFLite, PyTorch not importable
  (not needed for ONNX conversion).

## System packages (`bin/check-linux-dependency.sh` needs root; not run)

Checked read-only with `dpkg-query` instead:

| Package | Status |
| --- | --- |
| libncurses5, libgl1, clang (14.0), software-properties-common, python3-dev, python3.10-venv | installed |
| libc++-dev, libc++abi-dev | **not installed** |
| libc++1-14, libc++abi1-14 (runtime libs) | not installed; apt candidate 1:14.0.0-1ubuntu1.1 |

## Converter startup (blocker found and resolved without root)

First attempt: `snpe-onnx-to-dlc --help` and `snpe-dlc-info --help` both exited 1:

```
ImportError: cannot import name 'libPyIrGraph' from partially initialized module 'qti.aisw.converters.common'
```

Cause (`ldd lib/python/qti/aisw/converters/common/linux-x86_64/libPyIrGraph.so`):
`libpython3.10.so.1.0` resolves, but **`libc++.so.1` and `libc++abi.so.1` are not found**. The SDK does not ship
them in `lib/x86_64-linux-clang`, and no copy exists on the host (`ldconfig -p`, `/usr/lib`, `/usr/lib/llvm-*`,
`/usr/local/lib`, `/opt`). Installing them requires system packages (`libc++1-14`, `libc++abi1-14`, or
`libc++-dev libc++abi-dev`, which the SDK's Linux script installs as root).

Fix (no root, nothing changed on the system): `apt-cache depends` showed `libc++1-14` and `libc++abi1-14` both
also need `libunwind-14` (not installed; `libc6`, `libgcc-s1` are). The three official Ubuntu jammy-updates
packages were fetched with `apt-get download` (checked by apt against the host's configured Ubuntu mirror
index) and unpacked with `dpkg-deb -x` into `~/umetrack_b/deps/`:

| .deb (version 1:14.0.0-1ubuntu1.1, amd64) | bytes | sha256 |
| --- | --- | --- |
| libc++1-14 | 288,672 | `e6ff175a61d9c51808c31d0a1333519380d5dd1e5256c4c592e88683544f642f` |
| libc++abi1-14 | 66,910 | `37690fe144864f8469a27a99a70237c361e3314ba1e2ce91cc547c577452bf07` |
| libunwind-14 | 20,282 | `21bd0aeff037b4ea6f238165360e73df5d518bfbc764ca9c7d31904d1d14049e` |

Every converter invocation therefore uses, after `source bin/envsetup.sh` in the venv:

```sh
export LD_LIBRARY_PATH=$HOME/umetrack_b/deps/usr/lib/llvm-14/lib:$LD_LIBRARY_PATH
```

With that, `ldd libPyIrGraph.so` resolves `libc++.so.1`, `libc++abi.so.1` and `libunwind.so.1` from
`deps/usr/lib/llvm-14/lib`, and `snpe-onnx-to-dlc --help` and `snpe-dlc-info --help` both exit 0. Undo: delete
`~/umetrack_b/deps/`.

## Raw logs

On the host in `~/umetrack_b/logs/`, copied to ignored `results/raw/conversion_host/` on the Mac:

| File | sha256 |
| --- | --- |
| check-python-dependency.log | `9d2a51f8fecfad28c27dc8aaf2a33134313f4e9fa1c37049b0a3dfd4583a965b` |
| pip-onnx.log | `7380857feb2959b34d80375eceadfdc80e669b892045c2de649a0346b239dba5` |
| pip-protobuf.log | `3dd4680aadf13cc3c3ee520da9257d7af07de1eb4e698c8d3cc2c6f369c71adb` |
| pip-check.log (after protobuf pin; same two typing-extensions notes) | `1788fceb68b1835ab1bae48f5d59058d35b779198eb9f5ff55bd9cfc4d39c0a8` |
| pip-freeze.txt (after protobuf pin) | `bbe96190877d811e8548278c5a84d29201f142234e0c843ec2a779603482aaf4` |
| envcheck.log | `d18d87455d0eac1293c8aa115a555a38be2d3cc74174315f2d8680a1930c3d1e` |
| snpe-onnx-to-dlc.help.txt (working help) | `4b14f23c23e8bfa12d4321afb94a2457753ea2e380160da4928cf5035cbab462` |
| snpe-dlc-info.help.txt (working help) | `ae45236439aa14378a4687fea1756e9375633afa2625229f9703f7387263a1e0` |

Earlier failing `--help` outputs (the `libPyIrGraph` traceback) had sha256 `1cce1f84…` / `88886cf5…`; they were
overwritten on the host by the working runs.

## Conversion notes (verbatim, with explanation)

Full logs: ignored `results/raw/conversion/` (`dry_run.log`, `convert.log`, `dlc_info.txt`); hashes in
`results/conversion/summary.md`.

**Dry run** (`snpe-onnx-to-dlc ... --dry_run info`, exit 1). It printed:

```
|node_relu      | Relu: unsupported version    |         |          |                                         |
|node_max_pool2d| MaxPool: unsupported version |         |          | storage_order: unsupported in Converter |
```

(one such row for each of the 25 Relu nodes), then crashed:

```
  File ".../lib/python/qti/aisw/converters/onnx/onnx_loader.py", line 376, in native_checker
    sys.exit(0)
NameError: name 'sys' is not defined
```

- The crash is an SDK bug: `onnx_loader.py` calls `sys.exit(0)` at the end of dry-run mode but never imports
  `sys`. It happens after the table is printed and says nothing about the model.
- The table comes from the legacy dry-run checker (`onnx/model_evaluator.py`, `LATEST_SUPPORTED_ONNX_VERSION = 11`),
  not from the conversion path. At opset 17 the model's ops are Conv-11 (x29), Add-14 (x12), MaxPool-12 (x1) and
  Relu-14 (x25). The converter registers Conv [1, 11], Add [1, 6, 7, 13, 14], MaxPool [1, 8, 10, 11, 12] and
  Relu [1, 6]. MaxPool-12 is therefore supported; its flag is the legacy checker. The model's MaxPool has
  `storage_order = 0` (the default, row-major), so the attribute note concerns the attribute's presence, not a
  non-default value.

**Conversion** (exit 0, `INFO_CONVERSION_SUCCESS: Conversion completed successfully`). Warnings, verbatim:

```
WARNING - WARNING_OP_VERSION_NOT_SUPPORTED: Operation Relu Not Supported. Expected operator version: [1, 6], instead got version: [14]
```

(25 times, once per Relu), plus:

```
WARNING - Onnx model simplification failed due to: No module named 'onnxsim'
WARNING - Simplified model validation failed
WARNING - --model_version option is deprecated, use --set_model_version.
```

- Relu: in the conversion path (`onnx_translations.py`, `validate_op_ver`) an unregistered op version only logs
  this warning and conversion continues with the registered Relu translation. Relu-13/14 only added data types
  (bfloat16, integer types); for float32 Relu is `max(x, 0)` in every version. Whether the converted graph is
  numerically right is checked by the development check on the conversion host and by the RB5 CPU run, not by
  this argument.
- onnxsim is not in the SDK's tested package list, so the converter skipped its optional ONNX simplification and
  converted the model as exported.
- `--model_version` still took effect (`snpe-dlc-info`: `Model Version: umetrack-encoder-onnx-1e865865`).

## Source model on the host

`~/umetrack_b/model/encoder.onnx`: 15,098,488 bytes, sha256
`1e86586510d928febb2db643caaef6e13fc48e4e5dd236424422cd89f34fa483` (matches the Mac artifact).
