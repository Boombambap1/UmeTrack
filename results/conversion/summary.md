# D6 — float DLC conversion (Part B)

Date: 2026-11-26. Converted on the conversion host (`convhost`, Ubuntu 22.04.5 x86_64; see
`docs/env_conversion_host.md`). **No RB5 run is part of this summary.**

## Result

| Item | Value |
| --- | --- |
| Converted model | `encoder_float.dlc` (kept out of Git; local copy `artifacts/encoder_float.dlc`) |
| DLC bytes / sha256 | 15,145,523 / `d0342d35a65f9de43967ffc1d2f6bfb2356e92b80b3dacd35bf57e5bf1572f7e` (identical on host and Mac) |
| Converter | `snpe-onnx-to-dlc`, QAIRT 2.33.2.250410; DLC reports "created with converter version: 2.33.2.250410134701_117956" (same build as the RB5 runner observed on Nov 21) |
| Model version string | `umetrack-encoder-onnx-1e865865` |
| Source ONNX | `artifacts/encoder.onnx`, 15,098,488 bytes, sha256 `1e86586510d928febb2db643caaef6e13fc48e4e5dd236424422cd89f34fa483`, opset 17, IR 10 (re-verified on the host) |
| Checkpoint | `pretrained_weights.torch` sha256 `72eab0ea2c5c4518188ede4cf62e98f62eb1e4ef13628c0f20ad3fc56d1bf18d`, upstream `e263ec0844abf73457468a050a75853444d33d69` |
| Export helper | `scripts/export_encoder.py` sha256 `9fa0cb0c3ace7f629894414173bfadb77c749b3962f9735884361d356dfe06c0`; metadata `results/export/export_meta.json` |
| Parameters / MACs | 3,769,928 / 484M per inference (snpe-dlc-info) |

Command (run in `~/umetrack_b/`, exit 0, `INFO_CONVERSION_SUCCESS: Conversion completed successfully`):

```sh
snpe-onnx-to-dlc -i model/encoder.onnx --input_dim crops 1,1,96,96 --input_layout crops NCHW --out_name features --float_bitwidth 32 --model_version umetrack-encoder-onnx-1e865865 -o conversion/encoder_float.dlc
```

Environment for every SDK command: venv activated, `source bin/envsetup.sh`, and
`LD_LIBRARY_PATH=$HOME/umetrack_b/deps/usr/lib/llvm-14/lib:$LD_LIBRARY_PATH` (user-space libc++; see env doc).

## Converted I/O (from `snpe-dlc-info`)

| | Name | Dims | dtype | Layout | Batch |
| --- | --- | --- | --- | --- | --- |
| Input | `crops` | `[1,96,96,1]` | Float_32 | NHWC | fixed 1 |
| Output | `features` | `[1,6,6,72]` | Float_32 | NHWC | fixed 1 |

- Input is already preprocessed (uint8 / 255 -> float32); no encoding/normalisation in the DLC ("No encoding info").
- The converter reordered the graph to spatial-last order (`perform_axes_to_spatial_first_order=True` in its recorded
  command). For C = 1 the NHWC input bytes equal NCHW `[1,1,96,96]`. The output must be transposed explicitly
  NHWC `[1,6,6,72]` -> NCHW `[1,72,6,6]` to match the canonical contract.
- `snpe-net-run` writes one file per `Result_<i>/`: `features.raw`, little-endian float32, 10,368 bytes.
- A float32 container does not by itself prove FP32 arithmetic on an accelerator; `--float_bitwidth 32` was used.

## Warnings and notes (verbatim; explanation in `docs/env_conversion_host.md`)

Conversion log, 25 times (once per Relu):

```
WARNING - WARNING_OP_VERSION_NOT_SUPPORTED: Operation Relu Not Supported. Expected operator version: [1, 6], instead got version: [14]
```

and

```
WARNING - Onnx model simplification failed due to: No module named 'onnxsim'
WARNING - Simplified model validation failed
WARNING - --model_version option is deprecated, use --set_model_version.
```

Dry run (`--dry_run info`, exit 1) listed `Relu: unsupported version` for every Relu and
`MaxPool: unsupported version | storage_order: unsupported in Converter` for the MaxPool, then crashed with
`NameError: name 'sys' is not defined` at `onnx_loader.py` line 376 (`sys.exit(0)`; the module never imports
`sys`, an SDK bug).

In short: the dry-run table comes from the legacy checker (`LATEST_SUPPORTED_ONNX_VERSION = 11`); MaxPool-12 is
registered by the converter and `storage_order` is 0 (default). Relu-14 is not registered ([1, 6]); the conversion
path only warns and uses its Relu translation. Relu-13/14 only added data types; for float32 Relu is `max(x, 0)`.
onnxsim is not in the SDK's tested list, so the optional simplification was skipped. The development and smoke checks below
are the numerical evidence for this DLC on the host; the RB5 CPU run is still required.

## Development check on the conversion host (not the RB5 experiment)

Development check on the conversion host: the SDK's x86_64 `snpe-net-run` (CPU) ran `encoder_float.dlc` on 8 of
Part B's 60 development crops (rows 0, 8, 17, 25, 34, 42, 51, 59; written in the reported NHWC layout by
`scripts/write_snpe_inputs.py --purpose dev`) and the outputs were compared with ONNX Runtime features by
`scripts/read_snpe_outputs.py --purpose dev --target conversion-host`. Part B's crops are development only (they are
not Steven's frozen crops) and the host is not the RB5: this says nothing about RB5 CPU accuracy or timing.

Tolerance `tolerances.host_cpu_vs_onnx_dev` (max_abs 1e-3, per-crop rel_l2 1e-4) declared in
`configs/experiment.yaml` before the run.

| Metric (8 crops, after explicit NHWC -> NCHW) | Value | Limit | Within |
| --- | --- | --- | --- |
| max_abs | 2.265e-06 | 1.0e-03 | yes |
| worst per-crop rel_l2 | 6.919e-07 (crop 3) | 1.0e-04 | yes |
| overall rel_l2 | 3.869e-07 | — | — |
| MAE | 4.324e-08 | — | — |
| RMSE | 8.309e-08 | — | — |

- numeric_match: **True**.
- backend_verified: **True** (requested CPU; all 67 per-layer lines in the snpe-diagview output report `CPU`; no
  fallback/unavailable/error lines in the run or diag log). First run: **False, as measured**, because the reader's
  diag pattern did not match the real per-layer format (`0: _node_Conv_102 : 1919 us : CPU`) and it reported
  "no per-layer runtime in diag log". The parser was then updated to this format in a separate commit (all safety
  rules kept, tests from real-format lines) and the comparison re-run. The format is confirmed on the conversion
  host's x86_64 log only and must be re-confirmed on the first RB5 log.
- Layout mapping check: reading the same outputs as NCHW `[1,72,6,6]` (wrong) gives max_abs 1.890e+00, rel_l2
  1.281e+00, so only the NHWC -> NCHW mapping matches.
- Full per-crop table: `results/conversion/host_dev_check/summary.md`.

## Smoke check on the conversion host (x86_64 CPU), not the RB5 experiment

Steven's seven frozen smoke crops (frame 0 of user_19 and user_23 `recording_00`) run through `encoder_float.dlc`
with the host's x86_64 `snpe-net-run` (`--runtime_order cpu --profiling_level detailed`, exit 0), compared with
Steven's own reference features (`expected-nchw/`). This is a host check: it says nothing about RB5 CPU accuracy or
timing. **The RB5 CPU run on Steven's crops is Steven's and is still pending.**

- Archive `artifacts/qualcomm-smoke-2026-11-21.zip` (151,000 bytes, sha256 `fcbcad10e9b53795451506380734bc9f8ba6f883df725230e9c0b4e52eafbdac`)
  verified by `scripts/canonical_crops.py --set smoke` against `manifests/smoke-exchange.json` (internal manifest
  `cc0f5ca7…`, input list `59c41769…`). Inputs and `input_list.txt` copied to the host unchanged (sha256 re-checked
  there); host DLC sha256 re-checked (`d0342d35…`).
- References: `canonical_crops.py --references` extracted only `expected-nchw/` into their own folder; every file is
  re-verified against `expected_sha256` in the pinned manifest and matched by crop_id in the verified order
  (`read_snpe_outputs.py --reference-raw-dir`). Inputs, references and runtime outputs are kept in separate folders.
- Tolerance `tolerances.host_cpu_vs_reference_smoke` (max_abs 1e-3, per-crop rel_l2 1e-4; proposed, pending agreement
  with Steven) declared and committed before the run.

| Metric (7 crops, host NHWC -> NCHW vs Steven's NCHW) | Value | Limit | Within |
| --- | --- | --- | --- |
| max_abs | 1.073e-06 (crop 6) | 1.0e-03 | yes |
| worst per-crop rel_l2 | 4.223e-07 (crop 4) | 1.0e-04 | yes |
| overall rel_l2 | 3.006e-07 | — | — |
| MAE | 4.189e-08 | — | — |
| RMSE | 6.442e-08 | — | — |

- numeric_match: **True**. backend_verified: **True** (requested CPU; all 67 snpe-diagview per-layer lines report
  `CPU`; no fallback/unavailable/error lines).
- Layout mapping check: reading the same outputs as NCHW (wrong) gives max_abs 2.027e+00, rel_l2 1.285e+00.
- Per-crop max_abs / MAE / RMSE / rel_l2: `results/conversion/host_smoke_check/summary.md`.

## Raw logs (ignored `results/raw/`)

| File | sha256 |
| --- | --- |
| conversion/convert_cmd.txt | `41ec5aca3d38659fbd5d78cad9d8e5b48a45a72f32a60d7212f0282be88d14aa` |
| conversion/convert.log | `57c1d921e654efcfb2c07bc6f82f9462302a7d061cb524863c9249102d1c2aa3` |
| conversion/convert.exit (`exit=0`) | `19eaf43821a7660ec323a87c8457bf74823beb296c39f5e01aa8a683aa50f061` |
| conversion/dry_run.log | `b910feab51679daf204a665b62b9c3cbc1bb7c5b2c3b799ba0a80c4968cf362a` |
| conversion/dry_run.exit (`exit=1`) | `cf205dbb8cea84897b488abcc281bf96698d5e94b1096b16657b4caba9082a22` |
| conversion/dlc_info.txt | `35b34a54bf08454f54f68985d65d05f8a1c47631cd43d61b741228f1cef44778` |
| conversion/dlc_info.csv | `b8b29df7432b9e34459e41a8e9b1d4a9b5656f3359b0f0f3cb216078090f56f0` |
| conversion/sha256.txt | `3a7e81fe8710a4f81cbf4302a19fc62f2811475eba2786619ed1da82167fa169` |
| conversion/host_dev/net_run.log | `4bc4979e23b8cba09deaed7e52f2b9491e35ced3c9da8779cc498e0f60281484` |
| conversion/host_dev/diag.txt | `db4e7c6d64c95d0a5f506639dc2f8d06d8bdcd9faa1cd7580d4df379e4a8cb41` |
| conversion/host_dev/net_run.help.txt | `9ab760a797c20c68b5a52ff4dabe72907076f1c5f3402d4ede6ed7f1d4013e72` |
| conversion/host_smoke/net_run.log | `2952b356441002b64a531ebcca8cdb8fe1895d3a10185f44b017751131bb7b70` |
| conversion/host_smoke/net_run.exit (`exit=0`) | `19eaf43821a7660ec323a87c8457bf74823beb296c39f5e01aa8a683aa50f061` |
| conversion/host_smoke/diag.txt | `65aaa4452d29ed0c81dcbc091158f2a81a0d8bd622d412901ff3b2d50c9ccdb8` |
| conversion/host_smoke/snpe_diag/SNPEDiag.log (moved out of `output/`; `SNPEDiag_0.log` identical) | `d83e6ed878ce26e196baf028d651199dc9f050aa928734df9f450f207a1b1d17` |
| conversion_host/ (env setup logs) | listed in `docs/env_conversion_host.md` |
