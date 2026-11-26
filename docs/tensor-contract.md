# Encoder capture/replay contract v1

The canonical bundle is artifacts/reference-bundle. Its inputs are the two sequence-start clips frozen in manifests/reference-clips.json: user_19 and user_23 recording_00, frames 0–49 each. There are 100 frames, 335 crops, 200 tracked hand-frame instances, and no preceding context frames. Both clips start with fresh model/tracker state.

## Tensors and identities

- Input: float32 NCHW (N, 1, 96, 96), already perspective-warped, normalized by 255, and mirrored for right hands using upstream conventions. Do not re-normalize or re-mirror it.
- Output: canonical float32 NCHW (N, 72, 6, 6), captured immediately after _image_backbone and before camera transforms, fusion, or temporal processing. Float32 files do not require an accelerator to use FP32 arithmetic internally.
- Frame identity: clip ID plus zero-padded source frame index. Crop identity additionally includes hand and source camera ID. Match by identity and explicit crop order, not directory enumeration.
- Metadata: camera intrinsics/extrinsics, hand sample ranges, memory index/use flags, hand IDs, and known skeleton arrays. Exact input and metadata digests must match on replay.
- Source indices are recording frame indices, not measured sensor timestamps. Temporal state follows the sequential upstream evaluator. The selected recordings have 30 fps video; no live timestamp synchronization has been implemented.

features/manifest.json contains provenance, per-record identities, metadata, tensor digests, file hashes, and filenames. Each NPZ contains inputs and features; load with allow_pickle=False. frames.json records crop IDs/counts even if a frame has no encoder call. Clip-specific pose NPZs preserve predictions, ground truth, and valid masks.

## Validated behavior

On the fixed 100-frame bundle, normal execution matches the saved full-baseline prefixes. Capture preserves outputs, and saved PC-feature replay through fresh models returns bit-identical poses, truth arrays, and masks. Maximum pose difference is 0 mm. The subset's 8.961398 mm error is its own reference; it does not replace the full-test 9.425145 mm result.

The provider rejects duplicate/missing/unused records, crop/order/metadata mismatches, invalid feature shapes/dtypes, nonfinite/corrupted tensors, checksum failures, unexpected feature files, and provenance mismatches. No missing-feature fallback executes the live encoder. Reversed manifest listing order passes because matching uses IDs. Tests also cover a no-hand frame followed by hand reappearance with upstream tracking history reset.

## Qualcomm handoff work remaining

Raw export, output import, and downstream replay are now implemented. The candidate NHWC batch-one input package is artifacts/qualcomm-inputs, with archive artifacts/qualcomm-inputs.zip. Both NCHW and NHWC raw-output paths passed exact Mac-reference round trips for all 335 crops and 100 frames. This validates file plumbing, not Qualcomm execution. B must still confirm actual DLC tensor names/layouts, batch-one execution, and matching SDK versions. See docs/qualcomm-handoff.md for the execution record and limitations.

Keep original Mac reference artifacts immutable. Write each backend/precision run into a new bundle with its own execution provenance and output hashes. Validate conversion on a few shared crops first, then replay all scored frames with identical metadata/history. GPU uses the float DLC under the selected experiment protocol; DSP quantization is B's lane. No Qualcomm results have been produced yet.

## Exported model and tensor names (Part B)

Merged from Part B's former `docs/tensor_contract.md`. Values come from `results/export/export_meta.json` and
`configs/experiment.yaml`.

| Item | Value |
| --- | --- |
| Module | `model._feature_extractor._image_backbone` (ResNet `resnet_layers_2352-f32` + 1x1 conv to 72 channels) |
| Upstream / checkpoint | `e263ec0844abf73457468a050a75853444d33d69` / sha256 `72eab0ea2c5c4518188ede4cf62e98f62eb1e4ef13628c0f20ad3fc56d1bf18d` |
| ONNX file | `artifacts/encoder.onnx` (not in Git; `scripts/export_encoder.py`), sha256 `1e86586510d928febb2db643caaef6e13fc48e4e5dd236424422cd89f34fa483`, 15,098,488 bytes; re-export is byte-identical |
| Exporter | `torch.onnx.export(..., dynamo=True)`; torch 2.15.0, onnxscript 0.7.2, onnx 1.23.2 |
| Input tensor | `crops`, float32, `[1, 1, 96, 96]`, NCHW, fixed batch 1 |
| Output tensor | `features`, float32, `[1, 72, 6, 6]`, NCHW |
| Opset / IR version | opset 17 (default domain only), IR version 10. The dynamo exporter builds at opset 18 and down-converts. QAIRT 2.33.2's converter targets onnx 1.16.1 (IR ≤ 10, opset ≤ 21), so no compatibility copy is needed; `snpe-onnx-to-dlc` accepted it on 2026-11-26 (see "Converted float DLC"). |
| Graph | `Add`, `Conv`, `MaxPool`, `Relu`; 67 nodes, no local functions. Eval-mode BatchNorm (28 layers) is folded into Conv by the exporter. |

The tensor names above are the ONNX names. The DLC's names, shapes and layouts below are read from
`snpe-dlc-info`, not assumed.

## Converted float DLC (Part B, D6)

Converted 2026-11-26 on the conversion host; details, verbatim warnings and raw-log hashes in
`results/conversion/summary.md`, environment in `docs/env_conversion_host.md`, handoff checklist in
`results/conversion/handoff.json`.

| Item | Value |
| --- | --- |
| File | `encoder_float.dlc` (not in Git), 15,145,523 bytes, sha256 `d0342d35a65f9de43967ffc1d2f6bfb2356e92b80b3dacd35bf57e5bf1572f7e` |
| Converter / SDK | `snpe-onnx-to-dlc`, QAIRT 2.33.2.250410; "DLC created with converter version: 2.33.2.250410134701_117956" (matches the RB5 runner observed Nov 21) |
| Source | ONNX sha256 `1e86586510d928febb2db643caaef6e13fc48e4e5dd236424422cd89f34fa483`; checkpoint sha256 `72eab0ea2c5c4518188ede4cf62e98f62eb1e4ef13628c0f20ad3fc56d1bf18d`; upstream `e263ec0844abf73457468a050a75853444d33d69` |
| Arguments | `--input_dim crops 1,1,96,96 --input_layout crops NCHW --out_name features --float_bitwidth 32 --model_version umetrack-encoder-onnx-1e865865` |
| Input tensor | `crops`, Float_32, `[1, 96, 96, 1]`, **NHWC**, fixed batch 1, no encoding (already uint8/255) |
| Output tensor | `features`, Float_32, `[1, 6, 6, 72]`, **NHWC**; raw file `Result_<i>/features.raw`, 10,368 bytes |
| Layout transform | The converter reorders to spatial-last (`perform_axes_to_spatial_first_order=True`). Input: NHWC bytes equal NCHW for C = 1. Output: transpose NHWC `[1,6,6,72]` -> NCHW `[1,72,6,6]` explicitly (`(0, 3, 1, 2)`). |
| Operator warnings | 25 x `WARNING_OP_VERSION_NOT_SUPPORTED: Operation Relu Not Supported. Expected operator version: [1, 6], instead got version: [14]`; onnxsim absent so simplification skipped; dry-run legacy checker flags Relu/MaxPool and crashes (SDK bug). Explained in the summary. |
| Development check on the conversion host | x86_64 `snpe-net-run` CPU, 8 Part B dev crops, NHWC -> NCHW: max_abs 2.265e-06, worst crop rel_l2 6.919e-07 (limit 1e-3 / 1e-4); backend_verified True from per-layer diag (all 67 CPU). Wrong-layout read: max_abs 1.89, so the mapping is confirmed. Not the RB5 experiment. |
| Pending | Steven's seven-crop smoke check; RB5 CPU run on Steven's canonical crops. |

## Board raw files and layout

- SNPE raw files are little-endian float32, one file per crop, listed in `input_list.txt`; outputs arrive as
  `Result_<i>/<output>.raw` for line i of that list.
- The candidate input layout is NHWC `[1, 96, 96, 1]` (for C = 1 its bytes equal NCHW `[1, 1, 96, 96]`); the
  output may be NHWC `[1, 6, 6, 72]`. NCHW and NHWC outputs have the same byte count, so the layout cannot be
  inferred from size: take it from `snpe-dlc-info` and transpose explicitly in code.
- Part B's reader (`scripts/read_snpe_outputs.py`) requires each output to be exactly the expected byte size,
  exactly `Result_0`..`Result_{n-1}` with no extras, matches results to crop IDs through the input list, and
  reports `numeric_match` and `backend_verified` (runtime parsed from the snpe-net-run log) separately.

## Crop sets

| Set | Owner | Crops | Source | Use |
| --- | --- | --- | --- | --- |
| Formal set | Part A | 335 | user_19 and user_23 `recording_00`, frames 0–49 (`manifests/reference-clips.json`); package `manifests/qualcomm-input-package.json` | Formal board-vs-reference comparison and pose replay (8.961398 mm paired subset) |
| Smoke set | Part A | 7 | Frame 0 of both frozen clips (`manifests/smoke-exchange.json`) | First conversion/layout check on the board |
| Development set | Part B | 60 | user_19 `recording_00`, frames 0–19 (`manifests/crops_d5.csv`, `purpose` = development) | **Development only**: ONNX-vs-PyTorch parity and Mac-side script tests |
| Calibration set | Part B | 184 | Training users 15 and 16, `real/separate_hand/training/.../recording_00`, every 10th frame (`manifests/calibration.csv`, `purpose` = calibration) | **INT8 calibration only**; these users appear in no testing split |

Formal comparisons use Part A's frozen crops, never regenerated ones. Part B's development and calibration
sets were regenerated in Part B's environment and are not a substitute:

> "Crop IDs match, but the decoding stacks produce different pixels from the same video. The first decoded
> frame differed by up to two intensity levels. The geometry shim also introduced small additional crop
> differences. Formal hardware comparisons must use the frozen canonical crop files, rather than independently
> regenerating crops on each machine."
>
> — Part A (Steven), audit of commit a21a692

The development set's frame 0 crop IDs match the smoke set's user_19 IDs; the pixels are not assumed to.

Enforcement: `scripts/canonical_crops.py` verifies Part A's archive, package manifest, input list and every raw
input against the sha256 values in `manifests/qualcomm-input-package.json` (formal) or
`manifests/smoke-exchange.json` (smoke) and extracts them unchanged. `scripts/read_snpe_outputs.py --purpose
formal|smoke` refuses to run unless that verified package re-checks; `--purpose dev` (default) is labelled
"development only". `scripts/write_snpe_inputs.py` only writes development inputs.

Part B's CSV manifests store the same identity as columns: `recording` (dataset-relative path),
`frame`, `hand_idx`, `cam_idx`, plus the `crop_id` string above
(`<user>-<recording>:<frame:06d>:hand<h>:camera<c>`, built by `scripts/common.py` `crop_id`). The `crop_id`
string drops the real/synthetic and split parts of the path, so keep the `recording` column alongside it.

## Numerical tolerances

**Proposed, pending agreement** between Part A and Part B (`configs/experiment.yaml`, `tolerances`). They must
be agreed before any formal comparison and are never loosened to make a result pass.

| Comparison | max_abs | rel_l2 per crop | Other |
| --- | --- | --- | --- |
| ONNX vs PyTorch | 1.0e-4 | 1.0e-5 | |
| Board float (CPU/GPU) vs ONNX | 1.0e-3 | 1.0e-4 | |
| Pose error change vs the paired Mac subset | | | absolute MPJPE change < 0.1 mm against 8.961398 mm |

`rel_l2 = ||candidate − reference|| / ||reference||`. Part A's conversion template also asks for atol/rtol,
MAE, RMSE and max absolute error; Part B's summaries report max_abs, mean_abs (MAE), RMSE and rel_l2, so both
can be filled. DSP/INT8 allowances are not yet proposed; the ORT INT8 preview
(`results/int8_preview/summary.md`) is characterisation only, not SNPE/DSP and not pass/fail.

Measured so far: ONNX vs PyTorch on the 60-crop development set, worst max_abs 3.695e-6 and worst per-crop
rel_l2 9.078e-7 (`results/onnx_parity/summary.md`). No board result exists yet.
