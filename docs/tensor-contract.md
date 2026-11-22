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
