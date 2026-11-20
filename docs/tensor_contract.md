# UmeTrack image encoder — tensor contract (D1)

The interface between the host pipeline and the encoder that Part B exports, converts and runs on the RB5.
Part A (PC reference, replay adapter) and Part B (conversion, board runs) both build on this contract.

Sources: `CLAUDE.md`, `configs/experiment.yaml`, `results/export/export_meta.json`, `manifests/hashes.txt`,
and upstream code at the pinned commit.

## Provenance

| Item | Value |
| --- | --- |
| Upstream repo | https://github.com/facebookresearch/UmeTrack |
| Upstream commit | `e263ec0844abf73457468a050a75853444d33d69` (submodule `third_party/UmeTrack`) |
| Checkpoint | `third_party/UmeTrack/pretrained_models/pretrained_weights.torch` |
| Checkpoint sha256 | `72eab0ea2c5c4518188ede4cf62e98f62eb1e4ef13628c0f20ad3fc56d1bf18d` |
| Module | `model._feature_extractor._image_backbone` (ResNet `resnet_layers_2352-f32` + 1x1 conv to 72 channels) |
| ONNX file | `artifacts/encoder.onnx` (not in git; regenerate with `scripts/export_encoder.py`) |
| ONNX sha256 | `1e86586510d928febb2db643caaef6e13fc48e4e5dd236424422cd89f34fa483` (15,098,488 bytes; re-export is byte-identical) |
| Exporter | `torch.onnx.export(..., dynamo=True)` — torch 2.15.0, onnxscript 0.7.2, onnx 1.23.2 |
| Opset | 17 (default domain). The dynamo exporter builds at opset 18 and down-converts with the onnx version converter. |
| ONNX IR version | 10 (confirm QAIRT 2.33 accepts it in D6) |
| Graph changes at export | Eval-mode BatchNorm (28 layers) folded into Conv by the exporter's optimizer |

## Input

| Field | Value |
| --- | --- |
| Name | `crops` |
| Shape | `[1, 1, 96, 96]` — fixed batch 1 |
| Dtype | float32 |
| Layout | NCHW |
| Value range | [0, 1] |
| Preprocessing | `uint8 crop / 255.0` → float32, then add the channel axis |

Where it comes from in upstream:

- `lib/tracker/tracker.py` `HandTracker._make_inputs` (lines 210–211):
  `crop_image = _warp_image(view_data.camera, crop_camera, view_data.image)` then
  `left_images.append(crop_image.astype(np.float32) / 255.0)`.
- `_warp_image` (`tracker.py:54`) remaps the monochrome fisheye view into a 96x96 pinhole crop camera
  (`cv2.remap`, bilinear) and returns uint8.
- Crop cameras come from `HandTracker.gen_crop_cameras` → `lib/tracker/perspective_crop.py`
  `gen_crop_cameras_from_pose` (`HandTrackerOpts` defaults: 63 crop points, `hand_ratio_in_crop` 0.95,
  `min_required_vis_landmarks` 19, at most 2 views per hand, camera indices sorted).
- Right-hand crops are mirrored horizontally (`mirror_right_hand=True`), so the encoder always sees a left-hand-like image.
- The source image is monochrome: `lib/tracker/video_pose_data.py` takes channel 0 of the decoded frame.
- The model adds the channel axis itself: `left_images.unsqueeze(1)` in
  `lib/models/umetrack_model.py` `_forward_feature_extractor` (line 128).
- Upstream runs all crops of a frame as one batch (1–4 crops: up to 2 hands × up to 2 views).
  The exported model has a fixed batch of 1, so crops are run one at a time and stacked in manifest order.

## Output

| Field | Value |
| --- | --- |
| Name | `features` |
| Shape | `[1, 72, 6, 6]` |
| Dtype | float32 |
| Layout | NCHW |
| Observed range (D5, 60 real crops) | min −0.5602, max 1.6906, mean 0.0851, std 0.1994 |

What consumes it downstream (unchanged, on the host): `UmeTrackModel._forward_feature_extractor`
(`lib/models/umetrack_model.py`).

- Hand seen in one view: `FeatureExtractor.compute_singlev_features` applies FTL
  (`model_utils.apply_ftl_to_feature_maps`) using the crop camera intrinsics.
- Hand seen in two views: `FeatureExtractor.compute_multiv_features` applies FTL into a canonical view,
  fuses the two views, and applies FTL back to camera 0 (uses intrinsics and extrinsics).
- The fused features then go to the temporal module, skeleton conditioning and pose regressors.

The replay adapter (Part A) substitutes features at exactly this boundary: the `_image_backbone` output,
before FTL and view fusion, with the same order, dtype and shape.

## ONNX ops

`Add`, `Conv`, `MaxPool`, `Relu` — 67 nodes, no local functions.

## Conversion-specific changes

_To be filled in D6 (e.g. NHWC raw layout for SNPE input/output files, tensor renames, any transposes)._

## Sample identity

Part A and Part B match crops by the columns of the crop manifest (`manifests/crops_d5.csv` for D5):

| Column | Meaning |
| --- | --- |
| `row` | Position in the crops array (`artifacts/crops/crops_uint8.npy`) and in every feature array derived from it |
| `recording` | Recording path relative to the dataset root, without extension (e.g. `raw_data/real/separate_hand/testing/user_19/recording_00`) |
| `frame` | 0-based frame index in the recording's mp4 / json |
| `hand_idx` | 0 = left, 1 = right (`lib/common/hand.py` `LEFT_HAND_INDEX` / `RIGHT_HAND_INDEX`) |
| `cam_idx` | Index into the recording json's `cameras` list (0–3) |

- The identity key is (`recording`, `frame`, `hand_idx`, `cam_idx`); `row` is only a position.
- Within a frame, crops are ordered by hand, then by camera ascending — the same order as `_make_inputs`.
- `make_crops.py` checks every frame against upstream `_make_inputs`: `left_images == crops / 255` exactly.

## Tolerances

**Proposed, to agree with Steven** before any formal comparison (`configs/experiment.yaml`, `tolerances`).
Never loosen one to make a result pass.

| Comparison | max_abs | rel_l2 | Other |
| --- | --- | --- | --- |
| ONNX vs PyTorch (`onnx_vs_torch`) | 1.0e-4 | 1.0e-5 | |
| Board CPU vs ONNX (`board_cpu_vs_onnx`) | 1.0e-3 | 1.0e-4 | |
| Pose error change vs paired PC baseline | | | 0.1 mm (`pose_error_change_mm`) |

`rel_l2 = ||candidate − reference|| / ||reference||`, applied per crop.

## Notes

- The D5 crops were placed using a `so3_exp_map` stand-in for pytorch3d (`scripts/pytorch3d_shim.py`),
  because pytorch3d is not installed on the Mac. It matches `scipy` `Rotation.from_rotvec` within 1e-6
  (`tests/test_so3_standin.py`). Part A's full evaluation uses real pytorch3d.
- `VideoStream.__iter__` is patched at runtime (`scripts/upstream_compat.py`) because its fps log line fails
  on PyAV ≥ 19; frame decoding is unchanged. `third_party/` is never edited.
