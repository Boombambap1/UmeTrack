# Current reference architecture

1. SyncedImagePoseStream reads video, splits the four camera views, and supplies calibration, camera-to-world transforms, hand geometry, and reference poses for each frame.
2. HandTracker.gen_crop_cameras uses the reference hand pose to select up to two views per hand. _make_inputs warps perspective crops, normalizes pixels, mirrors right-hand input, and constructs tensors and hand/view ranges.
3. _image_backbone maps N x 1 x 96 x 96 images to N x 72 x 6 x 6 per-view features. This is our only replacement boundary.
4. Feature transforms incorporate camera geometry, and view fusion combines observations of each hand. A recurrent temporal module transforms/updates memory using prior/current camera geometry and per-hand state flags.
5. Known-skeleton features condition regression. Root decoding and coordinate transforms recover world-space hand pose; upstream uses meter-space model geometry internally and converts root translations to millimeters for output.
6. Forward kinematics/skinning produces the 21 evaluated landmarks per hand. The evaluator averages world-space Euclidean landmark errors in millimeters over valid hand-frame instances.

The local EncoderProvider supports normal, capture, and replay modes without changing downstream stages. ContextTracker attaches exact frame/crop IDs and model-input metadata. Each run/clip gets a fresh model and tracker; frames within a clip are never shuffled. Empty frames preserve sequence ordering and reset upstream hand-history flags. The wrapper is removed in a finally block after each run.

Profiling uses the original backbone plus timing hooks and explicitly separates decoding, crop selection, input preparation, encoder, remaining model, postprocessing, and landmark evaluation. It does not include capture/replay disk IO and does not measure live capture or rendering.

No hand detector, live camera driver, sensor synchronization, Monado/OpenXR adapter, or Qualcomm execution backend has been added. The Mac currently provides the complete reference pipeline; future board experiments will replace only encoder outputs while preserving host downstream evaluation.
