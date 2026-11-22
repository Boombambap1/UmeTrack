# Part A capture/replay and serial profile

Completed on the Mac CPU on November 19, 2026. Upstream source was not edited; the private encoder boundary is wrapped by our local adapter and restored at the end of each run.

## Replay evidence

- Fixed 100 source frames from two users, starting at sequence start.
- 335 crops: 15 frames with two crops, 35 with three, and 50 with four.
- 200 valid hand-frame instances, subset MPJPE 8.961397993859494 mm.
- Normal vs capture: exact match.
- Normal vs saved full-baseline prefixes: exact match.
- Capture vs replay using fresh model/tracker per clip: exact match, 0 mm maximum pose difference.
- Six test groups pass for single/multiple views, input/order/metadata rejection, corrupted/invalid/unused features, file checksums/provenance/listing order, duplicate calls, and no-hand/reappearance state handling.

The real scored clips contain two hands in every selected frame and do not establish broad occlusion or reacquisition quality. No-hand/reappearance behavior was checked separately. This is a conversion-equivalence bundle, not a new accuracy benchmark.

## Serial timing

Three serial passes, two CPU threads, native-library limits below; first 10 frames of each 50-frame clip excluded from timing. This gives 80 unique measured frames, repeated three times (240 frame instances). No feature-capture serialization runs during profiling. Model/label construction is outside timers; video decode, assisted cropping, and evaluation landmark construction are included. Timing hooks add overhead.

| Stage | Median ms/frame | p95 ms/frame |
| --- | --- | --- |
| Video decode and frame/label assembly | 6.490 | 8.169 |
| Assisted crop selection | 3.854 | 4.792 |
| Warp, normalization and input preparation | 4.425 | 5.627 |
| Image encoder, actual frame batch | 12.232 | 14.143 |
| Remaining model and CPU output transfer | 1.238 | 1.495 |
| Pose postprocessing | 0.041 | 0.056 |
| Predicted and ground-truth landmark construction | 1.417 | 1.702 |
| Measured total | 29.960 | 33.896 |

Stage medians do not necessarily sum to the total median. Fractions use sums of measured stage times: encoder 41.098% of the offline pipeline and 90.641% of model time. Average encoder compute per crop within the actual batches was 3.717 ms; this is not an independently measured batch-one latency. The sampled timing frames contain 2–4 crops, and are not a representative full-test workload.

These are Mac CPU timings, not RB5 measurements, capture-to-display latency, or a promise of real-time live tracking. Accelerating only the encoder cannot remove decode/cropping/evaluation costs. Amdahl's ideal upper bound using this measured pipeline fraction is about 1.70x even with zero encoder cost, before offload overhead, and applies only to this measured workload.

## Reproduce

From the project root, use new output directories:

```sh
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=2 .venv/bin/python scripts/build_reference_bundle.py --output artifacts/reference-bundle-new
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=2 .venv/bin/python scripts/profile_reference.py --output artifacts/runs/serial-profile-new
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=2 .venv/bin/python -m unittest discover -s tests -v
```

Bundle creation checks the recording/label hashes and the existing full-baseline prefixes. tests require the pinned checkpoint; the hand-reappearance test additionally uses the earlier smoke recording. Missing dependencies/data fail rather than silently substituting inputs. Only sequence-start clips are currently supported; nonzero start_frame is explicitly rejected to avoid omitting temporal history.

Code: src/umetrack_rb5/encoder_provider.py and reference.py, scripts/build_reference_bundle.py and profile_reference.py. Portable results: results/reference-replay.json and mac-serial-profile.json. Detailed frame timings and canonical arrays remain under ignored artifacts/. Final test log: artifacts/replay-tests-final.log.

Raw SDK input export, strict output import, and imported-feature replay were subsequently implemented and validated using Mac fixtures; see docs/qualcomm-handoff.md. Next: confirm actual model metadata and SDK builds, verify board readiness, exchange a few real encoder crops with B, then run actual CPU/GPU comparisons on RB5. Isolated Mac batch-one timing was subsequently measured November 21; see below. Nothing was committed/pushed during those initial experiments; the shared implementation is now prepared for publication. Subsequent board readiness diagnostics and their temporary outputs are recorded separately in docs/board-readiness.md; no board configuration was changed.

## Isolated batch-one encoder — November 21

scripts/benchmark_encoder.py measured 335 preloaded real crops, three passes (1,005 calls), two PyTorch threads, one interop thread, eval/no_grad, and 30 additional warm-up calls after validating all crops. Median was **3.944 ms/crop** and p95 **4.147 ms/crop**. All batch-one feature outputs matched the saved frame-batch reference exactly (MAE/RMSE/max absolute error 0).

This is a distinct experiment from the November 19 frame-batch profile. Its environment snapshot reports macOS 27.2.1, and the Mac was on battery power. Thermal state and other workloads were not measured. Do not attribute the difference from 3.717 average ms/crop solely to batch size. Decode, preprocessing, transfer, model startup, output serialization, and downstream pose computation are excluded. Raw call timings, per-pass summaries, source hashes, and power observations are under artifacts/runs/mac-encoder-batch-one-2026-11-21; the small shareable summary is results/mac-encoder-batch-one.json. Reproduce using the new command in README.md and a fresh output directory.
