# Full Mac CPU reference baseline

Completed November 19, 2026. Protocol: known skeleton / real / separate_hand / testing.

- All 143 recordings across five users completed, with no failures.
- 63,798 source frames and 102,885 valid hand-frame instances.
- Mean joint position error: **9.42514460875624 mm**. This rounds to the upstream README's 9.4 mm reference and lies in the task's 9.3–9.5 mm reproduction band.
- Mean joint acceleration: 3.92336136493353 in upstream's discrete-frame metric (not automatically physical mm/s²).
- Upstream AUC: 81.1501044855907.
- Valid tracking slots: 102,885 / 127,596 = 80.6334%. The denominator counts both hand slots in every frame, including untracked/unlabeled hands; this percentage is not detector recall.

The evaluator used ground-truth-assisted crop selection and known hand geometry. This is a pose-estimation reference, not a live hand-detection or headset-readiness test.

## Per-user results

| User | Recordings | Frames | Valid hand-frame instances | MPJPE (mm) |
| --- | --- | --- | --- | --- |
| user_19 | 29 | 12706 | 20414 | 9.896242 |
| user_23 | 29 | 13079 | 22523 | 9.529450 |
| user_29 | 30 | 13530 | 21053 | 8.781470 |
| user_34 | 25 | 11272 | 18243 | 8.628778 |
| user_39 | 30 | 13211 | 20652 | 10.205366 |

Overall MPJPE weights valid hand-frame instances, not users or recordings equally.

## Reproduce

Run from the repository root with the installed Mac environment, pinned upstream source, dataset Git metadata, and manifests/separate-hand-test.json:

```sh
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=2 .venv/bin/python -u scripts/run_full_baseline.py --output artifacts/runs/separate-hand-new
.venv/bin/python scripts/validate_baseline.py artifacts/runs/separate-hand-new
```

Choose a new run directory. The runner refuses to overwrite prior results, reuses only hash-verified input videos, and checks extracted labels against the pinned Git blobs. Each recording gets a fresh model/tracker. Four processes with two PyTorch threads each are used by default; interop threads are limited to one. This concurrency is for accuracy evaluation throughput, not a latency benchmark.

The initial attempt without explicit native-library limits was interrupted for excessive CPU contention and preserved in artifacts/runs/separate-hand-baseline. Only artifacts/runs/separate-hand-bounded contributes to the reported result. Its 901.2-second wall time includes remaining downloads, model initialization, decoding, processing and IO; it must not be labeled encoder latency or RB5 runtime.

## Evidence

- results/separate-hand-baseline.json: portable summary, provenance, and per-user results.
- manifests/separate-hand-test.json: complete pinned input inventory, LFS sizes and SHA256 hashes.
- artifacts/runs/separate-hand-bounded/config.json: exact run configuration.
- artifacts/runs/separate-hand-bounded/sequences/: generated predictions, masks, and per-sequence records.
- artifacts/runs/separate-hand-bounded/validation.txt: upstream aggregate and saved-prediction cross-check.
- artifacts/full-baseline-bounded.log: process/download log.

Source/checkpoint hashes and environment snapshot hash are in the summary. Large inputs, predictions, logs, and environments remain ignored. Nothing was committed or pushed, and no RB5 commands were executed.

Reference: https://github.com/facebookresearch/UmeTrack#results

Next milestone: profile the serial Mac pipeline, freeze the canonical reference clip bundle, and implement PC-feature capture/replay before substituting board features.
