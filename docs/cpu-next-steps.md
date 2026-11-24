# Part A preparation before the float DLC handoff

Updated November 23, 2026. Steven owns Part A; Ali owns float conversion and the DSP lane. This is the implementation runbook for the independent preparation stage. The Notion task remains [Hand tracking research — Steven and Ali](https://app.notion.com/p/3e6bc072402f80f984a1f73bc57c50dc).

## Work completed without the handoff

| Step | Evidence | Status |
| --- | --- | --- |
| Reconcile progress against the task and recorded A/B split | M1 reference complete; M2 export validated, conversion pending; M3 encoder inference pending | Reviewed November 23 |
| Verify the frozen seven-crop and 335-crop archives | Hash-pinned manifests, input-list order, crop IDs, individual raw hashes | Passed November 23 |
| Compare the full raw package directly against canonical reference arrays | All 335 crop bytes match | Passed November 23 |
| Check the recovered ONNX identity | SHA256 `1e86586510d928febb2db643caaef6e13fc48e4e5dd236424422cd89f34fa483` | Matched November 23 |
| Exercise export, raw output import and downstream replay | Both NCHW and NHWC; 335 crops and 100 frames per layout; zero feature/pose difference | Passed November 23 |
| Preserve the reference | All 105 reference files have unchanged before/after hashes | Passed November 23 |
| Make earlier canonical ONNX evidence reviewable | [Small ONNX summary](../results/canonical-onnx-validation.json), derived from preserved November 21 audit | Documented November 23 |

Fresh results and exact commands: [pre-handoff readiness](../results/pre-handoff-readiness.json). The local verification script, expanded packages, reference hash snapshot and round-trip outputs are preserved under ignored `artifacts/pre-handoff-2026-11-23/`. Prior artifacts were not overwritten.

The round-trip check copies saved Mac encoder features into raw result files, then replays them through fresh host tracker instances. It checks file exchange, layout handling and temporal replay. It does not simulate the RB5 or test Qualcomm software. Existing November 23 tests (16 Part A passes; 77 passes and one expected skip under the Part B environment) were already recorded in the repository assessment and were not rerun here; runtime source is unchanged.

## Frozen experiment definition

Use the first 50 frames of `user_19/recording_00` and `user_23/recording_00`, resetting model/tracker state at each clip boundary. There are 100 scored frames, no preceding context frames, 335 crops and 200 valid hand-frame instances. Crop counts are 15 frames with two crops, 35 with three, and 50 with four.

The paired reference is **8.961397993859494 mm MPJPE**. The full-test **9.42514460875624 mm** result uses a different sample population. CPU acceptance uses absolute MPJPE change below **0.1 mm** against the paired reference, with identical ground truth and masks. Also report direct joint displacement and feature errors; matching aggregate MPJPE alone is insufficient evidence about every joint.

Use the existing canonical raw bytes. The development environments decode different pixels from the same video, so regenerating crops independently can change the comparison. NHWC `(1,96,96,1)` is the current candidate input representation; canonical output is NCHW `(1,72,6,6)`. Actual DLC names/layouts remain unknown. A matching byte count does not establish layout.

The November 21 Mac ONNX experiment used all 335 frozen crops and returned **8.9613971812326 mm** after downstream replay, an absolute error change of **8.126268937758141e-7 mm**. That is export evidence, not an RB5 CPU result. The new summary identifies the original evidence hashes and clearly dates this historical result.

## Recheck the local exchange if inputs or replay code change

Run from the repository root using the existing Part A environment and a new output directory:

```sh
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=2 .venv/bin/python scripts/check_raw_exchange.py --output artifacts/raw-exchange-pre-handoff-new
```

Do not rerun full dataset evaluation or download additional training data merely to wait for conversion. The agreed evaluation data, source/checkpoint identity and local plumbing are already available.

## Stop: the next step requires the float DLC handoff

Before model-specific board preparation, obtain and verify:

1. Float encoder DLC and SHA256; verify the downloaded file locally.
2. Conversion SDK release/build and converter identity. The last observed board runner is `2.33.2.250410134701_117956`; match the intended runtime or explicitly review a change.
3. Actual input/output names, shapes, batch size, layouts and dtypes; confirm normalization is already applied and establish the exact raw output filename.
4. Exact conversion command and full log, plus the source ONNX/checkpoint identities.
5. Runtime setup/location usable by the benchmark account. The last recorded shared-account access could not read the staged SDK inside another account's private home.

`examples/conversion-handoff.template.json` is a checklist, not a completed record. Leave unobserved model fields unset. Proposed feature tolerances in `configs/experiment.yaml` remain unchanged and still require agreement; do not loosen them after seeing results.

At this gate, stop rather than finalize tensor mapping, alter SDK permissions, install a different runtime, or fabricate backend evidence. Board access/runtime reports remain historical November 21 observations. No new board contact or configuration change was performed during this preparation.

After the handoff is verified, use [the A/B runbook](part-a-part-b-handoff.md): seven-crop CPU smoke test first, then the 335-crop CPU run and paired replay, followed by separate warm timing. Inspect actual runtime/profiling evidence before choosing a maintained timing collection implementation. Keep startup, file I/O and Tailscale transfer outside encoder compute timing. DSP repair and INT8 work are not prerequisites for CPU testing; GPU support remains unverified.
