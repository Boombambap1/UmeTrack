# UmeTrack RB5 progress report

Report date: November 28, 2026. Steven owns Part A: the Mac reference, frozen evaluation inputs, replay, RB5 CPU and GPU experiments. Ali owns Part B: export/conversion, SDK handoff and DSP work.

## Current outcome

The pretrained UmeTrack image encoder runs correctly on the RB5 CPU. Its features reproduce the paired Mac hand-pose result with an MPJPE change of only **0.000000737463 mm**, comfortably below the agreed 0.1 mm CPU limit. Dedicated batch-one CPU timing is **20.779 ms/crop median and 27.088 ms/crop p95**, with profiling and thermal-observation limitations described below.

GPU work is blocked before model execution: the current board environment exposes an OpenCL loader but **zero OpenCL platforms**. DSP board execution remains unverified in the evidence available here. The overall CPU/GPU/DSP comparison and live headset integration are unfinished.

| Plan stage | Current status |
| --- | --- |
| Host reference and feature replay | Complete for the agreed evaluation |
| Float encoder export/conversion and CPU handoff | Validated, including actual RB5 CPU execution |
| RB5 CPU correctness and dedicated timing | Complete with recorded timing limitations |
| RB5 GPU | Readiness blocked; seven-crop smoke test not started |
| RB5 DSP | Partner workstream; no verified board result in these records |
| Consolidated comparison and headset suitability | Partial; accelerator evidence and representative integration remain |

## How the experiment operates

The Mac decodes the public recordings and makes the fixed hand/view crops. Either the Mac encoder or the converted encoder on the RB5 produces feature tensors. The Mac then runs the remaining camera-aware fusion, temporal tracking and pose decoding on those features. Paired replay measures whether changing the encoder changes the final hand poses.

Tailscale provides connectivity. SSH runs commands and transfers files. Neither performs model conversion or simulates the RB5. Only the encoder runs on the board in this experiment; the full tracker remains on the Mac.

## Completed work

1. **Established the Mac reference environment.** Restored pinned UmeTrack/data sources and checkpoint, built real PyTorch3D for the reference environment, and kept the separate Part B environment explicit. Source/dependency pins and setup commands are recorded in repository manifests and docs.
2. **Completed the public accuracy evaluation.** Evaluated all 143 recordings in the real separate-hand known-skeleton test split: 63,798 frames and 102,885 valid tracked hand-frame instances. MPJPE was 9.425145 mm, within the task's 9.3–9.5 mm reproduction band. Coverage was 80.6334% of all 127,596 two-hand frame slots; this denominator is not visible-hand detection recall.
3. **Froze the paired evaluation bundle.** Selected the first 50 frames from each of two fixed clips: 100 frames, 335 crops and 200 valid hand-frame instances. There are 15 frames with two crops, 35 with three, and 50 with four. Crop identity, ordering, preprocessing, camera metadata, hand geometry and temporal reset policy are preserved.
4. **Validated capture and replay.** Original, capture and replay paths produced exactly identical reference poses. Fresh tracker/model state is used per clip and mode. NCHW and NHWC raw-file round trips also reproduced the saved features and poses exactly using Mac fixtures.
5. **Profiled the host.** The isolated Mac batch-one encoder reference is 3.944 ms/crop median and 4.147 ms/crop p95. A separate serial pipeline profile measured 29.960 ms/frame median total and 12.232 ms/frame median encoder execution in frame batches. The encoder represented 41.10% of aggregate measured pipeline time in that host workload; this is not an RB5 fraction.
6. **Validated ONNX and the float DLC.** The ONNX encoder passed on all 335 frozen crops. Ali converted it on an Ubuntu 22.04 x86_64 host using the matching QAIRT/SNPE build and validated development crops plus the seven canonical smoke crops on the conversion-host CPU. The resulting DLC and account-readable CPU runtime were subsequently hash-checked and exercised on the RB5.
7. **Completed actual RB5 CPU correctness.** Seven smoke crops and all 335 formal crops passed the declared feature limits. All 67 layers were positively verified on CPU. The formal outputs were replayed through the same Mac downstream tracker with matching masks and ground truth.
8. **Completed dedicated CPU timing.** One loaded model processed 30 unscored warm-ups and three ordered passes through 335 crops. Individual SDK execution records supplied 1,005 measured samples; averaged diagnostic rows and process wall time were not used as substitutes.
9. **Investigated GPU readiness.** Verified the current model/runtime identity, dependencies and library resolution, inspected accessible GPU/OpenCL configuration, and made one bounded platform-enumeration probe. Stopped at the confirmed infrastructure blocker without launching the encoder on GPU.

## Accuracy results

| Condition | MPJPE mm | Change from paired Mac result mm | Evidence |
| --- | --- | --- | --- |
| Mac full separate-hand split | 9.425144609 | Separate evaluation denominator | [Full baseline](../results/separate-hand-baseline.json) |
| Mac paired 100-frame subset | 8.961397994 | Reference | [Reference replay](../results/reference-replay.json) |
| RB5 CPU encoder plus Mac downstream tracker | 8.961397256 | 0.000000737463 | [CPU correctness](../results/rb5-cpu-correctness-2026-11-28.json) |
| RB5 GPU / DSP | Not measured | Not measured | Readiness/partner work remains |

For the formal CPU feature comparison, worst max_abs was **2.38419e-6**, against a limit of 1e-3. Worst per-crop rel_l2 was **6.01666e-7**, against a limit of 1e-4. Both limits applied to every crop. Mean direct joint displacement was 0.00005643 mm and maximum displacement was 0.00017925 mm. These displacement values are diagnostics, separate from MPJPE change.

The converted output is float32 NHWC `[1,6,6,72]`, explicitly transposed with `(0,3,1,2)` to canonical NCHW `[1,72,6,6]`. Float32 files alone do not establish an accelerator's internal arithmetic mode.

## Dedicated RB5 CPU timing

| Statistic | ms per crop |
| --- | ---: |
| Median | 20.779 |
| p95 | 27.088 |
| Mean | 21.493 |
| Minimum | 18.736 |
| Maximum | 64.253 |

| Measured pass | Samples | Median ms/crop | p95 ms/crop |
| --- | ---: | ---: | ---: |
| 1 | 335 | 19.421 | 26.946 |
| 2 | 335 | 19.827 | 26.619 |
| 3 | 335 | 21.242 | 30.572 |

The batch size was one. SNPE CSV format 2 retained a `TOTAL_INFERENCE_TIME` event for each call; statistics use the 1,005 measured events after excluding the 30 warm-ups. Model loading, initialization, file reading/writing, transfer and numerical comparison are outside this SDK timer. All 1,035 outputs were checked after timing, and all 69,345 per-layer records reported CPU.

The run used `system_settings` with detailed profiling. The observed governor was `performance`, with ten total process threads and affinity masks permitting CPUs 0–7; the compute-worker count was not separately established. Inputs were pre-read into the page cache, while the CLI still performed its own file reads outside the timer. No declared contention stop rule triggered.

Detailed profiling adds instrumentation overhead. Pass medians rose 9.38% from first to third. Observed CPU/cluster sensors spanned approximately 47.3–69.8 C, but buffered input announcements prevented exact before/after temperature attribution to each pass. These are warm, instrumented execution measurements, not proof of stationary or sustained thermally stable performance. Thermal throttling was not established as the cause of variation.

The saved Mac timing is useful context, but hardware, runtime, profiling, thread configuration, power and buffer handling differ. A controlled hardware-only speed ratio or full-tracker speedup cannot be inferred. Grouped crop durations, if used, are derived serial encoder costs rather than measured headset latency.

## GPU blocker and DSP status

The fresh GPU probe found `/usr/lib/libOpenCL.so.1.0.0`, owned by `ocl-icd 2.3.5-1`. The inspected vendor directories were absent, no OpenCL override was configured, and `clGetPlatformIDs(0, NULL, &count)` returned **-1001 / CL_PLATFORM_NOT_FOUND_KHR, with zero platforms**. The loader exists, but it currently exposes no usable implementation to the benchmark account. [Khronos API definition](https://registry.khronos.org/OpenCL/specs/unified/refpages/man/html/clGetPlatformIDs.html).

The render node was readable/writable, but its sysfs device driver resolved to `msm_dpu`, a display-subsystem driver. That observation does not establish an accessible Adreno compute device. The library search was bounded to standard accessible locations; it does not prove a usable implementation is absent everywhere on the board.

No SNPE GPU runtime validator was rerun after this blocker, and no GPU model inference, output comparison, precision verification or latency test occurred. The historical November 21 validator SIGSEGV, exit 139, remains unexplained; the current missing-platform finding does not establish its cause.

DSP remains Ali's workstream. Historical November 21 diagnostics reported missing `libadsprpc.so` and failed validation. That is historical evidence, not a fresh claim about today's DSP state. An ONNX INT8 preview is not proof of DSP execution.

## Reproducibility and validation

- Experiments used the immutable reviewed PR #1 snapshot `2fe815709db28b1f71fa8b05f705c1cd2c83f578`, while the working branch remained `main` at `f2f0ac658278af7deb0dd528f4ddf068d7f1ee1e`.
- QAIRT/SNPE runtime build: `2.33.2.250410134701_117956`, matched to conversion.
- Float DLC SHA256: `d0342d35a65f9de43967ffc1d2f6bfb2356e92b80b3dacd35bf57e5bf1572f7e`.
- Source, crop and reference hashes were checked; all 105 original reference files remained unchanged. Canonical input lists and earlier experiments were preserved.
- Original diagnostic aliases were preserved. Only the known `SNPEDiag.log -> SNPEDiag_0.log` alias was materialized into a byte-identical analysis copy for the strict reader; arbitrary links were not broadly accepted.
- Fresh checks on the exact PR snapshot: **104 pytest tests passed, one skipped** because Part B's environment lacks real PyTorch3D. **All 16 Part A unittest tests passed** with real PyTorch3D, including that skipped case. These counts overlap and should not be added.
- During the benchmark stages, no board packages, drivers, permissions, services, firmware, governors or image were changed, no unrelated jobs were terminated, and no Git or external-service publication actions were performed. Subsequent repository publication is recorded below.

## Repository publication status

[PR #1](https://github.com/Boombambap1/UmeTrack/pull/1) was reviewed, approved and merged on November 28, 2026. The review was anchored to tested head `2fe815709db28b1f71fa8b05f705c1cd2c83f578`; merge commit is `9c1f210921b4366f1f126db8f748fb2c2bba8437`. The code validated by the CPU experiments is now on `main`. The GPU infrastructure blocker does not invalidate the conversion/CPU result.

The local checkout was fast-forwarded to that merge without losing any publication file. This follow-up uses branch `codex/rb5-cpu-results-gpu-readiness` for the README refresh, this report and the three November 28 summaries: CPU correctness, CPU timing and GPU readiness. The results update is intended for a separate pull request and teammate review.

SDK binaries, DLC/ONNX files, datasets, bulk raw outputs, credentials, AI plans and agent guidance remain outside this publication. Existing ignore rules cover these categories. Any needed evidence bundle should be shared separately with hashes and controlled retrieval instructions; the JSON summaries refer to ignored local artifact directories.

The maintained pipeline/import code is in the repository, while the board orchestration and timing-analysis scripts for these runs are preserved under ignored experiment directories. Publishing the summaries alone does not make those exact runs reproducible from a clean clone without the artifact handoff. A portable runner/evidence handoff is useful follow-up work; experiment scripts containing machine-specific connection paths should not be copied blindly into public Git history.

The earlier experiment JSON files retain the original code/branch and no-publication observations from when each experiment ran. Those historical records have not been rewritten to imply that the experiments ran on the later merge commit.

## Remaining work and ownership

1. **Vincent:** identify/provide a GPU OpenCL implementation and compute-device path compatible with the board image, demonstrating a nonzero platform count and an accessible GPU device under the benchmark account. Any infrastructure changes require separate authorization.
2. **Ali:** confirm the matching SNPE build's GPU dependency requirements and supply missing accessible runtime artifacts if necessary; continue DSP readiness, INT8 calibration/conversion and actual-backend evidence independently.
3. **Steven:** once GPU platform readiness is restored, run a bounded SNPE GPU check, then the seven-crop float smoke test. Apply max_abs <= 1e-3 and per-crop rel_l2 <= 1e-4, verify actual per-layer runtime and reject fallback as a pure GPU result.
4. **Steven:** after the smoke gate passes, run all 335 crops and paired downstream replay. Record precision, coverage and pose differences. Keep later GPU timing separate and improve synchronized thermal/workload observations for comparison with CPU.
5. **Both:** consolidate the final CPU/GPU/DSP comparison and commands on the Notion task when requested. Agree accelerator acceptance and representative headset latency/accuracy targets before claiming deployment suitability.

The evaluation uses known hand geometry and ground-truth-assisted crops. Live cameras, autonomous hand detection/recovery, full tracker deployment, OpenXR interaction, sustained operation and commercial-use clearance remain outside the demonstrated result. The existing [license note](license.md) records the upstream non-commercial terms and open questions.

## Evidence locations

Small summaries suitable for repository review:

- [Full Mac baseline](../results/separate-hand-baseline.json), [paired reference/replay](../results/reference-replay.json), [serial profile](../results/mac-serial-profile.json), and [batch-one timing](../results/mac-encoder-batch-one.json).
- [Canonical ONNX validation](../results/canonical-onnx-validation.json) and [raw exchange checks](../results/raw-exchange-check.json).
- [Actual RB5 CPU correctness](../results/rb5-cpu-correctness-2026-11-28.json), [dedicated CPU timing](../results/rb5-cpu-timing-2026-11-28.json), and [GPU readiness blocker](../results/rb5-gpu-readiness-2026-11-28.json).

Ignored local evidence directories (not supplied by a normal Git clone):

| Experiment | Directory |
| --- | --- |
| CPU correctness, replay, immutable code | `artifacts/rb5-cpu-correctness-2026-11-28-61515952/` |
| CPU timing, 1,005 samples, protocol and commands | `artifacts/rb5-cpu-timing-2026-11-28-53a765da/` |
| GPU library inventory and bounded platform probe | `artifacts/rb5-gpu-readiness-2026-11-28-f7147f6f/` |
| Report preparation and validation record | `artifacts/status-report-2026-11-28-a4a11e6f/` |

Ali's conversion/setup records are in PR #1. Earlier dated setup/readiness documents remain historical; the November 28 results above establish the current CPU/GPU findings.
