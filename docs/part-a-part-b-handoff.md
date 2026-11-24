# Part A / Part B handoff to RB5 testing

Updated November 23, 2026 against the [Notion task](https://app.notion.com/p/3e6bc072402f80f984a1f73bc57c50dc), recorded A/B split, repository evidence and Ali's response relayed by Steven. Steven owns Part A. ONNX export and Mac parity/replay are verified; float DLC conversion remains pending while Ali seeks a compatible Linux host. Ali prioritizes the minimal float delivery independently of DSP repair. Board diagnostics below remain November 21 observations. See [CPU next steps](cpu-next-steps.md) for completed independent checks and the stopping gate. No new deadline or host availability is inferred.

## Scope and division

| Work | Owner under the recorded split | Current evidence / next action |
| --- | --- | --- |
| Mac accuracy, canonical inputs/features, replay and timing | Part A — Steven | Complete; result summaries in results/ |
| ONNX export, supported Linux conversion environment, float DLC and SDK provenance | Part B — Ali | ONNX recovered and validated; conversion host and float DLC pending |
| RB5 CPU execution and paired pose evaluation | Part A — Steven | Ready to consume a verified float DLC; does not depend on fixing DSP/GPU |
| RB5 GPU float experiment | Part A — Steven | Investigate validator SIGSEGV alongside CPU work |
| DSP runtime/quantization/calibration and INT8 experiment | Part B | Investigate missing RPC library; keep calibration separate from scored crops |
| Runtime/driver/image changes and unresolved toolchain issues | Relevant lane owner with board administrator/Vincent | Preserve existing configuration; coordinate changes and board time |
| Final four-backend table | Both | Combine measured results and explicitly documented unavailable backends |

The Notion card still says lab PC and phrases GPU after INT8 quantization. The accepted local protocol uses Steven's Mac CPU reference, float DLC for CPU/GPU, and a separately quantized model for DSP. Mac accuracy reproduces the reference band; Mac timing does not reproduce the paper's GPU timing. These clarifications are documented here without silently rewriting the source card.

## Part A delivers

Source, setup instructions, replay/import code, tests, tensor contract, small result summaries, and manifests are versioned in this repository. Large reference artifacts, model/SDK binaries, and credentials stay outside Git.

| Artifact | Purpose | Identity |
| --- | --- | --- |
| Seven-crop archive | First-frame crops from both frozen clips plus expected NCHW features; early conversion/layout check | manifests/smoke-exchange.json |
| 335-crop archive | All batch-one encoder inputs for the fixed 100-frame pose comparison | manifests/qualcomm-input-package.json |
| Canonical Mac bundle | Crops, features, metadata, poses, masks and provenance used by Steven's replay | artifacts/reference-bundle on the Mac; source hashes in manifests/results |
| Board diagnostics | Observed runtime identity, GPU crash and DSP dependency failure | results/rb5-readiness.json |

The two archives and their manifests have been staged and SHA256-verified through SCP/SSH at this directory relative to the documented shared board account's home:

```text
umetrack-handoff/steven/ref-7eab946c7430/
    qualcomm-inputs.zip
    qualcomm-smoke-2026-11-21.zip
    qualcomm-input-package.json
    smoke-exchange.json
```

See manifests/board-artifact-handoff.json for byte counts and hashes. Obtain board access details from the existing private Notion instructions, not Git. No direct teammate notification or SDK/model upload was performed. These are archives; extract into new experiment directories before running a model. Do not run against the small archive using the full-package importer schema.

## Part B returns before CPU testing

Use examples/conversion-handoff.template.json as a checklist; it is a template, not a validated conversion record.

1. Exact upstream/checkpoint identities matching manifests/upstream.json; export helper/source and ONNX SHA256/opset.
2. Conversion host OS/architecture, QAIRT/SNPE SDK release/build, tool version output, and exact export/conversion commands with logs. Use a supported host for the selected release; Mac-only access is not evidence of a supported conversion environment.
3. Float encoder DLC and its SHA256. The observed board runner reports SNPE v2.33.2.250410134701_117956. Match the intended conversion/runtime SDK build, or coordinate an intentional runtime change and repeat readiness checks. Do not silently upgrade either side.
4. Actual converted tensor names, shapes, batch size, layouts and dtypes. Input is already preprocessed. Candidate raw input is NHWC (1,96,96,1); canonical output is NCHW (1,72,6,6). Explicitly map the actual model's layout.
5. PyTorch/ONNX/converted-output numerical checks on real crops, with tolerances declared before acceptance. Seven-crop expected features are available now; a float32 output container alone does not establish FP32 internal math.
6. A reproducible target runtime setup and a proposed artifact location accessible to Steven's benchmark account. Keep credentials and licensed SDK binaries outside this public repository.

The float conversion delivery is sufficient to unblock CPU testing. DSP quantization, DSP RPC repair, and GPU debugging can proceed independently. Returning all accelerator work first is unnecessary.

## CPU execution sequence

First validate model/package hashes and actual I/O metadata. Run the seven-crop diagnostic before the complete input package. Compare returned features against the saved Mac reference after the explicitly declared layout conversion.

For the installed runner version, the observed help supports the following CPU command shape. Set the staged SDK path and actual model path through local configuration, then run from a newly extracted package directory. This command has not been executed on the encoder because the DLC is still pending.

```sh
LD_LIBRARY_PATH="$UMETRACK_SDK_TARGET_ROOT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" "$UMETRACK_SDK_TARGET_ROOT/bin/snpe-net-run" --container "$UMETRACK_FLOAT_DLC" --input_list input_list.txt --output_dir cpu-output-new --runtime_order cpu --profiling_level detailed
```

The actual release's help and model metadata take precedence. Use new output directories, preserve exact input-list order, and record the model version, invocation, requested/actual backend, output filename/layout/dtype, and logs. A startup smoke result is not a latency benchmark. Separate warm inference from loading, file I/O, transfer, and warm-up.

For the complete 335-crop run, return every required output plus a completed execution.json based on examples/board-execution.template.json. The importer currently accepts one flat output filename per Result_i, little-endian float32, with explicit NCHW/NHWC layout. Adapt and review other formats before consuming them; do not guess by byte count.

On Steven's Mac, from the repository root and using fresh output paths:

```sh
.venv/bin/python scripts/import_raw_outputs.py --package artifacts/qualcomm-inputs --returned /path/to/returned-cpu --output artifacts/imported-cpu-new
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=2 .venv/bin/python scripts/replay_imported_features.py --imported artifacts/imported-cpu-new
```

CPU acceptance: absolute MPJPE change below 0.1 mm against the paired **8.961398 mm** Mac subset, with identical masks/history and direct joint-displacement diagnostics. The 9.425145 mm full-split reference is a different denominator. Report median/p95 inference timing and actual backend evidence separately.

## Accelerator issues to reconcile in parallel

- GPU: staged validator reported prerequisites Present then SIGSEGV during RuntimeCheck, exit 139. Investigate this runtime/image combination; GPU hardware is not proven permanently unsupported.
- DSP: prerequisites Absent, missing /usr/lib/libadsprpc.so, unit/overall result Failed. The process returned 0, so treat the explicit test result as authoritative. Correct the runtime/RPC setup for this image before benchmarking.
- No UmeTrack encoder has run on RB5. Built-in diagnostic execution is distinct from actual encoder compatibility, accuracy, and performance.
- Agree accelerator accuracy allowances before a pass/fail verdict; until then report characterization. CPU/GPU use float, DSP INT8 uses separate calibration provenance.
- The source task calls for consulting Vincent after one day blocked in SNPE before switching toolchains. No purchase, toolchain switch, or message to another person is implied by this handoff.

Reserve separate board measurement windows and record concurrent workloads/thermal conditions. Use separate branches for each lane's subsequent code changes and review shared tensor/interface changes together. No new deadline or teammate acceptance is invented here.
