# Qualcomm raw-input and output handoff

## Completed and tested locally

- scripts/export_raw_inputs.py writes one little-endian float32 raw file per crop, input_list.txt, and a source-bound package.json. Candidate package: artifacts/qualcomm-inputs; zip: artifacts/qualcomm-inputs.zip; sizes/checksums: manifests/qualcomm-input-package.json.
- src/umetrack_rb5/raw_exchange.py validates input bytes, crop/list order and provenance, then reconstructs frame batches from explicit-layout outputs.
- scripts/import_raw_outputs.py imports a returned directory into a new canonical feature bundle, preserving the immutable Mac reference.
- scripts/replay_imported_features.py evaluates the imported features through the same host downstream pipeline and reports feature/pose error changes, direct joint displacements, and masks.
- scripts/check_raw_exchange.py uses test-only saved-reference fixtures. For both NCHW and NHWC, all 335 crops returned identical features and 100 frames returned identical poses, with zero MPJPE change. These are not SDK outputs and do not emulate RB5 hardware. Results: results/raw-exchange-check.json. Eleven test groups passed; artifacts/raw-exchange-tests-final.log.

## Package specification

The current candidate input package declares NHWC (1,96,96,1), batch one, 335 crops, 36,864 bytes per crop. Inputs are already normalized/mirrored/warped. Do not preprocess again. B must confirm actual DLC metadata before this declaration is treated as correct for the deployed model. NCHW input export is also available with an explicit flag; no layout is inferred automatically.

Expected default float32 output size is 10,368 bytes per crop: 72 x 6 x 6 x 4. NCHW and NHWC outputs have the same byte count but different ordering. A wrong, plausible layout declaration cannot be detected by byte counts alone. Confirm metadata and compare a few actual reference feature outputs before formal acceptance.

The current importer expects one flat output filename under Result_0, Result_1, ..., in the exact order of the unmodified input_list.txt. Additional, missing, truncated, or nonfinite raw files fail. Native quantized/float16 output files, nested tensor output paths, and batched models are not supported automatically and need reviewed adaptation.

## B's execution record

Fill examples/board-execution.template.json or the included execution.template.json. Save it as execution.json inside the returned-output directory. Record device, requested backend, reported actual backend, exact conversion/runtime SDK build versions, actual DLC SHA256, source checkpoint SHA256, explicit output layout/dtype/filename, exact invocation, input-list/package hashes, and hashes of evidence logs copied into that directory.

For execution_kind=board the importer requires a DLC model, matching SDK versions, and the expected source checkpoint. It checks the supplied provenance and evidence file hashes, but does not authenticate a remote device or automatically establish that a backend avoided fallback. Review actual runtime logs separately. Input-list identity and a trusted run procedure establish crop association; swapping two same-sized raw results cannot be discovered from the raw bytes alone.

Run the command shape in the archive README only after confirming the chosen release's runner syntax and model metadata. CPU/GPU use the float model under this plan. DSP uses its separately calibrated quantized model. Tailscale/SSH/file transfer are separate from model conversion/execution. No model conversion or board execution occurred during package preparation.

## Host commands

Choose new output directories; existing artifacts are not overwritten:

```sh
.venv/bin/python scripts/export_raw_inputs.py --output artifacts/qualcomm-inputs-new --input-layout NHWC
.venv/bin/python scripts/import_raw_outputs.py --package artifacts/qualcomm-inputs --returned /path/to/returned --output artifacts/imported-cpu
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=2 .venv/bin/python scripts/replay_imported_features.py --imported artifacts/imported-cpu
```

The import command additionally validates the local reference/input package before consuming outputs. The replay command keeps ground truth and valid masks fixed; mismatches fail instead of shrinking the scoring denominator. Full pose comparison requires the complete 335-crop package. A few-crop startup smoke test is a separate diagnostic, not a full replay result.

Rerun the local plumbing check with a fresh directory:

```sh
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=2 .venv/bin/python scripts/check_raw_exchange.py --output artifacts/raw-exchange-new
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=2 .venv/bin/python -m unittest discover -s tests -v
```

At the end of the initial packaging run, no teammate invitation, file transmission, Git commit/push, or Notion update had been performed. The subsequent board artifact staging and publication preparation are recorded below.

Official workflow references (actual installed release takes precedence):
- https://qdc.qualcomm.com/support/user-guide/ai-models/ai-interactive-sessions
- https://www.thundercomm.com/rubik-pi-3/en/docs/rubik-pi-3-user-manual/1.1.2/ai-developer-workflow/

## Early exchange and access update — November 21

Steven owns Part A. The local early encoder exchange archive artifacts/qualcomm-smoke-2026-11-21.zip contains 7 crops from frame zero of each frozen clip, expected NCHW raw features, and an identity/hash manifest. It is a conversion/layout diagnostic, not the complete importer schema or pose-evaluation package. See manifests/smoke-exchange.json. Actual conversion/runtime metadata must still be confirmed. The archive was subsequently staged on the board as recorded below.

Isolated Mac batch-one timing is now recorded separately in results/mac-encoder-batch-one.json: 3.944 median and 4.147 p95 ms/crop, 1,005 calls, exact saved-feature equality. These are warm Mac compute measurements on battery power.

Board access now succeeds through the documented unprivileged account after reconnecting Tailscale. The historically staged runtime is under an owner-only personal home and cannot be inspected through that login. Documented administrator access subsequently verified that staged SNPE v2.33.2.250410134701_117956 starts without changing permissions. Its GPU validator crashed (exit 139), and DSP diagnostics failed due to missing /usr/lib/libadsprpc.so despite returning exit 0. Obtain the matching float DLC and investigate these runtime issues; details are in docs/board-readiness.md and results/rb5-readiness.json. No UmeTrack encoder inference has occurred.

## Shared handoff

See docs/part-a-part-b-handoff.md for the reconciled division of work, conversion delivery checklist, and CPU-first execution gate. Both crop archives and their manifests are staged and SHA256-verified in the documented shared board account under umetrack-handoff/steven/ref-7eab946c7430; manifests/board-artifact-handoff.json records identities. SDK binaries, credentials, and models remain outside Git. This transfer ran no inference. GPU/DSP repairs do not block the first float CPU comparison once Part B delivers the matched DLC.

## Independent preparation update — November 23

The recovered float ONNX is validated on all 335 frozen crops, including downstream pose replay on the Mac; [the small summary](../results/canonical-onnx-validation.json) preserves the November 21 evidence. The float DLC remains pending, with Ali seeking a compatible conversion host. Fresh local checks verified both crop archives and repeated the 335-crop raw export/import/replay for NCHW and NHWC with exact feature/pose equality. These checks use Mac features and do not establish Qualcomm inference. See [CPU next steps](cpu-next-steps.md) and [readiness results](../results/pre-handoff-readiness.json).

Stop before model-specific board preparation until the DLC/hash, matched SDK build, actual tensor metadata, conversion command/log and accessible runtime setup are supplied. No board runtime, permissions, drivers or services were changed during these checks.
