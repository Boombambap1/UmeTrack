# RB5 readiness — November 21, 2026

## Access and runtime identity

The Mac's existing Tailscale profile was reconnected while retaining its existing accept-routes preference; no reset/account switch occurred. The documented hostname resolved and login through the documented unprivileged shared account succeeded. A repository-local ignored known-hosts file records the initial host key; global SSH settings were unchanged. Credentials were used transiently and were not stored in repository files or logs.

The board reports Linux aarch64, Arch Linux ARM, and eight processors. Package versions: glibc 2.43+r22+g8362e8ce10b2-2 and GCC 16.1.1+r12+g301eb08fa2c5-1.

The shared login cannot read the historical personal account's SDK because its home has owner-only permissions. Documented administrator access allowed the SDK to be inspected without changing those permissions. snpe-net-run --help and --version work; the runner reports **SNPE v2.33.2.250410134701_117956**. The platform validator is also present. Binary/library hashes and diagnostic summaries are in results/rb5-readiness.json. No DLC was found under the inspected staged SDK tree; this is not a global search for models.

## Actual backend diagnostics

Using administrator access and the staged SDK library directory via LD_LIBRARY_PATH, ran snpe-platform-validator for GPU and DSP with --coreVersion --libVersion --testRuntime and separate task-specific temporary output directories. Each command had a 12-second limit; neither reached that timeout.

| Backend | Observed result | Interpretation |
| --- | --- | --- |
| CPU | No model execution attempted | Requires a suitable sample or matching encoder DLC |
| GPU | Prerequisites Present, then segmentation fault during RuntimeCheck; exit 139 | This validator/configuration failed; GPU support and model execution are unverified |
| DSP | Missing /usr/lib/libadsprpc.so; prerequisites Absent; unit test and CSV overall result Failed | Current RPC/runtime library setup is incomplete |

The DSP process returned exit code 0 despite its explicit failure report. Its CSV labels the runtime Supported but reports missing prerequisites and Overall Result Failed. Interpret the diagnostic contents, not the process status or architecture label alone. It reported Hexagon Architecture V66; that does not establish successful DSP execution.

No UmeTrack encoder inference, feature output, board latency, or paired pose comparison has occurred. The GPU check reported a core dump; no dump was retrieved or removed. Task temporary diagnostic outputs were created. Board drivers, permissions, accounts, services, and SDK files were not changed.

## Next actions

Part B needs to supply a float encoder DLC with checkpoint/model identities, actual I/O metadata, and a conversion SDK build matched to the intended runtime release. Record the full tool version strings, rather than assuming the historical SDK directory alone identifies every binary.

Investigate the GPU validator crash and missing DSP RPC library against the board image/runtime package before formal accelerator benchmarking. A bounded regular-file search of /usr/lib and /opt found no matching adsprpc/cdsprpc library; it was not a complete filesystem inventory. Do not classify the hardware as permanently unsupported from these results or report a fallback run as acceleration.

Stage the intended matched runtime for Steven's benchmark account, then verify real model execution. Use the seven-crop exchange for numerical/layout checks and the full 335-crop bundle for paired pose scoring. CPU work can continue once a suitable float DLC exists while accelerator issues are investigated. The task's escalation rule applies if SNPE work remains blocked for one day; no message to Vincent or another teammate has been sent.

Private endpoints, authentication diagnostics, exact board paths/commands, and host-key records remain under ignored artifacts/board-readiness-2026-11-21. No credentials or SDK binaries belong in Git. The public-sized result JSON contains artifact hashes and bounded findings.
