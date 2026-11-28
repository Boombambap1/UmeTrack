# RB5 CPU runtime setup (Part B package for Part A's RB5 CPU run)

Prepared 2026-11-28 by Part B. This is for Steven's benchmark account on the RB5. It contains no hostnames,
accounts or credentials. Paths below use placeholders: `$PKG_PARENT` is a folder that Steven's benchmark
account owns, and `$RUN` is a new, empty run folder.

## What the package is

| Item | Value |
| --- | --- |
| File | `snpe-2.33.2-aarch64-oe-cpu-runtime.tar.gz` (not in Git) |
| Size / sha256 | 19,797,559 bytes / `eea77d92ff7dd2eef2a4f240c60566b58bc7b165653e7fbbf0341c84545f39f1` |
| Inside | `snpe-2.33.2-aarch64-oe-cpu-runtime/` with `bin/`, `lib/`, `BUILD.txt` and `MANIFEST.sha256` (sha256 `a1a8cac7eaf33cb31b854a81167ca629b764bc54278ce3bc9976b8e694fa030b`) |
| SDK release / build | QAIRT/SNPE 2.33.2.250410, build `250410134701_117956` (SDK zip sha256 `37487de4d9dd770f15826224bc915d138417eb44445ebe6bd8718b6ca172d617`) |
| Target | `aarch64-oe-linux-gcc11.2`, copied unchanged from the unpacked SDK's `bin/` and `lib/` folders |
| Runtime | **CPU only.** No DSP, GPU or HTP libraries are included. |

Files (`MANIFEST.sha256` lists every file except itself):

| Path | Bytes | sha256 | Why it's included |
| --- | --- | --- | --- |
| `bin/snpe-net-run` | 571,576 | `587043a3b6466df57d169f44b8b774f6ccf48e370e771c839d59ab366af90de8` | the runner |
| `bin/snpe-diagview` | 633,736 | `3337fb2c54c5756ab9033ffef59ca62b4bc81935397aba149d14ebc5302800dd` | converts `SNPEDiag.log` to text (per-layer runtimes) |
| `bin/snpe-platform-validator` | 16,481,464 | `e508dae2500b54948679b2561fb72c8a98cd7a645e1ce25e9cd984e1d9b3ef32` | the same validator as the board's staged runtime; not needed for the CPU run |
| `lib/libSNPE.so` | 21,522,304 | `f43ca6e72c05db526b4cf4336a10bea388fcfea6ccc0fa7e74a98a6ab2d9d9bc` | the SNPE runtime, which includes the CPU runtime |
| `lib/libPlatformValidatorShared.so` | 16,464,624 | `ec5688fa5cb07e2152ad19a1524803eb3e5826f8c8d9d30be06064c4461c6de3` | loaded at runtime by the validator (not listed as NEEDED) |
| `BUILD.txt` | 359 | `2d92dafd6a91591cf0bf21c868dd8fbb4b874f2ddb20716923a3e9ad9148fe23` | release, target and prerequisites |

**Same build as the board.** The four binaries and libraries recorded in `results/rb5-readiness.json` (the
board's staged runtime, inspected 2026-11-21) have byte-identical sha256 values here: `snpe-net-run`,
`snpe-platform-validator`, `libSNPE.so` and `libPlatformValidatorShared.so`. The package is that same build,
in a folder your account can read.

**How the file list was found.**
- `readelf -d` NEEDED was followed from the three tools and `libSNPE.so`, keeping only files in the SDK's
  `lib/aarch64-oe-linux-gcc11.2/`. Only `libSNPE.so` resolves there; `libPlatformValidatorShared.so` was added
  because the validator loads it at runtime.
- Every file is AArch64 and requires at most GLIBC 2.34.
- The tools look for their libraries at `$ORIGIN/../lib` and `libSNPE.so` at `$ORIGIN`, so keep `bin/` and
  `lib/` side by side as unpacked.
- `libSNPE.so` contains the names `libcdsprpc.so` (DSP) and `libOpenCL-pixel.so` (GPU), which it can load at
  runtime for those runtimes. They are deliberately not in this package and are not needed for
  `--runtime_order cpu`.
- If either name, or any fallback, unavailable or error line, appears in a CPU run's log, Part B's reader marks
  the run `backend_verified = false`. That run then needs review, not a pass.

## Board prerequisites (system libraries, not in the package)

Required by the NEEDED entries: `libatomic.so.1`, `libstdc++.so.6`, `libgcc_s.so.1`, `libm.so.6`, `libc.so.6`,
with GLIBC ≥ 2.34. The board reported glibc 2.43 and GCC 16.1.1 (`results/rb5-readiness.json`). Check without
installing anything:

```bash
ldconfig -p | grep -E 'libatomic\.so\.1|libstdc\+\+\.so\.6|libgcc_s\.so\.1|libm\.so\.6|libc\.so\.6'
```

If one is missing, stop and tell Part B. Don't install system packages for this run without agreeing it first.

## Unpack and environment

```bash
cd "$PKG_PARENT"            # a folder your benchmark account owns
sha256sum snpe-2.33.2-aarch64-oe-cpu-runtime.tar.gz   # must be eea77d92...f39f1
tar -xzf snpe-2.33.2-aarch64-oe-cpu-runtime.tar.gz
export SNPE_RT="$PKG_PARENT/snpe-2.33.2-aarch64-oe-cpu-runtime"
export PATH="$SNPE_RT/bin:$PATH"
export LD_LIBRARY_PATH="$SNPE_RT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

The RPATH already finds `lib/`. `LD_LIBRARY_PATH` is set as well so that a different `libSNPE.so` elsewhere on
the board can't be picked up first.

## Verify before running

```bash
cd "$SNPE_RT" && sha256sum -c MANIFEST.sha256       # every line must say OK
ldd "$SNPE_RT/bin/snpe-net-run" | grep -E 'not found|libSNPE'   # no "not found"; libSNPE from $SNPE_RT/lib
which snpe-net-run snpe-diagview                     # both from $SNPE_RT/bin
snpe-net-run --version                               # must show 2.33.2.250410134701_117956
```

The board's staged runner previously printed `SNPE v2.33.2.250410134701_117956`. Any other build means stop.

## Seven-crop smoke run (CPU)

Inputs:
- your smoke archive `qualcomm-smoke-2026-11-21.zip` (sha256 `fcbcad10e9b53795451506380734bc9f8ba6f883df725230e9c0b4e52eafbdac`);
- `encoder_float.dlc` (sha256 `d0342d35a65f9de43967ffc1d2f6bfb2356e92b80b3dacd35bf57e5bf1572f7e`, 15,145,523 bytes).

```bash
mkdir "$RUN" && cd "$RUN"                 # new, empty folder
cp /path/to/encoder_float.dlc . && sha256sum encoder_float.dlc   # must be d0342d35...572f7e
unzip -q /path/to/qualcomm-smoke-2026-11-21.zip inputs/'*' input_list.txt
sha256sum input_list.txt                  # must be 59c41769cf063f13610fa12fe3bc8102c0be74bcc407c383b19b693c37bc29bb
test ! -e output && snpe-net-run --container encoder_float.dlc --input_list input_list.txt \
  --output_dir output --runtime_order cpu --profiling_level detailed > net_run.log 2>&1; echo "exit=$?" | tee net_run.exit
snpe-diagview --input_log output/SNPEDiag.log > diag.txt 2>&1; echo "diagview exit=$?"
grep -c ': CPU$' diag.txt                 # expect 67, one line per layer
```

Run it from `$RUN`. The input list uses relative paths (`inputs/crop-0000i.raw`), and `Result_<i>` matches line
i of the list.

**Where things appear** (all under `$RUN`):
- `output/Result_0` … `output/Result_6`, each containing only `features.raw`: NHWC `[1,6,6,72]`, little-endian
  float32, 10,368 bytes.
- `output/SNPEDiag.log` (and `output/SNPEDiag_0.log`), written by `snpe-net-run` at the top level of
  `output/`. Part B's reader accepts only these `SNPEDiag*.log` files there, records their sha256 and never uses
  them for the numbers; anything else extra in `output/` is refused.
- `net_run.log`, `net_run.exit` and `diag.txt` next to `output/`.

Return all of `output/`, `net_run.log`, `net_run.exit`, `diag.txt`, the `snpe-net-run --version` output and the
`sha256sum -c MANIFEST.sha256` output. Keep the timing runs (warm-up excluded, median) separate from this
correctness run.

## Acceptance (declared before the run; `configs/experiment.yaml`)

- **Features:** `tolerances.rb5_cpu_vs_reference` (agreed with Steven 2026-11-28). On every crop, max_abs ≤ 1e-3
  and rel_l2 ≤ 1e-4 vs Part A's frozen reference features, after an explicit NHWC → NCHW transpose.
- **Backend:** `backend_verified` must be true. That means every per-layer line in `diag.txt` reports `CPU`,
  with no fallback, unavailable or error lines. A silent fallback is a failed run.
- **Pose:** `tolerances.rb5_cpu_pose`, absolute MPJPE change < 0.1 mm vs the paired 8.961398 mm.
- MAE, RMSE and direct joint displacement are diagnostics only.

For reference, the same DLC and crops on the conversion host (x86_64 CPU, not the RB5) gave max_abs 1.073e-06
and worst per-crop rel_l2 4.223e-07 (`results/conversion/host_smoke_check/summary.md`).
