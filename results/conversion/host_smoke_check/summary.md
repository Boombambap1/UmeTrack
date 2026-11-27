# Smoke check on the conversion host (x86_64 CPU), not the RB5 experiment — snpe-net-run vs Part A's reference features

**Smoke check on the conversion host (x86_64 CPU), not the RB5 experiment: Part A's frozen crops and reference features, x86_64 host runtime. Says nothing about RB5 CPU accuracy or timing; the RB5 CPU run is Part A's.**

**Purpose: smoke - Part A frozen crops verified against manifests/smoke-exchange.json**

**Result: PASS** on 7 crops.

- numeric_match: **True** (tolerances `tolerances.host_cpu_vs_reference_smoke`, declared before this run, unchanged)
- backend_verified: **True** — requested `CPU`, actual from log `CPU` (diag runtime matches request)

| Check | Limit | Measured (worst) | Within |
| --- | --- | --- | --- |
| max_abs over all crops | 1.0e-03 | 1.073e-06 | True |
| rel_l2 per crop | 1.0e-04 | 4.223e-07 (idx 4) | True |

Overall: max_abs 1.073e-06, mean_abs (MAE) 4.189e-08, RMSE 6.442e-08, rel_l2 3.006e-07.
No PyTorch diagnostic: Part B's PyTorch features come from regenerated crops, not Part A's frozen ones.

Host output: `features.raw` dims [1, 6, 6, 72] layout NHWC -> explicit transpose to NCHW [1,72,6,6]. Result_<i> matched to input_list line i, reference by crop_id.
Reference: Part A's features `artifacts/reference/smoke-2026-11-21` (NCHW), 7 files, each sha256 verified against the pinned package manifest (sha256 `cc0f5ca73df17a85e7ef660da8c0c6fb07e66cc5da16c6cc21b45d07bc826fb5`, pinned by `manifests/smoke-exchange.json`); matched by crop_id in the verified order.
Run log: `results/raw/conversion/host_smoke/net_run.log` sha256 `2952b356441002b64a531ebcca8cdb8fe1895d3a10185f44b017751131bb7b70`.
snpe-net-run SNPEDiag*.log in the outputs dir (tolerated, not used for numeric checks): none
Runtime evidence (source: diag): `0: _node_Conv_102 : 2583 us : CPU`; `1: _node_relu : 638 us : CPU`; `2: _node_max_pool2d : 574 us : CPU`; `3: _node_Conv_104 : 2397 us : CPU`; `4: _node_relu_1 : 573 us : CPU`
Negative lines: none

## Per crop

| idx | crop_id | reference row | max_abs | mean_abs (MAE) | RMSE | rel_l2 | within |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | `user_19-recording_00:000000:hand0:camera1` | 0 | 4.768e-07 | 3.713e-08 | 5.241e-08 | 2.454e-07 | True |
| 1 | `user_19-recording_00:000000:hand0:camera2` | 1 | 5.960e-07 | 3.941e-08 | 5.673e-08 | 2.631e-07 | True |
| 2 | `user_19-recording_00:000000:hand1:camera2` | 2 | 5.960e-07 | 3.784e-08 | 5.530e-08 | 2.585e-07 | True |
| 3 | `user_23-recording_00:000000:hand0:camera1` | 3 | 6.557e-07 | 4.511e-08 | 6.930e-08 | 2.819e-07 | True |
| 4 | `user_23-recording_00:000000:hand0:camera2` | 4 | 9.537e-07 | 5.022e-08 | 8.230e-08 | 4.223e-07 | True |
| 5 | `user_23-recording_00:000000:hand1:camera1` | 5 | 4.172e-07 | 3.737e-08 | 5.333e-08 | 2.869e-07 | True |
| 6 | `user_23-recording_00:000000:hand1:camera2` | 6 | 1.073e-06 | 4.614e-08 | 7.486e-08 | 3.325e-07 | True |
