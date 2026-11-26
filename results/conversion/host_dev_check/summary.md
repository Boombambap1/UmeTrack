# Development check on the conversion host — x86_64 snpe-net-run vs ONNX Runtime encoder features

**Development check on the conversion host, not the RB5 experiment: x86_64 host runtime, Part B development crops. Says nothing about RB5 CPU accuracy or timing.**

**Purpose: development only - Part B regenerated crops, not a formal comparison**

**Result: NOT PASS (numeric_match=True, backend_verified=False)** on 8 crops.

- numeric_match: **True** (tolerances `tolerances.host_cpu_vs_onnx_dev`, declared before this run, unchanged)
- backend_verified: **False** — requested `CPU`, actual from log `None` (no per-layer runtime in diag log)

| Check | Limit | Measured (worst) | Within |
| --- | --- | --- | --- |
| max_abs over all crops | 1.0e-03 | 2.265e-06 | True |
| rel_l2 per crop | 1.0e-04 | 6.919e-07 (idx 3) | True |

Overall: max_abs 2.265e-06, mean_abs (MAE) 4.324e-08, RMSE 8.309e-08, rel_l2 3.869e-07.
Diagnostic, host vs PyTorch: max_abs 1.550e-06, rel_l2 3.043e-07.

Host output: `features.raw` dims [1, 6, 6, 72] layout NHWC -> explicit transpose to NCHW [1,72,6,6]. Result_<i> matched to input_list line i, reference by crop_id.
Reference: `artifacts/reference/onnx_features.npy` sha256 `2e54dd25e3129b5c978aca03289b64c8b0b19d7ef4bcc10daffbafc4f2f6a414`.
Run log: `results/raw/conversion/host_dev/net_run.log` sha256 `4bc4979e23b8cba09deaed7e52f2b9491e35ced3c9da8779cc498e0f60281484`.
Runtime evidence (source: diag): none
Negative lines: none

## Per crop

| idx | crop_id | reference row | max_abs | mean_abs | rel_l2 | within |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | `user_19-recording_00:000000:hand0:camera1` | 0 | 4.768e-07 | 3.253e-08 | 2.384e-07 | True |
| 1 | `user_19-recording_00:000002:hand1:camera2` | 8 | 4.023e-07 | 3.262e-08 | 2.289e-07 | True |
| 2 | `user_19-recording_00:000005:hand1:camera2` | 17 | 1.073e-06 | 4.597e-08 | 3.773e-07 | True |
| 3 | `user_19-recording_00:000008:hand0:camera2` | 25 | 2.265e-06 | 6.327e-08 | 6.919e-07 | True |
| 4 | `user_19-recording_00:000011:hand0:camera2` | 34 | 5.960e-07 | 3.960e-08 | 2.857e-07 | True |
| 5 | `user_19-recording_00:000014:hand0:camera1` | 42 | 8.345e-07 | 4.888e-08 | 3.692e-07 | True |
| 6 | `user_19-recording_00:000017:hand0:camera1` | 51 | 5.960e-07 | 3.631e-08 | 2.636e-07 | True |
| 7 | `user_19-recording_00:000019:hand1:camera2` | 59 | 1.192e-06 | 4.673e-08 | 3.968e-07 | True |
