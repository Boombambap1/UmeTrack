# D5 — PyTorch vs ONNX Runtime encoder parity

**Result: PASS** on 60 real crops (`artifacts/crops/crops_uint8.npy`, frames from `raw_data/real/separate_hand/testing/user_19/recording_00`), batch size 1, ORT CPUExecutionProvider, 1 thread(s).

Tolerances (`tolerances.onnx_vs_torch`, declared before this run, unchanged):

| Check | Limit | Measured (worst) | Pass |
| --- | --- | --- | --- |
| max_abs over all crops | 1.0e-04 | 3.695e-06 | True |
| rel_l2 per crop | 1.0e-05 | 9.078e-07 (row 47) | True |

Overall: max_abs 3.695e-06, mean_abs 5.535e-08, rel_l2 4.406e-07.

PyTorch feature range: min -0.5602, max 1.6906, mean 0.0851, std 0.1994, fraction exactly zero 0.000.

Preprocessing: `uint8 / 255 -> float32 [N,1,96,96]` NCHW.
ONNX: `artifacts/encoder.onnx` sha256 `1e86586510d928febb2db643caaef6e13fc48e4e5dd236424422cd89f34fa483`.
Crops: sha256 `0a5e042d0a4971dac990164f433c9b89c3890070069b1ed5a3aaa4269a28e7f4`; manifest `manifests/crops_d5.csv`.
Versions: torch 2.15.0, onnxruntime 1.32.0, numpy 2.5.5.

## Per crop

| row | max_abs | mean_abs | rel_l2 | pass |
| --- | --- | --- | --- | --- |
| 0 | 1.073e-06 | 5.338e-08 | 4.128e-07 | True |
| 1 | 1.192e-06 | 5.152e-08 | 3.865e-07 | True |
| 2 | 1.431e-06 | 5.751e-08 | 4.923e-07 | True |
| 3 | 1.431e-06 | 5.447e-08 | 4.241e-07 | True |
| 4 | 8.345e-07 | 5.504e-08 | 3.841e-07 | True |
| 5 | 4.768e-07 | 4.899e-08 | 3.266e-07 | True |
| 6 | 7.153e-07 | 4.835e-08 | 3.446e-07 | True |
| 7 | 1.311e-06 | 5.781e-08 | 4.412e-07 | True |
| 8 | 1.311e-06 | 5.088e-08 | 3.823e-07 | True |
| 9 | 8.345e-07 | 4.963e-08 | 3.385e-07 | True |
| 10 | 7.153e-07 | 5.669e-08 | 3.728e-07 | True |
| 11 | 1.431e-06 | 5.498e-08 | 4.216e-07 | True |
| 12 | 2.265e-06 | 7.051e-08 | 5.948e-07 | True |
| 13 | 4.768e-07 | 5.101e-08 | 3.282e-07 | True |
| 14 | 5.960e-07 | 4.957e-08 | 3.546e-07 | True |
| 15 | 9.537e-07 | 5.539e-08 | 4.047e-07 | True |
| 16 | 1.669e-06 | 5.812e-08 | 4.862e-07 | True |
| 17 | 7.153e-07 | 4.716e-08 | 3.197e-07 | True |
| 18 | 5.960e-07 | 5.014e-08 | 3.121e-07 | True |
| 19 | 2.027e-06 | 5.917e-08 | 5.465e-07 | True |
| 20 | 8.345e-07 | 5.029e-08 | 3.515e-07 | True |
| 21 | 1.669e-06 | 6.181e-08 | 5.513e-07 | True |
| 22 | 3.576e-07 | 4.543e-08 | 2.873e-07 | True |
| 23 | 1.311e-06 | 6.268e-08 | 4.878e-07 | True |
| 24 | 1.788e-06 | 6.859e-08 | 6.343e-07 | True |
| 25 | 2.265e-06 | 6.812e-08 | 6.296e-07 | True |
| 26 | 9.537e-07 | 5.041e-08 | 3.487e-07 | True |
| 27 | 4.768e-07 | 4.380e-08 | 2.743e-07 | True |
| 28 | 1.192e-06 | 5.652e-08 | 4.272e-07 | True |
| 29 | 1.311e-06 | 6.183e-08 | 4.405e-07 | True |
| 30 | 7.153e-07 | 4.992e-08 | 3.436e-07 | True |
| 31 | 1.788e-06 | 6.561e-08 | 5.646e-07 | True |
| 32 | 8.345e-07 | 4.949e-08 | 3.449e-07 | True |
| 33 | 1.431e-06 | 5.790e-08 | 4.488e-07 | True |
| 34 | 5.364e-07 | 5.388e-08 | 3.729e-07 | True |
| 35 | 4.172e-07 | 4.540e-08 | 2.975e-07 | True |
| 36 | 7.153e-07 | 5.089e-08 | 3.321e-07 | True |
| 37 | 5.960e-07 | 5.070e-08 | 3.385e-07 | True |
| 38 | 1.311e-06 | 6.031e-08 | 4.576e-07 | True |
| 39 | 1.073e-06 | 5.439e-08 | 3.885e-07 | True |
| 40 | 4.172e-07 | 5.302e-08 | 3.464e-07 | True |
| 41 | 2.146e-06 | 5.938e-08 | 5.255e-07 | True |
| 42 | 8.345e-07 | 5.954e-08 | 4.177e-07 | True |
| 43 | 3.338e-06 | 8.236e-08 | 8.518e-07 | True |
| 44 | 3.576e-07 | 4.629e-08 | 3.105e-07 | True |
| 45 | 7.153e-07 | 4.931e-08 | 3.337e-07 | True |
| 46 | 2.503e-06 | 6.472e-08 | 6.310e-07 | True |
| 47 | 3.695e-06 | 8.299e-08 | 9.078e-07 | True |
| 48 | 1.073e-06 | 5.076e-08 | 3.716e-07 | True |
| 49 | 4.768e-07 | 4.615e-08 | 3.145e-07 | True |
| 50 | 3.576e-07 | 4.427e-08 | 3.045e-07 | True |
| 51 | 7.153e-07 | 5.177e-08 | 3.510e-07 | True |
| 52 | 6.557e-07 | 5.199e-08 | 3.625e-07 | True |
| 53 | 4.172e-07 | 4.856e-08 | 3.237e-07 | True |
| 54 | 1.550e-06 | 5.755e-08 | 4.894e-07 | True |
| 55 | 1.073e-06 | 5.179e-08 | 3.731e-07 | True |
| 56 | 9.537e-07 | 5.468e-08 | 4.223e-07 | True |
| 57 | 1.192e-06 | 5.191e-08 | 3.692e-07 | True |
| 58 | 1.311e-06 | 5.937e-08 | 4.612e-07 | True |
| 59 | 1.550e-06 | 6.632e-08 | 5.520e-07 | True |
