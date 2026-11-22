# ORT INT8 preview — not SNPE/DSP, no pass/fail

Float ONNX vs ONNX Runtime static INT8 (QDQ) encoder features on the 60 D5 crops. Characterisation only: no tolerance, no pass/fail. The SNPE quantizer and Hexagon DSP arithmetic are different, so these numbers do not predict the D8 DSP result exactly.

## Setup

- Float model: `artifacts/encoder.onnx` sha256 `1e86586510d928febb2db643caaef6e13fc48e4e5dd236424422cd89f34fa483`
- INT8 model (not in git): `artifacts/int8_preview/encoder_int8_qdq.onnx` sha256 `7c087f342abe57c4ef1d9522040d883191e2efe877d074b906931a18659b7306`
- Quantization: onnxruntime 1.32.0 `quantize_static`, QDQ, activations QInt8, weights QInt8, per_channel=True, calibration MinMax
- Calibration: 184 crops from `manifests/calibration.csv` (training users user_15, user_16; no overlap with evaluation), crops sha256 `a3455c836d860a6b1b5195cfe85b3f1b4f896491a4010ee7022c020957f6628b`
- Evaluation: 60 D5 crops `artifacts/crops/crops_uint8.npy` sha256 `0a5e042d0a4971dac990164f433c9b89c3890070069b1ed5a3aaa4269a28e7f4`, batch 1, ORT CPU provider, 1 thread
- Quantized graph: Add 12, Conv 29, DequantizeLinear 101, MaxPool 1, QuantizeLinear 43

## Overall

| max_abs | mean_abs (MAE) | RMSE | rel_l2 |
| --- | --- | --- | --- |
| 8.6931e-02 | 4.7352e-03 | 6.4262e-03 | 2.9645e-02 |

Float feature range: min -0.5602, max 1.6906, mean 0.0851, std 0.1994 (span 2.2508). Worst max_abs is 3.86% of the span.

Per-crop spread: max_abs median 4.363e-02 / worst 8.693e-02, mean_abs median 4.690e-03 / worst 5.775e-03, rmse median 6.261e-03 / worst 8.255e-03, rel_l2 median 2.968e-02 / worst 3.758e-02

## Per crop

| row | crop_id | max_abs | mean_abs (MAE) | RMSE | rel_l2 |
| --- | --- | --- | --- | --- | --- |
| 0 | `user_19-recording_00:000000:hand0:camera1` | 6.421e-02 | 4.446e-03 | 6.184e-03 | 2.899e-02 |
| 1 | `user_19-recording_00:000000:hand0:camera2` | 5.766e-02 | 4.553e-03 | 6.212e-03 | 2.881e-02 |
| 2 | `user_19-recording_00:000000:hand1:camera2` | 3.343e-02 | 4.266e-03 | 5.633e-03 | 2.637e-02 |
| 3 | `user_19-recording_00:000001:hand0:camera1` | 4.374e-02 | 4.357e-03 | 5.856e-03 | 2.713e-02 |
| 4 | `user_19-recording_00:000001:hand0:camera2` | 3.141e-02 | 4.341e-03 | 5.675e-03 | 2.605e-02 |
| 5 | `user_19-recording_00:000001:hand1:camera2` | 2.561e-02 | 4.069e-03 | 5.294e-03 | 2.468e-02 |
| 6 | `user_19-recording_00:000002:hand0:camera1` | 4.061e-02 | 4.446e-03 | 6.042e-03 | 2.769e-02 |
| 7 | `user_19-recording_00:000002:hand0:camera2` | 3.133e-02 | 4.603e-03 | 6.097e-03 | 2.789e-02 |
| 8 | `user_19-recording_00:000002:hand1:camera2` | 6.312e-02 | 4.606e-03 | 6.657e-03 | 3.104e-02 |
| 9 | `user_19-recording_00:000003:hand0:camera1` | 5.583e-02 | 4.909e-03 | 6.929e-03 | 3.145e-02 |
| 10 | `user_19-recording_00:000003:hand0:camera2` | 4.360e-02 | 5.238e-03 | 7.063e-03 | 3.256e-02 |
| 11 | `user_19-recording_00:000003:hand1:camera2` | 2.915e-02 | 4.442e-03 | 5.799e-03 | 2.691e-02 |
| 12 | `user_19-recording_00:000004:hand0:camera1` | 3.710e-02 | 5.031e-03 | 6.817e-03 | 3.045e-02 |
| 13 | `user_19-recording_00:000004:hand0:camera2` | 6.348e-02 | 5.154e-03 | 6.949e-03 | 3.172e-02 |
| 14 | `user_19-recording_00:000004:hand1:camera2` | 4.661e-02 | 4.527e-03 | 6.065e-03 | 2.788e-02 |
| 15 | `user_19-recording_00:000005:hand0:camera1` | 3.545e-02 | 4.407e-03 | 5.848e-03 | 2.593e-02 |
| 16 | `user_19-recording_00:000005:hand0:camera2` | 4.901e-02 | 5.193e-03 | 7.200e-03 | 3.313e-02 |
| 17 | `user_19-recording_00:000005:hand1:camera2` | 5.807e-02 | 4.571e-03 | 6.467e-03 | 2.965e-02 |
| 18 | `user_19-recording_00:000006:hand0:camera1` | 2.816e-02 | 4.759e-03 | 6.295e-03 | 2.788e-02 |
| 19 | `user_19-recording_00:000006:hand0:camera2` | 4.142e-02 | 4.957e-03 | 6.542e-03 | 3.015e-02 |
| 20 | `user_19-recording_00:000006:hand1:camera2` | 2.692e-02 | 4.347e-03 | 5.759e-03 | 2.593e-02 |
| 21 | `user_19-recording_00:000007:hand0:camera1` | 2.901e-02 | 4.275e-03 | 5.598e-03 | 2.479e-02 |
| 22 | `user_19-recording_00:000007:hand0:camera2` | 8.437e-02 | 5.775e-03 | 8.255e-03 | 3.754e-02 |
| 23 | `user_19-recording_00:000007:hand1:camera2` | 5.382e-02 | 5.024e-03 | 6.992e-03 | 3.135e-02 |
| 24 | `user_19-recording_00:000008:hand0:camera1` | 2.702e-02 | 4.456e-03 | 5.867e-03 | 2.609e-02 |
| 25 | `user_19-recording_00:000008:hand0:camera2` | 8.693e-02 | 5.042e-03 | 7.071e-03 | 3.196e-02 |
| 26 | `user_19-recording_00:000008:hand1:camera2` | 4.283e-02 | 4.798e-03 | 6.560e-03 | 2.911e-02 |
| 27 | `user_19-recording_00:000009:hand0:camera1` | 4.507e-02 | 4.618e-03 | 6.227e-03 | 2.762e-02 |
| 28 | `user_19-recording_00:000009:hand0:camera2` | 3.647e-02 | 4.691e-03 | 6.135e-03 | 2.804e-02 |
| 29 | `user_19-recording_00:000009:hand1:camera2` | 2.144e-02 | 4.284e-03 | 5.407e-03 | 2.423e-02 |
| 30 | `user_19-recording_00:000010:hand0:camera1` | 4.262e-02 | 4.575e-03 | 6.055e-03 | 2.734e-02 |
| 31 | `user_19-recording_00:000010:hand0:camera2` | 6.568e-02 | 4.963e-03 | 6.992e-03 | 3.211e-02 |
| 32 | `user_19-recording_00:000010:hand1:camera2` | 3.007e-02 | 4.212e-03 | 5.444e-03 | 2.472e-02 |
| 33 | `user_19-recording_00:000011:hand0:camera1` | 6.822e-02 | 4.910e-03 | 6.780e-03 | 3.108e-02 |
| 34 | `user_19-recording_00:000011:hand0:camera2` | 4.662e-02 | 4.993e-03 | 6.824e-03 | 3.290e-02 |
| 35 | `user_19-recording_00:000011:hand1:camera2` | 4.460e-02 | 4.511e-03 | 6.026e-03 | 2.796e-02 |
| 36 | `user_19-recording_00:000012:hand0:camera1` | 2.972e-02 | 4.982e-03 | 6.589e-03 | 3.018e-02 |
| 37 | `user_19-recording_00:000012:hand0:camera2` | 5.068e-02 | 4.986e-03 | 6.944e-03 | 3.234e-02 |
| 38 | `user_19-recording_00:000012:hand1:camera2` | 2.677e-02 | 4.359e-03 | 5.659e-03 | 2.576e-02 |
| 39 | `user_19-recording_00:000013:hand0:camera1` | 4.727e-02 | 4.767e-03 | 6.575e-03 | 3.016e-02 |
| 40 | `user_19-recording_00:000013:hand0:camera2` | 4.564e-02 | 5.664e-03 | 8.071e-03 | 3.758e-02 |
| 41 | `user_19-recording_00:000013:hand1:camera2` | 3.658e-02 | 4.522e-03 | 5.932e-03 | 2.762e-02 |
| 42 | `user_19-recording_00:000014:hand0:camera1` | 3.301e-02 | 4.701e-03 | 6.169e-03 | 2.799e-02 |
| 43 | `user_19-recording_00:000014:hand0:camera2` | 3.929e-02 | 4.830e-03 | 6.385e-03 | 2.999e-02 |
| 44 | `user_19-recording_00:000014:hand1:camera2` | 4.366e-02 | 5.088e-03 | 6.837e-03 | 3.248e-02 |
| 45 | `user_19-recording_00:000015:hand0:camera1` | 4.301e-02 | 5.072e-03 | 6.926e-03 | 3.177e-02 |
| 46 | `user_19-recording_00:000015:hand0:camera2` | 3.739e-02 | 4.772e-03 | 6.506e-03 | 3.068e-02 |
| 47 | `user_19-recording_00:000015:hand1:camera2` | 5.449e-02 | 4.427e-03 | 6.172e-03 | 2.971e-02 |
| 48 | `user_19-recording_00:000016:hand0:camera1` | 3.809e-02 | 4.689e-03 | 6.204e-03 | 2.852e-02 |
| 49 | `user_19-recording_00:000016:hand0:camera2` | 6.722e-02 | 5.077e-03 | 7.066e-03 | 3.368e-02 |
| 50 | `user_19-recording_00:000016:hand1:camera2` | 5.917e-02 | 5.095e-03 | 7.001e-03 | 3.411e-02 |
| 51 | `user_19-recording_00:000017:hand0:camera1` | 4.897e-02 | 4.558e-03 | 6.147e-03 | 2.819e-02 |
| 52 | `user_19-recording_00:000017:hand0:camera2` | 4.775e-02 | 4.761e-03 | 6.475e-03 | 3.080e-02 |
| 53 | `user_19-recording_00:000017:hand1:camera2` | 3.313e-02 | 4.632e-03 | 6.077e-03 | 2.980e-02 |
| 54 | `user_19-recording_00:000018:hand0:camera1` | 4.669e-02 | 4.528e-03 | 6.064e-03 | 2.785e-02 |
| 55 | `user_19-recording_00:000018:hand0:camera2` | 5.758e-02 | 5.061e-03 | 6.798e-03 | 3.213e-02 |
| 56 | `user_19-recording_00:000018:hand1:camera2` | 4.975e-02 | 4.788e-03 | 6.481e-03 | 3.160e-02 |
| 57 | `user_19-recording_00:000019:hand0:camera1` | 3.124e-02 | 4.582e-03 | 5.983e-03 | 2.762e-02 |
| 58 | `user_19-recording_00:000019:hand0:camera2` | 5.243e-02 | 5.205e-03 | 7.166e-03 | 3.396e-02 |
| 59 | `user_19-recording_00:000019:hand1:camera2` | 4.194e-02 | 4.616e-03 | 6.101e-03 | 2.985e-02 |
