# PyTorch3D on the Mac reference host

Installed PyTorch3D 0.7.8 from tag V0.7.8, commit 75ebeeaea0908c5527e7b1e305fbc7681382db47. This release lists PyTorch 2.4.1 as supported. Source: https://github.com/facebookresearch/pytorch3d/blob/75ebeeaea0908c5527e7b1e305fbc7681382db47/INSTALL.md

The normal CPU build failed because the installed Apple Clang rejects std::is_arithmetic specialization in PyTorch 2.4.1's strong_type.h. A first retry with CXXFLAGS did not pass the flag through this build system. Supplying CFLAGS=-Wno-invalid-specialization allowed the full CPU extension to build without editing either upstream source tree. This is a local toolchain workaround, not an upstream-certified macOS configuration. CFLAGS overrides default compiler flags; benchmark reports must retain the recorded build configuration.

After installing requirements-mac-reference.txt and cloning the pinned source into third_party/pytorch3d, the successful command from the repository root was:

```sh
PATH="$PWD/.venv/bin:$PATH" MAX_JOBS=2 MACOSX_DEPLOYMENT_TARGET=14.0 CC=clang CXX=clang++ CFLAGS=-Wno-invalid-specialization PYTORCH3D_FORCE_NO_CUDA=1 .venv/bin/python -m pip install --no-build-isolation ./third_party/pytorch3d
```

The built wheel was tagged macosx_26_0_arm64 despite the explicit deployment target; this installation was verified only on the current Mac, not on older macOS releases. Logs for all attempts are in ignored artifacts/build/. The previous environment snapshot and its content hash are preserved there as well.

Validation completed:

- `.venv/bin/python scripts/check_reference_dependencies.py`: analytic rotations, compiled CPU nearest-neighbor operation, upstream joint-pivot check, tracker and evaluation imports passed.
- `.venv/bin/python -m pip check`: no broken requirements.

No dataset was downloaded or evaluated during this installation. No source replacement or approximate rotation implementation was introduced. No RB5 software or runtime was changed.
