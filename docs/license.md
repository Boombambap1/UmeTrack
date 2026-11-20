# UmeTrack licensing note (D2)

This is a summary for the team, not legal advice.

## Sources

| Material | License text | Where |
| --- | --- | --- |
| Code + pretrained weights | CC BY-NC 4.0 (`Attribution-NonCommercial 4.0 International`) | `third_party/UmeTrack/LICENSE` at upstream commit `e263ec0844abf73457468a050a75853444d33d69`, sha256 `336255dc30193e8e15d689d9481bb05673d89055718f3a96923a7ffb99adbbaf`; README §License: "UmeTrack is licensed under the Creative Commons Attribution-NonCommerial 4.0 International License" |
| Dataset (`UmeTrack_data`) | CC BY-NC 4.0 — byte-identical to the upstream LICENSE (same sha256, git blob `108b5f00…`) | `LICENSE` on `facebookresearch/UmeTrack_data` branch `main`, commit `950bda2ab602d0ca5591476e04c993ec6f3cac4f` (2023-05-03) |

The upstream README says the pre-trained inference model, the sample code and the dataset "are all included in
this project", so all three are treated as CC BY-NC 4.0.

**Gap:** the `UmeTrack_data` commit our submodule pins (`355df534a1c761ac416427925485b311f22ce50e`,
2023-04-28, the one D5 downloaded from) has no LICENSE file and no license text in its README. The license was
published with the later `main` history (a separate single-commit history). We assume the same terms apply to
the pinned data; worth confirming if it ever matters.

## What CC BY-NC 4.0 allows and restricts

Common to all three (§2(a)(1)): a worldwide, royalty-free, non-exclusive licence to reproduce and share the
material, and to produce and share adapted material, **for NonCommercial purposes only**. NonCommercial
means "not primarily intended for or directed towards commercial advantage or monetary compensation" (§1(i)).
Patent and trademark rights are not licensed (§2(b)(2)). The material is provided as-is, without warranty (§5).

| Material | Allowed (non-commercial) | Restricted |
| --- | --- | --- |
| Code | Run, study, modify, wrap (our `scripts/`), share with attribution | Any commercial use; relicensing under terms that stop recipients complying |
| Pretrained weights | Run inference, export to ONNX, convert to DLC, quantize to INT8, benchmark, share with attribution | Commercial deployment (e.g. inside a sold product); the converted/quantized models carry the same NC restriction |
| Dataset | Download, evaluate, make crops and calibration sets, publish metrics; extracting/sharing a substantial portion is NC only (§4) | Commercial use of the data or of substantial extracts; sharing without attribution |

Open interpretation point: format conversion may count as a "technical modification" (§2(a)(4)), which does
not produce Adapted Material, while quantization or other changes probably do. Either way the result stays
non-commercial and needs attribution if shared.

## What we must do

- **Attribution** whenever we share upstream material or anything derived from it (code, weights, ONNX/DLC files,
  crops, data extracts), per §3(a): credit the creators (Meta Platforms, Inc. and affiliates / the UmeTrack
  authors), keep the copyright notice, state the CC BY-NC 4.0 license with a link, keep the disclaimer notice,
  link to the source, and say what we modified.
- **No relicensing**: we must not put upstream-derived material under a licence that prevents recipients from
  complying with CC BY-NC 4.0 (§3(a)(4)). Keep the upstream LICENSE with anything we redistribute.
- **Non-commercial use only** for code, weights and data, and for everything derived from them.
- Keep upstream code in the pinned submodule rather than copying it, so provenance stays clear.
- Our own repository has no LICENSE file yet; choosing one for our scripts is a separate team decision and must
  not appear to relicense the upstream material.

## Open questions for Vincent

1. Is Project Standalone non-commercial research?
2. Could the headset (or software from it) ever be sold or licensed?
3. If yes: what is the plan — seek permission or a commercial licence from Meta, or switch to an alternative
   hand-tracking model with suitable terms? Until that is answered, UmeTrack results should be treated as a
   research benchmark only.

This is a summary for the team, not legal advice.
