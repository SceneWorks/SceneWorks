# FLUX.2-dev Candle prompt-enhancement memory gate

This note records the bounded CUDA evidence used for the SC-18474 catalog gate. It is not a
replacement for the repository memory-calibration harness and does not promote a 1024x1024 matrix
cell.

## Source identity

- Product inference pin: `e199538429fda38c1a22682c5d0582ba8f88e920`.
- Q8 edit was built and run directly from that exact merge.
- Q4 text-to-image, Q4 edit, and Q8 text-to-image were built from
  `351a482fa2a5ccd3276a5ed01e0c3b309c18bd9e`. That commit is an ancestor of the product pin, and
  the complete tree delta from it to `e1995384` is only
  `crates/media/mlx-gen/mlx-gen-flux2/src/model.rs`; the Candle provider, gen-core contract, and
  CUDA runtime used by these captures are byte-identical.
- Live provider calibration fingerprint:
  `flux2-dev-cuda-caption-upsample-staged-host-full-edge-decode-bounded-attention-device-format-blocks-v3`.
- Live real-weight residency fingerprint: `flux2-cuda-residency-caption-upsample-v2`.

## Durable captures

All runs used GPU 0 on the RTX PRO 6000 Blackwell CUDA host, 256x256 output, one diffusion step,
one caption token, and VramProbe. Each enhancement request emitted exactly one typed `Enhanced`
report with no fallback.

| Route | Tier | Overall peak | Output SHA-256 |
| --- | --- | ---: | --- |
| text-to-image | Q4 | 40.8 GB | `01D50AC669D5073558DB0B1E8E0FD7088E295204D84A0E1FFD598B0631080748` |
| single-reference edit | Q4 | 42.7 GB | `416CF39763746042666368AA7529259A44C55BFCBDCC15B0C0B4802A585FBEB9` |
| text-to-image | Q8 | 70.8 GB | `7C95377853CC8BEA42445C09EA5C2A6A38EB33CA5F03481B9E6F1CA0EB1997B3` |
| single-reference edit | Q8 | 70.8 GB | `E2C953E7BDA518DCFA982D4233507CF132B4410E77D9A3C8302C5F29DCB1A783` |

The first three raw logs and checksum manifest are in
`D:/repos/SceneWorks/.codex/evidence/sc-18474/inference-351a482fa2a5ccd3276a5ed01e0c3b309c18bd9e`.
The exact-merge Q8 edit command, raw stdout/stderr, evidence summary, checksum manifest, and PNG are
in
`D:/repos/SceneWorks/.codex/evidence/sc-18474/inference-e199538429fda38c1a22682c5d0582ba8f88e920`.

## Gate derivation

The catalog uses the larger measured route peak for each tier: 42.7 GB for Q4 and 70.8 GB for Q8.
The worker adds the repository-wide 2 GB dedicated-VRAM reserve, so the resulting requirements are
44.7 GB and 72.8 GB. Rounded up to the next supported hardware tier, Q4 starts at 48 GB and Q8 at
80 GB. The sequential rows deliberately repeat those same high-waters; these bounded captures do
not establish a smaller caption-aware sequential peak.

`vramMeasuredPixels` is 65,536 (256 squared), matching the captures. BF16 has no row. The five
older Q4 1024x1024 v2 bindings remain unchanged as historical records, but runtime selection now
requires their calibration fingerprint to match the live provider. They therefore cannot authorize
v3 execution, including the obsolete low bounded-rung peaks that could otherwise admit 16 GB or
24 GB hardware.

These values are conservative admission floors, not claims of full 1024x1024/28-step calibration.
Fresh harness measurements are required before the matrix can publish current v3 cells or before
any lower tier may be admitted.
