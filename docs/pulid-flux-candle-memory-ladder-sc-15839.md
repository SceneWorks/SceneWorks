# PuLID-FLUX Candle/CUDA memory ladder — SC-15839

SC-15839 implements the shared five-rung image memory ladder for the bespoke
PuLID-FLUX Candle route. The worker admits only the exact PuLID identity route:
PiD and requests carrying any LoRA bypass the ladder. The provider keeps the
EVA-CLIP, IdFormer/cross-attention, SCRFD, ArcFace, and BiSeNet identity stack
resident while applying staged/windowed residency to the FLUX.1-dev trunk.

## Revisions and fixture

- SceneWorks: `a87a977d89b44f11dd02921404873f98a04ed192`
- inference: `5f973a73bf00307240afd81d2778ba9d89349e51`
- Backend: Candle/CUDA 12.9 on GPU 0 only
- GPU: NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition, compute 12.0,
  driver 596.36, 97,887 MiB
- Geometry: Q4, identity overlay, one reference, batch 1, 1024x1024, 4 steps
- Cache: `E:\huggingface`

The reference is a visible, frontal real-person portrait used as a local
operator validation fixture. Its source PNG hashes to
`d9dfdaaac866ac410dfaa7a624c674ff027ca36728ae656be8d03891883bf53d`;
the 502x724 PPM used by the test hashes to
`475cf7e86e0470a026efe12ae19b71286a283603c8bb0d0bea80a369b0e2645f`.
The original remote source URL was not preserved, so this evidence does not
claim external dataset provenance.

The exact Q4 base inventory contains 515 files and 17,013,959,875 bytes with
inventory digest
`ca76b88ccb101284b734a6a7a894a3caec98e3a191507e8c9bd12980aff39459`.
The evidence bundle also records exact revisions, byte counts, and SHA-256
digests for the PuLID adapter, EVA-CLIP weights, and three-model face bundle.

## Physical CUDA results

Every rung ran in a fresh process with `CUDA_VISIBLE_DEVICES=0`. An external
500 ms `nvidia-smi` sampler measured the full request. The baseline was 19 MiB
for every accepted run.

| Rung | Peak MiB | Delta peak MiB | Reduction from resident | Provider ms | Wall ms | Output SHA-256 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| resident | 26,680 | 26,661 | — | 19,123 | 93,502 | `dc2b53f…2575` |
| staged residency | 22,072 | 22,053 | 4,608 MiB (17.3%) | 45,182 | 49,581 | `dc2b53f…2575` |
| bounded decode | 15,000 | 14,981 | 11,680 MiB (43.8%) | 77,220 | 80,983 | `5759f659…283f` |
| bounded attention | 12,568 | 12,549 | 14,112 MiB (52.9%) | 81,690 | 85,924 | `5759f659…283f` |
| bounded transformer residency | 7,480 | 7,461 | 19,200 MiB (72.0%) | 148,511 | 149,897 | `5759f659…283f` |

The calibration receipt was
`pulid-flux-cuda-identity-stack-staged-decode-attention-block-window-v1`
for every rung. Staged residency is byte-identical to resident. Bounded
decode, bounded attention, and bounded transformer residency are mutually
byte-identical. Compared with resident, their RGB8 maximum error is 3, mean
error is 0.1348972321, RMSE is 0.3673233971, and PSNR is 56.8298 dB.

The first bounded-transformer attempt could not write a device-format cache
sidecar under sandbox restrictions and stopped before generation. Its partial
7,448 MiB sample is explicitly excluded. The table contains the subsequent
clean run, which completed and produced the expected output hash.

## Identity and lifecycle checks

The real-weight identity gate detected the reference face and produced a
512-dimensional ArcFace embedding. The identity output detected a face at
0.840 confidence with 0.8516 reference cosine similarity. The no-identity
control detected a face at 0.797 confidence with -0.0992 reference cosine
similarity; its mean pixel difference from the identity output was 149.47.
Pre-denoise cancellation and mid-denoise cancellation after three steps both
returned `Canceled`.

The raw, immutable receipts are:

- `docs/calibration/sc-15839-pulid-q4-resident.log`
- `docs/calibration/sc-15839-pulid-q4-staged-residency.log`
- `docs/calibration/sc-15839-pulid-q4-bounded-decode.log`
- `docs/calibration/sc-15839-pulid-q4-bounded-attention.log`
- `docs/calibration/sc-15839-pulid-q4-bounded-transformer-residency.log`
- `docs/calibration/sc-15839-pulid-q4-quality.log`

They are registered as schema-v4 `sourceSessions` in
`docs/generated/memory-calibration-evidence.json`.

## Admission status

This evidence deliberately does not create a `complete` or `runtime_complete`
calibration record and does not add `vramGb` values to the matrix. The schema's
runtime-complete shape is base-only (`overlay=none`), whereas this route is an
identity overlay. Full completion also requires evidence that was not captured
here: a warm-repeat request, injected-error cleanup with a warm follow-up,
measured negative mutation, and official per-phase observed/predicted metrics.

Consequently the generated matrix remains **Implemented/unverified** and the
selector remains fail-closed until a current, full PuLID identity calibration
record is packaged. This run certifies neither Q8 nor BF16.
