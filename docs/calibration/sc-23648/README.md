# Production MLX admission regression (sc-23648)

The 0.8.10 manifests and exact q4 snapshots reproduce both reported errors for the
red-raincoat prompt: SDXL 20.90 GiB and SANA Sprint 18.08 GiB against an 8 GiB host
with a 2 GiB reserve. Z-Image passes. The regression now calls the same
`prepare_mlx_load_policy` used by production and binds quality rows to the resolved
snapshot identity, rather than directly forcing Sequential + Deferred.

SDXL omitted q4/q8 streaming declarations; RealVisXL already declared them.
RealVisXL Lightning also omitted its bf16/q8 streaming tiers. Provider validation
still decides whether each artifact/mode supports the declared strategy. CLIP
window demand now comes from the production tokenizer, with the byte bound retained
only as a fallback when assets cannot be read.

SANA supports staged, native tiled decoding, but its unknown phase workspace was
priced with a generic multi-gigabyte allowance. Its old whole-decoder tile sweep
cannot describe the current dense-head/tiled-tail implementation. The focused
current-runtime evidence is in `sana-sprint-phases.json`; no catalog measurement
matrix was run. The shared estimator charges actual simultaneous component bytes,
uses the observed conditioning/denoising residues, and retains the entire decode
observation as workspace. Scaling that whole observation by the maximum of image
area ratio, tile area ratio and one preserves both the dense head and output
accumulation without inventing a split from one data point. Base SANA additionally
charges two CFG forwards, merge tensors and solver state. The geometry/domain guards
and existing reserve/uncertainty policy remain in force.

A cached or precomputed conditioning phase can have a counter below the resident
weight total. Such a phase remains unknown, but no longer discards usable denoise
and decode observations. This shared issue affects retained anchors for both SANA
variants, FLUX.2-dev, SD3.5 and SenseNova. Existing deferred/bounded-regime exclusions
remain unchanged; this does not claim those larger models fit 8 GiB.

The production-selected four-step render regression passed for Z-Image (two images),
SDXL and SANA Sprint with maximum active peaks of 4.20, 4.61 and 3.29 GiB respectively.
`selected-runtime.json` records the source identity, request, assertions and limits.
These are execution/memory checks, not image-quality equivalence measurements.
Validation commands and final integration evidence are recorded on sc-23648.
The development host has 128 GiB; physical 8 GiB runtime validation remains distinct
from passing an emulated admission budget.
