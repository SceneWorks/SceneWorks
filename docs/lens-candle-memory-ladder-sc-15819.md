# Lens Candle/CUDA shared image memory ladder (SC-15819)

SC-15819 implements the shared five-rung image memory contract for both inference provider IDs,
`lens` and `lens_turbo`. The SceneWorks catalog remains entry-specific: this document records
implementation evidence, not a catalog calibration promotion.

## Runtime contract

- staged conditioning/denoise/decode residency with synchronized release;
- native FLUX.2 tiled VAE decode at the published tile/overlap candidates;
- request-scoped bounded joint attention with typed cancellation;
- device-format transfer of the 48-block DiT in request-selected windows;
- exact packed q4/q8 artifact eligibility, with adapters, PiD, control routes, stale calibration,
  wrong numeric tiers, and bounded flags without staged residency rejected fail-closed;
- independent registration for `lens` and `lens_turbo`.

The runtime fingerprint is
`lens-candle-cuda-shared-ladder-device-format-blocks-v1`. The same implementation fingerprint does
not make entry-level measurements interchangeable.

## Authoritative CUDA implementation evidence

The serial ignored harness
`integration_tests::sc15819_real_weights_five_rung_sequence` ran on GPU 0 only, with no other GPU 0
workload. It used CUDA 12.9, MSVC 14.44 vcvars64, `CUDA_COMPUTE_CAP=120`, and the shipped
`SceneWorks/lens-turbo-mlx` q4 snapshot revision
`d3f485c320039595cff16d4f686a5f9378714f25`. Each request used 1024x1024, batch 1, one denoise
step, seed 15819, and the same prompt. Transfer-ready sidecars were prepared outside measured
requests in a story-local external cache.

| Rung | Live high-water | Reserved high-water | Pixel checksum |
| --- | ---: | ---: | --- |
| Resident | 30.522 GiB | 33.812 GiB | `500e4a522431550f` |
| Staged residency | 17.088 GiB | 20.594 GiB | `500e4a522431550f` |
| Bounded decode | 7.572 GiB | 9.344 GiB | `dadaf92a9abc1b42` |
| Bounded attention | 7.572 GiB | 9.156 GiB | `dadaf92a9abc1b42` |
| Bounded transformer residency | 3.455 GiB | 5.312 GiB | `dadaf92a9abc1b42` |

Resident and staged output bytes matched. Bounded-decode, bounded-attention, and bounded-transformer
output bytes matched one another. The decode checksum differs because the native tiled VAE is an
explicit alternate decode plan. The same run proved pre-cancel cleanup and invalid-prerequisite
cleanup, then returned GPU 0 to its 19 MiB idle floor.

## Calibration and catalog ownership

These measurements prove the implementation is executable and the rungs materially reduce the
request envelope for the exact Lens-Turbo q4 route above. They do not certify q8, bf16, base
`lens`, other geometry/batch points, or a reusable fit. The generated matrix must therefore remain
Implemented/unverified where applicable.

SC-15874 retains Lens-Turbo entry-level ownership. During SC-15819 integration the matrix correctly
rejected reusing the MLX base story SC-15462 or the family story SC-15819 for a Candle model cell.
Existing follow-up SC-17489 owns base `lens` Candle/CUDA calibration, admission values, and
catalog/matrix promotion. Until it completes, the current base Candle model cell remains Missing and
no `lens.candle` manifest block is fabricated.
