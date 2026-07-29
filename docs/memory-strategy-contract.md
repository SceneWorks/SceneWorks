# Shared memory-strategy contract

SC-15449 defines one worker-owned selection path for every image provider. The generated
[`memory-matrix.json`](generated/memory-matrix.json) is the authoritative inventory and
evidence schema; `sceneworks_worker::memory_strategy::select_strategy` is the authoritative runtime
selector. Providers may reject a selected strategy defensively, but must not contain a second
least-cost selector.

## Ownership

| Owner | Responsibilities |
| --- | --- |
| Provider | Capabilities, cache/load/drop lifecycle, estimate formula, backend realization, calibration ABI fingerprint, and defense-in-depth validation |
| Manifest/generated evidence | Coefficients, envelopes, provenance, backend/tier/mode/overlay/geometry/parameter coverage, conformance state, and structural N/A |
| Worker | Live budget and reclaimable memory, ordered least-cost selection, precision invariance, rejection/fallback, telemetry, and user advice |

The five ordered strategies are `Resident`, `StagedResidency`, `BoundedDecode`,
`BoundedAttention`, and `BoundedTransformerResidency`. Candle/CUDA budgets discrete live VRAM and
may credit the evicted single-slot cache. MLX/Metal budgets total unified memory and reserves the OS
share as explicit headroom. In both cases effective capacity is
`min(total - reserve, available + reclaimable - reserve)`, saturating at zero. Those backend
semantics are not interchangeable.

The selector accepts only `Verified` evidence with all six generated evidence dimensions, matching
provider/calibration fingerprints, current SceneWorks and inference revisions, an exact tier/mode/
overlay match, and an in-envelope geometry. `Implemented/unverified`, unknown, stale, fingerprint
mismatch, structural N/A, route unavailable, or out-of-envelope evidence returns `Unverified`; it
never selects an optimized rung. Equality fits (`needed_gb <= available_gb`). A calibration that
needs tolerance must include it in the provider estimate and golden evidence.

Strategy changes never change precision. A lower precision tier is a separate candidate evaluated
by the existing tier chooser before the memory selector.

## Lifecycle and telemetry

A provider capability declaration is selectable only when cancel and error transitions are safe:
each transition must either leave the warm cache reusable or invalidate it atomically. Warm cache
hits retain their selected realization; cold loads are re-estimated. The worker records the selected
strategy, estimate, effective live budget, rejection/unverified reason, and whether reclaimable
memory affected the outcome. Advice may name only a lower geometry or strategy that has verified
evidence.

## Reconciliation

- Krea 2 Turbo CUDA text-to-image keeps its measured boundaries. Resident admission now runs
  through the shared selector first; `threeStage`, `tiledVae`, `chunkedAttention`, and
  `streamedBlocks` map respectively to staged residency, bounded decode, bounded attention, and
  bounded transformer residency. Its manifest phase curves and maximum pixel envelope now carry
  explicit revision/fingerprint provenance and exact cumulative tile, overlap,
  attention-budget, and transformer-window parameters before flowing through the shared selector.
- Generic MLX cold loads keep the existing recursive safetensors disk sum, provider footprint
  component split, architecture headroom, OS reserve, and weights-fit safety floor. Resident and
  staged choices now flow through the shared selector; the provider capability registry remains the
  defense-in-depth check.
- Mage must not copy either selector. Its provider estimator must add request geometry, mode, overlay,
  and parameter terms to the provider-owned formula, publish matching evidence cells, and submit
  ordinary strategy candidates to the worker selector. Until that calibration is verified,
  request-aware optimized selection remains `Unverified` and the existing MLX weights safety floor
  is the only permissible fallback.

The provider ABI types and compatibility defaults live in the pinned SceneWorks inference
repository. A provider that does not implement the additive contract exposes resident-only,
unverified defaults, preserving its current behavior without claiming optimization.
