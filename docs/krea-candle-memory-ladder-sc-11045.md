# Krea 2 Turbo Candle/CUDA five-rung memory ladder (SC-11045)

This record separates implementation/loadability evidence from calibration. It **does not promote any
Krea Candle cell to Verified**, and it does not move any shipped calibration binding. The capture is
published separately at `docs/generated/krea-candle-five-rung-sc-11045.json`, outside the runtime
admission bundle, following the SC-15817 precedent.

## Why this capture certifies nothing

Every record carries `status: "gated"` by the harness's own declaration. Each one states the reason
in `diagnostics.blockers`:

> five-rung oracle capture measures exact per-rung memory and strategy identity for SC-16059; it
> intentionally remains gated because this run does not repeat the full promotion-quality,
> negative-mutation, and lifecycle scenario suite

That is a design property of the fixture, not a failure of this run — the step exited `success`, the
adapter reports `diagnostics.execution: "executed"`, and every rung carries real device deltas from a
real probe. But `scripts/generate-memory-matrix.mjs` promotes only records whose `status` is
`complete` or `runtime_complete` (see its `record-not-complete` reason and the
`currentEnvironmentVerification` filter), so these five are excluded from promotion by construction.

Measured, not assumed — ingesting these records into `docs/generated/memory-calibration-evidence.json`
and regenerating gives:

- `summary.currentCalibrationRuns`: **0 → 0** (unchanged).
- `summary.calibrationRunsByStatus`: `{complete: 85, runtimeComplete: 23}` (unchanged — the five are
  counted in neither bucket). sc-21715 has since made this tally PARTITION the bundle, so the same
  ingest today reads `{complete: 85, runtimeComplete: 23, gated: 5, negativeComplete: 0}` — the five
  are named rather than uncounted, and neither certifying bucket moves, which is the claim this
  bullet was making.
- `npm run report:stale-lanes`: `9 stale, 0 current` (unchanged); `candle:krea_2_turbo` still ranked
  #8 at `1/1` bindings, **`0/0` records**, `status=stale`.

So the lane does not move even on the record half, and the §7d binding half has no certifying record
to stand on. Stamping the shipped bindings to this revision would assert a promotion the evidence
does not support, which is the "launder a stale measurement into a current one" failure the
calibration runbook refuses by design. **§7d is deliberately not performed here.**

One further reason to keep these records out of the admission bundle at the time: they would have
been the first `gated` records ever to enter it, and that surfaced a latent inconsistency in the
generator — `summary.calibrationRuns` was `records.length` (113) while
`"the published summary re-derives from the evidence bundle and the closure ledger (sc-17774)"`
recomputed it as `complete + runtimeComplete` (108). The two agreed only because every record in the
bundle had always been complete. That was a real defect in its own right, not this story's to fix,
and forcing it open was never a precondition for publishing a gated oracle. It was filed as sc-21715
and fixed there: the tally now partitions the bundle over every admitted status, so admitting these
five is no longer blocked by it. Promotion still is — for the reasons above, which are about
evidence, not about counting.

## Capture provenance

Captured by the guarded `windows-candle.yml` dispatch
([run 33000590976](https://github.com/SceneWorks/SceneWorks/actions/runs/33000590976), conclusion
`success`) — the documented route; local capture is refused by the pinned `VramProbe` idle gate
because GPU 1 hosts the desktop.

- Fixture: `fresh-five-rung-krea-q4-1024-seed16402-step2`, `--fresh-per-case`.
- Artifact: `SceneWorks/krea-2-turbo-mlx@d009674080cc1bccf2b629d834c34bf5eccdb723`, variant `q4`.
- SceneWorks `769363b7bd7d0e8d73a82e48f521aa5dadcb9d5a`, inference
  `3cd86ba2165f35db7c4fceecbeb7fbd12bca1c0c`, both worktrees clean.
- Provider closure digest `60d4cc2d8214…` — this is the lane's **live** digest, so the capture is at
  the current closure; only the gated status keeps it from certifying.
- Contract fingerprint `krea-turbo-cuda-phase-curves-v1`; load shape `deferred_materialization`;
  target `krea_2_turbo` q4 `text_to_image` at 1024x1024, batch 1.
- Device: NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition, compute capability 12.0,
  102,641,958,912 bytes, CUDA 12.9, driver 596.36, GPU 0 selected through `CUDA_VISIBLE_DEVICES`.

## The five rungs

Observed overall device peak delta, one fresh process per rung:

| Rung | Parameters | Observed peak (bytes) | GiB |
| --- | --- | ---: | ---: |
| Resident | none | 25,171,066,880 | 23.44 |
| Staged residency | none | 22,352,494,592 | 20.82 |
| Bounded decode | tile 512, overlap 128 | 14,800,650,240 | 13.78 |
| Bounded attention | tile 512, overlap 128, score budget 134,217,728 | 11,310,989,312 | 10.53 |
| Bounded transformer residency | prior parameters, block window 1 | 3,996,123,136 | 3.72 |

The ladder is strictly monotone across all five rungs — high-water **23.44 GiB** at `resident` down
to a **3.72 GiB** floor at full bounding, a 6.3x span.

Per-phase deltas confirm where the peak lives. At `bounded_decode` the phases are conditioning
3.57 GiB, denoise 13.78 GiB, decode 8.57 GiB — **denoise attention is the peak, not VAE decode**,
matching the Wan finding. The exception is `resident`, whose decode phase (23.44 GiB) dominates
because nothing is bounding it; that is exactly the rung the ladder exists to retire.

## Reuse comparison

The workflow also captured the same fixture with `--batch-rungs` and ran `compare-reuse` between the
two passes. Verdict: **`unable_to_amortize`** — one of the two verdicts the lane accepts.

Batched rungs track the fresh ones closely at the bounded end (`bounded_decode` 13.88 vs 13.78 GiB,
`bounded_attention` 10.35 vs 10.53 GiB, `resident` identical at 23.44 GiB) but diverge where a
carried-over allocator state changes the phase shape — `staged_residency` reads 22.69 vs 20.82 GiB
with conditioning jumping 3.57 → 11.44 GiB, and `bounded_transformer_residency` reads 4.79 vs
3.72 GiB. Sharing one process across rungs does not recover the load cost and distorts the very
phase boundaries the ladder measures, so `--fresh-per-case` remains the authoritative shape. The
comparison and batched bundles are retained on the run artifact
(`sc-16059-candle-reuse-33000590976`); only the authoritative fresh capture is committed.

## What would make this lane current

A certifying capture — one that also runs the promotion-quality, negative-mutation, and lifecycle
scenario suite so the records land `complete` rather than `gated` — followed by the binding half in
§7d of the calibration runbook. Until then `candle:krea_2_turbo` keeps serving its measured numbers
behind the widened 2.00% margin, which is a signal and not a gate.
