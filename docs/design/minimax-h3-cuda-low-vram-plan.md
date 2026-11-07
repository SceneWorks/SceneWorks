# MiniMax H3 CUDA memory optimization plan

Date: 2026-11-05

Status: **SW-F329** implementation and runtime acceptance verified. Matched-policy
control 42335898368 produced identical Resident/window media for the original
request. The permanent inference pin is `2b09938e33370915d4f853389961709b8bbf8bf9`.
[PR3331](https://github.com/SceneWorks/SceneWorks/pull/3331) and the feature
acceptance report record final CI and protected delivery status.

Make MiniMax H3 render long clips on CUDA using bounded GPU residency and accurate
request accounting. At the planning baseline, the acceptance case was
**1344Ã—768, 345 frames, q4, with Veda enabled**, failing preflight with an
approximately 106 GB requirement against approximately 95 GB available. Deliver a usable
24 GiB execution configuration as well as correcting that workstation case.

## Current behavior and demonstrated gaps

The source baseline is SceneWorks `40bac3f2d`, with inference pinned to
`178eb45e5ad986d2a7236c8f6b800139675baa31`. Reconcile relevant changes at execution
start before editing; the source references below describe this baseline.

| Area | Existing behavior | Gap to close |
| --- | --- | --- |
| Component staging | CUDA H3 forces sequential execution and releases the text encoder, DiT, and VAEs between phases. | Preserve this behavior and expose it truthfully to strategy selection. |
| Transformer residency | CUDA constructs the transformer on the device and retains its block stack during denoising. | Stream a bounded window of blocks for both the DiT and text encoder. MLX already has corresponding deferred loaders and window planning. |
| VAE decoding | Spatial tiling and temporal chunking already exist. | Reuse them and price their actual lifetimes; do not rebuild the decoder. |
| Dense attention | The CUDA runtime enables media FlashAttention by default; supported calls use fused attention, with a query-chunked materializing fallback. | Price and bound the route that can actually execute, including fallback. A compile-time feature alone is insufficient proof of runtime kernel support. |
| Strategy contract | CUDA H3 advertises only `Resident`; all four optimized strategies are `Missing`. | Register existing mechanisms and new streaming with executable support and correct parameter domains. |
| Stock cold admission | The q4 catalog peak is 40.068 decimal GB, measured at 1344Ã—768 and **124 frames**; the gate adds its reserve without receiving the requested duration. | Use resolved request geometry and selected strategy, including 345 frames, before allocation. Keep the measurement as evidence for its measured configuration. |
| Veda admission | Both cold and queued checks add `WorkspaceBound.total_bytes` to a dense baseline. That total includes host allocations, QKV/output storage, and dense fallback storage. | Separate host and device pools and compose simultaneous lifetimes within each phase. Determine actual overlap rather than assuming every term is an incremental allocation. |

The 106 GB message is an analytical admission result, not a measured peak of the
reported render. Its excess cannot all be attributed to host RAM: overlapping
baseline, input/output, and fallback lifetimes also need inspection. Conversely,
the 124-frame measurement does not prove that a 345-frame dense render fits.

Some CUDA strategy comments still attribute missing support to an absent
calibration identity, while production fingerprint construction now exists.
Inspect executable declarations and runtime selection; do not perpetuate that
historical explanation or create a new measurement-freshness requirement.

## Requirements

- **E1 â€” Bounded transformer residency.** Support deferred CUDA loading and a
  bounded block window for both H3 transformers. Preserve stock q4, q8, and bf16
  loading and the semantics of existing accepted native/imported checkpoints,
  adapters, and H3 task partitions. Stream compatible sources without silently
  changing format or precision. Report an unsupported streaming configuration
  before materializing tensors, while preserving its existing resident route.
- **E2 â€” Bounded working memory.** Bound attention, feed-forward, staging, and
  predictor allocations for the resolved request. Streamed execution must not
  first load or expand the entire transformer on the GPU. Account for shared
  projections, token refiners, AdaLN tables/precompute, adapters, retained
  activations, current blocks, and transfers that really coexist. Reuse the
  existing native H3 allocation facts and window planners.
- **E3 â€” Accurate admission.** Calculate separate host and GPU requirements in
  bytes, with explicit conversions for decimal GB and GiB. Use phase maxima for
  successive lifetimes and sums for overlapping lifetimes, including actual
  dense fallback. Both cold and queued checks must use the same request and
  selected strategy, actual free memory, reference geometry, and adapter stack.
  Unknown allocations must not become zero, and stale measurement provenance
  must not change runtime behavior.
- **E4 â€” Product integration.** Automatically select a fitting supported strategy
  before the fixed stock peak can reject a streamable request. Preserve the
  prompt, seed, resolution, frame count, steps, requested tier, conditioning,
  adapters, audio, and Veda controls. If nothing fits, explain the limiting
  phase/pool and offer remedies available for that request. Do not suggest a
  smaller quant tier when q4 is already the smallest supported tier.
- **E5 â€” Numerical and lifecycle correctness.** Preserve dense semantics within
  established backend tolerances. Compare Veda streaming with the equivalent
  Veda route, not with dense output as an equality oracle. Cancellation, failure,
  and subsequent jobs must release block windows, staging buffers, temporary
  tensors, and provider state. Keep existing FlashAttention and VAE tiling active.
- **E6 â€” Demonstrated delivery.** Complete the exact workstation case and a
  24 GiB configuration at the same 1344Ã—768 / 345-frame geometry. Validate dense
  fallback within an admitted envelope. Reuse focused item evidence and perform
  one terminal real-weight acceptance campaign after integration.

## Implementation items

The local plan references map to the tracked CodeGraph items below.

| Ref | CodeGraph | Type | Repository | Outcome | Blocked by |
| --- | --- | --- | --- | --- | --- |
| H3-1 | SW-6921 | Story | inference | Execute H3 with bounded CUDA transformer residency | None |
| H3-2 | SW-6922 | Bug | inference | Account for Veda host and GPU allocations by lifetime | None |
| H3-3 | SW-6923 | Story | inference | Select truthful H3 CUDA memory strategies | H3-1, H3-2 |
| H3-4 | SW-6924 | Story | SceneWorks | Admit and run long H3 requests through the shared strategy path | H3-3 |
| H3-5 | SW-6925 | Task | SceneWorks | Integrate the inference pin and prove long-clip memory targets | H3-4 |

### H3-1 Execute H3 with bounded CUDA transformer residency

**Goal:** Make the CUDA runtime capable of completing conditioning and denoising
without keeping either complete transformer on the GPU. Inherits E1, E2, E5.

Reuse the MLX deferred-load/window design where its contracts apply, with CUDA
device-format transfer and explicit resource ownership. Keep any necessary
checkpoint repacking or preparation outside the per-block hot path and bound
its memory separately. Stream shared/root modules and precompute AdaLN according
to their actual lifetimes; model blocks alone are not the complete working set.
Inspect existing MLP chunking, which currently addresses element-count limits,
and make its allocation bound sufficient for the selected memory budget.

Acceptance criteria:

1. Both the text encoder and DiT execute with a declared bounded block window;
   device allocation traces show no full-transformer load or dense expansion
   before streaming, including the existing packed loading paths.
2. Streamed and equivalent resident computations agree within established
   tolerances for the affected routes; adapters, both task partitions, accepted
   source topology, and generated audio retain their existing semantics.
3. Working allocations obey the selected window/chunk limits, and cancellation
   or an injected load/forward failure leaves no surviving window that affects
   the next job.

Expected paths: inference `candle-gen-minimax-h3/src/{model.rs,dit/,text_encoder/,
memory_strategy.rs}`, its loaders, and shared `gen-core-h3` H3 memory/window code.
Validation uses existing loader, parity, allocation, and cancellation seams;
reuse an existing deferred storage abstraction where possible. Terminal
real-weight evidence belongs to H3-5.

### H3-2 Account for Veda host and GPU allocations by lifetime

**Goal:** Replace the mixed aggregate bound with usable pool and lifetime facts,
while retaining a safe bound on every executable fallback. Inherits E2, E3, E5.

Inspect actual CUDA predictor/scorer/kernel ownership before changing formulas.
Distinguish host-only allocations, device allocations, transfer copies, shared
dense buffers, and the lifetimes retained during fallback. Treat fused and
materialized attention according to runtime capability and dispatch. If a
fallback can allocate more than the admitted route, bound it to the same budget
or refuse that request before generation rather than risking a mid-render OOM.
Keep MLX unified-memory accounting correct while changing shared types.

Distinguish Veda sparse-to-dense fallback from fused-attention-unavailable
materializing fallback: the Veda dense path can still dispatch FlashAttention.
Force and identify both reachable routes in dedicated inference validation,
including the selected query chunk sizes and their byte bounds. A forced Veda
fallback alone does not establish materializing-fallback coverage.

Acceptance criteria:

1. Reported host and GPU byte bounds cover their respective allocations;
   host-only storage is excluded from the VRAM requirement, shared storage is
   counted once per overlapping lifetime, and successive alternatives use a
   maximum rather than an unconditional sum.
2. The admitted bound covers predictor loading, sparse execution, self-test,
   and forced dense fallback for the actual backend and geometry, including
   unsupported fused-kernel cases and constrained chunk sizes.
3. Exact-budget and just-under-budget fixtures exercise each pool independently;
   Veda-off behavior and MLX accounting remain correct, and overflow or unknown
   allocation facts cannot silently admit an unsafe configuration.

Expected paths: inference `gen-core/src/veda/workspace.rs`,
`gen-core-h3/src/veda/workspace.rs`, CUDA H3 Veda execution, and native H3
allocation facts. Use source allocation/lifetime tests and existing sparse/dense
kernel parity checks. Do not substitute reduced accounting for an unimplemented
memory optimization.

### H3-3 Select truthful H3 CUDA memory strategies

**Goal:** Let the existing planner choose working H3 optimizations with complete
load, capability, and allocation facts. Inherits E1, E2, E3, E5.

Acceptance criteria:

1. Staged residency and bounded VAE decode are declared and selectable where the
   existing pipeline implements them; bounded transformer residency is selectable
   for the deferred loaders from H3-1 with the actual component/window domain.
2. Contract, lifecycle, allocation facts, strategy composition, and executable
   conformance behavior agree. A declaration alone cannot make an unavailable
   window, precision, source, or attention route selectable.
3. At a constrained budget the shared selector chooses a fitting optimized
   configuration; impossible budgets reject with a limiting phase/pool, and
   changing measurement age or pin provenance does not change that decision.

Expected paths: inference CUDA H3 `memory_strategy.rs` and its submodules,
provider behavior registration, and shared H3 strategy/allocation facts. Reuse
existing conformance and selection tests. Remove assertions and comments that
encode obsolete missing-support assumptions as each mechanism becomes real.

### H3-4 Admit and run long H3 requests through the shared strategy path

**Goal:** Carry the complete request into strategy selection before cold loading,
then execute the selected configuration without a contradictory second gate.
Inherits E1, E3, E4, E5.

Acceptance criteria:

1. Stock and existing native/imported H3 paths use resolved request and source
   geometry, adapters, Veda facts, and selected memory strategy before tensor
   materialization. A fitting deferred request reaches execution without being
   rejected by the fixed 124-frame catalog baseline.
2. Cold and queued checks use consistent host/device units and allocation
   composition, recheck the current budget, and reject impossible requests
   before heavy allocation. Existing source validation remains enforced.
3. The generated request and replayed recipe preserve all user settings and
   audio; any refusal names the actual limiting pool/phase and supported
   remedies, including offload, without misleading q4 tier advice.

Expected paths: SceneWorks `crates/sceneworks-worker/src/video_jobs/{minimax_h3.rs,
wan.rs}`, `video_admission.rs`, `vram_gate/video.rs`, and their existing request
tests. Change catalog metadata only where the new behavior requires it; do not
overwrite the historical measurement with an estimate. Reuse shared admission
and native source contracts rather than adding a separate H3 planner or bypass.

### H3-5 Integrate the inference pin and prove long clip memory targets

**Goal:** Deliver the two repositories together and establish actual fit at the
reported duration. Inherits E1 through E6.

Acceptance criteria:

1. The integrated SceneWorks request completes at 1344Ã—768 / 345 frames / q4
   with Veda enabled under the approximately 95 GiB available workstation
   envelope, preserving requested content settings and producing valid video
   and generated audio for the requested duration.
2. A configuration completes at the same geometry within a **24 GiB total
   device-memory envelope including reserve**, using real weights and bounded
   CUDA execution. Forced dense fallback also completes within an admitted
   envelope. Observed whole-process/device peaks cover loading, conditioning,
   denoising, and decoding; a reduced preflight budget alone is not fit proof.
3. Final inference and SceneWorks heads pass applicable CI and integrated
   acceptance; SceneWorks main pins the accepted inference-main commit, and
   results identify revisions, device, settings, limits, peaks, and outputs.

Use one terminal campaign for these outcomes, reusing bounded numerical,
cancellation, and contract evidence from earlier items. Recover the original
job or recipe for its remaining settings when available; otherwise record the
chosen prompt, seed, steps, and sparse controls explicitly. The bounded campaign
uses the workstation normal-Veda case, a 24 GiB case with forced dense fallback,
and one focused normal-Veda24 GiB render. The third render establishes normal
sparse fit and supplies complete media diagnostics against the workstation's
resident Veda execution: identical creative settings, positive sparse calls on
both, different executed residency policies, and actual video/audio comparison.
E5 additionally requires matched-input component parity through the affected
production paths and dtypes; independent successful renders are not parity proof.
The forced-dense render remains a separate fallback/envelope proof. An actual 24 GiB
GPU or an enforceable allocation cap plus whole-device peak observation can
establish the memory envelope; a Blackwell run must not be reported as RTX 3090
hardware or performance validation.

Establish the 24 GiB proof mechanism before the terminal campaign. Use a real
24 GiB device or an enforceable allocation cap covering the process's CUDA
allocation routes, with whole-device peaks including context, reserve, and
other processes. Record the enforcement mechanism and its coverage.
`SCENEWORKS_CUDA_VRAM_CAP_GB` changes admission budgets only; sampled telemetry
with cooperative cancellation also does not enforce an allocation limit. An
uncapped larger-card run cannot establish this envelope.

Record Veda sparse-to-dense fallback and fused-unavailable materializing
fallback as distinct executed routes; the first can still use FlashAttention.
Reuse focused runtime validation for the materializing route, with its selected
query chunk sizes and byte bounds. Add a full render only when the existing
anchor cannot establish the required envelope. All inference computation stays
in separate dedicated validation workflows.

Check video dimensions, frame count, duration, changing nonblack frames, and
the generated audio stream. Retain full decoded RGB and native-rate stereo
comparison diagnostics for the two normal-Veda anchors, including their
job/media hashes and executed policy witnesses. The sc-26691
`maximumBaselineDelta:0.001` measures sampled unchanged-repeat stability; it is
not an established full-frame, per-channel resident/window tolerance. Do not
promote it to that role or choose a replacement threshold after seeing outputs.

Verify equivalent-route numerical semantics with identical fixed inputs through
the affected production-dtype components, including real q4/Turbo projections,
adapters, Q/K preprocessing, TE/DiT and audio paths as affected. Hold the CUDA
workspace policy equivalent when comparing residency, and isolate default versus
explicit workspace as a separate diagnostic. A workspace-policy difference is
an unproven confound until a component witness establishes its effect. Keep the
existing exact Q/K contract and exact Veda mask/score assertions for common
inputs. Existing small F32 DiT parity does not certify production BF16; no
general BF16 projection/FFN bound is established by it. Locate and explain the
first differing intermediate, enforce applicable source-defined component
contracts, and repair violations. Do not infer a final-media bound from those
contracts or declare the observed media drift acceptable without this evidence.

Preserve campaign 42243645213 (consumer `c8815c7cd`, inference `2ff035382`) as failed
under its predeclared harness criterion. Its three successful fit/media children
and cleanup remain evidence
with their original source/device identities. The measured video RMS 0.09263595
and later audio RMS 0.01594603 remain divergence diagnostics, not accepted
equivalence or permission to loosen a threshold. Matched control 42335898368 now
reproduces the enforced 24 GiB output's encoded hash and zero full RGB/stereo
differences, alongside the accepted production-path component evidence. The
legacy numeric regression values remain unchanged. PR3331 and the feature
acceptance report separately record final-head checks and protected delivery.
Record host peak separately from device peak and account for other GPU
processes. Timing is reported to expose transfer overhead, with no invented
latency target. Failed anchors require a focused diagnosis and repair, not a
larger routine grid or an unverified claim of fit.

## Execution and integration

H3-1 and H3-2 are independent implementation fronts. H3-3 joins their contracts,
H3-4 performs the product cutover, and H3-5 closes the two-repository delivery.
Related items may share reviewed PR batches and CI runs while retaining separate
acceptance evidence. Refresh affected source and reconcile the live feature,
items and overlapping work before implementation.

Follow [FEATURE_DEVELOPMENT.md](../../FEATURE_DEVELOPMENT.md) for mirrored feature
branches, review, inference merge ordering, final pin integration, and required
CI. Keep temporary work, builds, and evidence on D:. Discover the live runner
pool and physical GPU ownership before GPU work using
[Host and Runner Guidance](../host-and-runner-guidance.md). Keep existing device
restrictions and concurrency. Run the terminal campaign once after review and
stable integration; do not reinstate disabled pin-keyed gates or demand renewed
measurements for provenance changes.

This work optimizes the existing H3 CUDA feature. It does not introduce a Python
sidecar, require different checkpoints, reduce clip settings automatically,
replace MLX streaming, build a new memory framework, or publish a release. The
24 GiB target can require substantial host RAM and PCIe transfers; admission must
check that host requirement explicitly.

## Source references

- SceneWorks [stock H3 cold gate](../../crates/sceneworks-worker/src/vram_gate/video.rs),
  [shared video admission](../../crates/sceneworks-worker/src/video_admission.rs),
  [H3 worker path](../../crates/sceneworks-worker/src/video_jobs/minimax_h3.rs), and
  [catalog measurements](../../config/manifests/builtin.models.jsonc).
- Pinned inference [CUDA H3 pipeline](https://github.com/SceneWorks/inference/blob/178eb45e5ad986d2a7236c8f6b800139675baa31/crates/media/candle-gen/candle-gen-minimax-h3/src/model.rs),
  [CUDA strategy contract](https://github.com/SceneWorks/inference/blob/178eb45e5ad986d2a7236c8f6b800139675baa31/crates/media/candle-gen/candle-gen-minimax-h3/src/memory_strategy.rs),
  [Veda workspace pricing](https://github.com/SceneWorks/inference/blob/178eb45e5ad986d2a7236c8f6b800139675baa31/crates/contracts/gen-core-h3/src/veda/workspace.rs),
  and [MLX H3 streaming precedent](https://github.com/SceneWorks/inference/blob/178eb45e5ad986d2a7236c8f6b800139675baa31/crates/media/mlx-gen/mlx-gen-minimax-h3/src/memory_strategy.rs).
- [Miowtion](https://github.com/veda-sparse/Miowtion) and the
  [Veda deployment recipe](https://huggingface.co/Veda-Sparse/Minimax-H3-T2VA-Veda-8NFE-600Step-Preview/blob/9a1fd3a41b4a754a7886e64e82edbddf599fd1bd/AGENTS.md)
  demonstrate the low-VRAM design: host-resident streamed blocks, sparse/fused
  attention, and MLP chunking. Their results motivate the implementation; they
  do not establish SceneWorks fit or identical checkpoint/kernel behavior.
