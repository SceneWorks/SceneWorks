# The shared memory-strategy pipeline — target design

**Status:** proposal, not ratified. Supersedes nothing until agreed.
**Reference implementation:** MLX `z_image_turbo`. Every other model/backend refactors onto this.
**Scope:** the epic remains the 53 image catalog entries. The *vocabulary* is lane-neutral by design
(§13) so video and audio can adopt without a second contract; whether the ladder itself transfers is
unproven and explicitly not claimed here.
**Relationship to existing docs:** extends [`memory-strategy-contract.md`](memory-strategy-contract.md) (SC-15449) and epic [15448](https://app.shortcut.com/trefry/epic/15448).

---

## 0. The one-sentence goal

**One pipeline decides, once, per request, which quality-preserving strategy runs — and every fact
that decision rests on is declared in exactly one place.**

Everything below is a consequence of that sentence. Where today's code has a fact written in three
places, or a second pipeline making the same decision differently, this design removes the duplicate
rather than reconciling it.

### What "candle" means in this document

Candle is an **adopter**, not a constraint. No current candle behaviour justifies a design choice
here. Where candle is mentioned it is to scope migration work, never to shape the contract. The same
applies to any MLX family other than z-image: the reference implementation is built first, proven,
and then everything else moves onto it.

---

## 1. What a rung is, and what it is not

A **rung** is a *quality-preserving execution strategy* that bounds a specific memory peak.

Two properties are definitional:

1. **It preserves the output.** Same precision, same schedule, same seed, same conditioning, same
   output geometry. Byte-identical where the operation permits; a documented deterministic tolerance
   where it does not.
2. **It is an execution choice, not a weight choice.** It changes *how* a render runs, not *what
   weights exist*.

Anything failing either test is **not a rung** and must not enter the ladder:

| thing | why it is not a rung | where it belongs |
|---|---|---|
| Quantization tier (q4/q8/bf16) | changes precision | user-facing creative choice, chosen before memory selection |
| Control-branch packing | load-time weight transformation | tier integrity — §7 |
| Resolution reduction | changes the requested output | nowhere — §6 |

### The two rung families

The ladder currently flattens two structurally different things into one ordered enum. The design
makes the distinction explicit, because it determines what each rung must declare.

| family | rungs | bounds | lifetime of the thing it bounds |
|---|---|---|---|
| **Scratch-bounding** | 2 Bounded decode, 3 Bounded attention | activations, tiles, score matrices | born and dies inside one request |
| **Residency-bounding** | 1 Staged residency, 4 Bounded transformer residency | materialized weights | persists across requests unless explicitly released |

Scratch-bounding rungs have no preconditions — they can always run. Residency-bounding rungs must
be able to *release* what they bound, which is the entire source of today's load-time/request-time
seam.

---

## 2. The four facts every rung declares

Each rung declares exactly four things, each in exactly one place, each derived everywhere else.

| fact | answers | owner | scope | shape |
|---|---|---|---|---|
| **Availability** | Can *this loaded model* execute this rung at all? | provider, built per `LoadSpec` | static, per load | `Implemented` / `StructurallyNotApplicable` / `Missing` |
| **Prerequisites** | Which rungs must also engage for this one to mean anything? | the shared contract, declared per rung | static | explicit edge set — **never** derived from enum order |
| **Engagement** | Is this rung on for *this request*, and with what parameters? | worker selection → `GenerationMemory` | per request | one boolean + its owned parameters |
| **Cost rank** | What order does the selector try rungs in? | worker | policy | the normative order of `MemoryStrategy::ALL` |

### The invariant that makes it checkable

> **A rung — or a parameter of one — that cannot move the REQUEST peak is never selected by
> default, and never recorded as a saving.**

The unit is the **request** peak, not a phase peak. A strategy that halves one phase while a
different phase still binds the request has saved the user nothing, and a calibration row claiming
otherwise is a false green. SC-15794 is the worked example: rung 4's text-encoder scope cuts the
z-image conditioning phase 2.718 → 1.455 GiB (−46.5%) and moves the request peak by **0.0%**, because
decode binds every tier at ~4.365 GiB. The capability is real and published; the default is not it.

Availability is the mechanism that enforces this. A typed `Error::Unsupported` at the provider is a
**backstop for a selector bug**, not a control-flow path — if it fires in production, the selector
is wrong, and that is a defect rather than a fallback.

The corollary is the honesty rule that motivates the whole epic: a rung that silently degrades
writes a calibration row for a run that saved nothing. That is a false green, and it is worse than
an error.

---

## 3. The ladder, restated

| # | rung | family | bounds | availability facts | engagement knob | parameters |
|---|---|---|---|---|---|---|
| 0 | Resident | — | nothing | always | *(absence of all others)* | — |
| 1 | Staged residency | residency | weights across phases | model has a separable phase-A component | `stage_residency` | — |
| 2 | Bounded decode | scratch | decoder scratch | decoder is tileable | `tile_vae_decode` | `decode_tile_edge`, `decode_overlap` |
| 3 | Bounded attention | scratch | attention scratch | attention is chunkable | `chunk_attention` | `attention_chunk_size` |
| 4 | Bounded transformer residency | residency | trunk weights | weights are **re-openable** from a snapshot | `stream_transformer_blocks` | `transformer_window_size`, `transformer_window_component` |

Rung 4's `transformer_window_component` (`Dit` / `TextEncoder` / `Both`, SC-15794) is a **parameter
with a published candidate domain**, exactly like `decode_tile_edges` and `attention_chunk_sizes` —
not a new axis. Availability stays one verdict per rung; which transformer the window applies to is
chosen within it. This is the shape any future component scope should take, including Wan's MoE
expert residency (§14).

### The prerequisite graph — explicit, and small

```
rung 1  →  (none)
rung 2  →  (none)
rung 3  →  (none)
rung 4  →  requires rung 1 ENGAGED in the same request
```

That is the entire graph. Two things follow:

**Rungs 2 and 3 depend on nothing.** Bounding decode scratch or attention scratch is correct whether
or not the text encoder was shed. Today's numeric-order walk asserts otherwise, and that is the
latent trap described in §5.

**Rung 4's single edge is a correctness constraint, not a cost ordering.** A block window bounds
nothing if the trunk is already materialized — it *adds* a copy on top. This is the fact currently
encoded three separate times (§9).

### Cost-order defaults are a different thing, and must stay separate

A rung-4 selection today also sets `tile_vae_decode` and `chunk_attention`. That is **not** a
dependency — it is the selector's cost-order policy: *if you needed the most expensive rung, you
would have taken the cheaper ones on the way up.* The epic already words it as defeasible:

> "Strategies are cumulative unless a provider documents and verifies a cheaper equivalent composition."

So the design keeps them apart:

- **Prerequisite** — enforced by the contract. Violating it is an error.
- **Cost-order default** — applied by the selector. A provider may publish a verified cheaper
  composition and opt out, with evidence.

Conflating these is what makes today's `validate_selection` reject compositions that are perfectly
correct.

---

## 4. Reshaping rung 1

### What is wrong today

Rung 1 is the only rung with **no engagement knob**. `GenerationMemory` has three booleans — one
each for rungs 2, 3 and 4 — and nothing for rung 1. Staging is gated on `Residency::is_sequential()`,
which comes from the *load-time* `LoadSpec::offload_policy`.

Three consequences, all live:

1. **Selecting rung 1 per request on a `Resident`-loaded generator silently does nothing.**
   `z_image_generation_memory` maps rung 1 to an all-false `GenerationMemory` for exactly this
   reason. There is nothing to turn on.
2. **It is inert but declared `Implemented`.** Rung 4 declares `Missing` off the *identical* fact
   (`streamable` requires `OffloadPolicy::Sequential`). Same input, opposite answers, one file apart.
3. **Residency policy is fixed for the life of the generator.** A warm generator cannot serve a
   constrained request, and a staged generator pays re-load on every request forever — including the
   ones that had headroom.

### The target: rung 1 becomes a full member

**Residency becomes a per-request decision.** `GenerationMemory` gains `stage_residency: bool`, and
rung 1 declares availability and engagement like every other rung.

This is a smaller change than it sounds, because the machinery already exists. `Residency` today is
a two-variant enum:

```rust
Inner::Resident(pair)       // holds built components
Inner::Sequential(loaders)  // holds two closures; builds per run, frees after
```

Under `Sequential`, the heavy bundle is **already** rebuilt on every `run`. The per-request capability
is present and gated behind a load-time enum. The target is the *union* of the two variants:

```rust
pub struct Residency<Text, Heavy> {
    loaders: SeqLoaders<Text, Heavy>,        // always held — cheap, two boxed closures
    warm:    Option<ResidentPair<Text, Heavy>>, // held iff currently warm
}
```

Per request:

| `stage_residency` | behaviour |
|---|---|
| `false` | if `warm.is_none()`, run the loaders and **store** the products; borrow and run the resident path |
| `true` | if `warm.is_some()`, **drop it** and `clear_cache()`; run the staged loader path; leave `warm` empty |

### Eviction is not a new cost — it is the cost sequential loading always had

Staging frees nothing unless the staged components are released. That has always been true: it is
what `mempolicy` means by "only the cross-request weight cache is lost." Request-scoping does not
add a cost; it changes **who pays and when**.

Today a load-time policy gives exactly two fixed regimes:

| regime | every request | can any request stage? |
|---|---|---|
| `Resident` | warm, no reload | no |
| `Sequential` | reloads | yes — but *all* of them, always |

Request-scoped residency is Pareto-better than both:

- **vs `Resident`** — identical when no request stages; strictly more capable when one does.
- **vs `Sequential`** — identical when every request stages; strictly cheaper when some don't.

So the ledger only improves. What is genuinely new is **variance, not cost**: the reload's *timing*
becomes data-dependent. A 1024² (warm) → 2048² (staged) → 1024² sequence pays a reload on the third
render that fixed-`Resident` would not. Same cost category, unpredictable placement.

That is a telemetry and explainability requirement, not a design objection:

- The worker records cache eviction alongside the selected strategy, so "why was that render slow"
  is answerable.
- The existing contract rule already covers the safety half: *"each transition must either leave the
  warm cache reusable or invalidate it atomically."*

**Two things the union type must not lose.** Both already exist in the current drivers:

1. **Materialize before shed.** Under MLX's lazy graph an un-evaluated output keeps its producer's
   weights referenced, so a drop before the `eval` frees nothing. `run_staged` encodes this at both
   phase boundaries and it must survive the refactor intact.
2. **Drop before load.** Evicting warm and loading staged must not overlap, or the transition peak
   exceeds both regimes it sits between.

**Concurrency is not a hazard.** `LoadSpec`'s own documentation records that the crate runs
single-device because the MLX default device is not thread-safe, and the worker serializes jobs per
thread. Eviction cannot race a live borrow.

### What `LoadSpec::offload_policy` becomes — **ignored by image providers, not deleted**

An earlier draft of this document said "deleted." That is wrong, and the audit says why:
`offload_policy` has consumers outside the image lane.

| lane | crates |
|---|---|
| image | ~20 mlx-gen / candle-gen crates |
| **video** | `mlx-gen-wan`, `mlx-gen-ltx`, `candle-gen-wan`, `candle-gen-svd` |
| **audio** | `candle-audio-stable-audio-3` |

Deleting the field is a cross-lane change well outside a 53-image-entry epic.

**The target instead:** for an image provider that has adopted the ladder, `offload_policy` is
**ignored** — rung 1's authority is `GenerationMemory::stage_residency` and nothing else. The field
remains in `LoadSpec` for the video and audio lanes, unchanged, and is deleted only when they
migrate.

This keeps one authority per fact. The field is *not* retained for image providers as a "warm-up
hint" — that would re-create exactly the two-place ambiguity this design removes. Deferring the load
entirely is also strictly better: nothing loads until a request has been selected, so model
switching gets faster.

### Amendment (SC-15794 / PR #323): the loader must take residency as an argument

PR #323 makes the loader choose the **form** of a component from the load policy:

```rust
// z-image loader.rs
let streamable = matches!(spec_text.offload_policy, OffloadPolicy::Sequential);
load_text_encoder_only(root, spec_text.quantize, streamable)
```

A streamable encoder holds no resident layers, so it is ~2× slower under `Resident` (measured
250/148/165 ms resident vs 445/428/419 ms streamed) — a real regression the PR pins a test against.
The gate is correct **given a load-time policy**. Under request-scoped residency it cannot stand: the
loader runs before the request exists.

**The fix is already precedented in the same type.** `Residency::sequential` takes
`load_heavy: impl Fn(bool) -> Result<Heavy>` — the `bool` is `use_pid`, a *per-request* flag threaded
into a loader closure. `streamable` becomes a second such argument:

```rust
load_text:  impl Fn(bool /* streamable */) -> Result<Text>
load_heavy: impl Fn(bool /* use_pid */, bool /* streamable */) -> Result<Heavy>
```

The request's selected strategy decides the form; the loader builds it. No form is chosen before a
request exists, and the 2× resident regression is structurally unreachable because a request that
did not select a streaming scope never asks for a streamable component.

**The warm-cache consequence is the one already declared.** A warm *non*-streamable encoder cannot
serve a request that selects the text-encoder scope, so that request evicts and rebuilds — the same
eviction rung 1 already declares, for the same reason. No new cost category.

### What this unlocks for rung 4

Rung 4's availability currently carries two facts. With residency per request, one of them
disappears:

```rust
// today
let streamable = matches!(spec.weights, WeightsSource::Dir(_))
    && matches!(spec.offload_policy, OffloadPolicy::Sequential);

// target
let streamable = matches!(spec.weights, WeightsSource::Dir(_));
```

The `Sequential` clause becomes the declared prerequisite edge `rung 4 → rung 1`. The provider stops
hand-maintaining it, and `resolve_block_window`'s check reverts to the pure backstop it is already
documented as being.

**One capability change removes the load-time coupling from both residency-bounding rungs.**

### The mutation seam — settled by the existing trait

`run_staged` is `&self` today, and a per-request warm cache needs to mutate. This is not a judgment
call: `Generator::generate(&self, …)` is the trait every provider in the workspace implements, so
`&mut self` would mean changing that signature everywhere — a large blast radius unrelated to this
epic.

**Interior mutability**, therefore: `warm: Mutex<Option<ResidentPair<Text, Heavy>>>` inside
`Residency`. It is uncontended in practice because the worker already serializes jobs per thread,
and it keeps `Residency` `Sync` without touching `Generator`.

---

## 5. Prerequisites become explicit

### The problem

`gen_core::validate_selection` derives dependency from the enum's numeric order:

```rust
for rung in MemoryStrategy::ALL.into_iter().filter(|rung| *rung < selection.strategy) {
    if matches!(support, Some(Missing) | None) { return Err(Error::Unsupported(...)); }
}
```

Three sites read the order, and they encode **two different policies**:

| site | policy on a `Missing` lower rung |
|---|---|
| `gen_core::validate_selection` prerequisite walk | **fatal** — `Error::Unsupported` |
| `gen_core::validate_selected_parameters` | tolerant — guards on `implemented(...)`, stops requiring that rung's parameters |
| `sceneworks_worker::memory_strategy::select_strategy` | tolerant — `continue`s past any non-`Implemented` rung, keeps looking |

The selector and the validator **already contradict each other**: `select_strategy` will return rung
2 on a contract where rung 1 is `Missing`, and `validate_selection` will hard-error on exactly that
selection. It is unreachable today only because no provider ever declares a lower rung `Missing`.

Any honest declaration at the bottom of the ladder trips it.

### The fix

Replace the order-derived walk with the declared graph from §3.

### As shipped (SC-15805, inference PR #325 / SceneWorks PR #1956)

- `MemoryStrategy::requires() -> &'static [MemoryStrategyPrerequisite]` publishes the graph, and
  `MemoryStrategyCapability::requires()` exposes it per rung — empty for rungs 1, 2, 3;
  `[StagedResidency]` for rung 4.
- It is **contract-owned, not a provider-settable field**, which is the one deviation from the
  sketch above. Three reasons: the story's own goal is to move the fact *out of* a provider's private
  condition and *into* the contract; a settable field is a false green waiting to happen, since a
  provider that never sets it silently drops the edge; and the rung-4 edge is arithmetic, not policy,
  so no provider may opt out of it. Providers still control the graph's **effect** — declaring rung 1
  `Missing` makes rung 4 unselectable, which is the honest outcome.
- The engagement-vs-availability distinction is a **type**: `MemoryPrerequisiteScope::EngagedInSameRequest`.
  The refusal message says so, and states that availability does not satisfy it.
- `validate_selection` walks that graph, not `< selection.strategy`.
  `StructurallyNotApplicable` on a prerequisite rung satisfies it **vacuously** — it asserts the
  architecture has no such component, which is not evidence the trunk is eagerly materialized. The
  result is a strict *narrowing* of the old fatality set: nothing that validated before stops
  validating.
- The cost-order default survives, named and separated: `MemoryStrategy::engages` is the pure policy
  and `MemoryProviderContract::engages` is that policy intersected with what the provider implements.
  The intersection is where defeasibility lives — a rung the provider does not implement is not
  engaged, so its parameters stop being *required* rather than the selection being refused.
- `validate_selected_parameters` and `select_strategy` needed no behavioural change, as predicted.
  Three *other* sites that derived engagement from the enum's order were routed through the new seam:
  `mlx-gen-z-image`'s route-aware decode check, `sceneworks_worker::mlx_fit_gate::memory_for_selection`,
  and `validate_selected_parameters`' own `requires_*` flags. After this, **no ordering comparison on
  `MemoryStrategy` remains anywhere in either repo.**

**Correction to the problem statement above.** `select_strategy` already funnels every candidate
through `contract.validate_selection` (`candidate_exclusion`), so the two layers could never return
*different* selections and the "typed failure at generate time" was not reachable. The contradiction
surfaced instead as **over-refusal**: on a contract with rung 1 `Missing` the selector silently
excluded rungs 2 and 3 as `Invalid` and fell back to a more expensive rung or rejected outright. Same
root cause, same fix — but a "both layers agree" test would pass vacuously on the old code (both
refusing is agreement too), so the shipped test asserts the expected verdict per rung as well.

---

## 6. Resolution reduction is deleted, not relocated

### Position

**Never silently change the requested output.** If the requested geometry cannot be rendered, the
system either does not offer it, or refuses it with a truthful, measured alternative. It does not
quietly produce something smaller.

This is already the epic's stated non-negotiable — *"If nothing fits, reject before OOM with
truthful, measured alternatives"* — and the contract doc already constrains the advice:
*"Advice may name only a lower geometry or strategy that has verified evidence."*

`Lever::ResolutionReduction` predates both and contradicts them.

### It is live today, on one route

`mlx-gen-krea/src/model_control.rs:325` calls `plan_control_resolution`, which returns a smaller
16-aligned `(render_width, render_height)` when the requested size does not fit. On substitution it
emits an `eprintln!` — **stderr only**. Not the API response, not telemetry, not the UI.

So a `krea_2_turbo_control` request for 1024² can return 768², and the requester is never told.
The code comment calls this "never a SILENT capability drop" because it prints to stderr; from the
user's side it is exactly a silent drop.

### The replacement — three surfaces, none of which alter the request

**1. Do not offer what cannot be rendered.** Resolution choices presented in the UI are filtered by
the same measured evidence the selector uses. This is the existing SC-15613 work (backend-aware web
memory surfaces) — today four independent copies of one transient-headroom measurement drive these
surfaces, and none of them read the calibration contract.

**2. Refuse, with a measured alternative.** A request for an unreachable geometry is rejected before
the render with a typed error naming a geometry that has *verified* evidence — not an estimate, not
a guess. `plan_control_resolution`'s existing `Err` arm is already this shape; it just fires two
scales too late.

**3. Record it.** The rejection reason, the requested geometry, and the named alternative go to
telemetry, so "users keep asking for a size we can't serve" becomes visible rather than absorbed.

### Disposition

- `Lever::ResolutionReduction` — deleted.
- `plan_control_resolution` — deleted. Its `fits()` predicate survives as a **feasibility check**:
  same arithmetic, but it answers *"can we?"* and never answers *"then do this instead."*
- `CONTROL_RESOLUTION_SCALES`, `scale_render_dim` — deleted.
- The `eprintln!` substitution notice — deleted along with the substitution.

---

## 7. Branch quantization is not a lever — it is how packing works

### Position

**No component is resident above the user's selected tier unless a declared, measured exception says
otherwise.** Choosing q8 or q4 *is* a memory decision; carrying any component at bf16 defeats it.

This is [SC-15799](https://app.shortcut.com/trefry/story/15799), and it is already the written norm
in the manifest: *"every component is packed at the chosen tier."*

### Why `BranchQuant` is a workaround wearing a lever's clothes

`mempolicy`'s own doc describes it as *"pack the control branch **to the base quant tier** at load
time."* That is the invariant, implemented as an escalation step. Reading it against the numbers:

- q8 is measured **near-lossless** and saves **8.4 GiB**. There is no quality argument for defaulting
  to bf16 when the user asked for q8.
- It is load-time-only — *"it cannot be re-packed mid-render"* — which is what an install-time
  artifact choice looks like, not a runtime lever.
- It sits **last** in the escalation order, so a constrained machine engages residency and decode
  tiling first to claw back single-digit GiB while 8.4 GiB of unrequested precision sits in the
  branch the entire time.

Qwen already publishes its ControlNet per tier (q4 0.99 GB / q8 1.87 GB / bf16 3.51 GB), so the
compliant shape ships today.

### The one real carve-out

q4 branch quant is measured *"pose-locked; non-pose details drift."* So a q4 base floors its branch
at **q8** — a declared, measured exception, which is exactly what the invariant permits. It is not a
hole in the rule; it is the rule working.

### Disposition

- `Lever::BranchQuant` — deleted.
- `should_quantize_control_branch` — deleted. The branch's tier is decided by the installed artifact.
- The q4→q8 floor — declared in the manifest with its quality evidence attached.

---

## 8. `mempolicy` is deleted entirely

Not thinned. Deleted. Every one of its four levers has a home elsewhere, and its decision function
duplicates the shared selector.

| `Lever` | disposition |
|---|---|
| `SequentialResidency` | **is** rung 1 (§4) |
| `VaeDecodeTiling` | **is** rung 2 |
| `BranchQuant` | not a lever — tier integrity (§7) |
| `ResolutionReduction` | not a lever — refuse or don't offer (§6) |

With all four gone, `plan_memory_adaptation`, `StagePeaks`, `LaneLevers` and `MemoryPlan` have
nothing left to decide. Their job — *given a budget and shape-derived peaks, pick the lightest
sufficient setting* — is `sceneworks_worker::memory_strategy::select_strategy`'s job. The contract doc
already forbids the duplication:

> "Providers may reject a selected strategy defensively, but must not contain a second least-cost
> selector."

### What survives, and must not be lost

`mlx-gen-krea/src/memory.rs` is **two things welded together**, and only one of them is redundant:

- **The escalation policy half** — the `plan_memory_adaptation` call sites. Deleted.
- **The cost model half** — first-principles param counts re-fit on real weights on a 128 GB Metal
  Mac (SC-11847), with measured slopes ~44 (bf16) / ~61 (q4) B/(token·hidden) for denoise and
  ~5211 B/px for decode, under an over-predict/never-under-shoot convention. **This is expensive,
  calibrated evidence and it survives**, retargeted to produce provider estimates that feed
  `select_strategy` as ordinary candidates.

Deleting the cost model along with the policy would discard the most costly artifact in the module.

### Blast radius

`plan_memory_adaptation` has exactly one consumer: `mlx-gen-krea/src/memory.rs`. The "backend-neutral
escalation core both backends share" is used by one model on one backend — which is itself the
argument that the shared selector, not this, is the pipeline.

---

## 9. What makes it obviously correct

### The single-encoding audit

The test this design holds itself to is countable: **every fact is declared once and derived
everywhere else.** Current scores:

| fact | encodings today | where | target |
|---|---|---|---|
| rung 4 needs a non-materialized trunk | **4** *(was 3)* | `streamable`, numeric-order walk, `resolve_block_window`, **+ the loader's `streamable` flag (PR #323)** | **1** — prerequisite edge |
| rung 1 is inert under `Resident` | **0** | prose in a module doc; declared nowhere | **0 needed** — the condition ceases to exist |
| the escalation order | **2** | `MemoryStrategy::ALL`, `mempolicy::Lever` | **1** |
| the control branch's tier | **2** | installed artifact, `BranchQuant` lever | **1** |
| the attention budget constant | **2** | `gen_core::attention_budget`, candle's copy | **1** (SC-15796) |
| what a `Missing` lower rung means | **2 policies / 3 sites** | fatal vs tolerant | **1** |

A fact scoring >1 is a place two implementations can disagree without the compiler noticing. A fact
scoring 0 is a rule nothing enforces.

### Test discipline

Two failure modes this codebase has already produced, both of which the reference implementation
must actively guard against:

**Mutation-check every test that pins a shared contract.** SC-15793's own kernel-binding test was
near-vacuous — it iterated the shared table, but only one row was small enough to run in a unit test
and that row happened to be non-chunking, so it stayed green while the kernel forked its arithmetic.
The mutation run is what caught it.

**A test asserting a default value is a false green.** If the assertion passes because the field was
never set, it proves nothing. Every conformance assertion must be shown red under a deliberate
perturbation.

### Evidence honesty

Unchanged from the epic, restated because this design depends on it: a green result covers only the
parameter range exercised, dynamic evidence is authoritative only from suitable hardware, and a
calibration-fingerprint mismatch makes evidence stale. **A rung that engaged but saved nothing must
never produce a calibration row.**

---

## 10. Selection, end to end

```
1. REQUEST arrives with geometry, tier, mode, overlay, seed.
        │  tier is a creative choice, already made — the pipeline never changes it
        ▼
2. FEASIBILITY. Is this geometry renderable on this machine, by ANY rung, with verified evidence?
        ├─ no  → REFUSE with a measured alternative that has verified evidence.  ✋ STOP
        └─ yes → continue.        (never substitute a different geometry)
        ▼
3. CANDIDATES. Provider submits a per-rung estimate from its cost model.
        ▼
4. SELECT. Worker walks the normative cost order 0→4, skipping rungs that are
   unavailable or lack verified evidence, and takes the cheapest that fits the live budget.
        ▼
5. VALIDATE. Contract checks the selection: availability, declared PREREQUISITES
   (not enum order), and parameter domain.
        │  a failure here is a SELECTOR BUG, not a fallback
        ▼
6. ENGAGE. Selection maps to GenerationMemory:
        stage_residency / tile_vae_decode / chunk_attention / stream_transformer_blocks
        + the parameters each rung owns.
        ▼
7. RUN. Provider executes. Cancellation and error paths synchronize and release
   active phases and windows.
        ▼
8. RECORD. Telemetry: requested + effective tier, selected strategy and parameters,
   phase peaks, live budget, cache eviction, evidence revision, exclusion reasons.
```

Step 2 is the change that makes §6 real: feasibility is answered *before* selection, and its only
two answers are "proceed at the requested geometry" and "refuse."

---

## 11. Migration order

Sequenced so that nothing is measured twice and no story is blocked on a decision it cannot make.

| # | work | why here |
|---|---|---|
| 0 | **Rename to the lane-neutral vocabulary** (§13) — `gen_core::memory_strategy`, `Memory*`, `MEMORY_CALIBRATION_ABI`, `harnessVersion` → `sceneworks-memory-v3` | Purely mechanical, and it invalidates prior evidence via the `harnessVersion` const. Committed records today: **zero**. After the calibration wave: hundreds. |
| 1 | **Explicit prerequisites** — replace the numeric-order walk with the declared graph | Unblocks everything. Rung 1 cannot be declared honestly while the walk treats it as a hard prerequisite for rungs 2–4. |
| 2 | **Rung 1 reshape** — `Residency` union with a `Mutex` warm slot, `stage_residency` on `GenerationMemory`, image providers stop reading `LoadSpec::offload_policy` | The core capability. Simplifies rung 4's availability as a side effect. The field itself stays for the video/audio lanes. |
| 3 | **Feasibility + refusal** — delete `plan_control_resolution`, keep its predicate as a feasibility check | Independent of 1–2; removes the live output-substitution defect. |
| 4 | **Tier integrity (SC-15799)** — branch follows the base tier, q4→q8 floor declared | **Must precede control-lane calibration.** Repacking the branch invalidates every measured control peak and fingerprint. |
| 5 | **Delete `mempolicy`** — retarget Krea's cost model to submit candidates | Requires 1–4; all four levers must have homes first. |
| 6 | **z-image reference certification** — all rungs, all tiers, mutation-checked, evidence promoted | The template is not a template until it is proven. |
| 7 | **Refactor adopters onto it** — other MLX families, then candle | Only after 6. |

Step 4's placement is not cosmetic: calibration is the epic's stated dominant cost, and repacking
after measuring means paying for those measurements twice.

---

## 12. Decisions — resolved

All four are settled. Two were answered by the code, two are recommendations.

### D1 — mutation seam: **interior mutability** *(settled by the code)*

`Generator::generate(&self, …)` is implemented by every provider in the workspace. `&mut self` would
change that trait everywhere for an unrelated reason. So: `warm: Mutex<Option<ResidentPair<..>>>`
inside `Residency`, uncontended because the worker serializes per thread. See §4.

### D2 — `LoadSpec::offload_policy`: **ignored by image providers, not deleted** *(settled by the code)*

The field has consumers in the **video** lane (`mlx-gen-wan`, `mlx-gen-ltx`, `candle-gen-wan`,
`candle-gen-svd`) and the **audio** lane (`candle-audio-stable-audio-3`). Deleting it is a cross-lane
change outside this epic's scope. Image providers that adopt the ladder ignore it entirely; the field
is deleted when the other lanes migrate. See §4.

### D3 — is eviction refusable? **No, not in production** *(recommendation)*

A "stage but keep the warm cache" mode saves nothing by construction — the staged components are
still resident. Engaging rung 1 under it would write a rung-1 calibration row for a run that bounded
nothing, which is precisely the false green the invariant in §2 exists to prevent.

If A/B measurement needs the arm, it is a **calibration-only field**, not a production lever.
`GenerationMemory::calibration_error_phase` is the existing precedent: a field production selectors
leave `None`, so every ordinary request is a `None` comparison away from being unaffected.

### D4 — may a refusal name a tier? **No — geometry or strategy only** *(recommendation)*

Two independent reasons:

1. The contract already constrains it: *"Advice may name only a lower geometry or strategy that has
   verified evidence."* Tier is not in that list.
2. Tier is a creative choice. A memory pipeline that suggests one — even as information — becomes a
   tier-recommendation engine operating on memory grounds, and the boundary between "informing" and
   "nudging" is not one the code can hold.

Tier availability is an existing, separate surface. Keeping them apart is what stops the memory
pipeline from acquiring an opinion about image quality.

---

## 13. Naming — the vocabulary is lane-neutral

**Decision: drop `Image` from every type, and name the module `gen_core::memory_strategy`.**
**Status: done — SC-15804, paired inference + SceneWorks PRs. This section is now a record of what
was renamed, not a plan.**

Nothing in the ladder is image-specific. Rung 1 sheds a conditioning component; rung 2 bounds decoder
scratch; rung 3 bounds attention; rung 4 bounds transformer residency. Video and audio have all four.
The `Image` prefix records which lane happened to be built first, not what the contract covers.

**The epic's scope does not change.** It remains the 53 image catalog entries. Only the vocabulary
generalizes, so video and audio can adopt without a second contract.

### Why not bare `gen_core::memory`

Five crates already have a `memory` module, meaning two unrelated things: `mlx_gen::memory` is the
MLX budget interface (`safe_budget_gib`, `clamp_budget_to_cap`, `apply_memory_cap_env`), while
`mlx-gen-sam2::memory` is SAM2's *model* memory bank. A third sits one crate away —
`mlx_rs::memory` is the allocator itself (`clear_cache`, `get_peak_memory`). MLX provider files
import the first directly, so `gen_core::memory` would sit next to it in the same `use` block.
`memory_strategy` collides with nothing and says what it is.

The **types** stay bare — `MemoryStrategy`, `MemoryBudget`, `MemoryPhase` — because none of them
collide. The existing bare `Memory*` names in the workspace (`MemoryEncoder`, `MemoryAttention`,
`MemoryTokens`, `MemoryFeatures`, `MemoryProfile`, `MemoryConfig`) are all SAM2/model-bank concepts
in other crates, and `MemoryPlan` is deleted by §8.

### The Rust surface — mechanical

The module exported **36 public items**, every one prefixed `ImageMemory*`, plus
`IMAGE_MEMORY_CALIBRATION_ABI`. The rename was a prefix strip with no judgment calls:

```
gen_core::image_memory              →  gen_core::memory_strategy
IMAGE_MEMORY_CALIBRATION_ABI        →  MEMORY_CALIBRATION_ABI
ImageMemoryStrategy                 →  MemoryStrategy
ImageMemoryProviderContract         →  MemoryProviderContract
ImageMemorySelection                →  MemorySelection
ImageMemoryEvidence                 →  MemoryEvidence
…and 32 more, identically
sceneworks_worker::image_memory     →  sceneworks_worker::memory_strategy
```

`GenerationMemory` and `TransformerComponent` are already lane-neutral and did not change.

Three items outside that 36 carried the same prefix and were renamed with it, because leaving them
would have left the contract half-named: `gen_core::registry::ImageMemoryRegistration` →
`MemoryRegistration` (it *holds* a `MemoryProviderContract`), its builder method
`register_image_memory` → `register_memory_strategy`, and the three `Generator` hooks
(`image_memory_contract`, `image_memory_safety_check`, `begin_image_memory_request`). Provider-local
`IMAGE_MEMORY_REGISTRATION` / `IMAGE_MEMORY_CALIBRATION_FINGERPRINT` constants and the
`{Krea,Mage,ZImage}ImageMemoryScope` types followed the same strip.

### The SceneWorks surface — file and crate names, not payloads

| kind | items |
|---|---|
| crate | `sceneworks-image-memory-adapter` → `sceneworks-memory-adapter` (+ `Cargo.toml`, `Cargo.lock`, workspace manifest, `docker/rust.Dockerfile`, both `[[bin]]` targets, and the `macos-mlx.yml` dispatch input/env/artifact names that reference them) |
| schemas | `packages/schemas/image-memory-{matrix,calibration}.schema.json` → `memory-{matrix,calibration}.schema.json` |
| generated | `docs/generated/image-memory-{matrix.json,matrix.md,calibration-evidence.json}` → `memory-*` |
| config | `config/image-memory-calibration-plan.json` → `config/memory-calibration-plan.json` |
| scripts | `generate-image-memory-matrix.mjs`, `image-memory-calibration-harness.mjs`, their tests, the four `scripts/fixtures/image-memory-provider-*.mjs`, plus the path/identifier citations in `bump-inference.mjs`, `platform-review-contracts.test.mjs`, `split-memory-strategy-stories.mjs`, and the four `package.json` npm scripts |
| tests | `tests/test_image_memory_matrix.py` → `tests/test_memory_matrix.py` |
| docs | `image-memory-strategy-contract.md`, `image-memory-calibration-harness.md`, this file |

**What did *not* change: the JSON payloads.** There were zero `imageMemory*` keys in
`config/manifests/builtin.models.jsonc` or in either schema. No manifest migration, no key rewrite —
the only manifest edit in the whole rename is one `//` comment line, confirmed by diff.

The one `imageMemory` key that existed anywhere was `generatedFrom.sources.imageMemory` in the
**generated** matrix — a source-path map entry emitted by `generate-memory-matrix.mjs`, under a
`generatedFrom` that the matrix schema types as an unconstrained object. It regenerates as
`memoryStrategy`.

**What did change in content**, and it is exactly two things:

1. The schema `$id` URLs — now `https://sceneworks.ai/schemas/memory-*.schema.json`.
2. The `harnessVersion` **const**, asserted twice in the calibration schema:
   `"sceneworks-image-memory-v2"` → `"sceneworks-memory-v3"`, matched by `HARNESS_VERSION` in the
   harness. `scripts/memory-calibration-harness.test.mjs` asserts that a record which is well-formed
   and self-consistent in every other respect is still rejected at the prior version, with a control
   case proving the same record passes at the current one.

### The timing argument, precisely

That `harnessVersion` const is embedded in every emitted record, so bumping it invalidates all prior
evidence. **That is not a cost — it is the mechanism working.** The const is already the epic's
stale-evidence gate, and a vocabulary change is exactly the kind of thing it exists to invalidate.
The rename rides a versioning path that is already designed.

The cost is only the evidence that has to be re-captured, and at the time of the rename that was
nothing:

- `docs/generated/memory-calibration-evidence.json` contained **`"records": []`** — 92 bytes.
- The ~9 real records (1 Candle/RTX, 8 MLX/M5 Max) live in gated CI artifacts, unpromoted.

After the calibration wave there will be hundreds, each bound to a revision pair. **Do this before
any calibration story starts** — it is step 0 in §11 for that reason.

### What video and audio adoption would still need to check

The vocabulary generalizes. Whether the *ladder* does is unproven, and this design does not claim it.
Three things to establish before either lane adopts, flagged so they are not assumed:

1. **Rung 2 for video is temporal as well as spatial.** `gen_core::tiling` already serves candle
   video (ltx / mochi / svd / wan), and the epic already words rung 2 as "spatial or temporal." But
   SC-15325 is a temporal tile-context/overlap defect, so video's rung 2 has a known open issue that
   image's does not.
2. **Wan's MoE expert swap may not be either residency rung.** Swapping experts per step is
   weight-residency bounding, but it is neither "shed a phase" (rung 1) nor "window the trunk"
   (rung 4). The likely answer mirrors SC-15794: a **component scope** on rung 4 rather than a fifth
   rung. That should be decided on evidence, not assumed.
3. **Audio already uses `OffloadPolicy::Sequential`** (`candle-audio-stable-audio-3`), so it has
   rung 1 in the old shape and would need the same reshape §4 describes.

---

## 14. Portability to the video and audio lanes

**Intent:** the same ladder serves video and audio once the image lane is through. This section
records what that requires, so the image work does not quietly foreclose it.

### The mechanism already ports — rung 2 proves it

`gen_core::tiling` — the only rung whose planner is fully shared today — is a port of the
`mlx_video` LTX and Wan tiling references. It carries `temporal_scale`, `causal_temporal` and
`writable_frame_cap`, and its consumers already span both lanes:

| lane | consumers |
|---|---|
| image | `qwen-image`, `sana`, `sdxl`, `z-image` |
| video | `ltx`, `mochi`, `svd`, `wan`, `krea-realtime`, `scail2` |

Rung 2 was never image-only. Nothing in §2 (the four facts), §3's graph shape, §6 (never substitute
output), §7 (tier integrity), §8 (one pipeline) or §9 (single-encoding) is image-specific either.

### Three things that did not port — the first is now done

**1. The naming and the calibration ABI — done, SC-15804.**
The contract *was* image-named throughout: 36 public `ImageMemory*` items plus
`IMAGE_MEMORY_CALIBRATION_ABI`. An ABI bump makes existing evidence stale *by design*, and every one
of the 53 calibration stories adds records keyed to that ABI — so the rename had to land while there
were still only three provider adopters and near-zero promoted evidence, which is exactly when it
did. §13 records what it touched. The vocabulary is lane-neutral today, so this is no longer a
portability blocker; the two items below still are.

**2. The contract asserts the ladder has exactly five rungs.**
`MemoryStrategy::ALL: [Self; 5]`, and `conformance_errors` rejects any contract whose
`strategies.len()` differs — so every provider must declare all five. A sixth rung is a breaking
change to every declaration. Two candidates video plausibly needs, neither of which is any current
rung:

- **KV-cache bounding.** Krea Realtime 14B measures 7.14 GiB of KV cache — activations, so
  quantization does not shrink it. It is scratch, but unlike attention scratch it *persists across
  steps*, so rung 3 does not reach it.
- **Expert residency.** Wan A14B swaps between two expert weight sets mid-denoise. Residency-bounding,
  partitioned by expert rather than by block — a sibling of rung 4, not an instance of it.

**3. `MemoryGeometry` carries an image's axes.** Video needs frames; audio needs duration.

### Audio's structural difference — streaming output

Audio can emit incrementally. A provider still streaming output cannot release its decoder, which
directly conflicts with rung 1's "completed phases are released before the next materializes." Rung 1
for a streaming lane needs either a later release point or a declared exclusion.

### §6 extends verbatim, on a different axis

**Never silently drop frames, shorten a clip, or truncate audio duration.** Same rule, same three
surfaces: do not offer what cannot be rendered, refuse with a measured alternative, record the
refusal.
