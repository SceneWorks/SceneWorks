# Calibration invalidation survey (sc-17775)

Every calibrated provider, and what can force a re-capture of its memory measurements without
changing the behaviour those measurements describe.

This is characterisation, not repair. The remedy is sc-17776's job; the numbers here exist so it can
be prioritised against something real.

**Read against** SceneWorks `main` at inference pin `fbb00d6b` and
`docs/generated/memory-matrix.json` `schemaVersion 5`, surveyed 2026-08-05. Every count below is
reproducible from the commands in [Method](#method).

---

## 1. The headline

| | |
| --- | --- |
| Bound calibrations in `config/manifests/builtin.models.jsonc` | **31** |
| Calibrated providers | **6** |
| Bound calibrations that are `current` at the live pin today | **11** |
| Bound calibrations demoted to `historical` today | **20** |
| Providers with **any** mechanism to extend a capture past a pin bump | **1 of 6** (`flux2_dev`) |
| …and that mechanism is **already spent**: it authorizes one revision, and the pin has moved past it | ✅ |

The epic's framing — "invalidation over-triggers" — is one of two exposures, and it is the smaller
one. It is visible only because `flux2_dev` is the single provider that has a relief mechanism at
all. The other five have no extension path of any kind: a pin bump demotes them permanently and the
only remedy on offer is a fresh capture at the new pin.

**The sharpest single number in this survey:** `flux1_dev` and `flux1_schnell` hold 10 bound
calibrations, demoted by a window of **65 commits, exactly zero of which touched
`crates/media/candle-gen/candle-gen-flux`** — the crate that implements them.

---

## 2. How invalidation actually works

There are **two independent gates**, and only the second one is revision-sensitive. Conflating them
is what makes the exposure hard to see.

### Gate 1 — `calibrationBinding` (content)

`scripts/generate-memory-matrix.mjs:532`. Compares a calibration record against the matrix cell on:
record status, quality result, sweep range verification, **calibration fingerprint**, engaged-rung
composition, runtime strategy parameters, geometry envelope, batch/frames, artifact loadability
identity.

It carries **no revision term at all**. A record whose fingerprint, parameters and artifact still
match stays eligible forever, regardless of how far the pin has moved.

### Gate 2 — `evidenceSemantics` (revision)

`scripts/memory-calibration-harness.mjs:614`. An eligible record is `current` if and only if

```js
record.repositories.inference.revision === pin ||
  compatibleCapturedInferenceRevisions.includes(record.repositories.inference.revision)
```

Everything else is `historical`, which is what strips a cell of `Verified` / `Runtime verified`
(`generate-memory-matrix.mjs:2194`).

So the unit of invalidation is **the whole inference repository, at commit granularity**. Any commit
to any of ~2620 no-merge commits in the last 60 days moves the pin and demotes every bound
calibration that has not been re-captured.

### The one escape hatch, and why only one provider has it

`compatibleCapturedInferenceRevisions` is populated exclusively by `compatibilityAuthorizes`
(`generate-memory-matrix.mjs:1485`):

```js
function compatibilityAuthorizes(binding, { modelId, provider, inferenceRevision, audit }) {
  return modelId === audit.modelId &&
    provider === audit.provider &&
    binding.inferenceRevision === audit.capturedInferenceRevision &&
    binding.compatibleInferenceRevision === audit.compatibleInferenceRevision &&
    inferenceRevision === audit.compatibleInferenceRevision;
}
```

`audit` is the single frozen constant `FLUX2_COMPATIBILITY_AUDIT`
(`generate-memory-matrix.mjs:65`), whose `modelId` and `provider` are both the literal `flux2_dev`.
The mechanism is therefore **not a mechanism at all** in the general sense — it is one hard-coded
provider carrying one hard-coded `(captured → compatible)` revision pair. It authorizes exactly one
target revision, so it is spent the moment the pin moves one commit further.

That is the state on `main` today: the audit's `compatibleInferenceRevision` is `a4f409ae`, the pin
is `fbb00d6b`, and all five `flux2_dev` cells read `historical`. (Already tracked as **sc-17760**,
which blocks sc-15922 — not a new break.)

---

## 3. Per-provider table

| Provider | Backend | Crate | Crate-mates in the same crate | Fingerprint constant | Bound calibrations | Capture pin | Commits behind live pin | Semantics today | Artifact layer | Trigger classes that apply |
| --- | --- | --- | --- | --- | ---: | --- | ---: | --- | --- | --- |
| `flux2_dev` | candle | `candle-gen-flux2` | `flux2_klein_9b` (generator + memory), `flux2_dev_control`, `flux2_dev_edit` | `CALIBRATION_FINGERPRINT` (`memory_strategy.rs:41`) | 5 | `5ffd7612` | 41 | **historical** (extension spent at `a4f409ae`) | **yes**, v5, exhausted | 1, 2, 3, 4, 5, 6 |
| `flux1_dev` | candle | `candle-gen-flux` | `flux1_schnell` — **shares the same fingerprint constant** | `CALIBRATION_FINGERPRINT` (`memory_strategy.rs:29`) | 5 | `5f973a73` | 65 | **historical** | none | 1, 2, 3, 4, 5 |
| `flux1_schnell` | candle | `candle-gen-flux` | `flux1_dev` — **shares the same fingerprint constant** | same constant as above | 5 | `5f973a73` | 65 | **historical** | none | 1, 2, 3, 4, 5 |
| `z_image_turbo` | mlx | `mlx-gen-z-image` | **3** — `z_image` base, `z_image_base_control`, `z_image_control`, all with registered memory strategies | `MEMORY_CALIBRATION_FINGERPRINT` (`memory_strategy.rs:395`) | 5 | `d4802320` | 195 | **historical** | none (and see §6 — not producible on this box) | 1, 2, 3, 5 |
| `qwen_image` | mlx | `mlx-gen-qwen-image` | **2** — `qwen_image_control`, `qwen_image_edit`, both with registered memory strategies | `MEMORY_CALIBRATION_FINGERPRINT` (`memory_strategy.rs:22`) | 9 | `fbb00d6b` | 0 | **current** | none | 1, 2, 3, 5 (all latent — fires on the next bump) |
| `krea_2_turbo_control` | mlx | `mlx-gen-krea` | **4** — `krea_2_turbo`, `krea_2_raw`, `krea_2_edit`, `krea_2_turbo_edit`; those four share `block_memory_strategy.rs`'s separate fingerprint, the calibrated control route uses `memory_strategy.rs` | `MEMORY_CALIBRATION_FINGERPRINT` (`memory_strategy.rs:24`) | 2 | `fbb00d6b` | 0 | **current** | none | 1, 2, 3, 5 (all latent) |

Trigger-class key: **1** crate-mates · **2** `gen-core` · **3** `#[cfg(test)]` inside the audited
crate · **4** vendored `candle-kernels` · **5** `Cargo.lock` / workspace build inputs · **6**
pre-layout revisions the audit cannot run on (new, §5.6).

### 3.1 Fingerprint without a bound calibration — no exposure yet

**30** constants matching the fingerprint naming exist at `fbb00d6b` — 13 in `candle-gen`, 17 in
`mlx-gen` — of which 29 are calibration fingerprints (`mlx-gen-lens::LEGACY_TEXT_ENCODER_FINGERPRINT`
is not). Only **5** distinct strings are bound. The other 24 name providers with no bound
calibration: no exposure today, but they inherit this entire table the moment one is captured.

| Crate | Unbound fingerprint constants |
| --- | --- |
| `candle-gen-flux` | `RESIDENCY_CALIBRATION_FINGERPRINT` |
| `candle-gen-flux2` | `RESIDENCY_CALIBRATION_FINGERPRINT`, `KLEIN_CALIBRATION_FINGERPRINT` |
| `candle-gen-krea` | `RESIDENCY_`, `TURBO_MEMORY_`, `CONTROL_MEMORY_CALIBRATION_FINGERPRINT` |
| `candle-gen-lens`, `candle-gen-pulid`, `candle-gen-qwen-image` | one each |
| `candle-gen-z-image` | `CALIBRATION_`, `CONTROL_CALIBRATION_FINGERPRINT` |
| `mlx-gen-flux2` | `CALIBRATION_`, `KLEIN_MEMORY_CALIBRATION_FINGERPRINT` |
| `mlx-gen-sana` | `MEMORY_`, `RESIDENT_MEMORY_CALIBRATION_FINGERPRINT` |
| `mlx-gen-sensenova` | `QUALITY_`, `FAST_CALIBRATION_FINGERPRINT` |
| `mlx-gen-anima`, `mlx-gen-flux`, `mlx-gen-mage`, `mlx-gen-pulid`, `mlx-gen-sd3`, `mlx-gen-krea::block_memory_strategy` | one each |

The story's warning about non-uniform naming is not theoretical: this survey's first inventory pass
undercounted by 8 by grepping one spelling against the wrong revision. The reliable form is
`git grep -nE 'const [A-Z_]*FINGERPRINT[A-Z_]*: &str' <pin> -- 'crates/media/*/*/src/*.rs'`.

### 3.2 Bound calibration without a fingerprint — none

Checked explicitly, because it would be the worse finding: a bound calibration with no fingerprint
constant has no content gate at all. All five bound fingerprint strings still resolve to a live
constant at the pin (`git grep -l "<string>" fbb00d6b -- crates`). **No provider is in this state.**

---

## 4. Measured evidence

Every row below is a real `scripts/inference-artifact-audit.mjs --lane cuda` run on the RTX PRO 6000
box (rustc 1.96.0, nvcc 12.9, `--features cuda`, warm `D:\repos\inference-audit\sc-17524-target`).
Records are checked in under `docs/calibration/sc-17775/`.

**Reproducibility.** Three separate runs, from three different captured revisions whose FLUX.2
closure is byte-identical, each independently produced
`sha256:d80844f24dcb95f957c1cd893f9238c9d753db8e1e40c5deefe9f6b6f740f9aa` — the same digest frozen in
`FLUX2_COMPATIBILITY_AUDIT.artifactProof`. An `ARTIFACTS DIFFER` verdict below is therefore a real
signal, not link nondeterminism.

### 4.1 Historical windows

| # | Class | Window | Closure paths that moved | Captured digest | Compatible digest | Verdict |
| --- | --- | --- | --- | --- | --- | --- |
| A | 1 crate-mate — **contaminated**, see below | `b0fb8e77 → 41464d57` `feat: add FLUX.2 Klein memory ladder (sc-15831)` | `candle-gen-flux2` | `d80844f2…` | `38d11122…` | **DIFFER** |
| B | 2 `gen-core` | `78b78d37 → d02b8fcf` `feat(mlx): add the Anima 2B shared memory ladder` | `crates/contracts/gen-core` | `d80844f2…` | `d80844f2…` | **identical — absorbed** |
| C | 5 `Cargo.lock` | `dbb435e8 → 278bc946` `feat(mlx): add FLUX.2 Klein shared memory ladder` | `Cargo.lock` | `d80844f2…` | `d80844f2…` | **identical — absorbed** |
| D | 1 + 2 combined | `b1752354 → a6a4f057` `fix(media): overflow-proof image buffer guards (sc-12571)` | `gen-core`, `candle-gen-flux2` | `cee0e2e6…` | `4322da37…` | **DIFFER** (not attributable to one class) |
| F | 4 vendored `candle-kernels` | `80917bb5 → 9f997ef6` `docs: correct vendored changes heading (sc-12492)` | `vendor/candle-kernels` | `f6cb6a5b…` | `f6cb6a5b…` | **identical — absorbed** |

Row **B** is the cleanest over-trigger in the survey: an **MLX** memory ladder for **Anima** edits
`gen-core`, which moves FLUX.2's Candle/CUDA closure by object identity and forces a build — and the
linked FLUX.2 measurement binary is byte-identical. Link-time DCE absorbed the entire change.

Row **C** is the same shape one layer down: an MLX-only commit that moves nothing but `Cargo.lock`.

Row **F** is class 4's only observable shape (see §5.4).

Row **A** is reported honestly rather than as the crate-mate proof it looks like. `41464d57` is a
Klein commit, but it is not additive: it generalises `load_with_memory_context` over `Flux2Variant`
and deletes the `if pipe.variant.is_dev()` branches, so `flux2_dev`'s own control flow was rewritten
in the same diff. Its `ARTIFACTS DIFFER` verdict is therefore **not** demonstrably a false positive.
The same is true of row D, which spans two closure paths. That is why the crate-mate and
`#[cfg(test)]` claims rest on M1 and M2 instead: a historical commit that changes exactly one thing
and nothing else is rare, and pretending otherwise would be reasoning dressed as measurement.

This also sharpens the sc-17760 picture: extending the FLUX.2 audit to `fbb00d6b` has to cross
`41464d57` and `fbb00d6b`, both of which touch shared `candle-gen-flux2` code, so that re-run should
be expected to return `ARTIFACTS DIFFER` and demand a real re-capture.

### 4.2 Hand mutations

Historical commits are noisy; these isolate one variable each, on a commit built out-of-tree on top
of `fbb00d6b` (`docs/calibration/sc-17775/mutations.md` records the exact edit).

All four are captured at `fbb00d6b`, whose FLUX.2 binary hashes `sha256:fee1c2de…` — reproduced
identically by all four independent runs.

| # | Class | Mutation | Compatible digest | Verdict | Reads as |
| --- | --- | --- | --- | --- | --- |
| M1 | 1 crate-mate | bump `KLEIN_CALIBRATION_FINGERPRINT`, a constant `flux2_dev` provably cannot execute | `73fe78d5…` | **DIFFER** | over-trigger, unadjudicable by the current audited unit |
| M2 | 3 `#[cfg(test)]` | add a `#[cfg(test)] mod` with one trivial test to `candle-gen-flux2` | `750a0cfe…` | **DIFFER** | over-trigger, and confirms the audited artifact really is a test binary |
| M3 | positive control | bump the **dev** `CALIBRATION_FINGERPRINT` the calibration is bound to | `eb9a6ead…` | **DIFFER** | correct — the audit fails closed on a genuine invalidation |
| M4 | 2 `gen-core` | add an unreferenced `pub fn` to `gen-core` | `fee1c2de…` | **identical** | DCE absorbs it; matches row B on a real commit |

M3 is the one row that is *supposed* to differ. Without it, M1/M2/M4 would be a suite that passes
with the artifact layer ripped out.

---

## 5. Trigger classes

### 5.1 Crate-mates — over-triggers, and the artifact layer cannot help

**All six calibrated providers share their crate** with at least one sibling that has its own
registered memory strategy — see the crate-mate column in §3. There is no calibrated provider that
owns its crate alone. For any of them, a commit aimed at a sibling invalidates it at **both** gates:

- **Gate 2** always, because the pin moves.
- **The artifact layer**, because the audited unit is the **whole-crate `--lib` test binary**
  (`AUDIT_ARTIFACT_TARGET`, `inference-artifact-audit.mjs:232`). Sibling code — including the
  sibling's own `#[cfg(test)]` tests, which keep it alive against DCE — is linked into it. Every
  crate-mate instance measured here returned `ARTIFACTS DIFFER`, including M1, whose entire diff is
  a string constant `flux2_dev` has no path to. The layer that exists to absolve over-triggers
  cannot absolve this one.

`flux1_dev` / `flux1_schnell` are the worst case and need no build to demonstrate: they share **one**
`CALIBRATION_FINGERPRINT` constant with one use site (`candle-gen-flux/src/memory_strategy.rs:97`,
inside the shared `provider_contract(provider_id, spec)`). A schnell-only ladder revision that bumps
that string invalidates dev's 5 bound calibrations at **gate 1** as well, where not even a re-pin
would help.

Measured: rows A, D and M1.

### 5.2 `gen-core` — the highest-volume *code* class, and largely absorbed

117 no-merge commits touched `crates/contracts/gen-core` in the last 60 days (~4.5% of 2620) — the
most frequent closure mover after `Cargo.lock`, and the most frequent one that is actually code.

**Linked fraction.** Row B and mutation M4 both measure the mechanism directly: a `gen-core` change
that FLUX.2's linked code does not reference produces a byte-identical binary. Of the `gen-core`
commits inside the currently demoted windows, the one that reached `flux1`'s window (`d02b8fcf`) is
*measured* inert for the Candle closure. This is an estimate, not a census: the honest statement is
that the artifact layer resolves this class case-by-case for ~40 s per window, and that the observed
cases so far all resolved to "absorbed".

Measured: rows B, D (combined) and M4.

### 5.3 `#[cfg(test)]` inside the audited crate — accepted false positive, quantified

`candle-gen-flux2` took 45 non-merge commits in 60 days. **3** are test-region-only, and they split
into two sub-classes the epic's footnote does not distinguish:

| Sub-class | Commits | Reaches the `--lib` test binary? | Free path | Artifact layer |
| --- | ---: | --- | --- | --- |
| `tests/` integration files (`conformance.rs`, `convert_real_weights.rs`) | 2 (`0553f2eb`, `5f45bd82`) | **no** — separate test binaries | fails (tree moved) | should absolve |
| `#[cfg(test)] mod` inside `src/` | 1 (`1211ee58`) | **yes** | fails | **cannot** absolve |

So the accepted false positive is ~1 commit in 45 (~2%) for this crate, not the whole test-commit
rate. The `tests/` sub-class is cheaper than the footnote implies — it costs a 40 s build, not a
capture.

Measured: M2. The one historical in-`src` instance (`1211ee58`, 2026-07-13) is **unmeasurable** — see
§5.6.

### 5.4 Vendored `candle-kernels` — 3 moves in 60 days, all documentation

Only 3 non-merge commits touched `crates/media/candle-gen/vendor/candle-kernels` in 60 days:

| Commit | Date | Content |
| --- | --- | --- |
| `3738860c` | 2026-07-20 | `VENDORED.md` only (+29/−9) |
| `9f997ef6` | 2026-07-18 | `VENDORED.md` only (1 line) |
| `9e857956` | 2026-07-12 | tree move into the ownership layout |

**Zero `.cu` changes.** Every observed move of this closure path was provably non-compiling, and the
artifact layer absorbs them (row F). The dangerous shape this closure path exists to catch — a `.cu`
edit that would silently change kernels — is **unobserved in the surveyed window**. Recorded as
unobserved rather than dropped: the path earns its place on the risk, not on the frequency.

### 5.5 `Cargo.lock` and the workspace build inputs — a class the epic did not list

81 non-merge commits touched `Cargo.lock` in 60 days — the **highest-frequency closure mover of all**,
more than `gen-core`. Most are workspace crate-version bumps riding along with unrelated feature
work; the sample in row C is an MLX ladder for a different family.

The artifact layer absorbs these (row C), but on the free path a lockfile move alone forces a build.
For the five providers with no artifact layer it is an unconditional demotion. This class deserves
first-class treatment in sc-17776 because of its rate.

`rust-toolchain.toml` (1 commit) and `.cargo/config.toml` (2) are the same class at negligible rate.

### 5.6 New: the audit cannot run on pre-layout revisions

`scripts/inference-artifact-audit.mjs` fails hard on any revision before the 2026-07-12 ownership
layout move (`9e857956`), because `crates/contracts/gen-core` did not exist:

```
fatal: path 'crates/contracts/gen-core' exists on disk, but not in '3f2f6f3e…'
Command failed: git -C D:\repos\inference rev-parse 3f2f6f3e…:crates/contracts/gen-core
```

This is not a live problem — every capture pin in the manifest is post-layout — but it bounds what
the artifact layer can ever adjudicate, and it is why §5.3's historical in-`src` `#[cfg(test)]`
instance had to be replaced with a hand mutation.

---

## 6. Is an artifact layer applicable to each capture shape?

| Provider | Applicable? | What it would take |
| --- | --- | --- |
| `flux2_dev` (candle) | already exists | keep `FLUX2_COMPATIBILITY_AUDIT` current — sc-17760 |
| `flux1_dev`, `flux1_schnell` (candle) | **yes, directly** | `candle-gen-flux` already carries the equivalent measurement targets `flux_dev_probed_generate_for_offload_ab` and `flux_schnell_probed_generate_for_offload_ab` (`lib.rs:959`, `:968`). The script's closure, target and constants are hard-coded to FLUX.2; parameterising them is the whole job. |
| `z_image_turbo`, `qwen_image`, `krea_2_turbo_control` (mlx) | **not on this box** | The captures are Metal/M5. `--lane metal` exists but yields a deliberately inert record, and `mlx-gen` does not build on Windows at all, so an MLX artifact layer can only ever be produced on the Mac. `candle-gen-z-image` has no `probed_generate_for_offload_ab`-shaped target, so even the Candle mirror would need one written. |

The one layer that **is** host-independent is the resolved-feature witness — `cargo tree`, no
compile, `--target` pinned rather than taken from the host (`inference-artifact-audit.mjs:252`). It
resolves identically from any machine.

---

## 7. Findings of the opposite sign (would a real invalidation be MISSED?)

A false green is strictly worse than the false positives this epic is about, so these were looked
for deliberately.

1. **A fingerprint bump inside an authorized window.** The scenario: the pin moves to a revision the
   artifact audit authorizes, but the provider bumped its own `CALIBRATION_FINGERPRINT` in between —
   i.e. explicitly declared the calibration invalid. SceneWorks compares the record's fingerprint
   against its **own manifest copy**, not against inference source, so gate 1 would not see it.
   *Result:* **not a hole.** The dev constant is used in production code
   (`memory_strategy.rs:180`, inside `provider_contract`), so it is linked into the audited binary
   and the artifact layer fails closed. Mutation M3 is the positive control for this.

2. **`compatibleInferenceRevision` on `turboFit.evidenceRecords`.** `krea_2_turbo` carries six
   `compatibleInferenceRevision: a4f409ae` entries in its `turboFit` evidence block
   (`builtin.models.jsonc:5375`+). These look like a second provider with an artifact layer, and they
   are **not** — `compatibilityAuthorizes` gates on `modelId === "flux2_dev"`, so those fields are
   inert for the semantics gate. *Result:* not a correctness hole, but a real readability trap: the
   field name is identical to the load-bearing one and a reader is entitled to assume it does
   something.

3. **Nothing compares the bound fingerprint against the inference source constant.** Gate 1 compares
   `record.calibrationFingerprint` (evidence bundle, SceneWorks-authored) against
   `cell.calibrationFingerprint` (manifest, SceneWorks-authored). **Both sides are SceneWorks
   copies.** A fingerprint transcribed wrong into both — the manifest and the record are written
   together — binds happily while inference's provider advertises something else, and no check in
   either repo would notice. For `flux2_dev` the artifact layer covers this indirectly (finding 1);
   for the other five there is no artifact layer, so nothing covers it at all.
   *Result:* **no live instance** — all five bound strings resolve to a live constant at the pin
   (§3.2) — but that is a fact established by hand in this survey, not by any gate. This is the one
   genuinely unguarded false-green surface found. It is cheap to close: assert the five bound strings
   against `git grep` of the pinned inference source at matrix-generation time.

4. **Whether a crate-mate change can be silently absolved.** Not with the current audited unit — the
   whole-crate test binary made every measured crate-mate instance fail loudly (§5.1). Worth stating
   because sc-17776 will be tempted to narrow the audited unit to fix the over-trigger, and doing so
   naively (per-crate objects, or an rlib) reintroduces the two false-green units
   `inference-artifact-audit.mjs:17-30` already measured and rejected.

**One unguarded surface (finding 3), no live false green.** The shipped mechanism's failure mode is
otherwise uniformly over-triggering.

---

## 8. Counts for sc-17776 to prioritise against

| Exposure | Providers | Bound calibrations | Status |
| --- | ---: | ---: | --- |
| Demoted today, **no** relief mechanism | `flux1_dev`, `flux1_schnell`, `z_image_turbo` | **15** | needs a fresh capture per provider |
| Demoted today, mechanism exists but **spent** | `flux2_dev` | **5** | one audit re-run — sc-17760 |
| `current` today, **no** relief mechanism | `qwen_image`, `krea_2_turbo_control` | **11** | demoted by the next pin bump, whatever it contains |
| **Total** | 6 | **31** | |

Relevance of the demoting windows:

| Provider | Commits in the window | …touching its own crate | …touching `gen-core` |
| --- | ---: | ---: | ---: |
| `flux1_dev` / `flux1_schnell` | 65 | **0** | 1 (measured inert, row B) |
| `z_image_turbo` | 195 | 1 (`d74f96f3`, a test commit) | 6 |
| `flux2_dev` | 41 | 2 (both Klein) | 1 |

Cost of an artifact-layer adjudication, measured this session: **~40 s warm** for two release
`cargo test --no-run` builds against a surviving `-target`. No GPU and no weights. That is the number
to weigh a re-capture against, not the ~47.6 GB capture session the v1 audit implied.

---

## Method

Everything above is reproducible.

```bash
# fingerprint constants (the naming is not uniform — do not grep one spelling, and pin the revision)
git -C D:/repos/inference grep -nE 'const [A-Z_]*FINGERPRINT[A-Z_]*: &str' fbb00d6b \
  -- 'crates/media/*/*/src/*.rs'

# bound calibrations, from the shipped manifest
node scripts/generate-memory-matrix.mjs      # then read docs/generated/memory-matrix.json

# closure-path commit frequency, 60 days
for p in Cargo.toml Cargo.lock rust-toolchain.toml .cargo/config.toml \
         crates/contracts/gen-core crates/media/candle-gen/candle-gen \
         crates/media/candle-gen/candle-gen-pid crates/media/candle-gen/candle-gen-flux2 \
         crates/media/candle-gen/vendor/candle-kernels; do
  echo "$p $(git rev-list --count --no-merges --since=2026-06-05 fbb00d6b -- $p)"
done

# one measured window (needs MSVC 14.44 vcvars + CUDA_COMPUTE_CAP=120 in the same shell)
node scripts/inference-artifact-audit.mjs --repo D:/repos/inference \
  --captured <sha40> --compatible <sha40> --lane cuda \
  --workdir D:/repos/inference-audit/sc-17524 --out docs/calibration/sc-17775/<name>.json
```

Related: `docs/inference-artifact-audit-sc-17497.md`, sc-17760 (extend the FLUX.2 audit to the live
pin), sc-17776 (recommend an invalidation unit that tracks behaviour).
