# What should invalidate a packaged calibration (sc-17776)

Research and a recommendation. **No implementation** — nothing here changes the shipped record, the
frozen constants or either validator.

The question sc-17497 answered was "did the compiled code change?", and it answered it well: hash the
linked test binary. The question this story asks is the one underneath it — "did the **behaviour**
those measurements describe change?" — and the gap between the two is paid in ~47.6 GB re-captures.
[sc-17775](https://app.shortcut.com/trefry/story/17775) measured the gap; this story asks whether a
better unit exists.

**Read against** SceneWorks `main` at inference pin `fbb00d6b`, measured 2026-08-05 on the RTX PRO
6000 box (rustc 1.96.0, MSVC 14.44, nvcc 12.9, `--features cuda`). Everything below is reproducible
from [Method](#method); the raw records are in `docs/calibration/sc-17776/`.

---

## The finding that reframes the epic

**The epic's worked example is not an over-trigger.** The window it is built on —
`5ffd7612 → fbb00d6b`, where `flux2_dev`'s `CALIBRATION_FINGERPRINT` is untouched — changes the
code the measurements ran, and it changes it inside the memory ladder itself. Every unit measured
here says so, including the narrowest one:

| unit | `5ffd7612` | `fbb00d6b` | |
| --- | --- | --- | --- |
| shipped whole test binary | `d80844f2…` | `fee1c2de…` | DIFFER |
| non-test example binary | `1ffa3084…` | `8fd614b0…` | DIFFER |
| reachable from the measured test | `1c40ed58…` | `aaa1297d…` | DIFFER |
| reachable from the memory ladder alone | `35706634…` | `924a6b1b…` | DIFFER |

The reason is in the diff rather than in the digest. `41464d57` (sc-15831) did not add Klein
*alongside* dev — it **merged their ladders**: `provider_contract(spec)` now delegates to
`provider_contract_for(provider_id, spec)` reading a `ProviderProfile` from a shared
`profile(provider_id)` match, and the `if pipe.variant.is_dev()` branches were deleted in the same
commit. Dev's contract construction was rewritten. The only thing asserting the result is inert is
the author leaving `CALIBRATION_FINGERPRINT` alone, and [candidate 2](#candidate-2--grade-the-providers-own-calibration_fingerprint)
shows that constant has never once moved after a capture.

So **the re-capture sc-17760 demands is genuinely owed**, and no unit proposed in this story would
have avoided it. That does not dissolve the epic — the over-triggering class is real and is measured
below on isolated mutations — but it does move the FLUX.2 case out of it.

## The answer in one table

Seven one-variable mutations on `fbb00d6b` (`docs/calibration/sc-17776/mutations.md`), each run
through five candidate units. ✅ = the unit gives the answer a sound gate should give.

| mutation | should say | shipped unit (today) | non-test binary | reachable (all) | reachable (`.text`) | ladder-scoped |
| --- | --- | --- | --- | --- | --- | --- |
| `M-rms` — `RMS_EPS` `1e-5→2e-5` | DIFFER | DIFFER ✅ | DIFFER ✅ | DIFFER ✅ | DIFFER ✅ | **identical ❌** |
| `M-devfp` — dev fingerprint bump | DIFFER | DIFFER ✅ | DIFFER ✅ | DIFFER ✅ | **identical ❌** | DIFFER ✅ |
| `M-gencore` — unreferenced `gen-core` fn | identical | identical ✅ | identical ✅ | identical ✅ | identical ✅ | identical ✅ |
| `M-editprov` — klein-edit provider edit | identical | DIFFER ❌ | **identical ✅** | **identical ✅** | identical ✅ | identical ✅ |
| `M-cfgtest` — a `#[cfg(test)] mod` | identical | DIFFER ❌ | **identical ✅** | DIFFER ❌ | DIFFER ❌ | identical ✅ |
| `M-klein` — klein fingerprint bump | identical | DIFFER ❌ | DIFFER ❌ | DIFFER ❌ | identical | DIFFER ❌ |
| `M-safety` — widen the admission guard | DIFFER | DIFFER ✅ | DIFFER ✅ | DIFFER ✅ | DIFFER ✅ | DIFFER ✅ |

Read the two right-hand columns as disqualifications, not as near-misses: a `.text`-only digest
cannot see a provider **declaring its own calibration void**, and a ladder-scoped digest cannot see
a numerics change on the measured path. Both are false greens, produced on demand, and a false green
is the one failure mode this epic must not introduce.

Three results decide the recommendation:

1. **Narrowing the audited target to a non-test binary is the only candidate that absolves a class
   without opening a false green** — and it absolves two: `#[cfg(test)]` additions and crate-mate
   code the measured route does not link.
2. **Reachability scoping does not absolve `#[cfg(test)]`**, which was the class it was most
   expected to remove. See [why](#why-reachability-cannot-absolve-cfgtest) — the reason is CGU
   partitioning, and it disqualifies *every* code-hash unit from that class, not just this one.
3. **Nothing absolves `M-klein`**, because dev and Klein genuinely share `profile()` since sc-15831.
   The crate-mate class is irreducible for FLUX.2 by any code-based unit, which is also why
   [candidate 3](#candidate-3--split-the-multi-provider-crates-upstream) fails.

---

## Method

Every row is a real build. The instrument is two research scripts, checked in so the numbers can be
re-derived and so their limitations are readable rather than implied:

| | |
| --- | --- |
| `scripts/research/sc-17776-reachability-probe.mjs` | parses COFF objects, walks symbol relocations from a named root, digests the reachable set |
| `scripts/research/sc-17776-measure.mjs` | drives one build per case under `inference-artifact-audit.mjs`'s exact discipline and runs the probe over what the linker was handed |

Neither is wired into a gate, a test list or CI. They are measuring instruments for this story.

### The build discipline is the shipped one

The driver imports `reproducibleLinkFlags()` and `encodedRustflags()` from
`scripts/inference-artifact-audit.mjs` rather than restating them, and reuses the same detached
worktree, the same `--remap-path-prefix` set, `CARGO_INCREMENTAL=0`, `--locked` and the same warm
`D:\repos\inference-audit\sc-17524-target`. A digest computed under different flags is not
comparable with the one it is being weighed against.

Each case builds the audited target **twice**:

1. `cargo test -p candle-gen-flux2 --lib --release --locked --no-run --features cuda` — byte-for-byte
   the shipped audit's command, so the `shipped unit` column can be checked against the checked-in
   records.
2. `cargo rustc … --profile bench … -- -Csave-temps --print=link-args` — the same binary with its
   codegen units preserved and the linker's input list printed. Those extra flags change the rustc
   command line and therefore the bytes, which is exactly why the first build exists.

`-Csave-temps` is passed as a trailing `cargo rustc` argument rather than through
`CARGO_ENCODED_RUSTFLAGS`, so it applies to one unit and the ~700 dependency crates stay warm.

### Reproducing it

Needs the MSVC 14.44 vcvars environment and `CUDA_COMPUTE_CAP=120` in the same shell, and a worktree
at `--workdir` (the audit tool deletes its own on exit; recreate it with
`git worktree add --detach D:/repos/inference-audit/sc-17524 <rev>`). The derived `-target` must not
be deleted — a cold run is a multi-hour compile, a warm one is a few minutes per case.

```bash
node scripts/research/sc-17776-measure.mjs \
  --workdir D:/repos/inference-audit/sc-17524 \
  --cases "fbb0-a=<sha40>,M-rms=<sha40>+rms.patch,captured-5ffd=<sha40>" \
  --example flux2-txt2img \
  --out docs/calibration/sc-17776/reachability-matrix.json
```

Each `--cases` entry is `label=revision[+patchfile]`; the mutation patches are reproduced verbatim in
`docs/calibration/sc-17776/mutations.md`. The record this produced is checked in alongside it.

### Why the walk is over objects rather than the PE

The story frames candidate 1 as "a symbol + relocation walk of the PE". That is not possible on this
artifact, and the reason is the audit's own determinism fix: `/DEBUG:NONE` means no PDB is emitted,
and an MSVC-linked image carries no COFF symbol table. The linked binary has no symbol boundaries and
no symbol-level relocations left — only base relocations. A reachability walk of it would mean
disassembling 181 MB of x86-64.

The pre-link objects have what is missing: one COMDAT section per function, a symbol table, and a
relocation for every cross-section reference. That is also the graph `link.exe /OPT:REF` walks, so a
BFS over it models link-time DCE rather than approximating it from the far side. **This is a
correction to the story's framing, not a shortcut** — and it changes the cost estimate, because the
implementation has to keep objects that a normal build throws away.

### What the digest covers

Per reachable section: the canonical section kind, the raw bytes, and the `(offset, type, target
symbol name)` of every relocation out of it. The bytes alone are not enough — a call site in an
object is a zero placeholder plus a relocation, so two functions differing only in which callee they
call have identical bytes.

Two normalisations, both forced by measurement rather than chosen:

- **Container identity is not hashed.** Codegen-unit names move when modules are reshuffled.
- **`.llvm.<n>` suffixes are stripped from symbol names.** LLVM appends a module-derived number when
  it internalises a symbol, so the suffix moves when *any* code in the crate moves. Without this the
  probe reported `5ffd7612` and `a4f409ae` as different — a window the shipped audit certifies as
  producing byte-identical binaries. See [Instrument limitations](#instrument-limitations).

---

## Instrument limitations

Stated up front, because a research instrument that flatters its own candidate is worse than no
instrument.

1. **The object graph is noisier than the linked image.** Symbol-name suffixes, COMDAT duplication
   and section granularity all vary for reasons the linker normalises away. One such source
   (`.llvm.<n>`) was found and removed; the certified-identical window `5ffd7612 → a4f409ae` is the
   control that says whether any remain, and it is reported for every configuration below.
2. **COMDAT selection is not modelled.** Where several objects define the same external symbol the
   walk keeps the first in sorted order. Duplicates are identical COMDATs by construction, so the
   contents agree; the count is reported so the assumption stays visible.
3. **Reachability is static, not path-sensitive.** Every branch of a reachable function is reachable,
   including arms for other providers. This is not a defect to be fixed — a sound walk cannot assume
   a branch is not taken — but it bounds what candidate 1 can ever absolve, and the measurements
   below show that bound is the binding one.
4. **One provider, one lane.** Everything is `flux2_dev` on `cuda`. The five other calibrated
   providers have no artifact layer at all (sc-17775 §6), and the three MLX ones cannot have one
   produced on this box.

---

## Candidate 1 — reachability-scoped digest

Hash only the sections reachable from `tests::flux2_dev_probed_generate_for_offload_ab` via a symbol
+ relocation walk, instead of the whole image. Literally "the code the measurements ran".

### It works, and the instrument says so before it says anything else

| control | result |
| --- | --- |
| same revision built twice (`fbb0-a` / `fbb0-b`) | every digest identical |
| shipped unit reproduces the frozen records | `d80844f2…` at `5ffd7612`, `fee1c2de…` at `fbb00d6b` — byte-equal to `FLUX2_COMPATIBILITY_AUDIT.artifactProof` and to sc-17760's record |
| the certified-identical window `5ffd7612 → a4f409ae` | identical on **all six** units, object-level included |
| `M-gencore` (link-time-DCE control) | identical — reproduces sc-17775's M4 on the shipped unit too |
| rlib parse failures | 0 across 150 archives / 886 objects |

The third row is the one that earns the instrument its verdicts, and it did **not** hold on the first
attempt. Un-canonicalised, the object digests reported `5ffd7612` and `a4f409ae` as different while
their linked binaries were byte-identical — a false positive of the candidate, not of the code. The
cause was LLVM's `.llvm.<n>` internalisation suffix, which is derived from the module and therefore
moves when any code in the crate moves. That is recorded rather than quietly fixed because it is the
shape of the maintenance burden this candidate carries: **the object graph is noisier than the linked
image, and every source of that noise has to be found and normalised away, one toolchain at a time.**

### What it buys

| | sections | bytes |
| --- | ---: | ---: |
| all linked objects | 204,470 | 78.1 MB |
| reachable from the measured test | 35,800 (17.5%) | 44.9 MB |
| reachable from the memory ladder alone | 1,256 (0.6%) | 0.19 MB |

An 82% narrowing by section count. On the measured mutations that narrowing buys exactly one
absolution — `M-editprov`, a bespoke klein-edit provider that is registered nowhere and that the dev
route never links. Real, and the class is real: `edit_provider.rs` is 1,091 of the crate's 11,218
lines and took 122 lines of the last Klein window alone.

### Why reachability cannot absolve `#[cfg(test)]`

`M-cfgtest` appends a `#[cfg(test)] mod` with one trivial test. Nothing in it is reachable from the
measured test, and yet the reachable digest moves — while the **non-test example binary built from
the same tree is byte-identical**.

Those two facts together locate it: the added module is not reaching the measured code, the measured
code is being *compiled differently*. rustc partitions codegen units by module, so adding one
repartitions the crate, and repartitioning changes which functions are internalised and which
cross-CGU calls can be inlined. The production code the calibration describes really does come out
as different bytes in the test build — correctly, from the digest's point of view — even though no
production source changed.

(The mechanism is inferred from the two controls rather than proven by instrumenting rustc. What is
*measured* is the discrimination: test build moves, non-test build does not.)

This is the finding with the widest consequence in this story: **no code-hash unit, however scoped,
can absolve the `#[cfg(test)]` class while the audited artifact is a test binary.** sc-17497 recorded
that class as an accepted false positive and sc-17775 quantified it at ~1 commit in 45 for this
crate. Both were right about the frequency and understated the mechanism — it is not that test code
is hashed, it is that test code perturbs the codegen of production code around it.

### Cost, and the two costs the story asked about specifically

- **Does it survive `#[inline]` and generic instantiation?** Yes, and by construction rather than by
  luck: the walk is over the objects the linker was handed, so an inlined or monomorphised body is
  in the *consumer's* codegen unit and is reached through the consumer. This is exactly the failure
  sc-17497 measured when it rejected per-crate `.o` hashing, and scoping to the reachable set across
  all objects avoids it rather than repeating it. The `M-rms` row is the check: `RMS_EPS` feeds
  `rms_norm` call sites in `transformer.rs`, and the reachable digest moves.
- **Is the walk deterministic across builds under `/Brepro /DEBUG:NONE`?** Yes for the walk; the
  determinism risk moved somewhere worse. Those flags stabilise the *linked image*; the objects were
  already stable, but the object graph carries content-derived symbol suffixes the linker discards.
  One such source was found and normalised. There is no argument that it was the last one — only the
  certified-window control, which has to be re-run against every toolchain bump.
- **Maintenance against a toolchain bump.** The unit depends on rustc's symbol mangling, CGU naming
  and COMDAT emission, and on MSVC's archive long-name convention (the MSVC/GNU difference silently
  produced an *empty* graph during this research — a failure that reads as a stable digest). A
  `rust-toolchain.toml` bump already hard-stops the audit (`assertComparableToolchains`), which
  contains the risk but does not remove the re-validation work.
- **A second parser for the Metal lane.** The probe is COFF-only. `--lane metal` exists so the tool
  can be exercised on a Mac; a reachability layer would need a Mach-O implementation or the layer
  becomes cuda-only, which is a regression in how much of the tool is exercisable off the RTX box.
- **The build has to keep objects.** `-Csave-temps` and a `--print=link-args` parse, or an
  equivalent, for every audited revision.

Rough implementation estimate: **2–3 days** for the probe, the record schema bump (v6), both
validators, the frozen constants and the doc — plus re-establishing a determinism baseline of the
same order as sc-17524's thirteen builds, plus the `adjudicates` re-derivation described in
[composition](#how-any-new-unit-has-to-compose-with-sc-17606-and-sc-17607).

**Verdict: reject for now.** It is sound, it is measurably narrower, and it absolves one real class.
It also costs a bespoke object-graph parser, a permanent toolchain-coupled maintenance obligation and
a second parser for Metal — to absolve a class that [candidate 4](#candidate-4--narrow-the-audited-target)
absolves for the price of one `cargo build`, along with a second class candidate 1 cannot touch at
all. Revisit only if the non-test target turns out to be unsound.

---

## Candidate 2 — grade the provider's own `CALIBRATION_FINGERPRINT`

The constant exists, upstream maintains it, and the audit ignores it. The question is what it may be
allowed to decide.

### It cannot authorize, and the reason is measured rather than argued

The known-live mutation — `RMS_EPS` `1e-5 → 2e-5` in `candle-gen-flux2/src/transformer.rs`, the
change sc-17524 used to prove the digest is not blind — **does not move the fingerprint**. Nothing
about a numerics edit reaches a string constant in `memory_strategy.rs`. A gate that reads "the
fingerprint held still, so the calibration extends" authorizes it. That is the false green the story
names, produced on demand.

The history says the same thing more strongly. Over 60 days on inference `main`:

| | |
| --- | ---: |
| non-merge commits touching `candle-gen-flux2/src` | 41 |
| …that changed the **value** of `flux2_dev`'s `CALIBRATION_FINGERPRINT` | **4** |
| …and all four are dated 2026-08-04, while the ladder itself was being authored | ✅ |
| commits touching `memory_strategy.rs` since the current value was set (`51622dcb`) | 2 — both Klein, neither bumped it |
| `flux1`'s shared `CALIBRATION_FINGERPRINT` value changes since introduction | **0** |

So the fingerprint has **no observed post-capture variance at all**. Its entire history is the
author revising their own ladder before anyone captured against it. A signal that has never moved
after a capture cannot distinguish a valid extension from an invalid one — it says "extend" every
time.

### Nothing makes the bump reliable, and that is a cross-repo problem

Per the story's own framing, grading this makes inference authors' discipline load-bearing for
SceneWorks' calibration validity. Measured state of that discipline today:

- **No CI check in inference** connects a fingerprint to anything. The two `.github/` matches are a
  `--expected-fingerprint` argument in the real-weights workflow and an unrelated oracle-producer
  string; neither fails when a provider's compiled surface moves without its fingerprint.
- The one declared-vs-observed comparison that exists is **capture-side and in SceneWorks**
  (`scripts/sc-15833-flux2-evidence.mjs:403-412`), and its "observed" side is
  `MEMORY_EXPECTED_FINGERPRINT` — an environment variable the operator sets. It proves the operator
  and the provider agree at capture time. It proves nothing about a later revision.

### What it may decide: a veto, never a pass

There is a sound, cheap use, and it is the opposite of the one the story's framing suggests.
Record `inferenceFingerprint` at **both** revisions, read from inference source, and let it
**fail** the audit when it moved — while giving it no authority at all when it held still.

That asymmetry is what makes it safe. A veto can only add refusals, so it cannot manufacture a false
green; and it closes a hole the artifact layer only closes by luck. sc-17775's finding 1 notes that
`flux2_dev`'s constant happens to be linked into the audited binary today
(`memory_strategy.rs:76`, inside the registered contract table), so a bump moves the digest. That is
a property of the current code, not a guarantee: move the constant behind a `#[cfg]`, or into a
provider whose contract table the measured path does not reach, and the digest stops seeing it. The
veto does not depend on that.

It also generalises to the five providers with **no** artifact layer, which is the larger exposure
(sc-17775 §8: 26 of the 31 bound calibrations). For them a source-read fingerprint comparison is the
*only* revision-sensitive check available without a build, and it directly closes sc-17775's
finding 3 — the one genuinely unguarded false-green surface in the survey, where both sides of the
comparison are SceneWorks' own copies.

**Verdict: adopt, as a veto only.** It is not the unit; it is a second, independent way to fail.

---

## Candidate 3 — split the multi-provider crates upstream

Four crates host more than one calibrated provider, and the split argument is that a sibling's
commit would then stop moving the audited binary.

Measured against the epic's own worked example, it does not.

`41464d57` (sc-15831) did not *add* Klein alongside dev — it **merged their ladders**.
`provider_contract(spec)` now delegates to `provider_contract_for(provider_id, spec)`, which reads a
`ProviderProfile` from a shared `profile(provider_id)` match, and the `if pipe.variant.is_dev()`
branches were deleted in the same diff. Across `a4f409ae → fbb00d6b` the crate moved 655 lines in
three files, and two of the three — `lib.rs` and `memory_strategy.rs` — are files dev compiles.

A split into `candle-gen-flux2-dev` and `-klein` therefore has to either duplicate the shared ladder
(undoing sc-15831 and reintroducing the drift it removed) or put it in a third crate both link — in
which case dev's binary still moves on every Klein ladder revision. The class the split addresses is
*independent* providers sharing a crate, and FLUX.2 is no longer one of those.

Where it would pay is `candle-gen-flux`: `flux1_dev` and `flux1_schnell` share **one**
`CALIBRATION_FINGERPRINT` with a single use site, so a schnell-only revision invalidates dev at gate
1, where no audit layer can help. But those two are also the providers with no artifact layer at all,
so the split is not the first thing they need.

**Cost, with the self-tax the story asked for.** SceneWorks SHA-pins inference, so landing a split
requires a pin bump — the exact event that demotes every `current` calibration (sc-17775: ~1.5 pin
bumps/day, 11 calibrations `current` today). The refactor itself is the smaller half: `candle-gen-flux2`
is 13 files and 11,218 lines, with five downstream dependents in inference
(`candle-gen-catalog`, `-ideogram`, `-lens`, `-sd3`, and itself) plus SceneWorks' own consumers.

**Verdict: reject for FLUX.2** — it cannot absolve the case that motivated the epic. Record as a
possible future for `candle-gen-flux`, behind that family getting an artifact layer at all.

---

## Candidate 4 — narrow the audited target

Hash a **non-test** binary built from the same tree, so the `#[cfg(test)]` class disappears at the
source rather than being accepted.

sc-17497 chose the test binary deliberately — it is the one that produced the measurements — and any
move here has to answer that. It also has to answer a question sc-17497 never had to: nothing in the
crate other than the test harness drives the measured route, so what would the replacement even be?

**It already exists.** `candle-gen-flux2` ships `examples/flux2-txt2img.rs`, a real dev txt2img driver
compiled with no `#[cfg(test)]` code. Measured on the same six mutations it is the best-performing
unit in the matrix: it absolves `M-cfgtest` and `M-editprov`, and it catches `M-rms` and `M-devfp`.

That last pair is the load-bearing part, and it is what makes the unit credible rather than merely
quiet: `M-rms` proves the example links the transformer numerics on the measured path, and `M-devfp`
proves it links the memory-strategy contract table carrying `CALIBRATION_FINGERPRINT`. A unit that
absolved everything would be indistinguishable from a broken one.

### The gap this unit opens, and the check that closes it

The audited artifact stops being the binary that produced the measurements. The claim weakens from
"the code the measurements ran is unchanged" to "the production code the measured route links, as
linked into an artifact that drives that route, is unchanged". Two things follow, and both are
obligations rather than caveats:

1. **What the example does not link is invisible.** `M-rms` and `M-devfp` establish two reachable
   surfaces by mutation; they do not establish the whole measured route. The memory-strategy
   **admission** path — `registered_safety_check` → `admission_safety_check` → `validate_context` —
   is driven by `run_probed_offload_ab` and is not obviously driven by a plain txt2img example. A
   change there alters which requests the calibrated ladder accepts, which is squarely inside what
   the calibration authorizes, so if the example is blind to it this candidate is disqualified.
   Measured in [the safety-check probe](#the-safety-check-probe) below.
2. **The example's continued existence becomes load-bearing.** It is an `examples/` file today, with
   no test asserting it drives the dev route, and deleting or rewriting it is currently a free act
   upstream. Adopting it as the audited unit means the audit has to fail loudly if the target stops
   existing or stops exercising the route — the same discipline `COMPOSITION_ONLY_CRATES` applies to
   the closure list.

### The safety-check probe

`M-safety` widens the admission guard in `memory_strategy.rs::validate_context` from
`context.geometry.batch != 1` to `batch > 2`. That is not a cosmetic edit: it admits a two-image
batch into a ladder whose own rejection message reads *"memory calibration is single-image only"* —
a change that alters which requests the calibrated ladder accepts, with no compensating measurement.
If the non-test artifact were blind to it, candidate 4 would be disqualified as a gate.

| unit | verdict |
| --- | --- |
| shipped whole test binary | DIFFER ✅ |
| **non-test example binary** | **DIFFER ✅** `8fd614b0…` → `cac8959d…` |
| reachable from the measured test | DIFFER ✅ |
| ladder-scoped | DIFFER ✅ |

So `examples/flux2-txt2img.rs` links the memory-strategy **admission** path, not merely the contract
table. Together with `M-rms` (transformer numerics) and `M-devfp` (the calibration identity), three
independent mutations now place the three surfaces that matter inside the non-test artifact.

That is coverage established by mutation, which is the right standard here and is still not a proof
of completeness — it says these three surfaces are covered, not that every surface is. The
[recommendation](#recommendation) makes that obligation explicit rather than leaving it implied.

This case also carries an unplanned control worth keeping: it ran in a **separate session**, in a
re-created worktree, hours after the main matrix, and its `fbb0-a` row reproduced the earlier one
byte-for-byte on the shipped unit, the example binary *and* the object-level reachable digest.
Reproducibility across processes and worktree lifetimes, not just within one run.

---

## Candidate 5 — a behaviour witness: run the ladder instead of hashing it

The story's title asks for a unit that tracks behaviour, not bytes, and every candidate above is
still bytes. The obvious fifth option is to stop hashing code and start hashing **what the code
says**: execute the registered memory contract for `flux2_dev` at both revisions and hash a canonical
rendering of the result — capabilities and support levels, strategy parameters, decode tile edges and
overlaps, transformer block counts, the calibration identity — plus the safety-check decision over a
fixed grid of `MemoryRunContext` probes.

The infrastructure is already there and is weights-free and GPU-free:
`crates/contracts/gen-core-testkit/src/memory_strategy.rs` walks every memory-strategy registration
in a catalog (`check_memory_strategy_registry`), running the static contract conformance and driving
admission probes on each declared route. A witness is that walk with a canonical serialisation
instead of assertions.

**It is disqualified as a gate, and the measurement above already says so.** The witness's output is
a pure function of the ladder subgraph — the code rooted at `provider_contract_for` and
`registered_safety_check`, which is exactly the `ladder` scope measured here. `M-rms` leaves that
subgraph byte-identical (`924a6b1b…` at both), so the witness it produces is identical too, and the
story's named known-live change sails through. (An implication from a measurement, not a direct
measurement: it holds because the ladder code and its inputs are the witness's only determinants.)

That is not a flaw in the idea, it is the boundary of what "behaviour" means here. The contract table
is the *declared* ladder; the calibration measures *realised* VRAM. A change to attention chunking or
decode tiling moves the second without moving the first. Any witness of this shape must therefore
compose with a code digest rather than replace one.

**What it is good for is triage, and that is worth having.** It is the only candidate that answers
"did the ladder change?" in the vocabulary the calibration is written in, and unlike the artifact
layer it does not need the audited target to be the measurement binary — so it is the one candidate
that generalises to the five providers with no artifact layer at all (26 of the 31 bound
calibrations). It does need a build at both revisions and an inference-side dumper (~50 lines: a
`#[test]` that prints a canonical rendering), so it carries the same pin-bump self-tax as candidate 3
— once.

**Verdict: not the unit; file as the strongest follow-up.** Recommended as the second axis of a
triaged verdict, not as an authorization.

---

## How any new unit has to compose with sc-17606 and sc-17607

Neither existing layer is replaceable by a better code digest, and one of them constrains the
candidates in a way worth stating before the measurements.

**The resolved-feature witness (sc-17606) stays necessary, whatever the code unit becomes.** The
change it looks for — a crate *outside* FLUX.2's closure enabling a feature on a dependency FLUX.2
shares — moves no closure object and, critically, does not move the measurement build's code either:
the measurement build resolves features over a strictly smaller graph (`-p candle-gen-flux2`) than
the shipped bundle (`-p runtime-cuda`). Narrowing the code unit makes the witness *more* load-bearing,
not less, because a narrower unit sees strictly less.

**The composition check (sc-17607) is untouched.** "Which provider is registered under `flux2_dev`"
is not a codegen question in either direction, which is why it lives in
`crates/sceneworks-worker/src/flux2_composition_audit.rs` as a pointer-identity test against the
composed catalog rather than as a digest.

**A narrower unit takes on a soundness obligation the current one does not have.** The audit
adjudicates a moved closure path by *compile coverage*: `coveredClosurePaths` says a crate tree is
covered when cargo reports having compiled a package under it. That inference is exactly right for a
whole-binary digest — compiled and linked means "in the hashed bytes". It is **not** right for a
reachability-scoped digest, where a package can be compiled, linked, and still outside the hashed
set. Adopting a reachable-set unit means `adjudicates` has to be re-derived from the reachable set
rather than from the compile report, or the record will claim to have adjudicated a `gen-core` move
its digest never looked at. That is the same false-green shape sc-17497 refused when it excluded
`runtime-cuda` from `adjudicates`, arriving by a different route.

The same obligation applies, in a weaker form, to the recommended non-test target: a package cargo
compiled for the *test* build may not be linked into the example. Two of the nine closure paths are
already in that position by construction (`gen-core-testkit` never ships), so `adjudicates` has to be
derived from the artifact that is actually hashed rather than from the test build's compile report.

---

## Recommendation

**Change the audited artifact to a non-test binary, add the fingerprint as a veto, and keep
everything else exactly as it is.** In order of what each part is for:

| # | Change | What it is |
| --- | --- | --- |
| 1 | Audit `examples/flux2-txt2img` instead of the `--lib` test binary | the gate — narrower by two measured classes, no measured false green |
| 2 | Record `inferenceFingerprint` at both revisions and **fail** when it moved | a second, independent way to fail; no authority when it holds still |
| 3 | Keep the resolved-feature witness (sc-17606) and the composition check (sc-17607) unchanged | they answer questions no code digest can |
| 4 | Re-derive `adjudicates` from the audited artifact rather than the test build's compile report | the soundness debt (1) creates |

### Why this and not the reachability walk

The reachability walk is the more interesting answer and the worse trade. It absolves one class
(`M-editprov`); the non-test target absolves that one **and** the `#[cfg(test)]` class, which the walk
provably cannot. It costs a bespoke COFF parser, a Mach-O parser for the Metal lane, a build that
retains objects, a normalisation layer whose completeness is only ever evidenced by a control that
must be re-run against each toolchain, and a re-derivation of `adjudicates`. The non-test target costs
one `cargo build --example` and a target-existence assertion.

### What the recommendation does NOT cover

Stated as obligations, because the story asked for exactly this and because a list of gaps is only
useful if something else carries each one.

| Gap | Carried by |
| --- | --- |
| A crate-mate change that reaches code `flux2_dev` genuinely shares — `M-klein` | **Nothing, and nothing can.** Since sc-15831 dev and Klein share `provider_contract_for`/`profile()`. This is now a *deliberate* over-trigger with a named cause, not an unexplained one. |
| A feature change outside FLUX.2's closure reaching a shared dependency | the resolved-feature witness (sc-17606), unchanged and now more load-bearing |
| Which provider is registered under `flux2_dev` | the composition check (sc-17607), unchanged |
| A production surface the example does not link | **Not closed.** Three mutations place three surfaces inside it; completeness is not established. Item (1) must ship with a test asserting the example still drives the dev route, and any future ARTIFACTS-DIFFER triage should widen the mutation set rather than trust this table. |
| Whether the *declared* ladder changed, in the calibration's own vocabulary | **Nothing today** — candidate 5, the strongest follow-up |
| Realised VRAM changing while the declared ladder holds still | the code digest; this is precisely why candidate 5 cannot replace it |
| SceneWorks' own feature unification widening FLUX.2's linked packages | sc-17639, unchanged |
| The five providers with no artifact layer at all — 26 of 31 bound calibrations | **Nothing.** Item (2) is the only part of this recommendation that reaches them, and only as a veto. |
| A ladder that is registered upstream but never admitted by the worker selector — sc-17775 §9's exposure 3 | **Nothing here, and nothing here should.** That is an admission bug, not an invalidation one: the calibration is valid and simply never consulted. A better invalidation unit cannot see it, and a reader who takes this recommendation as covering it will be wrong in the expensive direction. |

That last row is the one to weigh against the rest. This recommendation improves the one provider
whose relief mechanism already exists. sc-17775 measured the larger exposure as the five that have
none, and neither this story's candidates nor the epic's framing address them — candidate 5 is the
only one that would, because it does not require the audited target to be the measurement binary.

### Cost estimate

| Item | Estimate |
| --- | --- |
| (1) switch the audited target, `AUDIT_ARTIFACT_TARGET` + `selectMeasurementExecutable` + the target-drives-the-route assertion | half a day |
| (2) `inferenceFingerprint` in the record, the veto in both validators, frozen constants | half a day |
| (4) re-derive `adjudicates` from the audited artifact | half a day |
| record schema `v6`, refusal of `v5` in both validators, `docs/inference-artifact-audit-sc-17497.md` | half a day |
| one re-run of the shipped window to emit a `v6` record, warm | ~15 min of machine time |
| **total** | **~2 days, SceneWorks only — no inference commit, no pin bump** |

For comparison: candidate 1 is 2–3 days *plus* a permanent toolchain-coupled maintenance obligation;
candidate 3 is a multi-day upstream refactor plus a pin bump that demotes every `current`
calibration; candidate 5 is ~1 day plus one inference commit and its pin bump.

None of them shrinks sc-17760's re-capture, which [the first section](#the-finding-that-reframes-the-epic)
shows is genuinely owed.
