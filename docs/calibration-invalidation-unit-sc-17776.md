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

**The epic's worked example is an over-trigger, and no digest of compiled code can see that it is.**

The window it is built on — `5ffd7612 → fbb00d6b`, where `flux2_dev`'s `CALIBRATION_FINGERPRINT` is
untouched — moves every code-based unit measured here, down to the memory-ladder subgraph alone:

| unit | `5ffd7612` | `fbb00d6b` | |
| --- | --- | --- | --- |
| shipped whole test binary | `d80844f2…` | `fee1c2de…` | DIFFER |
| non-test example binary | `1ffa3084…` | `8fd614b0…` | DIFFER |
| reachable from the measured test | `1c40ed58…` | `aaa1297d…` | DIFFER |
| reachable from the memory ladder alone | `35706634…` | `924a6b1b…` | DIFFER |
| **the ladder `flux2_dev` declares** | `5b3dd71f…` | `5b3dd71f…` | **IDENTICAL** |

The last row is a [behaviour witness](#candidate-5--a-behaviour-witness-run-the-ladder-instead-of-hashing-it):
the registered provider contract rendered over a grid of load specs — every rung's support level and
parameters, the decode tile edges and overlaps, the transformer window, the calibration identity —
executed at both revisions and compared. It is byte-identical, all 18,764 characters of it.

Two supporting facts, both checkable in seconds:

- **The window touches no numerics.** Its entire footprint inside FLUX.2's nine-path closure is
  `Cargo.lock`, `gen-core/src/residency.rs` (+30, an MLX Anima addition), `candle-gen/src/preview.rs`
  (+42, doc comments — sc-16961's original false positive), and three `candle-gen-flux2` files:
  `lib.rs`, `memory_strategy.rs`, `edit_provider.rs`. `transformer.rs`, `vae.rs`, `pipeline.rs` and
  `quant.rs` do not move.
- **The dev-affecting diff is a refactor.** `41464d57` (sc-15831) merged dev's and Klein's ladders:
  `provider_contract(spec)` now delegates to `provider_contract_for(provider_id, spec)` reading a
  `ProviderProfile` from a shared `profile(provider_id)` match. For `FLUX2_DEV_ID` that profile
  returns exactly the five values the old dev-only body used inline — which is *why* the witness does
  not move.

**What this means for sc-17760.** The evidence now points away from the re-capture, not toward it.
It is not a proof: the witness covers the *declared* ladder, and a code change could in principle
move realised VRAM while leaving the declaration alone (`M-rms` below is exactly that shape). But no
numerics file moved and the declaration is byte-identical, which is a materially different position
from a bare `ARTIFACTS DIFFER`. Whether that is enough to skip a ~47.6 GB capture is a policy call
this research story does not have the authority to make — it is put on the record so the call can be
made on evidence.

**This sharpens sc-17775 rather than contradicting it.** That survey deliberately declined to call
row A a false positive — *"pretending otherwise would be reasoning dressed as measurement"*
(`docs/calibration-invalidation-survey-sc-17775.md` §4.1). It was right to decline: the measurement
that settles it did not exist yet. It does now, and it says the declared ladder held still.

## The answer in one table

Seven one-variable mutations on `fbb00d6b` (`docs/calibration/sc-17776/mutations.md`), each run
through six candidate units. ✅ = the unit gives the answer a sound gate should give.

| mutation | should say | shipped unit (today) | non-test binary | reachable (all) | reachable (`.text`) | ladder-scoped | behaviour witness |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `M-rms` — `RMS_EPS` `1e-5→2e-5` | DIFFER | DIFFER ✅ | DIFFER ✅ | DIFFER ✅ | DIFFER ✅ | **identical ❌** | **identical ❌** |
| `M-safety` — widen the admission guard | DIFFER | DIFFER ✅ | DIFFER ✅ | DIFFER ✅ | DIFFER ✅ | DIFFER ✅ | **identical ❌**¹ |
| `M-devfp` — dev fingerprint bump | DIFFER | DIFFER ✅ | DIFFER ✅ | DIFFER ✅ | **identical ❌** | DIFFER ✅ | DIFFER ✅ |
| `M-gencore` — unreferenced `gen-core` fn | identical | identical ✅ | identical ✅ | identical ✅ | identical ✅ | identical ✅ | identical ✅ |
| `M-editprov` — klein-edit provider edit | identical | DIFFER ❌ | **identical ✅** | **identical ✅** | identical ✅ | identical ✅ | identical ✅ |
| `M-cfgtest` — a `#[cfg(test)] mod` | identical | DIFFER ❌ | **identical ✅** | DIFFER ❌ | DIFFER ❌ | identical ✅ | identical ✅ |
| `M-klein` — klein fingerprint bump | identical | DIFFER ❌ | DIFFER ❌ | DIFFER ❌ | identical | DIFFER ❌ | **identical ✅** |

¹ A gap in the probe, not in the idea — see [the admission gap](#the-admission-gap-a-limit-of-the-probe-not-of-the-idea).

Read the two false-green columns as disqualifications, not near-misses: a `.text`-only digest cannot
see a provider **declaring its own calibration void**, and neither a ladder-scoped digest nor a
behaviour witness can see a numerics change on the measured path. A false green is the one failure
mode this epic must not introduce.

Four results decide the recommendation:

1. **No single unit is sound.** Every unit that absolves anything the shipped unit over-triggers on
   also opens a false green somewhere, except one — see (2). The answer is therefore two axes, not a
   better single digest.
2. **Narrowing the audited target to a non-test binary is the only *code* candidate that absolves a
   class without opening a false green** — and it absolves two: `#[cfg(test)]` additions and
   crate-mate code the measured route does not link.
3. **The behaviour witness is the only unit that absolves the crate-mate class** (`M-klein`) **and
   the epic's own window**. It is also blind to numerics, so it can never be the gate.
4. **Reachability scoping does not absolve `#[cfg(test)]`**, the class it was most expected to
   remove — see [why](#why-reachability-does-not-absolve-cfgtest).

---

## Method

Every row is a real build. The instruments are checked in so the numbers can be re-derived and so
their limitations are readable rather than implied:

| | |
| --- | --- |
| `scripts/research/sc-17776-reachability-probe.mjs` | parses COFF objects, walks symbol relocations from a named root, digests the reachable set |
| `scripts/research/sc-17776-measure.mjs` | drives the builds per case under `inference-artifact-audit.mjs`'s exact discipline and runs the probe over what the linker was handed |
| `docs/calibration/sc-17776/behaviour-witness/sc17776-witness.rs` | the behaviour witness, applied to a scratch worktree by `git apply` and never committed to inference |

None of them is wired into a gate, a test list or CI. They are measuring instruments for this story.

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

`--profile bench` is what `cargo test --release` selects, and inference's root `Cargo.toml` declares
no `[profile.bench]` overrides at this pin, so the two builds differ only in the trailing rustc
arguments. That is a property of the current manifest rather than a guarantee; if a `[profile.bench]`
block ever appears, the two columns stop being comparable and this note is where to look.

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

Each `--cases` entry is `label=revision[+patchfile]`. The patch files are **not** checked in — the
`patch` paths recorded in the JSON are scratch paths from the measuring session and will not exist
for a re-runner. `docs/calibration/sc-17776/mutations.md` reproduces every edit as a `sed`/append
recipe against `fbb00d6b`; regenerate the patches from it with `git diff > <name>.patch`. `M-safety`'s
recipe is line-addressed and is only valid at that pin.

The behaviour witness is run separately, because it executes rather than hashes:

```bash
git apply docs/calibration/sc-17776/behaviour-witness/sc17776-witness.rs.patch   # regenerate from the .rs
cargo run -q --profile bench --locked -p candle-gen-flux2 --features cuda --example sc17776-witness
```

### Why the reachability walk is over objects rather than the PE

The story frames candidate 1 as "a symbol + relocation walk of the PE". That is not possible on this
artifact, and the reason is the audit's own determinism fix: `/DEBUG:NONE` means no PDB is emitted,
and an MSVC-linked image carries no COFF symbol table. The linked binary has no symbol boundaries and
no symbol-level relocations left — only base relocations. A reachability walk of it would mean
disassembling 181 MB of x86-64.

The pre-link objects have what is missing: one COMDAT section per function, a symbol table, and a
relocation for every cross-section reference. That is also the graph `link.exe /OPT:REF` walks, so a
BFS over it approximates link-time DCE closely rather than from the far side — with one known
under-approximation, recorded in [limitations](#instrument-limitations). **This is a correction to
the story's framing, not a shortcut**, and it changes the cost estimate: the implementation has to
keep objects a normal build throws away.

### What the reachability digest covers

Per reachable section: the canonical section kind, the raw bytes, and the `(offset, type, target
symbol name)` of every relocation out of it. The bytes alone are not enough — a call site in an
object is a zero placeholder plus a relocation, so two functions differing only in which callee they
call have identical bytes.

Two normalisations, both forced by measurement rather than chosen:

- **Container identity is not hashed.** Codegen-unit names move when modules are reshuffled.
- **`.llvm.<n>` suffixes are stripped from symbol names.** LLVM appends a module-derived number when
  it internalises a symbol, so the suffix moves when *any* code in the crate moves. Without this the
  probe reported `5ffd7612` and `a4f409ae` as different — a window the shipped audit certifies as
  producing byte-identical binaries.

---

## Instrument limitations

Stated up front, because a research instrument that flatters its own candidate is worse than no
instrument.

1. **The object graph is noisier than the linked image, and the noise control is partial.** One
   source (`.llvm.<n>`) was found and removed. The control that says whether any remain is the
   certified-identical window `5ffd7612 → a4f409ae`, on which all eight measured digests agree — but
   that window's closure delta is `Cargo.lock` plus `candle-gen`, and it **never touches
   `candle-gen-flux2`**. It therefore does not bound *in-crate* noise, which is precisely what the
   `M-cfgtest` result turns on. `M-editprov` is a partial in-crate control and a genuinely
   load-bearing one, but it edits an existing line rather than adding a module.
2. **COMDAT selection and associativity are not modelled.** `parseCoff` skips auxiliary symbol
   records, which is where `IMAGE_COMDAT_SELECT_ASSOCIATIVE` lives. `.pdata`/`.xdata` unwind sections
   point *at* `.text` rather than the reverse, so a reachable function's unwind data is never
   reached. That is an under-approximation of what `/OPT:REF` keeps, **in the false-green direction**:
   a change confined to unwind or EH data would be invisible to the reachability digest. Not
   exercised by any mutation here, and an undisclosed instance of it would be a defect in candidate 1
   rather than in this measurement.
3. **Native `.lib`/`.a` link inputs are dropped silently.** Both the link-input extraction and the
   object loader match only `rlib|o|obj`, so import libraries and native static archives are never
   listed and therefore never reported as parse failures — the omission reads as clean input. It does
   not affect these results (the CUDA kernels reach the binary as PTX `include_str!`ed into Rust
   `.rdata`), but it is a real limit on candidate 1 that its cost section has to carry.
4. **One unexplained residual.** `M-devfp` and `M-klein` are same-length string-constant edits in the
   same file. `M-klein` leaves the whole-object `.text`-only control identical, as expected of a
   `.rdata` change; `M-devfp` moves it (`43756b46…` → `c2800d4f…`). No verdict in this document rests
   on that control, but the model of what `.text` sees is incomplete and it is recorded rather than
   dropped.
5. **Reachability is static, not path-sensitive.** Every branch of a reachable function is reachable,
   including arms for other providers. Not a defect to fix — a sound walk cannot assume a branch is
   untaken — but it bounds what candidate 1 can absolve, and the measurements show that bound binds.
6. **One provider, one lane.** Everything is `flux2_dev` on `cuda`. The five other calibrated
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
| same revision, **separate session**, re-created worktree (`fbb0-c`) | identical on the shipped unit, the example binary and the object-level reachable digest |
| shipped unit reproduces the frozen records | `d80844f2…` at `5ffd7612` (= `FLUX2_COMPATIBILITY_AUDIT.artifactProof`), `fee1c2de…` at `fbb00d6b` (= sc-17760's record) |
| the certified-identical window `5ffd7612 → a4f409ae` | identical on **all eight** digests, object-level included |
| `M-gencore` (link-time-DCE control) | identical — reproduces sc-17775's M4 on the shipped unit too |
| sc-17775's M1/M3 shipped digests | `73fe78d5…` / `eb9a6ead…` reproduce exactly |
| rlib parse failures | 0 across 133 rlibs + 17 loose objects (886 COFF objects) |

The certified-window row did **not** hold on the first attempt. Un-canonicalised, the object digests
reported `5ffd7612` and `a4f409ae` as different while their linked binaries were byte-identical — a
false positive of the candidate, not of the code, caused by LLVM's `.llvm.<n>` internalisation
suffix. That is recorded rather than quietly fixed because it is the shape of the maintenance burden
this candidate carries: **the object graph is noisier than the linked image, and every source of that
noise has to be found and normalised away, one toolchain at a time.**

### What it buys

| | sections | bytes |
| --- | ---: | ---: |
| all linked objects | 204,470 | 78.1 MB |
| reachable from the measured test | 35,800 (17.5%) | 44.9 MB (57.4%) |
| reachable from the memory ladder alone | 1,256 (0.6%) | 0.19 MB (0.2%) |

An 82% narrowing by section count, but only 43% by bytes — the reachable set is the *large* sections.
On the measured mutations that narrowing buys exactly one absolution: `M-editprov`, a bespoke
klein-edit provider that is registered nowhere and that the dev route never links. Real, and the
class is real — `edit_provider.rs` is 1,091 of the crate's 11,218 lines and took 122 lines of the
last Klein window alone.

### Why reachability does not absolve `#[cfg(test)]`

`M-cfgtest` appends a `#[cfg(test)] mod` with one trivial test. Nothing in it is reachable from the
measured test, and the reachable digest moves anyway.

Measured, rather than guessed at: dumping the reachable set on both sides and diffing them,
**33 of 35,800 sections moved (0.09%), with zero sections entering or leaving the set** and the total
byte count unchanged at 44,854,083. Every moved section is an anonymous `.text`/`.rdata` COMDAT with
no defining external symbol.

That is consistent with rustc repartitioning codegen units — it partitions by module, so adding one
changes which functions are internalised and which cross-CGU calls inline — and the non-test example
binary being identical is consistent with the effect being confined to test builds. **But the
mechanism is inferred, not proven**: nothing here instruments rustc, and the example-binary control
cannot discriminate, because `#[cfg(test)]` code does not exist in a non-test compilation and that
binary was *always* going to be identical. What is measured is the discrimination and its size.

The consequence for this story does not depend on the mechanism: **while the audited artifact is a
test binary, a code digest refuses on this class, whatever it is scoped to** — and it refuses over a
33-section residue nobody can attribute. Narrowing the *target* removes the question instead of
answering it.

### Cost, and the two costs the story asked about specifically

- **Does it survive `#[inline]` and generic instantiation?** Yes, by construction: the walk is over
  the objects the linker was handed, so an inlined or monomorphised body sits in the *consumer's*
  codegen unit and is reached through the consumer. That is the failure sc-17497 measured when it
  rejected per-crate `.o` hashing; scoping to the reachable set across all objects avoids it rather
  than repeating it. `M-rms` is the check — `RMS_EPS` feeds `rms_norm` call sites in
  `transformer.rs`, and the reachable digest moves.
- **Is the walk deterministic across builds under `/Brepro /DEBUG:NONE`?** Yes for the walk; the risk
  moved somewhere worse. Those flags stabilise the *linked image*; the objects were already stable,
  but the object graph carries content-derived symbol suffixes the linker discards. One was found and
  normalised. There is no argument that it was the last one — only the certified-window control,
  which [limitation 1](#instrument-limitations) shows does not cover in-crate noise.
- **Maintenance against a toolchain bump.** The unit depends on rustc's symbol mangling, CGU naming
  and COMDAT emission, and on MSVC's archive long-name convention — the MSVC/GNU difference silently
  produced an *empty* graph during this research, a failure that reads as a stable digest. A
  `rust-toolchain.toml` bump already hard-stops the audit (`assertComparableToolchains`), which
  contains the risk without removing the re-validation work.
- **A second parser for the Metal lane.** The probe is COFF-only; `--lane metal` would need Mach-O or
  the layer becomes cuda-only.
- **The build has to keep objects**, plus a `--print=link-args` parse or equivalent, at every audited
  revision.
- **Associativity and native archives** ([limitations 2 and 3](#instrument-limitations)) are open
  work before this could be a gate, and the first is in the false-green direction.

Rough estimate: **2–3 days** for the probe, schema bump, both validators, frozen constants and doc —
plus a fresh determinism baseline of the order of sc-17524's thirteen builds, plus the `adjudicates`
re-derivation in [composition](#how-any-new-unit-has-to-compose-with-sc-17606-and-sc-17607).

**Verdict: reject.** It is sound in the direction it was designed for and measurably narrower, and it
absolves one real class. It costs a bespoke object-graph parser, a Mach-O parser, a permanent
toolchain-coupled maintenance obligation and two open false-green questions — to absolve a class that
[candidate 4](#candidate-4--narrow-the-audited-target) absolves for the price of one `cargo build`,
alongside a second class candidate 1 provably cannot touch.

---

## Candidate 2 — grade the provider's own `CALIBRATION_FINGERPRINT`

The constant exists, upstream maintains it, and the audit ignores it. The question is what it may be
allowed to decide.

### It cannot authorize, and the reason is measured rather than argued

The known-live mutation — `RMS_EPS` `1e-5 → 2e-5`, the change sc-17524 used to prove the digest is
not blind — **does not move the fingerprint**. Nothing about a numerics edit reaches a string
constant in `memory_strategy.rs`. A gate reading "the fingerprint held still, so the calibration
extends" authorizes it. That is the false green the story names, produced on demand.

The history says the same thing more strongly. Over 60 days on inference `main`:

| | |
| --- | ---: |
| non-merge commits touching `candle-gen-flux2/src` | 41 |
| …that changed the **value** of `flux2_dev`'s `CALIBRATION_FINGERPRINT` | **4** |
| …and all four are dated 2026-08-04, while the ladder itself was being authored | ✅ |
| commits touching `memory_strategy.rs` since the current value was set (`51622dcb`) | 2 — both Klein, neither bumped it |
| `flux1`'s shared `CALIBRATION_FINGERPRINT` value changes since introduction | **0** |

So the fingerprint has **no observed post-capture variance at all**. Its entire history is the author
revising their own ladder before anyone captured against it. A signal that has never moved after a
capture cannot distinguish a valid extension from an invalid one — it says "extend" every time.

### Nothing makes the bump reliable, and that is a cross-repo problem

Per the story's framing, grading this makes inference authors' discipline load-bearing for SceneWorks'
calibration validity. Measured state of that discipline:

- **No CI check in inference** connects a fingerprint to anything. The two `.github/` matches are a
  `--expected-fingerprint` argument in the real-weights workflow and an unrelated oracle-producer
  string; neither fails when a provider's compiled surface moves without its fingerprint.
- The one declared-vs-observed comparison that exists is **capture-side and in SceneWorks**
  (`scripts/sc-15833-flux2-evidence.mjs:403-412`), and its "observed" side is
  `MEMORY_EXPECTED_FINGERPRINT` — an environment variable the operator sets. It proves the operator
  and the provider agree at capture time. It proves nothing about a later revision.

**What would make it reliable, if someone wanted to go that way.** The check that would earn the
fingerprint real authority is upstream and does not exist: a test in inference that fails when a
provider's compiled surface moves without its fingerprint moving. That is the artifact audit again,
one repo to the left, and it inherits every problem this document catalogues — so it is not
recommended. A cheaper partial is a review checklist, which is discipline with extra steps and
carries no evidence. **The honest answer to "what makes the bump reliable" is: nothing does, and
nothing cheap would**, which is exactly why the recommendation below gives the fingerprint no
authority to authorize.

### What it may decide: a veto, never a pass

There is a sound, cheap use, and it is the opposite of the one the framing suggests. Record
`inferenceFingerprint` at **both** revisions, read from inference source, and let it **fail** the
audit when it moved — while giving it no authority when it held still.

That asymmetry is what makes it safe: a veto can only add refusals, so it cannot manufacture a false
green. And it closes a hole the artifact layer only closes by luck. sc-17775's finding 1 notes that
`flux2_dev`'s constant happens to be linked into the audited binary today (`memory_strategy.rs:76`),
so a bump moves the digest — a property of the current code, not a guarantee. Move the constant
behind a `#[cfg]` or into a contract table the measured path does not reach and the digest stops
seeing it. The veto does not depend on that.

**What the veto does *not* do — a correction worth being explicit about.** It compares the
fingerprint at two *revisions*. sc-17775's finding 3 is a different problem: gate 1 compares
`record.calibrationFingerprint` against `cell.calibrationFingerprint`, and **both are SceneWorks'
own copies**, so a value transcribed wrong into both binds happily while inference advertises
something else. A revision-to-revision veto never compares against the bound string and therefore
does not close it. Closing that needs a *declared-vs-source* check — assert every bound
`calibrationFingerprint` against the constant in the pinned inference source at matrix-generation
time. That is cheap, SceneWorks-only, needs no build, and is the **only** thing in this document that
reaches all six calibrated providers. It is listed separately in the
[recommendation](#recommendation) for that reason.

**Verdict: adopt, as a veto only.** It is not the unit; it is a second, independent way to fail.

---

## Candidate 3 — split the multi-provider crates upstream

Four crates host more than one calibrated provider, and the split argument is that a sibling's commit
would then stop moving the audited binary.

Measured against the epic's own worked example, it does not. `41464d57` did not *add* Klein alongside
dev — it **merged their ladders**, and two of the three files it moved (`lib.rs`,
`memory_strategy.rs`) are files dev compiles. A split into `candle-gen-flux2-dev` and `-klein` has to
either duplicate the shared ladder (undoing sc-15831 and reintroducing the drift it removed) or put
it in a third crate both link — in which case dev's binary still moves on every Klein ladder revision.
The class a split addresses is *independent* providers sharing a crate, and FLUX.2 is no longer one.

Where it would pay is `candle-gen-flux`: `flux1_dev` and `flux1_schnell` share **one**
`CALIBRATION_FINGERPRINT` with a single use site, so a schnell-only revision invalidates dev at gate
1, where no audit layer can help. But those two also have no artifact layer at all, so the split is
not the first thing they need.

**Cost, with the self-tax.** SceneWorks SHA-pins inference, so landing a split requires a pin bump —
the exact event that demotes every `current` calibration (sc-17775: ~1.5 pin bumps/day, 11
calibrations `current` today). The refactor is the smaller half: `candle-gen-flux2` is 13 files and
11,218 lines, with five downstream dependents in inference plus SceneWorks' own consumers.

**Verdict: reject for FLUX.2** — it cannot absolve the case that motivated the epic. Record as a
possible future for `candle-gen-flux`, behind that family getting an artifact layer at all.

---

## Candidate 4 — narrow the audited target

Hash a **non-test** binary built from the same tree, so the `#[cfg(test)]` class disappears at the
source rather than being accepted.

sc-17497 chose the test binary deliberately — it is the one that produced the measurements — and any
move here has to answer that. It also has to answer a question sc-17497 never faced: nothing but the
test harness drives the measured route, so what would the replacement be?

**It already exists.** `candle-gen-flux2` ships `examples/flux2-txt2img.rs`, a real dev txt2img driver
compiled with no `#[cfg(test)]` code. On the same seven mutations it is the best-performing *code*
unit: it absolves `M-cfgtest` and `M-editprov`, and it catches `M-rms`, `M-devfp` and `M-safety`.

Those three catches are the load-bearing part, and they are deliberately on three different surfaces
so a unit cannot pass by linking one of them: the transformer numerics, the calibration identity, and
the admission guard. A unit that absolved everything would be indistinguishable from a broken one.

### The safety-check probe

`M-safety` widens the admission guard in `memory_strategy.rs::validate_context` from
`context.geometry.batch != 1` to `batch > 2`. That is not cosmetic: it admits a two-image batch into
a ladder whose own rejection message reads *"memory calibration is single-image only"*. If the
non-test artifact were blind to it, candidate 4 would be disqualified as a gate.

| unit | verdict |
| --- | --- |
| shipped whole test binary | DIFFER ✅ |
| **non-test example binary** | **DIFFER ✅** `8fd614b0…` → `cac8959d…` |
| reachable from the measured test | DIFFER ✅ |
| ladder-scoped | DIFFER ✅ |

So the example links the memory-strategy **admission** path, not merely the contract table.

### The gap this unit opens

The audited artifact stops being the binary that produced the measurements. The claim weakens from
"the code the measurements ran is unchanged" to "the production code the measured route links, as
linked into an artifact that drives that route, is unchanged". Three obligations follow:

1. **Coverage is established by mutation, not by proof.** Three surfaces are inside it; completeness
   is not established. Any future ARTIFACTS-DIFFER triage should widen the mutation set rather than
   trust that table.
2. **The example's continued existence becomes load-bearing.** It is an `examples/` file with no test
   asserting it drives the dev route, and deleting or rewriting it is currently a free act upstream.
   The audit has to fail loudly if the target stops existing or stops exercising the route — the
   discipline `COMPOSITION_ONLY_CRATES` applies to the closure list.
3. **It is not a shipped-profile artifact either.** `cargo build --example` enables dev-dependencies,
   so `candle-gen`'s `testkit` feature and `sceneworks-gen-core-testkit` are still resolved. What
   goes away is `#[cfg(test)]` code, which is what the class needs; the measured-vs-shipped **feature**
   delta sc-17497 records stays exactly as it is, and a dev-dependency edit still moves the digest.

---

## Candidate 5 — a behaviour witness: run the ladder instead of hashing it

Every candidate above is still bytes. The fifth option is to stop hashing code and hash **what the
code says**: execute the registered memory contract for `flux2_dev` and render it canonically — every
rung's support level and parameters, decode tile edges and overlaps, transformer window, the
calibration identity, the resolved numeric tier — over a grid of load specs.

This was built and run, not reasoned about. The probe
(`docs/calibration/sc-17776/behaviour-witness/sc17776-witness.rs`, 43 lines) is an `examples/` file
applied to a scratch worktree by `git apply` and never committed to inference — the same status as the
mutation patches. It needs no weights, no GPU work and no device allocation. It is comparable across
the window by construction: `MemoryProviderContract` derives `Debug`, and
`crates/contracts/gen-core/src/memory_strategy.rs` is **byte-identical** between `5ffd7612` and
`fbb00d6b`, so the type being rendered cannot have changed shape.

| case | witness | |
| --- | --- | --- |
| `fbb00d6b` (baseline) | `5b3dd71f…` | |
| **`5ffd7612` — the epic's window** | `5b3dd71f…` | **identical** — the declared ladder did not move |
| `M-klein` | `5b3dd71f…` | **identical ✅** — the only unit that absolves the crate-mate class |
| `M-editprov`, `M-cfgtest`, `M-gencore` | `5b3dd71f…` | identical ✅ |
| `M-devfp` | `3aa8c73e…` | DIFFER ✅ — the rendered `MemoryCalibrationIdentity.fingerprint` moves |
| `M-rms` | `5b3dd71f…` | **identical ❌ false green** |
| `M-safety` | `5b3dd71f…` | **identical ❌** — see below |

**It cannot be the gate.** `M-rms` is a production numerics change on the measured path and the
witness cannot see it, because the contract table is the *declared* ladder while the calibration
measures *realised* VRAM. Any witness of this shape has that boundary; it must compose with a code
digest rather than replace one.

**It is the only unit that absolves the two things nothing else can** — the epic's own window and the
crate-mate class. That is not a small result: `M-klein` is refused by every code unit measured here,
including the narrowest, because dev and Klein genuinely share `profile()` since sc-15831.

### The admission gap: a limit of the probe, not of the idea

`M-safety` widens an admission guard and the witness does not move, because the probe renders
`provider_contract` and stops there. That is a **false green in the probe as built**, and it is
recorded as a hard design requirement rather than a caveat: a production witness must also drive the
admission path over a probe grid. The infrastructure already exists and already does exactly that —
`crates/contracts/gen-core-testkit/src/memory_strategy.rs::check_memory_strategy_registry` walks
every memory-strategy registration in a catalog and drives per-route admission probes, weights-free
and GPU-free. A witness is that walk with a canonical serialisation instead of assertions. Building
the probe the short way found the gap; that is what the probe was for.

### Cost and reach

It needs a build at both revisions and an inference-side dumper (~50 lines), so it carries a pin-bump
self-tax — once. In exchange it is the **only** candidate that does not require the audited target to
be the measurement binary, and therefore the only one that generalises to the five providers with no
artifact layer at all (26 of 31 bound calibrations).

**Verdict: adopt as the second axis, never as the gate.**

---

## How any new unit has to compose with sc-17606 and sc-17607

**The resolved-feature witness (sc-17606) stays necessary, whatever the code unit becomes.** The
change it looks for — a crate *outside* FLUX.2's closure enabling a feature on a dependency FLUX.2
shares — moves no closure object and does not move the measurement build's code either: that build
resolves features over a strictly smaller graph (`-p candle-gen-flux2`) than the shipped bundle
(`-p runtime-cuda`). Narrowing the code unit makes the witness *more* load-bearing, because a
narrower unit sees strictly less.

**The composition check (sc-17607) is untouched.** "Which provider is registered under `flux2_dev`"
is not a codegen question in either direction, which is why it is a pointer-identity test in
`crates/sceneworks-worker/src/flux2_composition_audit.rs`.

**Any narrower unit takes on a soundness obligation the current one does not have.** The audit
adjudicates a moved closure path by *compile coverage*: `coveredClosurePaths` treats a crate tree as
covered when cargo reports having compiled a package under it. That is exactly right for a
whole-binary digest — compiled and linked means "in the hashed bytes". It is **not** right once the
hashed artifact is something else. For a reachability-scoped digest a package can be compiled,
linked, and outside the hashed set; for the recommended non-test target a package cargo compiled for
the *test* build may not be linked into the example at all (`gen-core-testkit` never ships). Either
way `adjudicates` has to be derived from the artifact actually hashed, or the record will claim to
have adjudicated a move its digest never looked at — the same false-green shape sc-17497 refused when
it excluded `runtime-cuda` from `adjudicates`.

---

## Recommendation

**Two axes, neither of which may authorize alone, plus two cheap independent vetoes.**

| # | Change | What it is |
| --- | --- | --- |
| 1 | Audit `examples/flux2-txt2img` instead of the `--lib` test binary | the gate — narrower by two measured classes, no measured false green |
| 2 | Add a **behaviour witness** as a second recorded axis, covering the declared contract *and* the admission probes `check_memory_strategy_registry` already drives | the axis that can say a DIFFER is a refactor; never authorizes alone |
| 3 | Record `inferenceFingerprint` at both revisions and **fail** when it moved | a veto; no authority when it holds still |
| 4 | Assert every bound `calibrationFingerprint` against the pinned inference source at matrix-generation time | closes sc-17775's finding 3; the only item here that reaches all six providers |
| 5 | Keep sc-17606 and sc-17607 unchanged; re-derive `adjudicates` from the artifact actually hashed | composition, and the soundness debt (1) creates |

**How the two axes combine.** Digest identical ⇒ extend, as today. Digest DIFFER **and** witness
DIFFER ⇒ re-capture, and the record says which rung moved. Digest DIFFER **and** witness identical ⇒
*the declared ladder did not move*: recorded, reported, and **not** auto-authorizing, because `M-rms`
lands in that cell too. Whether a human may extend a calibration on that evidence is a policy
decision this story deliberately leaves open — but it is the cell the epic's own window falls into,
and today the epic has no way to even name it.

### Why this and not the reachability walk

The walk is the more interesting answer and the worse trade. It absolves one class (`M-editprov`);
the non-test target absolves that one **and** `#[cfg(test)]`, which the walk provably cannot. It costs
a bespoke COFF parser, a Mach-O parser for Metal, a build that retains objects, a normalisation layer
whose completeness is evidenced only by a control that does not cover the in-crate case, two open
false-green questions, and an `adjudicates` re-derivation. The non-test target costs one
`cargo build --example` and a target-existence assertion.

### What the recommendation does NOT cover

| Gap | Carried by |
| --- | --- |
| A crate-mate change reaching code `flux2_dev` genuinely shares — `M-klein` | **The behaviour witness, and only it.** No code unit can: since sc-15831 dev and Klein share `provider_contract_for`/`profile()`. |
| A numerics change on the measured path | the code digest (item 1); the witness is blind to it by design |
| A feature change outside FLUX.2's closure reaching a shared dependency | the resolved-feature witness (sc-17606), unchanged and now more load-bearing |
| Which provider is registered under `flux2_dev` | the composition check (sc-17607), unchanged |
| A production surface the example does not link | **Not closed.** Three mutations place three surfaces inside it; completeness is not established. Item (1) must ship with a test asserting the example still drives the dev route. |
| A change confined to unwind/EH data | irrelevant to items 1–4 (they hash whole images), but an open false-green question for candidate 1 if it is ever revisited |
| SceneWorks' own feature unification widening FLUX.2's linked packages | sc-17639, unchanged |
| A ladder registered upstream but never admitted by the worker selector — sc-17775 §9's exposure 3 | **Nothing here, and nothing here should.** That is an admission bug, not an invalidation one: the calibration is valid and simply never consulted. A reader who takes this recommendation as covering it will be wrong in the expensive direction. |
| The five providers with no artifact layer at all — 26 of 31 bound calibrations | **Item (4) only**, as a declared-vs-source check. Items (1)–(3) are `flux2_dev`-shaped because the audit record is. Item (2) is the one that *would* generalise — it needs no measurement binary — and that is the strongest argument for building it. |

That last row is the one to weigh against the rest. sc-17775 measured the larger exposure as the five
families that have no relief mechanism at all; items 1–3 improve the one provider that already has
one. Item 4 reaches all six cheaply, and item 2 is the only unit in this study that could ever be
extended to them.

### Cost estimate

| Item | Estimate |
| --- | --- |
| (1) switch the audited target: `AUDIT_ARTIFACT_TARGET`, `selectMeasurementExecutable`, the target-drives-the-route assertion | half a day |
| (2) behaviour witness: ~50-line inference dumper over contract + admission probes, the record field, both validators | 1–1.5 days **plus one inference commit and its pin bump** |
| (3) `inferenceFingerprint` veto in the record and both validators | half a day |
| (4) declared-vs-source fingerprint assertion at matrix generation | half a day, SceneWorks only |
| (5) `adjudicates` re-derivation | half a day |
| record schema `v6`, refusal of `v5` in both validators, `docs/inference-artifact-audit-sc-17497.md` | half a day |
| one re-run of the shipped window to emit a `v6` record, warm | ~15 min of machine time |
| **total** | **~3–4 days; items 1, 3, 4, 5 are SceneWorks-only (~2 days); item 2 is the only one that needs an inference commit** |

For comparison: candidate 1 is 2–3 days *plus* a permanent toolchain-coupled maintenance obligation
and two open false-green questions; candidate 3 is a multi-day upstream refactor plus a pin bump that
demotes every `current` calibration.

**On sc-17760.** Items 1–5 do not by themselves cancel that re-capture, but item 2 produces the
evidence that would let someone decide not to spend it — and that evidence has now been measured, in
advance, in [the first section](#the-finding-that-reframes-the-epic): the declared ladder is
byte-identical across the window, and no numerics file moved.
