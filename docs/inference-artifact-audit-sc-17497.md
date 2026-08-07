# Extending a FLUX.2 calibration across an inference pin bump (sc-17497)

> **SUPERSEDED by sc-17774 — the mechanism this documents no longer exists.**
>
> Calibration currency is now decided by a per-provider compile-closure digest, one mechanism applied
> identically to every model. See **[calibration-invalidation-sc-17774.md](calibration-invalidation-sc-17774.md)**.
>
> `scripts/inference-artifact-audit.mjs`, `FLUX2_COMPATIBILITY_AUDIT` and
> `crates/sceneworks-worker/src/inference_compatibility_audit.rs` are all deleted, so the procedure
> below **cannot be run**. It is kept because its reasoning is still load-bearing: the finding under
> "Why the source tree was the wrong unit" is exactly why the replacement hashes SEMANTIC source
> rather than raw bytes, which gives every lane the comment-absolution this audit bought for one.
>
> What did NOT go: the composition check (sc-17607,
> `crates/sceneworks-worker/src/flux2_composition_audit.rs`). "Which provider is registered under
> `flux2_dev`" is not an invalidation question and no digest can answer it in either direction.

A packaged calibration is a set of measurements taken against a specific build of the inference
engine. Extending it to a newer pin means proving the code it describes has not changed. sc-15833
proved that with **git object identity** over FLUX.2's Candle/CUDA compile closure; sc-17497 keeps
that as the cheap layer and adds a **compiled-artifact** layer underneath it, so a commit that
cannot change the compiled code no longer costs a re-capture. sc-17524 widened the closure to the
workspace build inputs it had been missing and made the Windows build reproducible enough for the
compiled-artifact layer to mean anything. sc-17606 added a third layer — a **resolved-feature
witness** — for the one change neither of the others can see: a crate outside FLUX.2's closure
enabling a feature on a dependency FLUX.2 shares. sc-17607 then narrowed the closure itself by one,
moving the composition question out of a layer that could never answer it.


## Why the source tree was the wrong unit

The v1 audit's unit is a crate tree, so any commit into one of them invalidates the proof. inference
`35251a88` added 42 lines of `//!` doc comment to `candle-gen/src/preview.rs`. Every other object
stayed byte-identical, that one moved, the five `flux2_dev` q4 cells dropped to `historical`, and the
only remedy on offer was a ~47.6 GB-peak capture on an RTX PRO 6000 — for a change that could not
possibly have altered a single instruction.

## What is hashed, and why it is the linked binary

"The compiled code is identical" is the claim, so the audit hashes the thing that runs. Three
candidate units were measured before choosing (fixture reproduced in
`scripts/inference-artifact-audit.test.mjs`):

| Unit | Doc-comment edit | `#[inline]` body `*7` → `*8` | Verdict |
| --- | --- | --- | --- |
| the rlib | **moves** | moves | Wrong — rustc metadata carries doc strings, so this reproduces the exact false positive. |
| the rlib's `.o` members | stable | **stable** | Wrong, and dangerous — an `#[inline]`/generic body is codegen'd in the consumer, so the defining crate's objects never move. A false green over most of a numerics crate. |
| the linked binary | stable | moves | **Correct.** Code that link-time DCE removes does not run, so excluding it cannot weaken the claim. |

The binary hashed is the one that produced the measurements: the `candle-gen-flux2` **lib test
binary** carrying `tests::flux2_dev_probed_generate_for_offload_ab` — the target named in the
approved capture command that `scripts/sc-15833-flux2-evidence.mjs` prints.

Confirmed on the real closure at the real revisions, **on the Metal lane**: `277f4238` and
`d2216f6b` — the pin bump the doc comment blocked — both produce
`sha256:7d69fc2665da1a453209820bd4310aabfdcada9c852d711f97424c797423caea`, and the digest is stable
across a `277f → d2216 → 277f` round trip in one worktree.

That is evidence the mechanism works and that this particular delta is codegen-inert; it is **not**
the proof, and no validator will accept it — a Metal record is refused on `lane`.

sc-17524 produced the CUDA-lane digests on the RTX PRO 6000 box, and doing so exposed that the
Windows link step was not reproducible at all until `/Brepro /DEBUG:NONE` — see **Determinism**
below. A Metal round trip being stable is not evidence that a Windows one is.

## The three layers

1. **Object identity (free).** Every closure object byte-identical ⇒ nothing is built and the proof
   extends. This is the common case and it stays as cheap as it is today.
2. **Artifact identity (one build).** A path moved ⇒ the record must carry a compiled-artifact
   digest, produced by building the measurement binary at *both* revisions under one toolchain.
   Equal digests extend the proof; unequal digests mean the compiled code really did change and the
   calibration must be re-captured.
3. **Shipped-feature identity (seconds, always).** The feature set the shipped bundle resolves for
   every package FLUX.2 links, hashed at both revisions and required identical. Unlike (2) this
   runs on **every** audit, including the free path — see *The resolved-feature witness* below for
   why that asymmetry is the whole point.

Both proofs are frozen in source — `FLUX2_COMPATIBILITY_AUDIT.artifactProof` /
`.featureWitness` in `scripts/generate-memory-matrix.mjs`, and `FLUX2_AUDIT_ARTIFACT_PROOF` /
`FLUX2_FEATURE_WITNESS` in `crates/sceneworks-worker/src/inference_compatibility_audit.rs` — so the
checked-in record cannot authorize itself.

## The closure: five crate trees and four build inputs

sc-15833 declared it as seven paths, all crate trees bar the root manifest. sc-17524 added three
more build inputs that feed every one of those builds, bringing that kind to four; sc-17607 removed
`crates/bundles/runtime-cuda`, leaving exactly what the measurement binary compiles plus the inputs
to that compile:

| Entry | Kind |
| --- | --- |
| `Cargo.toml`, `Cargo.lock`, `rust-toolchain.toml`, `.cargo/config.toml` | workspace build inputs |
| `crates/contracts/gen-core` | crate tree |
| `crates/media/candle-gen/candle-gen`, `-pid`, `-flux2` | crate trees |
| `crates/media/candle-gen/vendor/candle-kernels` | crate tree |

`Cargo.lock` was a realized gap, not a theoretical one: it moved between `5ffd7612` and `277f4238`
while all seven audited trees stayed byte-identical, so the free path printed *"no build needed"*
over a changed build input. That particular diff (`sha2` reaching `candle-gen-sensenova` and some
mlx crates) is inert for FLUX.2, but that was luck. A `cargo update` bumping a semver-compatible
transitive dependency the measurement binary links moves **only** the lockfile.

`.cargo/config.toml` is the same class and the easiest to miss, being neither at the repo root nor a
manifest — cargo reads it for every build in the worktree, and a `[build] rustflags`, a `[target.*]`
linker override or an `[env]` entry a build script consumes changes the compiled code while moving
no crate tree. It is object `61d7be37…` at every revision in play — `5ffd7612`, `277f4238`,
`06e0c5e9`, `a4f409ae`. One part of it the tool cannot honestly adjudicate:
`CARGO_ENCODED_RUSTFLAGS`, which this script must set for the path
remaps, **replaces** config-declared rustflags rather than merging them. So the audit refuses to run
against a config that declares any (`assertNoConfigRustflags`) rather than certify a build it had
itself stripped the flags out of. inference declares none today.

**Scope note.** These are inputs to the *measurement* build, inside the inference workspace. They
are not the worker's: SceneWorks consumes `runtime-cuda` as a git dependency, so cargo ignores
inference's `Cargo.lock` entirely and resolves against SceneWorks' own, and SceneWorks pins
`channel = "stable"` against inference's `1.96.0`. That is the correct scope for the claim being
made — "the code the measurements ran has not changed" — but it is not a statement about the
toolchain or lockfile the shipped worker is built with.

## A digest only speaks for what the binary links

`runtime-cuda` depends on `candle-gen-flux2`, **not** the other way round. A commit into
`crates/bundles/runtime-cuda` therefore leaves the measurement binary byte-identical, and reading
that unchanged digest as proof over the whole closure would be a false green — strictly worse than
the false positive this story set out to remove. (That crate has since left the closure entirely —
see [below](#the-two-crates-above-the-provider-sc-17607) — but the rule is what made its exclusion
from `adjudicates` mandatory while it was still audited, and it is what will catch the next path
added to the closure that the audited target does not compile.)

So the proof carries an `adjudicates` set alongside its digest, and both validators refuse a moved
path that falls outside it. The two kinds of entry earn their place in it differently:

- **Crate trees** are covered when cargo reports having *compiled* a package under them. Derived
  from the `compiler-artifact` stream rather than a hand-maintained list, so it cannot drift from
  the dependency graph. For the `candle-gen-flux2` lib test binary that is gen-core, candle-gen,
  candle-gen-pid, candle-gen-flux2 and (on the CUDA lane) candle-kernels — which, since sc-17607,
  is the whole closure.
- **Build inputs** can never appear there — cargo's stream only ever names *package* manifests, and
  the root `Cargo.toml` is a virtual manifest. They are adjudicated anyway, and soundly: they are
  inputs to *this* build. A lockfile bump that moves a dependency the binary links, a
  `[workspace.dependencies]` edit, a `[target.*]` linker override — each recompiles the binary, so
  each shows up in its digest. One that leaves the digest byte-identical provably did not reach the
  measured code. The inference is anchored to the audited package's own tree being covered, so a
  cargo report that compiled nothing cannot hand back four adjudicable paths.

  Two carve-outs, because "the digest saw it" is not true of either. A `rust-toolchain.toml`
  **channel** bump never reaches a digest comparison — `assertComparableToolchains` hard-stops
  first, which is strictly stronger. And rustflags declared in `.cargo/config.toml` are replaced by
  this script's own, so `assertNoConfigRustflags` refuses the run instead. Both are refusals, not
  adjudications; the free path can no longer reach either case because both files are in the
  closure.

### The two crates above the provider (sc-17607)

`runtime-cuda` was in the closure on the argument that **the worker links it**. Follow that edge
properly and it does not single the bundle out:

```text
runtime-cuda -> candle-gen-catalog -> candle-gen-flux2 -> candle-gen -> gen-core
```

`runtime-cuda` does not depend on `candle-gen-flux2` directly. `candle-gen-catalog` is the
intermediate node, it is "linked by the worker" in exactly the same sense, and it was in **no**
closure list at all. Across the certified window `5ffd7612 → a4f409ae` the catalog moved **988
lines** (+949/−39) and the audited bundle moved **nothing**: the crate the closure watched was inert
and the one it ignored was the one that changed.

Both are above the provider, so **neither can change FLUX.2's compiled code** and the measurement
binary links neither. The choice was between making them symmetric in the closure or symmetric out
of it:

| Option | What it buys | What it costs |
| --- | --- | --- |
| Add `candle-gen-catalog` | symmetry | a re-audit every time the catalog moves — which is often — adjudicating nothing, since no build of the audited target compiles it |
| **Remove `runtime-cuda`** (taken) | every closure member is compile-covered, so every move has a remedy that is not a re-capture | the composition question needs a real answer somewhere else, and object identity stops watching the bundle's own manifest — both supplied, the first here and the second by [the witness](#the-resolved-feature-witness) |

The second is what the paths are actually for. A composition-root change is a *composition*
question — which provider is registered under which id — and a codegen digest cannot answer it in
either direction: an unchanged digest says nothing about the bundle, and a changed one would convict
it of nothing. Keeping `runtime-cuda` audited bought no proof and guaranteed a false positive: a
bundle edit registering some unrelated provider is unadjudicable by construction, and the only
remedy on offer was the ~47.6 GB re-capture this epic exists to stop demanding.

The catalog's exclusion was not an oversight either, which is the part worth keeping. sc-15833's v1
record dispositioned it explicitly — `changedPathDisposition` in
`inference-compatibility-277f.json`: *"Only SenseNova preview registration changed…"* — a considered
judgement, in a free-text field no validator reads and that v2+ records do not carry at all. So the
reasoning existed and evaporated. That is what this section and `COMPOSITION_ONLY_CRATES` replace:
the same judgement, in a place that fails when someone disagrees with it silently.

**Where the composition question is answered instead.**
`crates/sceneworks-worker/src/flux2_composition_audit.rs` asks it of the bundle the worker actually
links, on the `windows-candle` lane, with no weights and no GPU: the `flux2_dev` generator
registered in the composed CUDA catalog must be `candle-gen-flux2`'s own registration — same
`descriptor`, `load` and `footprint` function pointers as
`candle_gen_flux2::register_providers` produces on its own — and so must its **memory-strategy**
pair (`contract` + `safety_check`), which is the admission path SC-15833's rungs were actually
measured against. Identity rather than a frozen descriptor snapshot: a snapshot re-asserts what the
digest already covers and goes red on every innocuous field addition, while identity pins the one
fact the digest cannot reach. If the catalog stops registering the id, registers something else
under it, or wraps the load, that test is red — immediately, and without a re-capture.

The control it compares against is reached through `runtime_cuda::providers::flux2`, which is the
catalog's own `pub use candle_gen_flux2 as flux2` — so the crate under audit owns the alias on both
sides. That is closed rather than hoped over: `type_name_of_val` on the registrar function renders
its *defining* path, which no re-export can rewrite, and the test asserts it is `candle_gen_flux2`.
What the check cannot see is a **feature** change: the same function under a different unification
is pointer-identical. That half is not left open — it is exactly what the resolved-feature witness
below covers, which is why these two stories are complementary rather than competing.

Both crates are named in `COMPOSITION_ONLY_CRATES` (`scripts/inference-artifact-audit.mjs`) and a
test asserts they are in no closure list, so re-adding one is a deliberate act rather than a quiet
one.

### Why not audit `runtime-cuda`'s test binary instead

sc-17524 considered switching the audited artifact to one that links the *whole* closure, which
would make `runtime-cuda` adjudicable by compile coverage alone. Measured and rejected. That binary
links the entire CUDA bundle — `candle-gen-catalog` plus every provider and the audio lane — so its
digest moves on a commit to any of ~50 crates that have nothing to do with FLUX.2. Across the very
window this first had to adjudicate, `5ffd7612` → `06e0c5e9`, four of them moved
(`candle-gen-catalog`, `candle-gen-chroma`, `candle-gen-sdxl`, `candle-gen-sensenova`). It would
have demanded a 47.6 GB re-capture for FLUX.2 code that did not change: the exact false positive
this epic exists to remove, with a far wider trigger.

**What that trade gives up.** One thing: `crates/bundles/runtime-cuda` stays non-adjudicable and
must remain object-identical. Cheap — it has not moved since the capture, and a change there is a
*composition* change (which provider is registered), which a codegen digest could not answer anyway.
`candle-gen-catalog` sits in the same position and is in no closure list at all; that asymmetry is
[sc-17607](https://app.shortcut.com/trefry/story/17607).

1. **The bundle's own manifest stops being watched by object identity** — what
   [sc-17607](https://app.shortcut.com/trefry/story/17607) cost when it took
   `crates/bundles/runtime-cuda` out of the closure. The entry was a *tree* object, so it covered
   `crates/bundles/runtime-cuda/Cargo.toml`, where the shipped feature resolution is decided
   (`candle-gen-catalog = { features = ["cuda"] }`, `default = ["media", "audio"]`,
   `flash-attn = [...]`). An edit there is invisible to the artifact digest (the measurement binary
   is `-p candle-gen-flux2` and never links the bundle), to `Cargo.lock` (feature selections are not
   recorded there), to the root `Cargo.toml` (a virtual manifest), and to the composition check,
   which compares function pointers — the same function compiled under a different feature
   unification is pointer-identical.

   **The witness below is what covers it**, and covers it better than the tree object did. A
   manifest edit that changes how a package FLUX.2 links resolves moves the witness digest and is
   refused; one that changes nothing FLUX.2 links moves nothing — correctly, because it cannot
   affect FLUX.2. The tree object could not tell those two apart, and its only verdict was an
   unadjudicable "re-capture". This is the trade the two stories make together: object identity
   over a crate nobody compiles, replaced by a resolution witness over what actually ships.


## The resolved-feature witness

`cargo test -p candle-gen-flux2 --features cuda` unifies features over a strictly smaller graph than
`-p runtime-cuda` does. A crate outside FLUX.2's closure — `candle-llm`, the audio lane, another
provider — can enable a feature on a dependency FLUX.2 **shares** (`candle-core`, `serde_json`,
`tracing`) and change what FLUX.2 executes in the shipped runtime.

Nothing above sees that, and the reason is worth stating exactly: **feature flags are not recorded
in `Cargo.lock`**. Enabling a feature that pulls no new optional dependency moves no lockfile entry,
no crate tree in the closure, and therefore no artifact digest — the record takes the free path and
certifies. Realized precedent, not a hypothesis: `core-llm` unifying `serde_json/preserve_order`
flipped map ordering under CI, and `preserve_order` is on in this very bundle today.

So the record carries a third digest, over a canonical text derived from `cargo tree` — dependency
resolution, no compiler, no CUDA toolkit, seconds:

```
cargo tree -p runtime-cuda -e normal,build --target x86_64-pc-windows-msvc \
           --prefix none --format "{p}|{f}" --locked
```

restricted to the packages `candle-gen-flux2` links (the same query rooted at the provider). The
restriction is what keeps this from becoming the ~50-crate trigger rejected above: the bundle
resolves **243** packages, FLUX.2 links **173** of them, and a feature change confined to the other
70 moves nothing.

The 173 are not all numerics, though, and the trigger is not zero. `image`, `windows-sys`, `winapi`
and `tracing-core` are among them, so a bundle crate adding a `windows-sys` feature moves the
witness and costs a re-capture for a change that cannot touch VRAM. That is the conservative
direction and far narrower than the source-commit trigger this epic removed, but it is a real cost
rather than a free check.

| | |
| --- | --- |
| shipped package | `runtime-cuda`, default features (`media` + `audio`) — what `sceneworks-worker` and `sceneworks-memory-adapter` both take |
| scope root | `candle-gen-flux2 --features cuda` |
| target | `x86_64-pc-windows-msvc`, pinned — the calibration is a Windows/CUDA one, and `cargo tree --target` needs only rustc's cfg for the triple, never the toolchain |
| edges | `normal,build` — dev-dependencies do not ship |

Package locations are normalized to repo-relative POSIX before hashing, or the witness would
describe the worktree it was computed in rather than the code — verified by deriving `321625ed…` from
three different directories on this box. Everything above was run on the
Windows box; the `metal` lane's copy of it has not been exercised on a Mac, and if that triple's cfg
turns out to need something rustc there will not hand over, it fails loudly rather than silently
producing a different witness — the resolution is part of the hashed text.

The digest is a function of cargo's rendering as well as of the features, so a `rust-toolchain.toml`
bump can move it for reasons that have nothing to do with feature unification. That cannot produce a
false green — within one run both revisions are resolved by the same cargo (the worktree's pinned
one), so the *comparison* stays sound; only the constant frozen in source can go stale, and it goes
stale loudly.

**It runs on every audit, including the free path**, and both validators require it unconditionally.
That asymmetry with the artifact layer is the design: the change it looks for moves no closure
object, so it can only ever *arrive* on a record whose closure is quiet. A witness gated on a build
would have been checked on precisely the runs where the thing it looks for cannot be present.

**Measured, in both directions**, on the real bundle at `a4f409ae`:

| | witness |
| --- | --- |
| clean | `321625ed…` |
| `candle-llm` gains `serde_json/unbounded_depth` — outside the closure, reaches a shared dep | `bd19739d…` **moves** |
| `candle-audio-kokoro` gains a feature of its own — outside the closure, reaches nothing FLUX.2 links | `321625ed…` **unchanged** |

The first mutation touches only `crates/llm/candle-llm/Cargo.toml`: not one of the ten closure paths,
and `serde_json`'s entry in FLUX.2's linked set gains `unbounded_depth`. The second really does
change the shipped bundle's resolution — the audio crate's feature appears in it — and the witness
correctly ignores it.

The first was then committed to a real inference revision and driven through the whole tool, which is
the only way to see the interaction rather than the arithmetic. All ten closure objects came back
byte-identical, so it took the **free path** — the exact run that would previously have printed
*"all 10 closure objects are byte-identical; no build needed"* and exited 0:

```
[audit] all 10 closure objects are byte-identical; no build needed.
[audit] SHIPPED FEATURE SETS DIFFER
  a4f409ae…: sha256:321625ed…
  9d6ba903…: sha256:bd19739d…
  Something outside FLUX.2's closure changed how runtime-cuda compiles code FLUX.2 links.
exit 1
```

The record is still written on that path, as it is for ARTIFACTS DIFFER: an operator needs the
evidence, not just the verdict.

### What it does not close

The witness answers "did the shipped resolution **change** across this window?", which is the
question a two-revision audit can answer. It does not make the measured code equal to the shipped
code. That static gap is *described* rather than closed: `featureWitness.measurementDelta` lists,
for the compatible revision, every package whose features differ between the measurement build and
the shipped one. Today it is exactly two entries:

```
candle-gen v0.0.0 (crates/media/candle-gen/candle-gen): measured [cuda,default,testkit] shipped [cuda,default]
sceneworks-gen-core-testkit v0.1.0 (crates/contracts/gen-core-testkit): measured only
```

Both are the other face of the accepted false positive below — the audited binary is a test binary,
so it turns on `candle-gen/testkit` and links a testkit crate that never ships. `candle-gen` **is**
a shipped package: the measurement build compiles it with one feature the bundle does not. That is
additive test-support code, and the point of the field is that this is now a list you can read
rather than an assumption; if a dev-dependency ever widened a shipped package's features further,
this is where it would appear.

Two caveats on the field itself, so it is not read as more than it is. It is **descriptive, not
gated**: it is outside the hashed text (folding it in would make a dev-dependency edit demand a
re-capture) and neither validator checks it — for the shipped record it is pinned by
`sc-15833-flux2-evidence.test.mjs`, and a future record's is not. And the tool reports a *change* in
it across the window only when the shipped witness held still: the delta is `measured − shipped`, so
widening the shipped side widens the delta too, and reporting it there put the primary finding under
the wrong name. Either way the record is still written and the run still exits 1.

One boundary stays open, and it is the same one the **Scope note** above draws for the lockfile and
the toolchain: this is the resolution inside the *inference* workspace. SceneWorks consumes
`runtime-cuda` as a git dependency, so cargo unifies its features against SceneWorks' **own** graph
too — and that is wider. Measured at the live pin, five of FLUX.2's 173 linked packages resolve with
more features under `cargo tree -p sceneworks-worker --features backend-candle` than under
`-p runtime-cuda` inside inference:

| package | extra features in the SceneWorks build |
| --- | --- |
| `image` | `bmp`, `gif`, `tiff`, `webp` |
| `windows-sys` | `Wdk*`, `Win32_Networking`, `Win32_Security`, `Win32_System_Threading`, … |
| `winapi` | `winsock2` |
| `num-complex` | `std` |
| `tracing-core` | `default` |

None of those is a numerics feature, but "looks inert" is the argument this whole audit exists to
replace. It is not a two-revision question about inference and no witness of this shape can answer
it — tracked as [sc-17639](https://app.shortcut.com/trefry/story/17639).

## Known, accepted false positive

The audited binary is a **test** binary, so an edit confined to `#[cfg(test)]` code inside
`candle-gen-flux2` moves the digest even though production codegen is unchanged. That is the
conservative direction — it costs a re-capture that was not strictly necessary, rather than
authorizing one that was. It is accepted deliberately: the alternative is auditing something other
than the binary that produced the measurements.

## Running it

```bash
node scripts/inference-artifact-audit.mjs --repo ~/Repos/inference --captured 5ffd7612e7de4e76b6db00a7148ed3d9c15b4c0d --compatible <new-pin> --out docs/calibration/sc-15833/inference-compatibility-<short>.json
```

Exit `0` = the proof extends. Exit `1` = the artifacts differ, or the shipped feature sets do, and a
re-capture is owed. The tool reports which closure paths moved before it builds anything.

Both revisions are checked out into a worktree on **every** run, including the free path, because
the resolved-feature witness has to resolve each one's dependency graph. That is `cargo tree`: no
compiler, no CUDA toolkit and no target directory, so the free path is still buildless.

**Where this must run.** The `cuda` lane is the only one whose record can authorize a FLUX.2
calibration, and it needs a real CUDA toolkit — an RTX box, not a Mac. The tool compiles a probe
kernel and refuses to proceed if `nvcc` cannot produce real PTX: the stub written by
`npm run rust:check:candle` emits **empty** `.ptx`, which `candle-kernels` would then `include_str!`,
silently dropping the kernels out of the hashed artifact and going blind to every `.cu` change in the
closure. `--lane metal` builds the same closure on Apple Silicon for exercising the tool; its records
are inert by construction, because every consumer requires `lane: "cuda"`.

**Determinism.** Both revisions are checked out into one worktree and built there, because `file!()`
bakes source paths into panic locations; the worktree and `CARGO_HOME` are additionally remapped out
of the artifact, `CARGO_INCREMENTAL=0`, and `--locked`. The toolchain is re-read after **each**
checkout — a checkout swaps `rust-toolchain.toml` too — and two builds under different compilers are
a hard stop, because their digests are not comparable at all.

On Windows that was not sufficient, and sc-17524 found it by running the audit for real. `link.exe`
stamps the link time into the PE image, so the *same* revision built twice hashed differently —
`5ffd7612` gave `sha256:2164e988…` then `sha256:57f15abb…`, `06e0c5e9` gave `sha256:feb9ea68…` then
`sha256:b5fbcd64…`. The cuda lane could only ever report ARTIFACTS DIFFER: an unfalsifiable
"re-capture on an RTX PRO 6000" for code that had not changed. Byte-diffing two builds located it
exactly — **21 bytes out of 181 MB, none of them in `.text`**: the COFF `TimeDateStamp` at `0x138`,
its copy in each of the three debug-directory entries, and the 17-byte CodeView PDB GUID+age.
Codegen was already deterministic.

The fix is `-Clink-arg=/Brepro -Clink-arg=/DEBUG:NONE`, on `win32` only. Both are needed, which a
two-link experiment on a hello-world binary settled directly:

| flags | result |
| --- | --- |
| *(baseline)* | DIFFERS |
| `/Brepro` | DIFFERS — it hashes the image, which still contains the varying PDB signature |
| `/Brepro -Cstrip=symbols` | DIFFERS |
| `/Brepro -Cdebuginfo=0` | DIFFERS — rustc passes `/DEBUG` on MSVC regardless |
| `/DEBUG:NONE` | IDENTICAL |
| `/Brepro /DEBUG:NONE` | **IDENTICAL** |

`/DEBUG:NONE` is the one that decides it: no PDB is emitted, so there is no signature left to vary.
Dropping it does not weaken the claim — the audited binary is built `--no-run` and never executes,
debug info is not code, and `.text` is byte-identical with and without it. Verified at real scale:
two builds of `5ffd7612` differ in **0 of 181,235,712 bytes**, and the shipped record's digest
`d80844f2…` came back from **thirteen builds across four separate runs**: the two-build
same-revision probe, both revisions of `5ffd7612 → 06e0c5e9` in both build orders (4), the
unmutated build of the sensitivity check (1), both revisions again when the record was re-emitted
for the widened closure (2), both revisions of the shipped `5ffd7612 → a4f409ae` window (2), and
both revisions once more when sc-17607 re-emitted that window against the narrowed nine-path
closure (2) — a run that reproduced the record byte-for-byte, months of unrelated commits later. The
captured side of sc-17760's `5ffd7612 → fbb00d6b` run makes it fourteen across five, and that one
mattered: it is the control that makes that run's ARTIFACTS DIFFER verdict readable at all.

**Stable is not the same as blind**, and the flag that bought stability is the one most likely to
over-normalize — so the other direction was measured too. Mutating a production constant in the
closure (`RMS_EPS: f64` in `candle-gen-flux2/src/transformer.rs`, `1e-5` → `2e-5`) under the same
`/Brepro /DEBUG:NONE` moves the digest from `d80844f2…` to `43fcc369…`. The audit still sees code
changes; it no longer sees the clock.

Mach-O and ELF carry no such stamp, which is exactly why sc-17497's macOS round trip was stable and
this went unseen.

**Re-running is warm.** `--workdir PATH` builds at a fixed path and keeps the derived
`<PATH>-target` afterwards. The remapped paths make every rustc fingerprint path-specific, so this
is the only way a second run reuses the first one's dependency build; without it a re-run is a cold
multi-hour compile. Only the workspace crates rebuild, because `git worktree add` gives every source
file a fresh mtime.

## Schema versions

`schemaVersion: 5` is sc-17607's nine-path closure carrying sc-17606's resolved-feature witness.
v1 (object identity only), v2 (sc-17497's seven-path artifact layer), v3 (sc-17524's ten-path
closure, no witness) and v4 (the ten-path closure *with* the witness) are all **refused**, not
re-graded — in both directions. A record produced before `Cargo.lock`, `rust-toolchain.toml` and
`.cargo/config.toml` were audited cannot be read as evidence about them, and neither can one that
never looked at how the shipped bundle resolves features. A v4 record is the opposite case and is
refused just as firmly: it is not short of evidence, it audited a path this schema stopped asking
about, and dropping an entry out of someone else's record to make it fit is the same unearned
re-reading arrived at by subtraction. For v1/v2/v4 the closure-identity check would reject on size
regardless; the version check is the legible version of the same refusal, and the only thing that
catches v3.


Superseded record files stay on disk — `inference-compatibility-277f.json` is the v1 one, and the
current window `5ffd7612 → a4f409ae` strictly contains the window it proved. They are history, not
fallbacks: pointing `SOURCE_PATHS.inferenceCompatibility` back at one is a hard failure, because
`validatedInferenceCompatibility` throws rather than degrading.

## After running it

Three edits, in this order:

1. Write the record to `docs/calibration/sc-15833/inference-compatibility-<short>.json` (the tool
   does this with `--out`) and point `SOURCE_PATHS.inferenceCompatibility` at it.
2. Set `compatibleInferenceRevision`, `artifactProof` (`{ digest, adjudicates }`) and
   `featureWitness.digest`, all copied from the record, in `FLUX2_COMPATIBILITY_AUDIT` —
   `scripts/generate-memory-matrix.mjs`.
3. Set `FLUX2_COMPATIBLE_INFERENCE_REVISION`, `FLUX2_AUDIT_ARTIFACT_PROOF` and
   `FLUX2_FEATURE_WITNESS.digest` —
   `crates/sceneworks-worker/src/inference_compatibility_audit.rs`.

On the free path, leave the *artifact* proof `null`/`None`: one frozen while the closure is quiet
demands a build that is not due, and both validators reject that. The **feature witness is never
null** — it is produced on every run and required on every record.

Moving the live pin at all — with or without a build — also means:

- the five `flux2_dev` `compatibleInferenceRevision` bindings in
  `config/manifests/builtin.models.jsonc`;
- `SOURCE_PATHS.inferenceCompatibility` (`scripts/generate-memory-matrix.mjs`) and the same filename
  hardcoded in `scripts/sc-15833-flux2-evidence.test.mjs`;
- `LIVE_INFERENCE_REVISION` in `scripts/sc-15833-flux2-evidence.test.mjs`, and its assertion that
  every audited object is `capturedObject === compatibleObject` — true only on the free path;
- `crates/sceneworks-worker/src/candle_memory_strategy.rs`'s binding test, which asserts the literal
  revision pair;
- the evidence-classification test in `scripts/calibration-cost-model.test.mjs` and
  `tests/test_memory_matrix.py::test_calibration_evidence_is_schema_valid_and_matrix_ingested` on the
  parity lane.

Regenerate `docs/generated/*` afterwards; never hand-edit them. Running this audit is still a manual
step alongside `scripts/bump-inference.mjs`, but since sc-17760 that script no longer runs blind: it
reads `FLUX2_COMPATIBLE_INFERENCE_REVISION` and keys on the **transition**, refusing a bump that
moves the pin OUT of the audited window and warning on one that merely inherits a pin already
outside it. The asymmetry is deliberate and is explained on `verifyFlux2AuditWindow` — an
ARTIFACTS DIFFER verdict is terminal until a re-capture, so a fail-closed guard would block every
future bump over a demotion no future bump caused. It also reads the checked-in records first: a pin
this repo has already probed is told the verdict and told *not* to re-run, which is the whole reason
negative records are kept on disk. `--dry-run` reports the same thing before anything is written.

Nothing above touches `flux2_composition_audit.rs`: it pins an *invariant* rather than a revision,
so a pin bump that keeps `flux2_dev` wired to `candle-gen-flux2` needs no edit there and one that
does not is red on the `windows-candle` lane without anyone having to remember this list.

The Rust half deliberately lives in its own module rather than in `candle_memory_strategy`, which
compiles only under `all(not(target_os = "macos"), feature = "backend-candle")` and whose tests link
libcuda. The audit is pure JSON logic, so keeping it separate lets `cargo test -p sceneworks-worker
--lib` run it on any platform instead of first executing it on the `windows-candle` CI lane.


## When it exits 1: the window does not extend

Exit 1 is not a failure of the run — it is the run's answer, and the checklist above does **not**
apply to it. The frozen constants stay where they are, the record it wrote is evidence rather than
an authorization, and the five `flux2_dev` q4 cells are *correctly* demoted from `Runtime verified`
until the calibration is re-captured. Moving the constants anyway would assert compatibility a build
just disproved.

This is realized, not hypothetical. sc-17760 ran `5ffd7612 → fbb00d6b` (the pin #2120 moved to) on
the RTX box:

```
[audit] closure paths moved: Cargo.lock, crates/contracts/gen-core,
        crates/media/candle-gen/candle-gen, crates/media/candle-gen/candle-gen-flux2
[audit] ARTIFACTS DIFFER
  5ffd7612…: sha256:d80844f2…
  fbb00d6b…: sha256:fee1c2de…
```

Three things in that record are worth reading together, and `inference-compatibility-fbb0.json` is
checked in so they can be:

- **The captured side reproduced.** `d80844f2…` is byte-identical to the digest frozen from the
  earlier independent builds of the same revision, weeks and months of unrelated commits earlier —
  the control that says the toolchain and link flags are still deterministic across runs on this
  box, which on Windows is the thing that has actually failed before (see the determinism evidence
  under [Running it](#running-it)). It came for free, because the tool always rebuilds both ends.
  It is *a* control, not the complete one: it cannot rule out non-determinism introduced by
  `fbb00d6b`'s own content, which would need that revision built twice. Nothing in the closure delta
  below suggests it — no build script or proc-macro moved — but if a future ARTIFACTS DIFFER verdict
  is going to authorize spending a capture, build the compatible end twice first.
- **The feature witness held still** — `321625ed…` at both revisions. So the finding is squarely in
  the compiled code, not in how the shipped bundle resolves features around it.
- **`candle-gen-flux2` itself moved**, alongside `gen-core` and `candle-gen`. The provider crate is
  the code the measurements ran, so this is the one case object identity cannot route around: the
  binary that produced the numbers is not the binary that ships.

What that means for [sc-15922](https://app.shortcut.com/trefry/story/15922) is a narrowing, not a
new bill. That story was written before any of this machinery existed and reads as a full re-capture
of the FLUX.2 calibration. The audit does not shrink the capture it will need — the digest moved, so
the measurements genuinely have to be retaken — but it does tell it *why*, and the why is not
incidental. Across `a4f409ae → fbb00d6b` the *closure* moved 685 lines in four files (the range as a
whole is far wider; only these reach the audited binary):

| file | +/− | commit |
| --- | --- | --- |
| `candle-gen-flux2/src/lib.rs` | +276/−44 | `41464d57` FLUX.2 Klein memory ladder (sc-15831), `fbb00d6b` Klein decode parity |
| `candle-gen-flux2/src/memory_strategy.rs` | +259/−63 | same two |
| `candle-gen-flux2/src/edit_provider.rs` | +120/−2 | same two |
| `gen-core/src/residency.rs` | +30/−0 | `d02b8fcf` Anima 2B shared memory ladder — an **MLX** commit that lands in the closure via `gen-core` |

Three quarters of it is the memory-strategy and residency code, in the provider crate whose ladder
the q4 base cells measure. That does not prove `flux2_dev`'s own numbers moved — the two named
commits are about **Klein** — but it does mean the demotion cannot be argued away as pin drift
around untouched code, which is the only argument that would have let the cells stand.

It also fixes the cost model for the rest of that campaign. A window probe is ~15 min warm with no
GPU and no weights, so every future window can be tested before a capture is budgeted, and only the
ones that come back ARTIFACTS DIFFER cost anything more.