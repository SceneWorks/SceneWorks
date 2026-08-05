# Extending a FLUX.2 calibration across an inference pin bump (sc-17497)

A packaged calibration is a set of measurements taken against a specific build of the inference
engine. Extending it to a newer pin means proving the code it describes has not changed. sc-15833
proved that with **git object identity** over FLUX.2's Candle/CUDA compile closure; sc-17497 keeps
that as the cheap layer and adds a **compiled-artifact** layer underneath it, so a commit that
cannot change the compiled code no longer costs a re-capture. sc-17524 widened the closure to the
workspace build inputs it had been missing and made the Windows build reproducible enough for the
compiled-artifact layer to mean anything. sc-17606 added a third layer — a **resolved-feature
witness** — for the one change neither of the others can see: a crate outside FLUX.2's closure
enabling a feature on a dependency FLUX.2 shares.

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

## The closure: six crate trees and four build inputs

sc-15833 declared it as seven paths, all crate trees bar the root manifest. sc-17524 added three
more build inputs that feed every one of those builds, bringing that kind to four:

| Entry | Kind |
| --- | --- |
| `Cargo.toml`, `Cargo.lock`, `rust-toolchain.toml`, `.cargo/config.toml` | workspace build inputs |
| `crates/contracts/gen-core` | crate tree |
| `crates/bundles/runtime-cuda` | crate tree |
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
the false positive this story set out to remove.

So the proof carries an `adjudicates` set alongside its digest, and both validators refuse a moved
path that falls outside it. The two kinds of entry earn their place in it differently:

- **Crate trees** are covered when cargo reports having *compiled* a package under them. Derived
  from the `compiler-artifact` stream rather than a hand-maintained list, so it cannot drift from
  the dependency graph. For the `candle-gen-flux2` lib test binary that is gen-core, candle-gen,
  candle-gen-pid, candle-gen-flux2 and (on the CUDA lane) candle-kernels. `runtime-cuda` is **not**
  among them and must stay object-identical.
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

It used to give up a second, sharper thing — the audited build's feature unification being narrower
than the shipped bundle's — and that is what the witness below closes.

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
code. That static gap is recorded rather than argued: `featureWitness.measurementDelta` is computed
at both revisions and the tool refuses to emit a record where they disagree, because a delta that
*grows* means the hashed binary represents the shipped code less well at one end than the other —
which no digest comparison can see, since both sides drifted together.

Today it is exactly two entries, both test-only surface:

```
candle-gen v0.0.0 (crates/media/candle-gen/candle-gen): measured [cuda,default,testkit] shipped [cuda,default]
sceneworks-gen-core-testkit v0.1.0 (crates/contracts/gen-core-testkit): measured only
```

Both are the other face of the accepted false positive below — the audited binary is a test binary —
and the point of measuring it is that the list is now a fact rather than an assumption. No *shipped*
package resolves differently under the measurement build than under the bundle; if a dev-dependency
ever widened one, this is where it would appear.

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
`d80844f2…` came back from **eleven builds across three separate runs**: the two-build
same-revision probe, both revisions of `5ffd7612 → 06e0c5e9` in both build orders (4), the
unmutated build of the sensitivity check (1), both revisions again when the record was re-emitted
for the widened closure (2), and both revisions of the shipped `5ffd7612 → a4f409ae` window (2).

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

`schemaVersion: 4` is the ten-path closure plus the resolved-feature witness. v1 (object identity
only), v2 (sc-17497's seven-path artifact layer) and v3 (sc-17524's ten-path closure, no witness)
are **refused**, not re-graded: a record produced before `Cargo.lock`, `rust-toolchain.toml` and
`.cargo/config.toml` were audited cannot be read as evidence about them, and neither can a record
that never looked at how the shipped bundle resolves features. For v1/v2 the closure-identity check
would reject on size regardless; the version check is the legible version of the same refusal.

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

Regenerate `docs/generated/*` afterwards; never hand-edit them. `scripts/bump-inference.mjs` is the
tool that moves the pin and does not yet know about any of this — running this audit is still a
manual step alongside it.

The Rust half deliberately lives in its own module rather than in `candle_memory_strategy`, which
compiles only under `all(not(target_os = "macos"), feature = "backend-candle")` and whose tests link
libcuda. The audit is pure JSON logic, so keeping it separate lets `cargo test -p sceneworks-worker
--lib` run it on any platform instead of first executing it on the `windows-candle` CI lane.
