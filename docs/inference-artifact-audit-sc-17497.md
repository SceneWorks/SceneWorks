# Extending a FLUX.2 calibration across an inference pin bump (sc-17497)

A packaged calibration is a set of measurements taken against a specific build of the inference
engine. Extending it to a newer pin means proving the code it describes has not changed. sc-15833
proved that with **git object identity** over FLUX.2's Candle/CUDA compile closure; sc-17497 keeps
that as the cheap layer and adds a **compiled-artifact** layer underneath it, so a commit that
cannot change the compiled code no longer costs a re-capture. sc-17524 widened the closure to the
workspace build inputs it had been missing and made the Windows build reproducible enough for the
compiled-artifact layer to mean anything.

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
Windows link step was not reproducible at all until `/Brepro` — see **Determinism** below. A Metal
round trip being stable is not evidence that a Windows one is.

## The two layers

1. **Object identity (free).** Every closure object byte-identical ⇒ nothing is built and the proof
   extends. This is the common case and it stays as cheap as it is today.
2. **Artifact identity (one build).** A path moved ⇒ the record must carry a compiled-artifact
   digest, produced by building the measurement binary at *both* revisions under one toolchain.
   Equal digests extend the proof; unequal digests mean the compiled code really did change and the
   calibration must be re-captured.

The proof is frozen in source — `FLUX2_COMPATIBILITY_AUDIT.artifactProof` in
`scripts/generate-memory-matrix.mjs` and `FLUX2_AUDIT_ARTIFACT_PROOF` in
`crates/sceneworks-worker/src/inference_compatibility_audit.rs` — so the checked-in record cannot authorize
itself.

## The closure: six crate trees and four build inputs

sc-15833 declared it as seven paths, all crate trees bar the root manifest. sc-17524 added the three
build inputs that feed every one of those builds:

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
`06e0c5e9`, `a4f409ae`. One part of it the tool
cannot honestly adjudicate: `CARGO_ENCODED_RUSTFLAGS`, which this script must set for the path
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

**What that trade gives up.** Two things, and the second is the one to remember:

1. `crates/bundles/runtime-cuda` stays non-adjudicable and must remain object-identical. Cheap — it
   has not moved since the capture, and a change there is a *composition* change (which provider is
   registered), which a codegen digest could not answer anyway.
2. **Feature unification is narrower than the shipped bundle's.** `cargo test -p candle-gen-flux2
   --features cuda` resolves a smaller feature set than `-p runtime-cuda` does. A feature enabled by
   some crate outside the closure — `candle-llm`, the audio lane, another provider — changes how a
   *shared* dependency (`candle-core`, `candle-nn`, `candle-transformers`) is compiled into the
   shipped runtime, and therefore what FLUX.2 executes there. Feature flags are not recorded in
   `Cargo.lock`, so the lockfile does not see it either: every closure object stays byte-identical,
   the digest stays byte-identical, and the free path certifies. This is realized precedent in this
   codebase — `core-llm` unifying `serde_json/preserve_order` flipped map ordering under CI.

   Note that (2) is a gap between *the measured code* and *the shipped code*, not a regression in
   what this audit claims: sc-15833's capture command is itself `cargo test -p candle-gen-flux2
   --release --features cuda`, so the measurements were always taken under the provider-only
   resolution. The audit's claim — "the code the measurements ran has not changed" — holds exactly.
   Closing the wider gap wants a resolved-feature-set witness (`cargo tree -e features
   -p runtime-cuda`, no compile), tracked separately.

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

Exit `0` = the proof extends. Exit `1` = the artifacts differ and a re-capture is owed. The tool
reports which closure paths moved before it builds anything.

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
`d80844f2…` came back from **eleven builds across three separate runs** — both revisions of the
`5ffd7612 → 06e0c5e9` window in both build orders, a same-revision round trip, and both revisions of
the shipped `5ffd7612 → a4f409ae` window.

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

`schemaVersion: 3` is the ten-path closure. v1 (object identity only) and v2 (sc-17497's
seven-path artifact layer) are **refused**, not re-graded: a record produced before `Cargo.lock` and
`rust-toolchain.toml` were audited cannot be read as evidence about them. The closure-identity check
would reject them on size regardless; the version check is the legible version of the same refusal.

Superseded record files stay on disk — `inference-compatibility-277f.json` is the v1 one, and the
current window `5ffd7612 → a4f409ae` strictly contains the window it proved. They are history, not
fallbacks: pointing `SOURCE_PATHS.inferenceCompatibility` back at one is a hard failure, because
`validatedInferenceCompatibility` throws rather than degrading.

## After a build was needed

Three edits, in this order:

1. Write the record to `docs/calibration/sc-15833/inference-compatibility-<short>.json` (the tool
   does this with `--out`) and point `SOURCE_PATHS.inferenceCompatibility` at it.
2. Set `compatibleInferenceRevision` and `artifactProof` (`{ digest, adjudicates }`, both copied from
   the record) in `FLUX2_COMPATIBILITY_AUDIT` — `scripts/generate-memory-matrix.mjs`.
3. Set `FLUX2_COMPATIBLE_INFERENCE_REVISION` and `FLUX2_AUDIT_ARTIFACT_PROOF` —
   `crates/sceneworks-worker/src/inference_compatibility_audit.rs`.

On the free path, leave both proofs `null`/`None`: one frozen while the closure is quiet demands a
build that is not due, and both validators reject that.

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
