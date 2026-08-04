# Extending a FLUX.2 calibration across an inference pin bump (sc-17497)

A packaged calibration is a set of measurements taken against a specific build of the inference
engine. Extending it to a newer pin means proving the code it describes has not changed. sc-15833
proved that with **git object identity** over FLUX.2's seven-object Candle/CUDA compile closure;
sc-17497 keeps that as the cheap layer and adds a **compiled-artifact** layer underneath it, so a
commit that cannot change the compiled code no longer costs a re-capture.

## Why the source tree was the wrong unit

The v1 audit's unit is a crate tree, so any commit into one of the seven invalidates the proof.
inference `35251a88` added 42 lines of `//!` doc comment to `candle-gen/src/preview.rs`. Six objects
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

Confirmed on the real closure at the real revisions: `277f4238` and `d2216f6b` — the pin bump the doc
comment blocked — produce a **byte-identical** measurement binary.

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

## A digest only speaks for what the binary links

`runtime-cuda` depends on `candle-gen-flux2`, **not** the other way round. A commit into
`crates/bundles/runtime-cuda` therefore leaves the measurement binary byte-identical, and reading
that unchanged digest as proof over the whole closure would be a false green — strictly worse than
the false positive this story set out to remove.

So the proof carries an `adjudicates` set alongside its digest: the closure paths the audited binary
actually compiled. The tool derives it from cargo's own `compiler-artifact` stream rather than a
hand-maintained list, so it cannot drift from the dependency graph, and both validators refuse a
moved path that falls outside it. For the `candle-gen-flux2` lib test binary that set is gen-core,
candle-gen, candle-gen-pid, candle-gen-flux2 and (on the CUDA lane) candle-kernels. The workspace
`Cargo.toml` and `runtime-cuda` are **not** adjudicable by it and must stay object-identical, or a
different artifact has to be audited.

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
of the artifact, `CARGO_INCREMENTAL=0`, and `--locked`. The toolchain is whatever the inference
repo's `rust-toolchain.toml` pins, and it is recorded — a rustc bump genuinely does change the
compiled code, so it should invalidate the proof.

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

Moving the live pin at all — with or without a build — also means updating the five `flux2_dev`
`compatibleInferenceRevision` bindings in `config/manifests/builtin.models.jsonc` and the three
places that lock the window to a literal revision: `LIVE_INFERENCE_REVISION` in
`scripts/sc-15833-flux2-evidence.test.mjs`, the evidence-classification test in
`scripts/calibration-cost-model.test.mjs`, and
`tests/test_memory_matrix.py::test_calibration_evidence_is_schema_valid_and_matrix_ingested` on the
parity lane. Regenerate `docs/generated/*` afterwards; never hand-edit them.

The Rust half deliberately lives in its own module rather than in `candle_memory_strategy`, which
compiles only under `all(not(target_os = "macos"), feature = "backend-candle")` and whose tests link
libcuda. The audit is pure JSON logic, so keeping it separate lets `cargo test -p sceneworks-worker
--lib` run it on any platform instead of first executing it on the `windows-candle` CI lane.
