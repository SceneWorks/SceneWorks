# What invalidates a packaged calibration (sc-17774)

A packaged calibration is a set of measurements taken against a specific build of the inference
engine. Keeping it in force across a pin bump means answering one question: **did the code those
measurements describe change?**

There is now exactly **one mechanism**, applied identically to every calibrated model.

---

## The rule

A calibration is `current` when the **compile-closure digest of the provider it measured** is
unchanged. Nothing else is compared — in particular, **not the inference pin**.

```
record.repositories.inference.closureDigest === config/inference-provider-closures.json
                                                  .providers["<backend>:<provider>"].digest
```

`inferenceRevision` survives on every record and binding as **capture provenance**. It says where a
measurement came from. It is never a currency term.

### Why the pin was the wrong unit

Currency used to be `record.repositories.inference.revision === <the Cargo pin>`, in three places
independently. The unit of invalidation was therefore the whole inference repository at commit
granularity: a commit to `mlx-gen-z-image` demoted `flux2_dev`, and a documentation-only commit
demoted all six calibrated providers at once. Measured over the 90 days to `fbb00d6b`, **all 2812
non-merge commits demoted everything.** Re-capture costs ~47.6 GB.

Under the closure unit, on the same window, each lane sees roughly **9–10.5%** of those commits.

## What a closure is

Derived in [`scripts/inference-closure-digest.mjs`](../scripts/inference-closure-digest.mjs), with
no build, no GPU and no weights — it reads git objects at a revision:

| part | why it is in |
| --- | --- |
| the provider's inference crate | the code being measured |
| that crate's transitive **first-party** path/workspace dependencies | code it compiles |
| the locked external packages that closure **reaches** | a dependency bump changes codegen; the rest of `Cargo.lock` does not |
| `rust-toolchain.toml`, `.cargo/config.toml`, root `[profile]` / `[patch]` | build inputs that change codegen for everything |

Excluded on purpose: `dev-dependencies` (they do not ship), `tests/`, `benches/`, `examples/`.

Source is hashed **semantically** for `.rs` and `.toml` — whole-line comments are stripped via
`scripts/lib/source-revision.mjs`, so a documentation-only edit absolves. Everything else (`.metal`,
`.cu`, embedded fixtures) is hashed byte-for-byte, because stripping a language the tool does not
model is how a real change becomes invisible.

### The key is `<backend>:<provider>`

A provider id is **not** unique. `krea_2_turbo_control` exists on mlx (`mlx-gen-krea`) and on candle
(`candle-gen-krea`) — genuinely different code paths. Keying on the bare id would grade one
backend's measurements against the other backend's code.

## Bumping the pin

```bash
node scripts/inference-closure-digest.mjs --repo <inference clone> --write
```

```bash
node scripts/backfill-closure-digests.mjs --repo <inference clone> --write
```

Then regenerate the derived docs (`npm run generate:memory-matrix`,
`npm run generate:calibration-cost-model`). `scripts/bump-inference.mjs` refuses a bump that would
leave the closure config behind, and names both commands.

**Lanes whose closure did not move stay `current` across the bump.** Only the ones that actually
changed are demoted, and the regenerated files show which.

`--restamp` exists for a `CLOSURE_DIGEST_VERSION` change, which legitimately re-derives the same
underlying fact. It is not a way past a genuine conflict: rewriting a captured digest that disagrees
with its own revision would launder a stale measurement into a current one, and the backfill refuses
that by default.

## What grades what

| layer | checks |
| --- | --- |
| `npm run check` | the closure config is **keyed** to the live Cargo pin; every complete record carries a digest |
| `check.yml` (parity job) | re-derives every lane's digest from a `--depth=1` fetch of the pinned revision — the digests are **real**, not merely present |
| `scripts/inference-closure-digest.test.mjs` | the derivation itself, hermetically, over a synthetic workspace |
| `memory_strategy.rs` / `mlx_fit_gate.rs` tests | the runtime gate refuses a moved closure and admits an unmoved one |

The CI re-derivation matters more than it looks. Without it the currency term is checked-in data
that nothing grades — a hand-edited digest would pass. `SceneWorks/inference` is public, so the
fetch needs no token; see the module header for why the opposite was believed for a while.

## What this unit does NOT see

Read this before trusting a green.

- **Feature unification.** `Cargo.lock` records versions, not resolved features, so a crate *outside*
  a closure enabling a feature on a dependency *inside* it moves no byte here. Owned by the
  resolved-feature witness (sc-17606). Narrowing the code unit makes that witness strictly more
  load-bearing.
- **Crate-mates.** Providers sharing a crate share its digest: a `qwen_image_edit` edit moves
  `qwen_image`. Deliberate — Rust compiles a crate as one unit with cross-module inlining, so
  file-level scoping inside a crate would be a false green. sc-17776 measured that only a behaviour
  witness can absolve this class soundly. The over-trigger is real, and narrowed from "the whole
  repository" to "one crate plus what it links".
- **Realised VRAM.** This is a code-identity unit. It answers "did the code the measurements ran
  change?", never "did the measured quantity change?".

## What this replaced

Every one of these was a *separate, model-specific* mechanism. They are all deleted.

| gone | what it was |
| --- | --- |
| `inference_compatibility_audit.rs`, `FLUX2_COMPATIBILITY_AUDIT` | `flux2_dev`'s hand-audited window — one provider, one `(captured → compatible)` revision pair, spent the moment the pin moved once further |
| `compatibleInferenceRevision` | the same one-shot hatch in a second spelling, on manifest bindings |
| `KREA_CONTROL_INFERENCE_REVISION`, `KREA_TURBO_INFERENCE_REVISION` | krea's own frozen SHAs, on two lanes |
| `verifyFlux2AuditWindow` | a pin-bump gate that spoke for one lane and was silent about the other five |
| `scripts/inference-artifact-audit.mjs` + `scripts/research/sc-17776-*` | the tooling for the above, unused once the mechanism went |

The artifact audit's one real capability — absolving comment-only churn, which it bought by linking
a CUDA binary at two revisions on a specific Windows box, for one provider — is kept and generalised
by hashing semantic source. Every lane gets it for free.

`docs/inference-artifact-audit-sc-17497.md` documented that mechanism and is superseded by this
file. The composition check (`crates/sceneworks-worker/src/flux2_composition_audit.rs`, sc-17607)
is **not** an invalidation mechanism — "which provider is registered under `flux2_dev`" is a
question no digest can answer in either direction — and survives unchanged.

## Three defects the removal exposed

Recorded because each one had been green for months, and each is the same shape: a gate that had
stopped grading anything.

1. **The MLX selector compared candidates against themselves.** `expected_inference_revision` was
   read off the first candidate's own evidence, so the comparison could not fail. Every fit-ladder
   test in `mlx_fit_gate.rs` was passing under a dead gate.
2. **The candle selector test went dark after every pin bump.** It skipped its real assertions
   whenever the live pin had moved past the audited window — so the one lane with a compatibility
   hatch was also the one whose test stopped running.
3. **Recovering a candidate's digest by `MemoryEvidenceKey` search failed open.** A miss fell back to
   the live digest — precisely the value the request compares against — so an unplaceable candidate
   became automatically current. Digests are now carried from their push sites; there is no lookup
   to miss. Pinned by `a_moved_provider_closure_demotes_the_calibrated_ladder`, mutation-checked.
