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

Two more places survived that first sweep and were converted in sc-17726: the `identity_matches`
filter in `mlx_fit_gate::evidence_admission_route` and the same conjunct in
`verified_lower_alternative`. Both compared `binding.query.inference_revision` against
`catalog_semantic_jobs::INFERENCE_RUNTIME_REVISION` — the raw Cargo pin — so **every** MLX binding
was excluded the moment the pin moved, whether or not that provider's closure had. It reproduced
exactly: bumping the pin alone, with `config/inference-provider-closures.json` untouched so no
closure moved, turned four `mlx_fit_gate` tests red on the old code and none on the new.

## What a closure is

Derived in [`scripts/inference-closure-digest.mjs`](../scripts/inference-closure-digest.mjs), with
no build, no GPU and no weights — it reads git objects at a revision:

| part | why it is in |
| --- | --- |
| the provider's inference crate | the code being measured |
| that crate's transitive **first-party** path/workspace dependencies | code it compiles |
| the local root **`[patch]` targets that closure reaches** | the vendored CUDA kernels arrive *only* this way |
| every closure crate's **build script** | `build.rs` is a compile input, and it is what runs `nvcc` |
| the locked external packages that closure **reaches** | a dependency bump changes codegen; the rest of `Cargo.lock` does not |
| `rust-toolchain.toml`, `.cargo/config.toml`, root `[profile]` / `[patch]` | build inputs that change codegen for everything |

Excluded on purpose: `dev-dependencies` (they do not ship), `tests/`, `benches/`, `examples/`, and
crate-root files that are not compile inputs (`README.md`, `VENDORED.md`).

### Patch targets are resolved by reachability (sc-17935)

Nothing in inference declares `candle-kernels` as a dependency. It is substituted at the workspace
root:

```toml
[patch."https://github.com/huggingface/candle"]
candle-kernels = { path = "crates/media/candle-gen/vendor/candle-kernels" }
```

The v2 walk read `path =` dependencies and `workspace = true`, so it never saw that tree: all 42
files — every `.cu`/`.cuh` kernel and the `build.rs` that compiles them — could change while all five
Candle digests held still and the calibration read `current`. The deleted artifact audit *had*
covered this tree; sc-17919 dropped that coverage without noticing.

A local patch target now joins a closure when the **lock says that closure reaches it**. `Cargo.lock`
records `candle-kernels` under every Candle provider (through `candle-core`) and under no MLX one, so
a kernel edit moves the five Candle lanes and none of the three MLX lanes. Sweeping every patch entry
into every closure would also be sound, and would re-couple lanes that share nothing.

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

## What a demotion costs at runtime

> **Superseded by sc-18095/sc-18096 (epic 18093)**: in
> `memory_strategy::candidate_exclusion` a moved closure no longer excludes a candidate at all.
> Fully-verified measured evidence whose closure went stale stays **eligible**, graded at its peak
> widened by the backend's stale-measured margin
> (`crates/sceneworks-worker/src/ladder_margin_policy.rs`), with current evidence strictly
> preferred over stale for the same key. The structural verdicts (`Invalid`, `OutOfEnvelope`,
> `CompositionMismatch`, unverified conformance) still exclude. sc-18096 retired the MLX
> `evidence_admission_route` pre-demotion too — a stale binding of the installed artifact now
> reaches the selector and is graded there — leaving `verified_lower_alternative` (refusal
> advice) as the only surface still filtered to the current closure.
>
> **Do not conflate the two staleness axes.** CLOSURE-DIGEST staleness — the capture's provider
> compile closure no longer matching the live one — is the widened-margin *signal* described
> above. Dimension-level `Stale` **verdicts** from capture-time manifest flags
> (`vram_gate.rs` `historical_verification` / `krea_control_fit.rs` `historical_verification`)
> are claims that the evidence was ALREADY known non-current when recorded, and they still
> exclude via `optimized_eligibility`.
>
> One deliberate boundary on the signal (sc-18096): a stale-closure record keeps serving its OWN
> measured cell behind the stale margin, but it is **not** a legitimate basis for a fitted-curve
> ESTIMATE — the estimate margin was derived over same-closure re-capture variance plus
> extrapolation error, and cannot also absorb closure drift, so
> `mlx_fit_gate::collect_estimate_bases` restricts extrapolation bases to closure-current
> records.

A lane whose closure moved **degrades to the legacy estimator. It is not refused.** An expired
calibration is epistemically the same as no calibration, and every uncalibrated model already renders
on the generic formula; turning a routine pin bump into a product outage on ~10% of commits is not a
safety posture. `z_image_turbo` ships in exactly this state today and renders.

That places the comparison ahead of the admission-path decision on both backends, for different
structural reasons:

| | where currency is applied | why there |
| --- | --- | --- |
| candle | `memory_strategy::candidate_exclusion` — since sc-18095 a signal (widened margin), not an exclusion | the resident baseline is always in the candidate pool, so grading the calibrated cells conservatively still leaves something to select |
| mlx | since sc-18096, the same `candidate_exclusion` grading: `evidence_admission_route` no longer pre-demotes a stale binding | the pre-18096 fear — the Evidence path withholds the resident baseline, so a stale binding had nothing to fall back to — is retired: refusal now happens only when nothing fits with margins, and the estimate ladder covers the unmeasured rungs |

Pre-sc-18095, "the gate refuses a moved closure" meant it refused to admit that
**candidate** — not that the request dies.

`verified_lower_alternative` is the exception that has to carry its own copy: the geometry it names
in a refusal never becomes a `Candidate`, so nothing downstream grades it. Left unfiltered it would
name a smaller geometry that the very next request refuses for the identical staleness.

`--restamp` exists for a `CLOSURE_DIGEST_VERSION` change, which legitimately re-derives the same
underlying fact. It is not a way past a genuine conflict: rewriting a captured digest that disagrees
with its own revision would launder a stale measurement into a current one, and the backfill refuses
that by default.

## What grades what

| layer | checks |
| --- | --- |
| `npm run check` | the closure config is **keyed** to the live Cargo pin; every complete record carries a digest |
| `check.yml` (parity job) | re-derives every lane's digest from a `--depth=1` fetch of the pinned revision — the digests are **real**, not merely present |
| `check.yml`, same step | re-derives the **captured** half too: `backfill-closure-digests.mjs --verify` against a shallow fetch of every revision `--revisions` reports — 65 record digests and all 33 manifest digests, with none left directly maintained |
| `scripts/inference-closure-digest.test.mjs` | the derivation itself, hermetically, over a synthetic workspace |
| `scripts/backfill-closure-digests.test.mjs` | the stamper's locator over synthetic JSONC (comment-separated digest updated not duplicated, nested and sibling objects never claimed, a hex value inside a comment never mistaken for a key), and the `--verify` verdict itself |
| `memory_strategy.rs` / `mlx_fit_gate.rs` tests | the runtime gate demotes a moved closure and admits an unmoved one, on both the ladder and the named refusal alternative |

The captured-half gate exists because grading only `config/inference-provider-closures.json` left the
other side of every comparison — 65 record digests and 31 manifest bindings — checked by nothing. That
was not hypothetical: a constant in `scripts/sc-15833-flux2-evidence.test.mjs` carried the comment
"derive it with …" and had never been a real derivation, and it survived the whole of sc-17774
unnoticed. A plain dry run reports drift and still exits 0, which is right for a pin-bump preview and
useless as a gate, so `--verify` is a separate mode that fails on any drift and ignores `--restamp`.

sc-17989 brought the last two digests under it. `turboFit`'s and `candle.control`'s were maintained
by hand and invisible to the stamper, which locates a binding by pairing an `inferenceRevision` key
with a `provider` in the same object — `turboFit` had no `provider` and `candle.control` had neither
key. They have both now, so the gate grades all 33 of the manifest's digests rather than 31.

Widening the locator was the substance of that change, not a detail of it. Three properties are
load-bearing, and each replaced something that silently corrupted or silently passed:

- **Object boundaries come from brace depth**, counted off a copy of each line with string contents
  and comment tails blanked out. The previous version stopped at the first line *beginning* with `}`
  or `]`, which is a nested close as often as the real one — it broke early on any block with a
  sub-object above its digest and inserted a second `inferenceClosureDigest`, and it ran past the
  real close when a sibling opened mid-line and rewrote that sibling's digest instead.
- **Comment tails are excluded from every match**, not just from brace counting. This file narrates
  digest provenance in prose, so a scan that reads comments rewrites a 64-hex value quoted inside
  one and leaves the block with no real digest key at all.
- **The walk goes both directions** from the revision key, so a digest or a `provider` written above
  it is found rather than duplicated.

Coverage is now enforced rather than asserted. `--verify` fails on an `inferenceRevision` it cannot
pair with a `provider`, *and* on an orphan — an `inferenceClosureDigest` that no located pair
reaches. The orphan check is what makes "every digest is graded" a fact: dropping an
`inferenceRevision`, or merely its trailing comma, leaves a digest that still looks graded and is
checked by nothing, and the skipped-block check alone does not see either. Both are checked before
drift, because "this digest is not covered" outranks "this covered digest moved".

Two test-fixture constants were the same shape of hole and are now derived instead of transcribed:
`SUPERSEDED_KREA_CLOSURE_DIGEST` (`scripts/generate-memory-matrix.test.mjs`) and the sc-15833
constant (`scripts/sc-15833-flux2-evidence.test.mjs`) are read out of the evidence bundle through
`recordsNeedingDigest` — the gate's own eligibility predicate — so they inherit the CI derivation
rather than sitting beside it. A wrong value in either did not fail anything; it made the test assert
the right verdict for the wrong reason, which is exactly how `820bf106…` hid. By contrast the two
constants in `scripts/sc-15823-flux1-evidence.mjs` are deliberately left as they are: that script
*feeds* the evidence bundle, so its values land in records the gate re-derives and a drift there
already surfaces.

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
4. **Two pin comparisons outlived the sweep** (sc-17726, above). A per-model unit of invalidation is
   only worth what its *narrowest* remaining term is: while `evidence_admission_route` still keyed on
   the pin, the whole MLX lane invalidated repository-wide no matter what the digests said. Both are
   now the closure comparison, mutation-checked in each direction — dropping either conjunct, or
   restoring the compare-against-self shape of defect 1, turns tests red including one driven by the
   real shipped `z_image_turbo` opt-in.
