# Anchor currency attestations at inference `c6d6a4db`, extended to `1cd0e393` / `8a65db2a` / `563c44e1` (sc-22667 / sc-22765 / sc-22723 / sc-22738, epic sc-22657)

Terminal-story close-out of the memory-anchor currency question the sc-22667 review raised as a
blocker: at the landed pin `c6d6a4dbd61ab09c26ff5526632cae2cefea60ed`, none of the five anchors the
image derivation law prices from (`z_image_turbo:candle` q4 / q8 / bf16, `krea_2_turbo:candle` q4,
`z_image_turbo:mlx` q4) recorded the loader-closure digest `config/anchor-loader-closures.json`
declares, so `candle_image_anchor` / `krea_store_anchor` / the MLX floor would have refused every
one of them in production and the headline admission held only on stores the tests re-stamped
privately.

This repository's invalidation doctrine is two-gated (calibration runbook §7c-bis): a differing
closure digest says the loader's **source** moved; only a load-or-device-path change with **no
behaviour witness** says the loader's **memory behaviour** moved. The review offered the honest
alternative to a blanket re-capture — classify each anchor's diff and re-stamp with the
justification recorded — and that is what this document, `config/anchor-currency-attestations.json`
and the `--stamp-anchors` attestation path implement. Nothing was re-captured: every diff was read,
and every anchor is either accounting-only or witnessed unchanged.

## Method

For each non-current anchor:

1. `git -C <inference> diff --name-only <measuredRevision>..c6d6a4db -- <src/ of every crate its
   loader closure names in config/anchor-loader-closures.json>` — the file list in each attestation's
   `filesChangedSinceMeasurement` is that command's output, generated, not typed.
2. Every file classified: **accounting-only** (`memory_strategy.rs`, `architecture_facts.rs`,
   `wan_i2v_memory.rs`, `weightsmeta.rs`, re-export lines in `lib.rs`, `quant/sidecar.rs`'s
   constant re-home), **test-only** (`#[cfg(test)]` helper moves in gen-core `registry.rs` /
   `runtime.rs`), **same-value-constant** (`candle-gen-pid/src/engine.rs` naming the f32 load dtype
   it already used), or **load-or-device-path**.
3. A behaviour witness cited where one exists: the E6 falsification fixture
   (`candle-five-rung-falsification-sc-22667.json`) re-measured `z_image_turbo` q4 and
   `krea_2_turbo` q4 at `a5f643ae` on the same GPU, and its `coldStagedControl` (the staged rung
   alone in a fresh process — the shape the packaged anchors were captured in) reproduces the
   packaged anchors per phase.
4. Accounting-only, or load-path with a witness → attested. Load-path with no witness → would have
   been re-captured. No anchor fell in the last class.

## Classification

| Anchor | Measured at | Files changed in closure since | Class | Witness | Action | `is_current` now |
| --- | --- | --- | --- | --- | --- | --- |
| `z_image_turbo:candle:q4` (sc-15859) | `670dc1f4` | 11: gen-core `lib.rs` (re-exports), `memory_strategy.rs`, `registry.rs` + `runtime.rs` (`#[cfg(test)]` helper relocation, Windows test fixture), `wan_i2v_memory.rs`, `weightsmeta.rs` (materialized_* readers; hidden-dir skip of `.candle-device-format-v1/`); `candle-gen-pid/src/engine.rs` (`LOAD_DTYPE = F32` / `GEMMA_MERGED_FILE` named, same values); `candle-gen-z-image/src/memory_strategy.rs` (asset + architecture facts, loaded-path pricing); `candle-gen/src/architecture_facts.rs` (new snapshot-config reader), `lib.rs` (`pub mod`), `quant/sidecar.rs` (`CACHE_DIR` constant re-homed, same value) | accounting-only | E6 cold staged control at `a5f643ae`: conditioning 2.989 GB vs anchor 3.097 (−3.5%, under), denoise 8.056 vs 8.051 (+0.07%), decode 11.747 vs 11.742 (+0.04%); `a5f643ae..c6d6a4db` is five accounting files | re-stamped (attested) | true |
| `z_image_turbo:candle:q8` (sc-15859) | `670dc1f4` | same 11 (the closure is per model) | accounting-only | none for the tier itself; the q4 witness covers the shared loader across `670dc1f4..a5f643ae`, the rest is the five accounting files | re-stamped (attested) | true |
| `z_image_turbo:candle:bf16` (sc-15859) | `670dc1f4` | same 11 | accounting-only | as q8 | re-stamped (attested) | true |
| `krea_2_turbo:candle:q4` (sc-11045) | `3cd86ba2` | 49, including load-path files: `candle-gen-krea` `loader.rs`, `pipeline.rs`, `transformer/mod.rs`, `control*.rs`, `lib.rs`; `candle-gen` `quant/mod.rs`, `quant/adapt.rs`; `candle-gen-boogu` / `-qwen-image` / `-pid` files the closure reaches; plus the accounting set. The diff alone cannot classify this anchor | witnessed-unchanged | E6 cold staged control at `a5f643ae` (22 first-parent merges after `3cd86ba2`): conditioning 3.694 GB vs anchor 3.828 (−3.5%, under), denoise 15.099 vs 15.103 (−0.02%), decode 22.349 vs 22.352 (−0.01%). `a5f643ae..c6d6a4db` over the Krea closure is four accounting files; `candle-gen-krea` is byte-identical across it | re-stamped (attested) | true |
| `z_image_turbo:mlx:q4` (sc-22667 nax capture) | `a5f643ae` | 4: gen-core `lib.rs` (re-export), `memory_strategy.rs`, `weightsmeta.rs` (hidden-dir skip; MLX never writes that cache), `mlx-gen-z-image/src/memory_strategy.rs`. `model.rs`, the loader and `mlx-gen/src` untouched | accounting-only | the anchor itself was captured at `a5f643ae` (run 33817079839), after every loader-path change in the store's history | re-stamped (attested) | true |

The conditioning-phase −3.5% in both witnesses is the phase's cold-start jitter and sits **under**
the anchor; the E6 test documents the same band ("cold staged control lands between −3.6% … and
+0.04%") and admission charges the 2% recapture spread on top of every anchor derivation.

## What the attestation mechanically is

* `config/anchor-currency-attestations.json` — one entry per anchor: `measuredRevision`,
  `attestedRevision`, `attestedAt`, `story`, `class`, `why`, `witness`,
  `filesChangedSinceMeasurement[{path, class}]`. Reviewed data, never derived at the pin by a tool.
* `node scripts/anchor-loader-closure.mjs --repo <clone> --stamp-anchors` derives an attested
  anchor's key at `attestedRevision` (every other anchor at its own measurement revision, as
  before) and copies the justification — minus the file list — into the store as
  `source.currencyAttestation`. `--stamp-anchors --check` verifies both halves;
  `--anchor-revisions` lists attested revisions too, so CI's shallow clone fetches them.
* `scripts/extract-memory-anchors.mjs` carries `currencyAttestation` forward beside the digest it
  justifies. `sceneworks-core` validates its shape at load (`AnchorCurrencyAttestation`), and the
  matrix reports it per anchor (`yes — attested accounting-only 670dc1f4→c6d6a4db (sc-22667)`) and
  in `summary.attestedAnchors`.
* Bounded on both ends: `measuredRevision` must still equal the cited record's inference revision
  (a re-capture refuses the stale entry), and the next pin bump that moves the closure past
  `attestedRevision` stales the anchor again. It is not a way past a load-or-device-path change
  without a witness — that is the false green the currency key exists to prevent.

## Result

* `config/memory-anchors.json`: the five anchors now record the pin's declared digest with their
  attestation; `report:stale-lanes` / matrix: `15 anchors, 10 stale, 5 current by attestation`
  (the ten are `flux2_dev:mlx`, `ltx_2_3:mlx`, `ltx_2_5:mlx`, `qwen_image:{candle,mlx}` — unread
  diffs, not attested, honestly stale).
* The private re-stamp helpers (`z_image_live_anchor_store`, `sc_22667_live_anchor_store`,
  `vram_gate::krea_live_anchor_store`) are gone. The headline test, the E6 falsification and the
  Krea gate tests price from `CandleLadderAnchors::packaged(&contract)` / the packaged store
  **unmodified** and assert `anchor_currency_matches` on the packaged row, so they cannot pass on a
  re-stamp: flipping the packaged digest reds them. `mlx_fit_gate::flux2_live_anchor_store` remains
  and says why (the flux2 MLX rows are not attested and honestly stale).

## Extension to inference `1cd0e393` (sc-22765, main-line pin, 2026-09-05)

On `main`, in parallel with the epic sc-22723 feature branch, the pin moved `c6d6a4db` → `1cd0e393`
(the sc-22760 FLUX.2 Klein rehost fix). Under the bound stated above that would stale all five
attested anchors, so each entry was re-read for the new range and re-keyed to the new pin. The
reading is mechanical and exhaustive rather than a judgement call:

* `c6d6a4db..1cd0e393` changes 20 inference files — the sc-19699 / sc-22261 StarVector work
  (`core-llm` contracts and testkit, `candle-llm` StarVector plus its decode streaming, `mlx-llm`
  StarVector), the sc-22760 `mlx-gen-flux2` artifact inventory and its tests, and `release/` +
  `scripts/release/` tooling.
* **None of those files appears in any of the five anchors' loader closures.** Intersecting each
  anchor's `closureFiles` in `config/anchor-loader-closures.json` with the range's changed-file list
  is empty for `krea_2_turbo:candle` (143 files), `z_image_turbo:candle` (109) and
  `z_image_turbo:mlx` (137). The closures are byte-identical across the range, and their digests are
  unchanged in the regenerated config.
* So there was no file left to classify: the extension adds nothing to either the accounting-only
  reading or the measured witness the four/one entries already rest on, and each entry keeps its
  original `class`, `witness` and `filesChangedSinceMeasurement` (that list still covers
  `measuredRevision..attestedRevision` exactly, because the widened range contributed no closure
  file).

Three closures DID move on this bump — `flux2_dev:mlx`, `ltx_2_3:mlx` and `ltx_2_5:mlx`, all of
which reach `mlx-gen-flux2`'s artifact inventory. Those anchors were already honestly stale and
unattested before this bump, and they stay that way: nothing here re-keys them.

The two pin lines (`1cd0e393` on main, `8a65db2a` on the feature branch) rejoin at `563c44e1`, the
inference merge of the feature branch over `1cd0e393`; the final section below reads
`8a65db2a..563c44e1`, which is exactly the main-side content this section covers plus inference
PR #961's removal of the stale-marker tolerance.

## Extension to inference `8a65db2a` (sc-22723 feature pin, 2026-09-05)

The epic sc-22723 feature pin (`c6d6a4db` → `8a65db2aad0b54581794127f5c3f5621e107e321`, inference
PRs #946–#958) moved all three closures and staled the five attested anchors, as the bound above
says it must. The `c6d6a4db..8a65db2a` segment was read file by file with the same per-crate `src/`
rule and the five entries were **extended** (`attestedRevision` → `8a65db2a`, `story` → sc-22723,
the reading appended to `why`) rather than re-written: every earlier segment and witness still
holds, and nothing in the new segment is a load or device path.

| Closure | Files changed in `c6d6a4db..8a65db2a` | Reading |
| --- | --- | --- |
| `krea_2_turbo:candle` | gen-core `encoder_contract.rs`, `lib.rs`, `wan_i2v_memory.rs`; `candle-gen-krea/src/lib.rs` | accounting-only. `encoder_contract.rs`: `ValidatedEncoderSource` keeps the whole `PackedQuantization` marker (same-value accessors), a stale-packed-marker allowance only `mlx-gen-flux2` calls, and `hf_cache_discovery_roots` (a source-authorization helper) plus tests; `lib.rs`: its re-export; `wan_i2v_memory.rs`: Wan A14B identity tokens. `candle-gen-krea/src/lib.rs`: the per-(route, proven tier) identity table and weights-free registry-behavior surfaces — the q4 turbo identity this anchor is keyed to (`krea-turbo-cuda-phase-curves-v1`) is unchanged and `krea_provable_artifact_tier` reads the marker `validate_load_spec` / `actual_quant_tier` already consulted. `loader.rs`, `pipeline.rs`, `transformer/`, `control*.rs` and `candle-gen` `quant/` untouched. The class stays `witnessed-unchanged` for the earlier load-path range. |
| `z_image_turbo:candle` (q4 / q8 / bf16) | the same three gen-core files; `candle-gen-z-image/src/memory_strategy.rs` | accounting-only. `selected_encoder_discovery_roots` now delegates to `gen_core::hf_cache_discovery_roots`, widening the text-encoder discovery confinement to the HF-cache repository directory — a validation gate consumed only by the planning-facts readers in that file; the same bytes reach the same device the same way. `pipeline.rs`, `model.rs`, the loader and the device-format paths untouched. |
| `z_image_turbo:mlx` (q4) | `core-llm/src/starvector.rs`; the same three gen-core files | accounting-only. `starvector.rs` adds a StarVector decode `Progress` event the Z-Image encoder never executes; `mlx-gen-z-image`, `mlx-gen-pid` and `mlx-gen/src` untouched. |

No re-measure was taken for the new segment — the doctrine allows the diff alone when it carries
no load-or-device-path change, which is the case here. Matrix after `--stamp-anchors`:
`15 anchors, 10 stale, 5 current by attestation` (unchanged set).

## Extension to inference `563c44e1` (sc-22738, the epic's final pin, 2026-09-06)

The epic sc-22738 terminal pin (`8a65db2a` → `563c44e1652043bf8bd018de79cadadc393f91f2`, inference
`main` = the merge of feature PR #960 over main `1cd0e393`, synchronized through PR #961) moved
every loader closure in `config/anchor-loader-closures.json` and staled the five attested anchors
again. The `8a65db2a..563c44e1` range changes nine inference files; exactly two are in the attested
closures, and they are the same two for all three: gen-core `encoder_contract.rs` and `lib.rs`. The
five entries were **extended** (`attestedRevision` → `563c44e1`, `story` → `sc-22738`, the reading
appended to `why` and to the two files' `filesChangedSinceMeasurement` classes).

| Closure | Files changed in `8a65db2a..563c44e1` | Reading |
| --- | --- | --- |
| `krea_2_turbo:candle`, `z_image_turbo:candle` (q4 / q8 / bf16), `z_image_turbo:mlx` (q4) | gen-core `encoder_contract.rs`, `lib.rs` | accounting-only, and strictly a REMOVAL. Inference PR #961 drops the sc-22727 stale-packed-marker tolerance that sc-22760's corrected Klein rehosts made unnecessary: the two entry points `source_for_load_with_stale_packed_marker` / `validate_source_for_discovery_with_stale_packed_marker`, the `RequiredWithStalePackedMarker` policy arm and its `stale_packed_marker()` reader, the `declared_quant` rewrite in `validate_source_with_policy` (the identity for `Required` and `ProviderOwnedComfyUi`, the only policies these closures use), `PackedQuantization` back to crate-private, and the allowance test; `lib.rs` drops the matching re-export. The removed surface's only production caller was `mlx-gen-flux2`'s Klein turnkey inventory, which none of the three closures reaches. `source_for_load`, `validate_source_for_discovery`, `requires_config` for `Required` and every shard/header/quantization-evidence check these models execute are the same code path before and after. |

The other seven files in the range — `mlx-gen-flux2` `artifact_inventory.rs` / `loader.rs` /
`memory_strategy.rs` / `model.rs`, its two test files and `release/real-weight-models.toml` — are
outside every attested closure (they move `flux2_dev:mlx`, `flux2_klein_9b*:mlx`, `ideogram_4*:mlx`
and `lens*:mlx`, none of which is attested; those stay honestly stale as before).

No re-measure was taken for the new segment — the doctrine allows the diff alone when it carries no
load-or-device-path change, which is the case here. Matrix after `--stamp-anchors`:
`15 anchors, 10 stale, 5 current by attestation` (unchanged set).

## Extension to inference `3b922bac` (sc-22738, third terminal pin, 2026-09-06)

The sc-22738 pin moved once more (`563c44e1` → `3b922bac6094e06f98bb2598a6d6fe10dc92739e`, inference
`main` = the merge of PR #962 `fix/sc-22738-krea-calibration-fault-hooks` over `563c44e1`, its first
parent) so the krea re-runs measure on an engine whose MLX krea and krea-realtime generators honour
the harness-authorized `calibration_fault` at every phase exit. The `563c44e1..3b922bac` range
changes fourteen inference files, every one under `crates/media/mlx-gen/mlx-gen-krea` or
`crates/media/mlx-gen/mlx-gen-krea-realtime` (`memory_strategy.rs` fault helpers + unit tests,
`model.rs` / `model_control.rs` / `pipeline.rs` / `generate.rs` / `t2v.rs` hook sites, and their
integration tests). The bump script's remediation is the closure intersection, and for all five
attested anchors it is **empty**:

| Closure | `closureFiles` ∩ `563c44e1..3b922bac` | Reading |
| --- | --- | --- |
| `krea_2_turbo:candle` | ∅ | the candle krea closure (`candle-gen-krea`, `candle-gen`, gen-core) never links `mlx-gen-krea` or `mlx-gen-krea-realtime`; its digest is unchanged in `config/anchor-loader-closures.json`. |
| `z_image_turbo:candle` (q4 / q8 / bf16) | ∅ | `candle-gen-z-image` never links either krea crate; digest unchanged. |
| `z_image_turbo:mlx` (q4) | ∅ | `mlx-gen-z-image` / `mlx-gen-pid` / `mlx-gen/src` never link either krea crate; digest unchanged. |

Only `krea_2_raw:mlx` and `krea_2_turbo:mlx` moved in the anchor-loader closures (and
`mlx:krea_2_raw` / `mlx:krea_2_turbo` / `mlx:krea_2_turbo_control` / `mlx:krea_realtime_14b` in the
provider closures); none of those is attested, and they are exactly the closures the pending krea
re-runs re-capture. The five entries were **extended** (`attestedRevision` → `3b922bac`, the reading
appended to `why`); no file is added to any `filesChangedSinceMeasurement`, because none changed.

No re-measure was taken — an empty intersection is the byte-identical case the doctrine names as the
trivial extension. Matrix after `--stamp-anchors`: `15 anchors, 10 stale, 5 current by attestation`
(unchanged set).

## Extension to inference `e16c6a55` (sc-22738, fourth terminal pin, 2026-09-07)

The sc-22738 pin moved again (`3b922bac` → `e16c6a55e8cb1e2a911ec590e581b068fcc5b529`, inference
`main` = PR #965 `fix/sc-22738-wan-rehost-tolerance` over #964 `fix/sc-22738-wan-t2v-memory-contract`
over #963 `fix/sc-22738-mlx-progress-boundaries`, all first-parent over `3b922bac`) so the Bernini and
Mage-Flow MLX generators emit the lifecycle boundaries their memory contracts declare, the loaded Wan
T2V-A14B reaches its memory contract, and Wan tolerates the legacy inert config key and dense output
heads on packed tiers. The `3b922bac..e16c6a55` range changes eleven inference files: gen-core
`residency.rs` and `wan_i2v_memory.rs`; `mlx-gen-bernini/src/bernini.rs` and its conformance test;
`mlx-gen-mage/src/{model,pipeline}.rs`; `mlx-gen-wan/src/{config,i2v_memory_strategy,memory_strategy,model}.rs`;
and one `mlx-gen-flux` real-weight test. Unlike the previous two extensions the intersection is
**not** empty — every one of the five attested closures links gen-core, and gen-core `residency.rs`
changed — so each anchor was read file by file:

`residency.rs` (+195/−8, of which 147 lines are new unit tests) is instrumentation-only.
`ensure_warm_locked` drops its `on_progress` parameter and no longer emits
`Loading(TextEncoder)`/`Loading(Renderer)` itself; the two warm (`!stage_residency`) arms of
`run_request_scoped` and `run_staged_request_scoped` emit `Loading(TextEncoder)` before the warm
load and `Loading(Renderer)` **after** the prompt encode, both gated on the new `warm_needs_load`
(a populated warm pair still reports no load). The staged arms, `run_two_phase`, the components
loaded, their order, their devices and the offload policy are byte-identical.

| Closure | `closureFiles` ∩ `3b922bac..e16c6a55` | Reading |
| --- | --- | --- |
| `krea_2_turbo:candle` (q4) | `gen-core/src/residency.rs` | measured on the **staged** arm (`measuredRegime.staged = true`), which the diff does not touch: neither what it loads nor where the boundaries it was cut on fall changed. |
| `z_image_turbo:candle` (q4 / q8 / bf16) | `gen-core/src/residency.rs` | `candle-gen-z-image` never drives `gen_core::residency::Residency` (no `run_request_scoped` / `run_staged_request_scoped` / `run_two_phase` call in the crate); the file is in the closure only as a gen-core module and is unreached by the Z-Image Candle load and render path. |
| `z_image_turbo:mlx` (q4) | `gen-core/src/residency.rs` | `mlx-gen-z-image` drives `run_staged_request_scoped` and the anchor was measured on the **warm** arm (`measuredRegime.staged = false`), so the moved `Loading(Renderer)` is on its path — but the loads it triggers are unchanged, and the MLX adapter's Z-Image capture (`run_z_image_reference_loaded`) cuts conditioning on the first `Progress::Step` and denoise on `Progress::Decoding`, neither of which `residency.rs` emits or moved. Both the memory behaviour and the phase attribution the packaged record was minted under are unaffected. |

`wan_i2v_memory.rs` is in none of the five closures. The five entries were **extended**
(`attestedRevision` → `e16c6a55`, the reading appended to `why`); no file is added to any
`filesChangedSinceMeasurement`. No re-measure was taken. Matrix after `--stamp-anchors`:
`15 anchors, 10 stale, 5 current by attestation` (unchanged set).

## Measured lower bounds attested at `e16c6a55` (sc-22738, 2026-09-07)

A measured lower bound (`exceededBounds`, sc-22738) is stamped by the same `--stamp-anchors` walk
as an anchor and reads its attestation from this same file, keyed by the bound's store id. It is
held to the same two-gated doctrine, with one difference in what is claimed: an anchor's attestation
says the loader still produces the measured *phase decomposition*; a bound's says the diff cannot
change the *peak the stop was reached at*, so a re-run would only re-establish the same inequality
and the probe tooling (`measure-memory-catalog.mjs` `exceeded_current`; the MLX memory adapter's
pre-load refusal, which since this story refuses only a **current** bound) should not book the
guarded render again. Nothing here reaches production: the runtime never demotes a stale bound and
never reads currency, so an attestation changes what the campaign schedules and nothing else.

Two bounds are on file in this tree, both measured at `3b922bac` on the 137,438,953,472-byte
capture Mac; both are attested to `e16c6a55` with `class: accounting-only`, each closure read file
by file against `3b922bac..e16c6a55`:

| Bound | `closureFiles` ∩ `3b922bac..e16c6a55` | Reading |
| --- | --- | --- |
| `bernini:mlx` bf16 (`exc-15a6249c…`, 848x480x49, footprint hard stop at 97,147,294,328 B mid z16 VAE decode) | `gen-core/wan_i2v_memory.rs`, `mlx-gen-bernini/src/bernini.rs`, `mlx-gen-wan/src/{config,i2v_memory_strategy,memory_strategy,model}.rs` | `bernini.rs` is the only executed file that changed and its whole change is two `Progress::Loading` emissions (before the planner load, before the expert loads) plus an import and a weights-free test — instrumentation. `wan_i2v_memory.rs` is packed-tier contract pricing Bernini never calls (and this is the dense bf16 tier). The four `mlx-gen-wan` files change the TI2V-5B contract identity (`config.rs`, `memory_strategy.rs`), tests only (`i2v_memory_strategy.rs`), and `Wan14b`'s own loader/decode-tiling wrapper (`model.rs`), none of which Bernini links or calls (its decode tiling is `mlx_gen_bernini::memory_strategy::decode_tiling` / `TilingConfig::auto`). The z16 decode tile budget the stop was reached in — `mlx-gen-wan/src/pipeline.rs` `auto_tiling_budgeted_z16*`, `mlx-gen/src/vae_tiling.rs` — is not in the diff. |
| `flux2_dev:mlx` bf16 (`exc-2c3953f5…`, 512x512x1, Metal submissions-ignored refusal at 86,988,010,336 B against the 87,044,670,532 B wired limit) | `gen-core/src/residency.rs` | `mlx-gen-flux2` drives `Residency` (`from_policy`, `sequential`; the bf16 resident load is the warm arm), so the moved boundaries are on its path — but the diff is the instrumentation-only reading recorded above: components loaded, order, devices, offload policy and every allocation are byte-identical, so the resident working set that reached the wired limit is the same working set. |

No re-measure was taken for either; under the doctrine none is needed, because no changed file
alters what is loaded or how much memory the peak path takes. Matrix after `--stamp-anchors`:
`15 anchors, 10 stale, 5 current by attestation` (unchanged — the matrix publishes anchors only);
both bounds now stamp at the pin's declared digest and classify `exceeded_current` in the catalog
walk.

**Deliberately not attested in PR #2782, attested in the campaign evidence PR.** At the time of
PR #2782 the sc-22738 MLX campaign branch carried six more bounds measured at `3b922bac` —
`bernini` q4/q8, `bernini_image` bf16, `krea_realtime_14b` bf16/q4/q8 — and the running campaign
later added `wan_2_2_i2v_14b` bf16. Their evidence records were not in that tree, so an entry for
them could not be checked against a record (`measuredRevision` must equal the cited record's
revision) and would have been silently inert in `--stamp-anchors` until the campaign branch merged.
The seven entries were written in the PR that assembles the campaign evidence (walk-1 ∪ rerun),
beside the records they cite; each was re-read against `3b922bac..e16c6a55` rather than copied:

| Bound | `closureFiles` ∩ `3b922bac..e16c6a55` | Reading |
| --- | --- | --- |
| `bernini:mlx` q4 (`exc-3ff33b42…`, 848x480x49, stop at 108,484,649,848 B) and q8 (`exc-086e91ed…`, 848x480x49, stop at 108,595,791,424 B) | the same six files as the bf16 row above | the bf16 reading verbatim, with one clause re-checked because it no longer follows from "bf16 is dense": `packed_transformer_bytes` is the Wan I2V contract walker's packed-tier pricing arm, and `mlx-gen-bernini/src` carries no reference to `wan_i2v_memory` (grep: 0), so the tier does not matter. |
| `bernini_image:mlx` bf16 (`exc-55ce7578…`, 848x480x1, stop at 94,852,699,512 B) | the same six files — `bernini_image:mlx`'s closure is the same 117 files as `bernini:mlx`, differing only in its entry points | the bf16 reading verbatim. |
| `krea_realtime_14b:mlx` bf16 (`exc-596a1ae3…`, stop at 109,921,567,216 B), q4 (`exc-c8c7f41a…`, 110,079,800,312 B), q8 (`exc-e3635aa6…`, 110,094,811,152 B), all 832x480x45 | `gen-core/wan_i2v_memory.rs`, `mlx-gen-wan/src/{config,i2v_memory_strategy,memory_strategy,model}.rs` | `mlx-gen-krea-realtime/src` has no reference to `wan_i2v_memory`, `Wan14b`, `load_t2v_14b` or `load_i2v_14b` (grep: 0); from `model.rs` it takes only `effective_te_quant` (t2v.rs:59, :995) and `descriptor_t2v_14b` (pipeline.rs:881), both untouched; its decode tiling is `pipeline.rs` `auto_tiling_budgeted_z16_quality_overlap` (t2v.rs:60, :292), not in the diff; `config.rs` only adds `LEGACY_INERT_CONFIG_KEYS` / `without_legacy_inert_keys` beside the `WanModelConfig` / `WanQuant` / `GuideScale` it imports; `memory_strategy.rs`'s `canonical_config` is reached only from the `wan2_2_ti2v_5b` `load` (model.rs:441); `i2v_memory_strategy.rs`'s single hunk starts at line 659, inside `mod tests`. |
| `wan_2_2_i2v_14b:mlx` bf16 (`exc-8071db43…`, 1280x720x77, stop at 106,174,403,704 B) | the same five files | the route loads through `load_i2v_14b` (model.rs:2214) → `i2v_memory_strategy::prepare(…, MODEL_ID_I2V_14B)`, unchanged; `Wan14b::generate_impl`'s decode tiling is now selected by `a14b_decode_tiling`, and for an `image_to_video` request the arm taken is the same `i2v_memory_strategy::decode_tiling` call with or without `memory.tile_vae_decode`, while a request with no memory contract falls to `auto_tiling_budgeted_z16` at both revisions — identical for every I2V input; `load_t2v_14b`'s receipt seal is the T2V route; `packed_transformer_bytes` is the `(Denoise, Some(quant))` arm (wan_i2v_memory.rs:1429) and bf16 takes the dense arm; `canonical_config` is the TI2V-5B loader's. |

None of the seven records names the phase the stop landed in (a bound record carries the ceiling,
the observed footprint and the geometry), and the entries claim nothing about that. No re-measure
was taken. After `--stamp-anchors` all nine sc-22738 bounds stamp at the pin's declared digest and
classify `exceeded_current` in the catalog walk. The 49 walk-1 anchors measured at `563c44e1`
(chroma, flux_dev/schnell, flux2_dev q4/q8, klein, ideogram, kolors, lens, qwen, sana) are
**not** attested: `563c44e1..e16c6a55` is 25 inference files, a different and larger diff, and
nobody has read it per closure. They keep their own measurement-revision keys and read stale in
the probe tooling; the runtime treats them as measured either way.
