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
