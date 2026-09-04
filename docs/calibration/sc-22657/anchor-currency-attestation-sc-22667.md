# Anchor currency attestations at inference `c6d6a4db` (sc-22667, epic sc-22657)

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
