# The 47 `manifest_tier_declaration` cells that remain at sc-22667 close-out

Epic sc-22657's AC1 reads "no `manifest_tier_declaration` cell remains for a model whose contract
publishes both fact sets". At inference `c6d6a4db` the store still classifies 47 analytic-only
candle cells that way (`config/memory-anchors.json`, `analyticOnly[].basis`). This note records the
mechanism, so the close-out can cite it rather than the count: **none of the 47 is priced by the
contract ladder at admission, and the extractor's classification is the truth about them.**

## The mechanism

`scripts/extract-memory-anchors.mjs` (`contractEstimateEvidence`, and the doc block above
`CONTRACT_LADDER_BACKENDS`) classifies a cell `contract_estimate` only when the worker would
actually build the per-rung pseudo-anchor for it. The worker does that in exactly one place —
`candle_memory_strategy.rs`'s `floor_pseudo_anchor` (via `floor_anchor`) — and only when **all
three** hold:

1. the model's `candle` block in `config/manifests/builtin.models.jsonc` publishes a
   `memoryStrategyContract` whose `implementations` declare rungs;
2. the manifest declares the RAW staged row the law decomposes, `candle.sequentialPeakGb[tier]`
   (`vram_gate::measured_sequential_peak_gb`, with its `q8` fallback for an unmeasured `nvfp4`);
3. the route is not receipt-priced (`is_receipt_priced` / `RECEIPT_PRICED_ROUTES`): a
   receipt-priced family's floor is a structural weights-plus-headroom sum sealed from the provider
   receipt and is never rescaled.

The contract's own asset facts and architecture facts (the "both fact sets") are read by the worker
at admission behind the provider surface; the generator cannot resolve them at the pin, and they
are not what gates the ladder anyway — the staged row is. A model that publishes both fact sets but
no `sequentialPeakGb` row has nothing for the ladder to rescale, so the pseudo-anchor is never
built and the cell falls through to the manifest's per-tier envelope, which is what
`manifest_tier_declaration` says.

## The 47, by which input is missing

| Lane (candle) | Cells | Contract published | `sequentialPeakGb` row | Receipt-priced | Why the ladder does not run |
| --- | ---: | --- | --- | --- | --- |
| `boogu_image`, `boogu_image_turbo` | 6 | yes | none | no | no staged row to decompose |
| `sd3_5_large` | 2 | yes | none | yes | no staged row, and receipt-priced |
| `ideogram_4`, `ideogram_4_turbo` | 6 | yes | yes | yes | receipt-priced: the floor is the sealed receipt sum |
| `kolors` | 3 | no | none | yes | no contract block in the manifest |
| `scail2_14b` | 3 | no | none | no | no contract block in the manifest |
| `sensenova_u1_8b` + `_fast` + `_infographic_v2/v3` (+ `_fast`) | 18 | no | none | no | no contract block in the manifest |
| `wan_2_2`, `wan_2_2_i2v_14b`, `wan_2_2_t2v_14b` | 9 | no | none | no | no contract block in the manifest (video lanes; the image ladder is not their mechanism) |

Only the first three rows (14 cells) are models whose manifest block publishes a contract at all;
for every one of them the missing input is the staged row and/or the receipt-priced route, never
the fact sets. The remaining 33 cells belong to models whose manifest `candle` block publishes no
`memoryStrategyContract`, so the AC's premise ("a model whose contract publishes both fact sets")
does not describe them.

## What would move a cell out of this class

* A measured `candle.sequentialPeakGb[tier]` row for a contract-bearing, non-receipt route
  (boogu, sd3_5_large) — one staged capture per tier, on demand, never per pin.
* Nothing, for a receipt-priced route: the receipt floor is the designed pricing and the ladder is
  deliberately not applied to it.
* A contract block, for the 33 contract-less lanes — which is a provider port question, not a
  memory-derivation one.

No code change accompanies this note (sc-22667 review, minor): the extractor already states the
mechanism in its own reason string per cell, and the worker's `floor_pseudo_anchor` is the single
implementation it mirrors.
