// RAM-based quant-tier download suggestion (sc-8509, epic 8506). The Models page lets a user pick
// which quant tier(s) of a model to DOWNLOAD, with a suggested default: the highest-fidelity tier
// that should fit the host's memory. This module is the pure, unit-testable core of that logic —
// deliberately separate from the React screen (like sc-8515's quantTier.js) so sc-8516 can later
// refine the thresholds/constants in ONE place once measured footprints land.
//
// SUGGEST, NEVER WITHHOLD (epic 8506 decision 1): every tier stays installable regardless of RAM.
// The suggestion only preselects/highlights the recommended tier; it never removes a tier from the
// installable set. `suggestTier` returning a smaller tier does not make bf16 un-installable — the
// screen keeps every tier's checkbox enabled.
//
// Consumes the sc-8508 per-variant catalog shape: each `model.variants[]` entry carries a `variant`
// key (bf16/q8/q4) and a `footprint` object. `footprint.diskSizeBytes` is always populated;
// `footprint.residentMemoryBytes` / `peakMemoryBytes` are the MEASURED memory fields sc-8516 populates
// (for the tiers it measured on-device; the rest stay null and fall back to the calibrated estimate).
//
// sc-8516 CALIBRATION (basis for the constants below): on-device measurement of steady-state resident
// + load+gen peak GPU memory (harness crates/sceneworks-worker/src/footprint_measure.rs — ONE tier per
// fresh process, resident sampled post-gen AFTER releasing the transient cache, peak = load+gen
// high-water — using the MLX counters mlx_rs::memory::{get_active_memory, get_peak_memory} the worker
// already publishes). Measured set: sdxl/q8, z_image/q4, z_image_turbo/q4, lens_turbo/q4. Two findings:
//   1. resident ≈ on-disk size (ratio 0.81–1.01, mean 0.94) — the old disk×1.5 resident estimate was
//      too high; packed weights sit resident at roughly their on-disk size.
//   2. peak − resident is a FIXED 14.04 GiB transient (1024² VAE decode + attention working set),
//      measured to within ~4 MiB across a 3.9→16.5 GiB resident spread over 3 different VAEs. It is
//      genuinely resolution-bound, NOT weight-bound (a real physical property of the 1024² decode, not
//      a "resident + constant" measurement artifact) — so peak = resident + a fixed addend, not ×N.
// The suggestion therefore budgets against PEAK (the real ceiling a generation must fit).

import { installedTiers, isSelectableTier, tierHostEligible, tierQuantize } from "./quantTier.js";

// Fidelity order, HIGHEST first. The suggestion walks this list and picks the first tier that both
// exists on the model AND fits memory; bf16 is preferred, then q8, then q4 (the smallest, always-fits
// fallback). Any declared tier not in this list is considered lowest-fidelity and only chosen last.
const TIER_FIDELITY = ["bf16", "q8", "q4"];

// Resident-memory estimate multiplier over on-disk size, used ONLY when a variant lacks any measured
// footprint. CALIBRATED by sc-8516 from on-device measurement (harness: crates/sceneworks-worker/src/
// footprint_measure.rs). Across sdxl-q8 / z-image-q4 / z-image-turbo-q4 / lens-turbo-q4 the measured
// steady-state RESIDENT/disk ratio was 0.81–1.01 (mean 0.94): packed weights sit resident at roughly
// their on-disk size, NOT the old 1.5× guess. We estimate resident ≈ disk × 1.0, then add the fixed
// transient below (which is where the real generation headroom lives).
export const DISK_TO_RESIDENT_MULTIPLIER = 1.0;

// Fixed transient working-set (bytes) a single 1024² generation needs ON TOP OF the resident weights —
// VAE decode buffers + attention activations/latents + framework scratch. sc-8516 measured this as the
// PEAK − RESIDENT gap and found it 14.04 GiB to within ~4 MiB across a 3.9→16.5 GiB resident spread
// (sdxl / z-image / lens, 3 different VAEs) — genuinely size-INDEPENDENT and resolution-bound, so it is
// modeled as a fixed addend, not a multiplier. 14 GiB ≈ the measured value, used directly. This is what
// makes the estimated budget (disk×MULT + transient) track the MEASURED peak the RAM suggestion must
// actually fit — the true install-time/run ceiling.
export const TRANSIENT_HEADROOM_BYTES = 14 * 1024 * 1024 * 1024;

// Fraction of detected unified/GPU memory a tier's peak footprint must fit UNDER to be suggested. The
// remainder is left for the OS, other apps, and margin. sc-8516 raised this from 0.8 → 0.9: because the
// budget is now peak-inclusive (weights + the fixed transient) rather than resident-only, the extra
// slack the old 0.8 baked in for un-modeled transient is already accounted for explicitly, and 0.9
// keeps the suggestion from needlessly under-picking on right-sized hardware. So on a 32 GB Mac a tier
// "fits" only if its peak is under 32 × 0.9 = 28.8 GB.
export const MEMORY_HEADROOM_FRACTION = 0.9;

const BYTES_PER_GB = 1024 * 1024 * 1024;

// A variant's PEAK memory requirement in bytes (the ceiling the suggestion must fit), or null when it
// can't be estimated. Basis, in priority order:
//   1. `footprint.peakMemoryBytes` — the MEASURED load+gen high-water mark, used verbatim (sc-8516).
//      This is the true ceiling: a tier whose peak OOMs during generation must not be suggested.
//   2. `footprint.residentMemoryBytes` + TRANSIENT_HEADROOM_BYTES — measured resident weights plus the
//      fixed transient working set, when resident was measured but peak was not.
//   3. `footprint.diskSizeBytes` × DISK_TO_RESIDENT_MULTIPLIER + TRANSIENT_HEADROOM_BYTES — the
//      estimate (the common case for un-measured tiers): weights ≈ on-disk size, plus the transient.
//   4. `downloadSizeBytes` × DISK_TO_RESIDENT_MULTIPLIER + TRANSIENT_HEADROOM_BYTES — last-resort when
//      the footprint object is absent but the catalog still knows the tier's download size.
// `measured` reports whether the value came from a measured field (peak or resident) vs the estimate.
export function variantFootprintBytes(variant) {
  const footprint = variant?.footprint;
  const peak = numberOrNull(footprint?.peakMemoryBytes);
  if (peak !== null && peak > 0) {
    return { bytes: peak, measured: true };
  }
  const resident = numberOrNull(footprint?.residentMemoryBytes);
  if (resident !== null && resident > 0) {
    return { bytes: resident + TRANSIENT_HEADROOM_BYTES, measured: true };
  }
  const disk = numberOrNull(footprint?.diskSizeBytes);
  if (disk !== null && disk > 0) {
    return {
      bytes: Math.round(disk * DISK_TO_RESIDENT_MULTIPLIER) + TRANSIENT_HEADROOM_BYTES,
      measured: false,
    };
  }
  const download = numberOrNull(variant?.downloadSizeBytes);
  if (download !== null && download > 0) {
    return {
      bytes: Math.round(download * DISK_TO_RESIDENT_MULTIPLIER) + TRANSIENT_HEADROOM_BYTES,
      measured: false,
    };
  }
  return null;
}

function numberOrNull(value) {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

// ---------------------------------------------------------------------------------------------------
// Per-tier MEASURED memory floor (sc-15400), PER BACKEND
//
// `mlx.minMemoryGb` / `candle.minMemoryGb` is a SINGLE per-model integer, and it OVER-states the
// requirement for every user who only installed a lighter tier: krea_realtime_14b is forced to 64 while
// its default q4 tier peaks at 27.90 GiB (`downloads[].footprint.peakMemoryBytes` 29,957,689,344 — with
// q8 at 34.44 and bf16 at 46.71 GiB). The manifest already carries the real per-tier ladder, so a
// consumer that knows WHICH tier is in play should read that tier's measured peak.
//
// WHAT THE BLANKET INTEGER IS NOT: it is NOT a ceiling over the model's tiers. An earlier revision of
// this comment asserted that the single integer "must admit the model's heaviest INSTALLABLE tier";
// that premise is wrong, and both owners of the field say so outright:
//   * the schema (`model-manifest.schema.json`, `candle.minMemoryGb`) — "the measured overall-peak of
//     the DEFAULT (lightest, typically q4) hosted tier, plus headroom. This gates the default tier
//     only ... heavier hosted tiers (q8 / bf16) can exceed this value, and their per-tier measured
//     ceilings live in vramGbByTier";
//   * `crates/sceneworks-worker/src/vram_gate.rs:180` — "`minMemoryGb` is the WRONG floor to land on
//     here: per the manifest schema it is the measured overall-peak of the DEFAULT (lightest, typically
//     q4) hosted tier, which heavier tiers are explicitly allowed to exceed — so falling through to it
//     would ... fail PERMISSIVELY (an under-prediction admits a load that can OOM)".
// The catalog bears this out: `flux_dev` declares `candle.minMemoryGb` 24 with a measured
// `candle.vramGbByTier.q8` of 31.8, and `krea_2_turbo` declares 32 against a measured bf16 of 47.2.
// So the blanket is a FLOOR, never a fallback that can stand in for a heavier tier's requirement — and
// a consumer that cannot measure every installed tier must take the MAX of the two, not the blanket
// alone (`installedFloorHostGb` below).
//
// THE TWO LANES CARRY INDEPENDENT EVIDENCE AND MUST NEVER BE CROSSED (sc-15613). The per-download
// `footprint` block is an MLX-ONLY measurement: the harness that produces it is
// `crates/sceneworks-worker/src/footprint_measure.rs`, sampling `mlx_rs::memory::get_peak_memory` — an
// Apple unified-memory high-water mark. A discrete CUDA card does not share system RAM, so that number
// does not describe the VRAM it must hold, and the schema says so outright at the `candle` block
// (model-manifest.schema.json): "the mlx.minMemoryGb ... does not describe the VRAM a discrete GPU must
// hold". The candle lane's own per-tier evidence is `candle.vramGbByTier` (populated on 33 catalog
// entries), with `candle.minMemoryGb` as its blanket.
//
// Crossing them is NOT merely imprecise, it can UNDER-state, which is the one direction that OOMs:
// `lens_turbo` measures 30.50 GiB on MLX while its candle q4 row declares 37.3 GB of VRAM at the same
// tier (that row is `candle.measured: false` — an explicit ESTIMATE, not a measurement; see below), and
// `qwen_image` declares `mlx.minMemoryGb` 50 against a `candle.minMemoryGb` of 56. So there is no "the MLX number is
// at least conservative" shortcut — every helper below takes the lane explicitly and reads only that
// lane's fields. `options.backend === "candle"` selects the candle lane; anything else is MLX (the
// convention every existing caller already uses: `macGatingActive ? "mlx" : "candle"`).
//
// MEASURED EVIDENCE IS PREFERRED, BUT AN UNEVIDENCED INSTALLED TIER IS *ESTIMATED*, NOT IGNORED
// (sc-15400 round 4). An earlier revision of this header claimed these helpers "never SYNTHESIZE a
// floor" — only ever reading a number the manifest declares for the lane, with "no disk-size modelling
// anywhere below". That framing produced a wrong CLAIM rather than a conservative one, because the
// ceiling is taken over the INSTALLED SET: with q4 measured and q8/bf16 carrying only `diskSizeBytes`,
// `z_image_edit` (which declares NO `mlx` block, so there is no blanket to fall back on) quoted q4's
// 19.42 GiB peak — 22 GB — as the requirement for a `{q4,bf16}` install whose bf16 tier this module's
// OWN estimator puts at 33.13 GiB, i.e. a 37 GB host. `tierFits(bf16, 22)` is false and
// `suggestTier(entry, 22)` is `q4`, so the label contradicted the picker on the same screen.
//
// So: a per-tier MEASURED number is used verbatim where it exists (that is still the whole point of
// sc-15400), and an installed tier with NO number on this lane is filled from `variantFootprintBytes`
// — the SAME disk-based estimate `tierFits` and `suggestTier` already rank with — so the ceiling bounds
// the WHOLE install set instead of a strict subset of it. Two rules keep the estimate honest:
//   * it is MLX-LANE ONLY (`mlxEstimatedPeakGb`). `variantFootprintBytes` prefers
//     `footprint.peakMemoryBytes`, an Apple unified-memory measurement, and 18 shipped (model, os, tier)
//     rows are an unevidenced CANDLE tier that carries exactly that field (`z_image_edit`/`z_image` q4,
//     `sdxl` q8, `illustrious_xl_v1`/`v2` all three on windows+linux) — filling from it would be the
//     lane-crossing this module forbids. Its disk fallback is no better on candle: the addend is
//     `TRANSIENT_HEADROOM_BYTES`, a 14 GiB MLX 1024²-VAE-decode measurement with no candle counterpart,
//     and candle converts additively, so the two would not even compose into a comparable quantity.
//   * a filled tier makes the answer ESTIMATED, so it can never LOWER the blanket (`maxHostGb`), exactly
//     as `laneEvidenceEstimated` does. Without that, `qwen_image`'s `{q4}` install would read its
//     filled q4 estimate INSTEAD OF the larger blanket 50 — trading one under-statement for another.
// When a tier has no lane number AND no estimable size either (`krea_2_raw` ships neither a footprint
// nor a `diskSizeBytes`), the set stays genuinely unbounded and `installedFloorHostGb` says so.
//
// "Declared" is not the same as "measured", and the candle lane says which it is. `candle.measured`
// is `false` on 13 shipped entries; five of the 13 also carry `vramGbByTier` (`lens_turbo`,
// `flux_schnell`, `krea_2_raw`, `sd3_5_large_turbo`, `sd3_5_medium`) and are the set meant here, the
// other eight being blanket-only audio entries (`kokoro_82m`, `chatterbox_tts`, …) where the flag has
// nothing per-tier to qualify — schema: "false when ESTIMATED (scaled from weight size + a measured
// sibling) pending a real measurement". The flag covers `minMemoryGb` AND `vramGbByTier` TOGETHER,
// which decides what to do about it: on such an entry the blanket integer is ITSELF an estimate, so
// "distrust the per-tier row and fall through to the blanket" would merely swap one estimate for
// another. What it CAN do is refuse to let either estimate lower the other, so `laneEvidenceEstimated`
// makes the blanket a FLOOR under the per-tier figure (`max`, identical to the incomplete-coverage rule
// above). Concretely: `lens_turbo`'s estimated candle q4 row of 37.3 converts to a 40 GB host, below the
// curated blanket of 44 — so the row reports 44. There is no `mlx.measured` counterpart because the MLX
// per-tier evidence is a harness measurement by construction (`footprint_measure.rs`), so the flag is
// read on the candle lane only.
// ---------------------------------------------------------------------------------------------------

// A single variant's MEASURED peak memory in GB (base 1024³, the same unit `variantFootprintBytes`
// budgets in), or null when this tier has no measured `footprint.peakMemoryBytes`. MLX LANE ONLY — see
// the harness note above.
function measuredVariantPeakGb(variant) {
  const peak = numberOrNull(variant?.footprint?.peakMemoryBytes);
  return peak !== null && peak > 0 ? peak / BYTES_PER_GB : null;
}

// Declared quant tier → the MLX measured peak (GB) for that tier, for the real bits-based tiers only
// (bf16/q8/q4). The "default" pseudo-variant of a single-variant model and the non-quant "training" base
// are excluded, so neither can ever be mistaken for a generation tier's footprint.
//
// DUPLICATE tier keys resolve to the MAX measured peak rather than last-write-wins. The manifest
// legitimately declares several downloads under one `variant` key — `mage_flow` carries each of q4/q8/bf16
// THREE times (main weights, plus a text_encoder and a vae component), and `wan_2_2_t2v_14b` carries each
// tier twice (a macOS/MLX entry with footprints and a windows/linux candle entry without).
//
// Three upstream filters currently make the emitted keys unique — the component rows are `coRequisite:
// true` (`is_co_requisite_download`), the per-platform rows are dropped by `retain_downloads_for_os`, and
// `model_variant_downloads` then de-dupes first-wins (sc-8508) — so no duplicate reaches this Map today.
// That is verified against the real catalog by memoryFloorCatalogParity.test.js rather than assumed here.
// A Map keyed blindly on `variant.variant` would nonetheless depend silently on all three: the LAST
// duplicate is a component row with no `footprint`, so last-write-wins would erase a real measurement and
// quietly fall back to the blanket. Taking the max over the duplicates that DO carry a measurement agrees
// with the first-wins projection today and errs toward over-stating if any of those filters ever change.
function mlxPeakGbByTier(model) {
  const byTier = new Map();
  for (const variant of model?.variants ?? []) {
    const tier = variant?.variant;
    if (tierQuantize(tier) === null) {
      continue;
    }
    const gb = measuredVariantPeakGb(variant);
    const prior = byTier.has(tier) ? byTier.get(tier) : null;
    if (gb === null) {
      // A duplicate with no footprint is a component subdir, not evidence that the tier is unmeasured.
      if (!byTier.has(tier)) {
        byTier.set(tier, null);
      }
      continue;
    }
    byTier.set(tier, prior === null ? gb : Math.max(prior, gb));
  }
  return byTier;
}

// Declared quant tier → the candle/CUDA measured peak VRAM (GB) for that tier, from
// `candle.vramGbByTier`.
//
// Keyed off `isSelectableTier`, NOT `tierQuantize` — the candle lane's tier vocabulary is WIDER than
// bf16/q8/q4. `int8-convrot` and `nvfp4` are real, user-selectable candle-only tiers with no
// `mlxQuantize` value (they select via `advanced.convRot` / `advanced.quantTier`), and the manifest
// really does ship a measured row for one: `krea_2_turbo` declares `int8-convrot: 34.7` alongside its
// q4/q8/bf16 ladder, and offers it as a download. Keying off `tierQuantize` dropped that row while
// `installedTiers` still returned the tier, so ONE candle-only tier in the install set collapsed the
// whole measured ceiling to null and the row fell back to the blanket 32 — under-stating a measured
// 34.7 GB requirement. That was the mechanism, not an edge case.
function candlePeakGbByTier(model) {
  const byTier = new Map();
  const declared = model?.candle?.vramGbByTier;
  if (!declared || typeof declared !== "object") {
    return byTier;
  }
  for (const [tier, value] of Object.entries(declared)) {
    if (!isSelectableTier(tier)) {
      continue;
    }
    const gb = numberOrNull(value);
    byTier.set(tier, gb !== null && gb > 0 ? gb : null);
  }
  return byTier;
}

// Declared quant tier → that tier's measured peak (GB) ON `backend`. The single seam where the lane is
// chosen; everything below reads through it so no consumer can accidentally mix the two.
function peakGbByTier(model, backend) {
  return backend === "candle" ? candlePeakGbByTier(model) : mlxPeakGbByTier(model);
}

// A tier only the candle lane can serve: selectable, but with no bits-based `mlxQuantize` (INT8-ConvRot,
// NVFP4). Expressed as the conjunction rather than a third hand-written list of the two keys.
function isCandleOnlyTier(tier) {
  return isSelectableTier(tier) && tierQuantize(tier) === null;
}

// One tier's declared peak (GB) on `backend`, or null when this lane has no number for it.
//
// A candle-only tier with no measured row DEGRADES TO THE `q8` ROW, never to `minMemoryGb` — mirroring
// `vram_gate::predicted_peak_gb` (sc-11042), whose doc comment is explicit that `minMemoryGb` "is the
// WRONG floor to land on here ... falling through to it would size an FP4 render against the lightest
// tier's number and fail PERMISSIVELY". The q8 row over-predicts NVFP4 (q8's weights are ~2× NVFP4's
// ~4.5 effective bits), which is safe.
//
// Latent today, deliberately: `krea_2_turbo` is the only entry offering a candle-only tier and it ships
// the matching `int8-convrot` row, so nothing currently reaches the degrade (pinned by
// memoryFloorCatalogParity.test.js). One honest caveat if it ever does: sc-12425 measured INT8-ConvRot
// ABOVE q8 on a real trunk (42.9 vs 35.9), so for that tier the q8 row is a known UNDER-prediction of
// the real peak.
//
// Which is why the degrade is reported as `exact: false`. An earlier revision of this comment instead
// reassured that "`installedFloorHostGb` floors the answer at the blanket regardless, so the degrade can
// never report below today's number" — that was FALSE: a degraded figure is non-null, so the install set
// counted as fully evidenced, `installedFloorHostGb` took its `complete` branch, and the converted q8
// number was returned with no blanket floor under it. (`vram_gate::predicted_peak_gb` degrades only
// `nvfp4` and otherwise falls to `minMemoryGb`, so there is no upstream floor either.) `exact: false`
// makes the claim true rather than merely deleting it: a proxy figure is treated exactly like the
// disk-based fill — it may raise the answer, never lower it below the blanket.
function declaredPeakGb(byTier, backend, tier) {
  const own = byTier.get(tier) ?? null;
  if (own !== null) {
    return { gb: own, exact: true };
  }
  if (backend === "candle" && isCandleOnlyTier(tier)) {
    const degraded = byTier.get("q8") ?? null;
    // A non-null answer here is the q8 PROXY, not this tier's own evidence.
    return { gb: degraded, exact: degraded === null };
  }
  return { gb: null, exact: true };
}

// The model's BLANKET declared floor (GB) on `backend`, or null when that lane declares none. The
// fallback for every helper below — and deliberately NOT cross-lane: a candle host with no candle
// evidence gets null (the caller shows nothing) rather than the MLX integer, because `qwen_image`'s
// mlx 50 / candle 56 proves the MLX blanket can sit BELOW the candle requirement.
export function blanketFloorGb(model, backend) {
  const block = backend === "candle" ? model?.candle : model?.mlx;
  const gb = numberOrNull(block?.minMemoryGb);
  return gb !== null && gb > 0 ? gb : null;
}

// Fixed transient/runtime headroom (GB) the CANDLE lane adds on top of a per-tier measured peak. Quoted
// from its owner, `crates/sceneworks-worker/src/vram_gate.rs` `HEADROOM_GB`, which is the gate that
// actually rejects a candle load: "Fixed transient/runtime headroom (GB) added on top of a per-tier
// MEASURED peak (`candle.vramGbByTier`) to cover allocator slack + activation spikes not captured by the
// steady peak. Not added on top of `candle.minMemoryGb`, which the manifest already pads over the
// measured peak (sc-9094)."
//
// SC-15614 owns unifying reserve policy across the lanes. Until it lands this is a MIRROR of the Rust
// constant, not a fourth independent source: if the gate's number moves, this must move with it. There
// is no shared JS home for it today — `resolutionMemory.js` imports `MEMORY_HEADROOM_FRACTION` from here,
// so here is the one place both lanes' reserve criteria are already stated.
export const CANDLE_HEADROOM_GB = 2.0;

// The host size (GB) at which a `peakGb` requirement satisfies the fit criterion OF ITS OWN LANE. This is
// what a USER-FACING "needs N GB" must report: a measured peak is a RAW high-water mark, so quoting it
// directly claims a host size at which the model would not actually run.
//
// THE TWO LANES USE DIFFERENT CRITERIA, and quoting the wrong one over-warns on the very lane this
// module just started reading:
//   * MLX — MULTIPLICATIVE. `peak <= host * MEMORY_HEADROOM_FRACTION` (0.9), the budget `tierFits` and
//     `resolutionMemory.resolutionFitsMemory` both apply, so the host is `ceil(peak / 0.9)`. A 31 GB Mac
//     fails lens_turbo's 30.50 GiB q4 because 31 × 0.9 = 27.9.
//   * candle — ADDITIVE. `vram_gate` admits a load at `measured peak + CANDLE_HEADROOM_GB`, so the host
//     is `ceil(peak + 2)`. Dividing instead over-states against the gate that actually rejects: on 11 of
//     the 95 shipped candle rows the fractional form lands >= 5 GB high, worst at `flux2_dev` bf16 —
//     143 shown against 130 required, +13.0 — with `qwen_image_edit_2511_lightning` bf16 at 98 vs 89.4
//     and `qwen_image` bf16 at 92 vs 84.5.
//
// The blanket `minMemoryGb` integers are already stated in host terms on both lanes — the manifest says
// so at krea_realtime_14b ("~47 GiB active peak. 64 covers that with OS/app headroom") and `vram_gate`
// says so for candle ("which the manifest already pads over the measured peak") — so converting here is
// what makes the measured basis and the blanket basis comparable, which is what lets them be `max`ed.
export function hostGbForPeakGb(peakGb, backend) {
  if (peakGb == null || !Number.isFinite(peakGb) || peakGb <= 0) {
    return null;
  }
  return backend === "candle"
    ? Math.ceil(peakGb + CANDLE_HEADROOM_GB)
    : Math.ceil(peakGb / MEMORY_HEADROOM_FRACTION);
}

// Whether `backend`'s per-tier numbers for `model` are ESTIMATED rather than measured. See the
// `candle.measured` discussion in the section header: the flag covers `minMemoryGb` and `vramGbByTier`
// together, so it does not make one of them trustworthy — it makes NEITHER a safe lower bound on the
// other, and the caller takes the max. MLX has no counterpart flag: its per-tier evidence is a harness
// measurement by construction.
export function laneEvidenceEstimated(model, backend) {
  return backend === "candle" ? model?.candle?.measured === false : false;
}

// The larger of two host sizes, tolerating either (or both) being null.
function maxHostGb(a, b) {
  if (a === null) {
    return b;
  }
  if (b === null) {
    return a;
  }
  return Math.max(a, b);
}

// The declared memory ceiling (GB) on `options.backend` for the tier(s) `model` will actually run at, or
// null unless EVERY installed tier carries a number on that lane.
//
// The ceiling is the MAX over the INSTALLED tiers, because a studio tier picker can switch among them at
// will. The all-or-nothing rule makes this the STRICT primitive: a non-null answer is a real bound on the
// whole install set, which is what the lane-independence and fit-criterion tests assert against. It is
// deliberately NOT the number a display surface should quote on its own — a null here means "unbounded",
// not "no requirement", and `installedFloorHostGb` is the composer that turns both cases into an honest
// host size (it floors the partial ceiling at the blanket rather than discarding it).
//
// `options` also forwards `installedTiers`' host-eligibility gates (convRotEligible / nvfp4Eligible), so
// a tier this host cannot serve never contributes to the ceiling.
export function installedTierPeakGb(model, options = {}) {
  // NO fill: this is the strict primitive, so an unevidenced installed tier makes it null.
  const { ceiling, complete, count } = installedCeiling(model, options);
  // An installed tier we have no number for means the set is unbounded — the caller must not treat the
  // ceiling as the requirement (see `installedFloorHostGb`, which fills or floors it instead).
  return count === 0 || !complete ? null : ceiling;
}

// One installed tier's MLX-lane ESTIMATED peak (GB) — `variantFootprintBytes` in the same base-1024³ unit
// the rest of this section works in, so `hostGbForPeakGb(_, "mlx")` (`ceil(peak / 0.9)`) reproduces
// `tierFits`'s budget EXACTLY and the label can never sit at a host the picker would refuse.
//
// Returns null on candle, unconditionally — see the lane rule in the section header. Also null when the
// catalog knows no size for the tier at all, which keeps "unbounded" distinguishable from "estimated".
function mlxEstimatedPeakGb(model, tier, options) {
  if (options.backend === "candle") {
    return null;
  }
  const variant = (model?.variants ?? []).find((entry) => entry?.variant === tier);
  const footprint = variant ? variantFootprintBytes(variant) : null;
  return footprint === null ? null : footprint.bytes / BYTES_PER_GB;
}

// The max peak (GB) over the installed tiers, plus how it was arrived at:
//   * `complete` — every installed tier contributed a number. `false` with a non-null `ceiling` is the
//     partially-covered case: real and common (`flux_dev` ships measured q4/q8 candle rows and offers a
//     bf16 tier with no row at all).
//   * `estimated` — at least one contribution was NOT that tier's own lane evidence: either a `fill`
//     estimate or `declaredPeakGb`'s candle-only q8 proxy. Such a ceiling may raise the reported host
//     size but must never lower it below the blanket.
//
// `fill` is optional so `installedTierPeakGb` keeps its strict all-or-nothing semantics unchanged.
function installedCeiling(model, options, fill = null) {
  const byTier = peakGbByTier(model, options.backend);
  const tiers = installedTiers(model, options);
  let ceiling = null;
  let complete = true;
  let estimated = false;
  for (const tier of tiers) {
    const declared = declaredPeakGb(byTier, options.backend, tier);
    let gb = declared.gb;
    if (gb !== null && !declared.exact) {
      estimated = true;
    }
    if (gb === null && fill !== null) {
      gb = fill(model, tier, options);
      if (gb !== null) {
        estimated = true;
      }
    }
    if (gb === null) {
      complete = false;
      continue;
    }
    if (ceiling === null || gb > ceiling) {
      ceiling = gb;
    }
  }
  return { ceiling, complete, count: tiers.length, estimated };
}

// The host size (GB) to quote for the tiers a model ACTUALLY HAS INSTALLED on `options.backend`, or null
// when the model has no installed tiers (or the lane no evidence of any kind). The whole per-tier policy
// lives here so the display surface composes wording only.
//
// Three inputs, and the rule is one-directional throughout — under-stating is what makes a user download
// a model that then OOMs, over-stating is merely conservative:
//
//   1. EVERY installed tier carries this lane's OWN measured number, and the lane calls its numbers
//      measured ⇒ the converted ceiling ALONE. This is the whole point of sc-15400: krea_realtime_14b
//      with only q4 on disk reports the 32 GB its 27.90 GiB peak needs, not the blanket 64.
//   2. SOME installed tier has no number of its own ⇒ it is FILLED from `mlxEstimatedPeakGb` where that
//      lane can estimate, and the answer becomes `max(blanket, converted ceiling)`. Two distinct
//      under-statements make both halves load-bearing:
//        * the ceiling must cover the WHOLE set, not the subset that happens to be measured —
//          `z_image_edit` `{q4,bf16}` quoted q4's 22 against a bf16 estimate of 37, with no blanket to
//          catch it (see the section header);
//        * and the blanket must still floor it, because an ESTIMATE may not lower a curated number —
//          `qwen_image` `{q4}` would otherwise read its filled q4 estimate instead of the larger 50.
//      The blanket alone was never right either: it is the DEFAULT (lightest) tier's peak plus headroom,
//      so it can sit below a heavier installed tier's MEASURED requirement — `flux_dev`'s 24 against a
//      measured q8 of 31.8 (a 34 GB host), `krea_2_turbo`'s 32 against a measured bf16 of 47.2 (50 GB).
//   3. The lane flags its numbers ESTIMATED, or a candle-only tier fell back to the q8 proxy ⇒ `max` as
//      well, for the reason in the section header.
//   4. STILL unbounded after the fill (an installed tier has no lane number, and no estimable size on a
//      lane that can estimate) AND no blanket ⇒ null, so the surface says NOTHING. A ceiling over a
//      strict subset is not a bound on the set, and quoting it is a false claim rather than a
//      conservative one. Silence is the honest answer; `installedTierPeakGb`'s doc calls this
//      "unbounded", and this is where that distinction is finally respected instead of discarded.
//      Unreachable on the shipped catalog (pinned by memoryFloorCatalogParity.test.js) — every lane that
//      cannot estimate also ships a blanket wherever it ships any per-tier row.
//
// `options` forwards `installedTiers`' host-eligibility gates (convRotEligible / nvfp4Eligible), so a
// tier this host cannot serve never raises the number.
export function installedFloorHostGb(model, options = {}) {
  const backend = options.backend;
  const { ceiling, complete, estimated, count } = installedCeiling(
    model,
    options,
    mlxEstimatedPeakGb,
  );
  if (count === 0) {
    return null;
  }
  const converted = hostGbForPeakGb(ceiling, backend);
  if (complete && !estimated && !laneEvidenceEstimated(model, backend)) {
    return converted;
  }
  const blanket = blanketFloorGb(model, backend);
  if (!complete && blanket === null) {
    return null;
  }
  return maxHostGb(converted, blanket);
}

// The host size (GB) to quote as an ENTRY floor for a model that is NOT installed yet, or null when the
// lane has no per-tier evidence. Same estimated-lane rule as above; see `cheapestDeclaredTierPeakGb` for
// why this is a floor rather than a specific tier's requirement.
export function declaredFloorHostGb(model, options = {}) {
  const backend = options.backend;
  const converted = hostGbForPeakGb(cheapestDeclaredTierPeakGb(model, options), backend);
  if (converted === null || !laneEvidenceEstimated(model, backend)) {
    return converted;
  }
  return maxHostGb(converted, blanketFloorGb(model, backend));
}

// The CHEAPEST measured tier peak (GB) on `options.backend` among the model's declared quant tiers, or
// null when none is measured. This answers "what does it take to run this model AT ALL", which is the
// only honest per-tier statement available for a model that is NOT installed yet: the catalog does not
// emit the manifest's per-download `default` flag (see the `variant.default` note in quantTier.js
// `defaultTierSelection`), so the client cannot know which tier a download will fetch — and it must not
// assume the lightest, because 3 of the 53 shipped matrix models declare a heavier default (krea_2_raw /
// krea_2_turbo default to q8, instantid_realvisxl to bf16).
//
// Callers MUST therefore present this as a FLOOR ("from N GB"), never as a specific tier's requirement.
//
// `options` forwards the same host-eligibility gates as `installedTierPeakGb`: a tier this host cannot
// serve must not set the entry price either. Without the gate an ineligible host could be quoted the
// `int8-convrot` row of a tier it will never be offered.
export function cheapestDeclaredTierPeakGb(model, options = {}) {
  let cheapest = null;
  for (const [tier, gb] of peakGbByTier(model, options.backend)) {
    if (!tierHostEligible(tier, options)) {
      continue;
    }
    if (gb !== null && (cheapest === null || gb < cheapest)) {
      cheapest = gb;
    }
  }
  return cheapest;
}

// The model's real quant tiers (bf16/q8/q4), in fidelity order (highest first). Excludes the
// single-variant "default" pseudo-tier and any unknown key. Every declared tier is included whether
// or not it's installed — the suggestion is about what to DOWNLOAD, so uninstalled tiers count.
export function declaredTiers(model) {
  if (!model?.hasVariantMatrix || !Array.isArray(model.variants)) {
    return [];
  }
  const keys = model.variants
    .map((variant) => variant?.variant)
    .filter((key) => tierQuantize(key) !== null);
  const unique = [...new Set(keys)];
  return unique.sort((a, b) => fidelityRank(a) - fidelityRank(b));
}

// Rank a tier by fidelity, HIGHEST first (bf16=0). Unknown tiers sort last.
function fidelityRank(tier) {
  const index = TIER_FIDELITY.indexOf(tier);
  return index === -1 ? TIER_FIDELITY.length : index;
}

// Whether a variant's PEAK footprint fits within `unifiedMemoryGb` with headroom. When the memory
// signal is unknown (null) OR the tier has no estimable footprint, we treat it as fitting — we never
// withhold or block a tier on missing data; the worst case is suggesting a heavier tier.
export function tierFits(variant, unifiedMemoryGb) {
  if (unifiedMemoryGb == null || !Number.isFinite(unifiedMemoryGb)) {
    return true;
  }
  const footprint = variantFootprintBytes(variant);
  if (footprint === null) {
    return true;
  }
  const budgetBytes = unifiedMemoryGb * BYTES_PER_GB * MEMORY_HEADROOM_FRACTION;
  return footprint.bytes <= budgetBytes;
}

// The suggested default download tier for `model` given the host's `unifiedMemoryGb` (GPU VRAM off
// Mac). Picks the HIGHEST-FIDELITY tier that fits memory with headroom; if none fits (a tiny host,
// or every tier over budget) it falls back to the SMALLEST declared tier so there's always a
// suggestion. Returns null only when the model has no quant matrix.
//
// This never affects installability — it just picks which tier to pre-select/highlight. A 32 GB host
// lands on q4, a 512 GB Studio on bf16, and either can override.
export function suggestTier(model, unifiedMemoryGb) {
  const tiers = declaredTiers(model);
  if (tiers.length === 0) {
    return null;
  }
  const byKey = new Map(
    (model.variants ?? [])
      .filter((variant) => tierQuantize(variant?.variant) !== null)
      .map((variant) => [variant.variant, variant]),
  );
  // `tiers` is already highest-fidelity first; the first that fits wins.
  for (const tier of tiers) {
    const variant = byKey.get(tier);
    if (variant && tierFits(variant, unifiedMemoryGb)) {
      return tier;
    }
  }
  // Nothing fit (every tier over budget) — suggest the smallest so we still preselect something.
  return tiers[tiers.length - 1];
}
