#!/usr/bin/env node

/**
 * SC-16056 — how many measurement runs does certifying all 53 catalog entries actually take?
 *
 * `docs/generated/memory-matrix.json` has thousands of cells. Read naively — one cell, one run —
 * calibration looks unaffordable. This generator refuses to answer that from prose. It reads the
 * generated matrix, derives the cell -> run collapsing rule from the harness code that actually
 * emits records, and keeps three tiers of confidence strictly separated:
 *
 *   CELLS are a FACT           — counted from the artifact.
 *   RUNS are DERIVED           — from the collapsing rule below, every axis cited to code.
 *   HOURS are PARAMETERISED    — per-run cost is an INPUT with a swept range, never an invented fact.
 *
 * Nothing here is an estimate presented as a measurement. Where a value cannot be known today
 * (the final Structurally-N/A count, per-run seconds) the model exposes it as a parameter and
 * reports the sensitivity instead of picking a number.
 *
 * ## sc-18099 — what "all cells" now means
 *
 * The matrix no longer PUBLISHES the catalog cross-product; `cells` there is the planned-or-evidenced
 * subset and `summary.cells` is the resolved coordinate total. This model still counts what the
 * artifact contains, so its population changed from "every coordinate the catalog can express" to
 * "every coordinate somebody planned, measured, bound or cited". That is the workload epic 18093
 * actually intends to price — R6 replaced the per-model measurement campaign with a runbook applied
 * to selected lanes — but it is a DIFFERENT question from the one this model answered before, so the
 * publication counts are republished verbatim in `matrixPublication` rather than left to be inferred
 * from a shrunken total. Read them before comparing this document to an older revision of itself.
 */

import { createHash } from "node:crypto";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";

import { stripJsoncComments } from "./lib/jsonc.mjs";
import { canonicalSourceText, semanticSourceBody } from "./lib/source-revision.mjs";
import { REQUIRED_SCENARIOS } from "./memory-calibration-harness.mjs";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const OUTPUT_JSON = "docs/generated/calibration-cost-model.json";
const OUTPUT_MD = "docs/generated/calibration-cost-model.md";

/**
 * The five ladder rungs, in the epic's normative cost order. `resident` is rung 0 and is
 * deliberately first: it is the one rung that can never be Structurally N/A, because it is the
 * unoptimised baseline every model has by definition. That makes its cell count an irreducible
 * measurement floor, which is the single most load-bearing fact in the N/A sensitivity below.
 */
const RUNG_ORDER = [
  "resident",
  "staged_residency",
  "bounded_decode",
  "bounded_attention",
  "bounded_transformer_residency",
];

const SESSION_KEY_FIELDS = ["modelId", "backend", "tier", "mode", "overlay"];
// Overlay dropped. This keying prices a capability the SHIPPED LOAD CONTRACT DOES NOT OFFER, and it
// is retained only because SC-16072 needs the figure to know what the capability would be worth.
// It is not a cheaper way to run the campaign as the code stands. See OVERLAY_LOAD_CONTRACT.
const OVERLAY_AMORTISED_SESSION_KEY_FIELDS = ["modelId", "backend", "tier", "mode"];

/**
 * Why the overlay axis does NOT collapse on the model-load axis today.
 *
 * An earlier version of this model said the overlay axis "DOES collapse on the model-load axis",
 * "mostly yes", and called it "legitimate for a LoRA a warm generator can swap". That was asserted
 * AGAINST the evidence in this repository. Four findings, three structural and one decisive:
 *
 *   1. ADAPTERS ARE LOAD-TIME, NOT RUNTIME. `resolve_adapters` (base.rs:2585) runs while the load
 *      spec is being built and its output becomes `LoadSpec::adapters` via `spec.with_adapters`
 *      (base.rs:2664-2666), so which adapters a generator has is fixed at construction. The adapter
 *      set is even folded into `GeneratorCacheKey` (generator_cache.rs), so changing it forces a
 *      COLD RELOAD by design rather than mutating a resident generator.
 *   2. NO DETACH/UNLOAD/SWAP API EXISTS. Nothing anywhere in `crates/` removes, disables, replaces
 *      or unfuses an adapter on an already-loaded generator, and the `gen_core::Generator` trait
 *      exposes no such method, so the call is not even expressible. (The engine pin has private
 *      `clear_adapters`/`set_adapters` internals used for per-phase toggling INSIDE one render;
 *      they are not on the trait and are unreachable from SceneWorks.) The collapse assumes an
 *      operation the codebase does not implement.
 *   3. CARRYING ADAPTERS VOIDS THE WHOLE MEASURED LADDER, not just one rung. `allow_streamed_blocks`
 *      is false whenever `adapter_count != 0` (base.rs:5737), which sets `overlay: Some("adapter")`
 *      on the request scope (vram_gate.rs:609) while every evidence cell is built with
 *      `overlay: None` (vram_gate.rs:829-830) — so `memory_strategy.rs:194-200` rules EVERY
 *      candidate out of envelope, including the resident cell, and the job falls back to the older
 *      sequential gate. This is also the exact origin of the 6 `Structurally N/A` cells in this
 *      matrix: all 6 are krea_2_turbo/candle x {q4,q8,bf16} x {lora,control} at
 *      bounded_transformer_residency, cited by the matrix generator to
 *      `base.rs#allow_streamed_blocks` with the reason "load-time adapters are incompatible with
 *      streamed transformer blocks". The overlay axis and the rung axis are not independent.
 *   4. DECISIVE — THE COLLAPSE PERTURBS THE MEASURED QUANTITY. Even a hypothetical toggleable LoRA
 *      leaves its weights RESIDENT. A `none` measurement taken on a generator that was loaded with
 *      an adapter is inflated by that adapter's own bytes. This campaign measures peak memory, so
 *      amortising a load across overlays does not merely assume a missing feature — it corrupts the
 *      baseline. Sharing a load and measuring an unperturbed `none` peak are mutually exclusive.
 *
 * So the honest verdict is UNAVAILABLE IN THE SHIPPED LOAD CONTRACT, not "collapsible but unproven".
 */
export const OVERLAY_LOAD_CONTRACT = {
  verdict: "unavailable in the shipped load contract",
  adaptersAreLoadTime:
    "crates/sceneworks-worker/src/image_jobs/base.rs#resolve_adapters (base.rs:2585) feeds " +
    "`LoadSpec::adapters` through `spec.with_adapters` (base.rs:2664-2666), so a generator's adapter " +
    "set is fixed when it is constructed; crates/sceneworks-worker/src/image_jobs/" +
    "krea_multiphase.rs:74-81 documents per-phase adapter refs as INDICES into that load-time stack, " +
    "and crates/sceneworks-worker/src/generator_cache.rs folds the adapter set into " +
    "`GeneratorCacheKey` so changing it forces a cold reload by design",
  noRuntimeSwapApi:
    "no detach / unload / remove / clear / disable / swap / unfuse API for an adapter on a loaded " +
    "generator exists anywhere in `crates/`, and the `gen_core::Generator` trait exposes no such " +
    "method, so the call is not expressible — the load-axis collapse assumes an operation that is " +
    "not implemented (SceneWorks' only adapter-facing generator call is the read-only " +
    "`adapter_apply_reports()`)",
  adaptersDisableARung:
    "carrying adapters voids the ENTIRE measured Krea ladder, not just the streamed-blocks rung: " +
    "`allow_streamed_blocks = adapter_count == 0` (base.rs:5737) sets `overlay: Some(\"adapter\")` on " +
    "the request scope (vram_gate.rs:609) while every evidence cell carries `overlay: None` " +
    "(vram_gate.rs:829-830), so memory_strategy.rs:194-200 marks every candidate OutOfEnvelope and " +
    "the job drops to the older sequential gate. This is the origin of the 6 `Structurally N/A` " +
    "cells here — all krea_2_turbo/candle x {q4,q8,bf16} x {lora,control} at " +
    "bounded_transformer_residency, which scripts/generate-memory-matrix.mjs cites to " +
    "`base.rs#allow_streamed_blocks`, reason \"load-time adapters are incompatible with streamed " +
    "transformer blocks\"",
  baselinePerturbation:
    "DECISIVE: a toggleable LoRA still leaves its weights resident, so a `none` scenario measured on " +
    "an adapter-loaded generator is inflated by the adapter's own bytes. The collapse perturbs the " +
    "exact quantity this campaign exists to measure, so sharing the load and measuring an " +
    "unperturbed `none` peak cannot both be done",
  perOverlayKind: {
    lora: {
      warmAttachDetachWithoutBaseReload: false,
      reason:
        "LoRAs are fixed in LoadSpec::adapters, changing that set changes GeneratorCacheKey, and " +
        "Generator exposes no detach or swap method",
      noneBaselineComparison: {
        status: "not_applicable",
        toleranceBytes: 0,
        reason:
          "the required warm swap does not exist; even a hypothetical disabled LoRA leaves its " +
          "weights resident and cannot establish a byte-identical no-adapter baseline",
      },
      loadSharingAvailable: false,
    },
    identity: {
      warmAttachDetachWithoutBaseReload: false,
      reason:
        "InstantID and PuLID materialise an identity adapter plus face/identity encoder weights, " +
        "and no loaded-generator API detaches that resident stack",
      noneBaselineComparison: {
        status: "not_applicable",
        toleranceBytes: 0,
        reason:
          "the required warm swap does not exist; the extra resident encoder network makes an " +
          "adapter-loaded process a different baseline population",
      },
      loadSharingAvailable: false,
    },
    control: {
      warmAttachDetachWithoutBaseReload: false,
      reason:
        "control branches are supplied at construction through LoadSpec::with_control (including " +
        "the measured Krea pose path), and no loaded-generator API replaces or clears them",
      noneBaselineComparison: {
        status: "not_applicable",
        toleranceBytes: 0,
        reason:
          "the required warm swap does not exist; a loaded control branch remains a second " +
          "resident network, so it cannot stand in for the base-only process",
      },
      loadSharingAvailable: false,
    },
  },
  counterfactualPriceTagDisposition:
    "SC-16072 needs to know what the capability would be worth before deciding whether to build it. " +
    "The counterfactual count is retained as structured metadata, but the unavailable " +
    "overlay-amortised COLUMN is dropped from campaign and wall-clock tables. It is not pursued as " +
    "an achievable campaign shape.",
};
const FIT_GROUP_FIELDS = ["modelId", "backend", "mode", "overlay"];

function sha256(body) {
  return createHash("sha256").update(body).digest("hex");
}

// Joined on an escaped NUL rather than a raw one: a literal NUL byte in the source makes git
// classify this file as binary, which silently costs every future reader a reviewable diff. The
// separator itself must stay collision-proof because these keys are compared for identity.
function keyOf(cell, fields) {
  return fields.map((field) => cell[field]).join("\u0000");
}

export function batchedSessionKeysFromPlan(plan) {
  const groups = new Map();
  for (const provider of plan.providers.filter((item) => item.modelLoadPolicy === "batch_rungs")) {
    if (!provider.modelLoadGroup) throw new Error("batch_rungs provider requires modelLoadGroup");
    if (!groups.has(provider.modelLoadGroup)) groups.set(provider.modelLoadGroup, []);
    groups.get(provider.modelLoadGroup).push(provider);
  }
  const keys = [];
  for (const [group, providers] of groups) {
    const rungs = providers.map((provider) => provider.rung).sort(
      (left, right) => RUNG_ORDER.indexOf(left) - RUNG_ORDER.indexOf(right),
    );
    if (JSON.stringify(rungs) !== JSON.stringify(RUNG_ORDER)) {
      throw new Error(`${group}: batched cost coverage requires one provider for every canonical rung`);
    }
    const sessions = new Set(providers.map((provider) => keyOf({
      modelId: provider.target.modelId,
      backend: provider.backend,
      tier: provider.target.tier,
      mode: provider.target.mode,
      overlay: provider.target.overlay.replace(/^control:\d+$/, "control"),
    }, SESSION_KEY_FIELDS)));
    if (sessions.size !== 1) throw new Error(`${group}: batched providers must share one session key`);
    keys.push(...sessions);
  }
  return [...new Set(keys)].sort();
}

function tally(values) {
  const counts = new Map();
  for (const value of values) counts.set(value, (counts.get(value) ?? 0) + 1);
  return Object.fromEntries([...counts.entries()].sort(([left], [right]) => left.localeCompare(right)));
}

/**
 * Deterministic rounding. Every derived number in the artifact goes through this so a re-run is
 * byte-identical regardless of how the platform prints the tail of a double.
 */
function round(value, places = 2) {
  if (!Number.isFinite(value)) return null;
  const factor = 10 ** places;
  return Math.round(value * factor + Number.EPSILON * factor) / factor;
}

/**
 * PARAMETERS. Every one of these is a value the model cannot derive from the artifact today.
 * Each carries its own provenance string, and each is swept rather than trusted.
 *
 * `derivedFrom: null` means ASSUMPTION — nothing in the repository establishes it.
 */
export const DEFAULT_PARAMETERS = {
  // Structurally N/A cells need NO measurement: the epic accepts static architecture evidence
  // where the impossibility is structural. SC-15969 (the per-family rung-4 applicability survey)
  // has NOT run, so the final count is unknowable. Expressed as a per-rung fraction and swept.
  structurallyNotApplicableFractionByRung: {
    resident: 0,
    staged_residency: 0,
    bounded_decode: 0,
    bounded_attention: 0,
    bounded_transformer_residency: 0,
  },
  // Seconds per provider invocation. The measured anchors below are a FLOOR that is known not to
  // cover the dominant cost, so this default is an ASSUMPTION and the sweep is the real answer.
  perRunSeconds: 300,
  perRunSecondsSweep: [30, 60, 120, 300, 600, 1200],
  // Geometry points needed to determine one affine phase curve `fixedGb + perMpxGb * megapixels`.
  // Two is the algebraic minimum for a line. See `krea_phase_curve` in vram_gate.rs.
  geometryPointsPerFit: 2,
  // Whether the geometry slope may be shared across tiers so only one tier needs two points.
  //
  // Default false on ABSENCE OF EVIDENCE, not on contradiction. An earlier version of this comment
  // justified `false` by saying the shipped Krea evidence contradicts sharing on the decode phase.
  // That reasoning was CIRCULAR: the q8/bf16 decode slopes it appealed to are exactly the
  // never-fitted zeros `kreaFitPrecedent` flags below (both tiers carry ONE geometry point), so they
  // cannot contradict anything.
  //
  // The informative comparison points the OTHER way. On the threeStage and tiledVae denoise phases
  // the manifest's slopes are 7.98 (q4) / 7.98 (q8) / 7.90 (bf16) — q8 and bf16 have a single
  // geometry point each, so those slopes cannot have been fitted independently and the manifest
  // evidently DID share q4's. The one place tiers visibly diverge is chunkedAttention denoise —
  // 0.59 (q4) / 1.18 (q8) / 0.22 (bf16), a 5.4x spread — but q8 and bf16 rest on one point there
  // too, so that spread is not measured evidence against sharing either.
  //
  // So NOTHING in the shipped evidence measures the same slope at two geometries on two tiers. That
  // is why the default is false: sharing is unestablished, not disproven. `fitSensitivity` below
  // reports what the assumption is worth instead of burying it.
  slopeSharedAcrossTiers: false,
  // Fraction of cells spot-checked against the fitted curve under a fit-then-validate strategy.
  validationSampleFraction: 0.05,
};

/**
 * The three parameters above that the wall-clock sweep does NOT sweep. The file's own policy is that
 * an unknowable value is exposed and swept rather than trusted, and `perRunSeconds` obeys it while
 * these three did not — even though they move the headline fit ratio by ~3x. Swept here so the
 * policy holds for every parameter, not just the one with a sweep already written.
 */
export const FIT_SENSITIVITY_GRID = {
  geometryPointsPerFit: [2, 3, 4],
  slopeSharedAcrossTiers: [false, true],
  validationSampleFraction: [0, 0.05, 0.25],
};

/**
 * THE COLLAPSING RULE.
 *
 * Derived by reading `scripts/memory-calibration-harness.mjs` and
 * `scripts/generate-memory-matrix.mjs`, not by assuming. Four candidate axes were tested; two
 * collapse, two do not. The two negative results are as load-bearing as the positive ones, so
 * they are recorded here rather than omitted.
 */
export const COLLAPSING_AXES = [
  {
    axis: "geometry",
    verdict: "collapsed for the implementation claim; NOT collapsed for the characterization claim",
    factor: "1x on `state`; `geometryPointsPerFit` per curve on `memoryCharacterization`",
    rule:
      "A cell carries a geometry ENVELOPE, not a geometry — and SC-16060 split what that licenses " +
      "into the two claims it was conflating. `state` (does the rung WORK) is not geometry-sensitive, " +
      "so one record binds the envelope and `calibrationBinding` accepts any record whose " +
      "`width x height` appears in `cell.geometryEnvelope.resolutions`. " +
      "`cells[].memoryCharacterization` (are its PEAKS known across the envelope) is geometry-" +
      "sensitive by construction, and reads `point` on one measured geometry, `fitted` only on two " +
      "or more.",
    citation:
      "scripts/generate-memory-matrix.mjs#calibrationBinding (implementation claim) and " +
      "#memoryCharacterization (characterization claim); vocabulary defined in the artifact's `claims`",
    caveat:
      "The manifest's evidenceRecords comment — 'exact request envelopes, not permission to " +
      "interpolate: a geometry absent here is Implemented/unverified even when a phase curve can " +
      "predict it' — is a statement about the CHARACTERIZATION claim, and is now true as written: a " +
      "geometry absent from the evidence is absent from `measuredGeometries` and cannot lift the " +
      "cell past `point`. It was only ever in tension with the binding rule while both claims shared " +
      "one field.",
  },
  {
    axis: "strategyParameters",
    verdict: "collapsed",
    factor: "one record retires every passed sweep point",
    rule:
      "One `complete` record carries a `sweep` whose passed cases each retire a distinct planned " +
      "logical case, so a provider that sweeps its parameter domain internally needs one record " +
      "for the whole domain rather than one per point.",
    citation:
      "scripts/memory-calibration-harness.mjs#completedLogicalIds (maps every passed sweep case " +
      "to its own logicalCaseId) and #validateComplete (every declared axis range must be derived " +
      "from passed executed cases)",
    caveat:
      "`runProviderPlan` recomputes completed logical identities after every provider response. A " +
      "complete record whose passed sweep covers other pending points retires those points before " +
      "the runner chooses its next invocation, so the collapse is now realised both within one run " +
      "and across `--resume` runs.",
  },
  {
    axis: "rung",
    verdict: "fresh per rung on both measured backends",
    factor: "1 record per model load; the 5-record warm floor remains counterfactual",
    rule:
      "A record names ONE rung (`strategy.rung` is a single enum and is part of the record " +
      "identity), and the matrix binds a record to exactly one cell on the " +
      "(modelId, provider, backend, tier, mode, overlay, rung) key, so records remain one-per-rung. " +
      "The harness and Candle adapter can execute Krea's five-rung group as one `run_batch` provider " +
      "invocation with `modelLoads: 1`, but the authoritative fresh/reused comparison exceeded the " +
      "committed tolerance, so the shipped plan keeps Candle fresh-per-rung. MLX also stays fresh: its Z-Image " +
      "ladder spans distinct eager and deferred calibration fingerprints, so one loaded generator " +
      "cannot preserve every rung's calibrated identity. No target is priced as batched.",
    citation:
      "scripts/memory-calibration-harness.mjs#runProviderPlan (canonical `run_batch`, one-load " +
      "attestation, and within-invocation sweep retirement); " +
      "crates/sceneworks-memory-adapter/src/bin/candle.rs#run_five_rung_batch; " +
      "crates/sceneworks-memory-adapter/src/bin/mlx.rs#assess_z_image_batch; " +
      ".github/workflows/windows-candle.yml (fresh/reused evidence gate) and macos-mlx.yml " +
      "(fresh capture plus provider-contract inability gate)",
    caveat:
      "The committed equivalence gate is the larger of 256 MiB absolute drift and 5% of the fresh " +
      "metric, applied to every phase and allocator/device/wired/reclaimable metric. Candle's measured " +
      "verdict and MLX's structural verdict are both `unable_to_amortize`. MLX's inability is structural, not " +
      "an inferred timing result: the eager and deferred fingerprints differ before a reused process " +
      "can truthfully execute all five identities, so the backend retains fresh-per-rung cost.",
  },
  {
    axis: "overlay",
    verdict: "NOT collapsed for records; NOT collapsible for model loads in the shipped contract",
    factor: "1 today (the current amortised load ratio prices a capability that does not exist)",
    rule:
      "A material share of the current matrix carries a non-`none` overlay. It does NOT collapse " +
      "on the record axis: `overlay` is part of the cell key and of " +
      "`record.target`, and `validateComplete` requires the `overlay` scenario to actually pass " +
      "(or carry a justified `not_applicable` reason) — a `none` record cannot certify a `lora` " +
      "cell. Nor does it collapse on the model-load axis: adapters are resolved at LOAD time into " +
      "`LoadSpec::adapters`, no API anywhere in `crates/` detaches or swaps an adapter on a loaded " +
      "generator, and carrying adapters makes `vram_gate.rs` disable streamed blocks outright. " +
      "Decisively, even a toggleable LoRA stays RESIDENT, so a `none` peak measured on an " +
      "adapter-loaded generator is inflated by the adapter's own bytes — the collapse perturbs the " +
      "very quantity being measured.",
    citation:
      "docs/generated/memory-matrix.json (overlay distribution); " +
      "scripts/memory-calibration-harness.mjs#validateRecord (target.overlay is required) and " +
      "#validateComplete (the overlay scenario must pass or be justified); " +
      "crates/sceneworks-worker/src/image_jobs/base.rs#resolve_adapters -> `LoadSpec::adapters` and " +
      "crates/sceneworks-worker/src/image_jobs/krea_multiphase.rs (adapters are load-time); " +
      "crates/sceneworks-worker/src/vram_gate.rs (adapters disable streamed blocks); " +
      "absence of any detach/unload/swap adapter API across `crates/`",
    caveat:
      "The overlay-amortised load count is still REPORTED, because SC-16072 has to know what the " +
      "capability would be worth before deciding to build it. It is a price tag, not a campaign " +
      "shape: it assumes a runtime attach/detach operation that does not exist, and if it were " +
      "built naively it would produce inflated `none` baselines rather than a cheaper campaign. An " +
      "`identity` overlay (InstantID / PuLID) additionally materialises extra encoder weights, so " +
      "even a real swap API would not make this axis uniformly free. Those overlay records do not " +
      "evaporate under any of it.",
  },
  {
    axis: "cell",
    verdict: "NOT collapsed",
    factor: "1 (cells map 1:1 to certifying records)",
    rule:
      "No record can certify two cells. The matrix matches a record to a cell on the full " +
      "(modelId, provider, backend, tier, mode, overlay, rung) key, and a `complete` record must " +
      "additionally carry its own measured negative mutation — so the negative case is INSIDE the " +
      "positive record rather than being a second record.",
    citation:
      "scripts/generate-memory-matrix.mjs (record->cell match on the full key); " +
      "scripts/memory-calibration-harness.mjs#validateComplete (negativeMutation must be measured " +
      "and fail as expected inside the complete record)",
    caveat:
      "This is the negative result. Tier, mode and overlay are all part of the cell key and none " +
      "of them collapses: a q4 record cannot certify a q8 cell, and a text_to_image record cannot " +
      "certify an edit_image cell.",
  },
];

function collapsingAxesFor(census, runs) {
  const overlayCells = census.total - (census.byOverlay.none ?? 0);
  const overlayShare = round((overlayCells / census.total) * 100, 1);
  return COLLAPSING_AXES.map((axis) => {
    if (axis.axis === "rung") {
      return {
        ...axis,
        factor:
          `${runs.shippedBatchCoverage.records} records in ${runs.shippedBatchCoverage.sessions} ` +
          `explicit group(s) save ${runs.shippedBatchCoverage.savedLoads} loads; combined ` +
          `${runs.certifyingRecords.total} -> ${runs.shippedModelLoads.total} ` +
          `(${runs.shippedLoadCollapseFactor}x)`,
      };
    }
    if (axis.axis !== "overlay") return axis;
    return {
      ...axis,
      factor:
        `1 today (the current ${runs.unavailableOverlayLoadPriceTag.collapseFactor}x amortised load ratio prices ` +
        "a capability that does not exist)",
      rule:
        `${overlayCells} of the ${census.total} cells carry a non-\`none\` overlay, so this axis ` +
        `is ${overlayShare}% of the matrix. ` +
        axis.rule.replace(
          "A material share of the current matrix carries a non-`none` overlay. ",
          "",
        ),
      caveat: axis.caveat.replace(
        "Those overlay records do not evaporate",
        `The ${overlayCells} overlay records do not evaporate`,
      ),
    };
  });
}

/**
 * PRODUCIBILITY. A cell nobody can currently measure is a different cost category from a cell that
 * is merely unmeasured, and conflating them is what makes a run count look like a GPU-hours problem
 * when it is a code problem.
 *
 * The gates are ORDERED and a cell is charged to the FIRST one that blocks it, so the buckets
 * partition the matrix exactly. Every gate is derived from a file that was read, not assumed.
 */
export const PRODUCIBILITY_GATES = [
  {
    id: "no-run-needed",
    label: "Structurally N/A — static architecture evidence is sufficient",
    blockedBy: "nothing; the epic exempts these",
    citation: "docs/generated/memory-matrix.json (cell.state) and the epic's conformance states",
  },
  {
    id: "overlay-declined",
    label: "Non-`none` overlay without a complete truthful overlay record",
    blockedBy: "provider-adapter code",
    citation:
      "docs/generated/memory-calibration-evidence.json (complete records whose exact overlay " +
      "scenario passed, normalized from control:N to the matrix control axis)",
  },
  {
    id: "no-provider-adapter",
    label: "No provider adapter covers this (entry, backend) at all",
    blockedBy: "provider-adapter code",
    citation: "config/memory-calibration-plan.json (the shipped plan's provider targets)",
  },
  {
    id: "adapter-gated",
    label: "An adapter exists, but it has never emitted an activation-eligible record",
    blockedBy: "provider-adapter code (phase telemetry and lifecycle injection)",
    citation:
      "docs/generated/memory-calibration-evidence.json (records with status `complete` or " +
      "base-only `runtime_complete`, per (entry, backend)); a pair remains gated only while that " +
      "set has no activation-eligible record",
  },
  {
    id: "producible-today",
    label: "A run executed right now could certify this cell",
    blockedBy: "nothing — GPU hours only",
    citation: "the remainder after the gates above",
  },
];

/**
 * The gates that represent MISSING CODE, as INDEPENDENT predicates. `no-run-needed` (exempt) and
 * `producible-today` (nothing wrong) are not blockers and are absent.
 *
 * These exist separately from the ordered partition because the ordered partition answers "which
 * gate is charged for this cell" and that is NOT the same question as "what is wrong with this
 * cell". A cell can be blocked by several of these at once, and the first-match partition shows only
 * the first. Ranking remediation off the partition therefore over-credits whichever gate happens to
 * be charged first — which is exactly the error this block exists to prevent.
 */
const CODE_BLOCKERS = [
  {
    id: "overlay-declined",
    blocks: (cell, ctx) =>
      cell.overlay !== "none" && !ctx.completeOverlays.has(ctx.overlayKey(cell)),
  },
  { id: "no-provider-adapter", blocks: (cell, ctx) => !ctx.adapters.has(ctx.pair(cell)) },
  {
    id: "adapter-gated",
    blocks: (cell, ctx) => ctx.adapters.has(ctx.pair(cell)) && !ctx.complete.has(ctx.pair(cell)),
  },
];

export function producibility(
  cells,
  { adapterPairs, pairsWithCompleteRecords, overlayKeysWithCompleteRecords = [] },
) {
  const adapters = new Set(adapterPairs);
  const complete = new Set(pairsWithCompleteRecords);
  const completeOverlays = new Set(overlayKeysWithCompleteRecords);
  const pair = (cell) => `${cell.modelId}|${cell.backend}`;
  const overlayKey = (cell) => `${pair(cell)}|${cell.overlay}`;
  const ctx = { adapters, complete, completeOverlays, pair, overlayKey };
  const buckets = new Map(PRODUCIBILITY_GATES.map((gate) => [gate.id, []]));

  // Every blocker on every cell, order-independent. This is the multiset the ordered partition
  // flattens away.
  const blockersOf = (cell) =>
    cell.state === "Structurally N/A"
      ? []
      : CODE_BLOCKERS.filter((blocker) => blocker.blocks(cell, ctx)).map((blocker) => blocker.id);

  const blockerSets = cells.map((cell) => ({ cell, blockers: blockersOf(cell) }));

  for (const cell of cells) {
    const gate =
      cell.state === "Structurally N/A"
        ? "no-run-needed"
        : (CODE_BLOCKERS.find((blocker) => blocker.blocks(cell, ctx))?.id ?? "producible-today");
    buckets.get(gate).push(cell);
  }

  const partition = PRODUCIBILITY_GATES.map((gate) => {
    const bucket = buckets.get(gate.id);
    return {
      ...gate,
      cells: bucket.length,
      mlx: bucket.filter((cell) => cell.backend === "mlx").length,
      candle: bucket.filter((cell) => cell.backend === "candle").length,
    };
  });

  // THE MULTISET. Deterministically ordered: most cells first, then by key, so a re-run is
  // byte-identical.
  const multisetCounts = new Map();
  for (const { cell, blockers } of blockerSets) {
    const key =
      cell.state === "Structurally N/A"
        ? "(exempt — no run needed)"
        : blockers.length === 0
          ? "(none — producible today)"
          : [...blockers].sort().join(" + ");
    if (!multisetCounts.has(key)) multisetCounts.set(key, { blockers: [...blockers].sort(), cells: [] });
    multisetCounts.get(key).cells.push(cell);
  }
  const blockersPerCell = [...multisetCounts.entries()]
    .map(([key, entry]) => ({
      blockers: entry.blockers,
      description: key,
      cells: entry.cells.length,
      mlx: entry.cells.filter((cell) => cell.backend === "mlx").length,
      candle: entry.cells.filter((cell) => cell.backend === "candle").length,
    }))
    .sort((left, right) => right.cells - left.cells || left.description.localeCompare(right.description));

  // INDEPENDENT BLOCKING. `soleBlocker` is the number of cells this gate is the ONLY thing wrong
  // with — i.e. the cells that fixing it alone makes producible. That, not the first-match bucket
  // size, is what a remediation ranking must use.
  const independentBlocking = CODE_BLOCKERS.map((blocker) => {
    const touched = blockerSets.filter((entry) => entry.blockers.includes(blocker.id));
    const sole = touched.filter((entry) => entry.blockers.length === 1);
    return {
      id: blocker.id,
      cellsTouched: touched.length,
      soleBlockerCells: sole.length,
      coBlockedCells: touched.length - sole.length,
      coBlockedShare: touched.length === 0 ? null : round((touched.length - sole.length) / touched.length, 4),
      coBlockedWith: tally(
        touched
          .filter((entry) => entry.blockers.length > 1)
          .map((entry) => entry.blockers.filter((id) => id !== blocker.id).sort().join(" + ")),
      ),
    };
  }).sort((left, right) => right.soleBlockerCells - left.soleBlockerCells || left.id.localeCompare(right.id));

  // GATE-ORDER SENSITIVITY, quantified rather than merely disclosed. The shipped order charges
  // overlay before adapter coverage; this is what the same matrix looks like the other way round.
  const firstMatchUnder = (order) => {
    const counts = new Map();
    for (const cell of cells) {
      const gate =
        cell.state === "Structurally N/A"
          ? "no-run-needed"
          : (order.find((blocker) => blocker.blocks(cell, ctx))?.id ?? "producible-today");
      counts.set(gate, (counts.get(gate) ?? 0) + 1);
    }
    return Object.fromEntries([...counts.entries()].sort(([left], [right]) => left.localeCompare(right)));
  };
  const byId = Object.fromEntries(CODE_BLOCKERS.map((blocker) => [blocker.id, blocker]));
  const adapterCoverageFirst = firstMatchUnder([
    byId["no-provider-adapter"],
    byId["overlay-declined"],
    byId["adapter-gated"],
  ]);

  const total = cells.length;
  const producible = buckets.get("producible-today").length;
  const noRunNeeded = buckets.get("no-run-needed").length;
  const behindCode = total - producible - noRunNeeded;
  const overlay = independentBlocking.find((entry) => entry.id === "overlay-declined");
  const adapterCoverage = independentBlocking.find((entry) => entry.id === "no-provider-adapter");
  const overlayCharged = partition.find((gate) => gate.id === "overlay-declined")?.cells ?? 0;

  return {
    partition,
    blockersPerCell,
    independentBlocking,
    gateOrderSensitivity: {
      shippedOrder: firstMatchUnder(CODE_BLOCKERS),
      adapterCoverageFirst,
      note:
        "The shipped order charges `overlay-declined` before `no-provider-adapter`. Both orders " +
        "partition the matrix exactly; they differ only in which gate is credited for the cells " +
        "that both block. The reordered column is the honest denominator for 'how much does overlay " +
        "support move'.",
    },
    // The ranking correction. Reading remediation priority off the first-match partition credits the
    // overlay gate with every cell it touches, including the ones that would not budge if overlay
    // support shipped tomorrow.
    blockerRanking: {
      rankedBy: "soleBlockerCells — cells this gate is the ONLY thing wrong with",
      ranking: independentBlocking.map((entry) => entry.id),
      overlayChargedByFirstMatch: overlayCharged,
      overlayAlsoBlockedByAdapterCoverage: overlay?.coBlockedCells ?? 0,
      overlayIndependentlyBlocking: overlay?.soleBlockerCells ?? 0,
      overlayBindingAfterAdapterCoverage: adapterCoverageFirst["overlay-declined"] ?? 0,
      finding:
        `\`overlay-declined\` is charged ${overlayCharged} cells by the first-match partition, but it ` +
        `is the SOLE blocker on ${overlay?.soleBlockerCells ?? 0} of them. ` +
        `${overlay?.coBlockedWith?.["no-provider-adapter"] ?? 0} of them ` +
        `(${round(((overlay?.coBlockedWith?.["no-provider-adapter"] ?? 0) / Math.max(overlayCharged, 1)) * 100, 1)}%) ` +
        "are also blocked by having NO PROVIDER ADAPTER AT ALL, and the remaining " +
        `${overlay?.coBlockedWith?.["adapter-gated"] ?? 0} by an adapter that has never emitted a ` +
        "`complete` record. Charge adapter coverage first and the overlay bucket falls to " +
        `${adapterCoverageFirst["overlay-declined"] ?? 0} while \`no-provider-adapter\` rises to ` +
        `${adapterCoverageFirst["no-provider-adapter"] ?? 0}. ` +
        `PROVIDER-ADAPTER COVERAGE, not overlay support, is the largest independent blocker: it is ` +
        `the only thing wrong with ${adapterCoverage?.soleBlockerCells ?? 0} cells and it touches ` +
        `${adapterCoverage?.cellsTouched ?? 0}. Closing the remaining overlay-support gaps is still ` +
        `necessary, but on its own it moves ${adapterCoverageFirst["overlay-declined"] ?? 0} cells, ` +
        `not ${overlayCharged}.`,
    },
    // The number that reframes the whole model: how much of the campaign is gated on code that does
    // not exist, rather than on machine time that could start today.
    behindProviderCode: behindCode,
    behindProviderCodeShare: round(behindCode / total, 4),
    producibleToday: producible,
    headline:
      producible === 0
        ? "ZERO cells are producible today. The binding constraint on this epic is provider-adapter " +
          `code, not GPU hours: ${complete.size} provider pair(s) have emitted a ` +
          `\`complete\` record, while ${adapters.size} provider pair(s) have an adapter at all. ` +
          "Every hour in the wall-clock sweep below is hypothetical until the remaining code gates " +
          "change."
        : `${producible} of ${total} cells could be certified by a run started right now.`,
  };
}

/** Layer 1 — CELLS. Pure counting off the artifact. No interpretation. */
export function cellCensus(cells) {
  const sessions = new Map();
  const fitGroups = new Map();
  let geometryExpandedCells = 0;

  for (const cell of cells) {
    const resolutions = cell.geometryEnvelope?.resolutions ?? [];
    const points = Math.max(resolutions.length, 1);
    geometryExpandedCells += points;

    const sessionKey = keyOf(cell, SESSION_KEY_FIELDS);
    if (!sessions.has(sessionKey)) {
      sessions.set(sessionKey, { backend: cell.backend, geometryPoints: points, rungs: new Set() });
    }
    sessions.get(sessionKey).rungs.add(cell.rung);

    const fitKey = keyOf(cell, FIT_GROUP_FIELDS);
    if (!fitGroups.has(fitKey)) fitGroups.set(fitKey, { backend: cell.backend, tiers: new Set() });
    fitGroups.get(fitKey).tiers.add(cell.tier);
  }

  const sessionList = [...sessions.values()];
  const rungSpans = new Set(sessionList.map((session) => session.rungs.size));

  return {
    total: cells.length,
    byState: tally(cells.map((cell) => cell.state)),
    byBackend: tally(cells.map((cell) => cell.backend)),
    byStateAndBackend: tally(cells.map((cell) => `${cell.state} @ ${cell.backend}`)),
    byRung: tally(cells.map((cell) => cell.rung)),
    byTier: tally(cells.map((cell) => cell.tier)),
    byMode: tally(cells.map((cell) => cell.mode)),
    byOverlay: tally(cells.map((cell) => cell.overlay)),
    // Geometry is the collapse the matrix ALREADY made. Keyed per resolution the matrix would be
    // this large; the ratio is how much collapsing is baked into the published cell count.
    geometryExpandedCells,
    geometryCollapseFactor: round(geometryExpandedCells / cells.length, 3),
    sessions: {
      total: sessions.size,
      byBackend: tally(sessionList.map((session) => session.backend)),
      geometryExpanded: sessionList.reduce((sum, session) => sum + session.geometryPoints, 0),
      geometryExpandedByBackend: Object.fromEntries(
        ["mlx", "candle"].map((backend) => [
          backend,
          sessionList
            .filter((session) => session.backend === backend)
            .reduce((sum, session) => sum + session.geometryPoints, 0),
        ]),
      ),
      // If this is not exactly [5] the rung collapse factor below is not a constant and the
      // model must not report one.
      rungsPerSession: [...rungSpans].sort((left, right) => left - right),
    },
    fitGroups: {
      total: fitGroups.size,
      byBackend: tally([...fitGroups.values()].map((group) => group.backend)),
      tierSum: [...fitGroups.values()].reduce((sum, group) => sum + group.tiers.size, 0),
      tierSumByBackend: Object.fromEntries(
        ["mlx", "candle"].map((backend) => [
          backend,
          [...fitGroups.values()]
            .filter((group) => group.backend === backend)
            .reduce((sum, group) => sum + group.tiers.size, 0),
        ]),
      ),
    },
  };
}

/**
 * Layer 2 — RUNS. Applies the collapsing rule and the N/A exemption to produce run counts under
 * each realisable strategy, split by hardware because MLX/Metal runs on a Mac and Candle/CUDA
 * needs rented CUDA — completely different cost structures.
 */
export function collapseToRuns(cells, parameters = DEFAULT_PARAMETERS, coverage = {}) {
  const naFractions = {
    ...DEFAULT_PARAMETERS.structurallyNotApplicableFractionByRung,
    ...(parameters.structurallyNotApplicableFractionByRung ?? {}),
  };

  // Already-classified N/A cells are exempt as a FACT. Additional exemptions are a PROJECTION.
  // The projection is applied to NAMED cells, not to a count, so the session arithmetic below
  // stays exact instead of becoming a division.
  //
  // Selection is the first N cells of the rung by ascending cell id. That is deterministic but
  // ARBITRARY, and it is arbitrary in a way that shows up in the output: cell ids lead with the
  // model id, so an alphabetical prefix is not evenly distributed across backends and the per-backend
  // split of a PROJECTION row is partly an artifact of this rule rather than a property of the
  // projection. An earlier version of this comment claimed the output "says out loud" that the
  // selection is arbitrary; it did not — neither generated artifact contained the word. It does now,
  // via `naSelectionRule` below, which is emitted into the artifact and rendered next to the table.
  const measurable = cells.filter((cell) => cell.state !== "Structurally N/A");
  const alreadyExempt = cells.length - measurable.length;

  const projectedExemptByRung = {};
  const projected = new Set();
  for (const rung of RUNG_ORDER) {
    const inRung = measurable
      .filter((cell) => cell.rung === rung)
      .sort((left, right) => left.id.localeCompare(right.id));
    const fraction = Math.min(Math.max(naFractions[rung] ?? 0, 0), 1);
    const exempt = Math.round(inRung.length * fraction);
    projectedExemptByRung[rung] = exempt;
    for (const cell of inRung.slice(0, exempt)) projected.add(cell.id);
  }

  const surviving = measurable.filter((cell) => !projected.has(cell.id));
  const certifyingRecords = {
    total: surviving.length,
    mlx: surviving.filter((cell) => cell.backend === "mlx").length,
    candle: surviving.filter((cell) => cell.backend === "candle").length,
  };

  // A warm session is one MODEL LOAD. It is NOT the record count divided by the rung span: a
  // session survives as long as ANY of its rungs still needs measuring, so exempting rungs
  // removes records without removing loads. Counting the surviving session keys directly is what
  // makes that visible; dividing would have hidden it.
  const sessionsKeyedBy = (fields) => {
    const keys = new Map();
    for (const cell of surviving) {
      const key = keyOf(cell, fields);
      if (!keys.has(key)) keys.set(key, cell.backend);
    }
    const backends = [...keys.values()];
    return {
      total: keys.size,
      mlx: backends.filter((backend) => backend === "mlx").length,
      candle: backends.filter((backend) => backend === "candle").length,
    };
  };
  const warmSessions = sessionsKeyedBy(SESSION_KEY_FIELDS);
  const overlayAmortisedSessions = sessionsKeyedBy(OVERLAY_AMORTISED_SESSION_KEY_FIELDS);

  const census = cellCensus(cells);
  const rungCollapse =
    census.sessions.rungsPerSession.length === 1 ? census.sessions.rungsPerSession[0] : null;
  const eligibleBatchKeys = new Set(coverage.batchedSessionKeys ?? []);
  const batchCounts = new Map();
  for (const cell of surviving) {
    const key = keyOf(cell, SESSION_KEY_FIELDS);
    if (!eligibleBatchKeys.has(key)) continue;
    const current = batchCounts.get(key) ?? { backend: cell.backend, records: 0 };
    current.records += 1;
    batchCounts.set(key, current);
  }
  const savedLoads = { mlx: 0, candle: 0 };
  for (const session of batchCounts.values()) {
    savedLoads[session.backend] += Math.max(0, session.records - 1);
  }
  const shippedModelLoads = {
    mlx: certifyingRecords.mlx - savedLoads.mlx,
    candle: certifyingRecords.candle - savedLoads.candle,
    total: certifyingRecords.total - savedLoads.mlx - savedLoads.candle,
  };

  return {
    exemptions: {
      alreadyStructurallyNotApplicable: alreadyExempt,
      projectedAdditional: projected.size,
      projectedAdditionalByRung: projectedExemptByRung,
    },
    // The unit the harness actually spawns: one `action:"run"` provider invocation.
    certifyingRecords,
    // The unit that costs a model load, IF a provider amortises one across a target's rungs.
    warmSessions,
    shippedModelLoads,
    shippedBatchCoverage: {
      sessionKeys: [...batchCounts.keys()].sort(),
      sessions: batchCounts.size,
      records: [...batchCounts.values()].reduce((sum, session) => sum + session.records, 0),
      savedLoads: savedLoads.mlx + savedLoads.candle,
    },
    shippedLoadCollapseFactor:
      shippedModelLoads.total === 0
        ? null
        : round(certifyingRecords.total / shippedModelLoads.total, 3),
    unavailableOverlayLoadPriceTag: {
      availability: "unavailable",
      disposition: "counterfactual metadata only; excluded from campaign and wall-clock columns",
      sessions: overlayAmortisedSessions,
      collapseFactor:
        overlayAmortisedSessions.total === 0
          ? null
          : round(warmSessions.total / overlayAmortisedSessions.total, 2),
    },
    // Three places deliberately: the exempt cells make this NOT exactly the rung span, and a
    // two-place round would print a clean "5" that reads as an identity rather than a ratio.
    recordsPerWarmSession:
      warmSessions.total === 0 ? null : round(certifyingRecords.total / warmSessions.total, 3),
    rungCollapseFactor: rungCollapse,
  };
}

/**
 * The projection's cell-selection rule, stated in the artifact rather than only in this source.
 * Minor but load-bearing: without it a reader takes a projection row's per-backend split for a
 * property of the projection, when it is partly a property of how cells were picked.
 */
export const NA_SELECTION_RULE =
  "Projected exemptions are applied to NAMED cells, not to a count, so the session arithmetic stays " +
  "exact rather than becoming a division. Within each rung the exempted cells are the first N by " +
  "ASCENDING CELL ID. That is deterministic but ARBITRARY: cell ids lead with the model id, so an " +
  "alphabetical prefix does not split evenly across backends. Consequently the MLX/Candle columns of " +
  "a PROJECTION row are partly an artifact of this rule. Only the `current` row's split is a fact. " +
  "Each projection row therefore also reports what its split would be under proportional allocation, " +
  "so the size of the artifact is visible instead of implied.";

/**
 * What a projection's exemptions would look like allocated in proportion to each backend's share of
 * the rung, rather than by sorted cell id. The gap between this and the actual split is exactly the
 * size of the selection artifact.
 */
function proportionalExemptions(cells, fractions) {
  const measurable = cells.filter((cell) => cell.state !== "Structurally N/A");
  const exempt = { mlx: 0, candle: 0 };
  for (const rung of RUNG_ORDER) {
    const inRung = measurable.filter((cell) => cell.rung === rung);
    const fraction = Math.min(Math.max(fractions[rung] ?? 0, 0), 1);
    if (fraction === 0) continue;
    for (const backend of ["mlx", "candle"]) {
      exempt[backend] += inRung.filter((cell) => cell.backend === backend).length * fraction;
    }
  }
  return { mlx: Math.round(exempt.mlx), candle: Math.round(exempt.candle) };
}

/**
 * N/A sensitivity. SC-15969 has not run, so the honest output is a curve, not a number.
 * The scenarios below are labelled projections; only `current` is a fact.
 */
export function naSensitivity(cells, coverage = {}) {
  const scenarios = [
    {
      name: "current",
      kind: "fact",
      note: "What the generated matrix classifies today.",
      fractions: {},
    },
    {
      name: "rung4-half",
      kind: "projection",
      note:
        "Half of all bounded_transformer_residency cells turn out structurally inapplicable. " +
        "SC-15969 names the two breakers: heterogeneous blocks (the floor is the largest single " +
        "block, so the saving collapses) and skip connections (encoder activations must persist).",
      fractions: { bounded_transformer_residency: 0.5 },
    },
    {
      name: "rung4-all",
      kind: "projection",
      note:
        "Upper bound on the rung-4 axis alone: every rung-4 cell is exempt. This is the most " +
        "optimistic single-rung outcome and it still leaves four rungs to measure.",
      fractions: { bounded_transformer_residency: 1 },
    },
    {
      name: "rungs-2-3-4-half",
      kind: "projection",
      note:
        "Half of bounded_decode, bounded_attention and bounded_transformer_residency exempt — " +
        "i.e. a large fraction of the catalog turns out to have a non-tileable decoder AND " +
        "non-chunkable attention. Aggressive; included to show the shape of the curve, not " +
        "because anything supports it.",
      fractions: { bounded_decode: 0.5, bounded_attention: 0.5, bounded_transformer_residency: 0.5 },
    },
    {
      name: "everything-but-baseline",
      kind: "bound",
      note:
        "Absolute floor. Every constrained rung on every entry is exempt; only the resident " +
        "baseline is measured. Rung 0 can never be Structurally N/A — it is the unoptimised " +
        "baseline every model has by definition — so this is the smallest campaign that can exist.",
      fractions: {
        staged_residency: 1,
        bounded_decode: 1,
        bounded_attention: 1,
        bounded_transformer_residency: 1,
      },
    },
  ];

  const baseline = collapseToRuns(cells, DEFAULT_PARAMETERS, coverage).certifyingRecords;

  return scenarios.map((scenario) => {
    const runs = collapseToRuns(cells, {
      ...DEFAULT_PARAMETERS,
      structurallyNotApplicableFractionByRung: {
        ...DEFAULT_PARAMETERS.structurallyNotApplicableFractionByRung,
        ...scenario.fractions,
      },
    }, coverage);
    const exemptedByBackend = {
      mlx: baseline.mlx - runs.certifyingRecords.mlx,
      candle: baseline.candle - runs.certifyingRecords.candle,
    };
    const proportional = proportionalExemptions(cells, scenario.fractions);
    return {
      name: scenario.name,
      kind: scenario.kind,
      note: scenario.note,
      certifyingRecords: runs.certifyingRecords,
      shippedModelLoads: runs.shippedModelLoads,
      warmSessions: runs.warmSessions,
      // Emitted so a projection's per-backend column is never mistaken for a finding. See
      // NA_SELECTION_RULE.
      exemptedByBackend,
      exemptedByBackendIfProportional: proportional,
      backendSplitIsSelectionArtifact:
        scenario.kind !== "fact" &&
        (exemptedByBackend.mlx !== proportional.mlx || exemptedByBackend.candle !== proportional.candle),
    };
  });
}

/**
 * Wall clock. Per-run seconds is an INPUT. The measured anchors are recorded with exactly what
 * they do and do not cover, and the default is labelled an assumption.
 */
export function wallClock(runs, parameters = DEFAULT_PARAMETERS) {
  const sweep = [...new Set([...(parameters.perRunSecondsSweep ?? []), parameters.perRunSeconds])]
    .filter((value) => Number.isFinite(value) && value > 0)
    .sort((left, right) => left - right);

  const hours = (count, seconds) =>
    count === null ? null : round((count * seconds) / 3600, 1);

  return {
    measuredAnchors: [
      {
        what:
          "8 real MLX provider invocations plus one hardware probe and a schema check, on an " +
          "Apple M5 Max self-hosted runner.",
        measurement: "30.76 s wall clock for the whole harness step",
        derivation:
          "GitHub Actions run 30380114921 job 90347750988: cargo 'Finished `release` profile' at " +
          "17:14:45.405Z, next step group (artifact upload) at 17:15:16.164Z. The 8 records are " +
          "the ones the epic ledger describes (7 passing tile points + 1 negative mutation).",
        perInvocationSeconds: round(30.76 / 8, 2),
        doesNotCover:
          "The Qwen MLX adapter exercises the VAE decode seam ONLY. Its records are `gated`, not " +
          "`complete`: no text-encoder or DiT load, no conditioning, no denoise steps, no " +
          "cancel/error injection at every phase, no warm A->B->A repeat. A `complete` record " +
          `needs all ${REQUIRED_SCENARIOS.length} required scenarios plus quality and a measured ` +
          "negative mutation. This is a FLOOR that is known not to cover the dominant cost.",
      },
      {
        what: "Warm-page-cache sequential read of two real safetensors shards on this Mac.",
        measurement: "11.3 GB/s and 12.3 GB/s (5.03 GB and 3.99 GB reads via dd)",
        derivation:
          "dd bs=4m over Z-Image-Turbo transformer shard 1 of 3 and text_encoder shard 2 of 3 in " +
          "the local Hugging Face cache.",
        perInvocationSeconds: null,
        doesNotCover:
          "Page cache was warm and could not be dropped without elevated privileges, so this is a " +
          "best case, not cold NVMe. It bounds ONLY the file-I/O component of a model load: at " +
          "this rate a 20 GiB tier streams in under 2 s, which is the useful conclusion — I/O is " +
          "not the bottleneck. Dequantisation, graph construction and denoise are unmeasured.",
      },
    ],
    assumption: {
      parameter: "perRunSeconds",
      value: parameters.perRunSeconds,
      status: "ASSUMPTION — nothing in this repository establishes it",
      reasoning:
        "A `complete` record requires at least the 8 required scenarios, a parameter sweep, an " +
        "identical-latent quality comparison and a measured negative mutation, each involving at " +
        "least one render, on top of one model load. The default sits mid-sweep so no reader " +
        "mistakes it for a measurement. The sweep, not the default, is the answer.",
    },
    sweep: sweep.map((seconds) => ({
      perRunSeconds: seconds,
      recordHours: {
        total: hours(runs.certifyingRecords.total, seconds),
        mlx: hours(runs.certifyingRecords.mlx, seconds),
        candle: hours(runs.certifyingRecords.candle, seconds),
      },
      shippedModelLoadHours: {
        total: hours(runs.shippedModelLoads.total, seconds),
        mlx: hours(runs.shippedModelLoads.mlx, seconds),
        candle: hours(runs.shippedModelLoads.candle, seconds),
      },
      warmSessionHours: {
        total: hours(runs.warmSessions.total, seconds),
        mlx: hours(runs.warmSessions.mlx, seconds),
        candle: hours(runs.warmSessions.candle, seconds),
      },
    })),
  };
}

/**
 * Fit-versus-exhaustive. The epic splits ownership: the provider owns the FORMULA KIND, the
 * manifest owns the MEASURED COEFFICIENTS. The shipped phase curve is affine —
 * `fixedGb + perMpxGb * megapixels` — so two coefficients per (tier, rung, phase), and two
 * geometry points determine a line.
 *
 * `kreaPrecedent` is parsed from the manifest rather than quoted, so it goes stale loudly.
 */
/**
 * Minor-but-real gap this closes: `geometryPointsPerFit`, `slopeSharedAcrossTiers` and
 * `validationSampleFraction` were the only parameters in this file NOT swept, contrary to the file's
 * own stated policy — and they move the headline exact-geometry/fit ratio by about 3x. `perRunSeconds`
 * already had a sweep; these now have one too.
 *
 * The INVERSION IS ROBUST: fit/per-cell stays above 1 at every grid point, so "fitting costs more
 * than the per-cell campaign" does not depend on the defaults. Only the magnitude moves.
 */
export function fitSensitivity(cells, grid = FIT_SENSITIVITY_GRID, coverage = {}) {
  const precedentStub = { measurementRecords: 0, tiers: [], curves: 0, coefficients: 0 };
  const rows = [];
  for (const points of grid.geometryPointsPerFit) {
    for (const shared of grid.slopeSharedAcrossTiers) {
      for (const fraction of grid.validationSampleFraction) {
        const result = fitVersusExhaustive(cells, precedentStub, {
          ...DEFAULT_PARAMETERS,
          geometryPointsPerFit: points,
          slopeSharedAcrossTiers: shared,
          validationSampleFraction: fraction,
        }, coverage);
        const fit = result.strategies.find((strategy) => strategy.name === "fit-then-validate");
        rows.push({
          geometryPointsPerFit: points,
          slopeSharedAcrossTiers: shared,
          validationSampleFraction: fraction,
          fitSessions: fit.fitSessions,
          validationSessions: fit.validationSessions,
          modelLoads: fit.sessions,
          exactGeometryOverFit: result.ratios.exactGeometryOverFit,
          fitOverPerCell: result.ratios.fitOverPerCell,
        });
      }
    }
  }
  const exactOverFit = rows.map((row) => row.exactGeometryOverFit);
  const fitOverPerCell = rows.map((row) => row.fitOverPerCell);
  const isDefault = (row) =>
    row.geometryPointsPerFit === DEFAULT_PARAMETERS.geometryPointsPerFit &&
    row.slopeSharedAcrossTiers === DEFAULT_PARAMETERS.slopeSharedAcrossTiers &&
    row.validationSampleFraction === DEFAULT_PARAMETERS.validationSampleFraction;

  return {
    grid,
    rows,
    defaultRow: rows.find(isDefault) ?? null,
    exactGeometryOverFitRange: [Math.min(...exactOverFit), Math.max(...exactOverFit)],
    fitOverPerCellRange: [Math.min(...fitOverPerCell), Math.max(...fitOverPerCell)],
    // The load-bearing robustness claim, computed rather than asserted.
    inversionHoldsEverywhere: fitOverPerCell.every((ratio) => ratio > 1),
    note:
      "These three parameters were previously fixed and unswept while the file's own policy says an " +
      "unknowable value must be exposed and swept. They move the exact-geometry/fit ratio across the " +
      `range above — a factor of about ${round(Math.max(...exactOverFit) / Math.min(...exactOverFit), 1)} ` +
      "— so the single headline ratio is a point on a surface, not a constant. What does NOT move is " +
      "the DIRECTION: fit/per-cell exceeds 1 at every grid point, so fitting costs more than the " +
      "per-cell campaign under every combination tested. The finding is robust; its magnitude is not.",
  };
}

export function fitVersusExhaustive(cells, kreaPrecedent, parameters = DEFAULT_PARAMETERS, coverage = {}) {
  const census = cellCensus(cells);
  const runs = collapseToRuns(cells, parameters, coverage);
  const points = Math.max(1, Math.trunc(parameters.geometryPointsPerFit ?? 2));
  // Only explicitly planned batch groups reduce provider invocations. The exact-geometry and fit
  // campaigns include unplanned geometry points, so they retain the conservative fresh-rung count.
  const rungSpan = runs.rungCollapseFactor ?? 1;

  // Exhaustive at the exact geometry the manifest's own policy demands ("not permission to
  // interpolate"): one session per (session key, resolution).
  const exhaustiveExactGeometry = census.sessions.geometryExpanded;
  // Exhaustive as the binding logic actually accepts it: one session per session key, one
  // geometry each.
  const exhaustivePerCell = census.sessions.total;
  // Fit: `points` geometries per tier, or `points` on one tier plus one on each other tier when
  // the slope may be shared. `tierSum` is the number of (group, tier) pairs, which equals the
  // session count; `fitGroups.total` is the number of groups needing the extra slope point.
  const fitSessions = parameters.slopeSharedAcrossTiers
    ? census.fitGroups.tierSum + census.fitGroups.total * (points - 1)
    : census.fitGroups.tierSum * points;
  const validationFraction = Math.min(Math.max(parameters.validationSampleFraction ?? 0, 0), 1);
  const validationSessions = Math.ceil(census.sessions.total * validationFraction);

  return {
    formula: {
      kind: "affine in megapixels",
      expression: "fixedGb + perMpxGb * megapixels",
      citation: "crates/sceneworks-worker/src/vram_gate.rs#krea_phase_curve",
      coefficientsPerCurve: 2,
      minimumPointsToDetermine: 2,
    },
    kreaPrecedent,
    strategies: [
      {
        name: "exhaustive-exact-geometry",
        sessions: exhaustiveExactGeometry,
        providerInvocations: exhaustiveExactGeometry * rungSpan,
        policy:
          "Measure every advertised resolution. Retained as the UPPER BOUND a purely " +
          "per-resolution reading of the evidence would have cost. SC-16060 established that " +
          "nothing needs this: the manifest's 'exact request envelopes, not permission to " +
          "interpolate' governs the characterization claim, which `fit-then-validate` prices.",
      },
      {
        name: "exhaustive-per-cell",
        sessions: exhaustivePerCell,
        providerInvocations: runs.shippedModelLoads.total,
        policy:
          "One measurement per cell at one geometry. Since SC-16060 this is the honest price of " +
          "the IMPLEMENTATION claim alone — it takes every cell to `state: Verified` and leaves " +
          "`memoryCharacterization` at `point`, which the artifact now says out loud instead of " +
          "reading as envelope-wide certification.",
      },
      {
        name: "fit-then-validate",
        sessions: fitSessions + validationSessions,
        providerInvocations: (fitSessions + validationSessions) * rungSpan,
        fitSessions,
        validationSessions,
        policy:
          `Measure ${points} geometry points per (entry, backend, mode, overlay, tier) to determine ` +
          "the affine curve, then spot-check a sample against the prediction. Covers every geometry " +
          "in the envelope that is at or below the tier's `maxMeasuredPixels`, including ones never " +
          "rendered — but ONLY up to that bound. Above it the curve is not consulted at all: " +
          "`krea_rung_phase_peaks` returns `None` and `krea_turbo_fit_with_runtime` returns " +
          "`Unverified { OutOfEnvelope }` (vram_gate.rs:481-485 and :578-583), and the manifest says " +
          "why in its own words — \"do not extrapolate these curves into unvalidated attention " +
          "shapes\". So a fit does not buy coverage of the whole envelope; it buys coverage of the " +
          "measured sub-envelope. Note this is UNKNOWN, not `Reject`: the job is not refused, the " +
          "measured ladder is simply declined and the older sequential gate decides " +
          "(vram_gate.rs:548-549 draws that distinction explicitly).",
      },
    ],
    ratios: {
      exactGeometryOverFit: round(exhaustiveExactGeometry / (fitSessions + validationSessions), 2),
      exactGeometryOverPerCell: round(exhaustiveExactGeometry / exhaustivePerCell, 2),
      fitOverPerCell: round((fitSessions + validationSessions) / exhaustivePerCell, 2),
    },
    finding:
      "Fitting does NOT beat the per-cell campaign on run count — it costs more, because the " +
      "matrix already collapsed geometry and a curve needs a second geometry point the per-cell " +
      "campaign never takes. Fitting wins only against the policy the manifest actually states " +
      "(no interpolation), where it is the difference between a campaign that covers every " +
      "advertised resolution and one that does not. The decision is therefore about which " +
      "geometry policy is true, not about which strategy is cheaper.",
  };
}

/** Parse the Krea turboFit block. Derived from the manifest so the precedent cannot go stale silently. */
export function kreaFitPrecedent(manifestJson) {
  const krea = manifestJson.models.find((model) => model.id === "krea_2_turbo");
  const turboFit = krea?.candle?.turboFit;
  if (!turboFit) throw new Error("could not locate krea_2_turbo candle.turboFit in the manifest");
  const records = turboFit.evidenceRecords ?? [];
  const curves = turboFit.phaseCurvesByTier ?? {};

  const geometriesByTier = {};
  for (const record of records) {
    const tier = record.tier;
    geometriesByTier[tier] = [...new Set([...(geometriesByTier[tier] ?? []), `${record.width}x${record.height}`])].sort();
  }

  // A zero `perMpxGb` on the `text` phase is CORRECT — conditioning does not scale with output
  // area. So the geometry-sensitive phases are counted separately; a flat denoise or decode slope
  // is the one that signals an undetermined fit rather than a real physical constant.
  const geometrySensitivePhases = ["denoise", "decode"];
  let curveCount = 0;
  const slopesByTier = {};
  for (const [tier, rungs] of Object.entries(curves)) {
    let flat = 0;
    let total = 0;
    for (const phases of Object.values(rungs)) {
      for (const [phaseName, phase] of Object.entries(phases)) {
        curveCount += 1;
        if (!geometrySensitivePhases.includes(phaseName)) continue;
        total += 1;
        if (phase.perMpxGb === 0) flat += 1;
      }
    }
    slopesByTier[tier] = {
      geometryPoints: (geometriesByTier[tier] ?? []).length,
      geometrySensitiveCurves: total,
      flatSlopes: flat,
    };
  }

  const underDetermined = Object.entries(geometriesByTier)
    .filter(([, geometries]) => geometries.length < 2)
    .map(([tier]) => tier)
    .sort();
  const detail = Object.entries(slopesByTier)
    .sort()
    .map(
      ([tier, stats]) =>
        `${tier} (${stats.geometryPoints} geometry point${stats.geometryPoints === 1 ? "" : "s"}): ` +
        `${stats.flatSlopes}/${stats.geometrySensitiveCurves} flat`,
    )
    .join("; ");

  return {
    source: "config/manifests/builtin.models.jsonc#models/krea_2_turbo/candle/turboFit",
    measurementRecords: records.length,
    geometriesByTier: Object.fromEntries(Object.entries(geometriesByTier).sort()),
    tiers: Object.keys(curves).sort(),
    curves: curveCount,
    coefficients: curveCount * 2,
    slopesByTier: Object.fromEntries(Object.entries(slopesByTier).sort()),
    underDeterminedTiers: underDetermined,
    honestyNote:
      underDetermined.length === 0
        ? "Every tier carries at least two geometry points, so every slope is determined."
        : `Tiers ${underDetermined.join(", ")} carry ONE geometry point each, so their perMpxGb ` +
          "slopes cannot have been fitted. The zero-slope counts on the geometry-SENSITIVE phases " +
          `(denoise, decode) show the consequence — ${detail}. A campaign budgeted from this ` +
          "precedent must budget two geometry points per tier, not the four records this evidence " +
          "actually cost.",
    // Sharper than the note above, and aimed at the code rather than the campaign: the gate's own
    // doc comment claims a provenance the machine-readable evidence does not support for 2 of 3
    // tiers. The gate reads `evidenceRecords`; the reassuring sentence is next to the reader.
    slopeProvenanceContradiction:
      underDetermined.length === 0
        ? null
        : {
            claim:
              'crates/sceneworks-worker/src/vram_gate.rs:460-463 documents the slope as "the ' +
              'geometry-dependent activation slope, fitted from real renders at multiple resolutions"',
            reality:
              `that is true for q4 only. Tiers ${underDetermined.join(", ")} carry ONE geometry point ` +
              "each in the same manifest block the gate reads, so their slopes cannot have been " +
              "fitted from multiple resolutions — they are either carried over from q4 or left at " +
              "zero. Both show up in the data: the threeStage/tiledVae denoise slopes are 7.98 / " +
              "7.98 / 7.90 across q4/q8/bf16 (evidently shared, not independently fitted), while " +
              `4 of 8 geometry-sensitive curves on each of ${underDetermined.join(" and ")} ship a ` +
              "zero slope versus 1 of 8 on q4",
            nuance:
              "the manifest's PROSE comment does quote 768x768 and 1024x1024 raw samples for all " +
              "three tiers, so the sentence is not invented — but the machine-readable " +
              "`evidenceRecords` the gate actually consumes contain one geometry cell for q8 and one " +
              "for bf16. The defect is a comment that reassures the reader about evidence the " +
              "artifact does not carry, which is exactly how a single-point slope gets trusted as a " +
              "fitted one",
            consequence:
              "any consumer budgeting or trusting per-tier slopes from this precedent should treat " +
              "q8 and bf16 as unfitted. Recorded on SC-16060 (geometry policy) alongside the " +
              "binding-vs-manifest contradiction",
          },
  };
}

/**
 * BIGGEST UNCERTAINTIES, RANKED BY INDEPENDENTLY-BLOCKING CELL COUNT.
 *
 * The previous version of this list ranked the overlay gate #1 on the strength of its first-match
 * bucket size ("the largest single bucket... gates 59.5% of the matrix") and prescribed scoping
 * overlay support everywhere. That mis-directs the wave: the overwhelming majority of those
 * cells are ALSO blocked by having no provider adapter at all, so overlay work moves far fewer cells
 * than the bucket size implies. Ranking is therefore derived from `independentBlocking` — cells a gate
 * is the ONLY thing wrong with — rather than from which gate happens to be charged first.
 *
 * `perRunSeconds` is not cell-gated (it is a multiplier on every hour) so it carries no blocking
 * count; it is placed directly after the top code gate because it scales everything the code gates
 * unlock.
 */
export function rankUncertainties(
  prod,
  { completeRecords = 0, adapterModelCount = 0, catalogEntryCount = 0 } = {},
) {
  const blocking = Object.fromEntries(prod.independentBlocking.map((entry) => [entry.id, entry]));
  const ranking = prod.blockerRanking;
  const partitionTotal = prod.partition.reduce((sum, entry) => sum + entry.cells, 0);
  const overlayFirstMatchShare = round(
    (ranking.overlayChargedByFirstMatch / partitionTotal) * 100,
    1,
  );

  const codeGated = [
    {
      gate: "no-provider-adapter",
      input: "how much provider-adapter work the CATALOG COVERAGE gap needs",
      why:
        `This is the largest INDEPENDENT blocker in the matrix: it is the only thing wrong with ` +
        `${blocking["no-provider-adapter"]?.soleBlockerCells ?? 0} cells and it touches ` +
        `${blocking["no-provider-adapter"]?.cellsTouched ?? 0} — ${ranking.overlayAlsoBlockedByAdapterCoverage} ` +
        `of them jointly with the overlay gap. Only ${adapterModelCount} of ${catalogEntryCount} ` +
        "catalog entries have a provider adapter " +
        "at all, so most of the matrix cannot be measured by any code that exists, for any overlay. " +
        "Charge this gate before overlay and it accounts for " +
        `${prod.gateOrderSensitivity.adapterCoverageFirst["no-provider-adapter"] ?? 0} cells.`,
      howToResolve:
        "Scope provider-adapter coverage across the catalog — this is the work that actually moves " +
        "the population, and it is a prerequisite for the overlay work rather than a parallel track. " +
        "Land it against SC-15508's harness contract. Nothing else on this list can be started " +
        "independently of it.",
      story: 15508,
    },
    {
      gate: "adapter-gated",
      input: "what an activation-eligible record actually requires from an existing adapter",
      why:
        `${blocking["adapter-gated"]?.soleBlockerCells ?? 0} cells are blocked ONLY by this, and ` +
        `${blocking["adapter-gated"]?.cellsTouched ?? 0} in total. The Krea MLX control adapter has ` +
        "now emitted complete exact records, proving the pipeline can close; this bucket applies " +
        "only to adapter-covered pairs that still have no activation-eligible record.",
      howToResolve:
        "Close the harness contract for each remaining adapter-covered pair: real phase telemetry, " +
        "bounded-output parity, exact-fit/stale/unknown selection, lifecycle recovery, loadability, " +
        "and a measured negative mutation. Complete records then move those exact targets out of " +
        "`adapter-gated`.",
      story: 15508,
    },
    {
      gate: "overlay-declined",
      input: "how much provider-adapter work the OVERLAY axis needs",
      why:
        `DEMOTED, and this is the correction that matters most in this document. The first-match ` +
        `partition charges this gate ${ranking.overlayChargedByFirstMatch} cells ` +
        `(${overlayFirstMatchShare}% of the ${partitionTotal}-cell matrix), which previously made ` +
        `it the #1 uncertainty. But it is the SOLE blocker on ` +
        `${ranking.overlayIndependentlyBlocking} of them: ${ranking.overlayAlsoBlockedByAdapterCoverage} ` +
        "are also blocked by having no adapter or no `complete` record at all. Scoping overlay " +
        `support where it remains absent therefore moves ${ranking.overlayBindingAfterAdapterCoverage} cells, ` +
        `not ${ranking.overlayChargedByFirstMatch}. It remains necessary work — those ` +
        `${ranking.overlayChargedByFirstMatch} cells cannot be certified without it — but it is ` +
        "not where the schedule is decided.",
      howToResolve:
        "Still SC-16072, but sequenced AFTER catalog adapter coverage rather than ahead of it. Note " +
        "that the load-side saving SC-16072 was expected to unlock is not available in the shipped " +
        "load contract at all (see the `overlay` axis above), so this work should be scoped for " +
        "RECORD correctness, not for a model-load reduction. Control coverage is now declared by " +
        "CONTROL_LANE_MODELS and is not part of this remediation.",
      story: 16072,
    },
  ]
    .map((entry) => ({
      ...entry,
      independentlyBlockingCells: blocking[entry.gate]?.soleBlockerCells ?? 0,
      cellsTouched: blocking[entry.gate]?.cellsTouched ?? 0,
    }))
    // The ranking rule itself, applied rather than assumed.
    .sort(
      (left, right) =>
        right.independentlyBlockingCells - left.independentlyBlockingCells ||
        right.cellsTouched - left.cellsTouched ||
        left.gate.localeCompare(right.gate),
    );

  const perRunSeconds = {
    input: "perRunSeconds",
    why:
      "It is a pure multiplier on every reported hour and it spans a 40x range in the sweep. The " +
      `current artifact has ${completeRecords} activation-eligible record(s) and ${prod.producibleToday} ` +
      "producible cell(s), but the records do not carry per-scenario duration telemetry, so they " +
      "do not narrow that multiplier yet.",
    howToResolve:
      "Add per-scenario timing instrumentation to the now-complete Krea MLX adapter path and retain " +
      "the exact geometry/strategy identity on every duration. This reuses an already-green real " +
      "campaign instead of extrapolating one end-to-end duration across unrelated providers, and " +
      "it narrows the 40x sweep without inventing timing for still-gated pairs.",
  };

  return [
    codeGated[0],
    perRunSeconds,
    ...codeGated.slice(1),
    {
      input: "how many geometry points each cell's curve needs",
      why:
        "RESOLVED as posed, and the resolution changed the question. SC-16060 found the two 'policies' " +
        "were answers to different claims: whether a rung WORKS is not geometry-sensitive, and whether " +
        "its PEAKS are known across the envelope is. The matrix now carries them separately — `state` " +
        "and `cells[].memoryCharacterization` — so one measured geometry establishes the first and " +
        "cannot pretend to establish the second. What remains open is a campaign question, not a " +
        "contradiction: the implementation claim collapses geometry entirely, while the " +
        "characterization claim needs at least `geometryPointsPerFit` points per (entry, backend, " +
        "mode, overlay, tier), and only the exhaustive-exact-geometry column ever needed the full " +
        "per-resolution count.",
      howToResolve:
        "Nothing to decide — read the two columns. `exhaustive-per-cell` now prices the implementation " +
        "claim and `fit-then-validate` prices the characterization claim; they are complementary " +
        "rather than rival. `exhaustive-exact-geometry` is retained as the upper bound a purely " +
        "per-resolution reading would have cost. The under-determined shipped tiers are reported by " +
        "`fitVersusExhaustive.kreaPrecedent` and now also by the matrix itself, where Krea's q8 and " +
        "bf16 cells read `point` rather than `fitted`.",
      story: 16060,
    },
  ];
}

// Every source this document is DERIVED from, exported for the tests that pin the tripwire's
// coverage (sc-16268). `sourceRevision`/`jsonc` are in the list because they DEFINE how every other
// entry is hashed: before sc-16268 that logic lived inside `generate-memory-matrix.mjs`, which is
// hashed here, so leaving them out would have narrowed provenance as a side effect of extracting a
// shared module.
export const SOURCE_PATHS = Object.freeze({
  matrix: "docs/generated/memory-matrix.json",
  manifest: "config/manifests/builtin.models.jsonc",
  harness: "scripts/memory-calibration-harness.mjs",
  matrixGenerator: "scripts/generate-memory-matrix.mjs",
  sourceRevision: "scripts/lib/source-revision.mjs",
  jsonc: "scripts/lib/jsonc.mjs",
  calibrationPlan: "config/memory-calibration-plan.json",
  vramGate: "crates/sceneworks-worker/src/vram_gate.rs",
  calibrationEvidence: "docs/generated/memory-calibration-evidence.json",
  candleAdapter: "crates/sceneworks-memory-adapter/src/bin/candle.rs",
  mlxAdapter: "crates/sceneworks-memory-adapter/src/bin/mlx.rs",
});

export async function buildCostModel({ sourceOverrides = {} } = {}) {
  const sourcePaths = SOURCE_PATHS;
  const sourceEntries = Object.entries(sourcePaths);
  const bodies = Object.fromEntries(
    await Promise.all(
      sourceEntries.map(async ([name, relative]) => [
        name,
        canonicalSourceText(
          Object.hasOwn(sourceOverrides, name)
            ? sourceOverrides[name]
            : await readFile(path.join(ROOT, relative), "utf8"),
        ),
      ]),
    ),
  );

  const matrix = JSON.parse(bodies.matrix);
  const manifest = JSON.parse(stripJsoncComments(bodies.manifest));
  // sc-16268: same provenance rule as the matrix generator — hash each source's semantic body, so a
  // comment-only edit to `vram_gate.rs` or to a generator script does not rotate this document's
  // fingerprint. Without this, wiring `check:calibration-cost-model` into `rust:check` would just
  // move the inert churn from CI to the pre-push gate.
  const revisionBodies = Object.fromEntries(
    sourceEntries.map(([name, relative]) => [name, semanticSourceBody(relative, bodies[name])]),
  );
  const plan = JSON.parse(bodies.calibrationPlan);
  const evidence = JSON.parse(bodies.calibrationEvidence);

  const cells = matrix.cells;
  const census = cellCensus(cells);
  const batchCoverage = { batchedSessionKeys: batchedSessionKeysFromPlan(plan) };
  const runs = collapseToRuns(cells, DEFAULT_PARAMETERS, batchCoverage);
  const precedent = kreaFitPrecedent(manifest);

  // Completed work is derived rather than assumed, from two independent places that must agree:
  // the committed evidence bundle and the matrix's own summary.
  const recordsByStatus = tally(evidence.records.map((record) => record.status));
  const completedBaseline = {
    evidenceBundleRecords: evidence.records.length,
    recordsByStatus,
    completeRecords: recordsByStatus.complete ?? 0,
    runtimeCompleteRecords: recordsByStatus.runtime_complete ?? 0,
    activationEligibleRecords:
      (recordsByStatus.complete ?? 0) + (recordsByStatus.runtime_complete ?? 0),
    matrixSummaryCalibrationRuns: matrix.summary.calibrationRuns,
    matrixSummaryCurrentCalibrationRuns: matrix.summary.currentCalibrationRuns,
    verifiedCells: census.byState.Verified ?? 0,
    note:
      `${recordsByStatus.complete ?? 0} Full complete and ${recordsByStatus.runtime_complete ?? 0} ` +
      `base-only runtime-complete record(s) exist in the evidence bundle; the matrix ` +
      `reports ${matrix.summary.currentCalibrationRuns} current calibration run(s) and ` +
      `${census.byState.Verified ?? 0} aggregate Verified cell(s). Exact records remain narrower ` +
      "than aggregate matrix cells, so a current exact run does not by itself promote a whole cell.",
  };

  // Producibility. Adapter coverage is derived from the shipped plan's provider targets; the
  // complete-record set is derived from the evidence bundle, so both self-update.
  const adapterPairs = plan.providers.map(
    (provider) => `${provider.target.modelId}|${provider.backend}`,
  );
  const pairsWithCompleteRecords = evidence.records
    .filter((record) => ["complete", "runtime_complete"].includes(record.status))
    .map((record) => `${record.target.modelId}|${record.backend}`);
  const matrixOverlay = (overlay) => (/^control:\d+$/.test(overlay) ? "control" : overlay);
  const overlayKeysWithCompleteRecords = evidence.records
    .filter(
      (record) =>
        record.status === "complete" &&
        record.target.overlay !== "none" &&
        record.scenarios.some(
          (scenario) => scenario.name === "overlay" && scenario.result === "passed",
        ),
    )
    .map(
      (record) =>
        `${record.target.modelId}|${record.backend}|${matrixOverlay(record.target.overlay)}`,
    );
  const producible = producibility(cells, {
    adapterPairs,
    pairsWithCompleteRecords,
    overlayKeysWithCompleteRecords,
  });

  // Control-overlay coverage is capability-declared, not inferred from measurement blocks
  // (sc-16069). `CONTROL_LANE_MODELS` is the generator's declaration source; every declared model
  // gets control cells for each backend that actually exists in the generated matrix.
  const controlCells = cells.filter((cell) => cell.overlay === "control");
  // sc-18099: the emitted PAIRS come from the matrix's published axes, not from the published cells.
  // A declared control lane with nothing planned or measured now publishes no cell — reading the
  // pairs off `cells` would report `kolors|candle` as having no control lane, which is exactly the
  // sc-16069 defect (absent evidence read as absent feature) that this section exists to disprove.
  const controlPairs = matrix.models
    .flatMap((model) =>
      Object.entries(model.axes)
        .filter(([, axes]) => axes.overlays.includes("control"))
        .map(([backend]) => `${model.id}|${backend}`),
    )
    .sort();
  const declaredControlBlock = bodies.matrixGenerator
    .split("export const CONTROL_LANE_MODELS = [")[1]
    ?.split("];")[0];
  if (declaredControlBlock === undefined) {
    throw new Error("matrix generator must expose CONTROL_LANE_MODELS");
  }
  const declaredControlModels = [...declaredControlBlock.matchAll(/"([^"]+)"/g)].map(
    (match) => match[1],
  );
  const cellsPerPairIfControlDeclared = Object.fromEntries(
    [...new Set(cells.map((cell) => `${cell.modelId}|${cell.backend}`))].sort().map((pair) => {
      const inPair = cells.filter((cell) => `${cell.modelId}|${cell.backend}` === pair);
      const tiers = new Set(inPair.map((cell) => cell.tier)).size;
      const modes = new Set(inPair.map((cell) => cell.mode)).size;
      return [pair, tiers * modes * (runs.rungCollapseFactor ?? RUNG_ORDER.length)];
    }),
  );
  const controlCoverage = {
    declaredControlModels,
    emittedControlPairs: controlPairs,
    controlCells: controlCells.length,
    citation:
      'scripts/generate-memory-matrix.mjs#overlaysFor emits "control" when ' +
      "`CONTROL_LANE_MODELS.includes(model.id)`",
    perPairCellCost: cellsPerPairIfControlDeclared,
    sensitivityNote:
      "Adding a model to CONTROL_LANE_MODELS adds tiers x modes x rungs cells for every backend " +
      "already in that model's matrix scope; measurement blocks affect evidence state, not whether " +
      "the capability axis exists.",
  };

  // The published sweep domain, derived from the shipped plan rather than asserted.
  const planProviders = plan.providers.map((provider) => ({
    name: provider.name,
    backend: provider.backend,
    rung: provider.rung,
    evidenceScope: provider.evidenceScope,
    plannedCases: provider.cases.length,
    positiveCases: provider.cases.filter((item) => item.expectedResult === "passed").length,
    modelLoadPolicy: provider.modelLoadPolicy ?? "fresh_per_case",
    modelLoadGroup: provider.modelLoadGroup ?? null,
  }));

  const model = {
    schemaVersion: 1,
    generatedFrom: {
      // NUL-separated for the same reason as the matrix generator (sc-16268): normalised bodies no
      // longer end in a newline, so a bare concatenation has ambiguous source boundaries.
      sceneWorksRevision: `source-tree:${sha256(
        sourceEntries.map(([name]) => revisionBodies[name]).join("\0"),
      )}`,
      matrixRevision: matrix.generatedFrom.sceneWorksRevision,
      inferenceRevision: matrix.generatedFrom.inferenceRevision,
      sources: Object.fromEntries(
        sourceEntries.map(([name, relative]) => [
          name,
          { path: relative, sha256: sha256(revisionBodies[name]) },
        ]),
      ),
    },
    // sc-18099. The matrix publishes a subset; these are its own numbers for what it resolved and
    // what it dropped, carried here so this model's `cells` total can never be mistaken for the
    // catalog cross-product.
    matrixPublication: {
      resolvedCoordinates: matrix.summary.cells,
      publishedCells: matrix.summary.publishedCells,
      elidedCoordinates: matrix.summary.elidedCells,
      elidedByState: matrix.summary.elidedByState,
      predicate: matrix.summary.publicationPredicate,
      note:
        "This model prices the PUBLISHED population. An elided coordinate is unplanned, unmeasured, " +
        "unbound and uncited; per-lane totals for every coordinate, published or not, are in " +
        "docs/generated/memory-matrix.json#coverage.",
    },
    confidenceTiers: {
      cells:
        "FACT — counted from the PUBLISHED cells of docs/generated/memory-matrix.json (sc-18099); " +
        "see `matrixPublication` for the resolved and elided coordinate totals",
      runs: "DERIVED — from the collapsing rule, every axis cited to code that was read",
      hours: "PARAMETERISED — per-run cost is an input with a swept range, not a measurement",
      producibility:
        "DERIVED — from the shipped plan, the evidence bundle, and the adapter sources; this is " +
        "the partition that says which of those hours can be spent today at all",
    },
    completedBaseline,
    producibility: producible,
    controlCoverage,
    cells: census,
    overlayLoadContract: OVERLAY_LOAD_CONTRACT,
    collapsing: {
      runUnit:
        "One provider invocation. `runProviderPlan` uses `action:\"run\"` for a fresh case and " +
        "`action:\"run_batch\"` for a compatible Candle five-rung group; either invocation attests " +
        "one model load.",
      runUnitCitation: "scripts/memory-calibration-harness.mjs#runProviderPlan",
      requiredScenariosPerRecord: REQUIRED_SCENARIOS.length,
      requiredScenarios: [...REQUIRED_SCENARIOS].sort(),
      axes: collapsingAxesFor(census, runs),
      shippedPlan: planProviders,
    },
    runs,
    naSensitivity: {
      unknowable:
        "SC-15969 (the per-family rung-4 applicability survey) has not run, so the final " +
        "Structurally N/A count cannot be known. Only the `current` row below is a fact.",
      selectionRule: NA_SELECTION_RULE,
      irreducibleFloor: {
        rung: "resident",
        cells: census.byRung.resident ?? 0,
        reason:
          "Rung 0 is the unoptimised baseline every model has by definition, so it can never be " +
          "Structurally N/A. No survey outcome can remove these cells from the campaign.",
      },
      scenarios: naSensitivity(cells, batchCoverage),
    },
    wallClock: wallClock(runs, DEFAULT_PARAMETERS),
    fitVersusExhaustive: {
      ...fitVersusExhaustive(cells, precedent, DEFAULT_PARAMETERS, batchCoverage),
      // Attached here rather than inside `fitVersusExhaustive` because the sweep calls that function
      // once per grid point and nesting it would recurse.
      sensitivity: fitSensitivity(cells, FIT_SENSITIVITY_GRID, batchCoverage),
    },
    parameters: DEFAULT_PARAMETERS,
    biggestUncertainties: rankUncertainties(producible, {
      completeRecords: completedBaseline.activationEligibleRecords,
      adapterModelCount: new Set(plan.providers.map((provider) => provider.target.modelId)).size,
      // sc-18099: the CATALOG's entry count, from `models`. Counting distinct ids in `cells` used to
      // be the same number and no longer is — the published cells are a subset, so it would say "9
      // of 13" and read as adapter coverage being nearly complete.
      catalogEntryCount: matrix.models.length,
    }),
  };

  return model;
}

function markdownTable(header, rows) {
  return [
    `| ${header.join(" | ")} |`,
    `| ${header.map(() => "---").join(" | ")} |`,
    ...rows.map((row) => `| ${row.join(" | ")} |`),
  ];
}

function renderMarkdown(model) {
  const { cells, runs, fitVersusExhaustive: fit, producibility: prod, controlCoverage } = model;
  const lines = [
    "# Calibration cost model",
    "",
    "> Generated by `scripts/calibration-cost-model.mjs`. Do not edit by hand.",
    "",
    `- Matrix revision: \`${model.generatedFrom.matrixRevision}\``,
    `- Inference revision: \`${model.generatedFrom.inferenceRevision}\``,
    "",
    "**Cells are a fact. Runs are derived. Hours are parameterised.** Nothing below mixes the three.",
    "",
    `> **What "cells" counts here (sc-18099).** ${model.matrixPublication.publishedCells} — the cells the memory matrix PUBLISHES, out of ${model.matrixPublication.resolvedCoordinates} coordinates the catalog resolves. The matrix stopped publishing the cross-product; ${model.matrixPublication.elidedCoordinates} unplanned, unmeasured, unbound and uncited coordinates are counted in \`docs/generated/memory-matrix.json#coverage\` instead of carrying a row. This document therefore prices the planned-or-evidenced workload, which is what epic 18093 intends to price — but every population figure below dropped by roughly that ratio against earlier revisions of this file, and that is a change of QUESTION, not of progress.`,
    "",
    "## 0. Read this first: the binding constraint is not GPU hours",
    "",
    `**${prod.headline}**`,
    "",
    `**${prod.behindProviderCode}** of ${cells.total} cells (**${round(prod.behindProviderCodeShare * 100, 1)}%**) are blocked by provider-adapter code that does not exist, not by machine time. **${prod.producibleToday}** could be certified by a run started now.`,
    "",
    ...markdownTable(
      ["Gate (first one that blocks the cell)", "Cells", "MLX", "Candle", "Blocked by"],
      prod.partition.map((gate) => [
        `\`${gate.id}\` — ${gate.label}`,
        String(gate.cells),
        String(gate.mlx),
        String(gate.candle),
        gate.blockedBy,
      ]),
    ),
    "",
    ...prod.partition
      .filter((gate) => gate.cells > 0)
      .map((gate) => `- \`${gate.id}\`: ${gate.citation}`),
    "",
    "### Blockers per cell — the partition above is ORDERED, so it hides co-blocking",
    "",
    "A cell is charged to the FIRST gate that blocks it, which makes the table above a partition but " +
      "**not** a measure of how much work each gate represents. Most blocked cells are blocked by more " +
      "than one thing at once. This is the order-independent view:",
    "",
    ...markdownTable(
      ["Blockers on the cell (multiset)", "Cells", "MLX", "Candle"],
      prod.blockersPerCell.map((entry) => [
        entry.blockers.length === 0 ? entry.description : entry.blockers.map((id) => `\`${id}\``).join(" + "),
        String(entry.cells),
        String(entry.mlx),
        String(entry.candle),
      ]),
    ),
    "",
    ...markdownTable(
      ["Blocker", "Cells touched", "Sole blocker (fixing it alone frees these)", "Co-blocked"],
      prod.independentBlocking.map((entry) => [
        `\`${entry.id}\``,
        String(entry.cellsTouched),
        String(entry.soleBlockerCells),
        String(entry.coBlockedCells),
      ]),
    ),
    "",
    `**${prod.blockerRanking.finding}**`,
    "",
    "Gate order is load-bearing and is therefore reported both ways rather than only disclosed in prose:",
    "",
    ...markdownTable(
      ["Gate", "Charged (shipped order: overlay first)", "Charged (adapter coverage first)"],
      [...new Set([
        ...Object.keys(prod.gateOrderSensitivity.shippedOrder),
        ...Object.keys(prod.gateOrderSensitivity.adapterCoverageFirst),
      ])]
        .sort()
        .map((gate) => [
          `\`${gate}\``,
          String(prod.gateOrderSensitivity.shippedOrder[gate] ?? 0),
          String(prod.gateOrderSensitivity.adapterCoverageFirst[gate] ?? 0),
        ]),
    ),
    "",
    `> ${prod.gateOrderSensitivity.note}`,
    "",
    `Remediation priority below is ranked by **${prod.blockerRanking.rankedBy}**, not by bucket size: ${prod.blockerRanking.ranking.map((id) => `\`${id}\``).join(" > ")}.`,
    "",
    `Completed-evidence baseline: ${model.completedBaseline.note}`,
    "",
    `The wall-clock sweep in section 5 is a gross campaign cost model, not a remaining-time estimate; promoted exact records are reported above rather than silently subtracted from aggregate cells.`,
    "",
    "## 1. Cells (fact)",
    "",
    ...markdownTable(
      ["Conformance state", "Cells"],
      Object.entries(cells.byState).map(([state, count]) => [`\`${state}\``, String(count)]),
    ),
    "",
    `Total **${cells.total}** cells across ${cells.sessions.total} (entry, backend, tier, mode, overlay) session keys, every one of which spans exactly ${cells.sessions.rungsPerSession.join("/")} rungs.`,
    "",
    ...markdownTable(
      ["Backend", "Cells", "Session keys"],
      Object.keys(cells.byBackend)
        .sort()
        .map((backend) => [
          backend,
          String(cells.byBackend[backend]),
          String(cells.sessions.byBackend[backend] ?? 0),
        ]),
    ),
    "",
    `Geometry is already collapsed: keyed per advertised resolution the matrix would hold **${cells.geometryExpandedCells}** cells, so the published count embeds a **${cells.geometryCollapseFactor}x** collapse.`,
    "",
    ...markdownTable(
      ["Overlay", "Cells"],
      Object.entries(cells.byOverlay).map(([overlay, count]) => [`\`${overlay}\``, String(count)]),
    ),
    "",
    `The overlay axis is **${cells.total - (cells.byOverlay.none ?? 0)}** cells — ${round(((cells.total - (cells.byOverlay.none ?? 0)) / cells.total) * 100, 1)}% of the matrix — and it is not uniformly real. \`lora\` and \`identity\` are genuine dimensions; \`control\` is a declared capability axis with ${controlCoverage.controlCells} cells on ${controlCoverage.emittedControlPairs.length} (entry, backend) pair(s).`,
    "",
    `**Control coverage is declared, not inferred from measurements.** ${controlCoverage.citation}. The declaration currently names ${controlCoverage.declaredControlModels.length} model(s), producing the ${controlCoverage.emittedControlPairs.length} backend pair(s) above. ${controlCoverage.sensitivityNote}`,
    "",
    "## 2. The collapsing rule (derived)",
    "",
    `${model.collapsing.runUnit} A \`complete\` record must pass all ${model.collapsing.requiredScenariosPerRecord} required scenarios plus quality and a measured negative mutation.`,
    "",
    ...markdownTable(
      ["Axis", "Verdict", "Factor"],
      model.collapsing.axes.map((axis) => [`\`${axis.axis}\``, axis.verdict, axis.factor]),
    ),
    "",
    ...model.collapsing.axes.flatMap((axis) => [
      `**\`${axis.axis}\` — ${axis.verdict}.** ${axis.rule}`,
      "",
      `> Citation: ${axis.citation}`,
      "",
      `> Caveat: ${axis.caveat}`,
      "",
    ]),
    "## 3. Runs (derived)",
    "",
    ...markdownTable(
      ["Unit", "Total", "MLX/Metal (this Mac)", "Candle/CUDA (rented)"],
      [
        [
          "Certifying records (1 per cell)",
          String(runs.certifyingRecords.total),
          String(runs.certifyingRecords.mlx),
          String(runs.certifyingRecords.candle),
        ],
        [
          "Shipped model loads (explicit groups only)",
          String(runs.shippedModelLoads.total),
          String(runs.shippedModelLoads.mlx),
          String(runs.shippedModelLoads.candle),
        ],
        [
          `All-backend warm-session floor (${runs.rungCollapseFactor} rungs each)`,
          String(runs.warmSessions.total),
          String(runs.warmSessions.mlx),
          String(runs.warmSessions.candle),
        ],
      ],
    ),
    "",
    `**Does the overlay axis evaporate? No — on either axis.** On the record axis, each of the ${cells.total - (cells.byOverlay.none ?? 0)} non-\`none\` cells needs its own record, because \`overlay\` is in the cell key and \`validateComplete\` requires the overlay scenario to actually pass.`,
    "",
    `On the model-load axis the answer is **${model.overlayLoadContract.verdict}**. The counterfactual price tag says dropping overlay from the session key would take loads from ${runs.warmSessions.total} to ${runs.unavailableOverlayLoadPriceTag.sessions.total} — a further ${runs.unavailableOverlayLoadPriceTag.collapseFactor}x — but that figure is structured metadata, not a campaign column, and it **prices a capability the shipped code does not offer**. An earlier version of this document wrongly reported it as "mostly yes". Four findings, the last decisive:`,
    "",
    `1. **Adapters are load-time.** ${model.overlayLoadContract.adaptersAreLoadTime}.`,
    `2. **No runtime swap API exists.** ${model.overlayLoadContract.noRuntimeSwapApi}.`,
    `3. **Adapters void the measured ladder.** ${model.overlayLoadContract.adaptersDisableARung}.`,
    `4. **The collapse perturbs the measurement.** ${model.overlayLoadContract.baselinePerturbation}.`,
    "",
    `> ${model.overlayLoadContract.counterfactualPriceTagDisposition}`,
    "",
    ...markdownTable(
      ["Overlay", "Warm attach/detach without base reload", "None-baseline comparison", "Tolerance (bytes)", "Load sharing"],
      Object.entries(model.overlayLoadContract.perOverlayKind).map(([overlay, verdict]) => [
        `\`${overlay}\``,
        String(verdict.warmAttachDetachWithoutBaseReload),
        verdict.noneBaselineComparison.status,
        String(verdict.noneBaselineComparison.toleranceBytes),
        String(verdict.loadSharingAvailable),
      ]),
    ),
    "",
    `Excluded: ${runs.exemptions.alreadyStructurallyNotApplicable} cells already classified \`Structurally N/A\`, which the epic exempts from measurement. Nothing else is excluded — \`Missing\` cells still need a run, they just need an implementation first.`,
    "",
    `The shipped campaign now costs **${runs.shippedModelLoads.total} model loads**. Exactly ${runs.shippedBatchCoverage.records} records in ${runs.shippedBatchCoverage.sessions} groups are configured for batching, saving ${runs.shippedBatchCoverage.savedLoads} loads; every record remains fresh. That is a **${runs.shippedLoadCollapseFactor}x** reduction from ${runs.certifyingRecords.total} fresh-per-record loads. The ${runs.warmSessions.total}-load floor remains counterfactual because MLX is structurally unable and Candle failed the committed fresh/reused tolerance.`,
    "",
    "## 4. Structurally N/A sensitivity",
    "",
    model.naSensitivity.unknowable,
    "",
    `Irreducible floor: the **${model.naSensitivity.irreducibleFloor.cells}** \`${model.naSensitivity.irreducibleFloor.rung}\` cells. ${model.naSensitivity.irreducibleFloor.reason}`,
    "",
    ...markdownTable(
      ["Scenario", "Kind", "Records", "MLX", "Candle", "Shipped loads", "Warm floor", "Exempted MLX/Candle", "...if proportional"],
      model.naSensitivity.scenarios.map((scenario) => [
        `\`${scenario.name}\``,
        scenario.kind,
        String(scenario.certifyingRecords.total),
        String(scenario.certifyingRecords.mlx),
        String(scenario.certifyingRecords.candle),
        String(scenario.shippedModelLoads.total),
        String(scenario.warmSessions.total),
        `${scenario.exemptedByBackend.mlx}/${scenario.exemptedByBackend.candle}`,
        scenario.kind === "fact"
          ? "—"
          : `${scenario.exemptedByBackendIfProportional.mlx}/${scenario.exemptedByBackendIfProportional.candle}${scenario.backendSplitIsSelectionArtifact ? " ⚠" : ""}`,
      ]),
    ),
    "",
    `> **How cells are selected, and why the projection rows' backend columns are not findings.** ${model.naSensitivity.selectionRule}`,
    "",
    `Rows marked ⚠ have a per-backend split that differs from proportional allocation — that gap is the selection rule showing through, not a property of the projection. Only the \`current\` row's split is a fact.`,
    "",
    `**The all-backend warm floor does not move.** A session survives as long as any one rung still needs measuring, and the resident baseline always does. The shipped-load column can move because both backends still pay per surviving rung; only the counterfactual warm floor is one load per surviving session.`,
    "",
    "## 5. Wall clock (parameterised)",
    "",
    `Per-run seconds is an **input**, defaulting to ${model.wallClock.assumption.value} s. ${model.wallClock.assumption.status}. ${model.wallClock.assumption.reasoning}`,
    "",
    "### What was actually measured",
    "",
    ...model.wallClock.measuredAnchors.flatMap((anchor) => [
      `**${anchor.what}** — ${anchor.measurement}.`,
      "",
      `> How: ${anchor.derivation}`,
      "",
      `> Does NOT cover: ${anchor.doesNotCover}`,
      "",
    ]),
    "### Sweep",
    "",
    ...markdownTable(
      [
        "s/run",
        "Record-hours (total)",
        "MLX",
        "Candle",
        "Shipped load-hours",
        "All-backend warm floor",
      ],
      model.wallClock.sweep.map((row) => [
        String(row.perRunSeconds),
        String(row.recordHours.total),
        String(row.recordHours.mlx),
        String(row.recordHours.candle),
        String(row.shippedModelLoadHours.total),
        String(row.warmSessionHours.total),
      ]),
    ),
    "",
    "The shipped load-hours column amortises only plan-declared, evidence-gated groups; all other records stay fresh. The final column is the unavailable all-backend warm floor. The unavailable overlay-amortised column is deliberately absent. And per section 0, none of these hours can be spent today at all.",
    "",
    "## 6. Fit coefficients versus measure every cell",
    "",
    `The phase curve is **${fit.formula.kind}** — \`${fit.formula.expression}\` (${fit.formula.citation}), so ${fit.formula.coefficientsPerCurve} coefficients per curve and ${fit.formula.minimumPointsToDetermine} geometry points to determine one.`,
    "",
    `Worked precedent: **${fit.kreaPrecedent.measurementRecords}** measurement records produced ${fit.kreaPrecedent.curves} curves (${fit.kreaPrecedent.coefficients} coefficients) across tiers ${fit.kreaPrecedent.tiers.join(", ")}.`,
    "",
    `> ${fit.kreaPrecedent.honestyNote}`,
    "",
    ...markdownTable(
      ["Strategy", "Session points", "Provider invocations (explicit batches only)"],
      fit.strategies.map((strategy) => [
        `\`${strategy.name}\``,
        String(strategy.sessions),
        String(strategy.providerInvocations),
      ]),
    ),
    "",
    ...fit.strategies.map((strategy) => `- \`${strategy.name}\`: ${strategy.policy}`),
    "",
    `Ratios: exact-geometry / fit = **${fit.ratios.exactGeometryOverFit}x**; exact-geometry / per-cell = ${fit.ratios.exactGeometryOverPerCell}x; fit / per-cell = ${fit.ratios.fitOverPerCell}x.`,
    "",
    `**${fit.finding}**`,
    "",
    "### Sensitivity of the fit ratio to its three previously-unswept parameters",
    "",
    `\`geometryPointsPerFit\`, \`slopeSharedAcrossTiers\` and \`validationSampleFraction\` were the only parameters in this model that were fixed rather than swept, which contradicted its own stated policy. Swept here. The exact-geometry/fit ratio ranges **${fit.sensitivity.exactGeometryOverFitRange[0]}x–${fit.sensitivity.exactGeometryOverFitRange[1]}x** across the grid, so the headline \`${fit.ratios.exactGeometryOverFit}x\` is one point on a surface.`,
    "",
    ...markdownTable(
      ["geometry points", "slope shared", "validation sample", "Fit loads", "Validation loads", "Total loads", "exact/fit", "fit/per-cell"],
      fit.sensitivity.rows.map((row) => [
        String(row.geometryPointsPerFit),
        String(row.slopeSharedAcrossTiers),
        String(row.validationSampleFraction),
        String(row.fitSessions),
        String(row.validationSessions),
        String(row.modelLoads),
        `${row.exactGeometryOverFit}x`,
        `${row.fitOverPerCell}x`,
      ]),
    ),
    "",
    `> ${fit.sensitivity.note}`,
    "",
    `**The inversion is robust${fit.sensitivity.inversionHoldsEverywhere ? "" : " — WARNING: not at every grid point"}:** fit/per-cell stays in ${fit.sensitivity.fitOverPerCellRange[0]}x–${fit.sensitivity.fitOverPerCellRange[1]}x, above 1 at ${fit.sensitivity.inversionHoldsEverywhere ? "every" : "only some"} grid point${fit.sensitivity.inversionHoldsEverywhere ? "" : "s"}, so "fitting costs more than the per-cell campaign" does not depend on the defaults. Only the magnitude does.`,
    "",
    ...(fit.kreaPrecedent.slopeProvenanceContradiction
      ? [
          "### The gate's slope-provenance comment overstates its own evidence",
          "",
          `**Claim:** ${fit.kreaPrecedent.slopeProvenanceContradiction.claim}.`,
          "",
          `**Reality:** ${fit.kreaPrecedent.slopeProvenanceContradiction.reality}.`,
          "",
          `> Nuance: ${fit.kreaPrecedent.slopeProvenanceContradiction.nuance}.`,
          "",
          `> Consequence: ${fit.kreaPrecedent.slopeProvenanceContradiction.consequence}.`,
          "",
        ]
      : []),
    "## 7. Biggest uncertainties",
    "",
    `Ranked by **${prod.blockerRanking.rankedBy}**, not by first-match bucket size. An earlier version of this document ranked the overlay gap #1 because it is the largest bucket in the ordered partition; ${prod.blockerRanking.overlayAlsoBlockedByAdapterCoverage} of those cells are also blocked by having no adapter at all, so that ranking pointed remediation at work that moves ${prod.blockerRanking.overlayBindingAfterAdapterCoverage} cells rather than ${prod.blockerRanking.overlayChargedByFirstMatch}.`,
    "",
    ...model.biggestUncertainties.flatMap((item, index) => [
      // `item.input` may itself contain backticks, so it is NOT wrapped in them here — nesting them
      // would break the heading's rendering.
      `### ${index + 1}. ${item.input}` +
        (item.gate
          ? ` — gate \`${item.gate}\`, **${item.independentlyBlockingCells}** cells independently blocked (${item.cellsTouched} touched)`
          : item.input === "perRunSeconds"
            ? " — not cell-gated; a pure multiplier on every reported hour"
            : " — not cell-gated"),
      "",
      item.why,
      "",
      `> Resolve by: ${item.howToResolve}`,
      "",
    ]),
  ];
  return `${lines.join("\n")}\n`;
}

async function main() {
  const model = await buildCostModel();
  const json = `${JSON.stringify(model, null, 2)}\n`;
  const markdown = renderMarkdown(model);
  if (process.argv.includes("--check")) {
    const [existingJson, existingMarkdown] = await Promise.all([
      readFile(path.join(ROOT, OUTPUT_JSON), "utf8"),
      readFile(path.join(ROOT, OUTPUT_MD), "utf8"),
    ]);
    if (canonicalSourceText(existingJson) !== json || canonicalSourceText(existingMarkdown) !== markdown) {
      throw new Error("generated calibration cost model is stale; run npm run generate:calibration-cost-model");
    }
    return;
  }
  await mkdir(path.join(ROOT, "docs/generated"), { recursive: true });
  await Promise.all([
    writeFile(path.join(ROOT, OUTPUT_JSON), json),
    writeFile(path.join(ROOT, OUTPUT_MD), markdown),
  ]);
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  await main();
}
