// Project the ENGINE capability dumps into the manifest's per-model memory-declaration blocks.
//
// WHY THIS EXISTS (sc-20246, Michael 2026-08-17)
// ---------------------------------------------
// `config/manifests/builtin.models.jsonc` used to carry a hand-maintained SECOND opinion about which
// memory-ladder rungs each provider implements at which tier. The engine registries already publish
// that inventory — `memoryContracts` (per provider: which rungs are implemented under which
// tier/offloadPolicy/loadShape selector) and `memoryRouteWitnesses` (per provider: which
// tier/mode/overlay/loadProfile coordinates production actually routes), both landed by PR #2386. Two
// sources of truth for one fact drift, and the sc-18460 reconciliation report measured the drift at
// 429 engine→manifest coordinates.
//
// So the engine becomes the single source of truth and the manifest MIRRORS it. This module owns the
// projection; `scripts/generate-manifest-memory-declarations.mjs` is the CLI.
//
// ## What is and is not projected
//
// The dumps are authoritative for exactly the axes they publish:
//
//   * `rung` + `tiers`        <- `memoryContracts[].surfaces[].implementedRungs` keyed by
//                                `selector.tier`. This is the axis pair the reconciliation's
//                                engine_manifest leg compares, so it is the axis pair that heals.
//   * `modes` / `overlays` /  <- `memoryRouteWitnesses`, then INTERSECTED with the catalog entry's
//     `loadProfiles`             own axes (`catalogAxes`). The route rules are coarse — several use
//                                `ALL_MODES`, so their witnesses claim `image_detail` for an entry
//                                with no detail capability — and a row that inherited that would
//                                declare cells the matrix has none for.
//
// A rung/tier with no witnessed, catalog-reachable coordinate emits NO ROW and is reported instead.
// An empty `modes`/`overlays` would match nothing at runtime and nothing in the matrix, so writing one
// would be paperwork that zeroes a report counter while declaring an unreachable capability — the
// failure mode the deleted waiver ledger was made of.
//
// The dumps publish NOTHING about `parameters`, `parameterRanges`, `fingerprint`, `engagedRungs`,
// `requestContexts`, `providerOverlay` or `sourceKinds`. Those are measured/hand-authored, so
// generated rows OMIT them rather than guessing. Two consequences worth knowing:
//
//   * `engagedRungs` is deliberately absent: the memory matrix reads it as a rung-1 prerequisite
//     proof (`stagedResidencyIsAvailable`), and a derived ladder prefix is not that proof.
//   * a row with no `parameters` is a COVERAGE claim, and `strategyStatus` in
//     `scripts/generate-memory-matrix.mjs` was taught (sc-20246) to let such a row yield to
//     calibration bindings and the rung-4 survey, which do publish parameters.
//   * an MLX lane whose route rules are non-legacy REQUIRES `requestContexts` on every row, and
//     reads a row without it as malformed — failing the whole load closed to `Refused + Eager`.
//     Those lanes are skipped outright; see `parseRequestContextLanes`.
//
// ## Why generated rows are APPENDED, never merged into hand-authored rows
//
// Existing implementation rows are left byte-identical. They carry measured narrow scopes — the
// `sdxl` MLX row that declares `bounded_decode` at q4 ONLY, on a 48/255-vs-84/255 quality
// withholding — and widening one to the engine's full tier set would silently overturn a measured
// verdict. Generated rows live between BEGIN/END marker comments so a rerun replaces exactly its own
// output and nothing else. That is also what makes the projection idempotent on a file full of
// hand-written comments.
//
// ## One host model per (backend, provider)
//
// The engine_manifest leg is provider-scoped, not model-scoped: one declaration anywhere satisfies
// it. Several catalog entries share a provider (five SceneWorks ids run `sdxl`), and some of them
// declare deliberately LESS than their siblings — `illustrious_xl_v1`/`v2` carry no
// bounded_transformer_residency row and SC-15525 records that as a MEASURED verdict. Writing into
// every sharing model would steamroll those. So the projection picks exactly ONE primary host and
// leaves every other sharing entry untouched.
//
// ## Not a gate
//
// This is a developer tool, run like the repo's other `generate-*` scripts. Nothing checks its
// output; the reconciliation stays report-only (Michael, 2026-08-17). Do not wire it into CI.

import { stripJsoncComments } from "./jsonc.mjs";

/** Ladder order. `resident` is the floor every provider implements and is never declared as an
 *  implementation row, which is why the reconciliation excludes it from the engine_manifest leg. */
export const RUNG_ORDER = Object.freeze([
  "resident",
  "staged_residency",
  "bounded_decode",
  "bounded_attention",
  "bounded_transformer_residency",
]);

/** Canonical orders, matching the vocabularies `scripts/lib/memory-contract-reconciliation.mjs`
 *  validates against and the order the hand-authored manifest rows already use. Sorting by these
 *  rather than lexically is what keeps a regenerated array byte-stable AND readable. */
export const TIER_ORDER = Object.freeze(["bf16", "q8", "q4", "nvfp4"]);
export const MODE_ORDER = Object.freeze([
  "text_to_image",
  "style_variations",
  "edit_image",
  "image_to_image",
  "image_inpaint",
  "image_detail",
  "character_image",
]);
export const OVERLAY_ORDER = Object.freeze(["none", "lora", "control", "identity"]);
/** Load profile -> the overlay cell it serves, precedence identity > control > lora > none. The same
 *  map `scripts/lib/memory-contract-reconciliation.mjs` validates witnesses against; it is repeated
 *  here as the ORDER plus the lookup because a generated row's `loadProfiles` must stay consistent
 *  with the `overlays` it survived the catalog intersection with. */
export const LOAD_PROFILE_OVERLAYS = new Map([
  ["plain", "none"],
  ["lora", "lora"],
  ["lora_pid", "lora"],
  ["pid", "none"],
  ["single_control", "control"],
  ["multi_control", "control"],
  ["lora_single_control", "control"],
  ["ip_adapter", "identity"],
  ["ip_adapter_pid", "identity"],
  ["lora_ip_adapter", "identity"],
  ["lora_ip_adapter_pid", "identity"],
  ["identity", "identity"],
]);
export const LOAD_PROFILE_ORDER = Object.freeze([...LOAD_PROFILE_OVERLAYS.keys()]);

export const GENERATED_BEGIN =
  "// BEGIN generated-from-engine-capabilities — scripts/generate-manifest-memory-declarations.mjs";
export const GENERATED_END = "// END generated-from-engine-capabilities";

const orderedBy = (order, values) => {
  const set = new Set(values);
  const known = order.filter((item) => set.has(item));
  // An unknown value is a dump the vocabularies here have not caught up with. Keep it — sorted, at
  // the end — rather than silently dropping engine truth on the floor.
  const unknown = [...set].filter((item) => !order.includes(item)).sort();
  return [...known, ...unknown];
};

/**
 * Per-provider rung inventory from `memoryContracts`.
 *
 * A rung counts as implemented at a tier when ANY selector surface for that tier implements it. The
 * selector's `offloadPolicy`/`loadShape` axes describe HOW a load is shaped, not whether the rung
 * exists, and the manifest declaration has no field for either — so collapsing them is the honest
 * projection of "this provider can reach this rung at this tier".
 */
export function engineContractInventory(engineFacts) {
  const out = new Map();
  for (const document of engineFacts) {
    const backend = document.backend;
    for (const contract of document.memoryContracts ?? []) {
      const rungTiers = new Map();
      for (const surface of contract.surfaces ?? []) {
        const tier = surface.selector?.tier;
        for (const rung of surface.implementedRungs ?? []) {
          if (rung === "resident") continue;
          if (!rungTiers.has(rung)) rungTiers.set(rung, new Set());
          rungTiers.get(rung).add(tier);
        }
      }
      out.set(`${backend}:${contract.id}`, {
        backend,
        provider: contract.id,
        rungs: new Map(
          RUNG_ORDER.filter((rung) => rungTiers.has(rung)).map((rung) => [
            rung,
            orderedBy(TIER_ORDER, rungTiers.get(rung)),
          ]),
        ),
      });
    }
  }
  return out;
}

/** Per-provider production route coordinates from `memoryRouteWitnesses`. */
export function routeWitnessInventory(engineFacts) {
  const out = new Map();
  for (const document of engineFacts) {
    const backend = document.backend;
    for (const witness of document.memoryRouteWitnesses ?? []) {
      const key = `${backend}:${witness.provider}`;
      if (!out.has(key)) {
        out.set(key, { byTier: new Map() });
      }
      const entry = out.get(key);
      if (!entry.byTier.has(witness.tier)) {
        entry.byTier.set(witness.tier, {
          modes: new Set(),
          overlays: new Set(),
          loadProfiles: new Set(),
        });
      }
      const tier = entry.byTier.get(witness.tier);
      tier.modes.add(witness.mode);
      tier.overlays.add(witness.overlay);
      tier.loadProfiles.add(witness.loadProfile);
    }
  }
  return out;
}

/**
 * The witnessed coordinates of ONE tier, or null when the route registry witnesses none.
 *
 * `memoryRouteWitnesses` is the DEFERRED route population — the worker's typed
 * `deferred_route_witnesses()` rules, plus (on Candle) the manifest's own `requestContexts` rows. A
 * provider with no witness at a tier implements the rung in the engine but production cannot reach
 * it there, so no row is generated for that tier: a declaration with an empty `modes`/`overlays`
 * matches nothing at runtime and nothing in the matrix, and emitting one would be paperwork that
 * zeroes a report counter while declaring an unreachable capability.
 */
export function witnessCoordinatesForTier(witnesses, tier) {
  const entry = witnesses?.byTier.get(tier);
  if (!entry) return null;
  return {
    modes: orderedBy(MODE_ORDER, entry.modes),
    overlays: orderedBy(OVERLAY_ORDER, entry.overlays),
    loadProfiles: orderedBy(LOAD_PROFILE_ORDER, entry.loadProfiles),
  };
}

/** The `capabilities` values that denote a generation MODE, as the memory matrix's `modesFor` reads
 *  them. `style_variations` is deliberately absent: it is not a catalog capability, it is a request
 *  shape the route registry can witness, so it can never be part of a model's catalog reach. */
const GENERATION_CAPABILITIES = new Set([
  "text_to_image",
  "edit_image",
  "image_to_image",
  "image_inpaint",
  "image_detail",
  "character_image",
]);

/** A `&[&str]` Rust const, read as text. The matrix generator reads the same two pose-family lists the
 *  same way; `image_jobs/base.rs` compiles only under macOS or `backend-candle`, so a text read is the
 *  only way for a portable script to see them at all. */
export function rustStringSlice(source, name) {
  const match = source.match(
    new RegExp(`const\\s+${name}:\\s*&\\[&str\\]\\s*=\\s*&\\[([\\s\\S]*?)\\];`),
  );
  if (!match) throw new Error(`could not derive ${name} from image_jobs/base.rs`);
  return new Set([...match[1].matchAll(/"([^"]+)"/g)].map((entry) => entry[1]));
}

/**
 * What the CATALOG can actually ask of this model on this backend.
 *
 * The third source of truth, alongside the engine contract (capability) and the route witnesses
 * (reach). The route registry's rules are coarse — several use `ALL_MODES`, so their witnesses claim
 * `image_detail` for a model whose catalog entry has no detail capability at all — and a declaration
 * that inherited that coarseness would advertise cells the catalog never routes. Derived exactly as
 * the matrix's `modesFor`/`overlaysFor` derive a cell's axes, so the projection cannot declare a
 * coordinate the matrix has no cell for.
 */
export function catalogAxes(model, backend, poseFamilies) {
  const modes = (model.capabilities ?? []).filter((capability) =>
    GENERATION_CAPABILITIES.has(capability),
  );
  // Tier axis, derived exactly as the matrix's `tiersFor` derives it. Candle advertises a tier only
  // through `vramGbByTier`; MLX also counts download variants and a single-dense `quantize`
  // declaration. `tiersFor`'s InstantID override is not mirrored: `instantid` publishes no engine
  // memory contract, so no projected row can ever reach it.
  const measured = Object.keys(model[backend]?.vramGbByTier ?? {});
  const downloads = (model.downloads ?? [])
    .map((download) => download.variant)
    .filter((variant) => typeof variant === "string" && /^(bf16|fp16|q\d+|nvfp4|int\d+)/.test(variant));
  const dense =
    model[backend]?.quantize === 4 ? ["q4"] : model[backend]?.quantize === 8 ? ["q8"] : [];
  const advertised =
    backend === "candle" && measured.length ? measured : [...measured, ...downloads, ...dense];
  const overlays = ["none"];
  if (model.loraCompatibility) overlays.push("lora");
  if (poseFamilies[backend]?.has(model.id)) overlays.push("control");
  if ((model.capabilities ?? []).includes("character_image")) overlays.push("identity");
  return {
    modes: new Set(orderedBy(MODE_ORDER, modes)),
    overlays: new Set(orderedBy(OVERLAY_ORDER, overlays)),
    // `tiersFor` falls back to the pseudo-tier `["default"]` when a catalog entry advertises none.
    // Left as-is rather than widened: a model with no advertised tier has no cell at `bf16`/`q4`/`q8`
    // for a declaration to describe, so every engine tier is correctly reported as unreachable.
    tiers: new Set(advertised.filter((tier) => tier !== "int8-convrot")),
  };
}

/** SceneWorks model id -> engine provider id, parsed from the worker's static routing table. */
export function parseEngineModelTable(source) {
  const table = source.match(/pub\(crate\) const MODEL_TABLE:[\s\S]*?=\s*&\[([\s\S]*?)\n\];/);
  if (!table) throw new Error("could not locate MODEL_TABLE in engines.rs");
  const routes = new Map();
  for (const row of table[1].matchAll(/ModelRow\s*\{([\s\S]*?)\n\s*\},/g)) {
    const model = row[1].match(/sceneworks_id:\s*"([^"]+)"/)?.[1];
    const engine = row[1].match(/engine_id:\s*"([^"]+)"/)?.[1];
    if (model && engine) routes.set(model, engine);
  }
  if (routes.size === 0) throw new Error("MODEL_TABLE parsed to zero routes");
  return routes;
}

/**
 * The strict-control engine ids, from the worker table that IS their source of truth.
 *
 * A control provider has no `MODEL_TABLE` row of its own — it is reached as an OVERLAY on its base
 * model's route, which is why the manifest declares it as a `runtimeProvider` split on the base
 * model's contract rather than as a separate entry. Reading the ids from `STRICT_CONTROL_ENGINES`
 * rather than pattern-matching every `*_control` string keeps the join anchored on code: a provider
 * only becomes control-hostable here if the worker actually registers it as one.
 */
export function parseStrictControlEngines(source) {
  const table = source.match(/const STRICT_CONTROL_ENGINES:\s*&\[[^\]]*\]\s*=\s*&\[([\s\S]*?)\n\];/);
  if (!table) throw new Error("could not locate STRICT_CONTROL_ENGINES in strict_control.rs");
  const ids = [...table[1].matchAll(/engine_id:\s*"([^"]+)"/g)].map((item) => item[1]);
  if (ids.length === 0) throw new Error("STRICT_CONTROL_ENGINES parsed to zero engines");
  return new Set(ids);
}

/**
 * The MLX lanes whose declaration rows MUST carry `requestContexts`.
 *
 * `memory_route_registry.rs` computes `requires_request_context` as
 * `matching_rules(selector).any(|rule| !rule.legacy_shaping)`, and on that path
 * `mlx_request_implementation_matches` returns `Err` — not `false` — for a row with no
 * `requestContexts`. A malformed row fails CLOSED: the whole load becomes `Refused + Eager`. So a
 * coverage-only projected row on such a lane does not merely fail to help, it takes the production
 * deferred load away (measured: the Krea MLX ladder dropped to EagerMaterialization).
 *
 * `requestContexts` is a provider-owned request predicate — reference counts, PiD, phase presence —
 * that no capability dump publishes. There is nothing here to project, so these lanes are skipped and
 * reported. Candle is unaffected: its request-strategy reader returns `Ok(false)` for a row with no
 * `requestContexts`, which is inert rather than malformed.
 */
export function parseRequestContextLanes(source) {
  const table = source.match(/const RULES: &\[MemoryRouteRule\] = &\[([\s\S]*?)\n\];/);
  if (!table) throw new Error("could not locate RULES in memory_route_registry.rs");
  const lanes = new Set();
  let seen = 0;
  for (const row of table[1].matchAll(/MemoryRouteRule \{([\s\S]*?)\n {4}\},/g)) {
    seen += 1;
    const backend = row[1].match(/backend: MemoryRouteBackend::(\w+)/)?.[1];
    const provider = row[1].match(/provider: "([a-z0-9_]+)"/)?.[1];
    const legacy = row[1].match(/legacy_shaping: (true|false)/)?.[1];
    if (!backend || !provider || !legacy) {
      throw new Error(`memory-route rule ${seen} is under-keyed`);
    }
    if (backend === "Mlx" && legacy === "false") lanes.add(`mlx:${provider}`);
  }
  if (seen === 0) throw new Error("RULES parsed to zero memory-route rules");
  return lanes;
}

/** Every (provider or runtimeProvider) a model's backend contract already declares. */
export function declaredProviders(model, backend) {
  const contract = model?.[backend]?.memoryStrategyContract;
  if (!contract) return new Set();
  return new Set(
    (contract.implementations ?? []).map(
      (implementation) => implementation.runtimeProvider ?? contract.provider,
    ),
  );
}

/** Overlay -> owning runtimeProvider, from the rows a contract already declares. The epic's control
 *  splits live in this map (`z_image_turbo` owns none/lora, `z_image_turbo_control` owns control), and
 *  a generated row must never claim an overlay another runtime provider owns: `providerFor` in the
 *  matrix generator throws on a contract that declares two runtime providers for one overlay. */
export function overlayOwners(model, backend) {
  const contract = model?.[backend]?.memoryStrategyContract;
  const owners = new Map();
  if (!contract) return owners;
  for (const implementation of contract.implementations ?? []) {
    const provider = implementation.runtimeProvider ?? contract.provider;
    for (const overlay of implementation.overlays ?? []) {
      if (!owners.has(overlay)) owners.set(overlay, new Set());
      owners.get(overlay).add(provider);
    }
  }
  return owners;
}

/**
 * A deliberate withhold: the manifest declaring LESS than the engine dumps, on purpose.
 *
 * Read from `models[].<backend>.memoryDeclarationWithhold` so the reason travels with the row it
 * constrains instead of living in a table over here that nobody editing the manifest would see.
 * Shape: `{ "rungs": ["bounded_decode"], "story": "SC-15525", "reason": "..." }`, or `"rungs": "all"`
 * to withhold the whole backend. The generator honors it and REPORTS it; it never overrides one.
 */
export function withheldRungs(model, backend) {
  const declaration = model?.[backend]?.memoryDeclarationWithhold;
  if (!declaration) return null;
  const rungs = declaration.rungs;
  if (rungs === "all") return { rungs: "all", declaration };
  if (!Array.isArray(rungs) || rungs.length === 0) {
    throw new Error(
      `${model.id}:${backend} memoryDeclarationWithhold.rungs must be "all" or a non-empty array`,
    );
  }
  for (const rung of rungs) {
    if (!RUNG_ORDER.includes(rung)) {
      throw new Error(`${model.id}:${backend} memoryDeclarationWithhold names unknown rung ${rung}`);
    }
  }
  return { rungs: new Set(rungs), declaration };
}

/**
 * Choose the ONE model that hosts a provider's generated declaration.
 *
 * Precedence, most specific first:
 *   1. the model whose routing-table engine id IS this provider (the canonical entry);
 *   2. a model that already declares this provider (an alias or control split the epic established);
 *   3. for a registered strict-control provider, the primary host of its base provider;
 *   4. any routed model carrying a block for this backend.
 * Within a tier of that precedence, manifest order decides, so the choice is stable across runs.
 *
 * Only `type: "image"` entries are eligible, because that is the only shape the reconciliation's
 * `manifestContracts` reads — a video provider cannot be healed through this file at all.
 */
export function resolveHost({ backend, provider, models, routes, controlEngines, hostsByProvider }) {
  const eligible = models.filter((model) => model.type === "image" && model[backend]);
  const routed = eligible.filter((model) => routes.get(model.id) === provider);
  const canonical = routed.find((model) => model.id === provider);
  if (canonical) return { model: canonical, via: "canonical-route" };
  const declaring = eligible.find((model) => declaredProviders(model, backend).has(provider));
  if (declaring) return { model: declaring, via: "existing-declaration" };
  if (controlEngines.has(provider)) {
    const base = provider.replace(/_control$/, "");
    const baseHost = hostsByProvider.get(`${backend}:${base}`);
    if (baseHost) return { model: baseHost.model, via: "strict-control-overlay" };
  }
  if (routed.length) return { model: routed[0], via: "routing-table" };
  return null;
}

/**
 * The generated implementation rows for one (backend, provider) on its host model.
 *
 * Returns `{ rows, skipped }`. `skipped` records rungs the engine implements that this projection
 * deliberately does not write — a hand-authored row already covering the tier, or a declared
 * withhold — so the CLI can report them instead of them vanishing.
 */
export function projectProviderRows({
  backend,
  provider,
  contractProvider,
  host,
  engine,
  witnesses,
  withhold,
  axes,
}) {
  const rows = [];
  const skipped = [];
  const existing = host?.[backend]?.memoryStrategyContract;
  const owners = overlayOwners(host, backend);
  const declaredTiers = (rung) =>
    new Set(
      (existing?.implementations ?? [])
        .filter(
          (implementation) =>
            (implementation.runtimeProvider ?? existing.provider) === provider &&
            implementation.rung === rung,
        )
        .flatMap((implementation) => implementation.tiers ?? []),
    );
  // Never claim an overlay another runtime provider on this same contract owns: one overlay, one
  // runtime provider, or `providerFor` in the matrix generator refuses to resolve the lane at all.
  const claimable = (overlays) =>
    overlays.filter((overlay) => {
      const owner = owners.get(overlay);
      return !owner || owner.has(provider);
    });
  for (const [rung, tiers] of engine.rungs) {
    if (withhold && (withhold.rungs === "all" || withhold.rungs.has(rung))) {
      skipped.push({ rung, tiers, reason: "withheld", declaration: withhold.declaration });
      continue;
    }
    const covered = declaredTiers(rung);
    const missing = tiers.filter((tier) => !covered.has(tier));
    if (missing.length === 0) {
      skipped.push({ rung, tiers, reason: "already-declared" });
      continue;
    }
    // One row per distinct witnessed coordinate set, tiers grouped under it. A single row unioning
    // the coordinates of several tiers would claim every tier x mode x overlay cross-product,
    // including cells no tier actually witnesses.
    const bySignature = new Map();
    const unreachable = new Map();
    const unreachableAt = (tier, reason) => {
      if (!unreachable.has(reason)) unreachable.set(reason, []);
      unreachable.get(reason).push(tier);
    };
    for (const tier of missing) {
      if (!axes.tiers.has(tier)) {
        unreachableAt(tier, "tier-not-advertised");
        continue;
      }
      const coordinates = witnessCoordinatesForTier(witnesses, tier);
      if (!coordinates) {
        unreachableAt(tier, "no-route-witness");
        continue;
      }
      // engine capability AND route witness AND catalog reach. Any of the three empty means there is
      // no coordinate to declare here.
      const modes = coordinates.modes.filter((mode) => axes.modes.has(mode));
      const overlays = claimable(coordinates.overlays).filter((overlay) => axes.overlays.has(overlay));
      const loadProfiles = coordinates.loadProfiles.filter((profile) =>
        overlays.includes(LOAD_PROFILE_OVERLAYS.get(profile)),
      );
      if (modes.length === 0 || overlays.length === 0 || loadProfiles.length === 0) {
        unreachableAt(tier, "no-catalog-coordinate");
        continue;
      }
      const scoped = { modes, overlays, loadProfiles };
      const signature = JSON.stringify(scoped);
      if (!bySignature.has(signature)) bySignature.set(signature, { scoped, tiers: [] });
      bySignature.get(signature).tiers.push(tier);
    }
    for (const [reason, tiers] of [...unreachable].sort(([left], [right]) =>
      left.localeCompare(right),
    )) {
      skipped.push({ rung, tiers, reason });
    }
    for (const [, { scoped, tiers: rowTiers }] of [...bySignature].sort(([left], [right]) =>
      left.localeCompare(right),
    )) {
      const row = { rung };
      if (provider !== contractProvider) row.runtimeProvider = provider;
      row.tiers = rowTiers;
      row.modes = scoped.modes;
      row.overlays = scoped.overlays;
      row.loadProfiles = scoped.loadProfiles;
      row.source = `config/engine-capabilities/capabilities.${backend}.json#memoryContracts/${provider}`;
      rows.push(row);
    }
  }
  return { rows, skipped };
}

/**
 * Plan the whole projection: which host model's backend contract gains which generated rows.
 *
 * Pure — it reads the parsed manifest and the dumps and returns a plan plus a report. The text
 * surgery that applies the plan lives in `applyProjection`, so the ordering-stability and
 * comment-preservation concerns stay separable from the semantics.
 */
export function planProjection({
  manifest,
  engineFacts,
  enginesSource,
  strictControlSource,
  imageRoutingSource,
  routeRegistrySource,
}) {
  const models = manifest.models ?? [];
  const routes = parseEngineModelTable(enginesSource);
  const controlEngines = parseStrictControlEngines(strictControlSource);
  const requestContextLanes = parseRequestContextLanes(routeRegistrySource);
  const poseFamilies = {
    mlx: rustStringSlice(imageRoutingSource, "WIRED_MLX_POSE_FAMILIES"),
    candle: rustStringSlice(imageRoutingSource, "WIRED_CANDLE_POSE_FAMILIES"),
  };
  const inventory = engineContractInventory(engineFacts);
  const witnessIndex = routeWitnessInventory(engineFacts);

  // Base providers first, so a strict-control provider can inherit its base's chosen host.
  const keys = [...inventory.keys()].sort();
  const baseKeys = keys.filter((key) => !controlEngines.has(key.split(":")[1]));
  const controlKeys = keys.filter((key) => controlEngines.has(key.split(":")[1]));

  const hostsByProvider = new Map();
  const plans = new Map();
  const unhosted = [];
  const withholds = [];
  const skipped = [];

  for (const key of [...baseKeys, ...controlKeys]) {
    const engine = inventory.get(key);
    const { backend, provider } = engine;
    // A provider that reaches nothing above `resident` has no declaration to project and cannot
    // appear on the engine_manifest leg either, so it is not an unhosted finding.
    if (engine.rungs.size === 0) continue;
    const host = resolveHost({
      backend,
      provider,
      models,
      routes,
      controlEngines,
      hostsByProvider,
    });
    if (!host) {
      unhosted.push({ backend, provider, rungs: [...engine.rungs.keys()] });
      continue;
    }
    // Recorded BEFORE the request-context skip below, so a strict-control provider can still inherit
    // its base's host even when the base lane itself takes no projected row.
    hostsByProvider.set(key, host);
    if (requestContextLanes.has(key)) {
      for (const [rung, tiers] of engine.rungs) {
        skipped.push({
          backend,
          provider,
          modelId: host.model.id,
          rung,
          tiers,
          reason: "requires-request-contexts",
        });
      }
      continue;
    }
    const existing = host.model[backend]?.memoryStrategyContract;
    const contractProvider = existing?.provider ?? routes.get(host.model.id) ?? provider;
    const withhold = withheldRungs(host.model, backend);
    if (withhold) {
      withholds.push({
        backend,
        provider,
        modelId: host.model.id,
        rungs: withhold.rungs === "all" ? "all" : [...withhold.rungs],
        declaration: withhold.declaration,
      });
    }
    const projected = projectProviderRows({
      backend,
      provider,
      contractProvider,
      host: host.model,
      engine,
      witnesses: witnessIndex.get(key),
      withhold,
      axes: catalogAxes(host.model, backend, poseFamilies),
    });
    for (const entry of projected.skipped) {
      skipped.push({ backend, provider, modelId: host.model.id, ...entry });
    }
    if (projected.rows.length === 0) continue;
    const planKey = `${host.model.id}:${backend}`;
    if (!plans.has(planKey)) {
      plans.set(planKey, {
        modelId: host.model.id,
        backend,
        contractProvider,
        hasContract: Boolean(existing),
        rows: [],
        providers: [],
      });
    }
    const plan = plans.get(planKey);
    plan.providers.push({ provider, via: host.via });
    plan.rows.push(...projected.rows);
  }

  // Manifest order for the plans, and ladder-then-provider order for the rows inside each, so the
  // rendered output is a function of the inputs alone.
  const modelIndex = new Map(models.map((model, index) => [model.id, index]));
  const ordered = [...plans.values()].sort(
    (left, right) =>
      modelIndex.get(left.modelId) - modelIndex.get(right.modelId) ||
      left.backend.localeCompare(right.backend),
  );
  for (const plan of ordered) {
    plan.rows.sort(
      (left, right) =>
        RUNG_ORDER.indexOf(left.rung) - RUNG_ORDER.indexOf(right.rung) ||
        (left.runtimeProvider ?? "").localeCompare(right.runtimeProvider ?? "") ||
        left.tiers.join(",").localeCompare(right.tiers.join(",")),
    );
    plan.providers.sort((left, right) => left.provider.localeCompare(right.provider));
  }
  return { plans: ordered, unhosted, withholds, skipped };
}

// --- jsonc text surgery -------------------------------------------------------------------------
// The manifest is 21k lines of hand-written comments, so the projection EDITS text rather than
// reserializing a parsed tree: a reserialize would drop every comment in the file, which is the one
// thing the story says must survive. These helpers walk the jsonc well enough to find the exact span
// of a key's value.

function skipTrivia(body, index) {
  for (;;) {
    while (index < body.length && /\s/.test(body[index])) index += 1;
    if (body[index] === "/" && body[index + 1] === "/") {
      while (index < body.length && body[index] !== "\n") index += 1;
      continue;
    }
    if (body[index] === "/" && body[index + 1] === "*") {
      index += 2;
      while (index < body.length && !(body[index] === "*" && body[index + 1] === "/")) index += 1;
      index += 2;
      continue;
    }
    return index;
  }
}

function scanString(body, index) {
  let cursor = index + 1;
  while (cursor < body.length) {
    if (body[cursor] === "\\") {
      cursor += 2;
      continue;
    }
    if (body[cursor] === '"') return cursor + 1;
    cursor += 1;
  }
  throw new Error(`unterminated string at ${index}`);
}

/** End index (exclusive) of the value starting at `index`. */
export function scanValue(body, index) {
  const char = body[index];
  if (char === '"') return scanString(body, index);
  if (char === "{" || char === "[") {
    const close = char === "{" ? "}" : "]";
    let cursor = index + 1;
    for (;;) {
      cursor = skipTrivia(body, cursor);
      if (body[cursor] === close) return cursor + 1;
      if (body[cursor] === "," ) {
        cursor += 1;
        continue;
      }
      if (body[cursor] === '"' && char === "{") {
        cursor = scanString(body, cursor);
        cursor = skipTrivia(body, cursor);
        if (body[cursor] !== ":") throw new Error(`expected ':' at ${cursor}`);
        cursor = skipTrivia(body, cursor + 1);
      }
      cursor = scanValue(body, cursor);
    }
  }
  let cursor = index;
  while (cursor < body.length && !/[,}\]\s]/.test(body[cursor])) cursor += 1;
  return cursor;
}

/** Direct entries of the object whose `{` is at `start`. */
export function objectEntries(body, start) {
  if (body[start] !== "{") throw new Error(`expected '{' at ${start}`);
  const entries = [];
  let cursor = start + 1;
  for (;;) {
    cursor = skipTrivia(body, cursor);
    if (body[cursor] === "}") return { entries, end: cursor + 1 };
    if (body[cursor] === ",") {
      cursor += 1;
      continue;
    }
    const keyStart = cursor;
    const keyEnd = scanString(body, cursor);
    const key = JSON.parse(body.slice(keyStart, keyEnd));
    cursor = skipTrivia(body, keyEnd);
    if (body[cursor] !== ":") throw new Error(`expected ':' after ${key} at ${cursor}`);
    const valueStart = skipTrivia(body, cursor + 1);
    const valueEnd = scanValue(body, valueStart);
    entries.push({ key, keyStart, valueStart, valueEnd });
    cursor = valueEnd;
  }
}

function entry(body, start, key) {
  return objectEntries(body, start).entries.find((item) => item.key === key) ?? null;
}

/** Every top-level `models[]` object span, by model id. The array is located by walking the document
 *  root rather than by searching for the string `"models"`, which appears in prose comments first. */
export function modelSpans(body) {
  const root = skipTrivia(body, 0);
  const modelsEntry = entry(body, root, "models");
  if (!modelsEntry) throw new Error('could not find "models" in the manifest');
  let cursor = modelsEntry.valueStart;
  if (body[cursor] !== "[") throw new Error('"models" is not an array');
  const spans = new Map();
  cursor += 1;
  for (;;) {
    cursor = skipTrivia(body, cursor);
    if (body[cursor] === "]") return spans;
    if (body[cursor] === ",") {
      cursor += 1;
      continue;
    }
    const start = cursor;
    const end = scanValue(body, start);
    const idEntry = entry(body, start, "id");
    if (!idEntry) throw new Error(`model object at ${start} has no id`);
    spans.set(JSON.parse(body.slice(idEntry.valueStart, idEntry.valueEnd)), { start, end });
    cursor = end;
  }
}

/** The index just past a value, skipped forward over a trailing same-line `//` comment. */
function endOfLineValue(body, index) {
  let cursor = index;
  while (cursor < body.length && (body[cursor] === " " || body[cursor] === "\t")) cursor += 1;
  if (body[cursor] === "/" && body[cursor + 1] === "/") {
    while (cursor < body.length && body[cursor] !== "\n") cursor += 1;
    return cursor;
  }
  return index;
}

const indentOf = (body, index) => {
  const lineStart = body.lastIndexOf("\n", index) + 1;
  return body.slice(lineStart, index).match(/^[ \t]*/)[0];
};

/** One implementation row, rendered in the manifest's established style: object per line, scalar
 *  arrays inline. Key order is fixed by construction in `projectProviderRows`. */
export function renderRow(row, indent) {
  const lines = Object.entries(row).map(([key, value]) => {
    const rendered = Array.isArray(value)
      ? `[${value.map((item) => JSON.stringify(item)).join(", ")}]`
      : JSON.stringify(value);
    return `${indent}  ${JSON.stringify(key)}: ${rendered}`;
  });
  return [`${indent}{`, ...lines.map((line) => `${line},`).slice(0, -1), lines.at(-1), `${indent}}`]
    .join("\n");
}

/** The banner every generated region opens with. Wraps the provider list so a block hosting several
 *  runtime providers still produces stable, readable lines. */
export function renderBanner(indent, providers) {
  const lines = [`Engine-derived rung x tier coverage for ${providers.join(", ")}.`];
  lines.push("Re-run scripts/generate-manifest-memory-declarations.mjs to update;");
  lines.push("hand edits inside this region are overwritten.");
  return lines.map((line) => `${indent}// ${line}\n`).join("");
}

/** The marked generated region: the two marker comments and the rows between them. */
export function renderGeneratedRegion(rows, indent, providers) {
  const banner = `${indent}${GENERATED_BEGIN}\n${renderBanner(indent, providers)}`;
  return `${banner}${rows.map((row) => renderRow(row, indent)).join(",\n")}\n${indent}${GENERATED_END}`;
}

/** Strip a previously generated region (with its trailing or leading comma) from an
 *  `implementations` array body, leaving hand-authored rows byte-identical. */
export function stripGeneratedRegion(text) {
  const begin = text.indexOf(GENERATED_BEGIN);
  if (begin < 0) return text;
  const end = text.indexOf(GENERATED_END, begin);
  if (end < 0) throw new Error("generated region has no END marker");
  const after = end + GENERATED_END.length;
  const head = text.slice(0, begin).replace(/,\s*$/, "");
  const tail = text.slice(after).replace(/^\s*,/, "");
  return `${head}${tail}`;
}

/**
 * Apply the plan to the manifest text.
 *
 * For a backend block that already has a `memoryStrategyContract`, the generated region replaces any
 * previous one at the tail of `implementations`. For a backend block that has none, a whole
 * `memoryStrategyContract` is inserted after the block's last entry.
 */
export function applyProjection(body, plans) {
  const spans = modelSpans(body);
  // Apply back-to-front so earlier offsets stay valid.
  const edits = [];
  for (const plan of plans) {
    const span = spans.get(plan.modelId);
    if (!span) throw new Error(`plan targets unknown model ${plan.modelId}`);
    const backendEntry = entry(body, span.start, plan.backend);
    if (!backendEntry) throw new Error(`${plan.modelId} has no ${plan.backend} block`);
    const contractEntry = entry(body, backendEntry.valueStart, "memoryStrategyContract");
    if (contractEntry) {
      const implementations = entry(body, contractEntry.valueStart, "implementations");
      if (!implementations) {
        throw new Error(`${plan.modelId}:${plan.backend} contract has no implementations array`);
      }
      const open = implementations.valueStart;
      const close = implementations.valueEnd - 1;
      const inner = stripGeneratedRegion(body.slice(open + 1, close));
      const arrayIndent = indentOf(body, implementations.keyStart);
      const region = renderGeneratedRegion(
        plan.rows,
        `${arrayIndent}  `,
        plan.providers.map((item) => item.provider),
      );
      const head = inner.replace(/\s*$/, "");
      const separator = head.trim() ? ",\n" : "\n";
      edits.push({
        start: open,
        end: implementations.valueEnd,
        text: `[${head}${separator}${region}\n${arrayIndent}]`,
      });
      continue;
    }
    const backendObject = objectEntries(body, backendEntry.valueStart);
    const last = backendObject.entries.at(-1);
    if (!last) throw new Error(`${plan.modelId}:${plan.backend} block is empty`);
    const indent = indentOf(body, last.keyStart);
    // No inner markers here: the whole contract is generated, so ONE marked region wraps it. Nesting
    // two regions would leave `clearProjection` pairing the outer BEGIN with the inner END.
    const region = plan.rows.map((row) => renderRow(row, `${indent}    `)).join(",\n");
    const inserted =
      `,\n${indent}${GENERATED_BEGIN}\n` +
      renderBanner(indent, plan.providers.map((item) => item.provider)) +
      `${indent}"memoryStrategyContract": {\n` +
      `${indent}  "abi": 1,\n` +
      `${indent}  "provider": ${JSON.stringify(plan.contractProvider)},\n` +
      `${indent}  "implementations": [\n${region}\n${indent}  ]\n` +
      // No comma after the closing brace: the generated contract becomes the block's LAST entry, and
      // the comma that joins it to the previous one is already emitted above.
      `${indent}}\n` +
      `${indent}${GENERATED_END}`;
    // Past any trailing same-line comment on the entry we append after, so the inserted comma cannot
    // land inside somebody's `// note` and change what that note says.
    edits.push({ start: endOfLineValue(body, last.valueEnd), end: endOfLineValue(body, last.valueEnd), text: inserted });
  }
  edits.sort((left, right) => right.start - left.start);
  let out = body;
  for (const edit of edits) {
    out = out.slice(0, edit.start) + edit.text + out.slice(edit.end);
  }
  return out;
}

/** Remove every generated region from the manifest text — the inverse of `applyProjection`, and how
 *  a rerun reaches a clean base before re-emitting (which is what makes the generator idempotent
 *  even when a provider stops being hosted, or a whole contract stops being generated). */
export function clearProjection(body) {
  let out = body;
  for (;;) {
    const begin = out.indexOf(GENERATED_BEGIN);
    if (begin < 0) return out;
    const end = out.indexOf(GENERATED_END, begin);
    if (end < 0) throw new Error("generated region has no END marker");
    const lineStart = out.lastIndexOf("\n", begin) + 1;
    let after = end + GENERATED_END.length;
    // A generated whole-contract insert owns the comma that introduced it; a generated region inside
    // an `implementations` array is introduced by a comma on the preceding hand-authored row.
    const head = out.slice(0, lineStart).replace(/,(\s*)$/, "$1");
    while (after < out.length && out[after] === "\n") after += 1;
    out = `${head}${out.slice(after)}`;
  }
}

/** Project the dumps into a manifest body. Pure text in, text out. */
export function projectManifestBody({
  body,
  engineFacts,
  enginesSource,
  strictControlSource,
  imageRoutingSource,
  routeRegistrySource,
}) {
  // Start from the manifest WITHOUT any previous generated region, so what a rerun reads as
  // "already declared by hand" never includes what the last run wrote. Without this the projection
  // is not idempotent: generated rows would be treated as hand coverage and suppress themselves.
  const base = clearProjection(body);
  const manifest = JSON.parse(stripJsoncComments(base));
  const plan = planProjection({
    manifest,
    engineFacts,
    enginesSource,
    strictControlSource,
    imageRoutingSource,
    routeRegistrySource,
  });
  return { body: applyProjection(base, plan.plans), ...plan };
}
