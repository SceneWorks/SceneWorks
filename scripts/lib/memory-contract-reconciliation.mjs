import { createHash } from "node:crypto";

const RUNGS = Object.freeze([
  "resident",
  "staged_residency",
  "bounded_decode",
  "bounded_attention",
  "bounded_transformer_residency",
]);

const AXES = Object.freeze([
  "leg",
  "direction",
  "backend",
  "provider",
  "modelId",
  "familyStory",
  "mode",
  "tier",
  "overlay",
  "rung",
  "selectorDigest",
]);

const laneKey = (backend, provider) => `${backend}:${provider}`;
const sortedUnique = (items) => [...new Set(items)].sort();

function mismatch(fields) {
  return Object.fromEntries(AXES.map((axis) => [axis, fields[axis] ?? null]));
}

export function reconciliationMismatchKey(value) {
  return JSON.stringify(Object.fromEntries(AXES.map((axis) => [axis, value[axis] ?? null])));
}

function requireObject(value, at) {
  if (!value || Array.isArray(value) || typeof value !== "object") {
    throw new Error(`${at} must be an object`);
  }
  return value;
}

function engineContractIndex(engineFacts) {
  const out = new Map();
  const seenBackends = new Set();
  for (const document of engineFacts) {
    requireObject(document, "engine capability facts");
    const backend = document.backend;
    if (!["mlx", "candle"].includes(backend)) {
      throw new Error(`engine capability facts name unknown backend ${JSON.stringify(backend)}`);
    }
    if (seenBackends.has(backend)) throw new Error(`duplicate engine capability facts backend ${backend}`);
    seenBackends.add(backend);
    if (!/^[0-9a-f]{40}$/.test(document.generatedFrom?.inferenceRevision ?? "")) {
      throw new Error(`engine capability facts for ${backend} have no valid inference revision`);
    }
    if (!Array.isArray(document.memoryContracts) || document.memoryContracts.length === 0) {
      throw new Error(`engine capability facts for ${backend} have no memoryContracts inventory`);
    }
    const providers = new Set();
    for (const contract of document.memoryContracts) {
      requireObject(contract, `${backend} memory contract`);
      if (typeof contract.id !== "string" || !contract.id) throw new Error(`${backend} memory contract has no provider id`);
      if (providers.has(contract.id)) throw new Error(`duplicate ${backend} memory-contract provider ${contract.id}`);
      providers.add(contract.id);
      if (!/^sha256:[0-9a-f]{64}$/.test(contract.selectorDigest ?? "")) {
        throw new Error(`${backend}:${contract.id} has no canonical selectorDigest`);
      }
      if (!Array.isArray(contract.surfaces) || contract.surfaces.length === 0) {
        throw new Error(`${backend}:${contract.id} publishes no contract surfaces`);
      }
      const selectors = new Set();
      const implemented = new Set();
      const structural = new Set();
      const deferred = new Set();
      const implementedByTier = new Map();
      const structuralByTier = new Map();
      const deferredByTier = new Map();
      for (const surface of contract.surfaces) {
        const selector = surface.selector;
        const selectorKey = `${selector?.tier}:${selector?.offloadPolicy}:${selector?.loadShape}`;
        if ([selector?.tier, selector?.offloadPolicy, selector?.loadShape].some((part) => !part)) {
          throw new Error(`${backend}:${contract.id} has an under-keyed selector`);
        }
        if (selectors.has(selectorKey)) throw new Error(`${backend}:${contract.id} repeats selector ${selectorKey}`);
        selectors.add(selectorKey);
        for (const [field, target, byTier] of [
          ["implementedRungs", implemented, implementedByTier],
          ["structurallyNotApplicableRungs", structural, structuralByTier],
          ["deferredMaterializationRungs", deferred, deferredByTier],
        ]) {
          if (!Array.isArray(surface[field])) throw new Error(`${backend}:${contract.id}:${selectorKey} has no ${field}`);
          if (!byTier.has(selector.tier)) byTier.set(selector.tier, new Set());
          for (const rung of surface[field]) {
            if (!RUNGS.includes(rung)) throw new Error(`${backend}:${contract.id}:${selectorKey} names unknown rung ${rung}`);
            target.add(rung);
            byTier.get(selector.tier).add(rung);
          }
        }
      }
      const canonicalSurfaces = contract.surfaces.map((surface) => ({
        selector: {
          tier: surface.selector.tier,
          offloadPolicy: surface.selector.offloadPolicy,
          loadShape: surface.selector.loadShape,
        },
        implementedRungs: surface.implementedRungs,
        structurallyNotApplicableRungs: surface.structurallyNotApplicableRungs,
        deferredMaterializationRungs: surface.deferredMaterializationRungs,
      }));
      const expectedDigest = `sha256:${createHash("sha256")
        .update(JSON.stringify(canonicalSurfaces))
        .digest("hex")}`;
      if (contract.selectorDigest !== expectedDigest) {
        throw new Error(
          `${backend}:${contract.id} selectorDigest does not bind its published contract surfaces`,
        );
      }
      out.set(laneKey(backend, contract.id), {
        backend,
        provider: contract.id,
        composed: contract.composed === true,
        selectorDigest: contract.selectorDigest,
        implemented,
        structural,
        deferred,
        implementedByTier,
        structuralByTier,
        deferredByTier,
      });
    }
  }
  if (!["mlx", "candle"].every((backend) => seenBackends.has(backend))) {
    throw new Error("memory-contract reconciliation requires both MLX and Candle capability facts");
  }
  return out;
}

export function validateMemoryContractFacts(engineFacts) {
  const providers = engineContractIndex(engineFacts).size;
  routeEligibilityFromEngineFacts(engineFacts);
  return providers;
}

const BESPOKE_WAIVER_FIELDS = Object.freeze([
  "providerId",
  "crateName",
  "owner",
  "reason",
  "contractPath",
  "verificationPath",
]);

/**
 * Audit upstream's typed disposition for a real, descriptor-less Candle route.
 *
 * This is deliberately not an engine-contract waiver. It cannot create a registration, a route
 * witness, or an optimized contract surface; it only closes the topology question for a bespoke
 * worker route that is present in the production matrix but cannot truthfully implement
 * `load(id, LoadSpec)`.
 */
function validateBespokeMemoryRouteWaivers(engineFacts, manifest, cells, engine, declarations) {
  const waivers = [];
  const seen = new Set();
  for (const document of engineFacts) {
    const rows = document.bespokeMemoryRouteWaivers ?? [];
    if (!Array.isArray(rows)) {
      throw new Error(`${document.backend ?? "(unset)"} bespokeMemoryRouteWaivers must be an array`);
    }
    if (document.backend !== "candle" && rows.length) {
      throw new Error(`${document.backend} engine facts cannot publish Candle bespoke-memory waivers`);
    }
    for (const [index, waiver] of rows.entries()) {
      requireObject(waiver, `bespoke memory-route waiver ${index}`);
      for (const field of BESPOKE_WAIVER_FIELDS) {
        if (!Object.hasOwn(waiver, field)) {
          throw new Error(`bespoke memory-route waiver ${index} is under-keyed: missing ${field}`);
        }
      }
      for (const field of Object.keys(waiver)) {
        if (!BESPOKE_WAIVER_FIELDS.includes(field)) {
          throw new Error(`bespoke memory-route waiver ${index} has unknown field ${field}`);
        }
      }
      if (BESPOKE_WAIVER_FIELDS.some((field) => waiver[field] === "*")) {
        throw new Error(`bespoke memory-route waiver ${index} contains a wildcard`);
      }
      if (!/^[a-z0-9][a-z0-9_]*$/.test(waiver.providerId ?? "")) {
        throw new Error(`bespoke memory-route waiver ${index} has an invalid providerId`);
      }
      if (!/^[a-z0-9][a-z0-9-]*$/.test(waiver.crateName ?? "")) {
        throw new Error(`bespoke memory-route waiver ${index} has an invalid crateName`);
      }
      if (waiver.owner !== `candle-gen-${waiver.crateName}`) {
        throw new Error(
          `bespoke memory-route waiver ${waiver.providerId} owner does not match its crateName`,
        );
      }
      if (typeof waiver.reason !== "string" || waiver.reason.trim().length < 24) {
        throw new Error(`bespoke memory-route waiver ${waiver.providerId} has no actionable reason`);
      }
      const expectedPrefix = `crates/media/candle-gen/${waiver.owner}/src/`;
      for (const field of ["contractPath", "verificationPath"]) {
        const path = waiver[field];
        if (
          typeof path !== "string" ||
          !path.startsWith(expectedPrefix) ||
          !path.endsWith(".rs") ||
          path.includes("..")
        ) {
          throw new Error(
            `bespoke memory-route waiver ${waiver.providerId} has an invalid ${field}`,
          );
        }
      }
      if (waiver.contractPath === waiver.verificationPath) {
        throw new Error(
          `bespoke memory-route waiver ${waiver.providerId} must name distinct contract and verification paths`,
        );
      }
      if (seen.has(waiver.providerId)) {
        throw new Error(`duplicate bespoke memory-route waiver ${waiver.providerId}`);
      }
      seen.add(waiver.providerId);

      if (engine.has(laneKey("candle", waiver.providerId))) {
        throw new Error(
          `bespoke memory-route waiver ${waiver.providerId} masks an ordinary Candle provider registration`,
        );
      }
      if (declarations.some((row) => row.backend === "candle" && row.provider === waiver.providerId)) {
        throw new Error(
          `bespoke memory-route waiver ${waiver.providerId} masks an ordinary manifest contract declaration`,
        );
      }
      if ((document.memoryRouteWitnesses ?? []).some((row) => row.provider === waiver.providerId)) {
        throw new Error(
          `bespoke memory-route waiver ${waiver.providerId} masks an ordinary route witness`,
        );
      }
      const routedCells = cells.filter(
        (cell) => cell.backend === "candle" && cell.provider === waiver.providerId,
      );
      if (routedCells.length === 0) {
        throw new Error(`stale bespoke memory-route waiver ${waiver.providerId} has no production route`);
      }
      for (const cell of routedCells) {
        const model = (manifest.models ?? []).find((candidate) => candidate.id === cell.modelId);
        if (model?.type !== "image" || !model.candle) {
          throw new Error(
            `bespoke memory-route waiver ${waiver.providerId} reaches unknown Candle model ${cell.modelId}`,
          );
        }
      }
      waivers.push({ backend: "candle", ...waiver });
    }
  }
  return waivers;
}

function manifestContracts(manifest) {
  const rows = [];
  for (const model of manifest.models ?? []) {
    if (model.type !== "image") continue;
    for (const backend of ["mlx", "candle"]) {
      const contract = model[backend]?.memoryStrategyContract;
      if (!contract) continue;
      if (typeof contract.provider !== "string" || !contract.provider) {
        throw new Error(`${model.id}:${backend} memoryStrategyContract has no provider`);
      }
      for (const implementation of contract.implementations ?? []) {
        const runtimeProvider = implementation.runtimeProvider ?? contract.provider;
        if (typeof runtimeProvider !== "string" || !runtimeProvider) {
          throw new Error(`${model.id}:${backend}:${contract.provider} has an invalid runtimeProvider`);
        }
        if (!RUNGS.includes(implementation.rung)) {
          throw new Error(`${model.id}:${backend}:${contract.provider} names unknown rung ${implementation.rung}`);
        }
        if (!Array.isArray(implementation.tiers) || implementation.tiers.length === 0) {
          throw new Error(`${model.id}:${backend}:${contract.provider}:${implementation.rung} has no tiers`);
        }
        rows.push({
          backend,
          provider: runtimeProvider,
          modelId: model.id,
          rung: implementation.rung,
          tiers: sortedUnique(implementation.tiers),
          modes: sortedUnique(implementation.modes ?? []),
          overlays: sortedUnique(implementation.overlays ?? []),
          loadProfiles: sortedUnique(implementation.loadProfiles ?? []),
        });
      }
    }
  }
  return rows;
}

const ROUTE_TIERS = new Set(["bf16", "q4", "q8", "nvfp4"]);
const ROUTE_MODES = new Set([
  "text_to_image",
  "style_variations",
  "edit_image",
  "image_to_image",
  "image_inpaint",
  "image_detail",
  "character_image",
]);
const ROUTE_OVERLAYS = new Set(["none", "lora", "control", "identity"]);
// Ordered views of the two vocabularies, for the axis probes in `collectMemoryContractMismatches`
// that need to vary one axis while holding the other.
const ROUTE_MODES_LIST = Object.freeze([...ROUTE_MODES]);
const ROUTE_OVERLAYS_LIST = Object.freeze([...ROUTE_OVERLAYS]);
// Which matrix overlay cell each load profile serves. A profile can COMPOSE several load-time
// concerns, and the cell it lands in follows one precedence: identity > control > lora > none. That
// is not invented here — it is what the existing rows already encode (`lora_pid` -> `lora`, because
// PiD alone is `none` and LoRA outranks it) and what both engine dumps independently declare on the
// witnesses themselves, which this map is then used to CHECK.
//
// The four composed profiles below were missing, and a missing entry is a hard throw rather than a
// finding: at pin 931366f62 the MLX registry emits 24 witnesses across them, so the reconciliation
// crashed on a legitimate route instead of reconciling it. Adding them widens what the consistency
// check COVERS; it does not widen what satisfies a manifest declaration — `MANIFEST_ROUTE_PROFILES`
// below is deliberately left alone, so a bare `identity` cell is still served only by the plain
// identity profiles and never by `lora_ip_adapter`.
//
// sc-20799: the two PiD-composed CONTROL profiles were the same gap one lane later. The epic's
// Kolors control work made `candle_kolors_control` emit them, and the first authoritative candle
// dump at pin ebcdc7da carries 9 `single_control_pid` and 9 `lora_single_control_pid` witnesses.
// Both map to `control` here because that is what `MemoryRouteLoadProfile::overlay()` — the
// authority this map exists to CHECK against — returns for them; the witnesses independently
// declare `control` too, so the equality assertion below still does real work rather than being
// satisfied by construction. This map now covers all 14 registry profiles, so a composed profile
// can no longer blank the whole reconciliation.
const ROUTE_LOAD_PROFILES = new Map([
  ["plain", "none"],
  ["lora", "lora"],
  ["lora_pid", "lora"],
  ["single_control", "control"],
  ["single_control_pid", "control"],
  ["multi_control", "control"],
  ["lora_single_control", "control"],
  ["lora_single_control_pid", "control"],
  ["ip_adapter", "identity"],
  ["ip_adapter_pid", "identity"],
  ["lora_ip_adapter", "identity"],
  ["lora_ip_adapter_pid", "identity"],
  ["pid", "none"],
  ["identity", "identity"],
]);
// A manifest coordinate denotes the production shape it actually serves. MultiControlNet and PiD
// are distinct load profiles, not aliases for the single-control or plain matrix coordinates.
const MANIFEST_ROUTE_PROFILES = new Map([
  ["none", new Set(["plain"])],
  ["lora", new Set(["lora"])],
  ["control", new Set(["single_control"])],
  ["identity", new Set(["ip_adapter", "identity"])],
]);

/** Read the executable worker registry's dumped route witness, never Rust source text. */
export function routeEligibilityFromEngineFacts(engineFacts) {
  const rows = [];
  const seen = new Set();
  for (const document of engineFacts) {
    const backend = document.backend;
    if (!Array.isArray(document.memoryRouteWitnesses) || document.memoryRouteWitnesses.length === 0) {
      throw new Error(`engine capability facts for ${backend ?? "(unset)"} have no memoryRouteWitnesses`);
    }
    for (const [index, witness] of document.memoryRouteWitnesses.entries()) {
      requireObject(witness, `${backend} memory route witness ${index}`);
      const allowed = new Set(["provider", "tier", "mode", "overlay", "loadProfile"]);
      for (const field of Object.keys(witness)) {
        if (!allowed.has(field)) throw new Error(`${backend} memory route witness ${index} has unknown field ${field}`);
      }
      if (typeof witness.provider !== "string" || !witness.provider) {
        throw new Error(`${backend} memory route witness ${index} has no provider`);
      }
      if (!ROUTE_TIERS.has(witness.tier)) throw new Error(`${backend}:${witness.provider} has unknown route tier ${witness.tier}`);
      if (!ROUTE_MODES.has(witness.mode)) throw new Error(`${backend}:${witness.provider}:${witness.tier} has unknown route mode ${witness.mode}`);
      if (!ROUTE_OVERLAYS.has(witness.overlay)) throw new Error(`${backend}:${witness.provider}:${witness.tier}:${witness.mode} has unknown route overlay ${witness.overlay}`);
      const expectedOverlay = ROUTE_LOAD_PROFILES.get(witness.loadProfile);
      if (!expectedOverlay) throw new Error(`${backend}:${witness.provider}:${witness.tier}:${witness.mode} has unknown load profile ${witness.loadProfile}`);
      if (expectedOverlay !== witness.overlay) {
        throw new Error(
          `${backend}:${witness.provider}:${witness.tier}:${witness.mode}:${witness.loadProfile} ` +
          `belongs to overlay ${expectedOverlay}, not ${witness.overlay}`,
        );
      }
      const row = { backend, ...witness };
      const key = JSON.stringify(row);
      if (seen.has(key)) throw new Error(`duplicate memory route witness ${key}`);
      seen.add(key);
      rows.push(row);
    }
  }
  return rows.sort((left, right) => JSON.stringify(left).localeCompare(JSON.stringify(right)));
}

function planRows(plan) {
  const rows = [];
  // sc-22514: one anchor per `<modelId>:<tier>:<backend>` key replaced the provider grid.
  for (const [key, entry] of Object.entries(plan.anchors ?? {})) {
    const [modelId, , backend] = key.split(":");
    const provider = entry.provider;
    const mode = entry.mode;
    if (!backend || !provider || !mode) throw new Error(`anchor plan entry ${key} is under-keyed`);
    rows.push({ backend, provider, mode, modelId: modelId ?? null });
  }
  return [...new Map(rows.map((row) => [JSON.stringify(row), row])).values()];
}

function closureRows(closures) {
  const rows = [];
  for (const lane of Object.keys(closures.providers ?? {})) {
    const split = lane.indexOf(":");
    if (split < 1 || split === lane.length - 1) {
      throw new Error(`inference closure lane ${JSON.stringify(lane)} is not backend:provider`);
    }
    rows.push({ backend: lane.slice(0, split), provider: lane.slice(split + 1) });
  }
  return rows;
}

export function collectMemoryContractMismatches({
  engineFacts,
  manifest,
  cells,
  calibrationPlan,
  closures,
  survey,
}) {
  const engine = engineContractIndex(engineFacts);
  const routeEligibility = routeEligibilityFromEngineFacts(engineFacts);
  const declarations = manifestContracts(manifest);
  const out = [];
  // Why each coordinate was emitted, recorded at the emit site because that is the only place the
  // discriminator is still in scope. Kept OUT of the finding objects themselves so the eleven AXES
  // stay the whole identity of a coordinate: `reconciliationMismatchKey` is the dedupe key and the
  // survey leg round-trips its coordinates through `JSON.parse(key)`, both of which a twelfth field
  // would silently change. Merged onto the unique findings at the end instead. First writer wins, so
  // a coordinate reached twice keeps the reason it was first found for.
  const causes = new Map();
  const because = (entry, cause) => {
    const key = reconciliationMismatchKey(entry);
    if (!causes.has(key)) causes.set(key, cause);
    return entry;
  };

  for (const contract of engine.values()) {
    for (const [tier, implemented] of contract.implementedByTier) {
      for (const rung of [...implemented].filter((item) => item !== "resident")) {
        if (!declarations.some((row) =>
          row.backend === contract.backend &&
          row.provider === contract.provider &&
          row.rung === rung &&
          row.tiers.includes(tier)
        )) {
          // No sub-cause is available here: WHY the manifest carries no declaration is a property of
          // the projection (unhosted provider, tier not in the catalog, no route witness, MLX
          // request-context lane), which lives in `scripts/lib/manifest-memory-declarations.mjs`.
          // `triageMemoryContractMismatches` joins it back on.
          out.push(because(mismatch({
            leg: "engine_manifest",
            direction: "engine_to_manifest",
            backend: contract.backend,
            provider: contract.provider,
            tier,
            rung,
            selectorDigest: contract.selectorDigest,
          }), "undeclared_in_manifest"));
        }
      }
    }
  }

  const eligible = (backend, provider, tier, mode, overlay, declaredProfiles = []) => routeEligibility.some(
    (row) =>
      row.backend === backend &&
      row.provider === provider &&
      row.tier === tier &&
      row.mode === mode &&
      row.overlay === overlay &&
      (declaredProfiles.length
        ? declaredProfiles.includes(row.loadProfile)
        : MANIFEST_ROUTE_PROFILES.get(overlay)?.has(row.loadProfile)),
  );
  for (const row of declarations) {
    const contract = engine.get(laneKey(row.backend, row.provider));
    for (const tier of row.tiers) {
      if (!contract?.deferredByTier.get(tier)?.has(row.rung)) continue;
      const modes = row.modes.length ? row.modes : [null];
      for (const mode of modes) {
        const overlays = row.overlays.length ? row.overlays : [null];
        for (const overlay of overlays) {
          const cell = cells.find((candidate) =>
            candidate.backend === row.backend &&
            candidate.provider === row.provider &&
            candidate.modelId === row.modelId &&
            candidate.rung === row.rung &&
            candidate.mode === mode &&
            candidate.tier === tier &&
            candidate.overlay === overlay,
          );
          const routed = eligible(row.backend, row.provider, tier, mode, overlay, row.loadProfiles);
          if (!cell || !routed) {
            out.push(because(mismatch({
              leg: "manifest_route",
              direction: "manifest_to_route",
              backend: row.backend,
              provider: row.provider,
              modelId: row.modelId,
              familyStory: cell?.owningFamilyStory ?? null,
              mode,
              tier,
              overlay,
              rung: row.rung,
              selectorDigest: contract.selectorDigest,
            }), cell ? "declared_route_unwitnessed" : "declared_cell_absent"));
          }
        }
      }
    }
  }

  const plans = planRows(calibrationPlan);
  for (const row of plans) {
    if (!engine.has(laneKey(row.backend, row.provider))) {
      out.push(because(mismatch({ leg: "plan_closure_engine", direction: "plan_to_engine", ...row }), "no_engine_contract"));
    }
  }
  const closure = closureRows(closures);
  for (const row of closure) {
    if (!engine.has(laneKey(row.backend, row.provider))) {
      out.push(because(mismatch({ leg: "plan_closure_engine", direction: "closure_to_engine", ...row }), "no_engine_contract"));
    }
  }

  const surveyFamilies = survey.families ?? {};
  for (const [familyStory, family] of Object.entries(surveyFamilies)) {
    for (const [backend, verdict] of Object.entries(family.backends ?? {})) {
      const familyCells = cells.filter((cell) =>
        cell.rung === "bounded_transformer_residency" &&
        cell.owningFamilyStory === Number(familyStory) &&
        cell.backend === backend,
      );
      const scopes = verdict.implementationScopes?.length
        ? verdict.implementationScopes.map((scope) => ({
            entries: scope.entries,
            tiers: scope.tiers,
            modes: scope.modes,
            overlays: scope.overlays,
          }))
        : verdict.implementation === "none"
          ? []
          : [{
              entries: verdict.implementedEntries ?? [],
              tiers: verdict.implementedTiers,
              modes: verdict.implementedModes,
              overlays: verdict.implementedOverlays,
            }];
      const surveyCoordinates = new Map();
      // A survey scope that OMITS an axis claims every value of it — `scope.tiers ?? [cell.tier]`
      // above matches whatever the cell happens to carry. So a coordinate the survey reaches only
      // through an omitted axis is not a per-coordinate assertion the survey ever made; it is the
      // wildcard being expanded against a per-coordinate engine fact. That distinction is the whole
      // difference between "the survey is wrong here" and "the two sides are keyed differently", so
      // it is recorded rather than flattened.
      //
      // The test is PER AXIS, not per scope. A first cut asked only whether the scope omitted *some*
      // axis, which excused a contradiction on an axis the verdict had named outright: the qwen and
      // SDXL verdicts name `tiers` explicitly and omit only `modes`, and 50 coordinates where the
      // contract implements the rung at bf16 ALONE were filed "count of work: 0" on the strength of
      // the unrelated `modes` omission. So: find the axes on which the two sides actually differ,
      // and call it benign only when every one of them is an axis this scope left open.
      const claimedScopes = new Map();
      for (const cell of familyCells) {
        const matched = scopes.filter((scope) =>
          scope.entries.includes(cell.modelId) &&
          (scope.tiers ?? [cell.tier]).includes(cell.tier) &&
          (scope.modes ?? [cell.mode]).includes(cell.mode) &&
          (scope.overlays ?? [cell.overlay]).includes(cell.overlay),
        );
        const claimed = matched.length > 0;
        if (claimed) claimedScopes.set(cell, matched);
        if (claimed) surveyCoordinates.set(reconciliationMismatchKey(mismatch({
          leg: "survey_engine",
          direction: "survey_to_engine",
          backend,
          provider: cell.provider,
          modelId: cell.modelId,
          familyStory: Number(familyStory),
          mode: cell.mode,
          tier: cell.tier,
          overlay: cell.overlay,
          rung: "bounded_transformer_residency",
          selectorDigest: engine.get(laneKey(backend, cell.provider))?.selectorDigest,
        })), cell);
      }
      // Clean-base claims per (provider, modelId, tier), for the withheld-overlay probe below. Keyed
      // WITHOUT the mode axis on purpose (sc-21510): a provider's rung-4 gate is a property of the
      // loaded overlay spec, not of the request mode — FLUX.1's `structurally_streamable` requires
      // `identity.is_none() && ip_adapter.is_none()` no matter which mode routed the load, and its
      // `validate_load_contract` rejects an identity spec for `flux1_dev` outright. A dump's contract
      // surface carries neither a mode nor an overlay axis, so when the survey claims the same
      // (provider, model, tier) at clean base in ANY mode, a loaded-overlay coordinate reached only
      // through the route witnesses is the engine side over-approximating, not a survey omission.
      const cleanBaseClaims = new Set();
      for (const cell of surveyCoordinates.values()) {
        if (cell.overlay === "none") {
          cleanBaseClaims.add(`${cell.provider}:${cell.modelId}:${cell.tier}`);
        }
      }
      const engineCoordinates = new Map();
      for (const cell of familyCells) {
        const contract = engine.get(laneKey(backend, cell.provider));
        const supported =
          contract?.implementedByTier.get(cell.tier)?.has("bounded_transformer_residency") &&
          eligible(backend, cell.provider, cell.tier, cell.mode, cell.overlay);
        if (supported) engineCoordinates.set(reconciliationMismatchKey(mismatch({
          leg: "survey_engine",
          direction: "survey_to_engine",
          backend,
          provider: cell.provider,
          modelId: cell.modelId,
          familyStory: Number(familyStory),
          mode: cell.mode,
          tier: cell.tier,
          overlay: cell.overlay,
          rung: "bounded_transformer_residency",
          selectorDigest: contract.selectorDigest,
        })), cell);
      }
      /**
       * On which axes do the survey and the engine actually disagree about this cell?
       *
       * The engine supports a rung-4 coordinate when its contract implements the rung AT THE CELL'S
       * TIER and a production route reaches (tier, mode, overlay). Those are two different failures
       * and they blame different axes, so they are probed separately rather than collapsed into one
       * "unsupported" bit.
       */
      const disagreeingAxes = (cell) => {
        const contract = engine.get(laneKey(backend, cell.provider));
        if (!contract?.implementedByTier.get(cell.tier)?.has("bounded_transformer_residency")) {
          return ["tiers"];
        }
        // The rung is implemented at this tier, so the miss is in the route. Hold each of mode and
        // overlay while varying the other: whichever one has no reachable partner is the axis that
        // carries the disagreement. When neither alone explains it, the PAIR does, and both axes are
        // blamed — being able to name either one is enough to make the claim an assertion.
        const modeReaches = ROUTE_MODES_LIST.some((mode) =>
          eligible(backend, cell.provider, cell.tier, mode, cell.overlay));
        const overlayReaches = ROUTE_OVERLAYS_LIST.some((overlay) =>
          eligible(backend, cell.provider, cell.tier, cell.mode, overlay));
        const axes = [];
        if (!modeReaches) axes.push("overlays");
        if (!overlayReaches) axes.push("modes");
        return axes.length ? axes : ["modes", "overlays"];
      };
      for (const [key, cell] of surveyCoordinates) {
        if (engineCoordinates.has(key)) continue;
        const axes = disagreeingAxes(cell);
        // Benign only if EVERY disagreeing axis was left open by every scope that claimed this cell.
        // If any scope named one of them, the survey asserted that value and is simply wrong.
        const openEverywhere = axes.every((axis) =>
          (claimedScopes.get(cell) ?? []).every((scope) => !scope[axis]),
        );
        out.push(because(
          JSON.parse(key),
          openEverywhere ? "survey_wildcard_axis" : "survey_scope_overclaims",
        ));
      }
      for (const [key] of engineCoordinates) {
        if (surveyCoordinates.has(key)) continue;
        const row = JSON.parse(key);
        row.direction = "engine_to_survey";
        // Does the survey claim this provider/model/tier at CLEAN BASE and withhold only the loaded
        // overlay? Then the two sides are not in conflict — the engine side simply cannot express
        // the distinction. A dump's contract surface is keyed (tier, offloadPolicy, loadShape) with
        // NO overlay axis, so `implementedByTier` cannot say "rung 4 at overlay none only", while
        // the providers routinely mean exactly that: FLUX.1's `structurally_streamable` requires
        // `adapters.is_empty() && identity.is_none() && ip_adapter.is_none()`, FLUX.2 Klein's
        // `klein_streamable` requires `klein_overlay(spec).is_none()`, and Mage's
        // `surface_streamable` requires `spec.adapters.is_empty()`. The route witnesses DO reach
        // those overlays — routing is not rung capability — so the engine side over-approximates.
        // The probe deliberately ignores the mode axis; see `cleanBaseClaims` above.
        const withheldOverlay =
          row.overlay !== "none" &&
          cleanBaseClaims.has(`${row.provider}:${row.modelId}:${row.tier}`);
        // Does the verdict's own request-peak record mark this entry unmeasured at this coordinate?
        // Then the absence from the implementation scopes is a recorded measured-verdict decision,
        // not an omission (sc-21510): SC-15525 marks `illustrious_xl_v1`/`_v2` unmeasured in
        // `requestPeak.scopes`, and publishing the coordinates anyway would overturn that record.
        // Only an EXPLICIT entry name counts — an unmeasured top-level finding with no scopes says
        // nothing per-coordinate and must not launder ordinary underclaims.
        const withheldUnmeasured = (verdict.requestPeak?.scopes ?? []).some(
          (scope) =>
            scope.finding === "unmeasured" &&
            (scope.entries ?? []).includes(row.modelId) &&
            (scope.tiers ?? [row.tier]).includes(row.tier) &&
            (scope.modes ?? [row.mode]).includes(row.mode) &&
            (scope.overlays ?? [row.overlay]).includes(row.overlay),
        );
        out.push(because(
          row,
          verdict.implementation === "none"
            ? "survey_records_none"
            : withheldUnmeasured
              ? "survey_withholds_unmeasured_entry"
              : withheldOverlay
                ? "survey_withholds_loaded_overlay"
                : "survey_scope_underclaims",
        ));
      }
    }
  }

  const unique = new Map(out.map((entry) => [reconciliationMismatchKey(entry), entry]));
  return [...unique.entries()]
    .map(([key, entry]) => ({ ...entry, cause: causes.get(key) ?? null }))
    .sort((left, right) =>
      reconciliationMismatchKey(left).localeCompare(reconciliationMismatchKey(right)),
    );
}

/**
 * Reconcile the engine registries, the manifest declarations and the route witnesses, and REPORT what
 * disagrees. This never decides whether a build passes.
 *
 * # Report-only, and no waivers (Michael's decision, 2026-08-17)
 *
 * This function used to end in a waiver ledger: `config/memory-contract-reconciliation-waivers.json`
 * listed every accepted mismatch keyed by all eleven axes plus the provider's `selectorDigest`, and a
 * bijection check failed the build on any unwaived mismatch OR any waiver without a live mismatch.
 * The ledger, its schema, the `ownerStory`/`reason` fields and the bijection are all DELETED — not
 * demoted, not disabled behind a flag. Do not reintroduce them, and do not make this throw on a
 * finding.
 *
 * Why: the ledger was pin-keyed, so an inference pin bump staled it wholesale. Landing SC-18460 on the
 * current epic head produced 253 unwaived mismatches and 382 stale waivers at once — of which only 101
 * were the same coordinate with a rotated digest. The other 433 were coordinates appearing or
 * disappearing because provider surfaces legitimately moved. The only way to make that green was to
 * author 152 new waivers with invented owner stories, which is fabricated provenance, or to freeze the
 * pin. A gate whose green state depends on inventing paperwork is not measuring anything, and gate
 * volume is what makes these epics uncompletable. **Runtime catching is the chosen tradeoff.**
 *
 * What survives is the part that had value: this still enumerates every disagreement per coordinate —
 * that enumeration is how the 152 undeclared surfaces were found in the first place — and publishes it
 * through `scripts/report-memory-contract-reconciliation.mjs` and the matrix summary. Findings are for
 * a human to act on, never for CI to block on.
 *
 * `bespokeMemoryRouteWaivers` is NOT the removed concept and is deliberately kept: it is typed data the
 * engine dumper emits (see `validateBespokeMemoryRouteWaivers`), carries no owner story or accepted
 * digest, and cannot authorize anything — it records that a real worker route has no
 * `load(id, LoadSpec)` registration. It is engine facts, not human paperwork.
 */
export function reconcileMemoryContracts(input) {
  const engine = engineContractIndex(input.engineFacts);
  const declarations = manifestContracts(input.manifest);
  const bespokeWaivers = validateBespokeMemoryRouteWaivers(
    input.engineFacts,
    input.manifest,
    input.cells,
    engine,
    declarations,
  );
  const mismatches = collectMemoryContractMismatches(input);
  return {
    providers: input.engineFacts.reduce((count, document) => count + document.memoryContracts.length, 0),
    bespokeWaivers: bespokeWaivers.length,
    mismatches: mismatches.length,
    byLeg: Object.fromEntries(
      [...new Set(mismatches.map((entry) => entry.leg))].sort().map((leg) => [
        leg,
        mismatches.filter((entry) => entry.leg === leg).length,
      ]),
    ),
    // The full per-coordinate enumeration. Consumers report it; nothing gates on it.
    findings: mismatches,
  };
}
