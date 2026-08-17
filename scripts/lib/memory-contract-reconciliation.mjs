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

function engineContractIndex(engineFacts, pin) {
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
    if (document.generatedFrom?.inferenceRevision !== pin) {
      throw new Error(
        `engine capability facts for ${backend} are keyed to ` +
          `${document.generatedFrom?.inferenceRevision ?? "(unset)"}, but Cargo pins ${pin}`,
      );
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

export function validateMemoryContractFacts(engineFacts, pin) {
  const providers = engineContractIndex(engineFacts, pin).size;
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
const ROUTE_LOAD_PROFILES = new Map([
  ["plain", "none"],
  ["lora", "lora"],
  ["lora_pid", "lora"],
  ["single_control", "control"],
  ["multi_control", "control"],
  ["lora_single_control", "control"],
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

/** Read the executable worker registry's pin-bound route witness, never Rust source text. */
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
  for (const entry of plan.providers ?? []) {
    const backend = entry.backend;
    const provider = entry.target?.provider;
    const mode = entry.target?.mode;
    if (!backend || !provider || !mode) throw new Error(`calibration plan entry ${entry.name ?? "(unnamed)"} is under-keyed`);
    rows.push({ backend, provider, mode, modelId: entry.target?.modelId ?? null });
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
  pin,
  engineFacts,
  manifest,
  cells,
  calibrationPlan,
  closures,
  survey,
}) {
  const engine = engineContractIndex(engineFacts, pin);
  const routeEligibility = routeEligibilityFromEngineFacts(engineFacts);
  const declarations = manifestContracts(manifest);
  const out = [];

  for (const contract of engine.values()) {
    for (const [tier, implemented] of contract.implementedByTier) {
      for (const rung of [...implemented].filter((item) => item !== "resident")) {
        if (!declarations.some((row) =>
          row.backend === contract.backend &&
          row.provider === contract.provider &&
          row.rung === rung &&
          row.tiers.includes(tier)
        )) {
          out.push(mismatch({
            leg: "engine_manifest",
            direction: "engine_to_manifest",
            backend: contract.backend,
            provider: contract.provider,
            tier,
            rung,
            selectorDigest: contract.selectorDigest,
          }));
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
          if (!cell || !eligible(row.backend, row.provider, tier, mode, overlay, row.loadProfiles)) {
            out.push(mismatch({
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
            }));
          }
        }
      }
    }
  }

  const plans = planRows(calibrationPlan);
  for (const row of plans) {
    if (!engine.has(laneKey(row.backend, row.provider))) {
      out.push(mismatch({ leg: "plan_closure_engine", direction: "plan_to_engine", ...row }));
    }
  }
  const closure = closureRows(closures);
  for (const row of closure) {
    if (!engine.has(laneKey(row.backend, row.provider))) {
      out.push(mismatch({ leg: "plan_closure_engine", direction: "closure_to_engine", ...row }));
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
      for (const cell of familyCells) {
        const claimed = scopes.some((scope) =>
          scope.entries.includes(cell.modelId) &&
          (scope.tiers ?? [cell.tier]).includes(cell.tier) &&
          (scope.modes ?? [cell.mode]).includes(cell.mode) &&
          (scope.overlays ?? [cell.overlay]).includes(cell.overlay),
        );
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
      for (const key of surveyCoordinates.keys()) {
        if (!engineCoordinates.has(key)) out.push(JSON.parse(key));
      }
      for (const [key] of engineCoordinates) {
        if (surveyCoordinates.has(key)) continue;
        const row = JSON.parse(key);
        row.direction = "engine_to_survey";
        out.push(row);
      }
    }
  }

  const unique = new Map(out.map((entry) => [reconciliationMismatchKey(entry), entry]));
  return [...unique.values()].sort((left, right) =>
    reconciliationMismatchKey(left).localeCompare(reconciliationMismatchKey(right)),
  );
}

function validateWaivers(ledger, pin, mismatches) {
  requireObject(ledger, "memory-contract waiver ledger");
  const schemaPath = "../packages/schemas/memory-contract-reconciliation-waivers.schema.json";
  const allowedTopLevel = new Set(["$schema", "schemaVersion", "inferenceRevision", "waivers"]);
  for (const field of Object.keys(ledger)) {
    if (!allowedTopLevel.has(field)) throw new Error(`memory-contract waiver ledger has unknown field ${field}`);
  }
  if (ledger.$schema !== schemaPath) {
    throw new Error(`memory-contract waiver ledger $schema must be ${schemaPath}`);
  }
  if (ledger.schemaVersion !== 1) throw new Error("memory-contract waiver ledger schemaVersion must be 1");
  if (ledger.inferenceRevision !== pin) {
    throw new Error(`memory-contract waiver ledger is keyed to ${ledger.inferenceRevision ?? "(unset)"}, but Cargo pins ${pin}`);
  }
  if (!Array.isArray(ledger.waivers)) throw new Error("memory-contract waiver ledger has no waivers array");
  const mismatchByKey = new Map(mismatches.map((entry) => [reconciliationMismatchKey(entry), entry]));
  const waiverByKey = new Map();
  const allowed = new Set([...AXES, "ownerStory", "reason"]);
  for (const [index, waiver] of ledger.waivers.entries()) {
    requireObject(waiver, `waiver ${index}`);
    for (const axis of AXES) {
      if (!Object.hasOwn(waiver, axis)) throw new Error(`waiver ${index} is under-keyed: missing ${axis}`);
    }
    for (const field of Object.keys(waiver)) {
      if (!allowed.has(field)) throw new Error(`waiver ${index} has unknown field ${field}`);
    }
    if (Object.values(waiver).includes("*")) throw new Error(`waiver ${index} contains a wildcard`);
    if (!/^sc-[0-9]+$/.test(waiver.ownerStory ?? "")) throw new Error(`waiver ${index} has no ownerStory`);
    if (typeof waiver.reason !== "string" || waiver.reason.trim().length < 12) {
      throw new Error(`waiver ${index} has no actionable reason`);
    }
    const waiverKey = reconciliationMismatchKey(waiver);
    if (waiverByKey.has(waiverKey)) throw new Error(`duplicate waiver ${waiverKey}`);
    waiverByKey.set(waiverKey, waiver);
  }
  const missing = [...mismatchByKey.keys()].filter((entry) => !waiverByKey.has(entry));
  const stale = [...waiverByKey.keys()].filter((entry) => !mismatchByKey.has(entry));
  if (missing.length || stale.length) {
    throw new Error(
      `memory-contract reconciliation found ${missing.length} unwaived mismatch(es) and ` +
        `${stale.length} stale waiver(s)` +
        `${missing.length ? `\nunwaived:\n${missing.join("\n")}` : ""}` +
        `${stale.length ? `\nstale:\n${stale.join("\n")}` : ""}`,
    );
  }
}

export function reconcileMemoryContracts(input) {
  const engine = engineContractIndex(input.engineFacts, input.pin);
  const declarations = manifestContracts(input.manifest);
  const bespokeWaivers = validateBespokeMemoryRouteWaivers(
    input.engineFacts,
    input.manifest,
    input.cells,
    engine,
    declarations,
  );
  const mismatches = collectMemoryContractMismatches(input);
  validateWaivers(input.waiverLedger, input.pin, mismatches);
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
  };
}
