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
  return engineContractIndex(engineFacts, pin).size;
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
        if (!RUNGS.includes(implementation.rung)) {
          throw new Error(`${model.id}:${backend}:${contract.provider} names unknown rung ${implementation.rung}`);
        }
        if (!Array.isArray(implementation.tiers) || implementation.tiers.length === 0) {
          throw new Error(`${model.id}:${backend}:${contract.provider}:${implementation.rung} has no tiers`);
        }
        rows.push({
          backend,
          provider: contract.provider,
          modelId: model.id,
          rung: implementation.rung,
          tiers: sortedUnique(implementation.tiers),
          modes: sortedUnique(implementation.modes ?? []),
          overlays: sortedUnique(implementation.overlays ?? []),
        });
      }
    }
  }
  return rows;
}

function exactFunctionBody(source, name) {
  const marker = source.search(new RegExp(`\\bfn\\s+${name}\\s*\\(`));
  if (marker < 0) throw new Error(`route source has no Rust function ${name}`);
  const open = source.indexOf("{", marker);
  if (open < 0) throw new Error(`route source function ${name} has no body`);
  let depth = 0;
  for (let index = open; index < source.length; index += 1) {
    if (source[index] === "{") depth += 1;
    if (source[index] === "}") depth -= 1;
    if (depth === 0) return source.slice(open + 1, index);
  }
  throw new Error(`route source function ${name} has an unterminated body`);
}

function providerStrings(body) {
  return [...body.matchAll(/"([a-z][a-z0-9_]+)"/g)].map((match) => ({
    provider: match[1],
    offset: match.index,
  }));
}

function rustStatement(body, offset) {
  const start = body.lastIndexOf(";", offset) + 1;
  const next = body.indexOf(";", offset);
  return body.slice(start, next < 0 ? body.length : next + 1);
}

function eligibleOverlays(statement) {
  const overlays = ["none"];
  if (!statement.includes("spec.adapters.is_empty()")) overlays.push("lora");
  if (!statement.includes("spec.control.is_none()")) overlays.push("control");
  if (!statement.includes("spec.pid.is_none()") && !statement.includes("spec.identity.is_none()")) {
    overlays.push("identity");
  }
  return overlays;
}

/**
 * Read the production load-shape helpers themselves. SC-18457 replaces their literal population
 * with contract-derived eligibility; its branch must replace this transition parser with that
 * exported population before the final paired-pin regeneration.
 */
export function routeEligibilityFromRust({ imageRouting, mlxFitGate }) {
  const rows = [];
  const mlx = exactFunctionBody(imageRouting, "apply_measured_mlx_load_shape_for_request");
  for (const item of providerStrings(mlx)) {
    const statement = rustStatement(mlx, item.offset);
    for (const overlay of eligibleOverlays(statement)) {
      rows.push({
        backend: "mlx",
        provider: item.provider,
        mode: statement.includes("plain_text_to_image") ? "text_to_image" : null,
        overlay,
      });
    }
  }
  const sequential = exactFunctionBody(mlxFitGate, "with_selected_sequential_shape");
  for (const item of providerStrings(sequential)) {
    for (const overlay of eligibleOverlays(rustStatement(sequential, item.offset))) {
      rows.push({ backend: "mlx", provider: item.provider, mode: null, overlay });
    }
  }
  const candle = exactFunctionBody(imageRouting, "apply_candle_image_load_shape");
  for (const item of providerStrings(candle)) {
    for (const overlay of eligibleOverlays(rustStatement(candle, item.offset))) {
      rows.push({ backend: "candle", provider: item.provider, mode: null, overlay });
    }
  }
  return [...new Map(rows.map((row) => [JSON.stringify(row), row])).values()].sort((left, right) =>
    JSON.stringify(left).localeCompare(JSON.stringify(right)),
  );
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
  routeEligibility,
}) {
  const engine = engineContractIndex(engineFacts, pin);
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

  const eligible = (backend, provider, mode, overlay) => routeEligibility.some(
    (row) =>
      row.backend === backend &&
      row.provider === provider &&
      (row.mode === null || row.mode === mode) &&
      row.overlay === overlay,
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
          if (!cell || !eligible(row.backend, row.provider, mode, overlay)) {
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

  const familyProviders = new Map();
  for (const cell of cells) {
    if (!engine.has(laneKey(cell.backend, cell.provider))) continue;
    const familyKey = `${cell.owningFamilyStory}:${cell.backend}`;
    if (!familyProviders.has(familyKey)) familyProviders.set(familyKey, new Set());
    familyProviders.get(familyKey).add(cell.provider);
  }
  const surveyFamilies = survey.families ?? {};
  for (const [familyStory, family] of Object.entries(surveyFamilies)) {
    for (const [backend, verdict] of Object.entries(family.backends ?? {})) {
      const providers = familyProviders.get(`${familyStory}:${backend}`) ?? new Set();
      const implemented = [...providers].filter((provider) =>
        engine.get(laneKey(backend, provider))?.implemented.has("bounded_transformer_residency"),
      );
      if (verdict.implementation === "none") {
        for (const provider of implemented) {
          const contract = engine.get(laneKey(backend, provider));
          out.push(mismatch({
            leg: "survey_engine",
            direction: "survey_to_engine",
            backend,
            provider,
            familyStory: Number(familyStory),
            rung: "bounded_transformer_residency",
            selectorDigest: contract.selectorDigest,
          }));
        }
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
  const mismatches = collectMemoryContractMismatches(input);
  validateWaivers(input.waiverLedger, input.pin, mismatches);
  return {
    providers: input.engineFacts.reduce((count, document) => count + document.memoryContracts.length, 0),
    mismatches: mismatches.length,
    byLeg: Object.fromEntries(
      [...new Set(mismatches.map((entry) => entry.leg))].sort().map((leg) => [
        leg,
        mismatches.filter((entry) => entry.leg === leg).length,
      ]),
    ),
  };
}
