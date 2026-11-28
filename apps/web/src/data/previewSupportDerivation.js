// Stage 2 of the engine-capability pipeline (`SW-6983`, `REG-3`, `M8`): the PURE derivation shared
// by the generator (apps/web/scripts/generate-preview-support.mjs) and its drift guard
// (previewSupportCatalog.test.js), so the committed artifacts and the test derive from one
// implementation.
//
// ## What this derives
//
// Every media, bespoke-promoted, and audio route's live-preview capability comes directly from
// inference's canonical hardware-free `sga.capabilities/2` exports:
//   - `config/engine-capabilities/sga/macos-mlx.json` (`backend: "mlx"`, `audio_backend: "candle"`)
//   - `config/engine-capabilities/sga/candle-cuda.json` (`backend: "candle"`, `audio_backend: "candle"`)
//
// SceneWorks product IDs map onto inference `route_id`s through `MODEL_TABLE` in
// `crates/sceneworks-worker/src/engines.rs` plus the two identity-backbone aliases
// (`pulid_flux_dev` -> `pulid_flux`, `instantid_realvisxl` -> `instantid`). If a backend registers
// the exact product ID directly (e.g. Candle's `qwen_image_edit_2509` / `qwen_image_edit_2511` /
// `qwen_image_edit_2511_lightning` routes), that direct route takes precedence over the alias.
// Audio routes use the identity join (`audio_routes[modelId]`).

/** The artifact schema version, bumped when the derived shape changes incompatibly. */
export const PREVIEW_SUPPORT_VERSION = 1;

/** Named in the artifacts so a reader knows the one sanctioned way to update them. */
export const PREVIEW_SUPPORT_GENERATOR = "apps/web/scripts/generate-preview-support.mjs";

/** Required canonical schema version for `sga.capabilities/2` exports. */
export const SGA_CAPABILITIES_SCHEMA_VERSION = "sga.capabilities/2";

/** Shipping desktop/server `sga.capabilities/2` exports that govern SceneWorks preview support. */
export const REQUIRED_SGA_EXPORTS = [
  { file: "candle-cuda.json", backend: "candle", featureSet: "candle-cuda" },
  { file: "macos-mlx.json", backend: "mlx", featureSet: "macos-mlx" },
];

/** Required backend sets covered by the canonical desktop/server exports. */
export const REQUIRED_MEDIA_BACKENDS = ["candle", "mlx"];
export const REQUIRED_AUDIO_BACKENDS = ["candle"];

const MODEL_TABLE_OPEN = "const MODEL_TABLE: &[ModelRow] = &[";
const BESPOKE_ALIASES_OPEN = "const BESPOKE_IMAGE_ROUTE_ALIASES: &[(&str, &str)] = &[";
const MIN_EXPECTED_MODEL_ROWS = 50;

function stripRustComments(source, sourceLabel = "engines.rs") {
  let out = "";
  let index = 0;
  let blockDepth = 0;
  while (index < source.length) {
    if (blockDepth > 0) {
      if (source.startsWith("/*", index)) {
        blockDepth += 1;
        index += 2;
      } else if (source.startsWith("*/", index)) {
        blockDepth -= 1;
        index += 2;
      } else {
        out += source[index] === "\n" ? "\n" : " ";
        index += 1;
      }
      continue;
    }
    if (source.startsWith("/*", index)) {
      blockDepth = 1;
      index += 2;
      continue;
    }
    if (source.startsWith("//", index)) {
      while (index < source.length && source[index] !== "\n") index += 1;
      continue;
    }
    if (source[index] === '"') {
      out += source[index];
      index += 1;
      while (index < source.length) {
        const char = source[index];
        out += char;
        index += 1;
        if (char === "\\") {
          if (index < source.length) {
            out += source[index];
            index += 1;
          }
          continue;
        }
        if (char === '"') break;
      }
      continue;
    }
    out += source[index];
    index += 1;
  }
  if (blockDepth > 0) {
    throw new Error(`${sourceLabel} has an unterminated \`/*\` block comment`);
  }
  return out;
}

function sliceBracketedArrayBody(source, openToken, label, fileName) {
  const open = source.indexOf(openToken);
  if (open === -1) {
    throw new Error(
      `${label}: no ${JSON.stringify(openToken)} in ${fileName} — the table was renamed or ` +
        "restructured; update this parser and re-run `npm run gen:preview-support`",
    );
  }
  const bodyStart = open + openToken.length;
  let depth = 1;
  let index = bodyStart;
  while (index < source.length) {
    const char = source[index];
    if (char === '"') {
      index += 1;
      while (index < source.length) {
        const inner = source[index];
        index += 1;
        if (inner === "\\") {
          index += 1;
          continue;
        }
        if (inner === '"') break;
      }
      continue;
    }
    if (char === "[" || char === "{" || char === "(") depth += 1;
    else if (char === "]" || char === "}" || char === ")") {
      depth -= 1;
      if (depth === 0) return source.slice(bodyStart, index);
    }
    index += 1;
  }
  throw new Error(
    `${label}: opening \`&[\` in ${fileName} is never closed — the table is truncated or the ` +
      "brackets are unbalanced",
  );
}

/**
 * Parse the `sceneworks_id` -> `engine_id` rows out of `crates/sceneworks-worker/src/engines.rs`.
 */
export function parseEngineModelTable(rustSource) {
  if (typeof rustSource !== "string" || rustSource.length === 0) {
    throw new Error("parseEngineModelTable: expected the engines.rs source text");
  }
  const body = sliceBracketedArrayBody(
    stripRustComments(rustSource, "parseEngineModelTable: engines.rs"),
    MODEL_TABLE_OPEN,
    "parseEngineModelTable",
    "engines.rs",
  );

  const rows = [];
  for (const match of body.matchAll(/ModelRow\s*\{([^{}]*)\}/g)) {
    const fields = match[1];
    const sceneworksId = /\bsceneworks_id:\s*"([^"]+)"/.exec(fields)?.[1];
    const engineId = /\bengine_id:\s*"([^"]+)"/.exec(fields)?.[1];
    if (!sceneworksId || !engineId) {
      throw new Error(
        `parseEngineModelTable: a ModelRow is missing sceneworks_id/engine_id:\n${fields.trim()}`,
      );
    }
    rows.push({ sceneworksId, engineId });
  }

  const declared = (body.match(/\bModelRow\b/g) ?? []).length;
  if (rows.length !== declared) {
    throw new Error(
      `parseEngineModelTable: MODEL_TABLE declares ${declared} ModelRow entries but only ` +
        `${rows.length} parsed. A row the brace-free matcher cannot read (most likely a new field ` +
        'with a nested `{ … }` value) would otherwise be dropped silently and read as "unknown". ' +
        "Teach this parser the new shape, then re-run `npm run gen:preview-support`.",
    );
  }

  if (rows.length < MIN_EXPECTED_MODEL_ROWS) {
    throw new Error(
      `parseEngineModelTable: parsed only ${rows.length} rows (expected at least ` +
        `${MIN_EXPECTED_MODEL_ROWS}) — the parse is almost certainly wrong`,
    );
  }
  const seen = new Set();
  for (const row of rows) {
    if (seen.has(row.sceneworksId)) {
      throw new Error(
        `parseEngineModelTable: MODEL_TABLE declares ${row.sceneworksId} twice; the SceneWorks id ` +
          "must be unique or the derived flag is ambiguous",
      );
    }
    seen.add(row.sceneworksId);
  }
  return rows;
}

/**
 * Parse `BESPOKE_IMAGE_ROUTE_ALIASES` out of `crates/sceneworks-core/src/registry/route_bindings.rs`
 * so preview-support derivation shares the single canonical bespoke product-to-route binding table.
 */
export function parseBespokeRouteAliases(rustSource) {
  if (typeof rustSource !== "string" || rustSource.length === 0) {
    throw new Error("parseBespokeRouteAliases: expected the route_bindings.rs source text");
  }
  const body = sliceBracketedArrayBody(
    stripRustComments(rustSource, "parseBespokeRouteAliases: route_bindings.rs"),
    BESPOKE_ALIASES_OPEN,
    "parseBespokeRouteAliases",
    "route_bindings.rs",
  );

  const aliases = [];
  for (const match of body.matchAll(/\(\s*"([^"]+)"\s*,\s*"([^"]+)"\s*,?\s*\)/g)) {
    aliases.push({ sceneworksId: match[1], engineId: match[2] });
  }

  const declared = (body.match(/\(/g) ?? []).length;
  if (aliases.length !== declared || aliases.length === 0) {
    throw new Error(
      `parseBespokeRouteAliases: BESPOKE_IMAGE_ROUTE_ALIASES declares ${declared} tuple(s) but ` +
        `parsed ${aliases.length}. Update this parser if the table shape changed.`,
    );
  }
  const seen = new Set();
  for (const alias of aliases) {
    if (seen.has(alias.sceneworksId)) {
      throw new Error(
        `parseBespokeRouteAliases: duplicate bespoke product id ${alias.sceneworksId}`,
      );
    }
    seen.add(alias.sceneworksId);
  }
  return aliases;
}

/**
 * Assert the checked-in `sga.capabilities/2` exports cover every required backend.
 */
export function assertBackendCoverage(declaredBackends, presentBackends, lane = "media") {
  if (lane !== "media" && lane !== "audio") {
    throw new Error(`assertBackendCoverage: unknown lane ${JSON.stringify(lane)}`);
  }
  if (!Array.isArray(declaredBackends) || declaredBackends.length === 0) {
    throw new Error(
      `assertBackendCoverage: the declared ${lane} backend set is empty, so this assertion would be vacuous.`,
    );
  }
  const nameless = presentBackends.filter(
    (backend) => typeof backend !== "string" || backend.length === 0,
  );
  if (nameless.length > 0) {
    throw new Error(
      `assertBackendCoverage: ${nameless.length} sga.capabilities/2 export(s) carry no readable \`backend\` field.`,
    );
  }
  const present = new Set(presentBackends);
  const missing = declaredBackends.filter((backend) => !present.has(backend));
  if (missing.length > 0) {
    throw new Error(
      `sga.capabilities/2 ${lane} exports are missing for backend(s) [${missing.join(", ")}]. ` +
        "Regenerate config/engine-capabilities/sga/*.json with inference's hardware-free " +
        "sga-capabilities-export-* CLIs, then re-run `npm run gen:preview-support` from apps/web.",
    );
  }
  const undeclared = [...present].filter((backend) => !declaredBackends.includes(backend));
  if (undeclared.length > 0) {
    throw new Error(
      `sga.capabilities/2 ${lane} exports exist for undeclared backend(s) [${undeclared.join(", ")}] ` +
        `(declared: ${JSON.stringify(declaredBackends)}).`,
    );
  }
}

/**
 * Validate one `sga.capabilities/2` document and index its media and audio route preview flags.
 */
export function indexSgaExport(doc, sourceLabel) {
  if (!doc || typeof doc !== "object" || Array.isArray(doc)) {
    throw new Error(`${sourceLabel}: not a JSON object`);
  }
  if (doc.schema_version !== SGA_CAPABILITIES_SCHEMA_VERSION) {
    throw new Error(
      `${sourceLabel}: unsupported schema_version ${JSON.stringify(doc.schema_version)}; expected ${JSON.stringify(SGA_CAPABILITIES_SCHEMA_VERSION)}`,
    );
  }
  if (typeof doc.backend !== "string" || doc.backend.length === 0) {
    throw new Error(`${sourceLabel}: missing \`backend\``);
  }
  if (typeof doc.feature_set !== "string" || doc.feature_set.length === 0) {
    throw new Error(`${sourceLabel}: missing \`feature_set\``);
  }
  if (!doc.routes || typeof doc.routes !== "object" || Array.isArray(doc.routes)) {
    throw new Error(`${sourceLabel}: missing \`routes\` map`);
  }
  const routeEntries = Object.entries(doc.routes);
  if (routeEntries.length === 0) {
    throw new Error(
      `${sourceLabel}: \`routes\` is empty. A vacuous export would derive as "no route supports live preview".`,
    );
  }

  const routes = new Map();
  for (const [routeId, route] of routeEntries) {
    const supportsPreview = route?.capabilities?.supports_preview;
    if (typeof supportsPreview !== "boolean") {
      throw new Error(
        `${sourceLabel}: route ${routeId} is missing boolean capabilities.supports_preview`,
      );
    }
    routes.set(routeId, supportsPreview);
  }

  const audioRoutes = new Map();
  if (doc.audio_routes && typeof doc.audio_routes === "object" && !Array.isArray(doc.audio_routes)) {
    for (const [routeId, route] of Object.entries(doc.audio_routes)) {
      const supportsPreview = route?.capabilities?.supports_preview;
      if (typeof supportsPreview !== "boolean") {
        throw new Error(
          `${sourceLabel}: audio route ${routeId} is missing boolean capabilities.supports_preview`,
        );
      }
      audioRoutes.set(routeId, supportsPreview);
    }
  }

  return {
    backend: doc.backend,
    featureSet: doc.feature_set,
    audioBackend: typeof doc.audio_backend === "string" && doc.audio_backend.length > 0 ? doc.audio_backend : null,
    routes,
    audioRoutes,
  };
}

const byBackendName = (left, right) =>
  left.backend < right.backend ? -1 : left.backend > right.backend ? 1 : 0;

/**
 * Derive the served preview-support catalog purely from MODEL_TABLE rows (plus identity-backbone
 * route aliases) and the canonical `sga.capabilities/2` exports (`macos-mlx.json`, `candle-cuda.json`).
 */
export function derivePreviewSupport(rows, sgaExports) {
  if (!Array.isArray(sgaExports) || sgaExports.length === 0) {
    throw new Error(
      "derivePreviewSupport: no sga.capabilities/2 exports provided. Expected config/engine-capabilities/sga/{macos-mlx,candle-cuda}.json.",
    );
  }

  const indexed = sgaExports
    .map((doc) => indexSgaExport(doc, `${doc?.feature_set ?? "?"}.json`))
    .sort(byBackendName);

  const backendNames = indexed.map((entry) => entry.backend);
  if (new Set(backendNames).size !== backendNames.length) {
    throw new Error(
      `derivePreviewSupport: duplicate sga.capabilities/2 export for backend (${backendNames})`,
    );
  }

  const allMediaRows = [...rows];
  const models = {};
  for (const row of allMediaRows) {
    const byBackend = {};
    for (const entry of indexed) {
      // Prefer exact SceneWorks product ID match when a backend registers it directly (e.g.
      // Candle's qwen_image_edit_2509 / _2511 / _2511_lightning), else fall back to engineId.
      if (entry.routes.has(row.sceneworksId)) {
        byBackend[entry.backend] = entry.routes.get(row.sceneworksId);
      } else if (entry.routes.has(row.engineId)) {
        byBackend[entry.backend] = entry.routes.get(row.engineId);
      }
    }
    if (Object.keys(byBackend).length > 0) {
      models[row.sceneworksId] = byBackend;
    }
  }

  const mediaIds = new Set(allMediaRows.map((row) => row.sceneworksId));
  const audioBackendNames = new Set();
  for (const entry of indexed) {
    if (!entry.audioBackend || entry.audioRoutes.size === 0) continue;
    audioBackendNames.add(entry.audioBackend);
    for (const [audioId, supportsPreview] of entry.audioRoutes) {
      if (mediaIds.has(audioId)) {
        throw new Error(
          `derivePreviewSupport: ${audioId} is both a media model id and an audio route id.`,
        );
      }
      const existing = models[audioId]?.[entry.audioBackend];
      if (existing !== undefined && existing !== supportsPreview) {
        throw new Error(
          `derivePreviewSupport: audio route ${audioId}/${entry.audioBackend} disagrees across exports`,
        );
      }
      models[audioId] = { ...(models[audioId] ?? {}), [entry.audioBackend]: supportsPreview };
    }
  }

  const sortedModels = {};
  for (const id of Object.keys(models).sort()) sortedModels[id] = models[id];

  const featureSets = {};
  for (const entry of indexed) {
    featureSets[entry.backend] = entry.featureSet;
  }

  return {
    version: PREVIEW_SUPPORT_VERSION,
    generatedBy: PREVIEW_SUPPORT_GENERATOR,
    backends: [...new Set([...backendNames, ...audioBackendNames])].sort(),
    generatedFrom: {
      schemaVersion: SGA_CAPABILITIES_SCHEMA_VERSION,
      featureSets,
    },
    models: sortedModels,
  };
}

/**
 * The web app's slice of the catalog — the SAME derivation with the backend-only provenance dropped.
 */
export function catalogToWebPreviewSupport(catalog) {
  return {
    version: catalog.version,
    backends: catalog.backends,
    models: catalog.models,
  };
}
