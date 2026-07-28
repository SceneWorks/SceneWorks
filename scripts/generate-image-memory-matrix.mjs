#!/usr/bin/env node

import { createHash } from "node:crypto";
import { readFile, writeFile, mkdir } from "node:fs/promises";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";

import { stripJsoncComments } from "./lib/jsonc.mjs";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const OUTPUT_JSON = "docs/generated/image-memory-matrix.json";
const OUTPUT_MD = "docs/generated/image-memory-matrix.md";
const EXPECTED_IMAGE_COUNT = 53;
const EXPECTED_MLX_STAGED_COUNT = 39;
const RUNGS = [
  "resident",
  "staged_residency",
  "bounded_decode",
  "bounded_attention",
  "bounded_transformer_residency",
];
const GENERATION_CAPABILITIES = new Set([
  "text_to_image",
  "edit_image",
  "image_to_image",
  "image_inpaint",
  "image_detail",
  "character_image",
  "style_variations",
]);

// This is ownership metadata, not conformance data. Drift is checked against the
// source-owned EXPECTED_IMAGE_IDS list and the shipped manifest below.
const MODEL_STORIES = {
  mage_flow_edit_base: 15450,
  mage_flow_edit: 15451,
  mage_flow_edit_turbo: 15452,
  mage_flow_base: 15453,
  mage_flow: 15454,
  mage_flow_turbo: 15455,
  z_image_turbo: 15456,
  z_image: 15457,
  z_image_edit: 15458,
  qwen_image: 15459,
  qwen_image_edit_2511: 15460,
  qwen_image_edit_2511_lightning: 15461,
  lens: 15462,
  lens_turbo: 15463,
  sensenova_u1_8b: 15464,
  sensenova_u1_8b_infographic_v2: 15465,
  sensenova_u1_8b_infographic_v3: 15466,
  sensenova_u1_8b_fast: 15467,
  sensenova_u1_8b_infographic_v2_fast: 15468,
  sensenova_u1_8b_infographic_v3_fast: 15469,
  flux_schnell: 15470,
  flux_dev: 15471,
  ideogram_4: 15472,
  ideogram_4_turbo: 15473,
  boogu_image: 15474,
  boogu_image_turbo: 15475,
  boogu_image_edit: 15476,
  krea_2_turbo: 15477,
  krea_2_raw: 15478,
  flux2_klein_9b: 15479,
  flux2_klein_9b_kv: 15480,
  flux2_klein_9b_true_v2: 15481,
  flux2_dev: 15482,
  chroma1_hd: 15483,
  chroma1_base: 15484,
  chroma1_flash: 15485,
  kolors: 15486,
  sd3_5_large: 15487,
  sd3_5_large_turbo: 15488,
  sd3_5_medium: 15489,
  sana_1600m: 15490,
  sana_sprint_1600m: 15491,
  anima_base: 15492,
  anima_aesthetic: 15493,
  anima_turbo: 15494,
  sdxl: 15495,
  realvisxl: 15496,
  realvisxl_lightning: 15497,
  illustrious_xl_v1: 15498,
  illustrious_xl_v2: 15499,
  instantid_realvisxl: 15500,
  pulid_flux_dev: 15501,
  bernini_image: 15502,
};

function familyStory(modelId) {
  if (modelId.startsWith("mage_flow")) return 15509;
  if (modelId.startsWith("z_image")) return 15510;
  if (modelId.startsWith("qwen_image")) return 15511;
  if (modelId.startsWith("lens")) return 15512;
  if (modelId.startsWith("sensenova")) return 15513;
  if (modelId === "flux_schnell" || modelId === "flux_dev") return 15514;
  if (modelId.startsWith("ideogram")) return 15515;
  if (modelId.startsWith("boogu")) return 15516;
  if (modelId.startsWith("krea_2")) return 15517;
  if (modelId.startsWith("flux2_klein")) return 15518;
  if (modelId === "flux2_dev") return 15519;
  if (modelId.startsWith("chroma1")) return 15520;
  if (modelId === "kolors") return 15521;
  if (modelId.startsWith("sd3_5")) return 15522;
  if (modelId.startsWith("sana")) return 15523;
  if (modelId.startsWith("anima")) return 15524;
  if (["sdxl", "realvisxl", "realvisxl_lightning", "illustrious_xl_v1", "illustrious_xl_v2"].includes(modelId)) return 15525;
  if (modelId === "instantid_realvisxl") return 15526;
  if (modelId === "pulid_flux_dev") return 15527;
  if (modelId === "bernini_image") return 15528;
  throw new Error(`no family story for ${modelId}`);
}

function sha256(body) {
  return createHash("sha256").update(body).digest("hex");
}

function sortedUnique(values) {
  return [...new Set(values)].sort();
}

function parseExpectedImageIds(source) {
  const match = source.match(/const EXPECTED_IMAGE_IDS:\s*&\[&str\]\s*=\s*&\[([\s\S]*?)\n\s*\];/);
  if (!match) throw new Error("could not locate EXPECTED_IMAGE_IDS in engines.rs");
  return [...match[1].matchAll(/"([^"]+)"/g)].map((item) => item[1]);
}

function parseEngineRoutes(source) {
  const table = source.match(/pub\(crate\) const MODEL_TABLE:[\s\S]*?=\s*&\[([\s\S]*?)\n\];/);
  if (!table) throw new Error("could not locate MODEL_TABLE in engines.rs");
  const routes = new Map();
  for (const row of table[1].matchAll(/ModelRow\s*\{([\s\S]*?)\n\s*\},/g)) {
    const model = row[1].match(/sceneworks_id:\s*"([^"]+)"/)?.[1];
    const engine = row[1].match(/engine_id:\s*"([^"]+)"/)?.[1];
    const repo = row[1].match(/default_repo:\s*"([^"]+)"/)?.[1] ?? null;
    if (model && engine) routes.set(model, { engine, repo, kind: "registry" });
  }
  routes.set("instantid_realvisxl", {
    engine: "instantid",
    repo: null,
    kind: "bespoke",
  });
  routes.set("pulid_flux_dev", {
    engine: "pulid_flux",
    repo: null,
    kind: "bespoke",
  });
  return routes;
}

function parseMlxSequentialEngines(source) {
  const test = source.match(
    /fn engine_supports_sequential_is_derived_from_the_registered_capability\(\)\s*\{([\s\S]*?)\n\s*\}\n\n\s*\/\/\/ An id with no registered generator/,
  );
  if (!test) {
    throw new Error("could not locate the MLX sequential-capability registry sweep");
  }
  const beforeNegativeControl = test[1].split("assert!(!engine_supports_sequential")[0];
  return new Set([...beforeNegativeControl.matchAll(/"([^"]+)"/g)].map((item) => item[1]));
}

function inferencePin(cargo) {
  const match = cargo.match(
    /candle-kernels\s*=\s*\{[^}]*?github\.com\/SceneWorks\/inference[^}]*?rev\s*=\s*"([0-9a-f]+)"/,
  );
  if (!match) throw new Error("could not resolve the pinned SceneWorks/inference revision");
  return match[1];
}

function backendScopes(model, manifestById) {
  const inherited = model.id === "z_image_edit" ? manifestById.get("z_image_turbo") : model;
  const scopes = [];
  if (inherited?.mlx || ["instantid_realvisxl", "pulid_flux_dev"].includes(model.id)) scopes.push("mlx");
  if (inherited?.candle || ["instantid_realvisxl", "pulid_flux_dev"].includes(model.id)) scopes.push("candle");
  return scopes;
}

function tiersFor(model, backend) {
  const backendTiers = Object.keys(model[backend]?.vramGbByTier ?? {});
  const downloadTiers = (model.downloads ?? [])
    .map((download) => download.variant)
    .filter((variant) => typeof variant === "string" && /^(bf16|fp16|q\d+|nvfp4|int\d+)/.test(variant));
  const inferred = model[backend]?.quantize === 4 ? ["q4"] : model[backend]?.quantize === 8 ? ["q8"] : [];
  return sortedUnique([...backendTiers, ...downloadTiers, ...inferred]).filter(
    (tier) => tier !== "int8-convrot",
  ).length
    ? sortedUnique([...backendTiers, ...downloadTiers, ...inferred]).filter((tier) => tier !== "int8-convrot")
    : ["default"];
}

function modesFor(model) {
  const modes = (model.capabilities ?? []).filter((capability) => GENERATION_CAPABILITIES.has(capability));
  return modes.length ? sortedUnique(modes) : ["catalog_default"];
}

function overlaysFor(model, backend) {
  const overlays = ["none"];
  if (model.loraCompatibility) overlays.push("lora");
  if (model[backend]?.control) overlays.push("control");
  if ((model.capabilities ?? []).includes("character_image")) overlays.push("identity");
  return sortedUnique(overlays);
}

function geometryFor(model, backend) {
  const limits = { ...(model.limits ?? {}), ...(model[backend]?.limits ?? {}) };
  const defaults = { ...(model.defaults ?? {}), ...(model[backend]?.defaults ?? {}) };
  const envelope = {
    defaultResolution: defaults.resolution ?? null,
    resolutions: Array.isArray(limits.resolutions) ? limits.resolutions : [],
    minWidth: limits.minWidth ?? limits.minSize ?? null,
    maxWidth: limits.maxWidth ?? limits.maxSize ?? null,
    minHeight: limits.minHeight ?? limits.minSize ?? null,
    maxHeight: limits.maxHeight ?? limits.maxSize ?? null,
  };
  return Object.fromEntries(
    Object.entries(envelope).filter(
      ([, value]) => value !== null && (!Array.isArray(value) || value.length > 0),
    ),
  );
}

function artifactEvidence(model, route, tier) {
  const downloads = model.downloads ?? [];
  const tierMatches = downloads.filter((download) => download.variant === tier);
  const relevant = tierMatches.length
    ? [...tierMatches, ...downloads.filter((download) => download.variant == null)]
    : downloads;
  const artifacts = relevant.map((download) => ({
    repository: download.repo ?? null,
    revision: download.revision ?? null,
    variant: download.variant ?? null,
  }));
  if (!artifacts.length && route.repo) {
    artifacts.push({ repository: route.repo, revision: null, variant: null });
  }
  return [
    ...new Map(
      artifacts.map((artifact) => [
        `${artifact.repository}:${artifact.revision}:${artifact.variant}`,
        artifact,
      ]),
    ).values(),
  ];
}

function declaredEvidence(model, backend, tier) {
  const scope = model[backend] ?? {};
  const keys = [
    "minMemoryGb",
    "vramGbByTier",
    "sequentialPeakGb",
    "turboFit",
    "measured",
    "quantize",
    "standardTierLayout",
  ].filter((key) => scope[key] !== undefined);
  return keys.map((key) => ({
    source: `config/manifests/builtin.models.jsonc#models/${model.id}/${backend}/${key}`,
    tier,
  }));
}

function strategyStatus({ backend, rung, route, sequentialEngines, model }) {
  if (rung === "resident") {
    return {
      state: "Implemented/unverified",
      source: `crates/sceneworks-worker/src/engines.rs#${route.kind === "registry" ? "MODEL_TABLE" : "bespoke_advertised"}`,
      parameters: {},
    };
  }
  if (
    rung === "staged_residency" &&
    ((backend === "mlx" && sequentialEngines.has(route.engine)) ||
      (backend === "candle" &&
        (model.candle?.sequentialPeakGb !== undefined || model.candle?.turboFit !== undefined)))
  ) {
    return {
      state: "Implemented/unverified",
      source:
        backend === "mlx"
          ? "crates/sceneworks-worker/src/mlx_fit_gate.rs#engine_supports_sequential"
          : `config/manifests/builtin.models.jsonc#models/${model.id}/candle`,
      parameters: { phaseOrder: ["conditioning", "denoise", "decode"] },
    };
  }
  return { state: "Missing", source: null, parameters: {} };
}

function validateMatrix(matrix, expectedIds) {
  const ids = matrix.models.map((model) => model.id);
  if (ids.length !== EXPECTED_IMAGE_COUNT) {
    throw new Error(`expected exactly ${EXPECTED_IMAGE_COUNT} image entries, found ${ids.length}`);
  }
  if (
    new Set(expectedIds).size !== ids.length ||
    ids.some((id) => !expectedIds.includes(id)) ||
    expectedIds.some((id) => !ids.includes(id))
  ) {
    const manifestOnly = ids.filter((id) => !expectedIds.includes(id));
    const sourceOnly = expectedIds.filter((id) => !ids.includes(id));
    throw new Error(
      `manifest image ids, EXPECTED_IMAGE_IDS, and generated ownership rows disagree (manifest-only=${manifestOnly.join(",")}; source-only=${sourceOnly.join(",")})`,
    );
  }
  if (matrix.summary.mlxStagedStaticCoverage !== EXPECTED_MLX_STAGED_COUNT) {
    throw new Error(
      `expected MLX staged static coverage ${EXPECTED_MLX_STAGED_COUNT}/${EXPECTED_IMAGE_COUNT}, found ${matrix.summary.mlxStagedStaticCoverage}`,
    );
  }
  for (const cell of matrix.cells) {
    if (cell.state !== "Missing" && cell.evidence.staticImplementation.length === 0) {
      throw new Error(`${cell.id}: non-Missing classification has no static evidence`);
    }
    if (cell.state === "Structurally N/A" && cell.evidence.structural.length === 0) {
      throw new Error(`${cell.id}: Structurally N/A classification has no structural evidence`);
    }
    if (cell.state === "Verified") {
      const dynamic = cell.evidence.currentEnvironmentVerification;
      if (!dynamic.length || !cell.calibrationFingerprint || !Object.keys(cell.strategyParameters).length) {
        throw new Error(`${cell.id}: unsupported Full/Verified claim`);
      }
    }
  }
}

export async function buildMatrix() {
  const sourcePaths = {
    manifest: "config/manifests/builtin.models.jsonc",
    engines: "crates/sceneworks-worker/src/engines.rs",
    mlxFitGate: "crates/sceneworks-worker/src/mlx_fit_gate.rs",
    cargo: "Cargo.toml",
  };
  const [manifestBody, enginesBody, mlxFitBody, cargoBody] = await Promise.all(
    Object.values(sourcePaths).map((relative) => readFile(path.join(ROOT, relative), "utf8")),
  );
  const manifest = JSON.parse(stripJsoncComments(manifestBody));
  const images = manifest.models.filter((model) => model.type === "image");
  const manifestById = new Map(images.map((model) => [model.id, model]));
  const expectedIds = parseExpectedImageIds(enginesBody);
  const routes = parseEngineRoutes(enginesBody);
  const sequentialEngines = parseMlxSequentialEngines(mlxFitBody);
  const pin = inferencePin(cargoBody);
  const sceneWorksRevision = `source-tree:${sha256(manifestBody + enginesBody + mlxFitBody + cargoBody)}`;

  const models = images
    .map((model) => {
      const route = routes.get(model.id);
      if (!route) throw new Error(`${model.id}: no resolved route/provider`);
      if (!MODEL_STORIES[model.id]) throw new Error(`${model.id}: no owning model story`);
      return {
        id: model.id,
        name: model.name,
        family: model.family ?? null,
        resolvedRoute: route.engine,
        routeKind: route.kind,
        backends: backendScopes(model, manifestById),
        owningFamilyStory: familyStory(model.id),
        owningModelStory: MODEL_STORIES[model.id],
      };
    })
    .sort((left, right) => left.id.localeCompare(right.id));

  const cells = [];
  for (const modelSummary of models) {
    const model = manifestById.get(modelSummary.id);
    const route = routes.get(model.id);
    for (const backend of modelSummary.backends) {
      for (const tier of tiersFor(model, backend)) {
        for (const mode of modesFor(model)) {
          for (const overlay of overlaysFor(model, backend)) {
            for (const rung of RUNGS) {
              const status = strategyStatus({ backend, rung, route, sequentialEngines, model });
              const fingerprint =
                status.state === "Missing"
                  ? null
                  : sha256(
                      JSON.stringify({
                        sceneWorksRevision,
                        inferencePin: pin,
                        model: model.id,
                        route: route.engine,
                        backend,
                        tier,
                        mode,
                        overlay,
                        rung,
                        parameters: status.parameters,
                      }),
                    );
              cells.push({
                id: [model.id, route.engine, backend, tier, mode, overlay, rung].join(":"),
                modelId: model.id,
                resolvedRoute: route.engine,
                provider: route.engine,
                backend,
                tier,
                mode,
                overlay,
                rung,
                geometryEnvelope: geometryFor(model, backend),
                strategyParameters: status.parameters,
                state: status.state,
                evidenceRevision: {
                  sceneWorks: sceneWorksRevision,
                  inference: pin,
                },
                calibrationFingerprint: fingerprint,
                owningFamilyStory: modelSummary.owningFamilyStory,
                owningModelStory: modelSummary.owningModelStory,
                evidence: {
                  staticImplementation: status.source ? [{ source: status.source }] : [],
                  declaredCalibration: declaredEvidence(model, backend, tier),
                  historicalVerification: [],
                  currentEnvironmentVerification: [],
                  loadability: artifactEvidence(model, route, tier),
                  strategyParameterVerification: [],
                  structural: [],
                },
              });
            }
          }
        }
      }
    }
  }
  cells.sort((left, right) => left.id.localeCompare(right.id));

  const modelSlices = Object.fromEntries(
    models.map((model) => [
      model.id,
      cells.filter((cell) => cell.modelId === model.id).map((cell) => cell.id),
    ]),
  );
  const mlxStagedModels = new Set(
    cells
      .filter(
        (cell) =>
          cell.backend === "mlx" &&
          cell.rung === "staged_residency" &&
          cell.state === "Implemented/unverified",
      )
      .map((cell) => cell.modelId),
  );
  const matrix = {
    schemaVersion: 1,
    generatedFrom: {
      sceneWorksRevision,
      inferenceRevision: pin,
      sources: Object.fromEntries(
        Object.entries(sourcePaths).map(([name, source], index) => [
          name,
          { path: source, sha256: sha256([manifestBody, enginesBody, mlxFitBody, cargoBody][index]) },
        ]),
      ),
    },
    conformanceStates: [
      "Verified",
      "Implemented/unverified",
      "Structurally N/A",
      "Missing",
      "Route unavailable/broken",
    ],
    evidenceDimensions: [
      "staticImplementation",
      "declaredCalibration",
      "historicalVerification",
      "currentEnvironmentVerification",
      "loadability",
      "strategyParameterVerification",
    ],
    summary: {
      imageModels: models.length,
      cells: cells.length,
      mlxStagedStaticCoverage: mlxStagedModels.size,
      mlxStagedStaticCoverageDenominator: EXPECTED_IMAGE_COUNT,
      fullModels: 0,
    },
    models,
    cells,
    modelSlices,
  };
  validateMatrix(matrix, expectedIds);
  return matrix;
}

function renderMarkdown(matrix) {
  const lines = [
    "# Generated image memory-ladder matrix",
    "",
    "> Generated by `scripts/generate-image-memory-matrix.mjs`. Do not edit by hand.",
    "",
    `- SceneWorks revision: \`${matrix.generatedFrom.sceneWorksRevision}\``,
    `- Inference revision: \`${matrix.generatedFrom.inferenceRevision}\``,
    `- Catalog entries: ${matrix.summary.imageModels}`,
    `- Cells: ${matrix.summary.cells}`,
    `- MLX staged-residency static coverage: ${matrix.summary.mlxStagedStaticCoverage}/${matrix.summary.mlxStagedStaticCoverageDenominator}`,
    `- Full models: ${matrix.summary.fullModels}`,
    "",
    "Static capability is never promoted to dynamic verification. Generated cells contain separate declared, historical, current-environment, loadability, and strategy-parameter evidence arrays.",
    "",
    "| Catalog entry | Route | Backends | Family story | Model story | MLX staged |",
    "| --- | --- | --- | --- | ---: | --- |",
  ];
  for (const model of matrix.models) {
    const staged = matrix.cells.some(
      (cell) =>
        cell.modelId === model.id &&
        cell.backend === "mlx" &&
        cell.rung === "staged_residency" &&
        cell.state === "Implemented/unverified",
    );
    lines.push(
      `| \`${model.id}\` | \`${model.resolvedRoute}\` (${model.routeKind}) | ${model.backends.join(", ")} | SC-${model.owningFamilyStory} | SC-${model.owningModelStory} | ${staged ? "Implemented/unverified" : "Missing"} |`,
    );
  }
  lines.push(
    "",
    "Per-model consumers must use `modelSlices` in the JSON artifact. A cell is Full only when every applicable rung is Verified or Structurally N/A; this static baseline intentionally reports zero Full models.",
    "",
  );
  return lines.join("\n");
}

async function main() {
  const matrix = await buildMatrix();
  const json = `${JSON.stringify(matrix, null, 2)}\n`;
  const markdown = renderMarkdown(matrix);
  const check = process.argv.includes("--check");
  if (check) {
    const [existingJson, existingMarkdown] = await Promise.all([
      readFile(path.join(ROOT, OUTPUT_JSON), "utf8"),
      readFile(path.join(ROOT, OUTPUT_MD), "utf8"),
    ]);
    if (existingJson !== json || existingMarkdown !== markdown) {
      throw new Error("generated image memory matrix is stale; run npm run generate:image-memory-matrix");
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
