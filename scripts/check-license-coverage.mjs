// SceneWorks packaging guard: every shipped model must have an About→Licenses entry (sc-13803).
//
// WHY THIS EXISTS
// ---------------
// The About→Licenses screen (apps/desktop/licenses/manifest.json, rendered via
// apps/web/src/data/bundledLicenses.js) records the upstream license + attribution
// for every AI model SceneWorks downloads at runtime — the obligation that travels
// with weights we re-host on our Hugging Face org, and the disclosure that tells a
// user which models are NON-COMMERCIAL or otherwise restricted before they build a
// business on one.
//
// That page drifted badly: it covered the bundled binaries, three Wan2.2 video
// models, LTX and the audio models, while ~30 image/video/utility primaries — SDXL,
// FLUX, Krea, Ideogram, SANA, SD3.5, Anima, Qwen-Image, Z-Image, the PiD decoders
// and more — shipped with NO entry at all. Several of those are non-commercial.
// Nothing caught it because nothing checked: adding a model to the catalog had no
// mechanical link to the licenses page (sc-13803).
//
// This closes that loop. Every catalog entry that DOWNLOADS weights must be claimed
// by exactly one licenses component, and every claimed id must still exist. Adding a
// model without recording its license now fails the build.
//
// WHAT IT CHECKS
// --------------
//   1. Coverage      — every downloading catalog model id appears in some component's `models`.
//   2. No stale ids  — every id listed by a component still exists in the catalog.
//   3. No duplicates — no id is claimed by two components (an ambiguous attribution).
//   4. Wiring        — every component document `key` resolves in bundledLicenses.js,
//                      and its license text file exists on disk. A component whose text
//                      is missing renders as an EMPTY license — worse than absent,
//                      because the page then asserts coverage it does not have.
//
// UNDETERMINED UPSTREAM LICENSES are listed explicitly below rather than silently
// skipped, so the remaining decision is visible in code review instead of invisible.
//
// Usage: node scripts/check-license-coverage.mjs
import fs from "node:fs";
import path from "node:path";
import crypto from "node:crypto";
import { fileURLToPath } from "node:url";
import {
  parse as parseProvenanceCandidates,
  populationSha256,
} from "./scan-inference-provenance.mjs";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const CATALOG = path.join(ROOT, "config/manifests/builtin.models.jsonc");
const MANIFEST = path.join(ROOT, "apps/desktop/licenses/manifest.json");
const LICENSES_DIR = path.join(ROOT, "apps/desktop/licenses");
const BUNDLED_JS = path.join(ROOT, "apps/web/src/data/bundledLicenses.js");
const SOURCE_AUDIT = path.join(ROOT, "config/inference-third-party-source.json");
const PROVENANCE_CANDIDATES = path.join(ROOT, "config/inference-provenance-candidates.tsv");
const WORKER_CARGO = path.join(ROOT, "crates/sceneworks-worker/Cargo.toml");
const ROOT_CARGO = path.join(ROOT, "Cargo.toml");
const CARGO_LOCK = path.join(ROOT, "Cargo.lock");
const PACKAGE_JSON = path.join(ROOT, "package.json");
const TAURI_CONFIG = path.join(ROOT, "apps/desktop/tauri.conf.json");
const RUST_API_LIB = path.join(ROOT, "apps/rust-api/src/lib.rs");
const DESKTOP_PACKAGE = path.join(ROOT, "apps/desktop/package.json");
const BUILD_SIDECAR = path.join(ROOT, "apps/desktop/scripts/build-sidecar.mjs");
const BUILD_PLAN = path.join(ROOT, "apps/desktop/scripts/build-sidecar-platform.mjs");

// Catalog models whose UPSTREAM declares no license at all, so no entry can be
// recorded honestly yet. Each needs a licensing decision, not a guess — writing a
// plausible-looking license onto the product's legal page would be worse than the
// gap it papers over. Tracked on sc-13803; keep this list SHRINKING.
const UNDETERMINED = new Map([
  [
    "clip_vit_l14",
    "openai/clip-vit-large-patch14 declares no license in its HF cardData or model card.",
  ],
  [
    "joycaption_beta_one",
    "fancyfeast/llama-joycaption-beta-one-hf-llava declares no license; it is a Llama-3.1-8B-Instruct + SigLIP2 derivative, so Meta's Llama 3.1 Community License likely governs — needs confirmation.",
  ],
  [
    "prompt_refine_anubis_8b",
    "TheDrummer/Anubis-Mini-8B-v1 declares no license; it derives from a Llama-3.3-8B-Instruct finetune, so Meta's Llama 3.3 Community License likely governs — needs confirmation.",
  ],
]);

/** Strip `//` and `/* *​/` comments from JSONC without touching string contents. */
function stripJsonc(source) {
  return source.replace(
    /\\"|"(?:\\"|[^"])*"|(\/\/.*|\/\*[\s\S]*?\*\/)/g,
    (match, comment) => (comment ? "" : match),
  );
}

const catalog = JSON.parse(stripJsonc(fs.readFileSync(CATALOG, "utf8")));
const manifest = JSON.parse(fs.readFileSync(MANIFEST, "utf8"));
const bundledJs = fs.readFileSync(BUNDLED_JS, "utf8");
const sourceAudit = JSON.parse(fs.readFileSync(SOURCE_AUDIT, "utf8"));

// A model "ships weights" when it declares any download entry. Entries with no
// downloads (pure API/passthrough rows) redistribute nothing and need no notice.
const shipped = new Map(
  (catalog.models ?? [])
    .filter((model) => Array.isArray(model.downloads) && model.downloads.length > 0)
    .map((model) => [model.id, model]),
);

const errors = [];
const claimed = new Map();
const components = new Map(
  (manifest.components ?? []).map((component) => [component.id, component]),
);

function validateSourceAudit(audit, componentIndex, pinText, lockText, candidateText) {
  const auditErrors = [];
  const inferencePins = new Set(
    [...pinText.matchAll(/github\.com\/SceneWorks\/inference"[^}\n]*\brev\s*=\s*"([0-9a-f]{40})"/g)]
      .map((match) => match[1]),
  );
  if (inferencePins.size !== 1) {
    auditErrors.push(`expected exactly one inference revision across Cargo manifests, found: ${[...inferencePins].join(", ") || "none"}.`);
  } else if (!inferencePins.has(audit.inferenceRevision)) {
    auditErrors.push(
      `inference source audit is for ${audit.inferenceRevision}, but Cargo pins ${[...inferencePins][0]}. Re-audit inference NOTICE, LICENSE-*, and production include_str!/include_bytes! sites, then update config/inference-third-party-source.json.`,
    );
  }

  const canonical = JSON.stringify({
    artifacts: audit.artifacts,
    prospectiveDisclosures: audit.prospectiveDisclosures,
    provenanceScan: audit.provenanceScan,
    portedSourceAreas: audit.portedSourceAreas,
    includeSites: audit.includeSites,
  });
  const digest = crypto.createHash("sha256").update(canonical).digest("hex");
  if (audit.auditDigest !== digest) {
    auditErrors.push(`inference source audit digest mismatch: expected ${audit.auditDigest}, computed ${digest}. Re-run the exact pinned-revision audit; do not edit sites piecemeal.`);
  }

  const sourceIds = new Set();
  for (const artifact of audit.artifacts ?? []) {
    if (sourceIds.has(artifact.id)) {
      auditErrors.push(`inference source audit contains duplicate artifact id "${artifact.id}".`);
    }
    sourceIds.add(artifact.id);
    const component = componentIndex.get(artifact.component);
    if (!component) {
      auditErrors.push(`inference ${artifact.kind} "${artifact.id}" has no About→Licenses component "${artifact.component}".`);
      continue;
    }
    if (!Array.isArray(component.documents) || component.documents.length === 0) {
      auditErrors.push(`inference ${artifact.kind} "${artifact.id}" component "${artifact.component}" has no license document.`);
    }
  }

  const prospectiveIds = new Set();
  for (const disclosure of audit.prospectiveDisclosures ?? []) {
    prospectiveIds.add(disclosure.id);
    if (sourceIds.has(disclosure.id)) {
      auditErrors.push(`"${disclosure.id}" cannot be both pinned and prospective.`);
    }
    if (!componentIndex.has(disclosure.component)) {
      auditErrors.push(`prospective disclosure "${disclosure.id}" has no About→Licenses component "${disclosure.component}".`);
    }
  }

  const siteKeys = new Set();
  const validDispositions = new Set([
    "artifact",
    "model-asset",
    "shared-model-asset",
    "generated-numeric-data",
    "first-party-source",
    "model-data-no-separate-notice",
    "generated-build-output",
  ]);
  for (const site of audit.includeSites ?? []) {
    const key = `${site.source}|${site.included}`;
    if (siteKeys.has(key)) auditErrors.push(`duplicate audited include site "${key}".`);
    siteKeys.add(key);
    if (!validDispositions.has(site.disposition)) {
      auditErrors.push(`audited include site "${site.source}" has invalid disposition "${site.disposition}".`);
    }
    if (!site.evidence && site.disposition !== "artifact") {
      auditErrors.push(`audited include site "${site.source}" has no evidence-based disposition.`);
    }
    if (site.artifact && !sourceIds.has(site.artifact)) {
      auditErrors.push(`audited include site "${site.source}" maps to unknown artifact "${site.artifact}".`);
    }
    for (const component of [site.component, ...(site.components ?? [])].filter(Boolean)) {
      if (!componentIndex.has(component)) {
        auditErrors.push(`audited include site "${site.source}" maps to missing About component "${component}".`);
      }
    }
  }
  if (siteKeys.size === 0) auditErrors.push("inference source audit has no production include sites.");
  let candidates = [];
  try {
    candidates = parseProvenanceCandidates(candidateText);
  } catch (error) {
    auditErrors.push(`ported-source candidate inventory is malformed: ${error.message}`);
  }
  const candidatePopulationHash = populationSha256(candidates);
  if (audit.provenanceScan?.matchedFiles !== candidates.length) {
    auditErrors.push(`ported-source population count changed: audit says ${audit.provenanceScan?.matchedFiles}, inventory has ${candidates.length}.`);
  }
  if (audit.provenanceScan?.populationSha256 !== candidatePopulationHash) {
    auditErrors.push(`ported-source population hash changed: audit says ${audit.provenanceScan?.populationSha256}, inventory computes ${candidatePopulationHash}.`);
  }
  const areas = new Map();
  for (const area of audit.portedSourceAreas ?? []) {
    if (!area.id || areas.has(area.id)) {
      auditErrors.push(`ported-source area has missing or duplicate id "${area.id ?? ""}".`);
    }
    areas.set(area.id, area);
    if (!area.provenance || !area.disposition || (!area.pathPrefix && !area.paths)) {
      auditErrors.push(`ported-source area "${area.id}" lacks paths/provenance/disposition.`);
    }
    if (area.component && !componentIndex.has(area.component)) {
      auditErrors.push(`ported-source area "${area.id}" maps to missing About component "${area.component}".`);
    }
    if (!area.component && !area.evidence) {
      auditErrors.push(`excluded ported-source area "${area.id}" has no evidence.`);
    }
  }
  const candidatePaths = new Set();
  const usedAreas = new Set();
  for (const candidate of candidates) {
    if (candidatePaths.has(candidate.path)) {
      auditErrors.push(`duplicate ported-source candidate "${candidate.path}".`);
    }
    candidatePaths.add(candidate.path);
    const matches = [...areas.values()].filter((area) =>
      area.paths?.includes(candidate.path) ||
      (area.pathPrefix && candidate.path.startsWith(area.pathPrefix) &&
       !area.excludePaths?.includes(candidate.path)));
    if (matches.length !== 1) {
      auditErrors.push(`ported-source candidate "${candidate.path}" matches ${matches.length} disposition areas (expected exactly one).`);
      continue;
    }
    if (matches[0].id !== candidate.area) {
      auditErrors.push(`ported-source candidate "${candidate.path}" declares "${candidate.area}" but matches "${matches[0].id}".`);
    }
    usedAreas.add(matches[0].id);
  }
  for (const area of areas.keys()) {
    if (!usedAreas.has(area)) auditErrors.push(`ported-source area "${area}" has no candidates.`);
  }
  return auditErrors;
}

function validateDesktopNoticeContract(
  packageJson,
  tauriConfig,
  rustApiSource,
  bundledSource,
  desktopPackage,
  buildSidecar,
  buildPlan,
) {
  const contractErrors = [];
  if (tauriConfig.build?.frontendDist !== "ui") {
    contractErrors.push('Tauri build.frontendDist must remain "ui" for the signed desktop bootstrap.');
  }
  if (tauriConfig.bundle?.licenseFile || tauriConfig.bundle?.["license-file"]) {
    contractErrors.push("Tauri bundle.licenseFile must not become a competing third-party notice corpus.");
  }
  if (tauriConfig.build?.beforeBuildCommand !== "node scripts/build-sidecar.mjs") {
    contractErrors.push("Tauri beforeBuildCommand must invoke build-sidecar.mjs.");
  }
  if (!desktopPackage.scripts?.build?.includes("tauri build")) {
    contractErrors.push("desktop build script must invoke tauri build.");
  }
  if (!buildSidecar.includes('from "./build-sidecar-platform.mjs"') ||
      !buildSidecar.includes("sidecarBuildPlan(process.platform, process.env)") ||
      !buildSidecar.includes('run(npmCmd, ["run", buildPlan.npmScript]')) {
    contractErrors.push("build-sidecar must select and execute sidecarBuildPlan's embedded npm script.");
  }
  for (const platformContract of [
    'if (platform === "linux") return true',
    'platform === "win32"',
    'npmScript: "api:build:embedded"',
    'npmScript: "api:build:embedded:candle"',
  ]) {
    if (!buildPlan.includes(platformContract)) {
      contractErrors.push(`sidecar platform plan lost supported-platform contract: ${platformContract}`);
    }
  }
  if (!packageJson.scripts?.["api:build:embedded"]?.includes("web:build") ||
      !packageJson.scripts?.["api:build:embedded"]?.includes("embed-web")) {
    contractErrors.push("api:build:embedded must build and compile the web corpus into the Rust API sidecar.");
  }
  if (!packageJson.scripts?.["api:build:embedded:candle"]?.includes("web:build") ||
      !packageJson.scripts?.["api:build:embedded:candle"]?.includes("embed-web,backend-candle")) {
    contractErrors.push("api:build:embedded:candle must retain web:build plus embed-web,backend-candle.");
  }
  if (!rustApiSource.includes('#[folder = "../web/dist"]') ||
      !rustApiSource.includes("struct WebAssets")) {
    contractErrors.push("rust-api embed-web must embed apps/web/dist in the packaged sidecar.");
  }
  if (!(tauriConfig.bundle?.externalBin ?? []).includes("binaries/sceneworks-api")) {
    contractErrors.push("Tauri bundle.externalBin must package binaries/sceneworks-api.");
  }
  for (const key of ["cephes-bsd-3-clause", "cmudict-bsd-2-clause"]) {
    if (!bundledSource.includes(`"${key}"`)) {
      contractErrors.push(`packaged web notice mapping "${key}" is missing.`);
    }
  }
  return contractErrors;
}

for (const component of manifest.components ?? []) {
  if (!Array.isArray(component.models)) {
    errors.push(
      `component "${component.id}" has no \`models\` array. Every entry must declare which catalog ids it covers (use [] for a bundled binary or a co-requisite-only entry).`,
    );
    continue;
  }
  for (const id of component.models) {
    if (!shipped.has(id)) {
      errors.push(
        `component "${component.id}" claims model "${id}", which is not a downloading model in builtin.models.jsonc (renamed or removed?).`,
      );
    }
    if (claimed.has(id)) {
      errors.push(
        `model "${id}" is claimed by BOTH "${claimed.get(id)}" and "${component.id}" — attribution must be unambiguous.`,
      );
    }
    claimed.set(id, component.id);
  }

  // A document whose text never resolves renders as an empty license — the page
  // would then claim coverage it does not actually show.
  for (const doc of component.documents ?? []) {
    if (!bundledJs.includes(`"${doc.key}"`)) {
      errors.push(
        `component "${component.id}" document key "${doc.key}" is not wired in apps/web/src/data/bundledLicenses.js (import the text and add it to DOCUMENT_TEXT).`,
      );
    }
  }
  const dir = path.join(LICENSES_DIR, component.id);
  if (component.documents?.length && !fs.existsSync(dir)) {
    errors.push(`component "${component.id}" has documents but no apps/desktop/licenses/${component.id}/ directory.`);
  }
}

// Inference is a separate repository and its sources are not available in a clean
// SceneWorks CI checkout. Keep discovery deterministic by pinning the audit to the
// exact inference revision consumed here. Any pin bump therefore fails until a
// reviewer re-audits inference NOTICE/LICENSE-* plus production include_str!/
// include_bytes! sites and updates this inventory.
const pinSources = [
  fs.readFileSync(WORKER_CARGO, "utf8"),
  fs.readFileSync(ROOT_CARGO, "utf8"),
].join("\n");
const lock = fs.readFileSync(CARGO_LOCK, "utf8");
const provenanceCandidates = fs.readFileSync(PROVENANCE_CANDIDATES, "utf8");
errors.push(...validateSourceAudit(sourceAudit, components, pinSources, lock, provenanceCandidates));
const packageJson = JSON.parse(fs.readFileSync(PACKAGE_JSON, "utf8"));
const tauriConfig = JSON.parse(fs.readFileSync(TAURI_CONFIG, "utf8"));
const rustApiSource = fs.readFileSync(RUST_API_LIB, "utf8");
const desktopPackage = JSON.parse(fs.readFileSync(DESKTOP_PACKAGE, "utf8"));
const buildSidecar = fs.readFileSync(BUILD_SIDECAR, "utf8");
const buildPlan = fs.readFileSync(BUILD_PLAN, "utf8");
errors.push(...validateDesktopNoticeContract(
  packageJson, tauriConfig, rustApiSource, bundledJs, desktopPackage, buildSidecar, buildPlan,
));

if (process.argv.includes("--self-test")) {
  const withoutSite = structuredClone(sourceAudit);
  withoutSite.includeSites.pop();
  if (!validateSourceAudit(withoutSite, components, pinSources, lock, provenanceCandidates).some((error) => error.includes("digest mismatch"))) {
    console.error("self-test: deleting an audited include site did not fail closed");
    process.exit(1);
  }
  const addedSite = structuredClone(sourceAudit);
  addedSite.includeSites.push({ source: "new.rs:1", included: "new.dat", disposition: "first-party-source", evidence: "mutation" });
  if (!validateSourceAudit(addedSite, components, pinSources, lock, provenanceCandidates).some((error) => error.includes("digest mismatch"))) {
    console.error("self-test: adding an unaudited include site did not fail closed");
    process.exit(1);
  }
  const staleRevision = structuredClone(sourceAudit);
  staleRevision.inferenceRevision = "0".repeat(40);
  if (!validateSourceAudit(staleRevision, components, pinSources, lock, provenanceCandidates).some((error) => error.includes("but Cargo pins"))) {
    console.error("self-test: stale inference revision did not fail closed");
    process.exit(1);
  }
  const candidateLines = provenanceCandidates.trimEnd().split("\n");
  const provenanceMutations = [
    [
      "delete candidate",
      sourceAudit,
      [...candidateLines.slice(0, 1), ...candidateLines.slice(2), ""].join("\n"),
      "population count changed",
    ],
    [
      "add candidate",
      sourceAudit,
      [...candidateLines, candidateLines[1].replace(/^[^\t]+/, "crates/new-port/src/lib.rs"), ""].join("\n"),
      "population count changed",
    ],
    [
      "alter blob",
      sourceAudit,
      provenanceCandidates.replace(/\t[0-9a-f]{40}\t/, `\t${"0".repeat(40)}\t`),
      "population hash changed",
    ],
    [
      "unmatched candidate",
      sourceAudit,
      provenanceCandidates.replace(/\tarchitecture:[^\t\n]+/, "\tmissing-area"),
      "declares \"missing-area\"",
    ],
  ];
  const multiplyMatched = structuredClone(sourceAudit);
  multiplyMatched.portedSourceAreas.find((area) => area.id === "cfgpp-formula").paths.push(
    parseProvenanceCandidates(provenanceCandidates)[0].path,
  );
  provenanceMutations.push([
    "multiply-matched candidate",
    multiplyMatched,
    provenanceCandidates,
    "matches 2 disposition areas",
  ]);
  for (const [label, mutatedAudit, mutatedCandidates, expected] of provenanceMutations) {
    if (!validateSourceAudit(mutatedAudit, components, pinSources, lock, mutatedCandidates)
        .some((error) => error.includes(expected))) {
      console.error(`self-test: ${label} mutation did not fail closed with "${expected}"`);
      process.exit(1);
    }
  }
  const missingMapping = bundledJs.replace('"cmudict-bsd-2-clause"', '"removed-cmudict-key"');
  if (!validateDesktopNoticeContract(packageJson, tauriConfig, rustApiSource, missingMapping, desktopPackage, buildSidecar, buildPlan)
      .some((error) => error.includes("cmudict-bsd-2-clause"))) {
    console.error("self-test: removing a packaged notice mapping did not fail closed");
    process.exit(1);
  }
  const contractMutations = [
    ["beforeBuildCommand", packageJson, { ...tauriConfig, build: { ...tauriConfig.build, beforeBuildCommand: "broken" } }, rustApiSource, bundledJs, desktopPackage, buildSidecar, buildPlan],
    ["build-sidecar plan import", packageJson, tauriConfig, rustApiSource, bundledJs, desktopPackage, buildSidecar.replace('from "./build-sidecar-platform.mjs"', 'from "./broken.mjs"'), buildPlan],
    ["mac embedded selection", packageJson, tauriConfig, rustApiSource, bundledJs, desktopPackage, buildSidecar, buildPlan.replace('npmScript: "api:build:embedded"', 'npmScript: "broken"')],
    ["candle embedded selection", packageJson, tauriConfig, rustApiSource, bundledJs, desktopPackage, buildSidecar, buildPlan.replace('npmScript: "api:build:embedded:candle"', 'npmScript: "broken"')],
    ["candle web embed script", { ...packageJson, scripts: { ...packageJson.scripts, "api:build:embedded:candle": "cargo build --features backend-candle" } }, tauriConfig, rustApiSource, bundledJs, desktopPackage, buildSidecar, buildPlan],
    ["externalBin", packageJson, { ...tauriConfig, bundle: { ...tauriConfig.bundle, externalBin: [] } }, rustApiSource, bundledJs, desktopPackage, buildSidecar, buildPlan],
  ];
  for (const [label, ...args] of contractMutations) {
    if (validateDesktopNoticeContract(...args).length === 0) {
      console.error(`self-test: breaking ${label} did not fail closed`);
      process.exit(1);
    }
  }
  console.log("[license-coverage] self-test PASS — provenance population/dispositions, audit, and every Tauri→sidecar→embedded-web packaging-link mutation were rejected.");
}

for (const [id] of shipped) {
  if (claimed.has(id) || UNDETERMINED.has(id)) continue;
  errors.push(
    `model "${id}" ships weights but has NO About→Licenses entry. Add its upstream license under apps/desktop/licenses/<component>/, list the id in that component's \`models\`, and wire the document key in bundledLicenses.js.`,
  );
}

for (const [id, reason] of UNDETERMINED) {
  if (!shipped.has(id)) {
    errors.push(`UNDETERMINED entry "${id}" is no longer a shipped model — drop it from check-license-coverage.mjs.`);
  } else if (claimed.has(id)) {
    errors.push(
      `model "${id}" is now covered by "${claimed.get(id)}" — remove it from the UNDETERMINED list in check-license-coverage.mjs.`,
    );
  } else {
    console.warn(`[license-coverage] UNDETERMINED ${id}: ${reason}`);
  }
}

if (errors.length > 0) {
  console.error("License coverage check FAILED:\n");
  for (const error of errors) console.error(`  - ${error}`);
  console.error(
    `\n${errors.length} problem(s). The About→Licenses page must record every model whose weights SceneWorks downloads.`,
  );
  process.exit(1);
}

console.log(
  `[license-coverage] OK — ${claimed.size}/${shipped.size} shipped models covered by ${manifest.components.length} components (${UNDETERMINED.size} undetermined upstream).`,
);
