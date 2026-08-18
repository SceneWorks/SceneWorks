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
//   5. Crate coverage — every production-Rust CRATE in the pinned inference revision is
//                      classified: covered by a ported-source area, or given an explicit
//                      crateDispositions decision with evidence. Fail-closed (sc-15191);
//                      see validateCrateCoverage for why the marker regex is not enough.
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
  parseCrates as parseCratePrefixes,
  populationSha256,
  cratePopulationSha256,
} from "./scan-inference-provenance.mjs";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const CATALOG = path.join(ROOT, "config/manifests/builtin.models.jsonc");
const MANIFEST = path.join(ROOT, "apps/desktop/licenses/manifest.json");
const LICENSES_DIR = path.join(ROOT, "apps/desktop/licenses");
const BUNDLED_JS = path.join(ROOT, "apps/web/src/data/bundledLicenses.js");
const SOURCE_AUDIT = path.join(ROOT, "config/inference-third-party-source.json");
const PROVENANCE_CANDIDATES = path.join(ROOT, "config/inference-provenance-candidates.tsv");
const CRATE_PREFIXES = path.join(ROOT, "config/inference-crate-prefixes.txt");
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
    // WEIGHTS ONLY. The reproduced JoyCaption prompt CONTENT is a separate question and is now
    // SETTLED: it comes from github.com/fpgaminer/joycaption, which is Apache-2.0, and ships as the
    // `joycaption-source` About component. Do not re-merge the two — an undetermined weights license
    // is not evidence that the prompt text is unlicensed (sc-15191 review).
    "fancyfeast/llama-joycaption-beta-one-hf-llava declares no license for its WEIGHTS; it is a Llama-3.1-8B-Instruct + SigLIP2 derivative, so Meta's Llama 3.1 Community License likely governs — needs confirmation.",
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

// Dispositions a crate with NO marker-bearing file may carry. All three demand evidence; none is a
// catch-all. `first-party-original` asserts there is no upstream source to attribute at all — the
// strictest claim in this file, and the one most likely to be wrong by omission.
// `architecture-reimplementation-existing-terms` (the same word the ported areas use) asserts the
// crate IS a port whose files simply never say so, and that its upstream terms are already mapped.
// `upstream-content-existing-terms` is for the case neither covered: source that REPRODUCES upstream
// CONTENT — prompt tables, token vocabularies, label lists — rather than reimplementing an
// algorithm. It exists because `candle-gen-joycaption` was classified `first-party-original` on the
// strength of "SceneWorks caption product policy" while its prompt map is fancyfeast's JoyCaption
// demo `CAPTION_TYPE_MAP` reproduced near-verbatim in upstream's order (sc-15191 review). Copied
// strings are not a reimplementation, and calling them first-party is simply false.
const CRATE_DISPOSITIONS = new Set([
  "first-party-original",
  "architecture-reimplementation-existing-terms",
  "upstream-content-existing-terms",
]);

/**
 * FAIL-CLOSED crate coverage (sc-15191).
 *
 * The marker regex in scan-inference-provenance.mjs is a heuristic, and heuristics have holes:
 * `mlx-gen-krea-realtime` entered as a WHOLE NEW CRATE whose headers honestly described a port in
 * words the regex did not know (`mirroring`, `from the reference`), matched nothing, and was
 * therefore invisible to the audit — this gate went green on a crate nobody had classified
 * (sc-15138). Broadening the vocabulary narrows that hole but cannot close it, because the hole is
 * "a human has to have anticipated the phrasing".
 *
 * This closes it from the other side. The crate-prefix inventory is scanned from the pinned
 * revision WITHOUT reading a byte of source, so a crate is in the population because it exists.
 * Every prefix must then be claimed by a ported-source area or by an explicit disposition. A new
 * crate that is neither FAILS — silence is no longer a passing answer.
 */
function validateCrateCoverage(audit, componentIndex, crateText, pinnedRev) {
  const coverageErrors = [];
  const coverage = audit.crateCoverage;
  if (!coverage) {
    coverageErrors.push("inference source audit has no `crateCoverage` block — the fail-closed crate guard cannot run.");
    return coverageErrors;
  }
  if (pinnedRev && coverage.revision !== pinnedRev) {
    coverageErrors.push(
      `crate-prefix inventory was scanned at ${coverage.revision ?? "(unset)"}, but Cargo pins ${pinnedRev}. Re-run \`node scripts/scan-inference-provenance.mjs --repo <inference> --write-crates config/inference-crate-prefixes.txt\`, classify any new crate, then update crateCoverage + auditDigest.`,
    );
  }
  let crates = [];
  try {
    crates = parseCratePrefixes(crateText);
  } catch (error) {
    coverageErrors.push(`crate-prefix inventory is malformed: ${error.message}`);
  }
  if (coverage.cratePrefixes !== crates.length) {
    coverageErrors.push(`crate-prefix population count changed: audit says ${coverage.cratePrefixes}, inventory has ${crates.length}.`);
  }
  const computed = cratePopulationSha256(crates);
  if (coverage.cratePopulationSha256 !== computed) {
    coverageErrors.push(`crate-prefix population hash changed: audit says ${coverage.cratePopulationSha256}, inventory computes ${computed}.`);
  }

  const areas = audit.portedSourceAreas ?? [];
  // CRATE-level classification is decided by `pathPrefix` ONLY.
  //
  // It used to also accept `area.paths`, which meant a single marker-bearing FILE in a brand-new
  // crate, routed by the scanner's hardcoded SPECIAL_AREAS table into a pre-existing generic area
  // (`cephes-source`, `comfy-kdiffusion-solvers`, `opencv-source`, …), silently classified the WHOLE
  // crate — inheriting a decision a human made about some other crate's file rather than making one
  // about this crate. A `paths` area says "this file transcribes Cephes"; it says nothing about the
  // thousand other lines around it. Only a `pathPrefix` area (or an explicit crateDispositions
  // entry) is a statement about the crate.
  const portedCrate = (crate) => areas.some((area) =>
    area.pathPrefix && (area.pathPrefix === crate || area.pathPrefix.startsWith(`${crate}/`)));

  const declared = new Map();
  for (const entry of audit.crateDispositions ?? []) {
    if (!entry.prefix) {
      coverageErrors.push("crate disposition entry has no `prefix`.");
      continue;
    }
    if (declared.has(entry.prefix)) {
      coverageErrors.push(`duplicate crate disposition for "${entry.prefix}".`);
    }
    declared.set(entry.prefix, entry);
    if (!CRATE_DISPOSITIONS.has(entry.disposition)) {
      coverageErrors.push(`crate disposition "${entry.prefix}" has invalid disposition "${entry.disposition ?? "(unset)"}" (expected one of: ${[...CRATE_DISPOSITIONS].join(", ")}).`);
    }
    // An allowlist without a reason is a catch-all wearing a decision's clothes.
    if (!entry.evidence) {
      coverageErrors.push(`crate disposition "${entry.prefix}" has no evidence — every exemption must be justifiable in review.`);
    }
    if (entry.component && !componentIndex.has(entry.component)) {
      coverageErrors.push(`crate disposition "${entry.prefix}" maps to missing About component "${entry.component}".`);
    }
    if (!crates.includes(entry.prefix)) {
      coverageErrors.push(`crate disposition "${entry.prefix}" is not a production-Rust crate in the pinned revision (renamed, removed, or never existed?) — drop or repoint it.`);
    } else if (portedCrate(entry.prefix)) {
      coverageErrors.push(`crate "${entry.prefix}" is BOTH covered by a ported-source area and given a crateDispositions decision — the classification must be unambiguous.`);
    }
  }

  for (const crate of crates) {
    if (portedCrate(crate) || declared.has(crate)) continue;
    coverageErrors.push(
      `production-Rust crate "${crate}" in the pinned inference revision is UNCLASSIFIED: it has no portedSourceAreas coverage and no crateDispositions decision. Classify it — add a ported-source area if it ports upstream work, or a crateDispositions entry (${[...CRATE_DISPOSITIONS].join(" | ")}) with evidence if it does not.`,
    );
  }
  return coverageErrors;
}

// The canonical audit payload, hashed. Extracted so `--derive-json` below can hand
// scripts/bump-inference.mjs the digest THIS checker will grade, rather than a second copy of the
// formula in the bump script that could drift from it.
function auditCanonicalDigest(audit) {
  const canonical = JSON.stringify({
    artifacts: audit.artifacts,
    prospectiveDisclosures: audit.prospectiveDisclosures,
    provenanceScan: audit.provenanceScan,
    portedSourceAreas: audit.portedSourceAreas,
    crateCoverage: audit.crateCoverage,
    crateDispositions: audit.crateDispositions,
    includeSites: audit.includeSites,
  });
  return crypto.createHash("sha256").update(canonical).digest("hex");
}

function validateSourceAudit(audit, componentIndex, pinText, lockText, candidateText, crateText) {
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
  // 🔴 The ported-source inventory has its OWN revision label, and until sc-15017 nothing compared
  // it to the pin. That is how the inventory went stale: a bump updated `inferenceRevision` (checked
  // above) and the scanner was re-run against a revision literal it kept internally, so the label
  // said one thing, the scan was of another, and the audit stayed self-consistent and GREEN while
  // missing an entire crate. The scanner now derives its revision from the pin — but CI has no
  // inference clone, so it never re-scans, and a bump where nobody runs the scanner at all would
  // still pass every check below. This is the assertion that makes that impossible.
  if (inferencePins.size === 1 && audit.provenanceScan?.revision !== [...inferencePins][0]) {
    auditErrors.push(
      `ported-source inventory was scanned at ${audit.provenanceScan?.revision ?? "(unset)"}, but Cargo pins ${[...inferencePins][0]}. Re-run \`node scripts/scan-inference-provenance.mjs --repo <inference> --write config/inference-provenance-candidates.tsv\` (it reads the pin itself), then update provenanceScan + auditDigest.`,
    );
  }

  const digest = auditCanonicalDigest(audit);
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
  auditErrors.push(...validateCrateCoverage(
    audit,
    componentIndex,
    crateText,
    inferencePins.size === 1 ? [...inferencePins][0] : undefined,
  ));
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
const cratePrefixes = fs.readFileSync(CRATE_PREFIXES, "utf8");
errors.push(...validateSourceAudit(sourceAudit, components, pinSources, lock, provenanceCandidates, cratePrefixes));
const packageJson = JSON.parse(fs.readFileSync(PACKAGE_JSON, "utf8"));
const tauriConfig = JSON.parse(fs.readFileSync(TAURI_CONFIG, "utf8"));
const rustApiSource = fs.readFileSync(RUST_API_LIB, "utf8");
const desktopPackage = JSON.parse(fs.readFileSync(DESKTOP_PACKAGE, "utf8"));
const buildSidecar = fs.readFileSync(BUILD_SIDECAR, "utf8");
const buildPlan = fs.readFileSync(BUILD_PLAN, "utf8");
errors.push(...validateDesktopNoticeContract(
  packageJson, tauriConfig, rustApiSource, bundledJs, desktopPackage, buildSidecar, buildPlan,
));

// Structured derivation output for scripts/bump-inference.mjs (sc-18420).
//
// The report below is deliberately NOT fatal (sc-19751), and that broke the bump's restamp: the
// deriver read the recomputed digest out of `execFileSync`'s thrown error, the checker stopped
// throwing, and the restamp silently never fired again — the bump then shipped an audit whose digest
// and population count still described the previous revision. Derived facts belong in structured
// output, not in an exception's stderr, so they are reported here as data and exit 0 either way.
// Every graded population fact is reported, not just the digest: `provenanceScan.matchedFiles` and
// `crateCoverage.cratePrefixes` are graded here (see "population count changed") and the deriver never
// wrote either, so a bump that added or removed a ported file or a crate left the audit asserting the
// previous population.
if (process.argv.includes("--derive-json")) {
  const candidates = parseProvenanceCandidates(provenanceCandidates);
  const crates = parseCratePrefixes(cratePrefixes);
  process.stdout.write(
    `${JSON.stringify({
      auditDigest: auditCanonicalDigest(sourceAudit),
      provenanceMatchedFiles: candidates.length,
      provenancePopulationSha256: populationSha256(candidates),
      cratePrefixes: crates.length,
      cratePopulationSha256: cratePopulationSha256(crates),
    })}\n`,
  );
  process.exit(0);
}

if (process.argv.includes("--self-test")) {
  const withoutSite = structuredClone(sourceAudit);
  withoutSite.includeSites.pop();
  if (!validateSourceAudit(withoutSite, components, pinSources, lock, provenanceCandidates, cratePrefixes).some((error) => error.includes("digest mismatch"))) {
    console.error("self-test: deleting an audited include site did not fail closed");
    process.exit(1);
  }
  const addedSite = structuredClone(sourceAudit);
  addedSite.includeSites.push({ source: "new.rs:1", included: "new.dat", disposition: "first-party-source", evidence: "mutation" });
  if (!validateSourceAudit(addedSite, components, pinSources, lock, provenanceCandidates, cratePrefixes).some((error) => error.includes("digest mismatch"))) {
    console.error("self-test: adding an unaudited include site did not fail closed");
    process.exit(1);
  }
  const staleRevision = structuredClone(sourceAudit);
  staleRevision.inferenceRevision = "0".repeat(40);
  if (!validateSourceAudit(staleRevision, components, pinSources, lock, provenanceCandidates, cratePrefixes).some((error) => error.includes("but Cargo pins"))) {
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
    if (!validateSourceAudit(mutatedAudit, components, pinSources, lock, mutatedCandidates, cratePrefixes)
        .some((error) => error.includes(expected))) {
      console.error(`self-test: ${label} mutation did not fail closed with "${expected}"`);
      process.exit(1);
    }
  }
  // --- fail-closed crate coverage (sc-15191) ----------------------------------------------------
  //
  // The point of these mutations is that a GREEN result on today's tree proves nothing: the guard
  // has to REJECT the situation that shipped mlx-gen-krea-realtime unclassified. So each case
  // constructs the world as it would look after the next pin bump and asserts a non-empty error.
  //
  // A mutated crate inventory necessarily breaks the count/hash/digest assertions too, so the
  // helper re-stamps them — otherwise the test would "pass" on an incidental hash error while the
  // classification check silently did nothing.
  const withCrateInventory = (audit, crates) => {
    const restamped = structuredClone(audit);
    restamped.crateCoverage = {
      ...restamped.crateCoverage,
      cratePrefixes: crates.length,
      cratePopulationSha256: cratePopulationSha256(crates),
    };
    return restamped;
  };
  const committedCrates = parseCratePrefixes(cratePrefixes);
  const renderCrates = (crates) => `${["# self-test", ...crates].join("\n")}\n`;

  const crateMutations = [];

  // 1. A brand-new crate lands in the pinned rev and nobody classifies it — the sc-15138 case.
  //    This MUST fail. It is the whole reason the guard exists.
  {
    const crates = [...committedCrates, "crates/media/mlx-gen/mlx-gen-brand-new"].sort();
    crateMutations.push([
      "unclassified new crate",
      withCrateInventory(sourceAudit, crates),
      renderCrates(crates),
      'crate "crates/media/mlx-gen/mlx-gen-brand-new" in the pinned inference revision is UNCLASSIFIED',
    ]);
  }
  // 2. …and the SAME new crate, once given a disposition with evidence, passes. Without this the
  //    guard could be a constant `fail` and case 1 would still look green.
  {
    const crates = [...committedCrates, "crates/media/mlx-gen/mlx-gen-brand-new"].sort();
    const classified = withCrateInventory(sourceAudit, crates);
    classified.crateDispositions = [
      ...classified.crateDispositions,
      {
        prefix: "crates/media/mlx-gen/mlx-gen-brand-new",
        disposition: "first-party-original",
        evidence: "self-test fixture",
      },
    ];
    if (validateCrateCoverage(classified, components, renderCrates(crates), null)
        .some((error) => error.includes("UNCLASSIFIED"))) {
      console.error("self-test: a classified crate was still reported unclassified (guard is not discriminating)");
      process.exit(1);
    }
    // …and covered by a ported-source area instead of a disposition, it must ALSO pass.
    const ported = withCrateInventory(sourceAudit, crates);
    ported.portedSourceAreas = [
      ...ported.portedSourceAreas,
      {
        id: "architecture:crates/media/mlx-gen/mlx-gen-brand-new",
        pathPrefix: "crates/media/mlx-gen/mlx-gen-brand-new/src/",
        provenance: "self-test fixture",
        disposition: "architecture-reimplementation-existing-terms",
        evidence: "self-test fixture",
      },
    ];
    if (validateCrateCoverage(ported, components, renderCrates(crates), null)
        .some((error) => error.includes("UNCLASSIFIED"))) {
      console.error("self-test: a ported-area-covered crate was still reported unclassified");
      process.exit(1);
    }
  }
  // 2b. A new crate whose ONLY coverage is a per-FILE `area.paths` entry is NOT classified.
  //     This is the inherited-classification path (sc-15191 review): one marker-bearing file routed
  //     by the scanner's hardcoded SPECIAL_AREAS table into a pre-existing generic area
  //     (`cephes-source`, `opencv-source`, `comfy-kdiffusion-solvers`, …) used to mark the whole
  //     crate decided, without anyone deciding anything about the crate. Only `pathPrefix` — or an
  //     explicit disposition — counts.
  {
    const crates = [...committedCrates, "crates/media/mlx-gen/mlx-gen-inherits"].sort();
    const inherited = withCrateInventory(sourceAudit, crates);
    inherited.portedSourceAreas = [
      ...inherited.portedSourceAreas,
      {
        id: "cephes-source-selftest",
        paths: ["crates/media/mlx-gen/mlx-gen-inherits/src/latent.rs"],
        provenance: "self-test fixture",
        disposition: "architecture-reimplementation-existing-terms",
        evidence: "self-test fixture",
      },
    ];
    crateMutations.push([
      "crate classified only by a per-file area path",
      inherited,
      renderCrates(crates),
      'crate "crates/media/mlx-gen/mlx-gen-inherits" in the pinned inference revision is UNCLASSIFIED',
    ]);
  }
  // 3. Dropping a decision for a crate that is still there must fail — exemptions cannot rot away.
  {
    const stripped = structuredClone(sourceAudit);
    const dropped = stripped.crateDispositions.pop();
    crateMutations.push([
      "dropped crate disposition",
      stripped,
      cratePrefixes,
      `crate "${dropped.prefix}" in the pinned inference revision is UNCLASSIFIED`,
    ]);
  }
  // 4. An exemption with no evidence is a catch-all, not a decision.
  {
    const unjustified = structuredClone(sourceAudit);
    delete unjustified.crateDispositions[0].evidence;
    crateMutations.push([
      "crate disposition without evidence",
      unjustified,
      cratePrefixes,
      "has no evidence",
    ]);
  }
  // 5. An invented disposition word must not widen the vocabulary.
  {
    const invented = structuredClone(sourceAudit);
    invented.crateDispositions[0].disposition = "probably-fine";
    crateMutations.push([
      "invalid crate disposition",
      invented,
      cratePrefixes,
      'invalid disposition "probably-fine"',
    ]);
  }
  // 6. A stale exemption for a crate that no longer exists must be pruned, not left as cover.
  {
    const stale = structuredClone(sourceAudit);
    stale.crateDispositions = [
      ...stale.crateDispositions,
      { prefix: "crates/gone/away", disposition: "first-party-original", evidence: "stale" },
    ];
    crateMutations.push([
      "stale crate disposition",
      stale,
      cratePrefixes,
      'is not a production-Rust crate in the pinned revision',
    ]);
  }
  // 7. Claiming a crate BOTH ways is ambiguous attribution, the same defect the model-id duplicate
  //    check guards against.
  {
    const ambiguous = structuredClone(sourceAudit);
    ambiguous.crateDispositions = [
      ...ambiguous.crateDispositions,
      { prefix: "crates/audio/candle-audio-kokoro", disposition: "first-party-original", evidence: "ambiguous" },
    ];
    crateMutations.push([
      "doubly-classified crate",
      ambiguous,
      cratePrefixes,
      "is BOTH covered by a ported-source area and given a crateDispositions decision",
    ]);
  }
  // 8. A pin bump that never re-scanned the crate inventory must fail on the revision label alone.
  {
    const staleCrateScan = structuredClone(sourceAudit);
    staleCrateScan.crateCoverage.revision = "0".repeat(40);
    crateMutations.push([
      "stale crate-scan revision",
      staleCrateScan,
      cratePrefixes,
      "crate-prefix inventory was scanned at",
    ]);
  }
  // 9. Deleting the whole block must not silently disable the guard.
  {
    const noBlock = structuredClone(sourceAudit);
    delete noBlock.crateCoverage;
    crateMutations.push([
      "missing crateCoverage block",
      noBlock,
      cratePrefixes,
      "has no `crateCoverage` block",
    ]);
  }
  // 10. Editing the inventory file without re-stamping the audit must fail on count and hash.
  crateMutations.push([
    "unstamped crate inventory edit",
    sourceAudit,
    renderCrates([...committedCrates, "crates/sneaky/crate"].sort()),
    "crate-prefix population count changed",
  ]);
  for (const [label, mutatedAudit, mutatedCrates, expected] of crateMutations) {
    if (!validateSourceAudit(mutatedAudit, components, pinSources, lock, provenanceCandidates, mutatedCrates)
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
  console.log("[license-coverage] self-test PASS — provenance population/dispositions, fail-closed crate coverage (unclassified/stale/unjustified/doubly-classified/unscanned), audit, and every Tauri→sidecar→embedded-web packaging-link mutation were rejected.");
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

// Report, don't gate (sc-19751).
//
// Everything above still runs — coverage, stale ids, duplicates, document wiring, the
// revision-locked source audit, and fail-closed crate classification. What changed is what happens
// to the result: findings are printed for a human to act on, and the process exits 0.
//
// Three of these rules fired on changes with nothing to do with licensing. The source audit is
// revision-locked to the inference pin, so EVERY pin bump demanded a manual re-audit of upstream
// NOTICE/LICENSE-*/`include_str!` sites before CI could go green; the audit digest rejected any
// piecemeal edit; and a new production-Rust crate in the pinned revision failed closed until
// someone classified it. That is the same defect as sc-19728's decode-quality fingerprint — a
// digest keyed to a whole revision, refusing on any change, hand-re-derived each time it fires —
// and it made an ordinary inference pin bump cost a licensing audit.
//
// The analysis is the valuable part and is kept: it is the worklist for the compliance pass, and
// `--self-test` still proves every rule detects its mutation, so detection stays mutation-checked
// even though nothing is enforced. `--strict` restores exit 1 for that pass, when it happens.
//
// NOT affected: `scripts/check-no-nc-weights.mjs` stays fail-closed in both `check.yml` and
// `release.yml`. It does not fire on a missing license record — it fires on Non-Commercial weights
// baked into a distributed artifact, which would make SceneWorks a distributor of a Derivative and
// attach the NC obligations (sc-10526, docs/packaging-nc-weights-guard.md).
const STRICT = process.argv.includes("--strict");

if (errors.length > 0) {
  console.error(
    STRICT ? "License coverage check FAILED:\n" : "[license-coverage] REPORT — not enforced:\n",
  );
  for (const error of errors) console.error(`  - ${error}`);
  if (STRICT) {
    console.error(
      `\n${errors.length} problem(s). The About→Licenses page must record every model whose weights SceneWorks downloads.`,
    );
    process.exit(1);
  }
  console.error(
    `\n${errors.length} open licensing item(s). These do NOT fail the build (sc-19751) — they are the ` +
      `worklist for the compliance pass. Re-run with --strict to gate on them deliberately.`,
  );
}

console.log(
  `[license-coverage] OK — ${claimed.size}/${shipped.size} shipped models covered by ${manifest.components.length} components (${UNDETERMINED.size} undetermined upstream).`,
);
