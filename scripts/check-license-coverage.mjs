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
import { fileURLToPath } from "node:url";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const CATALOG = path.join(ROOT, "config/manifests/builtin.models.jsonc");
const MANIFEST = path.join(ROOT, "apps/desktop/licenses/manifest.json");
const LICENSES_DIR = path.join(ROOT, "apps/desktop/licenses");
const BUNDLED_JS = path.join(ROOT, "apps/web/src/data/bundledLicenses.js");
const SOURCE_AUDIT = path.join(ROOT, "config/inference-third-party-source.json");
const WORKER_CARGO = path.join(ROOT, "crates/sceneworks-worker/Cargo.toml");
const ROOT_CARGO = path.join(ROOT, "Cargo.toml");
const CARGO_LOCK = path.join(ROOT, "Cargo.lock");

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

function validateSourceAudit(audit, componentIndex, pinText, lockText) {
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
    if (artifact.package && !lockText.includes(`name = "${artifact.package}"`)) {
      auditErrors.push(`inference ${artifact.kind} "${artifact.id}" claims absent Cargo package "${artifact.package}".`);
    }
  }
  for (const required of ["cephes", "cmudict"]) {
    if (!sourceIds.has(required)) {
      auditErrors.push(`required ported/embedded inference artifact "${required}" is missing from config/inference-third-party-source.json.`);
    }
  }
  return auditErrors;
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
errors.push(...validateSourceAudit(sourceAudit, components, pinSources, lock));

if (process.argv.includes("--self-test")) {
  const withoutCmudict = structuredClone(sourceAudit);
  withoutCmudict.artifacts = withoutCmudict.artifacts.filter(({ id }) => id !== "cmudict");
  const missingErrors = validateSourceAudit(withoutCmudict, components, pinSources, lock);
  if (!missingErrors.some((error) => error.includes('required ported/embedded inference artifact "cmudict"'))) {
    console.error("self-test: removing CMUDICT did not fail closed");
    process.exit(1);
  }
  const staleRevision = structuredClone(sourceAudit);
  staleRevision.inferenceRevision = "0".repeat(40);
  if (!validateSourceAudit(staleRevision, components, pinSources, lock).some((error) => error.includes("but Cargo pins"))) {
    console.error("self-test: stale inference revision did not fail closed");
    process.exit(1);
  }
  console.log("[license-coverage] self-test PASS — missing disclosure and stale audit mutations were rejected.");
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
