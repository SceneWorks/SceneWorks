// Regenerate the derived live-preview support catalog (sc-16965, epic 16948).
// Run via `npm run gen:preview-support` (from apps/web). Emits BOTH:
//   - config/manifests/builtin.preview-support.jsonc  (the backend catalog block sceneworks-core
//                                                      embeds and rust-api merges onto /api/v1/models)
//   - apps/web/src/data/previewSupport.json           (the web app's offline-fallback table)
// from the SAME derivation, so the served and offline catalogs can never drift. The
// previewSupportCatalog.test.js drift guards fail CI if either committed artifact falls out of sync
// with its inputs, so this script is the only sanctioned way to update them.
//
// ## Stage 2 of two
//
// The inputs are both CHECKED IN, which is what lets the drift guard run on every ordinary PR:
//
//   1. config/engine-capabilities/capabilities.<backend>.json — the stage-1 dump of
//      `Capabilities.supports_preview` off the LINKED provider registry. Produced by
//      `cargo run -p sceneworks-worker --bin dump-engine-capabilities` on a lane that links
//      engines (macOS → mlx, `--features backend-candle` → candle). Re-run at every inference pin
//      bump; `scripts/bump-inference.mjs` fails closed on a stale one, beside the licence re-scan.
//   2. crates/sceneworks-worker/src/engines.rs — the static MODEL_TABLE joining SceneWorks model
//      ids to engine ids (many-to-one). Parsed, not mirrored, so a new row makes the artifacts
//      stale on the same PR that adds it.
//   3. config/engine-capabilities/audio/capabilities.<backend>.json — the same dump of the SEPARATE
//      audio registry (sc-17593), whose engine ids ARE SceneWorks model ids and so need no join.
//      Written by the same command on either lane: audio is candle-native everywhere.
//
// A single-stage generator would have had to link an engine registry itself, and both candle lanes
// (desktop-windows.yml, server-candle-linux.yml) are `workflow_dispatch`-only — its drift guard
// could never have run on an ordinary PR. That the stage-1 lane is dispatch-only is NOT an
// unguarded gap: sc-16951's `candle-gen-catalog` bidirectional test runs in the *inference* repo's
// CI on every PR, so descriptor-level truth is guarded continuously upstream; SceneWorks only needs
// to re-dump at pin time.
import { readdirSync, readFileSync, writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

import {
  assertBackendCoverage,
  catalogToWebPreviewSupport,
  derivePreviewSupport,
  parseEngineModelTable,
  parseSceneworksAudioBackends,
  parseSceneworksBackends,
} from "../src/data/previewSupportDerivation.js";

const factsDir = fileURLToPath(new URL("../../../config/engine-capabilities", import.meta.url));
// The audio registry's dumps (sc-17593). A SUBDIRECTORY, so `readdirSync` above — which is not
// recursive — cannot pick them up as media facts: the two registries share backend names ("candle"
// on every platform for audio) but have independent engine-id namespaces and different joins.
const audioFactsDir = `${factsDir}/audio`;
const enginesPath = fileURLToPath(
  new URL("../../../crates/sceneworks-worker/src/engines.rs", import.meta.url),
);
// The declared backend list (sc-17119). Read from the Rust const rather than the directory listing,
// because the directory listing cannot see a file that was never written.
const factsDeclarationPath = fileURLToPath(
  new URL("../../../crates/sceneworks-worker/src/engine_capability_facts.rs", import.meta.url),
);
const manifestPath = fileURLToPath(
  new URL("../../../config/manifests/builtin.preview-support.jsonc", import.meta.url),
);
const webPath = fileURLToPath(new URL("../src/data/previewSupport.json", import.meta.url));

/** Every checked-in stage-1 facts file, discovered rather than listed so a newly dumped backend
 *  (a third engine, a re-dumped macOS lane) is picked up with no edit here. */
export function readFactsFiles(dir = factsDir) {
  const names = readdirSync(dir)
    .filter((name) => /^capabilities\.[a-z0-9_-]+\.json$/.test(name))
    .sort();
  if (names.length === 0) {
    throw new Error(
      `no capabilities.<backend>.json under ${dir} — run the stage-1 dumper first ` +
        "(`cargo run -p sceneworks-worker --bin dump-engine-capabilities [--features backend-candle]`)",
    );
  }
  return names.map((name) => JSON.parse(readFileSync(`${dir}/${name}`, "utf8")));
}

/** The audio registry's dumps (sc-17593). Missing directory reads as "never dumped", which
 *  `assertBackendCoverage` then reports — the point being that it must not read as "nothing to do". */
export function readAudioFactsFiles(dir = audioFactsDir) {
  let names;
  try {
    names = readdirSync(dir)
      .filter((name) => /^capabilities\.[a-z0-9_-]+\.json$/.test(name))
      .sort();
  } catch {
    names = [];
  }
  return names.map((name) => JSON.parse(readFileSync(`${dir}/${name}`, "utf8")));
}

const factsFiles = readFactsFiles();
const audioFactsFiles = readAudioFactsFiles();
const factsDeclarationSource = readFileSync(factsDeclarationPath, "utf8");
// Refuse to write a catalog that is blind to a backend SceneWorks can run (sc-17119). Everything
// else here is scoped to what is on disk, so without this the artifacts regenerate perfectly
// happily with a whole engine's answers missing — reported downstream as "unknown", which renders
// exactly as it did before the feature shipped.
assertBackendCoverage(
  parseSceneworksBackends(factsDeclarationSource),
  factsFiles.map((facts) => facts.backend),
);
// The same assertion for the audio registry (sc-17593). Backend-level coverage above cannot stand in
// for it: `candle` is satisfied by the MEDIA dump, so the audio registry going undumped — as it did
// on every platform until this story — passes the check above every time.
assertBackendCoverage(
  parseSceneworksAudioBackends(factsDeclarationSource),
  audioFactsFiles.map((facts) => facts.backend),
  "audio",
);

const catalog = derivePreviewSupport(
  parseEngineModelTable(readFileSync(enginesPath, "utf8")),
  factsFiles,
  audioFactsFiles,
);

writeFileSync(manifestPath, `${JSON.stringify(catalog, null, 2)}\n`, "utf8");
writeFileSync(
  webPath,
  `${JSON.stringify(catalogToWebPreviewSupport(catalog), null, 2)}\n`,
  "utf8",
);

const ids = Object.keys(catalog.models);
const supporting = ids.filter((id) =>
  Object.values(catalog.models[id]).some((supported) => supported === true),
);
console.log(
  `builtin.preview-support.jsonc: ${ids.length} models over backends [${catalog.backends.join(", ")}]` +
    ` → ${manifestPath}`,
);
console.log(`previewSupport.json: mirror for the web fallback → ${webPath}`);
console.log(`live preview advertised by: ${supporting.join(", ") || "(none)"}`);
