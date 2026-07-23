import { access, constants, readFile } from "node:fs/promises";
import path from "node:path";
import process from "node:process";
import { promptGuideRequiredForModel } from "../apps/web/src/promptGuideContract.js";

const root = process.cwd();

const requiredPaths = [
  "apps/web/package.json",
  "apps/web/src/main.jsx",
  "apps/web/src/styles.css",
  "apps/rust-api/Cargo.toml",
  "apps/rust-api/src/main.rs",
  "apps/desktop/Cargo.toml",
  "apps/desktop/src/main.rs",
  "crates/sceneworks-core/Cargo.toml",
  "crates/sceneworks-core/src/lib.rs",
  "Cargo.toml",
  "rust-toolchain.toml",
  "packages/schemas/model-manifest.schema.json",
  "packages/schemas/lora-manifest.schema.json",
  "packages/schemas/recipe-preset.schema.json",
  "config/manifests/builtin.models.jsonc",
  "config/manifests/builtin.loras.jsonc",
  "config/manifests/builtin.recipe-presets.jsonc",
  "data/projects/.gitkeep",
  "data/models/.gitkeep",
  "data/loras/.gitkeep",
  "data/cache/.gitkeep",
  "docker-compose.yml",
  "docker/rust.Dockerfile",
  "docker/web.Dockerfile",
  ".cargo/config.toml",
];

const inferenceCargoWorkflows = [
  ".github/workflows/check.yml",
  ".github/workflows/desktop-windows.yml",
  ".github/workflows/macos-mlx.yml",
  ".github/workflows/release.yml",
  ".github/workflows/server-candle-linux.yml",
  ".github/workflows/windows-candle.yml",
];

const manifestSchemaPaths = [
  "packages/schemas/model-manifest.schema.json",
  "packages/schemas/lora-manifest.schema.json",
  "packages/schemas/recipe-preset.schema.json",
];

const manifestPaths = [
  "config/manifests/builtin.models.jsonc",
  "config/manifests/builtin.loras.jsonc",
  "config/manifests/builtin.recipe-presets.jsonc",
];

const manifestSchemaPairs = [
  {
    manifestPath: "config/manifests/builtin.models.jsonc",
    schemaPath: "packages/schemas/model-manifest.schema.json",
  },
  {
    manifestPath: "config/manifests/builtin.loras.jsonc",
    schemaPath: "packages/schemas/lora-manifest.schema.json",
  },
  {
    manifestPath: "config/manifests/builtin.recipe-presets.jsonc",
    schemaPath: "packages/schemas/recipe-preset.schema.json",
  },
];

async function assertReadable(relativePath) {
  const absolutePath = path.join(root, relativePath);
  await access(absolutePath, constants.R_OK);
}

async function assertContains(relativePath, expected) {
  const body = await readFile(path.join(root, relativePath), "utf8");
  if (!body.includes(expected)) {
    throw new Error(`${relativePath} does not contain ${expected}`);
  }
}

function stripJsoncComments(body) {
  let result = "";
  let inString = false;
  let escaped = false;
  for (let index = 0; index < body.length; index += 1) {
    const char = body[index];
    const next = body[index + 1];
    if (inString) {
      result += char;
      if (escaped) {
        escaped = false;
      } else if (char === "\\") {
        escaped = true;
      } else if (char === '"') {
        inString = false;
      }
      continue;
    }
    if (char === '"') {
      inString = true;
      result += char;
      continue;
    }
    if (char === "/" && next === "/") {
      while (index < body.length && body[index] !== "\n") {
        index += 1;
      }
      result += "\n";
      continue;
    }
    if (char === "/" && next === "*") {
      index += 2;
      while (index < body.length && !(body[index] === "*" && body[index + 1] === "/")) {
        index += 1;
      }
      index += 1;
      continue;
    }
    result += char;
  }
  return result;
}

async function readJsonc(relativePath) {
  return JSON.parse(stripJsoncComments(await readFile(path.join(root, relativePath), "utf8")));
}

async function assertManifestSchemasParse() {
  for (const schemaPath of manifestSchemaPaths) {
    const schema = JSON.parse(await readFile(path.join(root, schemaPath), "utf8"));
    if (schema.$schema !== "https://json-schema.org/draft/2020-12/schema") {
      throw new Error(`${schemaPath} must declare JSON Schema draft 2020-12`);
    }
    if (!schema.$id?.startsWith("https://sceneworks.local/schemas/")) {
      throw new Error(`${schemaPath} must declare a sceneworks.local schema id`);
    }
  }
}

async function assertManifestSchemaReferences() {
  for (const manifestPath of manifestPaths) {
    const manifest = await readJsonc(manifestPath);
    if (typeof manifest.$schema !== "string") {
      throw new Error(`${manifestPath} must declare a local $schema path`);
    }
    if (!manifest.$schema.startsWith("../../packages/schemas/")) {
      throw new Error(`${manifestPath} must reference packages/schemas, got ${manifest.$schema}`);
    }
    await assertReadable(path.normalize(path.join(path.dirname(manifestPath), manifest.$schema)));
  }
}

function assertJsonType(relativePath, key, value, expectedType) {
  if (expectedType === "integer") {
    if (!Number.isInteger(value)) {
      throw new Error(`${relativePath} ${key} must be an integer`);
    }
    return;
  }
  if (expectedType === "array") {
    if (!Array.isArray(value)) {
      throw new Error(`${relativePath} ${key} must be an array`);
    }
    return;
  }
  if (typeof value !== expectedType) {
    throw new Error(`${relativePath} ${key} must be a ${expectedType}`);
  }
}

async function assertManifestRootsMatchSchemas() {
  for (const { manifestPath, schemaPath } of manifestSchemaPairs) {
    const manifest = await readJsonc(manifestPath);
    const schema = JSON.parse(await readFile(path.join(root, schemaPath), "utf8"));
    for (const key of schema.required ?? []) {
      if (!(key in manifest)) {
        throw new Error(`${manifestPath} is missing required schema field ${key}`);
      }
      const expectedType = schema.properties?.[key]?.type;
      if (typeof expectedType === "string") {
        assertJsonType(manifestPath, key, manifest[key], expectedType);
      }
    }
  }
}

async function assertBuiltinPromptGuides() {
  const manifestPath = "config/manifests/builtin.models.jsonc";
  const manifest = await readJsonc(manifestPath);
  for (const model of manifest.models ?? []) {
    // Non-picker entries (type:"utility") never reach a Studio prompt-guide surface, so the schema
    // exempts them and this gate must too — otherwise a schema-valid utility entry REDs the lane
    // (sc-13783). The shared predicate is the single source of truth both authorities read; the
    // schema encodes the identical exemption and promptGuideScaffoldSchemaContract.test.js asserts
    // they can't diverge.
    if (!promptGuideRequiredForModel(model)) {
      continue;
    }
    const guide = model.ui?.promptGuide;
    if (!guide?.title || !guide?.path) {
      throw new Error(`${manifestPath} model ${model.id} is missing ui.promptGuide title/path`);
    }
    if (!Array.isArray(guide.sources) || guide.sources.length === 0) {
      throw new Error(`${manifestPath} model ${model.id} promptGuide needs source links`);
    }
    if (!guide.path.startsWith("/prompt-guides/") || !guide.path.endsWith(".md")) {
      throw new Error(`${manifestPath} model ${model.id} promptGuide path is invalid: ${guide.path}`);
    }
    await assertReadable(path.join("apps/web/public", guide.path.slice(1)));
  }
}

function parseCargoWorkspaceVersion(relativePath, body) {
  // The root Cargo.toml carries MANY `version = "…"` lines (workspace deps, the
  // [patch] rev, member crates inherit via `version.workspace = true`). Only the
  // one under [workspace.package] is the product version, so parse section-scoped
  // rather than grabbing the first `version =` in the file.
  const header = "[workspace.package]";
  const start = body.indexOf(header);
  if (start === -1) {
    throw new Error(`${relativePath} has no [workspace.package] table`);
  }
  const afterHeader = start + header.length;
  const rest = body.slice(afterHeader);
  const nextHeaderRel = rest.search(/\n\[/);
  const section = nextHeaderRel === -1 ? rest : rest.slice(0, nextHeaderRel);
  const match = section.match(/^version\s*=\s*"([^"]*)"/m);
  if (!match) {
    throw new Error(`${relativePath} [workspace.package] has no version key`);
  }
  return match[1];
}

async function readWorkspaceMemberCrateNames(cargoTomlBody) {
  // The [workspace] members array holds directory paths; each member's own
  // Cargo.toml declares the crate `name` that Cargo.lock records. Derive the set
  // dynamically (rather than hardcoding) so a new/renamed member is covered
  // automatically. Only the top-level `members = [...]` array — NOT
  // `default-members` (line starts with `default-`, so the ^members anchor skips
  // it) and NOT [workspace.dependencies].
  const membersMatch = cargoTomlBody.match(/^\s*members\s*=\s*\[([\s\S]*?)\]/m);
  if (!membersMatch) {
    throw new Error("Cargo.toml has no [workspace] members array");
  }
  const memberDirs = [...membersMatch[1].matchAll(/"([^"]+)"/g)].map((m) => m[1]);
  const names = [];
  for (const dir of memberDirs) {
    const memberToml = await readFile(path.join(root, dir, "Cargo.toml"), "utf8");
    const nameMatch = memberToml.match(/^\s*name\s*=\s*"([^"]+)"/m);
    if (!nameMatch) {
      throw new Error(`${dir}/Cargo.toml has no package name`);
    }
    names.push(nameMatch[1]);
  }
  return names;
}

function parseCargoLockVersions(body) {
  // Build a name -> version map from Cargo.lock's [[package]] blocks. Splitting on
  // the block delimiter keeps each package's own `name`/`version` together (so we
  // don't cross-read a dependency edge). Workspace members appear here with their
  // inherited version; a stale lock is exactly what makes `cargo fetch --locked`
  // fail in CI, so we assert these against the product version.
  const map = new Map();
  for (const block of body.split(/\n\[\[package\]\]\n/).slice(1)) {
    const name = block.match(/^name = "([^"]+)"/m)?.[1];
    const packageVersion = block.match(/^version = "([^"]+)"/m)?.[1];
    if (name && packageVersion) {
      map.set(name, packageVersion);
    }
  }
  return map;
}

async function assertVersionsAligned() {
  // sync-version.mjs's contract is that one root `npm version` bumps every
  // product-version field atomically (root + web + desktop package.json,
  // tauri.conf.json, and the root Cargo.toml [workspace.package] version), so
  // the git tag can never drift from the shipped version. Assert the invariant
  // here: if these diverge, a root `npm version patch` can emit a tag BELOW the
  // version users already run — the Tauri auto-updater (sc-1355) compares against
  // latest.json and would then serve no update. Cargo.toml is in the set because
  // its workspace version feeds logs, /health payloads, and the project-file
  // appVersion; it had silently drifted (0.2.0 vs a 0.8.0 product — sc-13613)
  // because sync-version.mjs did not touch it and this gate did not cover it.
  // Cargo.lock's workspace-member versions are asserted too: bumping the
  // workspace version without refreshing the lock leaves it stale, and CI's
  // `cargo fetch --locked` (parity/release/candle/windows) hard-fails on that
  // skew — so this gate catches lock drift before it ships, not in a red lane.
  const cargoTomlBody = await readFile(path.join(root, "Cargo.toml"), "utf8");
  const versionSources = [
    { path: "package.json", version: JSON.parse(await readFile(path.join(root, "package.json"), "utf8")).version },
    { path: "apps/web/package.json", version: JSON.parse(await readFile(path.join(root, "apps/web/package.json"), "utf8")).version },
    { path: "apps/desktop/package.json", version: JSON.parse(await readFile(path.join(root, "apps/desktop/package.json"), "utf8")).version },
    { path: "apps/desktop/tauri.conf.json", version: JSON.parse(await readFile(path.join(root, "apps/desktop/tauri.conf.json"), "utf8")).version },
    { path: "Cargo.toml", version: parseCargoWorkspaceVersion("Cargo.toml", cargoTomlBody) },
  ];
  const lockVersions = parseCargoLockVersions(await readFile(path.join(root, "Cargo.lock"), "utf8"));
  for (const crateName of await readWorkspaceMemberCrateNames(cargoTomlBody)) {
    versionSources.push({ path: `Cargo.lock:${crateName}`, version: lockVersions.get(crateName) });
  }
  // Scope of this assert: cross-file ALIGNMENT only. Using versionSources[0]
  // (root package.json) purely as the equality reference means we verify all
  // fields are equal, NOT that the version moved upward. A hypothetical
  // synchronized all-down edit would pass this check. That's deliberate:
  // upward-DIRECTION is owned by scripts/sync-version.mjs, whose only mutation
  // path is `npm version`, and `npm version` never moves a version down. There
  // is no committed floor to compare against here — these files ARE the
  // version sources — so a hardcoded baseline would be a rot-prone footgun
  // rather than a real guard. Alignment is the invariant this file owns.
  const reference = versionSources[0].version;
  const mismatched = versionSources.filter((source) => source.version !== reference);
  if (mismatched.length) {
    const detail = versionSources.map((source) => `${source.path}=${source.version}`).join(", ");
    throw new Error(
      `Product version fields are not aligned: ${detail}. Run one root ` +
        `\`npm version <x.y.z>\` (which invokes scripts/sync-version.mjs) so they all move together.`,
    );
  }
}

async function assertCharacterImageTuningSurface() {
  // Catch the sc-2017 picker UX mismatch at build time: a model that opts out
  // of the IP-Adapter Reference-strength slider via `ui.hideReferenceStrength`
  // must surface a `ui.variationStrength` slider in its place, otherwise the
  // "With character" picker leaves the user with no identity-tuning control.
  // The engine-wiring half of the sc-2018 guard lives in pytest because the
  // scaffold can't see worker MODEL_TARGETS without parsing Python.
  const manifestPath = "config/manifests/builtin.models.jsonc";
  const manifest = await readJsonc(manifestPath);
  const unbalanced = [];
  for (const model of manifest.models ?? []) {
    const ui = model.ui ?? {};
    if (ui.hideReferenceStrength && !ui.variationStrength) {
      unbalanced.push(model.id);
    }
  }
  if (unbalanced.length) {
    throw new Error(
      `${manifestPath} models hide the Reference-strength slider without declaring ` +
        `ui.variationStrength: ${unbalanced.join(", ")}. Add variationStrength or ` +
        `drop hideReferenceStrength.`,
    );
  }
}

for (const requiredPath of requiredPaths) {
  await assertReadable(requiredPath);
}

await assertContains("apps/web/src/App.jsx", "/api/v1/health");
await assertContains("Cargo.toml", "apps/rust-api");
await assertContains("Cargo.toml", "apps/desktop");
await assertContains("crates/sceneworks-core/src/lib.rs", "/api/v1/health");
await assertContains("docker-compose.yml", "NVIDIA_VISIBLE_DEVICES");
await assertContains("docker-compose.yml", "dockerfile: docker/rust.Dockerfile");
await assertContains("docker-compose.yml", "SCENEWORKS_RUST_WORKER_GPU_ID:-cpu");
await assertContains("docker-compose.yml", "/sceneworks/data/cache/jobs.db");
await assertContains("docker-compose.yml", "SCENEWORKS_ALLOW_OPEN_BIND");
await assertContains("docker-compose.yml", "environment: SCENEWORKS_INFERENCE_READ_TOKEN");
await assertContains(".env.example", "SCENEWORKS_RUST_WORKER_GPU_ID=cpu");
await assertContains(".env.example", "SCENEWORKS_INFERENCE_READ_TOKEN=");
await assertContains("docker/rust.Dockerfile", "sceneworks-rust-api");
await assertContains("docker/rust.Dockerfile", "type=secret,id=inference_token,required=true");
await assertContains("docker/rust.Dockerfile", "cargo build --offline");
await assertContains(".cargo/config.toml", "git-fetch-with-cli = true");
for (const workflowPath of inferenceCargoWorkflows) {
  await assertContains(workflowPath, "SCENEWORKS_INFERENCE_READ_TOKEN");
  await assertContains(workflowPath, 'CARGO_NET_OFFLINE: "true"');
  await assertContains(workflowPath, "cargo fetch --locked");
}
await assertContains("README.md", "SCENEWORKS_ACCESS_TOKEN");
await assertManifestSchemasParse();
await assertManifestSchemaReferences();
await assertManifestRootsMatchSchemas();
await assertVersionsAligned();
await assertBuiltinPromptGuides();
await assertCharacterImageTuningSurface();

console.log("SceneWorks scaffold check passed.");
