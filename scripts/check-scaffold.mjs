import { access, constants, readFile } from "node:fs/promises";
import path from "node:path";
import process from "node:process";

const root = process.cwd();

const requiredPaths = [
  "apps/web/package.json",
  "apps/web/src/main.jsx",
  "apps/web/src/styles.css",
  "apps/rust-api/Cargo.toml",
  "apps/rust-api/src/main.rs",
  "apps/desktop/Cargo.toml",
  "apps/desktop/src/main.rs",
  "apps/worker/scene_worker/runtime.py",
  "crates/sceneworks-core/Cargo.toml",
  "crates/sceneworks-core/src/lib.rs",
  "Cargo.toml",
  "rust-toolchain.toml",
  "packages/schemas/project.schema.json",
  "config/manifests/builtin.models.jsonc",
  "config/manifests/builtin.loras.jsonc",
  "data/projects/.gitkeep",
  "data/models/.gitkeep",
  "data/loras/.gitkeep",
  "data/cache/.gitkeep",
  "docker-compose.yml",
  "docker/rust-api.Dockerfile",
  "docker/web.Dockerfile",
  "docker/worker.Dockerfile",
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

async function assertBuiltinPromptGuides() {
  const manifestPath = "config/manifests/builtin.models.jsonc";
  const manifest = JSON.parse(stripJsoncComments(await readFile(path.join(root, manifestPath), "utf8")));
  for (const model of manifest.models ?? []) {
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

for (const requiredPath of requiredPaths) {
  await assertReadable(requiredPath);
}

await assertContains("apps/web/src/App.jsx", "/api/v1/health");
await assertContains("Cargo.toml", "apps/rust-api");
await assertContains("Cargo.toml", "apps/desktop");
await assertContains("crates/sceneworks-core/src/lib.rs", "/api/v1/health");
await assertContains("docker-compose.yml", "NVIDIA_VISIBLE_DEVICES");
await assertContains("docker-compose.yml", "dockerfile: docker/rust-api.Dockerfile");
await assertContains("docker-compose.yml", "SCENEWORKS_RUST_WORKER_GPU_ID:-cpu");
await assertContains("docker-compose.yml", "/sceneworks/data/cache/jobs.db");
await assertContains(".env.example", "SCENEWORKS_RUST_WORKER_GPU_ID=cpu");
await assertContains("docker/rust-api.Dockerfile", "sceneworks-rust-api");
await assertContains("README.md", "SCENEWORKS_ACCESS_TOKEN");
await assertBuiltinPromptGuides();

console.log("SceneWorks scaffold check passed.");
