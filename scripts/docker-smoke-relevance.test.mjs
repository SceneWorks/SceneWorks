import assert from "node:assert/strict";
import path from "node:path";
import { fileURLToPath } from "node:url";
import test from "node:test";
import { readFileSync, statSync } from "node:fs";

import { dockerSmokeDecision } from "./docker-smoke-relevance.mjs";

test("root Cargo.toml is Docker relevant", () => {
  assert.equal(dockerSmokeDecision(1, ["Cargo.toml"]).run, true);
});

// CI_IMPACT: rare build-configuration/evidence-schema edits use the existing Docker lane;
// existing broad source selection is preserved. This test performs no image/model work.
test("web build configuration and projected evidence dependencies are Docker relevant", () => {
  for (const file of [
    "apps/web/vite.config.js",
    "apps/web/sdxl-cuda-memory-projection.js",
    "packages/schemas/memory-calibration.schema.json",
    "packages/schemas/memory-anchor-plan.schema.json",
  ]) {
    assert.equal(dockerSmokeDecision(1, [file]).run, true, file);
  }
});

test("every static module imported by the evidence packager is Docker relevant", () => {
  const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
  const seen = new Set();
  function visit(file) {
    if (seen.has(file)) return;
    seen.add(file);
    assert.equal(dockerSmokeDecision(1, [file]).run, true, file);
    const source = readFileSync(path.join(root, file), "utf8");
    for (const match of source.matchAll(/^import\s+[^;]*?\s+from\s+["'](\.[^"']+)["']/gm)) {
      visit(path.posix.normalize(path.posix.join(path.posix.dirname(file), match[1])));
    }
  }
  visit("scripts/package-sdxl-cuda-endpoints.mjs");
  assert.ok(seen.size > 1, "must traverse the actual packager dependency graph");
});

test("an exhaustively listed unrelated pull request skips the Docker smoke", () => {
  assert.equal(
    dockerSmokeDecision(2, ["README.md", "docs/contributing.md"]).run,
    false,
  );
});

test("a paginated result mismatch runs conservatively instead of skipping", () => {
  const firstThreeThousand = Array.from(
    { length: 3000 },
    (_, index) => `docs/change-${index}.md`,
  );
  const decision = dockerSmokeDecision(3001, firstThreeThousand);
  assert.equal(decision.run, true);
  assert.match(decision.reason, /pagination returned 3000/);
});

test("vendored dependency changes rebuild the Docker Cargo graph", () => {
  for (const file of [
    "vendor/glib-0.18.5/src/variant_iter.rs",
    "vendor/tauri-plugin-updater-2.11.0/src/updater.rs",
  ]) {
    assert.equal(dockerSmokeDecision(1, [file]).run, true, file);
  }
});


test("Docker source, toolchain and embedded-data inputs cannot be skipped", () => {
  for (const file of ["crates/sceneworks-core/src/lib.rs", "apps/rust-api/src/main.rs", "apps/rust-worker/src/main.rs", "apps/web/src/App.jsx", "apps/web/package-lock.json", "apps/desktop/Cargo.toml", "apps/desktop/build.rs", "rust-toolchain.toml", ".cargo/config.toml", ".dockerignore", "docs/calibration/sc-22738/anchor.json", "docs/generated/video-memory-curves.json"]) {
    assert.equal(dockerSmokeDecision(1, [file]).run, true, file);
  }
});


test("every existing local Docker COPY input reaches the smoke selector", () => {
  const inputs = new Set();
  for (const dockerfile of ["docker/rust.Dockerfile", "docker/web.Dockerfile"]) {
    for (const match of readFileSync(dockerfile, "utf8").matchAll(/^COPY (.+)$/gm)) {
      if (match[1].includes("--from=")) continue; // stage output, not a repository input
      const tokens = match[1].split(/\s+/).filter((token) => !token.startsWith("--"));
      assert.ok(tokens.length >= 2 && !tokens[0].startsWith("["), "review new COPY syntax before narrowing its inputs");
      for (const source of tokens.slice(0, -1)) {
        const directory = statSync(source, { throwIfNoEntry: false })?.isDirectory();
        inputs.add(directory ? `${source.replace(/\/$/, "")}/selector-probe` : source);
      }
    }
  }
  for (const file of ["docs/generated/krea-candle-five-rung-sc-11045.json", "docs/generated/qwen-candle-five-rung-sc-15817.json", "docs/film-script-writing.md", "docs/film-editor.md", "apps/desktop/licenses/selector-probe"]) {
    assert.ok(inputs.has(file), `audit must see the actual COPY input ${file}`);
  }
  for (const input of inputs) assert.equal(dockerSmokeDecision(1, [input]).run, true, input);
});
