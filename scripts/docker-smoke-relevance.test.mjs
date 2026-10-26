import assert from "node:assert/strict";
import test from "node:test";

import { dockerSmokeDecision } from "./docker-smoke-relevance.mjs";

test("root Cargo.toml is Docker relevant", () => {
  assert.equal(dockerSmokeDecision(1, ["Cargo.toml"]).run, true);
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
