import assert from "node:assert/strict";
import test from "node:test";

import { backendScopes, canonicalSourceText } from "./generate-memory-matrix.mjs";

test("memory-strategy source hashing is independent of platform line endings", () => {
  const canonical = "alpha\nbeta\ngamma\n";
  assert.equal(canonicalSourceText(canonical), canonical);
  assert.equal(canonicalSourceText("alpha\r\nbeta\r\ngamma\r\n"), canonical);
  assert.equal(canonicalSourceText("alpha\rbeta\rgamma\r"), canonical);
});

// SC-15510: `z_image_edit` is a catalog id, not a provider. Both backends serve it from the
// `z_image_turbo` provider (MLX: `jobs_store::routing::mlx` maps `z_image_turbo | z_image_edit` to the
// same eligibility and engine; Candle: the `ZImageEdit` lane runs on Turbo weights), so its advertised
// backend scopes must be INHERITED from `z_image_turbo` rather than read off its own manifest entry —
// which carries no `mlx`/`candle` block of its own.
//
// Without the inheritance the entry silently advertises zero backends, and every one of its 150 matrix
// cells disappears instead of failing loudly. That is exactly the "route unavailable" state the epic
// distinguishes from "verified", so it is worth a test rather than a comment.
test("z_image_edit inherits its backend scopes from the z_image_turbo provider", () => {
  const manifestById = new Map([
    ["z_image_turbo", { id: "z_image_turbo", mlx: { quantize: 4 }, candle: { quantize: 4 } }],
    ["z_image_edit", { id: "z_image_edit" }],
  ]);
  const edit = manifestById.get("z_image_edit");
  assert.deepEqual(backendScopes(edit, manifestById), ["mlx", "candle"]);

  // The inheritance is specific, not a blanket fallback: an ordinary entry with no backend blocks
  // advertises nothing, and the alias tracks whatever Turbo actually advertises.
  assert.deepEqual(backendScopes({ id: "some_other_model" }, manifestById), []);
  const mlxOnly = new Map([
    ["z_image_turbo", { id: "z_image_turbo", mlx: { quantize: 4 } }],
    ["z_image_edit", { id: "z_image_edit" }],
  ]);
  assert.deepEqual(backendScopes(mlxOnly.get("z_image_edit"), mlxOnly), ["mlx"]);
});
