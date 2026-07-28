import assert from "node:assert/strict";
import test from "node:test";

import { canonicalSourceText } from "./generate-image-memory-matrix.mjs";

test("image-memory source hashing is independent of platform line endings", () => {
  const canonical = "alpha\nbeta\ngamma\n";
  assert.equal(canonicalSourceText(canonical), canonical);
  assert.equal(canonicalSourceText("alpha\r\nbeta\r\ngamma\r\n"), canonical);
  assert.equal(canonicalSourceText("alpha\rbeta\rgamma\r"), canonical);
});
