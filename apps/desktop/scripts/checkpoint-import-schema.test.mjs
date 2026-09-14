import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import test from "node:test";

import Ajv2020 from "ajv/dist/2020.js";

const scriptsDir = dirname(fileURLToPath(import.meta.url));
const schema = JSON.parse(
  readFileSync(join(scriptsDir, "..", "..", "..", "packages", "schemas", "checkpoint-import.schema.json"), "utf8"),
);
const digest = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

function locator(kind, relativePath) {
  if (kind === "linked") {
    return {
      kind,
      schemaVersion: 1,
      rootId: "root",
      relativePath,
      fingerprint: digest,
    };
  }
  return {
    kind,
    schemaVersion: 1,
    installId: "install",
    relativePath,
    sha256: digest,
    provenance: { source: "huggingface" },
  };
}

test("checkpoint-import Draft 2020-12 paths reject line-separator-hidden separators", () => {
  const ajv = new Ajv2020({ allErrors: true, strict: true });
  const validate = ajv.compile(schema);

  for (const kind of ["linked", "managed"]) {
    assert.equal(validate(locator(kind, "models/model.safetensors")), true, kind);
    assert.equal(validate(locator(kind, "a\u2028b")), true, `${kind} U+2028 ordinary path`);
    assert.equal(validate(locator(kind, "a\u2029b")), true, `${kind} U+2029 ordinary path`);
    for (const [label, path] of [
      ["slash", "a\u2028//b"],
      ["colon", "a\u2028:b"],
      ["trailing separator", "a\u2028x/"],
      ["slash", "a\u2029//b"],
      ["colon", "a\u2029:b"],
      ["trailing separator", "a\u2029x/"],
    ]) {
      assert.equal(validate(locator(kind, path)), false, `${kind} ${label} ${JSON.stringify(path)}`);
    }
  }
});
