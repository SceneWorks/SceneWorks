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

test("checkpoint-import schema2 requires a complete component commitment under strict AJV", () => {
  const ajv = new Ajv2020({ allErrors: true, strict: true, strictRequired: true });
  const validate = ajv.compile(schema);
  const download = { repo: "fixture/components", revision: "1".repeat(40), files: ["tokenizer/tokenizer.json"] };
  const declaration = {
    schemaVersion: 2, baseLineage: {}, declaredSemantics: {}, inheritedLicenseAcknowledgments: [],
    componentDownloads: [download],
    componentClosure: [{ download, revision: download.revision, files: [{
      relativePath: "tokenizer/tokenizer.json", sha256: `sha256:${digest}`, sizeBytes: 2,
    }] }],
  };
  const plan = (modelDeclaration) => ({
    schemaVersion: 1, planId: "fixture-plan", family: "minimax-h3",
    layers: [{ layerId: "dit", role: "transformer", targetPath: "model.safetensors",
      container: "safetensors", source: locator("managed", "model.safetensors") }],
    context: { modelDeclaration },
  });
  assert.equal(validate(plan(declaration)), true, JSON.stringify(validate.errors));
  for (const field of ["componentDownloads", "componentClosure"]) {
    const missing = structuredClone(declaration);
    delete missing[field];
    assert.equal(validate(plan(missing)), false, `schema2 requires ${field}`);
    const empty = structuredClone(declaration);
    empty[field] = [];
    assert.equal(validate(plan(empty)), false, `schema2 refuses empty ${field}`);
  }
  for (const mutate of [
    (value) => { value.componentDownloads[0].revision = "main"; },
    (value) => { value.componentClosure[0].files[0].sha256 = "unbound"; },
    (value) => { value.componentClosure[0].files[0].sizeBytes = -1; },
  ]) {
    const invalid = structuredClone(declaration);
    mutate(invalid);
    assert.equal(validate(plan(invalid)), false, "component revision/content constraints remain enforced");
  }
  const legacy = structuredClone(declaration);
  legacy.schemaVersion = 1;
  delete legacy.componentDownloads;
  delete legacy.componentClosure;
  assert.equal(validate(plan(legacy)), true, "legacy schema1 remains readable without inventing component authority");
});
