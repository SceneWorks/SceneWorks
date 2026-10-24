import test from 'node:test';
import assert from 'node:assert/strict';
import path from 'node:path';
import os from 'node:os';
import * as fs from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { SECURITY_BACKPORT, SECURITY_BACKPORT_MANIFEST_SHA256, verifyVendorBytes, validateCargoSources } from './zip9-source-overrides.mjs';
const root = fileURLToPath(new URL('../../', import.meta.url));
function graph() {
  const packages = SECURITY_BACKPORT.packages.map((p) => ({ name: p.name, version: p.version, source: null,
    id: p.name, manifest_path: path.join(root, p.path, 'Cargo.toml') }));
  for (const name of new Set(SECURITY_BACKPORT.audited_unchanged_cuda_anchors.map((a) => a.package))) {
    packages.push({ name, version: '0.0.0', id: name, manifest_path: `/fixture/${name}/Cargo.toml`,
      source: `git+https://github.com/SceneWorks/inference?rev=${SECURITY_BACKPORT.engine_revision}#${SECURITY_BACKPORT.engine_revision}` });
  }
  return { packages, workspace_members: ['app'], workspace_default_members: ['app'] };
}
test('all exact security leaf bytes and small port diff match declared source identities', async () => {
  const proof = await verifyVendorBytes();
  assert.equal(proof.manifest_sha256, SECURITY_BACKPORT_MANIFEST_SHA256);
  assert.equal(proof.packages.length, SECURITY_BACKPORT.packages.length);
  assert(proof.packages.every((p) => /^[a-f0-9]{64}$/.test(p.source_sha256)));
});
test('source graph retains exact security overrides without widening automatic workspace tests', () => {
  assert.match(validateCargoSources(graph()), /^[a-f0-9]{64}$/);
});
for (const [label, mutate] of [
  ['old registry/upstream implementation', (g) => { g.packages[0].source = 'registry+https://github.com/rust-lang/crates.io-index'; }],
  ['foreign path shadowing same package name', (g) => { g.packages[0].manifest_path = '/foreign/Cargo.toml'; }],
  ['missing overridden leaf', (g) => { g.packages.shift(); }],
  ['duplicate overridden leaf', (g) => { g.packages.push({ ...g.packages[0] }); }],
  ['changed package version', (g) => { g.packages[0].version = '99.0.0'; }],
  ['automatic workspace enrollment', (g) => { g.workspace_members.push(g.packages[0].id); }],
  ['default-member enrollment', (g) => { g.workspace_default_members.push(g.packages[0].id); }],
  ['changed CUDA producer source', (g) => { g.packages.at(-1).source = 'git+https://github.com/SceneWorks/inference#wrong'; }],
]) {
  test(`refuse ${label}`, () => { const value = graph(); mutate(value); assert.throws(() => validateCargoSources(value)); });
}
test('vendor source byte mutation fails rather than hiding behind the declared engine pin', async () => {
  const directory = await fs.mkdtemp(path.join(os.tmpdir(), 'sw-zip9-source-'));
  try {
    await fs.mkdir(path.join(directory, 'vendor'));
    await fs.copyFile(path.join(root, 'vendor/zip9-source-overrides.json'), path.join(directory, 'vendor/zip9-source-overrides.json'));
    const file = SECURITY_BACKPORT.packages[0].files[0];
    await fs.mkdir(path.dirname(path.join(directory, file.path)), { recursive: true });
    await fs.writeFile(path.join(directory, file.path), 'mutated source, not an approved security leaf');
    await assert.rejects(verifyVendorBytes(directory));
  } finally { await fs.rm(directory, { recursive: true, force: true }); }
});
