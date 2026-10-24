// Exact leaf security overrides are part of the effective producer, not hidden by a Git pin.
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import * as fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
const root = fileURLToPath(new URL('../../', import.meta.url));
const manifestFile = new URL('../../vendor/zip9-source-overrides.json', import.meta.url);
const sha = (data) => createHash('sha256').update(data).digest('hex');
export const SECURITY_BACKPORT_MANIFEST_SHA256 = sha(readFileSync(manifestFile));
export const SECURITY_BACKPORT = JSON.parse(readFileSync(manifestFile, 'utf8'));
const names = ['candle-audio-kokoro', 'candle-core', 'mlx-gen', 'tauri-plugin-updater'];

async function verifyDirectoryClosure(repository, pkg) {
  const declared = new Set(), directories = new Set([pkg.path]), seen = new Set();
  for (const file of pkg.files) {
    assert(file.path.startsWith(`${pkg.path}/`) && path.posix.normalize(file.path) === file.path);
    assert(!declared.has(file.path), `duplicate vendor input ${file.path}`);
    declared.add(file.path);
    for (let parent = path.posix.dirname(file.path); parent !== pkg.path; parent = path.posix.dirname(parent)) {
      assert(parent.startsWith(`${pkg.path}/`), 'vendor input escaped its audited root');
      directories.add(parent);
    }
  }
  async function walk(relative) {
    const stat = await fs.lstat(path.join(repository, relative));
    assert(stat.isDirectory() && !stat.isSymbolicLink(), `unsafe vendor directory ${relative}`);
    for (const entry of await fs.readdir(path.join(repository, relative), { withFileTypes: true })) {
      const name = `${relative}/${entry.name}`;
      assert(!entry.isSymbolicLink(), `unsafe vendor link ${name}`);
      if (entry.isDirectory()) {
        assert(directories.has(name), `unmanifested vendor directory ${name}`);
        await walk(name);
      } else {
        assert(declared.has(name), `unmanifested vendor input ${name}`);
        assert(entry.isFile(), `unsafe vendor input ${name}`);
        seen.add(name);
      }
    }
  }
  await walk(pkg.path);
  assert.deepEqual([...seen].sort(), [...declared].sort(), `incomplete vendor closure ${pkg.path}`);
}

export async function verifyVendorBytes(repository = root) {
  assert.equal(sha(await fs.readFile(path.join(repository, 'vendor/zip9-source-overrides.json'))), SECURITY_BACKPORT_MANIFEST_SHA256);
  assert.deepEqual(SECURITY_BACKPORT.packages.map((p) => p.name).sort(), [...names].sort());
  const packages = [];
  for (const pkg of SECURITY_BACKPORT.packages) {
    // Cargo may autodiscover untracked build.rs/bin targets; known hashes alone
    // are insufficient. Generated outputs belong outside these audited roots.
    await verifyDirectoryClosure(repository, pkg);
    const contents = [];
    for (const file of pkg.files) {
      assert(file.path.startsWith(`${pkg.path}/`) && !file.path.split('/').includes('..'));
      const full = path.join(repository, file.path), stat = await fs.lstat(full);
      assert(stat.isFile() && !stat.isSymbolicLink() && stat.nlink === 1, `unsafe vendor file ${file.path}`);
      const bytes = await fs.readFile(full);
      assert.equal(bytes.length, file.bytes, file.path); assert.equal(sha(bytes), file.sha256, file.path);
      contents.push([file.path, file.sha256]);
    }
    packages.push({ name: pkg.name, version: pkg.version, path: pkg.path, source_sha256: sha(JSON.stringify(contents)) });
  }
  assert.equal(sha(await fs.readFile(path.join(repository, 'vendor/zip9-security-backport.patch'))), SECURITY_BACKPORT.security_diff_sha256);
  return { kind: 'exact-pinned-leaf-security-backport', engine_revision: SECURITY_BACKPORT.engine_revision,
    manifest_sha256: SECURITY_BACKPORT_MANIFEST_SHA256, security_diff_sha256: SECURITY_BACKPORT.security_diff_sha256, packages };
}

export function validateCargoSources(metadata, repository = root) {
  assert(Array.isArray(metadata.packages));
  for (const expected of SECURITY_BACKPORT.packages) {
    const matches = metadata.packages.filter((p) => p.name === expected.name);
    assert.equal(matches.length, 1, `one effective ${expected.name} required`);
    const pkg = matches[0];
    assert.equal(pkg.version, expected.version);
    assert.equal(pkg.source, null, `${expected.name} must use the audited local security port`);
    assert.equal(path.resolve(pkg.manifest_path), path.resolve(repository, expected.path, 'Cargo.toml'));
    assert(!metadata.workspace_members.includes(pkg.id), 'security leaves must remain dependency-only');
    assert(!metadata.workspace_default_members.includes(pkg.id), 'security leaves must not widen default tests');
    assert(Array.isArray(pkg.targets) && pkg.targets.length > 0, `Cargo targets required for ${expected.name}`);
    const declared = new Set(expected.files.map((file) => path.resolve(repository, file.path)));
    for (const target of pkg.targets) {
      assert(path.isAbsolute(target.src_path) && declared.has(path.resolve(target.src_path)),
        `unaudited Cargo target ${expected.name}: ${target.src_path}`);
    }
  }
  const source = `git+https://github.com/SceneWorks/inference?rev=${SECURITY_BACKPORT.engine_revision}#${SECURITY_BACKPORT.engine_revision}`;
  for (const anchor of SECURITY_BACKPORT.audited_unchanged_cuda_anchors) {
    const matches = metadata.packages.filter((p) => p.name === anchor.package);
    assert.equal(matches.length, 1, `unambiguous CUDA anchor package ${anchor.package}`);
    assert.equal(matches[0].source, source, `CUDA anchor source changed: ${anchor.package}`);
  }
  const graph = metadata.packages.map((pkg) => ({ name: pkg.name, version: pkg.version,
    source: pkg.source ?? `path:${path.relative(repository, path.dirname(pkg.manifest_path)).split(path.sep).join('/')}`,
    targets: (pkg.targets ?? []).map((target) => ({ name: target.name, kind: target.kind,
      crate_types: target.crate_types, edition: target.edition, doc: target.doc, doctest: target.doctest, test: target.test,
      required_features: target['required-features'] ?? [],
      source: path.relative(path.dirname(pkg.manifest_path), target.src_path).split(path.sep).join('/'),
    })).sort((a, b) => JSON.stringify(a).localeCompare(JSON.stringify(b))),
  })).sort((a, b) => JSON.stringify(a).localeCompare(JSON.stringify(b)));
  return sha(JSON.stringify(graph));
}

export async function verifySecuritySources(metadata, repository = root) {
  const result = await verifyVendorBytes(repository);
  result.cargo_source_graph_sha256 = validateCargoSources(metadata, repository);
  result.audited_unchanged_cuda_anchors = [];
  for (const anchor of SECURITY_BACKPORT.audited_unchanged_cuda_anchors) {
    const pkg = metadata.packages.find((p) => p.name === anchor.package);
    const file = path.join(path.dirname(pkg.manifest_path), anchor.relative_path);
    const stat = await fs.lstat(file); assert(stat.isFile() && !stat.isSymbolicLink());
    assert.equal(sha(await fs.readFile(file)), anchor.sha256, `CUDA producer anchor changed: ${anchor.upstream_path}`);
    result.audited_unchanged_cuda_anchors.push({ ...anchor });
  }
  return result;
}
