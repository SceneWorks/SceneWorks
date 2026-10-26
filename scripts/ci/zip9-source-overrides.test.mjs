import test from 'node:test';
import assert from 'node:assert/strict';
import path from 'node:path';
import os from 'node:os';
import * as fs from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { execFileSync } from 'node:child_process';
import { SECURITY_BACKPORT, SECURITY_BACKPORT_MANIFEST_SHA256, CURRENT_UPDATER, verifyVendorBytes, validateCargoSources, verifySecuritySources } from './zip9-source-overrides.mjs';
const root = fileURLToPath(new URL('../../', import.meta.url));
function graph() {
  const packages = SECURITY_BACKPORT.packages.map((p) => ({ name: p.name, version: p.version, source: null,
    id: p.name, manifest_path: path.join(root, p.path, 'Cargo.toml'),
    targets: [{ name: p.name, kind: ['lib'], crate_types: ['lib'], src_path: path.join(root, p.path, 'src/lib.rs') }] }));
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

function upstreamGraph(revision = 'a'.repeat(40), repository = root) {
  const value = graph();
  for (const pkg of value.packages) {
    if (pkg.name === 'tauri-plugin-updater') {
      const declared = CURRENT_UPDATER;
      pkg.manifest_path = path.join(repository, declared.path, 'Cargo.toml');
      pkg.targets[0].src_path = path.join(repository, declared.path, 'src/lib.rs');
    } else {
      pkg.source = `git+https://github.com/SceneWorks/inference?rev=${revision}#${revision}`;
      pkg.manifest_path = `/inference/${pkg.name}/Cargo.toml`;
    }
  }
  return value;
}

test('a final upstream producer replaces all three engine overrides while retaining the updater port', () => {
  const revision = 'a'.repeat(40), value = upstreamGraph(revision);
  assert.match(validateCargoSources(value, root, revision), /^[a-f0-9]{64}$/);
  for (const name of ['candle-core', 'candle-audio-kokoro', 'mlx-gen']) {
    const mixed = structuredClone(value);
    mixed.packages.find((pkg) => pkg.name === name).source = null;
    assert.throws(() => validateCargoSources(mixed, root, revision), /current pinned inference producer/);
  }
  const skew = structuredClone(value);
  skew.packages.at(-1).source = `git+https://github.com/SceneWorks/inference?rev=${'b'.repeat(40)}#${'b'.repeat(40)}`;
  assert.throws(() => validateCargoSources(skew, root, revision), /mixed or foreign inference source/);
  assert.throws(() => validateCargoSources(value), /audited local security port/,
    'upstream identity cannot masquerade as the historical measured backport');
});

async function upstreamFixture({ withoutBuildScript = false } = {}) {
  const directory = await vendorFixture();
  try {
    const checkout = path.join(directory, 'inference');
    await fs.mkdir(checkout);
    const git = (args) => execFileSync('git', ['-C', checkout, ...args], { encoding: 'utf8' }).trim();
    git(['init', '-q']);
    await fs.writeFile(path.join(checkout, '.gitattributes'), '*.cmd text eol=crlf\n');
    const value = upstreamGraph('a'.repeat(40), directory);
    for (const pkg of value.packages.filter((p) => p.source)) {
      const base = path.join(checkout, pkg.name);
      await fs.mkdir(path.join(base, 'src'), { recursive: true });
      await fs.writeFile(path.join(base, 'Cargo.toml'), withoutBuildScript ? '[package]\n' : '[package]\nbuild = "build.rs"\n');
      await fs.writeFile(path.join(base, 'src/lib.rs'), 'mod inner; #[cfg(feature="optional")] mod optional; // never built\n');
      await fs.writeFile(path.join(base, 'src/inner.rs'), '// declared module, never executed\n');
      if (!withoutBuildScript) {
        await fs.writeFile(path.join(base, 'build.rs'), 'mod build_support; #[cfg(feature="optional")] mod optional_build; // never executed\n');
        await fs.writeFile(path.join(base, 'build_support.rs'), '// declared build module, never executed\n');
      }
      await fs.writeFile(path.join(base, 'install.cmd'), 'rem inert committed CRLF fixture\r\n');
      pkg.manifest_path = path.join(base, 'Cargo.toml');
      pkg.targets = [
        { name: pkg.name, kind: ['lib'], src_path: path.join(base, 'src/lib.rs') },
      ];
      if (!withoutBuildScript) pkg.targets.push({ name: 'build-script-build', kind: ['custom-build'], src_path: path.join(base, 'build.rs') });
    }
    git(['add', '.']);
    git(['-c', 'user.name=Codex', '-c', 'user.email=codex@example.invalid', 'commit', '-qm', 'owned source fixture']);
    const revision = git(['rev-parse', 'HEAD']);
    for (const pkg of value.packages.filter((p) => p.source)) {
      pkg.source = `git+https://github.com/SceneWorks/inference?rev=${revision}#${revision}`;
    }
    await fs.writeFile(path.join(checkout, '.cargo-ok'), '');
    return { directory, checkout, git, value, revision };
  } catch (error) {
    await fs.rm(directory, { recursive: true, force: true });
    throw error;
  }
}

test('upstream receipt binds actual dependency revision, tree, targets and remaining local bytes', async () => {
  const { directory, checkout, git, value, revision } = await upstreamFixture();
  try {
    const receipt = await verifySecuritySources(value, directory, revision);
    assert.equal(receipt.kind, 'pinned-inference-with-local-updater');
    assert.equal(receipt.engine_revision, revision);
    assert.equal(receipt.inference_checkouts[0].tree, git(['rev-parse', 'HEAD^{tree}']));
    assert.match(receipt.inference_checkouts[0].tracked_files_sha256, /^[a-f0-9]{64}$/);
    assert.deepEqual(receipt.packages.map((pkg) => pkg.name), ['tauri-plugin-updater']);
    assert(receipt.inference_targets.every((target) => /^[a-f0-9]{64}$/.test(target.sha256)));
    assert(!Object.hasOwn(receipt, 'audited_unchanged_cuda_anchors'), 'new closure does not inherit old CUDA audit');
    const target = value.packages.find((pkg) => pkg.source).targets[0].src_path;
    await fs.writeFile(target, '// changed actual source\n');
    await assert.rejects(verifySecuritySources(value, directory, revision), /must be clean/);
    git(['checkout', '--', '.']);
    await fs.writeFile(path.join(checkout, 'build.rs'), '// untracked automatic target\n');
    await assert.rejects(verifySecuritySources(value, directory, revision), /must be clean/);
  } finally {
    await fs.rm(directory, { recursive: true, force: true });
  }
});

for (const [flag, relative] of [
  ['--assume-unchanged', 'src/inner.rs'],
  ['--skip-worktree', 'src/inner.rs'],
  ['--assume-unchanged', 'build_support.rs'],
  ['--skip-worktree', 'build.rs'],
  ['--assume-unchanged', 'install.cmd'],
]) {
  test(`upstream receipt refuses hidden ${relative} edits with ${flag} before handoff`, async () => {
    const { directory, checkout, git, value, revision } = await upstreamFixture();
    try {
      const before = await verifySecuritySources(value, directory, revision);
      const pkg = value.packages.find((item) => item.source);
      const file = path.join(path.dirname(pkg.manifest_path), relative);
      git(['update-index', flag, '--', path.relative(checkout, file)]);
      await fs.writeFile(file, relative.endsWith('.cmd')
        ? 'rem changed hidden source, never executed\r\n' : '// changed hidden source, never executed\n');
      assert.equal(git(['status', '--porcelain', '--untracked-files=no']), '', 'mutation evades the old status check');
      assert.equal(git(['rev-parse', 'HEAD^{tree}']), before.inference_checkouts[0].tree);
      assert.equal(validateCargoSources(value, directory, revision), before.cargo_source_graph_sha256);
      let handedOff = false;
      await assert.rejects(async () => {
        await verifySecuritySources(value, directory, revision);
        handedOff = true;
      }, /inference tracked bytes differ/);
      assert.equal(handedOff, false, 'changed committed modules must never reach execution');
    } finally {
      await fs.rm(directory, { recursive: true, force: true });
    }
  });
}

test('upstream receipt refuses executable mode overlays hidden from git status', { skip: process.platform === 'win32' }, async () => {
  const { directory, checkout, git, value, revision } = await upstreamFixture();
  try {
    const pkg = value.packages.find((item) => item.source);
    const file = path.join(path.dirname(pkg.manifest_path), 'src/inner.rs');
    git(['config', 'core.filemode', 'false']);
    await fs.chmod(file, 0o755);
    assert.equal(git(['status', '--porcelain', '--untracked-files=no']), '');
    await assert.rejects(verifySecuritySources(value, directory, revision), /inference source mode changed/);
  } finally {
    await fs.rm(directory, { recursive: true, force: true });
  }
});

test('upstream receipt takes checkout EOL policy from the named commit, never hidden attributes', async () => {
  const { directory, checkout, git, value, revision } = await upstreamFixture();
  try {
    await verifySecuritySources(value, directory, revision);
    git(['update-index', '--assume-unchanged', '.gitattributes']);
    await fs.writeFile(path.join(checkout, '.gitattributes'), '*.cmd text eol=lf\n');
    assert.equal(git(['status', '--porcelain', '--untracked-files=no']), '');
    await assert.rejects(verifySecuritySources(value, directory, revision), /inference tracked bytes differ/);
  } finally {
    await fs.rm(directory, { recursive: true, force: true });
  }
});

test('current updater source identity refuses the retired path and a zip downgrade', async () => {
  const { directory, value, revision } = await upstreamFixture();
  try {
    const wrong = structuredClone(value);
    wrong.packages.find((pkg) => pkg.name === 'tauri-plugin-updater').manifest_path =
      path.join(directory, 'vendor/zip9-tauri-plugin-updater/Cargo.toml');
    assert.throws(() => validateCargoSources(wrong, directory, revision));
    const file = path.join(directory, CURRENT_UPDATER.path, 'Cargo.toml');
    const current = await fs.readFile(file, 'utf8');
    const downgraded = current.replace('version = "9"', 'version = "4"');
    assert.notEqual(downgraded, current);
    await fs.writeFile(file, downgraded);
    await assert.rejects(verifySecuritySources(value, directory, revision), /Cargo.toml/);
  } finally {
    await fs.rm(directory, { recursive: true, force: true });
  }
});

for (const relative of ['src/bin/ignored.rs', 'build.rs', 'src/optional.rs', 'optional_build.rs']) {
  test(`upstream receipt refuses ignored ${relative} before source handoff`, async () => {
    const { directory, checkout, git, value, revision } = await upstreamFixture({ withoutBuildScript: relative === 'build.rs' });
    try {
      const before = await verifySecuritySources(value, directory, revision);
      const pkg = value.packages.find((item) => item.source);
      const file = path.join(path.dirname(pkg.manifest_path), relative);
      await fs.appendFile(path.join(checkout, '.git/info/exclude'), `\n${path.relative(checkout, file)}\n`);
      await fs.mkdir(path.dirname(file), { recursive: true });
      await fs.writeFile(file, '// ignored source, never executed\n');
      assert.equal(git(['status', '--porcelain', '--untracked-files=no']), '');
      assert.equal(git(['status', '--porcelain', '--untracked-files=all']).replace('?? .cargo-ok', '').trim(), '');
      assert.equal(validateCargoSources(value, directory, revision), before.cargo_source_graph_sha256);
      if (relative.includes('/bin/')) pkg.targets.push({ name: 'ignored', kind: ['bin'], src_path: file });
      if (relative === 'build.rs') pkg.targets.push({ name: 'build-script-build', kind: ['custom-build'], src_path: file });
      let handedOff = false;
      await assert.rejects(async () => {
        await verifySecuritySources(value, directory, revision);
        handedOff = true;
      }, /extra inference source input|Cargo target absent from named inference tree/);
      assert.equal(handedOff, false);
    } finally {
      await fs.rm(directory, { recursive: true, force: true });
    }
  });
}
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
test('Cargo target graph rejects unmanifested auto-build source before any handoff', () => {
  const value = graph(), core = value.packages.find((p) => p.name === 'candle-core');
  core.targets.push({ name: 'build-script-build', kind: ['custom-build'], crate_types: ['bin'], src_path: path.join(root, 'vendor/zip9-candle-core/build.rs') });
  assert.throws(() => validateCargoSources(value), /unaudited Cargo target/);
});
test('Cargo target graph binds legitimate discovered targets as well as package identities', () => {
  const value = graph(), before = validateCargoSources(value);
  const updater = value.packages.find((p) => p.name === 'tauri-plugin-updater');
  updater.targets.push({ name: 'build-script-build', kind: ['custom-build'], crate_types: ['bin'], src_path: path.join(root, 'vendor/zip9-tauri-plugin-updater/build.rs') });
  assert.notEqual(validateCargoSources(value), before, 'authorized custom-build targets must change the source graph digest');
  updater.targets.at(-1).src_path = path.join(root, 'outside-audited-source.rs');
  assert.throws(() => validateCargoSources(value), /unaudited Cargo target/);
});

async function vendorFixture() {
  const directory = await fs.mkdtemp(path.join(os.tmpdir(), 'sw-zip9-closure-'));
  await fs.mkdir(path.join(directory, 'vendor'));
  for (const pkg of SECURITY_BACKPORT.packages) await fs.cp(path.join(root, pkg.path), path.join(directory, pkg.path), { recursive: true });
  await fs.cp(path.join(root, CURRENT_UPDATER.path), path.join(directory, CURRENT_UPDATER.path), { recursive: true });
  for (const name of ['zip9-source-overrides.json', 'zip9-security-backport.patch']) await fs.copyFile(path.join(root, 'vendor', name), path.join(directory, 'vendor', name));
  return directory;
}
for (const extra of ['build.rs', 'src/bin/extra.rs', '.cargo/config.toml', 'target/generated.rs']) {
  test(`complete vendor closure rejects unmanifested ${extra} before execution`, async () => {
    const directory = await vendorFixture(); let handedOff = false;
    try {
      const file = path.join(directory, 'vendor/zip9-candle-core', extra);
      await fs.mkdir(path.dirname(file), { recursive: true });
      await fs.writeFile(file, '// inert fixture: never built or executed\n');
      await assert.rejects(async () => { await verifyVendorBytes(directory); handedOff = true; }, /unmanifested vendor/);
      assert.equal(handedOff, false, 'an extra autodiscovered build/input cannot reach execution');
    } finally { await fs.rm(directory, { recursive: true, force: true }); }
  });
}

test('vendor source byte mutation fails rather than hiding behind the declared engine pin', async () => {
  const directory = await vendorFixture();
  try {
    const file = SECURITY_BACKPORT.packages[0].files[0];
    await fs.writeFile(path.join(directory, file.path), 'mutated source, not an approved security leaf');
    await assert.rejects(verifyVendorBytes(directory), (error) => error.message.includes(file.path));
  } finally { await fs.rm(directory, { recursive: true, force: true }); }
});
