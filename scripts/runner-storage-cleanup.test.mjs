import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, mkdir, writeFile, readFile, rm } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, resolve, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import { disposableTargets, mossOwner, retiredOwners, sweepMoss, runAttempt } from '../.github/actions/cleanup-runner-storage/index.mjs';

const execute = promisify(execFile);
const here = resolve(dirname(fileURLToPath(import.meta.url)), '../.github/actions/cleanup-runner-storage');
const day = 86400000;
const now = Date.parse('2026-10-27T12:00:00Z');
const runtime = root => ({ root, hfHome: join(root, 'hf'), backend: 'cuda', sourceSha: 'a'.repeat(40), inferenceSha: 'b'.repeat(40) });

test('Windows workflows register post cleanup before cache uploads are registered', async () => {
  const repo = resolve(here, '../../..');
  const action = await readFile(join(here, 'action.yml'), 'utf8');
  assert.match(action, /post: index\.mjs\s+post-if: always\(\)/);
  if (existsSync(join(repo, '.github/workflows/windows-candle.yml'))) {
    const workflow = await readFile(join(repo, '.github/workflows/windows-candle.yml'), 'utf8');
    assert.ok(workflow.indexOf('uses: ./.github/actions/cleanup-runner-storage') >= 0);
    assert.ok(workflow.indexOf('uses: ./.github/actions/cleanup-runner-storage') < workflow.indexOf('uses: ./.github/actions/prepare-rust-runner'));
    const moss = await readFile(join(repo, '.github/workflows/moss-edit-terminal.yml'), 'utf8');
    assert.match(moss, /ref: \$\{\{ github.workflow_sha \}\}\s+path: runner-storage-tooling\s+sparse-checkout: \.github\/actions\/cleanup-runner-storage/);
    assert.ok(moss.indexOf('uses: ./runner-storage-tooling/.github/actions/cleanup-runner-storage') < moss.indexOf('node source\\scripts\\moss-edit-terminal-actions.mjs smoke'));
    assert.match(moss, /cleanup-runner-storage\s+with:\s+moss: 'true'/);
  } else {
    const workflow = await readFile(join(repo, '.github/workflows/ci.yml'), 'utf8');
    for (const name of ['windows-cuda-check', 'windows-cuda']) {
      const block = workflow.split(`\n  ${name}:`)[1].split(/\n  [\w-]+:/)[0];
      const register = block.indexOf('uses: ./.github/actions/cleanup-runner-storage');
      assert.ok(register >= 0 && register < block.indexOf('uses: Swatinem/rust-cache@'), name);
    }
  }
});

async function fixture(t) {
  const path = await mkdtemp(join(tmpdir(), 'runner-storage-test-'));
  t.after(async () => {
    const full = resolve(path);
    assert.equal(dirname(full), resolve(tmpdir()));
    assert.ok(full.startsWith(join(resolve(tmpdir()), 'runner-storage-test-')));
    await rm(full, { recursive: true, force: true });
  });
  return path;
}

test('only checkout targets and the exact current MOSS attempt are disposable', () => {
  const workspace = resolve('runner/repo/repo');
  const current = resolve(workspace, '../moss-sc-26752/moss-execute-123-2-cuda');
  const env = { GITHUB_RUN_ID: '123', GITHUB_RUN_ATTEMPT: '2', MOSS_ROOT: current, CARGO_TARGET_DIR: resolve('shared/target') };
  assert.deepEqual(disposableTargets(workspace, env), [resolve(workspace, 'target'), resolve(workspace, 'source/target'), resolve(workspace, 'inference/target'), resolve(current, 'target')]);
  for (const bad of [resolve('shared/moss-execute-123-2-cuda'), resolve(workspace, '../moss-sc-26752/moss-execute-123-1-cuda')]) {
    assert.equal(disposableTargets(workspace, { ...env, MOSS_ROOT: bad }).length, 3);
  }
});

test('MOSS ownership binds the namespace, backend, source pair and private HF leaf', () => {
  const root = resolve('moss-execute-123-2-cuda');
  assert.deepEqual(mossOwner(root, runtime(root)), { run: '123', attempt: '2', root });
  for (const patch of [{ root: resolve('another') }, { hfHome: resolve('shared') }, { backend: 'mlx' }, { sourceSha: 'unknown' }]) assert.equal(mossOwner(root, { ...runtime(root), ...patch }), null);
  assert.equal(mossOwner(resolve('moss-bundle-123-2'), runtime(resolve('moss-bundle-123-2'))), null);
});

test('retain current, unknown and active owners, the newest two completed owners, and a full day of recovery', () => {
  const rows = [1, 2, 3, 4, 5].map(i => ({ run: String(i), status: 'completed', updated_at: new Date(now - i * day).toISOString() }));
  rows.push({ run: '6', status: 'in_progress', updated_at: new Date(now - 20 * day).toISOString() });
  rows.push({ run: '7', status: 'queued', updated_at: new Date(now - 20 * day).toISOString() });
  rows.push({ run: '8', status: 'completed', updated_at: 'unknown' });
  assert.deepEqual(retiredOwners(rows, now, '5').map(o => o.run), ['3', '4']);
  assert.deepEqual(retiredOwners(rows.slice(0, 3).map(o => ({ ...o, updated_at: new Date(now - 3600000).toISOString() })), now, '99'), []);
});

test('a completed attempt stays protected while a later attempt of its run is active', async t => {
  const run = { id: 123, run_attempt: 1, repository: { full_name: 'SceneWorks/SceneWorks' }, status: 'completed', updated_at: new Date(now - 5 * day).toISOString() };
  const replies = [run, { ...run, run_attempt: 2, status: 'in_progress' }];
  const urls = [];
  t.mock.method(globalThis, 'fetch', async url => { urls.push(url); return { ok: true, json: async () => replies.shift() }; });
  const value = await runAttempt({ run: '123', attempt: '1' }, { INPUT_TOKEN: 'test-token', GITHUB_REPOSITORY: 'SceneWorks/SceneWorks' });
  assert.equal(value.status, 'in_progress');
  assert.match(urls[0], /\/attempts\/1$/);
  assert.match(urls[1], /\/runs\/123$/);
});

async function mossFixture(t) {
  const root = await fixture(t), workspace = join(root, 'repo/repo');
  await mkdir(workspace, { recursive: true });
  const parent = resolve(workspace, '../moss-sc-26752');
  const owners = [];
  for (let i = 1; i <= 4; i++) {
    const path = join(parent, `moss-execute-${i}-1-cuda`);
    await mkdir(join(path, 'hf'), { recursive: true });
    await writeFile(join(path, 'runtime.json'), JSON.stringify(runtime(path)));
    owners.push(path);
  }
  const removed = [], calls = [];
  const storage = async (mode, path) => {
    if (['moss', 'cargo'].includes(mode)) removed.push(path);
    return { status: ['inspect-parent', 'moss-inputs'].includes(mode) ? 'verified' : 'removed', inputs: [] };
  };
  const attempt = async owner => { calls.push(owner.run); return { status: 'completed', updated_at: new Date(now - Number(owner.run) * day).toISOString() }; };
  return { workspace, parent, owners, removed, calls, storage, attempt, now, env: { GITHUB_RUN_ID: '99' } };
}

test('MOSS sweep deletes only expired owned leaves and rechecks completion', async t => {
  const f = await mossFixture(t);
  await sweepMoss(f.workspace, f.env, f);
  assert.deepEqual(f.removed, [join(f.owners[2], 'hf'), join(f.owners[3], 'hf')]);
  assert.equal(f.calls.filter(run => run === '3').length, 2);
  assert.equal(f.calls.filter(run => run === '1').length, 1);
});

test('MOSS sweep preserves explicit and bundle-referenced cache inputs', async t => {
  const f = await mossFixture(t), bundle = join(f.parent, 'moss-bundle-88-1');
  await mkdir(bundle);
  await writeFile(join(bundle, 'campaign.json'), JSON.stringify({ backends: { cuda: { hfHome: join(f.owners[3], 'hf') } } }));
  await sweepMoss(f.workspace, { ...f.env, MOSS_HF_HOME: join(f.owners[2], 'hf') }, f);
  assert.deepEqual(f.removed, []);
});

test('MOSS sweep preserves cache inputs referenced by a different listener', async t => {
  const f = await mossFixture(t);
  const storage = async (mode, path) => mode === 'moss-inputs' ? { status: 'verified', inputs: [join(f.owners[2], 'hf')] } : f.storage(mode, path);
  await sweepMoss(f.workspace, f.env, { ...f, storage });
  assert.deepEqual(f.removed, [join(f.owners[3], 'hf')]);
});

test('MOSS sweep skips if cross-listener inputs cannot be read', async t => {
  const f = await mossFixture(t);
  const storage = async (mode, path) => mode === 'moss-inputs' ? { status: 'skipped', reason: 'unreadable listener' } : f.storage(mode, path);
  await sweepMoss(f.workspace, f.env, { ...f, storage });
  assert.deepEqual(f.removed, []);
});

test('an API or ownership failure prevents the historical sweep', async t => {
  const f = await mossFixture(t);
  await sweepMoss(f.workspace, f.env, { ...f, attempt: async () => { throw Error('HTTP 403'); } });
  assert.deepEqual(f.removed, []);
  await writeFile(join(f.owners[0], 'runtime.json'), '{invalid');
  await sweepMoss(f.workspace, f.env, f);
  assert.deepEqual(f.removed, []);
});

test('a rerun which becomes active during the sweep retains its old cache', async t => {
  const f = await mossFixture(t), seen = new Set();
  const attempt = async o => {
    const result = await f.attempt(o);
    if (o.run === '3' && seen.has(o.run)) result.status = 'in_progress';
    seen.add(o.run); return result;
  };
  await sweepMoss(f.workspace, f.env, { ...f, attempt });
  assert.deepEqual(f.removed, [join(f.owners[3], 'hf')]);
});

test('a run which finishes again during the sweep gets a new recovery window', async t => {
  const f = await mossFixture(t), seen = new Set();
  const attempt = async o => {
    const result = await f.attempt(o);
    if (o.run === '3' && seen.has(o.run)) result.updated_at = new Date(now - 1000).toISOString();
    seen.add(o.run); return result;
  };
  await sweepMoss(f.workspace, f.env, { ...f, attempt });
  assert.deepEqual(f.removed, [join(f.owners[3], 'hf')]);
});

async function powershell(script, data) {
  const payload = Buffer.from(JSON.stringify(data)).toString('base64');
  const source = Buffer.from(join(here, 'storage.ps1')).toString('base64');
  const command = `$ErrorActionPreference='Stop'; . ([Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('${source}'))) -Library; $fixture=[Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('${payload}'))|ConvertFrom-Json; ${script}`;
  const { stdout } = await execute('powershell.exe', ['-NoProfile', '-NonInteractive', '-EncodedCommand', Buffer.from(command, 'utf16le').toString('base64')], { windowsHide: true });
  return stdout.trim();
}

test('Windows deletion removes only a validated Cargo target and respects active Cargo locks', { skip: process.platform !== 'win32' }, async t => {
  const root = await fixture(t), target = join(root, 'target');
  await mkdir(join(target, 'debug'), { recursive: true });
  await writeFile(join(target, '.rustc_info.json'), '{}');
  await writeFile(join(target, 'debug/.cargo-lock'), '');
  await writeFile(join(root, 'source.rs'), 'source');
  const blocked = await powershell(`$h=[IO.File]::Open(($fixture.target+'\\debug\\.cargo-lock'),[IO.FileMode]::Open,[IO.FileAccess]::ReadWrite,[IO.FileShare]::ReadWrite);try { try { Remove-StorageLeaf $fixture.target $fixture.root cargo $false @() 1|Out-Null;throw 'unexpected deletion' } catch { if(-not (Test-Path -LiteralPath $fixture.target)){throw};'blocked' } }finally{$h.Dispose()}`, { root, target });
  assert.equal(blocked, 'blocked');
  const output = JSON.parse(await powershell('Remove-StorageLeaf $fixture.target $fixture.root cargo $false @() 1 | ConvertTo-Json -Compress', { root, target }));
  assert.equal(output.status, 'removed');
  assert.equal(await readFile(join(root, 'source.rs'), 'utf8'), 'source');
});

test('Windows guards reject escaped paths, links, recent model writes and other active jobs', { skip: process.platform !== 'win32' }, async t => {
  const root = await fixture(t), target = join(root, 'target');
  await mkdir(target); await writeFile(join(target, '.rustc_info.json'), '{}');
  const data = { root, target };
  assert.equal(await powershell(`try{Remove-StorageLeaf $fixture.target ($fixture.root+'\\other') cargo $true @() 1|Out-Null;throw 'allowed'}catch{if($_.Exception.Message -eq 'allowed'){throw};'blocked'}`, data), 'blocked');
  assert.equal(await powershell(`$link=$fixture.root+'\\linked';New-Item -ItemType Junction -Path $link -Target $fixture.target|Out-Null;try{try{Get-StorageTree $link|Out-Null;throw 'allowed'}catch{if($_.Exception.Message -eq 'allowed'){throw};'blocked'}}finally{[IO.Directory]::Delete($link)}`, data), 'blocked');
  assert.equal(await powershell(`$link=$fixture.root+'\\linked';New-Item -ItemType Junction -Path $link -Target $fixture.target|Out-Null;try{try{Assert-OrdinaryAncestors ($link+'\\.rustc_info.json');throw 'allowed'}catch{if($_.Exception.Message -eq 'allowed'){throw};'blocked'}}finally{[IO.Directory]::Delete($link)}`, data), 'blocked');
  assert.equal(await powershell(`$hf=$fixture.root+'\\hf';New-Item -ItemType Directory -Path $hf|Out-Null;Set-Content -LiteralPath ($hf+'\\weights') -Value 'fixture';try{Remove-StorageLeaf $hf $fixture.root moss $false @() 1|Out-Null;throw 'allowed'}catch{if($_.Exception.Message -eq 'allowed'){throw};'blocked'}`, data), 'blocked');
  const blockers = JSON.parse(await powershell(`@((Get-StorageBlocker $fixture.target moss @([pscustomobject]@{ProcessId=2;Name='Runner.Worker.exe';CommandLine='worker';ExecutablePath='C:\\runner\\Runner.Worker.exe'}) 1),(Get-StorageBlocker $fixture.target cargo @([pscustomobject]@{ProcessId=3;Name='test.exe';CommandLine='';ExecutablePath=($fixture.target+'\\debug\\test.exe')}) 1))|ConvertTo-Json -Compress`, data));
  assert.match(blockers[0], /Another Actions/);
  assert.match(blockers[1], /references/);
});
