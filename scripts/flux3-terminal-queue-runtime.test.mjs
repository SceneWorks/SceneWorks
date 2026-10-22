// Real Windows PS5 execution of the EXACT workflow wrapper, substituting only an owned CPU dummy
// entrypoint. No queue/API/model/GPU work; no tool installation or configuration changes.
import assert from 'node:assert/strict';
import test from 'node:test';
import { readFile, writeFile, mkdtemp, rm } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { execFile as execCb } from 'node:child_process';
import { promisify } from 'node:util';
import path from 'node:path';
import os from 'node:os';
import { driverEnvironment } from './flux3-terminal-queue.mjs';
const exec = promisify(execCb), windows = process.platform === 'win32';
const NODE = 'D:/tmp/sc26242-node20-extract-20261020/node-v20.20.2-win-x64/node.exe';
const NODE_SHA = '56c1520ee33b801e8bdb92fb321cf2e98529735b6d12bd4a2a6dec0ac0bab937';
const ENTRY = 'D:/repos/SceneWorks/.claude/worktrees/agent-aafe2575fa619862f/scripts/flux3-terminal-queue.mjs';
const workflow = await readFile(new URL('../.github/workflows/flux3-image-klein-terminal-queue.yml', import.meta.url), 'utf8');
const inline = /- name: Reserve one slot[^\n]*\n        run: \|\n([\s\S]*?)\n        env:/.exec(workflow)?.[1]?.split('\n').map(l => l.replace(/^ {10}/, '')).join('\n');

test('workflow requires native Windows PowerShell and hash-pinned admitted Node20, preserving the homogeneous four-slot contract', () => {
  assert(inline, 'exact reservation wrapper must be extractable');
  assert.match(workflow, /shell: powershell/); assert.doesNotMatch(workflow, /\bpwsh\b/);
  assert(inline.includes(`& '${NODE}' '${ENTRY}' reserve`)); assert(inline.includes(NODE_SHA)); assert.match(inline, /Get-FileHash/);
  assert.match(inline, /Get-Command \$tool -ErrorAction Stop/);
  assert.match(workflow, /runs-on: \[self-hosted, Windows, X64, cuda\]/); assert.match(workflow, /slot: \[1, 2, 3, 4\]/); assert.match(workflow, /max-parallel: 4/);
  assert.match(workflow, /CUDA_VISIBLE_DEVICES: "-1"/); assert.doesNotMatch(workflow, /upload-artifact|actions\/cache|Stop-ScheduledTask|Stop-Service|Set-Acl|schtasks|winget|choco install/i);
});
test('actual Windows PS5 executes the exact wrapper with correct args, CPU mask and nonzero exit propagation', { skip: !windows, timeout: 60000 }, async t => {
  const dir = await mkdtemp(path.join(os.tmpdir(), 'flux3 PS5 CPU wrapper-')); t.after(() => rm(dir, { recursive: true, force: true }));
  assert.equal(createHash('sha256').update(await readFile(NODE)).digest('hex'), NODE_SHA, 'same admitted archive-backed interpreter bytes');
  const version = (await exec(NODE, ['--version'])).stdout.trim(); assert.equal(version, 'v20.20.2');
  const shell = path.join(process.env.SystemRoot ?? process.env.SYSTEMROOT, 'System32/WindowsPowerShell/v1.0/powershell.exe');
  const shellVersion = (await exec(shell, ['-NoProfile','-Command','$PSVersionTable.PSVersion.Major'])).stdout.trim(); assert.equal(shellVersion, '5');
  const dummy = path.join(dir, 'owned dummy.mjs');
  await writeFile(dummy, `import assert from 'node:assert/strict';assert.deepEqual(process.argv.slice(2),['reserve']);assert.equal(process.env.CUDA_VISIBLE_DEVICES,'-1');assert.equal(process.version,'v20.20.2');console.log(JSON.stringify({fixture:'CPU_ONLY',args:process.argv.slice(2),cuda:process.env.CUDA_VISIBLE_DEVICES,node:process.version}));process.exit(Number(process.env.FLUX3_FIXTURE_EXIT??0));`);
  const wrapped = inline.replace(ENTRY, dummy.replaceAll('\\', '/').replaceAll("'", "''"));
  assert.equal(wrapped.replace(dummy.replaceAll('\\', '/').replaceAll("'", "''"), ENTRY), inline, 'substitute ONLY the Node entrypoint, not wrapper logic');
  const file = path.join(dir, 'exact-wrapper.ps1'); await writeFile(file, wrapped, 'utf8');
  const env = driverEnvironment(process.env, null);
  const result = await exec(shell, ['-NoProfile','-NonInteractive','-File',file], { env, timeout: 20000 });
  assert.deepEqual(JSON.parse(result.stdout.trim()), { fixture:'CPU_ONLY',args:['reserve'],cuda:'-1',node:'v20.20.2' });
  await assert.rejects(exec(shell, ['-NoProfile','-NonInteractive','-File',file], { env:{...env,FLUX3_FIXTURE_EXIT:'17'}, timeout:20000 }), error => error.code === 17, 'native wrapper must propagate the owned child exit code');
});
