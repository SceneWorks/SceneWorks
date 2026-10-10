#!/usr/bin/env node
// Build the workspace's locked glib, then exercise its public API at opt-level=3.
// This avoids compiling the whole desktop just to run an FFI regression, and does
// not create a separate Cargo dependency resolution for the test.
import { spawnSync } from 'node:child_process';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

if (process.platform !== 'linux') throw new Error('Run this desktop dependency regression on Linux');

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
function run(command, args, capture = false) {
  const result = spawnSync(command, args, {
    cwd: root,
    encoding: 'utf8',
    stdio: capture ? ['ignore', 'pipe', 'inherit'] : 'inherit',
    maxBuffer: 32 * 1024 * 1024,
  });
  if (result.error) throw result.error;
  if (result.status !== 0) throw new Error(`${command} failed (${result.status ?? result.signal})`);
  return result.stdout;
}

// The desktop and this regression must consume the same glib patch.
const host = run('rustc', ['-vV'], true).match(/^host: (.+)$/m)?.[1];
if (!host?.includes('linux')) throw new Error('Expected a native Linux Rust toolchain');
const tree = run('cargo', ['tree', '--locked', '-p', 'sceneworks-desktop', '-i', 'glib',
  '--target', host], true);
if (!tree.includes(`glib v0.18.5 (${path.join(root, 'vendor/glib-0.18.5')})`) ||
    !tree.includes('sceneworks-desktop')) {
  throw new Error('Linux desktop no longer resolves the maintained glib patch; review the regression and provenance');
}
process.stdout.write(tree);
const output = run('cargo', ['build', '--locked', '--release', '-p', 'glib', '--message-format=json'], true);
const artifacts = output.trim().split('\n').map((line) => JSON.parse(line));
const artifact = artifacts.find((item) => item.reason === 'compiler-artifact' && item.target.name === 'glib');
const rlib = artifact?.filenames.find((file) => file.endsWith('.rlib'));
if (!rlib || artifact.profile.opt_level !== '3') throw new Error('Missing optimized glib artifact');
const temp = mkdtempSync(path.join(tmpdir(), 'sceneworks-glib-regression-'));
try {
  const binary = path.join(temp, 'glib-variant-iter');
  run('rustc', ['--edition=2021', '--test', '-C', 'opt-level=3',
    path.join(root, 'scripts/fixtures/glib-variant-iter.rs'),
    '--extern', `glib=${rlib}`, '-L', `dependency=${path.join(path.dirname(rlib), 'deps')}`, '-o', binary]);
  run(binary, ['--nocapture']);
} finally {
  rmSync(temp, { recursive: true, force: true });
}
