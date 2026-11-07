// DECLARED SYNTHETIC CPU fixtures only: no Mac, weights, personal inputs, device or campaign permit.
import assert from 'node:assert/strict';
import test from 'node:test';
import { createHash } from 'node:crypto';
import { mkdtemp, mkdir, readFile, rm, writeFile } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { resolveDispatch, hostRoot, readControl, verifyInputsManifest, laneAddendum, DATASET_INPUTS_MANIFEST_SHA256, NONE } from './flux3-terminal-host.mjs';
import { loadProductTargets, terminalPlan } from './flux3-image-klein-terminal-plan.mjs';
import { planHash } from './flux3-image-klein-terminal-budget.mjs';

const d = c => c.repeat(64), sha = bytes => createHash('sha256').update(bytes).digest('hex');
const source = 'a'.repeat(40), inference = 'b'.repeat(40);
const env = over => ({ FLUX3_TERMINAL_HOST: 'nax-macos-2', FLUX3_TERMINAL_STEP: 'prepare', FLUX3_TERMINAL_CONTROL_SHA256: d('c'), FLUX3_TERMINAL_SOURCE_SHA: source, FLUX3_TERMINAL_INFERENCE_SHA: inference, ...over });
async function temp(t) { const root = await mkdtemp(path.join(os.tmpdir(), 'flux3-terminal-host-DECLARED-SYNTHETIC-')); t.after(() => rm(root, { recursive: true, force: true })); return root; }

test('typed inputs: the "none" sentinel (what any empty dispatch value becomes) and unknown values are refused', () => {
  assert.equal(resolveDispatch(env()).runnerLabel, 'rw-mage');
  assert.equal(resolveDispatch(env({ FLUX3_TERMINAL_HOST: 'nax-macos' })).runnerTarget, 'primary');
  for (const [key, pattern] of [['FLUX3_TERMINAL_HOST', /host is the "none" sentinel/], ['FLUX3_TERMINAL_STEP', /step is the "none" sentinel/], ['FLUX3_TERMINAL_CONTROL_SHA256', /control_sha256 is the "none" sentinel/]]) {
    assert.throws(() => resolveDispatch(env({ [key]: NONE })), pattern); assert.throws(() => resolveDispatch(env({ [key]: '' })), pattern);
  }
  assert.throws(() => resolveDispatch(env({ FLUX3_TERMINAL_HOST: 'michael-trx50' })), /unknown terminal host/);
  assert.throws(() => resolveDispatch(env({ FLUX3_TERMINAL_STEP: 'measure' })), /unknown terminal step/);
  assert.throws(() => resolveDispatch(env({ FLUX3_TERMINAL_CONTROL_SHA256: 'C'.repeat(64) })), /64 lowercase hex/);
  assert.throws(() => resolveDispatch(env({ FLUX3_TERMINAL_SOURCE_SHA: 'main' })), /40-hex/);
});
test('wrong-host refusal: the actual runner name must be the dispatched host', () => {
  const dispatch = resolveDispatch(env());
  assert.equal(hostRoot(dispatch, { RUNNER_NAME: 'nax-macos-2', HOME: '/Users/declared' }), path.join('/Users/declared', 'sc26243-run'));
  assert.throws(() => hostRoot(dispatch, { RUNNER_NAME: 'nax-macos', HOME: '/Users/declared' }), /dispatched for nax-macos-2 but running on nax-macos/);
  assert.throws(() => hostRoot(dispatch, { HOME: '/Users/declared' }), /unnamed runner/);
});
test('the control file is pinned by the dispatched digest and must name this host, step and source', async t => {
  const root = await temp(t), dispatch = resolveDispatch(env({ FLUX3_TERMINAL_HOST: 'nax-macos' }));
  await assert.rejects(readControl(root, dispatch), /not recorded on this host/);
  await mkdir(path.join(root, 'control', 'nax-macos'), { recursive: true });
  const write = async value => { const bytes = JSON.stringify(value); await writeFile(path.join(root, 'control', 'nax-macos', 'prepare.json'), bytes); return sha(bytes); };
  const good = { kind: 'flux3-terminal-host-control', host: 'nax-macos', step: 'prepare', sourceSha: source, inferenceMainSha: inference };
  await write(good); await assert.rejects(readControl(root, dispatch), /differs from the dispatched sha256/);
  assert.equal((await readControl(root, { ...dispatch, controlSha256: await write(good) })).host, 'nax-macos');
  await assert.rejects(readControl(root, { ...dispatch, controlSha256: await write({ ...good, host: 'nax-macos-2' }) }));
});
test('the dataset host verifies every input against the approved manifest and refuses any mismatch', async t => {
  const root = await temp(t), png = Buffer.from('DECLARED SYNTHETIC NOT A PHOTO'), caption = Buffer.from('declared synthetic caption');
  await mkdir(path.join(root, 'references')); await mkdir(path.join(root, 'training'));
  const manifest = { schemaVersion: 1, references: [{ index: 0, file: 'ref-00.png', sha256: sha(png), bytes: png.length }], training: [{ index: 0, file: 'train-00.png', sha256: sha(png), bytes: png.length, captionSha256: sha(caption) }] };
  const manifestBytes = JSON.stringify(manifest); await writeFile(path.join(root, 'inputs-manifest.json'), manifestBytes);
  await writeFile(path.join(root, 'references/ref-00.png'), png); await writeFile(path.join(root, 'training/train-00.png'), png); await writeFile(path.join(root, 'training/train-00.txt'), caption);
  assert.deepEqual(await verifyInputsManifest(root, sha(manifestBytes)), { manifestSha256: sha(manifestBytes), references: 1, training: 1, files: 3 });
  await assert.rejects(verifyInputsManifest(root), /differs from the approved manifest/, 'only the approved a9551ff9 manifest is accepted by default');
  assert.equal(DATASET_INPUTS_MANIFEST_SHA256, 'a9551ff98b7088c3653052079ba19e00644190f0f7aac878b58d8835e886f0d9');
  await writeFile(path.join(root, 'training/train-00.txt'), 'changed'); await assert.rejects(verifyInputsManifest(root, sha(manifestBytes)), /differs from the manifest/);
  await writeFile(path.join(root, 'training/train-00.txt'), caption); await writeFile(path.join(root, 'references/extra.png'), png);
  await assert.rejects(verifyInputsManifest(root, sha(manifestBytes)), /unexpected input entry/);
  await rm(path.join(root, 'references/extra.png')); await rm(path.join(root, 'references/ref-00.png'));
  await assert.rejects(verifyInputsManifest(root, sha(manifestBytes)), /missing/);
});
test('run is refused until this host lane\'s addendum is recorded (another lane\'s never counts)', async t => {
  const sources = await loadProductTargets(), plan = terminalPlan(sources, { addendumBackends: ['mlx'], hostLanes: { mlx: { datasetHost: 'nax-macos', generalHost: 'nax-macos-2' } } });
  const root = await temp(t), approval = { schemaVersion: 1, kind: 'flux3-terminal-plan-approval', approvalId: 'DECLARED-SYNTHETIC', userPlanApprovalId: 'NOT_USER_AUTHORIZATION', approvedBy: 'CPU fixture', sourceSha: source, inferenceMainSha: inference, ledgerId: 'declared-host-ledger',
    ledgerRoots: { candle: path.join(root, 'c'), 'mlx@nax-macos': path.join(root, 'm1'), 'mlx@nax-macos-2': path.join(root, 'm2') }, inputSpecSha256: { candle: d('c') }, deferredInputSpecBackends: ['mlx@nax-macos', 'mlx@nax-macos-2'], plan };
  const addendum = lane => ({ schemaVersion: 1, kind: 'flux3-terminal-plan-input-spec-addendum', approvedPlanSha256: planHash(approval), ledgerId: approval.ledgerId, backend: 'mlx', lane, inputSpecSha256: d('e'),
    buildReceipt: { receiptSha256: d('5'), binarySha256: d('6') }, resourceBuildReceipt: { receiptSha256: d('7'), binarySha256: d('8') }, nativeBoundaryReceipt: null, parentNativeBoundaryVerificationId: 'DECLARED', addendumId: 'DECLARED', userApprovalId: 'NOT_USER_AUTHORIZATION', approvedBy: 'CPU fixture' });
  await assert.rejects(laneAddendum(approval, undefined, 'nax-macos-2'), /run refused: no recorded addendum for mlx@nax-macos-2/);
  // The shipped default plan keeps resource requests unbound, so no lane has READY_ON_ADDENDUM rows to prove here.
  assert.equal((await laneAddendum(approval, addendum('nax-macos-2'), 'nax-macos-2')).ledgerLane, 'mlx@nax-macos-2');
  await assert.rejects(laneAddendum(approval, addendum('nax-macos'), 'nax-macos-2'), /binds mlx@nax-macos, not mlx@nax-macos-2/);
});
test('the workflow can never upload: no artifact/upload step, no secrets, manual registered profile only', async () => {
  const text = await readFile(new URL('../.github/workflows/flux3-image-klein-terminal-host.yml', import.meta.url), 'utf8');
  const publicationRoutes = /upload-artifact|actions\/cache|upload-pages|gh release|\bgists?\b|secrets\./i;
  assert.match('gist', publicationRoutes);
  assert.doesNotMatch(text, publicationRoutes, 'no upload or secret path of any kind');
  assert.match(text, /^on:\n {2}workflow_call:/m, 'reusable only: reached through the manual campaign dispatcher');
  assert.doesNotMatch(text, /^ {2}(push|pull_request|schedule|workflow_dispatch):/m);
  assert.match(text, /\[\[ "\$RUNNER_NAME" == "\$FLUX3_TERMINAL_HOST" \]\]/, 'runner name checked before checkout');
  assert.match(text, /persist-credentials: false/);
  const campaign = await readFile(new URL('../.github/workflows/memory-catalog-campaign.yml', import.meta.url), 'utf8');
  for (const input of ['flux3_terminal_host', 'flux3_terminal_step', 'flux3_terminal_control_sha256']) {
    const block = campaign.slice(campaign.indexOf(`      ${input}:`)).split(/\n {6}[a-z_]+:\n/)[0];
    assert.match(block, /default: none/, `${input} defaults to the refused sentinel`);
  }
  assert.match(campaign, /uses: \.\/\.github\/workflows\/flux3-image-klein-terminal-host\.yml/);
  assert.match(campaign, /if: \$\{\{ needs\.moss-selector-preflight\.outputs\.enabled != 'true' && inputs\.profile == 'flux3-image-klein-terminal-host' \}\}/);
});
test('host script never echoes child output: children log to local files only', async () => {
  const text = await readFile(new URL('./flux3-terminal-host.mjs', import.meta.url), 'utf8');
  assert.match(text, /stdio: \['ignore', log\.fd, log\.fd\]/);
  assert.doesNotMatch(text, /stdio: 'inherit'|stdio: \[[^\]]*'inherit'/);
  // A Mac preparation is admitted on either known Mac (its runner must be the observed host).
  const prep = await readFile(new URL('./flux3-image-klein-preparation.mjs', import.meta.url), 'utf8');
  assert.doesNotMatch(prep, /assert\.equal\(a\.runner, 'nax-macos-2'\)/);
});
