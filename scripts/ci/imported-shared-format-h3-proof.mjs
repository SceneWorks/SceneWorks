// One existing full-AV product case. Reuse the owned-service, API, decoding and shutdown harness;
// this is not a rerun or acceptance of the complete historical H3 campaign.
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { copyFile, mkdir, readFile, stat, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { buildScenario, assertLicenseAuthority } from '../h3-finetune-terminal-cases.mjs';
import { operationalCaseById } from '../h3-finetune-terminal-reduced.mjs';
import { executeAPICase } from '../h3-finetune-terminal-runtime.mjs';
import { fileSha256 } from '../lib/file-sha256.mjs';
import { stripVerbatim } from '../lib/windows-verbatim-path.mjs';
import { lockedInferenceRevision } from '../select-windows-cargo-cache.mjs';

export function makeScenario(fixture, output, sourcePath, inputs, seeds) {
  const c = structuredClone(operationalCaseById('cuda.reduced.real-independent-fl2va'));
  assert(c && c.kind === 'full-av');
  const binding = structuredClone(fixture.binding);
  binding.model.sourcePath = sourcePath;
  for (const component of binding.model.components) {
    component.path = path.join(output, 'inputs', 'hf/hub', `models--${component.source.replaceAll('/', '--')}`,
      'snapshots', component.revision, component.file);
    assert(inputs.some(input => input.path === component.path && input.role === 'component'
      && input.sha256 === component.sha256), 'every inherited component must bind its exact staged file');
  }
  assertLicenseAuthority(binding); // Retain the actual prior operator scope; never fabricate consent.
  const runDir = path.join(output, 'services', c.id);
  const scenario = buildScenario(c, {
    projectId: { $capture: 'project/id' }, seed: fixture.geometry.seed, requestedGpu: '1',
    timeoutSeconds: 7200, baseUrl: 'http://127.0.0.1:0',
    setupSteps: [{ id: 'project', method: 'POST', path: '/api/v1/projects',
      body: { name: 'Shared ConvRot H3 acceptance' }, expectedStatus: 201 }],
  }, [binding]);
  for (const step of scenario.steps) {
    if (step.path === '/api/v1/models/import') {
      step.body.sourcePath = path.join(runDir, 'data/models/shared-format-source/model.safetensors');
      step.assert.push({ pointer: '/payload/manifestEntry/importComponentTier', equals: binding.model.componentTier });
    }
    if (step.path === '/api/v1/video/jobs') {
      // The immutable DiT source owns its precision. The assembly still explicitly chooses Q4
      // inherited components through its authoritative imported assembly.
      delete step.body.advanced.mlxQuantize;
      assert.equal(step.body.seed, fixture.geometry.seed);
      for (const key of ['width', 'height', 'fps', 'duration', 'mode']) assert.equal(step.body[key], fixture.geometry[key]);
    }
  }
  c.inputs = inputs;
  c.declaredSourceBindings = [binding];
  c.evidence = 'evidence.json'; c.timeoutSeconds = 7200;
  c.loadHostBudgetBytes = 64 * 2 ** 30; c.loadDeviceBudgetBytes = 96 * 2 ** 30;
  c.runtime = { port: 0, gpuId: '1', deviceId: 'GPU-e4b79931-7be6-f216-460a-f5405cfafffe',
    physicalGpuIndex: 1, physicalGpuUuid: 'GPU-e4b79931-7be6-f216-460a-f5405cfafffe', seedFiles: seeds };
  return { c, scenario };
}

const canonical = value => path.win32.normalize(stripVerbatim(value)).toLowerCase();
export function validateNativeH3Receipts(log, bundle) {
  assert.equal(bundle.jobs.length, 1, 'exactly one normal H3 generation required');
  const job = bundle.jobs[0]; assert.equal(job.status, 'completed');
  const binding = bundle.bindings.find(row => row.job.id === job.id);
  assert(binding, 'completed request must carry its actual sealed load closure');
  const event = binding.event.fields ?? binding.event;
  const prepared = typeof event.prepared_sources === 'string' ? JSON.parse(event.prepared_sources) : event.prepared_sources;
  const sources = new Set(prepared.map(row => canonical(row.path)));
  const events = log.split('\n').flatMap(line => { try { return [JSON.parse(line)]; } catch { return []; } })
    .map(row => row.fields ?? row).filter(row => row.event === 'checkpoint_weight_materialization' && row.jobId === job.id);
  assert.equal(events.length, 1, 'one post-forward receipt event for the completed request required');
  const receipts = JSON.parse(events[0].receipts);
  assert(receipts.length > 0, 'actual measured source receipts required');
  let nativeTensors = 0;
  for (const receipt of receipts) {
    assert(sources.has(canonical(receipt.sourcePath)), 'receipt source is outside this request sealed load closure');
    const materialization = receipt.checkpointWeightFacts?.materialization;
    assert.equal(materialization?.status, 'reported');
    nativeTensors += materialization.rows.filter(row => row.codecId === 'convrot-int8-native'
      && row.representation === 'native-packed').reduce((count, row) => count + row.tensorCount, 0);
  }
  assert(nativeTensors > 0, 'the completed AV request must have executed native ConvRot operands');
  return { jobId: job.id, nativeTensors, receipts };
}

async function main() {
  assert.equal(process.env.GITHUB_EVENT_NAME, 'workflow_dispatch');
  assert.equal(process.env.SCENEWORKS_GPU_ID, '1');
  assert(/^[1-9][0-9]*$/.test(process.env.SW_CI_CUDA_EXECUTION_ACTIVE ?? ''), 'physical guard must own this proof');
  const swRoot = process.cwd(), inferenceRoot = path.resolve(process.env.SHARED_FORMAT_INFERENCE_ROOT);
  const output = path.resolve(`${process.env.MODEL_PROOF_OUTPUT_DIR}-h3`);
  assert(!/^e:/i.test(output));
  const relative = path.relative(path.resolve(process.env.RUNNER_TEMP), output);
  assert(relative && !relative.startsWith('..') && !path.isAbsolute(relative));
  const source = execFileSync('git', ['rev-parse', 'HEAD'], { encoding: 'utf8' }).trim();
  assert.equal(source, process.env.GITHUB_SHA);
  const inference = lockedInferenceRevision(await readFile('Cargo.lock', 'utf8'));
  const fixture = JSON.parse(await readFile(new URL('../../docs/testing/shared-format-h3-convrot.json', import.meta.url), 'utf8'));
  await mkdir(output);
  const assets = path.join(output, 'inputs'); await mkdir(assets);
  const inputs = [], seeds = [];
  const stage = async (spec, role, destination) => {
    assert(/^[a-zA-Z0-9_.-]+\/[a-zA-Z0-9_.-]+$/.test(spec.source ?? spec.repo));
    assert(/^[a-f0-9]{40}$/.test(spec.revision));
    const repo = (spec.source ?? spec.repo).replaceAll('/', '--');
    const from = path.join(process.env.HF_HUB_CACHE, `models--${repo}`, 'snapshots', spec.revision, spec.file);
    const to = path.join(assets, destination);
    assert(!path.relative(assets, to).startsWith('..'));
    await mkdir(path.dirname(to), { recursive: true });
    const info = await stat(from); if (spec.bytes !== undefined) assert.equal(info.size, spec.bytes);
    await copyFile(from, to);
    assert.equal(await fileSha256(to), spec.sha256, 'staged bytes differ from the immutable source binding');
    inputs.push({ path: to, role, sha256: spec.sha256 });
    seeds.push({ input: to, destination, mode: 'hardlink' });
    return to;
  };
  const sourcePath = await stage({ ...fixture.source, sha256: fixture.binding.model.sha256 },
    'checkpoint', 'data/models/shared-format-source/model.safetensors');
  const refs = new Map();
  for (const component of [...fixture.binding.model.components, ...fixture.extraComponents]) {
    const repo = component.source.replaceAll('/', '--');
    await stage(component, 'component', `hf/hub/models--${repo}/snapshots/${component.revision}/${component.file}`);
    refs.set(repo, component.revision);
  }
  for (const [repo, revision] of refs) {
    const file = path.join(assets, `hf/hub/models--${repo}/refs/${revision}`);
    await mkdir(path.dirname(file), { recursive: true }); await writeFile(file, revision);
    inputs.push({ path: file, role: 'component', sha256: await fileSha256(file) });
    seeds.push({ input: file, destination: `hf/hub/models--${repo}/refs/${revision}`, mode: 'copy' });
  }
  const { c, scenario } = makeScenario(fixture, output, sourcePath, inputs, seeds);
  const scenarioPath = path.join(output, 'scenario.json'); await writeFile(scenarioPath, JSON.stringify(scenario));
  c.runtime.scenario = scenarioPath;
  c.inputs.push({ path: scenarioPath, role: 'scenario', sha256: await fileSha256(scenarioPath) });
  const p = { version: 1, scope: 'shared-format-single-functional-case', inventory: 'reduced-v1',
    sceneworks: source, inference, output, source: fixture.source, cases: [c], responsibleTask: 'SW-7234' };
  await writeFile(path.join(output, 'plan.json'), JSON.stringify(p, null, 2));
  process.env.H3_TERMINAL_AUTHORIZED_PAIR = `${source}:${inference}`;
  const dir = path.join(output, c.id); await mkdir(dir);
  await executeAPICase(p, c, swRoot, inferenceRoot, dir);
  const bundle = JSON.parse(await readFile(path.join(dir, 'source-job-output.json'), 'utf8'));
  const log = (await Promise.all(['api-1.log', 'worker-1.log'].map(name => readFile(path.join(dir, name), 'utf8')))).join('\n');
  const receipt = validateNativeH3Receipts(log, bundle);
  await writeFile(path.join(output, 'native-materialization.json'), JSON.stringify({ source, inference, ...receipt }, null, 2));
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  main().catch(error => { console.error(error); process.exitCode = 1; });
}
