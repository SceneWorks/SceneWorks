// The sole Windows queue executor. Invoked only as the verified supervisor's credential-free child.
// Photos/captions/images/reports stay on this machine; only compact hashes/counts are milestones.
import assert from 'node:assert/strict';
import { readFile, writeFile, mkdir, stat, statfs, appendFile, realpath } from 'node:fs/promises';
import { execFile as execCb } from 'node:child_process';
import { promisify } from 'node:util';
import { userInfo } from 'node:os';
import path from 'node:path';
import { CONTROL_FILE, QUEUE_ROOT, SOURCE_ROOT, digest, verifyQueueReservation, QueuePhaseFailure, QUEUE_PHASES } from './flux3-terminal-queue.mjs';
import { observeProcessTable, guardHardware, validateAdmission, fileHash } from './flux3-image-klein-terminal-session.mjs';
import { validateApprovedPlan, planHash, initializeLedger, auditLedger, appendCandleSourceTransition } from './flux3-image-klein-terminal-budget.mjs';
import { verifyTerminalBuild } from './flux3-image-klein-terminal-build.mjs';
import { loadProductTargets, targetRegistry, requestedDiskBudget } from './flux3-image-klein-terminal-plan.mjs';
import { artifactStagingPlan } from './flux3-image-klein-terminal-artifacts.mjs';
import { runTerminal, RETAINED_REPLAY_CASE, RETAINED_REPLAY_FIRST } from './flux3-image-klein-terminal-driver.mjs';
import { productionResourceRequest } from './flux3-native-boundary-self-test.mjs';
import { launchFlux3Resource } from './flux3-resource-launcher.mjs';
const exec = promisify(execCb), ROOT = 'D:/tmp/sc26243-run';
const json = async file => JSON.parse(await readFile(file, 'utf8'));
async function controlArtifact(binding) {
  assert(/^[a-f0-9]{64}$/.test(binding?.sha256 ?? ''));
  const actual = await realpath(binding.path), relative = path.relative(await realpath(ROOT), actual);
  assert(relative && relative !== '..' && !relative.startsWith(`..${path.sep}`) && !path.isAbsolute(relative), 'queue control artifact escaped owned external root');
  assert.equal(await fileHash(actual), binding.sha256, 'queue control artifact digest mismatch');
  return json(actual);
}
export function executionGroups(plan) {
  const invocations = plan.generationInvocations.filter(i => i.backend === 'candle'), groups = [];
  // Replay's two permits and training/export/zero/on must share sessions; no permutation expansion.
  for (let n = 0; n < invocations.length;) {
    const i = invocations[n];
    const group = i.purpose === 'training-reload' ? invocations.slice(n, n + 3) : invocations.slice(n).filter(j => j.caseId === i.caseId);
    if (i.purpose === 'training-reload') assert(group.length === 3 && group[1].adapterBinding && group[2].adapterBinding, 'one training/export/zero/on group required');
    groups.push(group); n += group.length;
  }
  assert.equal(groups.flat().length, 21, 'queue runs exactly the approved 21 Candle invocations');
  assert.deepEqual(groups.flat().map(i => i.invocationId), invocations.map(i => i.invocationId), 'queue order must equal the approved plan order');
  return groups;
}
export function assertGroupPassed(report, ids) {
  const selected = report.cases.filter(c => ids.includes(c.caseId));
  assert.equal(selected.length, ids.length);
  assert(!report.error && !report.cleanupError && !report.ledgerError && selected.every(c => c.executionStatus === 'PASSED'), 'FIRST_FAILURE_STOP: preserve local report and spent permits, never retry');
}
export function validateQueueOperation(operation) {
  assert(['run', 'admission-only'].includes(operation), 'explicit queue operation must be run or admission-only');
  return operation;
}
// Both modes reach this gate only AFTER the SAME production admission path. Diagnostic mode
// never even invokes ledger/provider/model callbacks, and is not terminal image evidence.
export async function executeAdmittedQueueOperation(operation, { initialize, execute, finalize }) {
  validateQueueOperation(operation);
  if (operation === 'admission-only') return { operation, status: 'DIAGNOSTIC_ONLY', generationCount: 0, campaignPermitsConsumed: 0, ledgerInitialized: false };
  let initialized = false;
  try { await initialize(); initialized = true; await execute(); }
  finally { if (initialized) await finalize(); }
  return { operation, status: 'COMPLETED', ledgerInitialized: true };
}
export async function runQueuedCandleCampaign(env) {
  const trace = { phase: 'queue_ownership', metrics: {} };
  try { return await runCampaignInner(env, trace); }
  catch (error) {
    const failure = new QueuePhaseFailure(error, trace.phase, trace.metrics);
    if (trace.failureFile) await writeFile(trace.failureFile, JSON.stringify(failure.report) + '\n', { flag: 'wx' }).catch(() => {});
    throw failure;
  }
}
async function runCampaignInner(env, trace) {
  const phase = value => { assert(QUEUE_PHASES.includes(value)); trace.phase = value; };
  assert.equal(process.platform, 'win32');
  const ref = JSON.parse(env.FLUX3_QUEUE_BINDING ?? 'null'); assert(ref, 'supervisor-bound queue child only');
  const initial = JSON.parse(await readFile(ref.path, 'utf8')); assert.equal(digest(JSON.stringify(initial)), ref.sha256);
  const identity = initial.identity;
  // Full four-slot barrier before opening ANY control artifact/model, ledger or CUDA context.
  await verifyQueueReservation({ sourceSha: identity.sourceSha, inferenceMainSha: identity.inferenceMainSha, queueReservation: ref }, await observeProcessTable(exec, 'win32'), process.pid);
  trace.failureFile = path.join(QUEUE_ROOT, `${identity.runId}-${identity.attempt}`, 'campaign.failure.json');
  phase('control');
  const bytes = await readFile(CONTROL_FILE); assert.equal(digest(bytes), identity.controlSha256);
  const control = JSON.parse(bytes); assert.equal(control.kind, 'flux3-queued-candle-control');
  validateQueueOperation(control.operation);
  assert.equal(control.sourceSha, identity.sourceSha); assert.equal(control.inferenceMainSha, identity.inferenceMainSha);
  phase('metadata');
  const approval = await controlArtifact(control.approval), inputSpec = await controlArtifact(control.inputSpec);
  const approved = validateApprovedPlan(approval); assert.equal(approval.sourceSha, identity.sourceSha); assert.equal(approval.inferenceMainSha, identity.inferenceMainSha);
  if (approval.candleSourceTransitionSha256 !== undefined) assert.equal(control.continuation?.sha256, approval.candleSourceTransitionSha256, 'current approval must pin the exact explicit Candle continuation authorization');
  else assert.equal(control.continuation, undefined, 'a continuation cannot be supplied outside its current approval');
  assert.equal(planHash(inputSpec), approval.inputSpecSha256.candle);
  const api = await controlArtifact(control.apiBuild), resource = await controlArtifact(control.resourceBuild);
  const plan = approval.plan, sources = await loadProductTargets(), registry = targetRegistry(sources), groups = executionGroups(plan);
  const outputRoot = path.join(QUEUE_ROOT, `${identity.runId}-${identity.attempt}`, 'outputs'); await mkdir(outputRoot);
  const milestoneFile = path.join(path.dirname(outputRoot), 'milestones.jsonl');
  const milestone = async v => { await appendFile(milestoneFile, JSON.stringify(v) + '\n'); console.log(JSON.stringify(v)); };
  phase('license');
  const admitted = control.admission;
  assert(admitted && admitted.trainingLicenseAcknowledged === true && typeof admitted.trainingLicenseAcceptance === 'string' && admitted.trainingLicenseAcceptance.includes('And I accept the license'), 'actual user training-license acceptance must be recorded verbatim');
  assert.equal(admitted.physicalGpuId, '1'); assert.equal(admitted.cudaVisibleDevices, '1'); assert.equal(admitted.logicalCudaOrdinal, 0);
  phase('disk_metadata');
  const present = {}, conversionOutputBytes = {};
  for (const t of registry.filter(t => t.backend === 'candle')) for (const d of t.model.downloads) {
    if (Object.values(inputSpec.packages).some(p => p.repository === d.repo && p.resolvedRevision === d.revision)) present[`${d.repo}@${d.revision}`] = d.estimatedSizeBytes;
    if (d.localDerivation) conversionOutputBytes[`${t.modelId}:candle:${d.variant}`] = Math.max(d.localDerivation.weightsBytes, d.footprint.diskSizeBytes - d.estimatedSizeBytes);
  }
  const cells = plan.cases.filter(c => c.backend === 'candle' && groups.flat().some(i => i.caseId === c.caseId));
  for (const p of artifactStagingPlan(cells, registry, inputSpec).derivedPackages) conversionOutputBytes[p.artifactKey] = Math.max(conversionOutputBytes[p.artifactKey] ?? 0, p.copyBytes);
  const available = async directory => { const fs = await statfs(directory, { bigint: true }); return Number(fs.bavail * fs.bsize); };
  const disk = { present, conversionOutputBytes, freeBytes: Math.min(await available(outputRoot), await available(admitted.hfHome)) };
  trace.metrics.diskFreeBytes = disk.freeBytes;
  phase('disk_budget');
  trace.metrics.diskRequiredBytes = requestedDiskBudget(registry, cells, disk).requiredBytes;
  const sealed = result => ({ binary: result.receipt.binary.path, binarySha256: result.receipt.binary.sha256, binarySourceSha: result.receipt.sourceSha, binaryBuildReceipt: { path: result.receiptPath, root: path.dirname(result.receiptPath), sha256: result.receiptSha256 } });
  const base = { actualRun: true, backend: 'candle', admissionId: control.admissionId, approvedBy: control.approvedBy, phases: plan.phases, runner: 'cuda-windows', runnerAccount: userInfo().username,
    sourceSha: approval.sourceSha, inferenceMainSha: approval.inferenceMainSha, inferenceRepository: admitted.inferenceRepository, inferenceMainTipSha: admitted.inferenceMainTipSha,
    outputRoot, hfHome: admitted.hfHome, timeoutSeconds: admitted.timeoutSeconds, approvedPlan: approval, approvedPlanSha256: approved.approvedPlanSha256, ledgerId: approval.ledgerId, ledgerRoot: approval.ledgerRoots.candle,
    inputSpecSha256: planHash(inputSpec), physicalGpuId: '1', cudaVisibleDevices: '1', logicalCudaOrdinal: 0, gpuUuid: admitted.gpuUuid, disk,
    trainingLicenseAcknowledged: true, trainingLicenseAcceptance: admitted.trainingLicenseAcceptance, queueReservation: ref };
  const admission = { ...base, ...sealed(api) }, resourceAdmission = { ...base, ...sealed(resource) };
  phase('source_pins'); await validateAdmission(admission, 'candle');
  phase('api_receipt'); await verifyTerminalBuild(admission);
  phase('resource_receipt'); await verifyTerminalBuild(resourceAdmission, { role: 'resource-adapter' });
  phase('queue_ownership'); const device = await guardHardware(admission, [], { onPhase: phase });
  // CUDA1 capacity excludes measured foreign desktop memory; do not subtract it from already-free
  // VRAM a second time. Every native resource capture also carries its live foreign-memory proof.
  phase('gpu_capacity');
  const smi = (await exec('nvidia-smi', ['-i', '1', '--query-gpu=memory.total', '--format=csv,noheader,nounits'], { timeout: 5000 })).stdout.trim();
  assert(/^\d+$/.test(smi));
  const capacityBytes = Number(smi) * 1024 ** 2, foreignBytes = device.foreignGraphics.totalBytes;
  trace.metrics.physicalCapacityBytes = capacityBytes; trace.metrics.foreignDedicatedBytes = foreignBytes;
  assert(capacityBytes > foreignBytes);
  phase('admission_write');
  await writeFile(path.join(path.dirname(outputRoot), 'admission.json'), JSON.stringify({ ...admission, operation: control.operation, diagnosticOnly: control.operation === 'admission-only', deviceAtAdmission: device, resourceHeadroom: { physicalCapacityBytes: capacityBytes, foreignDedicatedBytes: foreignBytes, capacityAfterForeignBytes: capacityBytes - foreignBytes, alreadyFreeVramMustNotBeSubtractedAgain: true } }, null, 2), { flag: 'wx' });
  await milestone({ kind: 'admission', operation: control.operation, physicalGpuId: '1', gpuUuid: admitted.gpuUuid, foreignBytes, capacityAfterForeignBytes: capacityBytes - foreignBytes, diskFreeBytes: disk.freeBytes });
  let continuationContext;
  const result = await executeAdmittedQueueOperation(control.operation, {
    initialize: async () => {
      if (approval.candleSourceTransitionSha256 !== undefined) {
        phase('ledger_transition');
        continuationContext = await appendCandleSourceTransition({ approval, ledgerRoot: base.ledgerRoot, continuation: control.continuation });
        await milestone({ kind: 'source-transition', historicalSourceSha: continuationContext.previousApproval.sourceSha, currentSourceSha: approval.sourceSha, authorizationSha256: approval.candleSourceTransitionSha256, headSha256: continuationContext.receipt.state.headSha256, generationCount: continuationContext.receipt.state.generationCount, noNewGeneration: true, historicalSupervisionStatus: 'FAILED' });
        return; // Never initialize or replace the existing claim/manifest/journal prefix.
      }
      phase('ledger_boundary');
      // Claim survives deletion of ledger dir. Neither an existing claim nor a journal is reset.
      for (const file of [base.ledgerRoot, path.join(path.dirname(base.ledgerRoot), `.flux3-${approval.ledgerId}-candle.initialized`)]) {
        try { await stat(file); throw new Error('existing ledger/claim requires coordinator reconciliation; no reset or reinit'); } catch (e) { if (e.code !== 'ENOENT') throw e; }
      }
      await mkdir(path.dirname(base.ledgerRoot), { recursive: true });
      phase('queue_ownership');
      await verifyQueueReservation(admission, await observeProcessTable(exec, 'win32'), process.pid);
      phase('ledger_init');
      await initializeLedger({ approval, backend: 'candle', ledgerRoot: base.ledgerRoot });
    },
    execute: async () => {
      phase('execution');
      const calibration = await json(path.join(SOURCE_ROOT, 'config/memory-calibration-plan.json'));
      for (const group of groups) {
        const start = Date.now(), invocation = group[0];
        if (invocation.kind === 'product_generation') {
          const ids = [...new Set(group.map(i => i.caseId))];
          const retained = continuationContext && invocation.caseId === RETAINED_REPLAY_CASE;
          const cellAdmission = retained ? { ...admission, replayPrefixAuthorizationSha256: approval.candleSourceTransitionSha256 } : admission;
          const { report } = await runTerminal({ admission: cellAdmission, inputSpec, sources, caseIds: ids, approvedPlan: approval }, { print: () => {} });
          const ledger = await auditLedger({ approval, backend: 'candle', ledgerRoot: base.ledgerRoot });
          if (retained) await milestone({ kind: 'retained-render-baseline', invocationId: RETAINED_REPLAY_FIRST, result: 'HISTORICAL_SUPERVISION_FAILED_RENDER_RETAINED', spent: true, newGeneration: false, pngSha256: continuationContext.baseline.pngSha256, semanticAcceptance: 'PENDING', generationCount: ledger.state.generationCount });
          for (const i of group) {
            if (retained && i.invocationId === RETAINED_REPLAY_FIRST) continue;
            const c = report.cases.find(c => c.caseId === i.caseId), index = retained ? 0 : group.filter(j => j.caseId === i.caseId).indexOf(i);
            const outputs = c?.kind === 'replay' ? (c.outputs ?? []).slice(index, index + 1) : c?.outputs ?? [];
            const spent = ledger.state.consumedIds.includes(i.invocationId);
            await milestone({ invocationId: i.invocationId, result: spent ? (c?.executionStatus ?? report.status) : 'NOT_EXECUTED', caseExecutionStatus: c?.executionStatus ?? report.status, spent, generationCount: ledger.state.generationCount, outputs: outputs.map(o => ({ sha256: o.pngSha256, width: o.width, height: o.height })), wallSeconds: (Date.now() - start) / 1000 });
          }
          assertGroupPassed(report, ids);
        } else {
          const pkg = inputSpec.packages[`${invocation.modelId}:candle:${invocation.tier}`], dense = inputSpec.packages[`${invocation.modelId}:candle:bf16`];
          assert.equal(control.resourceInputRoot, 'D:/tmp/sc26243-run/resource-inputs');
          const result = await launchFlux3Resource({ admission: resourceAdmission, request: productionResourceRequest(calibration, plan, invocation), invocationId: invocation.invocationId,
            resourceEnvironment: { SCENEWORKS_FLUX3_KLEIN_ROOT: pkg.root, SCENEWORKS_FLUX3_KLEIN_ORIGINAL_ROOT: dense.root, SCENEWORKS_FLUX3_KLEIN_REPOSITORY: dense.repository, SCENEWORKS_FLUX3_KLEIN_REVISION: dense.resolvedRevision, SCENEWORKS_FLUX3_RESOURCE_INPUT_ROOT: control.resourceInputRoot } });
          await milestone({ invocationId: invocation.invocationId, result: 'PASSED', spent: true, captureId: result.captureId, wallSeconds: (Date.now() - start) / 1000 });
        }
      }
    },
    finalize: async () => {
      const precedingPhase = trace.phase;
      phase('ledger_finalize');
      const receipt = await auditLedger({ approval, backend: 'candle', ledgerRoot: base.ledgerRoot });
      await writeFile(path.join(path.dirname(outputRoot), 'final-ledger-receipt.json'), JSON.stringify(receipt), { flag: 'wx' });
      await milestone({ kind: 'final-ledger', sequence: receipt.state.sequence, generationCount: receipt.state.generationCount, headSha256: receipt.state.headSha256 });
      phase(precedingPhase); // Successful cleanup must not mislabel an earlier execution failure.
    },
  });
  if (result.status === 'DIAGNOSTIC_ONLY') await milestone({ kind: 'admission-only', ...result, ...trace.metrics });
  return result;
}
