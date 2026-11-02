// sc-26691 single-SHA pair consolidation: builds the master plan (backendReadbacks) the
// documented `verify PLAN` path consumes, then runs verifyCampaign's steps in order, capturing the
// re-measured controls and rendered-content values. validateDelivery is run last and its result
// recorded (delivery records exist only after merge).
import path from 'node:path';
import { readFile, writeFile } from 'node:fs/promises';
import { pathToFileURL } from 'node:url';
// usage: node verify-pair.mjs SCENEWORKS_CHECKOUT EVIDENCE_ROOT OUT_DIR
// EVIDENCE_ROOT holds `gh run download` of both lane runs under mlx/ and cuda/.
const [SW, V, OUT] = process.argv.slice(2);
const T = await import(pathToFileURL(path.join(SW, 'scripts/h3-finetune-terminal.mjs')));
const R = await import(pathToFileURL(path.join(SW, 'scripts/h3-finetune-terminal-runtime.mjs')));
const json = async f => JSON.parse(await readFile(f, 'utf8'));
const lanes = {
  mlx: { run: 41379525090, root: `${V}/mlx/h3-terminal-mlx-reduced-41379525090-1`, anchors: `${V}/mlx/h3-anchors-mlx-1e23ef16e70327087a52504be374a1e8164262e8-a43724b82fbae35c9227cef4bcdabba6dd1da12c-0bada8f7287f414c30f77eb1d29ef4a544eb8ffc-attempt-1` },
  cuda: { run: 41379549283, root: `${V}/cuda/h3-terminal-cuda-reduced-41379549283-1`, anchors: `${V}/cuda/h3-anchors-cuda-1e23ef16e70327087a52504be374a1e8164262e8-a43724b82fbae35c9227cef4bcdabba6dd1da12c-0bada8f7287f414c30f77eb1d29ef4a544eb8ffc-attempt-1` },
};
const readbacks = {};
for (const b of ['mlx', 'cuda']) readbacks[b] = await json(`${lanes[b].root}/backend-readback-${b}.json`);
// Master plan: the selected backend's plan, with host paths of hash-bound retained files remapped
// to the downloaded artifacts (content is re-hashed against the plan's recorded sha256).
const master = structuredClone(readbacks.mlx.plan);
const remap = p => {
  const base = path.basename(p.replaceAll('\\', '/')), dir = path.basename(path.dirname(p.replaceAll('\\', '/')));
  if (dir === 'native-witnesses-mlx') return `${lanes.mlx.root}/native-witnesses-mlx/${base}`;
  if (dir === 'native-witnesses-cuda') return `${lanes.cuda.root}/native-witnesses-cuda/${base}`;
  if (dir === 'inline-anchor-bundle-mlx' || dir.startsWith('peer-inline-anchors-mlx')) return `${lanes.mlx.anchors}/${base}`;
  if (dir === 'inline-anchor-bundle-cuda' || dir.startsWith('peer-inline-anchors-cuda')) return `${lanes.cuda.anchors}/${base}`;
  throw new Error(`unmapped retained path ${p}`);
};
for (const proof of master.proofIndex) { proof.receiptInput.path = remap(proof.receiptInput.path); for (const l of proof.logInputs) l.path = remap(l.path); }
for (const a of (Array.isArray(master.matrixAnchors) ? master.matrixAnchors : Object.values(master.matrixAnchors).flat())) a.receiptInput.path = remap(a.receiptInput.path);
master.output = lanes.mlx.root;
master.backendReadbacks = Object.fromEntries(['mlx', 'cuda'].map(b => [b, { path: `${lanes[b].root}/backend-readback-${b}.json`, caseRoot: lanes[b].root }]));
await writeFile(path.join(OUT, 'master-plan.json'), JSON.stringify(master, null, 2) + '\n');

const result = { kind: 'sc26691-single-sha-pair-verification', sceneworks: master.sceneworks, inference: master.inference, nativeEvidence: master.nativeEvidence?.sha, matrixDecision: master.matrixDecision, steps: [], backends: {} };
const step = async (name, fn) => { try { const v = await fn(); result.steps.push({ name, ok: true }); return v; } catch (e) { result.steps.push({ name, ok: false, error: e.message }); throw e; } };
T.validatePlan(master);
await step('verifyMatrixAdmission', () => T.verifyMatrixAdmission(master));
const records = [];
for (const b of ['mlx', 'cuda']) {
  const spec = master.backendReadbacks[b];
  const r = await step(`validateBackendReadback:${b}`, async () => { const r = T.validateBackendReadback(master, b, await json(spec.path)); (await import('node:assert/strict')).deepEqual(r.plan.matrixDecision, master.matrixDecision); return r; });
  const lane = { run: lanes[b].run, planSha256: r.planSha256, verificationRunId: r.verificationRunId, cases: [], controls: null, content: [] };
  for (const row of r.cases) {
    const c = r.plan.cases.find(c => c.id === row.id), dir = path.join(spec.caseRoot, row.id);
    await step(`verifyPortableCase:${row.id}`, () => T.verifyPortableCase(r.plan, c, dir, row));
    records.push({ c, row, dir });
    const exec = await json(path.join(dir, 'execution.json'));
    lane.cases.push({ id: row.id, kind: c.kind, status: exec.status, runId: exec.runId, runAttempt: exec.runAttempt, sceneworks: exec.sceneworks ?? null, steps: c.runtime?.steps ?? c.backendSteps?.[b] ?? null });
    if (c.kind === 'full-av') {
      // Independent re-measurement of the rendered-content gate from the media + bound keyframes.
      const bundle = await json(path.join(dir, 'source-job-output.json'));
      for (const media of bundle.bindings.map(x => x.media)) {
        const job = bundle.bindings.find(x => x.media === media).job;
        const keyframes = R.renderedKeyframes(c, job, bundle.conditioningBindings, dir);
        const measured = await T.verifyRenderedContent(path.join(dir, media.av), keyframes);
        lane.content.push({ caseId: row.id, av: media.av, recordedAtRun: media.content ?? null, remeasured: measured });
      }
    }
  }
  lane.controls = await step(`verifyReducedControls:${b}`, () => R.verifyReducedControls(r.plan, b, { caseRoot: spec.caseRoot }));
  lane.laneReadbackControls = r.crossCaseControls ?? null;
  result.backends[b] = lane;
}
const coverage = await step('reducedCoverageReport', async () => T.reducedCoverageReport(master, records, await T.loadRetainedProofIndex(master)));
result.coverage = coverage;
try { T.validateDelivery(master); result.delivery = { recorded: true }; }
catch (e) { result.delivery = { recorded: false, error: e.message, meaning: 'validateDelivery needs post-merge records (final CI, inference/SceneWorks main merge, Shortcut readbacks); evidence verification above is complete without it.' }; }
result.status = result.delivery.recorded ? (coverage.unproven.length ? 'reduced-evidence-reverified-with-explicit-unproven' : 'reduced-evidence-reverified') : 'evidence-reverified-delivery-pending';
const slimControl = c => { const { media, locatorNormalization, ...rest } = c; return rest; };
for (const b of ['mlx', 'cuda']) {
  const L = result.backends[b];
  const out = {
    kind: 'sc26691-backend-verification', backend: b, run: L.run, planSha256: L.planSha256, laneReadbackRun: L.verificationRunId, sceneworks: result.sceneworks, inference: result.inference,
    cases: L.cases,
    // interventionDelta: re-measured here from the retained media; laneReadbackInterventionDelta: the
    // lane's own on-host readback (host ffmpeg differs, so the last digits differ).
    crossCaseControls: L.controls.map(c => ({ ...slimControl(c), laneReadbackInterventionDelta: L.laneReadbackControls?.find(x => x.caseId === c.caseId)?.interventionDelta ?? null })),
    renderedContent: L.content.map(x => ({ caseId: x.caseId, av: x.av, frames: x.remeasured.frames, spatialStd: x.remeasured.spatialStd, keyframes: x.remeasured.keyframes, matchesRunReceipt: JSON.stringify(x.recordedAtRun) === JSON.stringify(x.remeasured) })),
  };
  await writeFile(path.join(OUT, `verify-campaign-${b}.json`), JSON.stringify(out, null, 2) + '\n');
}
const { matrixDecision: _omit, ...coverageRest } = result.coverage;
const pair = {
  kind: result.kind, status: result.status, sceneworks: result.sceneworks, inference: result.inference, nativeEvidence: result.nativeEvidence,
  matrixDecision: result.matrixDecision, priorMatrixDecision: master.priorMatrixDecision ?? null,
  runs: Object.fromEntries(['mlx', 'cuda'].map(b => [b, { run: result.backends[b].run, cases: result.backends[b].cases.length, passed: result.backends[b].cases.filter(c => c.status === 'passed').length, planSha256: result.backends[b].planSha256 }])),
  steps: result.steps, coverage: coverageRest, delivery: result.delivery,
};
await writeFile(path.join(OUT, 'verify-campaign-pair.json'), JSON.stringify(pair, null, 2) + '\n');
console.log(JSON.stringify({ status: result.status, steps: result.steps.filter(s => !s.ok), delivery: result.delivery }, null, 2));
