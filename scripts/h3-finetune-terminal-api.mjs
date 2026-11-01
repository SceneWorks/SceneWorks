#!/usr/bin/env node
// Drive existing app routes, never an alternate importer/provider. No service startup/download.
import assert from 'node:assert/strict';
import {isDeepStrictEqual} from 'node:util';
import { readFile, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
export function pointer(value, p) {
  assert.ok(p === '' || p.startsWith('/'), 'JSON pointer required');
  return p === '' ? value : p.slice(1).split('/').reduce((v, key) => {
    key = key.replaceAll('~1', '/').replaceAll('~0', '~');
    assert.ok(v != null && Object.hasOwn(v, key), `missing JSON pointer ${p}`);
    return v[key];
  }, value);
}
export function expand(value, captures) {
  if (value && typeof value === 'object' && !Array.isArray(value) && Object.keys(value).length === 1 && Object.hasOwn(value, '$concat')) {
    assert.ok(Array.isArray(value.$concat));
    return value.$concat.map(v => { const part = expand(v, captures); assert.ok(typeof part === 'string'); return part; }).join('');
  }
  if (value && typeof value === 'object' && !Array.isArray(value) && Object.keys(value).length === 1 && Object.hasOwn(value, '$capture')) {
    const [name, ...keys] = value.$capture.split('/');
    assert.ok(Object.hasOwn(captures, name), `unknown capture ${name}`);
    return pointer(captures[name], keys.length ? '/' + keys.join('/') : '');
  }
  if (Array.isArray(value)) return value.map(v => expand(v, captures));
  if (value && typeof value === 'object') return Object.fromEntries(Object.entries(value).map(([k, v]) => [k, expand(v, captures)]));
  return value;
}
export function assertRecordedAssertion({actual,expected,operator='equals'}){
  if(operator==='contains'){
    assert.ok(Array.isArray(actual)&&expected&&Object.keys(expected).length>0,'typed catalog membership required');assert.ok(actual.some(row=>Object.entries(expected).every(([key,value])=>isDeepStrictEqual(row?.[key],value))),'selected compatible catalog entry absent');
  }else if(operator==='includesText'){
    assert.ok(typeof actual==='string'&&typeof expected==='string'&&expected.length>0&&actual.includes(expected),'actual refusal did not contain the expected specific reason');
  }else{assert.equal(operator,'equals','unknown assertion operator');assert.deepEqual(actual,expected);}
}
/**
 * The API commits a worker's terminal status BEFORE it persists the worker-reported
 * `result.assetWrites` into project assets and rewrites the result as `assets` / `assetIds`
 * (the durable two-phase handoff in `apps/rust-api/src/jobs.rs` `persist_reported_assets`). A
 * poller can therefore read `completed` with the raw facts still in place — campaign 40874652324's
 * hybrid case captured revision 82 (raw `assetWrites`) while revision 83, one write later, carried
 * `assetIds`. Mirrors `JobSnapshot::is_settled` in `apps/rust-api/src/film_harness.rs`: persisted
 * `assets`/`assetIds` are the positive signal; a completed result without non-empty `assetWrites`
 * (an import job, for one) has nothing to persist.
 */
export function isSettledJobSnapshot(job) {
  if (job?.status !== 'completed') return true;
  const result = job.result ?? {};
  return Array.isArray(result.assets) || Array.isArray(result.assetIds) || !Array.isArray(result.assetWrites) || result.assetWrites.length === 0;
}
/** Same bound the product's own film harness gives the terminal-to-persisted window. */
export const ASSET_SETTLE_GRACE_MS = 30000;
/** A mutating request running longer than this is logged as H3_SLOW_WRITE evidence. */
export const SLOW_WRITE_WARN_MS = 30000;
/** Upper bound for ONE non-retried mutating request (run 41299613206: an upload outlasted 30s). */
export const MUTATING_REQUEST_CAP_MS = 15 * 60 * 1000;
// `baseUrl` (a string or a function of no arguments) is the live owned service origin. A scenario
// planned against an OS-assigned port names `http://127.0.0.1:0` and requires it; the function form
// follows a restart that bound a new port.
export async function runScenario(s, { fetchImpl = fetch, now = Date.now, sleep = ms => new Promise(r => setTimeout(r, ms)), onRecord = async () => {}, beforeStep = async () => {}, deadlineMs = Infinity, uploadFile, baseUrl } = {}) {
  const loopback = url => { assert.ok(['127.0.0.1', '[::1]', 'localhost'].includes(url.hostname), 'task-owned loopback API required'); return url; };
  const planned = loopback(new URL(s.baseUrl));
  assert.ok(baseUrl !== undefined || planned.port !== '0', 'a scenario planned against an OS-assigned port needs the owned service\'s bound base URL');
  const currentBase = () => loopback(new URL(baseUrl === undefined ? s.baseUrl : typeof baseUrl === 'function' ? baseUrl() : baseUrl));
  currentBase();
  assert.ok(s.timeoutSeconds > 0 && s.timeoutSeconds <= 7200);
  assert.ok(s.caseId && Array.isArray(s.steps) && s.steps.length > 0);
  assert.ok(deadlineMs===Infinity||Number.isFinite(deadlineMs)&&deadlineMs>now(),'explicit case deadline expired');
  const deadline = Math.min(now() + s.timeoutSeconds * 1000,deadlineMs), captures = {}, transcript = [];
  let activeJob;
  const record = async r => { transcript.push(r); await onRecord(r); };
  async function request(method, route, body, expectedStatus, upload) {
    assert.ok(route.startsWith('/api/v1/') && !route.includes('..'), 'existing app API route required');
    const base = currentBase(), url = new URL(route, base); assert.equal(url.origin, base.origin);
    assert.ok(now() < deadline, 'scenario timeout');
    let wireBody=body===undefined?undefined:JSON.stringify(body),headers={'content-type':'application/json'},recordedBody=body;
    if(upload){assert.equal(method,'POST');assert.match(route,/^\/api\/v1\/projects\/[^/]+\/assets$/,'only the existing asset upload endpoint accepts files');assert.equal(typeof uploadFile,'function','source-bound upload adapter required');assert.equal(body,undefined,'upload fields are explicit, not arbitrary request bodies');const blob=await uploadFile(upload);assert.ok(blob instanceof Blob&&blob.size>0);const form=new FormData();form.set('file',blob,upload.fileName);wireBody=form;headers={};recordedBody={multipartFile:{path:upload.path,sha256:upload.sha256,mimeType:upload.mimeType,fileName:upload.fileName}};}
    // A read is safe to repeat: under heavy disk I/O (a checkpoint-sized staging copy on the same
    // volume as the jobs database) one status read can outlast 30s while the service is healthy.
    // Retry a timed-out GET until the scenario deadline.
    // A mutating request is NOT idempotent, so it is never retried, and a fixed 30s abort is the
    // wrong bound for it too: CUDA campaign run 41299613206 failed case
    // cuda.reduced.real-hybrid-first-last when the 5th multipart conditioning upload exceeded 30s
    // while the service was healthy under heavy disk I/O on D: (the previous upload took 5.1s).
    // Its single attempt is bounded by the remaining scenario deadline, capped at
    // MUTATING_REQUEST_CAP_MS so a hung service is still caught well before the hard case deadline.
    const isRead = method === 'GET';
    let response;
    for (;;) {
      const started = now(), remaining = Math.max(1, deadline - started);
      const bound = isRead ? Math.min(30000, remaining) : Math.min(MUTATING_REQUEST_CAP_MS, remaining);
      try { response = await fetchImpl(url, { method, headers, body:wireBody, signal: AbortSignal.timeout(bound) }); break; }
      catch (error) { if (!isRead || error?.name !== 'TimeoutError' || now() >= deadline) throw error; console.warn(`H3_READ_RETRY: ${method} ${route} timed out; retrying before the scenario deadline`); }
      finally { if (!isRead && now() - started > SLOW_WRITE_WARN_MS) console.warn(`H3_SLOW_WRITE: ${method} ${route} ${now() - started}`); }
    }
    const text = await response.text();
    const result = text ? JSON.parse(text) : null;
    await record({ method, route, request: recordedBody, status: response.status, result });
    assert.equal(response.status, expectedStatus, `${method} ${route}: ${text.slice(0, 400)}`);
    return result;
  }
  try {
    for (const step of s.steps) {
      await beforeStep({ step, captures });
      assert.ok(!Object.hasOwn(captures, step.id), 'duplicate step id');
      let result;
      if (step.waitJob) {
        const jobId = expand(step.waitJob, captures); assert.ok(typeof jobId === 'string' && jobId.length > 0);
        activeJob = jobId;
        let unsettledSince;
        for (;;) {
          result = await request('GET', `/api/v1/jobs/${encodeURIComponent(jobId)}`, undefined, 200);
          if (['completed', 'failed', 'cancelled', 'canceled', 'interrupted'].includes(result.status)) {
            if (isSettledJobSnapshot(result)) break;
            // Terminal but not yet settled: keep reading the same job until the API's asset
            // persistence rewrites the result, bounded by its own grace (never the case deadline).
            unsettledSince ??= now();
            assert.ok(now() - unsettledSince < ASSET_SETTLE_GRACE_MS, `job ${jobId} completed but its reported assetWrites were never persisted into result.assets/assetIds within ${ASSET_SETTLE_GRACE_MS}ms`);
          }
          await sleep(Math.min(1000, Math.max(0, deadline - now())));
        }
        assert.equal(result.status, step.expectedJobStatus ?? 'completed', 'job did not complete as expected');
        activeJob = undefined;
      } else {
        const requestedRoute = expand(step.path, captures);
        result = await request(step.method, requestedRoute, expand(step.body, captures), step.expectedStatus, step.upload?expand(step.upload,captures):undefined);
        // Submission identity is durable before waiting, including replay/retry failure recovery.
        if (step.method === 'POST' && result?.id && (['/api/v1/models/import','/api/v1/loras/import', '/api/v1/video/jobs'].includes(requestedRoute) || /^\/api\/v1\/jobs\/[^/]+\/(retry|duplicate)$/.test(requestedRoute))) activeJob = result.id;
      }
      captures[step.id] = result;
      for (const check of step.assert ?? []) {
        const actual = pointer(result, check.pointer),operator=Object.hasOwn(check,'contains')?'contains':Object.hasOwn(check,'includesText')?'includesText':'equals',expected=expand(check.contains??check.includesText??check.equals,captures);
        assertRecordedAssertion({actual,expected,operator});
        await record({ assertion: `${step.id}${check.pointer}`, actual, expected, operator });
      }
    }
    assert.ok(transcript.some(r => r.assertion), 'no observable assertions executed');
    return { caseId: s.caseId, captures, transcript };
  } catch (error) {
    // Cooperative product cancellation, never process-kill a native GPU buffer.
    if (activeJob) {
      try {
        const r = await fetchImpl(new URL(`/api/v1/jobs/${encodeURIComponent(activeJob)}/cancel`, currentBase()), { method: 'POST', signal: AbortSignal.timeout(10000) });
        await record({ cancellation: activeJob, status: r.status });
      } catch (cancelError) { await record({ cancellation: activeJob, error: cancelError.message }); }
    }
    throw error;
  }
}
if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  try {
    assert.equal(process.env.GITHUB_ACTIONS, 'true', 'actual app execution is explicit Actions only');
    const [scenarioPath, out] = process.argv.slice(2);
    const scenario = JSON.parse(await readFile(scenarioPath, 'utf8'));
    assert.equal(scenario.caseId, process.env.H3_CASE_ID);
    const records = [];
    const result = await runScenario(scenario, { onRecord: async r => { records.push(r); await writeFile(out, JSON.stringify({ caseId: scenario.caseId, transcript: records }, null, 2)); } });
    await writeFile(out, JSON.stringify(result, null, 2) + '\n');
    console.log(`H3_API_ASSERTIONS_EXECUTED=${result.transcript.filter(r => r.assertion).length}`);
  } catch (error) { console.error(error.message); process.exitCode = 1; }
}
