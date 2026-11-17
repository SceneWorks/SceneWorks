// Public failure metadata only. Never copy free-form errors, paths, argv or model data.
import { open } from 'node:fs/promises';

export function safeErrorClass(value) {
  return ['Error', 'AssertionError', 'TimeoutError', 'RuntimeError', 'CalledProcessError',
    'FileNotFoundError', 'PermissionError', 'OSError', 'ValueError', 'KeyboardInterrupt', 'SystemExit']
    .includes(value) ? value : value ? 'unknown' : null;
}

const POST_RENDER_CHECKS = new Set([
  'selected source changed', 'member map cardinality changed', 'member map missing or duplicated',
  'member escapes provisioned cache', 'duplicate physical member', 'selected member size changed',
  'selected member bytes changed', 'pinned runtime library changed',
  'CUDA driver not loaded by worker', 'duplicate loaded runtime libraries',
  'censused runtime library changed', 'qualified loaded runtime libraries changed',
  'loaded runtime libraries changed between renders', 'device/runtime changed after render',
]);

export function safePostRenderCheck(value) {
  const firstLine = typeof value === 'string' ? value.split('\n', 1)[0] : '';
  return POST_RENDER_CHECKS.has(firstLine) ? firstLine : 'unknown';
}

export async function readDiagnosticJson(file) {
  let handle;
  try {
    handle = await open(file, 'r');
    const limit = 1024 * 1024;
    const info = await handle.stat();
    if (!info.isFile() || info.size > limit) return null;
    const buffer = Buffer.alloc(limit + 1);
    let length = 0;
    while (length < buffer.length) {
      const { bytesRead } = await handle.read(buffer, length, buffer.length - length, null);
      if (bytesRead === 0) break;
      length += bytesRead;
    }
    if (length > limit) return null;
    const value = JSON.parse(buffer.subarray(0, length).toString('utf8'));
    return value && typeof value === 'object' && !Array.isArray(value) ? value : null;
  } catch {
    return null;
  } finally {
    await handle?.close().catch(() => {});
  }
}

export function failureDiagnostic(evidence, resource) {
  const stages = ['start', 'api_start', 'worker_start', 'project_create', 'exact_action', 'fresh_action',
    'queued_payload', 'selected_source', 'render', 'completed_receipt', 'completed_identity',
    'post_render_environment', 'compare_fresh'];
  const report = { version: 1, kind: 'sdxl-exact-replay-failure', accepted: false, bodyAccepted: false,
    proof: evidence ? { status: 'available', stage: stages.includes(evidence.failureStage) ? evidence.failureStage : 'unknown',
      errorClass: safeErrorClass(evidence.failureErrorClass) } : { status: 'unavailable' },
    resource: { status: 'unavailable' } };
  if (report.proof.stage === 'post_render_environment') {
    report.proof.check = safePostRenderCheck(evidence.failureCheck);
  }
  if (resource?.schema_version !== 1 || resource.kind !== 'cuda-execution-resource-lease-v1'
      || resource.actor !== 'sdxl-exact-replay' || typeof resource.resource_completed !== 'boolean') return report;
  const error = typeof resource.error === 'string' ? resource.error : '';
  const releaseFailed = error.startsWith('resource release failed: ');
  const errorType = (releaseFailed ? error.slice('resource release failed: '.length) : error).split(':', 1)[0];
  const wait = resource.lock_wait_seconds;
  report.resource = {
    status: 'available',
    // Last recorded boundary, not an inferred cause of failure.
    stage: releaseFailed ? 'resource-release' : resource.resource_completed ? 'resource-completed'
      : resource.child ? 'child-result-recorded' : Object.hasOwn(resource, 'lock_wait_seconds') ? 'lock-acquired'
        : resource.binding ? 'binding-recorded' : 'initialized',
    errorClass: safeErrorClass(errorType),
    lockWaitSeconds: typeof wait === 'number' && Number.isFinite(wait) && wait >= 0 && wait <= 86400
      ? Math.round(wait * 1000) / 1000 : null,
    childTimeout: typeof resource.child?.timeout === 'boolean' ? resource.child.timeout : null,
    resource_completed: resource.resource_completed,
  };
  return report;
}
