// Called synchronously at the native inner Generator.generate boundary, never by an outer
// launcher. This is reservation accounting, not measurement/quality evidence or permission.
import { readFile } from 'node:fs/promises';
import { pathToFileURL } from 'node:url';

export async function reserveResourceInvocation(input, dependencies = {}) {
  // Injection is an explicit CPU test seam, never accepted by the production CLI/input schema.
  const verifyHardware = dependencies.verifyHardware ?? (async (admission, context) => {
    const { verifyResourceProducerHardware } = await import('./flux3-image-klein-terminal-session.mjs');
    return verifyResourceProducerHardware(admission, context);
  });
  const { approvalFile, admissionFile, adapterPid, sourceSha, binarySha256, ledgerRoot, ledgerId, approvedPlanSha256, invocationId, backend, caseId, kind, requestSha256 } = input;
  if (!['resource_anchor', 'resource_supplemental'].includes(kind)) throw new Error('BLOCKED: native resource boundary requires a resource permit');
  const approval = JSON.parse(await readFile(approvalFile, 'utf8'));
  const admission = JSON.parse(await readFile(admissionFile, 'utf8'));
  const { reserveInvocation, planHash } = await import('./flux3-image-klein-terminal-budget.mjs');
  if (planHash(admission.approvedPlan) !== planHash(approval) || admission.backend !== backend || admission.ledgerId !== ledgerId
    || admission.ledgerRoot !== ledgerRoot || admission.approvedPlanSha256 !== approvedPlanSha256) throw new Error('BLOCKED: resource admission/immutable plan/ledger mismatch');
  const context = { adapterPid, caseId, invocationId, requestSha256, approvedPlanSha256, sourceSha, binarySha256 };
  const hardwareObservation = await verifyHardware(admission, context);
  for (const key of Object.keys(context)) if (hardwareObservation?.[key] !== context[key]) throw new Error(`BLOCKED: observed hardware ${key} binding mismatch`);
  const event = await reserveInvocation({ approval, ledgerRoot, ledgerId, approvedPlanSha256, invocationId, backend, caseId, kind, requestSha256, inputSpecAddendum: admission.inputSpecAddendum });
  return { ...event, hardwareObservation };
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  try {
    let text = '';
    for await (const chunk of process.stdin) text += chunk;
    const event = await reserveResourceInvocation(JSON.parse(text));
    process.stdout.write(JSON.stringify(event) + '\n');
  } catch (error) {
    process.stderr.write(`BLOCKED: FLUX3 inner-generate reservation: ${error.message}\n`);
    process.exitCode = 1;
  }
}
