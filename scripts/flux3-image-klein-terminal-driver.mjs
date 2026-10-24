// Terminal installed-product proof. CLI defaults to pure planning; no credentials or downloads.
import assert from 'node:assert/strict';
import { createHash, randomBytes } from 'node:crypto';
import { copyFile, mkdir, open, readFile, readdir, realpath, stat, statfs, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { inflateSync } from 'node:zlib';
import { createReadStream } from 'node:fs';
import { loadProductTargets, terminalPlan, terminalCatalog, targetRegistry, imageRequest, requestedDiskBudget, digestPattern, ROOT } from './flux3-image-klein-terminal-plan.mjs';
import { assertAccepted, assertSavedAsset, assertServedPng, assertClaimedWithin, assertDownloadWorker } from './flux3-image-klein-app-lifecycle.mjs';
import { contained, fileHash, validateAdmission, launchTerminalSession, ProvisioningBlocked } from './flux3-image-klein-terminal-session.mjs';
import { validateApprovedPlan, planHash, initializeLedger, auditLedger, boundInputSpecSha256, admittedHost, inHostLane, ledgerLane } from './flux3-image-klein-terminal-budget.mjs';
import { verifyTerminalMapping, artifactStagingPlan, revalidateMappedArtifacts, invalidateArtifactVerification, ArtifactReadinessBlocked } from './flux3-image-klein-terminal-artifacts.mjs';
import { bindTrainingExpectation, verifyNativeContinuation } from './flux3-image-klein-terminal-training-proof.mjs';
export const sha256 = bytes => createHash('sha256').update(bytes).digest('hex');
export const objectHash = object => sha256(JSON.stringify(object));
export function pngCrc32(bytes) { let crc = 0xffffffff; for (const byte of bytes) { crc ^= byte; for (let bit = 0; bit < 8; bit++) crc = (crc >>> 1) ^ ((crc & 1) ? 0xedb88320 : 0); } return (crc ^ 0xffffffff) >>> 0; }
export function assertFinishedPng(bytes, width, height, { allowBlank = false, forReference = false } = {}) {
  assertServedPng(bytes, bytes, width, height);
  let offset = 8, ended = false, ihdr = false, channels; const compressed = [];
  while (offset < bytes.length) {
    assert(offset + 12 <= bytes.length, 'truncated PNG chunk'); const length = bytes.readUInt32BE(offset), end = offset + 12 + length; assert(end <= bytes.length, 'truncated PNG data');
    const type = bytes.toString('ascii', offset + 4, offset + 8), data = bytes.subarray(offset + 8, offset + 8 + length);
    assert.equal(pngCrc32(bytes.subarray(offset + 4, offset + 8 + length)), bytes.readUInt32BE(offset + 8 + length), 'PNG CRC mismatch');
    if (type === 'IHDR') { assert(!ihdr && offset === 8 && length === 13); ihdr = true; assert.equal(data[8], 8, 'ordinary generated PNG must be 8-bit'); channels = { 2: 3, 6: 4 }[data[9]]; assert(channels && data[10] === 0 && data[11] === 0 && data[12] === 0, 'ordinary noninterlaced RGB/RGBA PNG required'); }
    if (forReference) assert(!['eXIf', 'tRNS'].includes(type), 'reference proof fixtures must use explicit RGB/RGBA PNG without EXIF/tRNS ambiguity; provision an oriented explicit-mode reference');
    if (type === 'IDAT') compressed.push(data);
    offset = end;
    if (type === 'IEND') { assert.equal(length, 0); assert.equal(offset, bytes.length, 'PNG trailing data'); ended = true; break; }
  }
  assert(ended && ihdr && compressed.length, 'finished PNG needs IHDR/IDAT/IEND');
  const row = width * channels + 1, decoded = inflateSync(Buffer.concat(compressed), { maxOutputLength: row * height }); assert.equal(decoded.length, row * height);
  const minimum = [255, 255, 255], maximum = [0, 0, 0], dimensions = Buffer.alloc(8); dimensions.writeUInt32LE(width); dimensions.writeUInt32LE(height, 4);
  // Exact worker OwnedReferenceBinding carrier digest, not the encoded PNG file digest.
  const carrier = createHash('sha256').update(Buffer.from([channels === 4 ? 1 : 0])).update(dimensions);
  let visiblePixels = 0, previous = Buffer.alloc(width * channels), current = Buffer.alloc(width * channels);
  const paeth = (a, b, c) => { const p = a + b - c, pa = Math.abs(p - a), pb = Math.abs(p - b), pc = Math.abs(p - c); return pa <= pb && pa <= pc ? a : pb <= pc ? b : c; };
  for (let y = 0; y < height; y++) {
    const filter = decoded[y * row]; assert(filter <= 4, 'invalid PNG scanline filter');
    for (let x = 0; x < current.length; x++) { const left = x >= channels ? current[x - channels] : 0, up = previous[x], upperLeft = x >= channels ? previous[x - channels] : 0; const predictor = filter === 0 ? 0 : filter === 1 ? left : filter === 2 ? up : filter === 3 ? Math.floor((left + up) / 2) : paeth(left, up, upperLeft); current[x] = (decoded[y * row + 1 + x] + predictor) & 255; }
    for (let x = 0; x < current.length; x += channels) if (channels === 3 || current[x + 3] > 0) { visiblePixels++; for (let c = 0; c < 3; c++) { minimum[c] = Math.min(minimum[c], current[x + c]); maximum[c] = Math.max(maximum[c], current[x + c]); } }
    carrier.update(current); [previous, current] = [current, previous];
  }
  const nonblank = visiblePixels > 0 && minimum.some((value, c) => value < maximum[c]);
  assert(allowBlank || nonblank, 'blank/uniform/transparent image is not finished generation proof');
  return { nonblank, visiblePixels, minimum, maximum, decodedContentSha256: carrier.digest('hex'), orientedSize: [width, height], sourceMode: channels === 4 ? 'rgba' : 'rgb', semanticAcceptance: 'PENDING' };
}
export class Blocked extends Error {}
export async function safetensorsHeader(file, { withOffset = false } = {}) {
  const handle = await open(file, 'r');
  try { const size = (await handle.stat()).size, prefix = Buffer.alloc(8); assert.equal((await handle.read(prefix, 0, 8, 0)).bytesRead, 8); const length = Number(prefix.readBigUInt64LE()); assert(length > 0 && length <= 16 * 1024 * 1024 && length + 8 <= size, 'bounded safetensors header required'); const bytes = Buffer.alloc(length); assert.equal((await handle.read(bytes, 0, length, 8)).bytesRead, length); const header = JSON.parse(bytes); return withOffset ? { header, dataOffset: length + 8, size } : header; } finally { await handle.close(); }
}
export async function assertCheckpoint(file, cell, step) {
  const { header, dataOffset, size } = await safetensorsHeader(file, { withOffset: true }), meta = header.__metadata__;
  assert.equal(meta?.format, cell.backend === 'candle' ? 'flux3-native-training-resume-v1' : 'flux3-mlx-training-resume-v1');
  const identity = JSON.parse(meta.identity), cursor = JSON.parse(meta.cursor);
  assert.equal(identity.base_repo, cell.repository); assert.equal(identity.base_revision, cell.resolvedRevision);
  assert(Object.values(identity.artifacts ?? {}).length && Object.values(identity.artifacts).every(h => digestPattern.test(h)));
  assert(digestPattern.test(identity.dataset ?? '') && digestPattern.test(identity.preprocessing ?? ''));
  assert(cursor.micro_step > 0 && cursor.update_step > 0 && cursor.rng_state?.length > 0 && cursor.update_step === step, 'checkpoint needs actual step/optimizer/RNG state');
  const factors = Object.keys(header).filter(key => key.startsWith('factor/')); assert(factors.length, 'checkpoint must contain learned factors');
  const optimizer = {};
  if (cell.backend === 'candle') {
    assert.equal(meta.optimizer, 'adamw', 'native Candle optimizer must match the proved AdamW profile');
    assert(typeof meta.optimizer_config === 'string' && meta.optimizer_config.startsWith('adamw;lr='), 'actual native optimizer configuration required');
    assert(/^[1-9][0-9]*$/.test(meta['optim.step'] ?? '') && Number.isSafeInteger(Number(meta['optim.step'])) && Number(meta['optim.step']) === cursor.update_step, 'actual native optimizer step differs from saved update cursor');
    const order = JSON.parse(meta.optimizer_order); assert(Array.isArray(order) && order.length === factors.length && new Set(order).size === order.length);
    order.forEach((name, index) => { assert(header[`factor/${name}`] && header[`optim.${index}.m`] && header[`optim.${index}.v`], 'Candle AdamW checkpoint needs actual ordered m/v tensors'); optimizer[name] = { first: `optim.${index}.m`, second: `optim.${index}.v`, step: Number(meta['optim.step']) }; });
  } else {
    assert(/^adamw;lr=[0-9a-f]{8};wd=[0-9a-f]{8};/.test(meta.optimizer_config ?? ''));
    const steps = JSON.parse(meta.optimizer_steps); assert(steps && typeof steps === 'object' && !Array.isArray(steps));
    // Native MLX creates moments only when an active factor receives gradients. Captioned
    // training intentionally leaves image_cond factors (and their absent moments) alone.
    for (const [key, count] of Object.entries(steps)) { assert(header[`factor/${key}`] && header[`moment_first/${key}`] && header[`moment_second/${key}`] && Number.isInteger(count) && count > 0 && count <= cursor.update_step, 'MLX checkpoint needs actual declared per-factor moment tensors/steps'); optimizer[key] = { first: `moment_first/${key}`, second: `moment_second/${key}`, step: count }; }
    assert(Object.keys(optimizer).length, 'native MLX checkpoint has no optimizer state');
  }
  const declared = [...factors, ...Object.values(optimizer).flatMap(o => [o.first, o.second]), ...JSON.parse(meta.pending_keys).map(key => `pending/${key}`)].sort();
  assert.deepEqual(Object.keys(header).filter(key => key !== '__metadata__').sort(), declared, 'undeclared/missing native optimizer or factor tensor');
  const manifest = JSON.parse(meta.tensor_manifest); assert.deepEqual(Object.keys(manifest).sort(), Object.keys(header).filter(key => key !== '__metadata__').sort());
  let nextOffset = 0; for (const tensor of Object.keys(manifest).map(key => header[key]).sort((a, b) => a.data_offsets[0] - b.data_offsets[0])) { assert.equal(tensor.data_offsets[0], nextOffset, 'native tensor payload ranges must be contiguous/nonoverlapping'); nextOffset = tensor.data_offsets[1]; } assert.equal(nextOffset, size - dataOffset);
  for (const [name, record] of Object.entries(manifest)) {
    const tensor = header[name]; assert.equal(record.dtype, 'F32'); assert.equal(tensor.dtype, 'F32'); assert.deepEqual(record.shape, tensor.shape); assert(digestPattern.test(record.sha256));
    const [start, end] = tensor.data_offsets; assert(Number.isSafeInteger(start) && Number.isSafeInteger(end) && start >= 0 && end > start && dataOffset + end <= size);
    assert.equal(end - start, tensor.shape.reduce((n, side) => { assert(Number.isSafeInteger(side) && side > 0); return n * side; }, 1) * 4);
    const h = createHash('sha256'); for await (const bytes of createReadStream(file, { start: dataOffset + start, end: dataOffset + end - 1 })) h.update(bytes); assert.equal(h.digest('hex'), record.sha256, 'native checkpoint tensor payload differs from recorded optimizer/factor state');
  }
  return { identity, cursor, format: meta.format, tensorManifest: manifest, optimizer, optimizerConfig: meta.optimizer_config, policy: meta.policy, gainBits: meta.gain_bits, pendingKeys: JSON.parse(meta.pending_keys) };
}
const need = (value, message) => { if (!value) throw new Blocked(message); };
const safeId = id => { assert(/^[a-zA-Z0-9_.-]+$/.test(id), 'invalid API identity'); return encodeURIComponent(id); };
export async function waitJob(session, id, report, { until = j => j.status === 'completed', allowCanceled = false, allowFailed = false, utility = false } = {}) {
  const first = Date.now();
  return session.wait(async () => {
    const job = await session.endpoint.call('GET', `/api/v1/jobs/${safeId(id)}`);
    assert.equal(job.id, id); assertClaimedWithin(job, first, Date.now());
    const event = { id, status: job.status, stage: job.stage, progress: job.progress, workerId: job.workerId, result: job.result };
    if (JSON.stringify(event) !== JSON.stringify(report.events.at(-1))) { assert(report.events.length < 10_000, 'case progress receipt exceeded bounded event budget'); report.events.push(event); }
    if (job.workerId) { if (utility || job.type === 'lora_import') assert(job.workerId.startsWith(`flux3-app-${session.identity.nonce}-utility`), 'job claimed by foreign utility worker'); else assert.equal(job.workerId, session.identity.workerId, 'job claimed by foreign worker'); }
    if (job.status === 'failed' && utility && !allowFailed && /401|403|gated|credential|authoriz/i.test(JSON.stringify(job.error ?? job.message))) throw new Blocked('Selected runner account cannot access the approved gated package (401/403). Accept the gate/refresh its own stored login; no fallback, alternate account transfer or skip-as-pass.');
    assert(![...(allowFailed ? [] : ['failed']), ...(allowCanceled ? [] : ['canceled'])].includes(job.status), `ordinary job ${id} ${job.status}: ${JSON.stringify(job.error ?? job.message ?? '')}`);
    if (until(job)) return job;
    assert(!['completed', 'canceled'].includes(job.status), 'job terminated before required observation'); return false;
  }, report.timeoutSeconds ?? 1800);
}
export async function uploadBinding(session, project, binding) {
  need(binding?.path && binding.root && digestPattern.test(binding.sha256 ?? ''), 'reference requires provisioned path/root/expected hash');
  const file = await contained(binding.path, binding.root); assert.equal(await fileHash(file), binding.sha256);
  const bytes = await readFile(file); assert.equal(sha256(bytes), binding.sha256, 'source bytes changed before upload'); need(bytes.length >= 24 && bytes.subarray(0, 8).toString('hex') === '89504e470d0a1a0a', 'terminal reference fixtures must be provisioned PNGs');
  const width = bytes.readUInt32BE(16), height = bytes.readUInt32BE(20); need(width > 0 && height > 0 && width * height <= 4096 ** 2, 'terminal reference view exceeds the explicit 16-Mpixel CPU proof budget');
  let decoded; try { decoded = assertFinishedPng(bytes, width, height, { allowBlank: true, forReference: true }); } catch (error) { throw new Blocked(`Provision an oriented explicit RGB/RGBA8 PNG reference for deterministic carrier proof: ${error.message}`); }
  const form = new FormData(); form.set('file', new Blob([bytes], { type: 'image/png' }), path.basename(file));
  const response = await session.endpoint.exchange('POST', `/api/v1/projects/${safeId(project.id)}/assets`, form, { multipart: true });
  assert.equal(response.status, 201); need(response.body?.id, 'upload did not return product asset identity');
  return { assetId: response.body.id, sha256: binding.sha256, decodedContentSha256: decoded.decodedContentSha256, orientedSize: decoded.orientedSize, sourceMode: decoded.sourceMode, bytes: bytes.length, originalPath: file };
}
async function verifyInventory(session, inventory) {
  need(inventory?.root && inventory.files?.length && inventory.authority === 'parent-approved-pinned-inventory', 'parent-approved upstream inventory required; self-reported receipts are not trusted pins');
  session.verifiedInventory ??= new Map();
  const key = objectHash(inventory), previous = session.verifiedInventory.get(key), current = [];
  for (const f of inventory.files) {
    assert(digestPattern.test(f.sha256) && Number.isSafeInteger(f.bytes) && f.bytes > 0);
    const file = await contained(path.join(inventory.root, f.path), inventory.containmentRoot ?? inventory.root), info = await stat(file, { bigint: true }); assert.equal(Number(info.size), f.bytes);
    const stamp = `${file}:${info.size}:${info.mtimeNs}:${info.ctimeNs}`; current.push(stamp);
    if (previous?.[current.length - 1] !== stamp) assert.equal(await fileHash(file), f.sha256);
  }
  session.verifiedInventory.set(key, current);
}
export function provisioningApproval(cell, target, admission) {
  const row = target.model.downloads.find(d => d.repo === cell.repository && d.revision === cell.resolvedRevision && (!d.variant || d.variant === 'bf16'));
  need(row && ['black-forest-labs/flux-3-image-klein', 'black-forest-labs/flux-3-image-klein-base', 'black-forest-labs/flux-3-image-klein-pup'].includes(row.repo), 'provisioning refuses arbitrary/non-frozen repository'); assert.deepEqual(row.files, []);
  const approved = admission.downloadApprovals?.find(a => a.repository === row.repo && a.resolvedRevision === row.revision);
  need(approved?.userDownloadApprovalId && approved.wholeRepository === true && approved.accountGateAcknowledged === true && approved.diskAcknowledged === true && approved.downloadBytes === row.estimatedSizeBytes, `BLOCKED: explicit user whole-repository download/disk/account-gate approval required for ${row.repo}@${row.revision} (${row.estimatedSizeBytes} bytes)`);
  return { repository: row.repo, resolvedRevision: row.revision, modelId: cell.options.provisionModelId, downloadBytes: row.estimatedSizeBytes, userDownloadApprovalId: approved.userDownloadApprovalId };
}
export async function provisionCell(session, cell, target, spec, report) {
  const authorization = session.authorizations?.find(a => a.repository === cell.repository && a.resolvedRevision === cell.resolvedRevision && a.httpStatus === 200);
  need(authorization && session.execution && authorization.sourceSha === session.execution.sourceSha && authorization.inferenceMainSha === session.execution.inferenceMainSha && authorization.runnerAccount === session.execution.runnerAccount, 'positive installation needs real matching MAIN/runner-account pinned-config authorization');
  const modelId = cell.options.provisionModelId, workers = await session.endpoint.call('GET', '/api/v1/workers'); assertDownloadWorker(workers, `flux3-app-${session.identity.nonce}`);
  await revalidateMappedArtifacts(session.artifactMapping);
  const body = { licenseAcknowledged: true }, accepted = await session.endpoint.call('POST', `/api/v1/models/${safeId(modelId)}/download`, body);
  assert.equal(accepted.type, 'model_download'); assert.equal(accepted.payload?.modelId, modelId); assert.equal(accepted.payload?.repo, cell.repository); assert.equal(accepted.payload?.revision, cell.resolvedRevision); assert.deepEqual(accepted.payload?.files, [], 'positive provisioning must request whole pinned repository');
  report.requests.push({ body, sha256: objectHash(body), repository: cell.repository, resolvedRevision: cell.resolvedRevision });
  const finished = await waitJob(session, accepted.id, report, { utility: true }); assert(finished.workerId, 'positive download must identify its owned utility worker'); report.jobs.push(finished);
  const marker = await contained(path.join(session.identity.state, 'data/models', modelId, '.sceneworks-download-complete.json'), session.identity.state), envelope = JSON.parse(await readFile(marker, 'utf8'));
  const receipt = (envelope.receipts ?? [envelope]).find(r => r.jobId === finished.id && r.modelId === modelId && r.repo === cell.repository && r.snapshotRevision === cell.resolvedRevision);
  assert(receipt && receipt.schemaVersion === 2 && receipt.completedAt, 'positive installation requires actual job-bound schema-v2 receipt, not pre-existing identity'); assert.deepEqual(receipt.manifestFiles, []);
  const inventory = spec.packages?.[cell.artifactKey]; await verifyInventory(session, inventory); assert.equal(inventory.repository, cell.repository); assert.equal(inventory.resolvedRevision, cell.resolvedRevision);
  assert.equal(new Set(inventory.files.map(f => f.path)).size, inventory.files.length); assert.deepEqual([...receipt.resolvedFiles].sort(), inventory.files.map(f => f.path).sort(), 'receipt must cover the complete approved pinned inventory');
  const installed = (await session.endpoint.call('GET', '/api/v1/models')).find(m => m.id === modelId); assert.equal(installed?.installState, 'installed'); assert.equal(await realpath(installed.installedPath), await realpath(inventory.root));
  assert.equal(await fileHash(await contained(path.join(inventory.root, 'config.json'), inventory.containmentRoot ?? inventory.root)), authorization.configSha256, 'installed config must match actual authenticated pinned config');
  if (modelId !== 'flux3_image_klein_pup') { assert(/FLUX.*Kommunity/i.test(await readFile(await contained(path.join(inventory.root, 'LICENSE.md'), inventory.containmentRoot ?? inventory.root), 'utf8'))); }
  const mapped = modelId === 'flux3_image_klein_pup' ? session.artifactMapping?.pup : session.artifactMapping?.snapshots[modelId]; if (mapped) mapped.install = { ...mapped.install, marker, receipt };
  invalidateArtifactVerification(); await revalidateMappedArtifacts(session.artifactMapping);
  report.provisioning = { modelId, repository: cell.repository, resolvedRevision: cell.resolvedRevision, authorization, jobId: finished.id, receipt, receiptSha256: await fileHash(marker), inventory, credentialIncluded: false };
}
export async function assertPackage(session, cell, target, inventory) {
  const model = (await session.endpoint.call('GET', '/api/v1/models')).find(m => m.id === cell.modelId);
  need(model?.installState === 'installed', `${cell.modelId} not installed on this runner: authorize the selected runner account for the gated pinned repository and separately approve its whole pinned repository download; never fallback.`);
  const row = target.downloads.find(d => d.variant === cell.tier);
  if (['fp8r', 'nvfp4'].includes(cell.tier)) need(session.worker?.capabilities?.includes(cell.tier === 'fp8r' ? 'fp8r' : 'flux3_nvfp4'), `owned GPU worker does not advertise the actual ${cell.tier} device floor; retain this cell BLOCKED`);
  need(row, `No trusted production catalog ${cell.modelId}/${cell.tier} row; endpoint refusal is required, not an unsupported exception.`);
  assert.deepEqual(row.files, []); assert.equal(row.repo, cell.repository); assert.equal(row.revision, cell.resolvedRevision);
  need(inventory?.root && inventory.files?.length && inventory.authority === 'parent-approved-pinned-inventory', 'parent-approved upstream inventory required; a self-reported conversion receipt is not trusted pins');
  assert.equal(inventory.repository, row.repo); assert.equal(inventory.resolvedRevision, row.revision);
  if (cell.tier === 'bf16') assert.equal(await realpath(model.installedPath), await realpath(inventory.root), 'installed product snapshot differs from pinned inventory');
  await verifyInventory(session, inventory);
  const resolverMapping = await verifyTerminalMapping(session, cell, inventory); assert.equal(await realpath(model.installedPath), await realpath(resolverMapping.snapshotRoot), 'catalog/worker selected a different snapshot tree');
  const names = new Set(inventory.files.map(f => f.path));
  for (const required of ['config.json', 'model.safetensors', ...(['q4','q8'].includes(cell.tier) ? ['flux3-conversion-receipt.json'] : ['manifest.json'])]) assert(names.has(required), `complete native package inventory missing ${required}`);
  if (!['fp8r', 'nvfp4'].includes(cell.tier)) for (const required of ['LICENSE.md', 'NOTICE.md']) assert(names.has(required), `package license inventory missing ${required}`);
  else { if (cell.tier === 'fp8r') assert(names.has('hf_quant_config.json'), 'FP8R publisher format configuration required'); const license = inventory.externalComponents?.find(c => c.role === 'license'); need(license && ['LICENSE.md', 'NOTICE.md'].every(name => license.files?.some(f => f.path === name)), 'publisher variant licenses/notices live at the pinned repository root, not inside variants; bind that real external license inventory'); assert.equal(license.repository, cell.repository); assert.equal(license.resolvedRevision, cell.resolvedRevision); await verifyInventory(session, license); assert.equal(await realpath(inventory.root), await realpath(path.join(model.installedPath, 'variants', cell.tier))); }
  if (cell.tier === 'bf16') for (const required of ['ae.safetensors', 'text_encoder/config.json', 'text_encoder/tokenizer.json', 'text_encoder/model.safetensors.index.json', 'text_encoder/model-00001-of-00002.safetensors', 'text_encoder/model-00002-of-00002.safetensors']) assert(names.has(required), `complete encoder/AE inventory missing ${required}`);
  else { need(inventory.externalComponents?.length >= 2, 'native tier must bind real external text encoder and autoencoder inventories'); for (const role of ['text_encoder', 'autoencoder']) { const component = inventory.externalComponents.find(c => c.role === role); need(component, `missing ${role} inventory`); assert.equal(component.repository, cell.repository); assert.equal(component.resolvedRevision, cell.resolvedRevision); await verifyInventory(session, component); } }
  if (cell.options.enhancePrompt) { const pup = inventory.components?.prompt_upsampler, row = target.model.downloads.find(d => d.componentId === 'prompt_upsampler'); need(pup && row, 'Install separately pinned optional PUP component, or leave PUP cases BLOCKED (never fallback PASS)'); assert.equal(pup.repository, row.repo); assert.equal(pup.resolvedRevision, row.revision); await verifyInventory(session, pup); }
  if (row.localDerivation) { const weight = inventory.files.find(f => f.path === row.localDerivation.weightsFile); assert(weight, 'converted primary weight missing'); assert.equal(weight.sha256, row.localDerivation.weightsSha256); assert.equal(weight.bytes, row.localDerivation.weightsBytes); }
  return { modelId: model.id, installState: model.installState, artifactKey: cell.artifactKey, inventory, actualWorkerPackage: resolverMapping.root, actualWorkerComponents: resolverMapping.components };
}
export function assertImageObservations(cell, body, asset, inputs) {
  const raw = asset.recipe.rawAdapterSettings;
  assert.equal(raw.quantTier, cell.tier, 'saved named native tier must match request');
  assert.equal(raw.mlxQuantize ?? null, ({ q4: 4, q8: 8 })[cell.tier] ?? null, 'worker-resolved native quantization bits mismatch');
  if (body.referenceAssetIds?.length) {
    const refs = raw.flux3ReferenceEditing?.references;
    assert.equal(refs?.length, body.referenceAssetIds.length, 'worker reference bindings required');
    for (let i = 0; i < refs.length; i++) { assert.equal(refs[i].assetId, body.referenceAssetIds[i]); const input = inputs[cell.inputBindings[i].binding]; assert.equal(refs[i].ordinal, i); assert.deepEqual(refs[i].orientedSize, input.orientedSize); assert.equal(refs[i].contentSha256, input.decodedContentSha256, 'worker decoded carrier binding must match retained input pixels/mode/dimensions, not encoded PNG hash'); }
    assert.deepEqual(raw.flux3_reference_ids, body.advanced.flux3_reference_ids);
  }
  if (body.advanced.flux3_layout) {
    assert.deepEqual(raw.flux3_layout, body.advanced.flux3_layout); assert.deepEqual(raw.flux3Layout?.authoring, body.advanced.flux3_layout);
    assert.deepEqual(raw.flux3Layout?.referenceInstanceIds ?? [], body.advanced.flux3_reference_ids ?? []);
    for (let i = 0; i < (body.referenceAssetIds?.length ?? 0); i++) { const binding = raw.flux3Layout?.references?.[i]; assert.equal(binding?.ordinal, i); assert.equal(binding?.assetId, body.referenceAssetIds[i]); assert.equal(binding?.contentSha256, inputs[cell.inputBindings[i].binding].decodedContentSha256); }
  }
  if (body.advanced.enhancePrompt) {
    const p = raw.promptEnhancement; assert(p && p.originalPrompt === body.prompt && p.effectivePrompt?.trim(), 'original/effective prompt required');
    assert.equal(p.mode, cell.options.pupMode); assert.equal(p.outcome, 'enhanced', 'fallback is not successful upsampler proof'); assert(p.fallbackReason == null, 'positive PUP proof must not carry an unexpected fallback reason');
    if (body.advanced.flux3_layout) { assert(typeof p.lockedAnnotations === 'string'); assert.equal(p.lockedAnnotations, raw.flux3Layout.lockedAnnotations, 'PUP must preserve the native compiled locked annotation bytes'); const rows = JSON.parse(p.lockedAnnotations); assert(Array.isArray(rows) && rows.length > 0); assert(p.effectivePrompt.includes(p.lockedAnnotations), 'locked annotation bytes must survive enhancement'); }
  } else assert(!raw.promptEnhancement || raw.promptEnhancement.outcome === 'absent', 'PUP-off cannot invoke enhancer');
  if (body.loras.length) {
    assert.deepEqual(asset.recipe.loras?.map(l => ({ name: l.name, weight: l.weight })), body.loras.map(l => ({ name: l.name, weight: l.weight })), 'saved LoRA identity and scale required');
  }
}
export async function retainImage(session, cell, body, job, project, directory, report, inputs) {
  assert(report.events.some(e => e.id === job.id && e.status === 'running' && typeof e.progress === 'number'), 'real running progress required; terminal status alone is insufficient');
  const library = await session.endpoint.call('GET', `/api/v1/projects/${safeId(project.id)}/assets`), asset = library.find(a => a.id === job.result?.assetIds?.[0]); assert(asset, 'completed image missing from GET library');
  const media = await contained(path.join(project.path, asset.file.path), project.path);
  const sidecarPath = await contained(media.replace(/\.png$/, '.sceneworks.json'), project.path), sidecarBytes = await readFile(sidecarPath), sidecar = JSON.parse(sidecarBytes.toString('utf8'));
  const proof = assertSavedAsset(job, asset, sidecar, { body, projectId: project.id, workerId: session.identity.workerId, backend: session.worker.gpuId, adapter: cell.backend === 'candle' ? 'candle_flux3' : 'mlx_flux3' });
  assertImageObservations(cell, body, asset, inputs);
  const served = await session.endpoint.served(`/api/v1/projects/${safeId(project.id)}/files/${asset.file.path}`), saved = await readFile(media); assert.equal(served.status, 200); assertServedPng(served.bytes, saved, body.width, body.height); const pixelValidity = assertFinishedPng(saved, body.width, body.height);
  await mkdir(directory, { recursive: true }); const retainedPng = path.join(directory, `${safeId(job.id)}.png`), retainedSidecar = path.join(directory, `${safeId(job.id)}.sceneworks.json`);
  await copyFile(media, retainedPng); await copyFile(sidecarPath, retainedSidecar);
  assert.equal(await fileHash(retainedPng), sha256(served.bytes)); assert.equal(await fileHash(retainedSidecar), sha256(sidecarBytes), 'retained sidecar must be the exact verified persistence bytes');
  return { ...proof, semanticAcceptance: 'PENDING', pixelValidity, visualObligations: cell.visualObligations ?? [], assetId: asset.id, pngSha256: sha256(saved), sidecarSha256: await fileHash(retainedSidecar), width: body.width, height: body.height, retainedPng, retainedSidecar, recipe: asset.recipe, device: await session.assertDeviceProof() };
}
async function runImage(session, cell, body, project, directory, report, inputs) {
  report.requests.push({ body, sha256: objectHash(body) }); const accepted = await session.endpoint.call('POST', '/api/v1/image/jobs', body); assertAccepted(accepted, body);
  const job = await waitJob(session, accepted.id, report); report.jobs.push(job);
  report.outputs.push(await retainImage(session, cell, body, job, project, directory, report, inputs)); return job;
}
export async function importAdapter(session, spec, cell, project, report) {
  need(spec?.root && spec.path && digestPattern.test(spec.sha256 ?? '') && spec.repository === cell.repository && spec.resolvedRevision === cell.resolvedRevision && spec.rank > 0 && spec.bytes > 0, 'provisioned compatible adapter identity/size/rank required (no fixture as real proof)');
  const file = await contained(spec.path, spec.root); assert.equal((await stat(file)).size, spec.bytes); assert.equal(await fileHash(file), spec.sha256);
  // Read actual safetensors metadata/tensor shapes; never accept rank only from a filename.
  const header = await safetensorsHeader(file); const ranks = Object.entries(header).filter(([k]) => /lora_(A|down)\.weight$/.test(k)).map(([, v]) => v.shape?.[0]);
  assert(ranks.length && ranks.every(rank => rank === spec.rank), 'actual LoRA tensor rank mismatch');
  const accepted = await session.endpoint.call('POST', '/api/v1/loras/import', { sourcePath: file, family: 'flux3', baseModel: cell.modelId, expectedSha256: spec.sha256, scope: 'project', projectId: project.id, name: `S10 ${cell.modelId} provisioned adapter` });
  const job = await waitJob(session, accepted.id, report); const loras = await session.endpoint.call('GET', `/api/v1/loras?projectId=${safeId(project.id)}`);
  const lora = loras.find(l => l.id === job.result?.loraId); assert(lora?.installState === 'installed', 'import must register real compatible adapter'); report.adapters.push({ spec, job, lora }); return lora;
}
export async function trainingSequence(session, cell, target, project, inputs, inputSpec, directory, report) {
  const edit = cell.options.trainingMode === 'edit', spec = inputSpec.training?.[cell.options.trainingMode];
  need(spec?.items?.length && spec.items.length <= 20 && spec.steps === 4 && spec.saveEvery === 1 && (spec.rank ?? 4) === 4, 'only TWO globally approved four-step training lifecycles (saveEvery=1, rank=4) are allowed; no expansion/retry');
  report.trainingInputs = spec.items.map(item => ({ target: inputs[item.binding], reference: edit ? inputs[item.referenceBinding] : undefined, caption: item.caption, captionSha256: typeof item.caption === 'string' ? sha256(item.caption) : null }));
  const items = spec.items.map(item => { need(inputs[item.binding], `missing training ${item.binding}`); need(typeof item.caption === 'string' && item.caption.trim(), 'real training caption required'); if (edit) need(inputs[item.referenceBinding], 'edit training requires one real reference binding per target'); return { assetId: inputs[item.binding].assetId, caption: { text: item.caption }, ...(edit ? { references: [{ assetId: inputs[item.referenceBinding]?.assetId }] } : {}) }; });
  const dataset = await session.endpoint.call('POST', `/api/v1/projects/${safeId(project.id)}/training/datasets`, { name: 'S10 real bounded dataset', items });
  const targets = await session.endpoint.call('GET', '/api/v1/training/targets');
  const trainingTarget = targets.targets?.find(t => t.id === (edit ? 'flux3_klein_edit_lora' : 'flux3_klein_lora')); need(trainingTarget?.baseModel === 'flux3_klein_base', 'shipped base training target unavailable');
  const config = { ...trainingTarget.defaults, steps: spec.steps, saveEvery: spec.saveEvery, rank: spec.rank ?? 4, alpha: spec.rank ?? 4, resolution: 512, advanced: { ...trainingTarget.defaults.advanced, resume: false, sampleEvery: 0 } };
  const body = { targetId: trainingTarget.id, datasetId: dataset.id, config, outputName: `S10 ${cell.options.trainingMode} adapter`, dryRun: false }; report.requests.push({ body, sha256: objectHash(body) });
  await session.pauseWorker(); // Native config/data proof must precede the first actual training start.
  const accepted = await session.endpoint.call('POST', `/api/v1/projects/${safeId(project.id)}/training/jobs`, body);
  assert.equal(accepted.payload?.dryRun ?? accepted.payload?.plan?.dryRun, false, 'real training must not be dryRun');
  let expected;
  try {
    const persisted = accepted.payload.plan; assert.equal(persisted.dataset.datasetId, dataset.id, 'persisted plan selected a different dataset'); if (dataset.version !== undefined) assert.equal(persisted.dataset.datasetVersion, dataset.version); assert.equal(persisted.target.targetId, trainingTarget.id);
    const relative = path.relative(session.identity.state, persisted.output.outputDir); assert(relative && !relative.startsWith('..') && !path.isAbsolute(relative), 'training output escaped owned state');
    let names; try { names = await readdir(persisted.output.outputDir); } catch (error) { if (error.code !== 'ENOENT') throw error; names = []; } assert(!names.some(name => name.endsWith('.safetensors')), 'fresh training output already contains checkpoint/export state');
    const witness = await session.trainingWitness(persisted); expected = await bindTrainingExpectation(session, persisted, config, report.trainingInputs, witness, assertFinishedPng); report.initialTrainingExpectation = expected;
  } catch (error) { await session.endpoint.call('POST', `/api/v1/jobs/${safeId(accepted.id)}/cancel`, {}); await waitJob(session, accepted.id, report, { until: j => j.status === 'canceled', allowCanceled: true }); session.failClosed(error); throw error; }
  await session.restartWorker();
  const checkpointJob = await waitJob(session, accepted.id, report, { until: j => j.status === 'running' && j.result?.resumeCheckpointStep > 0 });
  assert(checkpointJob.result.resumeCheckpointStep < spec.steps, 'interrupt must happen before training finishes');
  const plan = checkpointJob.payload.plan; assert.deepEqual(plan, accepted.payload.plan, 'running job changed the proved persisted training plan'); const stem = path.basename(plan.output.fileName, '.safetensors');
  // Worker checkpoint file names are persisted-plan-derived; retain the actual file, not a claimed step.
  const outputDir = plan.output.outputDir;
  const checkpoint = await contained(path.join(outputDir, `${stem}-step${String(checkpointJob.result.resumeCheckpointStep).padStart(6, '0')}.resume.safetensors`), outputDir);
  const checkpointHash = await fileHash(checkpoint); report.checkpoint = { path: checkpoint, sha256: checkpointHash, step: checkpointJob.result.resumeCheckpointStep, state: await assertCheckpoint(checkpoint, cell, checkpointJob.result.resumeCheckpointStep) };
  verifyNativeContinuation(report.checkpoint.state, expected);
  await session.endpoint.call('POST', `/api/v1/jobs/${safeId(accepted.id)}/cancel`, {});
  await waitJob(session, accepted.id, report, { until: j => j.status === 'canceled', allowCanceled: true });
  const oldPid = session.workerPid; assert.equal(await fileHash(checkpoint), checkpointHash);
  session.checkpointJobs ??= {};
  session.checkpointJobs[cell.modelId] = { id: accepted.id, plan: structuredClone(plan), checkpoint, checkpointHash };
  const altered = structuredClone(plan); altered.config.rank += 1;
  const { response: incompatible } = await ownedNegativeProbe(session, `/api/v1/jobs/${safeId(accepted.id)}/retry`, { resume: true, payloadChanges: { plan: altered } }, response => { assert.equal(response.status, 400); assert(/checkpoint identity mismatch/i.test(JSON.stringify(response.body)), 'incompatible checkpoint must produce actionable refusal'); }, report, { resumeWorker: false });
  assert.equal(await fileHash(checkpoint), checkpointHash); report.checkpointCompatibility = incompatible;
  const resumeBody = { resume: true }; report.requests.push({ route: `/api/v1/jobs/${accepted.id}/retry`, body: resumeBody, sha256: objectHash(resumeBody) });
  const resumed = await session.endpoint.call('POST', `/api/v1/jobs/${safeId(accepted.id)}/retry`, resumeBody);
  const expectedResumed = structuredClone(plan); expectedResumed.config.advanced.resume = true;
  try { assert.equal(resumed.payload?.plan?.config?.advanced?.resume, true); assert.notEqual(resumed.id, accepted.id); assert.deepEqual(resumed.payload.plan, expectedResumed, 'resume must preserve exact original plan except the explicit resume choice'); }
  catch (error) { try { await session.endpoint.call('POST', `/api/v1/jobs/${safeId(resumed.id)}/cancel`, {}); await waitJob(session, resumed.id, report, { until: j => j.status === 'canceled', allowCanceled: true }); } finally { session.failClosed(error); } throw error; }
  const next = await session.restartWorker(); assert.notEqual(next.pid, oldPid);
  const finished = await waitJob(session, resumed.id, report); assert(finished.result?.loraId && finished.result?.outputPath, 'completed job must export and register adapter');
  const finalStep = finished.result.resumeCheckpointStep; assert(finalStep > report.checkpoint.step, 'resumed job must advance real checkpoint state');
  const finalCheckpoint = await contained(path.join(outputDir, `${stem}-step${String(finalStep).padStart(6, '0')}.resume.safetensors`), outputDir);
  report.finalCheckpoint = { sha256: await fileHash(finalCheckpoint), state: await assertCheckpoint(finalCheckpoint, cell, finalStep) };
  await session.pauseWorker(); assert.deepEqual(finished.payload.plan, expectedResumed, 'completed resume job changed its persisted plan');
  const finalWitness = await session.trainingWitness(finished.payload.plan), finalExpected = await bindTrainingExpectation(session, finished.payload.plan, expectedResumed.config, report.trainingInputs, finalWitness, assertFinishedPng);
  for (const key of ['configSha256','configuration','factors','activeFactorKeys','trainingReference','baseRepository','baseRevision','artifacts','dataset','seed','policy','samplerLength']) assert.deepEqual(finalExpected[key], expected[key], 'final plan/dataset/component/config identity changed during resume');
  report.finalTrainingExpectation = finalExpected; verifyNativeContinuation(report.finalCheckpoint.state, finalExpected, report.checkpoint.state);
  const before = report.checkpoint.state.tensorManifest, after = report.finalCheckpoint.state.tensorManifest;
  assert(Object.keys(before).filter(key => key.startsWith('factor/')).some(key => before[key].sha256 !== after[key]?.sha256), 'real learned adapter factors must change after resume');
  assert.equal(finished.result.loraId, plan.output.loraId, 'registered export is not the actual planned adapter identity');
  const exported = await contained(finished.result.outputPath, outputDir); assert.equal(exported, await realpath(path.join(outputDir, plan.output.fileName)), 'export path is not the actual planned adapter file'); const exportedHash = await fileHash(exported);
  await mkdir(directory, { recursive: true }); await copyFile(checkpoint, path.join(directory, 'checkpoint.resume.safetensors')); await copyFile(exported, path.join(directory, 'exported-adapter.safetensors'));
  report.training = { firstJob: accepted.id, resumedJob: finished.id, oldPid, resumedPid: next.pid, checkpointSha256: checkpointHash, exportedSha256: exportedHash, loraId: finished.result.loraId };
  await session.restartWorker(); // Reload ordinary generation in another process, not the training session's cache.
  const imageCell = { ...cell, kind: 'image', options: { loraScale: 0.75 }, inputBindings: [] };
  const registered = (await session.endpoint.call('GET', `/api/v1/loras?projectId=${safeId(project.id)}`)).find(l => l.id === finished.result.loraId); assert(registered?.name, 'export must register a reloadable adapter');
  await runImage(session, imageCell, imageRequest(imageCell, target, project.id, inputs, registered), project, directory, report, inputs);
  assert.equal(report.outputs.at(-1).recipe.loras?.[0]?.hash, exportedHash, 'fresh ordinary generation must load the exact exported adapter file');
  // The ONLY adapter this backend's bound adapter cases may use: the exact file just exported here.
  const retained = path.join(directory, 'exported-adapter.safetensors');
  session.trainingExports ??= {};
  session.trainingExports[cell.backend] = { lifecycleId: cell.caseId, modelId: cell.modelId, root: directory, path: retained,
    sha256: exportedHash, bytes: (await stat(retained)).size, rank: config.rank, repository: cell.repository, resolvedRevision: cell.resolvedRevision };
}
/** Resolve the adapter for a LoRA case: a plan-bound training export, or a provisioned spec. */
export function boundAdapterSpec(session, cell, provisioned) {
  const binding = cell.adapterBinding;
  if (!binding) return provisioned;
  need(binding.source === 'training-export' && binding.backend === cell.backend, 'adapter case must bind its own backend training export');
  need(binding.modelId === cell.modelId, `${cell.caseId}: adapter trained on ${binding.modelId} cannot drive ${cell.modelId}`);
  const exported = session.trainingExports?.[cell.backend];
  need(exported && exported.lifecycleId === binding.lifecycleId, `${cell.caseId}: run its bound training lifecycle ${binding.lifecycleId} first; no other adapter substitutes`);
  need(exported.modelId === cell.modelId, `${cell.caseId}: exported adapter checkpoint differs from this case`);
  return exported;
}
// Negative ingress probes are CPU/API checks. Stop ONLY this nonce's GPU worker.
// If ingress unexpectedly creates any job, cancel and prove terminal while still paused.
export async function ownedNegativeProbe(session, route, body, check, report, { resumeWorker = true } = {}) {
  await session.pauseWorker();
  try {
    const before = await session.endpoint.call('GET', '/api/v1/jobs?limit=1000'); assert(before.length < 1000, 'cannot safely enumerate owned queue'); const known = new Set(before.map(j => j.id));
    const response = await session.endpoint.exchange('POST', route, body), after = await session.endpoint.call('GET', '/api/v1/jobs?limit=1000'); assert(after.length < 1000);
    const unexpected = after.filter(j => !known.has(j.id)); if (response.status < 300 && response.body?.id && !unexpected.some(j => j.id === response.body.id)) unexpected.push(response.body);
    for (const job of unexpected) {
      await session.endpoint.call('POST', `/api/v1/jobs/${safeId(job.id)}/cancel`, {});
      const terminal = await waitJob(session, job.id, report, { until: j => ['canceled', 'failed', 'completed'].includes(j.status), allowCanceled: true, allowFailed: true }); assert(!terminal.result?.assetIds?.length, 'negative probe unexpectedly published an image');
    }
    const remaining = await session.endpoint.call('GET', '/api/v1/jobs?limit=1000'); assert(!remaining.some(j => !known.has(j.id) && !['canceled', 'failed', 'completed'].includes(j.status)), 'unexpected queued jobs remain; never resume worker');
    assert.equal(unexpected.length, 0, 'negative probe unexpectedly accepted/mutated a job; it was canceled and terminal-verified with our GPU worker paused');
    check(response); const restarted = resumeWorker ? await session.restartWorker() : undefined; return { response, restarted };
  } catch (error) { session.failClosed(error); throw error; }
}
export function replayRequest(recipe, projectId) {
  const raw = recipe.rawAdapterSettings, settings = recipe.normalizedSettings;
  const guidance = recipe.model === 'flux3_klein' && raw.guidanceScale == null ? 0 : raw.guidanceScale;
  assert(Number.isInteger(raw.numInferenceSteps) && typeof raw.sampler === 'string' && typeof guidance === 'number', 'saved sampler/steps/guidance required for replay');
  return { projectId, mode: recipe.mode, model: recipe.model, prompt: recipe.prompt, negativePrompt: recipe.negativePrompt ?? '', seed: recipe.seed, width: settings.width, height: settings.height, count: settings.count, loras: [], advanced: { steps: raw.numInferenceSteps, sampler: raw.sampler, guidanceScale: guidance, enhancePrompt: raw.enhancePrompt === true, quantTier: raw.quantTier } };
}
export const RETAINED_REPLAY_CASE = 'candle.distilled.representative-repro.bf16';
export const RETAINED_REPLAY_FIRST = `${RETAINED_REPLAY_CASE}.generation-1`;
export const RETAINED_REPLAY_NEXT = `${RETAINED_REPLAY_CASE}.generation-2`;
export async function validateReplayPrefixBaseline({ previousApproval, previousReceipt, authorization, ownedRoot }) {
  const b = authorization.baseline;
  assert(b?.caseId === RETAINED_REPLAY_CASE && b.consumedInvocationId === RETAINED_REPLAY_FIRST && b.nextInvocationId === RETAINED_REPLAY_NEXT, 'only the explicitly approved first Candle replay prefix may be retained');
  assert.deepEqual(b.allowedLineageRemap, ['projectId'], 'only project lineage may be remapped for this reference-free baseline');
  const artifact = async binding => {
    assert(binding && Object.keys(binding).sort().join(',') === 'path,sha256' && path.isAbsolute(binding.path) && digestPattern.test(binding.sha256));
    const relative = path.relative(path.resolve(ownedRoot), path.resolve(binding.path));
    assert(relative && relative !== '..' && !relative.startsWith(`..${path.sep}`) && !path.isAbsolute(relative), 'retained replay artifact escaped lexical owned root');
    const file = await contained(binding.path, ownedRoot), bytes = await readFile(file);
    assert.equal(sha256(bytes), binding.sha256, 'retained replay evidence bytes changed'); return { file, bytes };
  };
  const reportFile = await artifact(b.report), png = await artifact(b.png), sidecar = await artifact(b.sidecar), readbackFile = await artifact(b.readback);
  const prior = JSON.parse(reportFile.bytes), readback = JSON.parse(readbackFile.bytes), saved = JSON.parse(sidecar.bytes);
  assert.equal(prior.sourceSha, previousApproval.sourceSha); assert.equal(prior.inferenceMainSha, previousApproval.inferenceMainSha);
  assert.equal(prior.approvedPlanSha256, planHash(previousApproval)); assert.equal(prior.ledgerId, previousApproval.ledgerId);
  assert.deepEqual(prior.ledgerReceipt, previousReceipt, 'render baseline must accompany the trusted final old ledger receipt');
  const cell = previousApproval.plan.cases.find(c => c.caseId === RETAINED_REPLAY_CASE), recorded = prior.cases.find(c => c.caseId === RETAINED_REPLAY_CASE);
  assert(cell?.backend === 'candle' && cell.kind === 'replay' && cell.modelId === 'flux3_klein' && cell.tier === 'bf16' && cell.inputBindings.length === 0);
  for (const [key, value] of Object.entries(cell)) if (key !== 'status') assert.deepEqual(recorded[key], value, 'retained replay case/options changed');
  assert.equal(recorded.executionStatus, 'FAILED', 'historical supervision failure must remain recorded, not relabeled PASS');
  assert(typeof recorded.error === 'string' && recorded.error.includes('Get-Counter') && recorded.error.includes('CounterApiError') && recorded.error.includes('performance counter samples is not valid'), 'only the known failed counter-supervision prefix is resumable');
  assert.equal(recorded.outputs.length, 1); assert.equal(recorded.jobs.length, 1); assert.equal(recorded.requests.length, 2);
  const output = recorded.outputs[0], job = recorded.jobs[0], first = recorded.requests[0], second = recorded.requests[1];
  assert.equal(path.resolve(output.retainedPng), path.resolve(b.png.path)); assert.equal(path.resolve(output.retainedSidecar), path.resolve(b.sidecar.path));
  assert.equal(await realpath(output.retainedPng), png.file); assert.equal(await realpath(output.retainedSidecar), sidecar.file);
  assert.equal(output.pngSha256, b.png.sha256); assert.equal(output.sidecarSha256, b.sidecar.sha256);
  assert.equal(first.sha256, b.firstRequestSha256); assert.equal(second.sha256, b.secondRecordedRequestSha256);
  assert.equal(objectHash(first.body), first.sha256); assert.equal(objectHash(second.body), second.sha256);
  assert.notEqual(first.sha256, second.sha256, 'retain the two original raw transport hashes; never assert false raw equality');
  assert.equal(previousReceipt.events[0].requestSha256, first.sha256);
  assert.deepEqual(first.body, second.body); assert.equal(first.body.count, 1);
  const expectedTarget = targetRegistry(await loadProductTargets()).find(t => t.backend === 'candle' && t.modelId === cell.modelId);
  assert(expectedTarget);
  assert.deepEqual(first.body, imageRequest(cell, expectedTarget, first.body.projectId, {}), 'retained request must match every approved authored inference parameter');
  assert(!first.body.referenceAssetIds?.length && first.body.loras.length === 0 && !first.body.advanced.enhancePrompt && !first.body.advanced.flux3_layout, 'retained prefix is reference-free/adapter-free/PUP-off only');
  assertAccepted(job, first.body); assert.equal(job.status, 'completed'); assert.equal(job.stage, 'completed'); assert.equal(job.error ?? null, null);
  assert.equal(job.backend, '1'); assert.equal(job.workerId, b.oldWorkerId); assert.equal(job.id, b.oldJobId);
  assert.equal(job.result.adapter, 'candle_flux3'); assert.equal(job.result.model, 'flux3_klein'); assert.equal(job.result.assetIds.length, 1); assert.equal(job.result.assets.length, 1); assert(!Object.hasOwn(job.result, 'assetWrites'));
  assert.deepEqual(job.result.assets[0], saved); assert.equal(job.result.assetIds[0], saved.id); assert.equal(output.assetId, saved.id);
  assert.equal(saved.projectId, first.body.projectId); assert.equal(saved.lineage.jobId, job.id); assert.equal(saved.type, 'image'); assert.equal(saved.generationSetId, job.result.generationSetId);
  assert.deepEqual(saved.recipe, output.recipe);
  for (const key of ['model', 'mode', 'prompt', 'seed']) assert.equal(saved.recipe[key], first.body[key]);
  assert.equal(saved.recipe.adapter, 'candle_flux3'); assert.equal(saved.recipe.rawAdapterSettings.realModelInference, true);
  assert.equal(saved.recipe.rawAdapterSettings.numInferenceSteps, first.body.advanced.steps); assert.equal(saved.recipe.rawAdapterSettings.repo, cell.repository);
  assert.equal(saved.recipe.normalizedSettings.family, 'flux3'); for (const key of ['width', 'height', 'count']) assert.equal(saved.recipe.normalizedSettings[key], first.body[key]);
  assertImageObservations(cell, first.body, saved, {});
  assert.equal(saved.file.mimeType, 'image/png'); assert.equal(saved.file.width, 1024); assert.equal(saved.file.height, 1024);
  const pixels = assertFinishedPng(png.bytes, 1024, 1024); assert.deepEqual(pixels, output.pixelValidity); assert.deepEqual(pixels, readback.pixelValidity);
  assert.equal(readback.pngSha256, b.png.sha256); assert.equal(readback.retainedSidecarHashVerified, true); assert.equal(readback.seedAndReplayReconstructionVerified, true);
  assert.equal(output.semanticAcceptance, 'PENDING'); assert.equal(readback.semanticAcceptance, 'PENDING'); assert.equal(recorded.visualInspection, 'NOT_INSPECTED');
  assert.equal(output.device.physicalGpuId, '1'); assert.equal(output.device.logicalCudaOrdinal, 0); assert(output.device.ownComputePids.includes(b.oldWorkerPid));
  // Historical proof remains GPU1 even when the separately authorized destination is GPU0.
  if (authorization.deviceTransition) {
    const prior = authorization.deviceTransition.previousDevice;
    assert.equal(output.device.physicalGpuId, prior.physicalGpuId);
    assert.equal(output.device.gpuUuid?.toLowerCase(), prior.gpuUuid.toLowerCase(), 'retained baseline UUID differs from authorized historical device');
    if (output.device.pciBusId !== undefined) assert.equal(output.device.pciBusId.toLowerCase(), prior.pciBusId.toLowerCase());
  }
  assert.deepEqual(replayRequest(saved.recipe, first.body.projectId), first.body);
  return { caseId: RETAINED_REPLAY_CASE, consumedInvocationId: RETAINED_REPLAY_FIRST, nextInvocationId: RETAINED_REPLAY_NEXT, pngSha256: b.png.sha256, sidecarSha256: b.sidecar.sha256, pngPath: png.file, sidecarPath: sidecar.file, oldJobId: job.id, oldWorkerId: job.workerId, oldWorkerPid: b.oldWorkerPid, oldSourceSha: previousApproval.sourceSha, oldDevice: structuredClone(output.device), ...(authorization.deviceTransition ? { deviceTransition: structuredClone(authorization.deviceTransition) } : {}), firstRequestSha256: first.sha256, secondRecordedRequestSha256: second.sha256, firstRequest: first.body, recordedReplayRequest: second.body, recipe: saved.recipe, historicalExecutionStatus: 'FAILED', historicalFailureKind: 'GPU_COUNTER_MEASUREMENT_FAILED', semanticAcceptance: 'PENDING' };
}
export function retainedReplayBody(prefix, projectId, authoredBody) {
  assert(prefix?.caseId === RETAINED_REPLAY_CASE && prefix.consumedInvocationId === RETAINED_REPLAY_FIRST && prefix.nextInvocationId === RETAINED_REPLAY_NEXT);
  assert(typeof projectId === 'string' && /^[a-zA-Z0-9_-]+$/.test(projectId));
  assert.deepEqual(authoredBody, { ...prefix.firstRequest, projectId }, 'only project lineage may change from the retained first request');
  const request = replayRequest(prefix.recipe, projectId);
  assert.deepEqual(request, { ...prefix.recordedReplayRequest, projectId }, 'retained replay inference parameters changed');
  assert.deepEqual(request, authoredBody); return request;
}
export async function executeRetainedReplayPrefix(session, cell, body, project, report, generate) {
  const prefix = session.replayPrefix;
  assert(cell.caseId === RETAINED_REPLAY_CASE && cell.kind === 'replay' && cell.backend === 'candle' && prefix?.caseId === cell.caseId);
  assert(prefix.authorizationSha256 && prefix.historicalExecutionStatus === 'FAILED' && prefix.semanticAcceptance === 'PENDING');
  assert(session.identity.workerId !== prefix.oldWorkerId, 'retained replay requires a fresh owned worker identity');
  assert.equal(report.outputs.length, 0); assert.equal(report.jobs.length, 0);
  const request = retainedReplayBody(prefix, project.id, body);
  if (prefix.deviceTransition) {
    const expected = prefix.deviceTransition.nextDevice, actual = session.deviceAtLaunch;
    assert(actual, 'cross-device replay requires actual launch-device proof');
    assert.equal(actual.physicalGpuId, expected.physicalGpuId);
    assert.equal(actual.gpuUuid?.toLowerCase(), expected.gpuUuid.toLowerCase());
    assert.equal(actual.pciBusId?.toLowerCase(), expected.pciBusId.toLowerCase());
    assert.equal(actual.logicalCudaOrdinal, 0);
  }
  report.replayComparison = { kind: prefix.deviceTransition ? 'CROSS_SOURCE_CROSS_DEVICE_FRESH_PROCESS_EXACT_OUTPUT_COMPARISON' : 'FRESH_PROCESS_EXACT_OUTPUT_COMPARISON', samePhysicalDeviceClaim: false, sameBinaryClaim: false, sameSourceClaim: false, historicalSourceSha: prefix.oldSourceSha, destinationSourceSha: prefix.deviceTransition?.nextSourceSha, historicalDevice: prefix.oldDevice, ...(prefix.deviceTransition ? { approvedDeviceTransition: prefix.deviceTransition, destinationDevice: session.deviceAtLaunch } : {}), exclusiveThroughputClaim: false, historicalMemoryTransferred: false };
  report.carriedBaseline = { caseId: prefix.caseId, invocationId: prefix.consumedInvocationId, sourceSha: prefix.oldSourceSha, authorizationSha256: prefix.authorizationSha256, jobId: prefix.oldJobId, pngSha256: prefix.pngSha256, sidecarSha256: prefix.sidecarSha256, firstRequestSha256: prefix.firstRequestSha256, secondRecordedRequestSha256: prefix.secondRecordedRequestSha256, historicalExecutionStatus: 'FAILED', historicalFailureKind: 'GPU_COUNTER_MEASUREMENT_FAILED', semanticAcceptance: 'PENDING', evidenceRole: 'RETAINED_RENDER_BASELINE_NOT_NEW_GENERATION' };
  await generate(request); // Production supplies ONLY runImage; CLI cannot inject a callback.
  assert.equal(report.outputs.length, 1, 'continuation creates only the actual second output');
  assert.equal(report.jobs.length, 1); assert.notEqual(report.jobs[0].id, prefix.oldJobId);
  assert.equal(report.jobs[0].workerId, session.identity.workerId);
  if (prefix.deviceTransition) {
    const expected = prefix.deviceTransition.nextDevice, actual = report.outputs[0].device;
    assert.equal(actual?.physicalGpuId, expected.physicalGpuId, 'cross-device output physical proof mismatch');
    assert.equal(actual?.gpuUuid?.toLowerCase(), expected.gpuUuid.toLowerCase());
    assert.equal(actual?.pciBusId?.toLowerCase(), expected.pciBusId.toLowerCase());
    assert.equal(actual?.logicalCudaOrdinal, 0);
  }
  assert.equal(report.outputs[0].pngSha256, prefix.pngSha256, 'fresh-process replay differs from the parent-approved retained render baseline');
  report.replayCompletedFromRetainedBaseline = true;
}
export async function executeCell(session, cell, target, project, inputs, spec, directory, report) {
  if (cell.kind === 'provision') return provisionCell(session, cell, target, spec, report);
  const body = imageRequest({ ...cell, options: Object.fromEntries(Object.entries(cell.options).filter(([key]) => key !== 'loraScale')) }, target, project.id, inputs);
  if (cell.kind === 'training') return trainingSequence(session, cell, target, project, inputs, spec, directory, report);
  if (cell.kind === 'installation') { report.observations.push('Pinned installed catalog + real file inventory verified; no installation performed'); return; }
  if (cell.kind === 'installation-failure') {
    const profile = cell.options.failure === 'authentication-denied' ? 'hf403' : 'missing-install';
    need(session.faultProfile === profile, `Requires separate explicit ${profile} fault admission and fresh empty owned cache; never modify existing cache or treat ordinary offline success as failure proof.`);
    const model = (await session.endpoint.call('GET', '/api/v1/models')).find(m => m.id === cell.modelId); assert(model && model.installState !== 'installed', 'fault profile must not use installed weights');
    // Both are CPU utility download faults against an empty owned hub + loopback denial.
    // Never enqueue an unbudgeted image generation just to discover missing weights.
    assertDownloadWorker(await session.endpoint.call('GET', '/api/v1/workers'), `flux3-app-${session.identity.nonce}`);
    const response = await session.endpoint.exchange('POST', `/api/v1/models/${safeId(cell.modelId)}/download`, { licenseAcknowledged: true });
    const expected = profile === 'hf403' ? /auth|credential|token|login|gated|403/i : /404|missing|not found|no files|install|offline/i;
    if (response.status >= 400) { assert([400, 401, 403, 404].includes(response.status)); assert(expected.test(JSON.stringify(response.body))); report.rejection = response; }
    else { assert.equal(response.body.type, 'model_download'); assert.equal(response.body.payload?.repo, cell.repository); assert.equal(response.body.payload?.revision, cell.resolvedRevision); assert.deepEqual(response.body.payload?.files, []); const failed = await waitJob(session, response.body.id, report, { until: j => j.status === 'failed', allowFailed: true, utility: true }); assert(expected.test(JSON.stringify(failed.error ?? failed.message))); report.jobs.push(failed); }
    report.faultRequests = [...session.faultRequests]; report.observations.push(`Injected loopback ${profile === 'hf403' ? 'HF403/no-credential' : 'missing-package 404'} fault, NOT evidence of successful runner login/upstream access; zero GPU generations`);
    return;
  }
  if (cell.kind === 'reject') {
    if (cell.options.invalid === 'bad-reference') { body.mode = 'edit_image'; body.referenceAssetIds = ['not-an-owned-asset']; }
    if (cell.options.invalid === 'reference-count') body.referenceAssetIds = Array(11).fill('not-an-owned-asset');
    if (cell.options.invalid === 'bad-box') body.advanced.flux3_layout = { version: 1, boxes: [{ id: 'invalid', bbox: [0.8, 0.2, 0.1, 0.9] }] };
    if (cell.options.invalid === 'checkpoint-compatibility') {
      const proof = session.checkpointJobs?.flux3_klein_base; need(proof, 'Run bounded base training on this owned session before checkpoint compatibility; no synthetic checkpoint accepted.');
      const altered = structuredClone(proof.plan); altered.target.baseModel = cell.modelId === 'flux3_klein_base' ? 'flux3_klein' : cell.modelId;
      const { response } = await ownedNegativeProbe(session, `/api/v1/jobs/${safeId(proof.id)}/retry`, { resume: true, payloadChanges: { plan: altered } }, response => { assert.equal(response.status, 400); assert(/checkpoint identity mismatch/i.test(JSON.stringify(response.body))); }, report);
      assert.equal(await fileHash(proof.checkpoint), proof.checkpointHash); report.rejection = response; return;
    }
    const { response } = await ownedNegativeProbe(session, '/api/v1/image/jobs', body, response => {
      assert.equal(response.status, 400); assert(!response.body.id);
      const expected = { 'bad-reference': /reference.*(not an asset|owned|missing)/i, 'reference-count': /10.*(references|ordered)|supplies 11/i, 'bad-box': /layout.*(positive|box|YXYX)/i }[cell.options.invalid]; assert(expected?.test(JSON.stringify(response.body)), 'rejection must identify the intended malformed input, not an unrelated missing-model error');
    }, report); report.rejection = response; return;
  }
  if (cell.kind === 'cancel') {
    const accepted = await session.endpoint.call('POST', '/api/v1/image/jobs', body); assertAccepted(accepted, body);
    await waitJob(session, accepted.id, report, { until: j => j.status === 'running' && j.progress > 0 });
    await session.endpoint.call('POST', `/api/v1/jobs/${safeId(accepted.id)}/cancel`, {});
    const canceled = await waitJob(session, accepted.id, report, { until: j => j.status === 'canceled', allowCanceled: true }); assert(!canceled.result?.assetIds?.length);
    const library = await session.endpoint.call('GET', `/api/v1/projects/${safeId(project.id)}/assets`); assert(!library.some(a => a.lineage?.jobId === accepted.id), 'canceled job must not publish a finished library image'); report.jobs.push(canceled); return;
  }
  const adapterSpec = Object.hasOwn(cell.options, 'loraScale') ? boundAdapterSpec(session, cell, spec.adapters?.[cell.modelId]) : undefined;
  if (adapterSpec) { const lora = await importAdapter(session, adapterSpec, cell, project, report); body.loras = [{ id: lora.id, name: lora.name, weight: cell.options.loraScale }]; }
  if (session.replayPrefix && cell.caseId === RETAINED_REPLAY_CASE) {
    return executeRetainedReplayPrefix(session, cell, body, project, report, request => runImage(session, cell, request, project, directory, report, inputs));
  }
  await runImage(session, cell, body, project, directory, report, inputs);
  if (Object.hasOwn(cell.options, 'loraScale')) {
    const proof = report.outputs.at(-1), specAdapter = adapterSpec; session.loraZeroProof ??= {};
    if (cell.options.loraScale === 0) session.loraZeroProof[cell.modelId] = { sha256: proof.pngSha256, adapterSha256: specAdapter.sha256 };
    else { const baseline = session.loraZeroProof[cell.modelId]; need(baseline?.adapterSha256 === specAdapter.sha256, 'Run matching scale-zero case before adapter-on effect proof'); assert.notEqual(proof.pngSha256, baseline.sha256, 'adapter-on must have an observable image effect'); assert.equal(proof.recipe.loras[0].hash, specAdapter.sha256, 'worker must report actual loaded adapter hash'); report.adapterEffect = { scaleZeroPngSha256: baseline.sha256, scaleOnPngSha256: proof.pngSha256, adapterSha256: specAdapter.sha256 }; }
  }
  if (cell.kind === 'replay') {
    const saved = report.outputs.at(-1).recipe; assert.equal(saved.seed, body.seed);
    const replay = replayRequest(saved, project.id); assert.deepEqual(replay, body, 'saved settings must reconstruct the authored ordinary request');
    await session.restartWorker(); await runImage(session, cell, replay, project, directory, report, inputs);
    assert.equal(report.outputs.at(-1).pngSha256, report.outputs.at(-2).pngSha256, 'same request/seed fresh-process reproduction differs');
  }
}
export async function runTerminal({ admission, inputSpec, sources, caseIds, approvedPlan, inputSpecAddendum }, dependencies = {}) {
  const approvedBudget = validateApprovedPlan(approvedPlan); assert.equal(admission.approvedPlanSha256, approvedBudget.approvedPlanSha256); assert.equal(admission.ledgerId, approvedPlan.ledgerId); assert.equal(admission.sourceSha, approvedPlan.sourceSha); assert.equal(admission.inferenceMainSha, approvedPlan.inferenceMainSha);
  // The admitted runner selects the ledger lane; every later check uses that host's lane only.
  const host = admittedHost(approvedPlan, admission); assert.equal(objectHash(inputSpec), boundInputSpecSha256(approvedPlan, admission.backend, inputSpecAddendum, admission.runner));
  const expected = new Map([...terminalCatalog(sources).cases, ...terminalPlan(sources).cases].map(c => [c.caseId, c]));
  const plan = approvedPlan.plan, registry = targetRegistry(sources), cells = plan.cases.filter(c => caseIds.includes(c.caseId)).sort((a, b) => plan.phases.indexOf(a.phase) - plan.phases.indexOf(b.phase));
  for (const cell of cells) assert.equal(planHash(cell), planHash(expected.get(cell.caseId)), 'approved case/options must name a shipped representative or explicit diagnostic case; no automatic expansion');
  assert.equal(new Set(caseIds).size, caseIds.length); assert.equal(cells.length, caseIds.length, 'unknown requested case ID'); assert(cells.length);
  await validateAdmission(admission, cells[0].backend, dependencies.admission);
  await auditLedger({ approval: approvedPlan, backend: admission.backend, ledgerRoot: admission.ledgerRoot, inputSpecAddendum, host: admission.runner });
  const staging = artifactStagingPlan(cells, registry, inputSpec);
  for (const p of staging.derivedPackages) assert((admission.disk?.conversionOutputBytes?.[p.artifactKey] ?? 0) >= p.copyBytes, 'disk admission must include all actual unique owned package copies before any copy');
  (dependencies.print ?? console.log)(JSON.stringify({ kind: 'exact-approved-physical-run-budget', approvedPlanSha256: approvedBudget.approvedPlanSha256, userPlanApprovalId: approvedPlan.userPlanApprovalId, budget: plan.budget, artifactStaging: staging, invocations: plan.generationInvocations, trainingLifecycles: plan.trainingLifecycles, selectedCaseIds: caseIds, resourceBoundaryUnverified: plan.resourceCases.filter(c => c.status !== 'READY').map(c => c.caseId) }));
  assert(cells.every(c => c.backend === admission.backend && admission.phases.includes(c.phase)), 'case outside backend/phase admission');
  // Wrong-host refusal before any launch/spend: a case whose permits belong to another host lane is never run here.
  for (const cell of cells) for (const id of [...plan.generationInvocations.filter(i => i.caseId === cell.caseId).map(i => i.invocationId), ...plan.trainingLifecycles.filter(t => t.caseId === cell.caseId).flatMap(t => [t.startId, t.resumeId])]) assert(inHostLane(plan, admission.backend, host, id), `BLOCKED: ${cell.caseId} is assigned to another ${admission.backend} host lane, not ${admission.runner}`);
  if (cells.some(c => c.kind === 'training')) need(admission.trainingLicenseAcknowledged === true, 'explicit acknowledgment of actual adaptation license conditions required; output exceptions are not blanket commercial-training permission');
  if (admission.faultProfile) assert(admission.installationFaultAdmission === true && cells.every(c => c.kind === 'installation-failure' && c.options.failure === (admission.faultProfile === 'hf403' ? 'authentication-denied' : 'install-missing')), 'fault profile can select only its explicitly admitted failure cases');
  // Conservative across split volumes: each admitted root must cover the combined budget.
  // Existing headroom constants are safety margins, not resource measurements.
  const free = dependencies.diskFree ?? (async root => { const fs = await statfs(root, { bigint: true }); return Number(fs.bavail * fs.bsize); });
  const actualFreeBytes = Math.min(await free(admission.outputRoot), await free(admission.hfHome));
  const budget = requestedDiskBudget(registry, cells, { ...admission.disk, freeBytes: Math.min(actualFreeBytes, admission.disk.freeBytes) });
  assert.equal(objectHash(inputSpec), admission.inputSpecSha256, 'input specification differs from admitted artifact bindings');
  await mkdir(admission.outputRoot, { recursive: true });
  const campaign = { schemaVersion: 1, kind: 'flux3-installed-product-terminal-report', admissionId: admission.admissionId, ledgerId: approvedPlan.ledgerId, approvedPlanSha256: approvedBudget.approvedPlanSha256, expandedPhysicalBudget: plan.budget, uncovered: plan.uncovered, sourceSha: admission.sourceSha, inferenceMainSha: admission.inferenceMainSha, runner: admission.runner, runnerAccount: admission.runnerAccount, status: 'NOT_RUN', budget, cases: plan.cases.map(c => ({ caseId: c.caseId, status: c.status, blocker: c.blocker })) };
  const output = path.join(admission.outputRoot, `terminal-${randomBytes(24).toString('hex')}`); await mkdir(output);
  let session;
  const save = () => { const text = JSON.stringify(campaign, null, 2) + '\n'; return writeFile(path.join(output, 'report.json'), session?.redact ? session.redact(text) : text); };
  await save();
  try {
    const approvedProvisioning = cells.filter(c => c.kind === 'provision').map(c => provisioningApproval(c, registry.find(t => t.backend === c.backend && t.modelId === c.modelId), admission));
    session = await (dependencies.launch ?? launchTerminalSession)({ ...admission, approvedProvisioning, approvedPlan, inputSpecAddendum, artifactInputs: inputSpec, artifactCells: cells, artifactTargets: registry }); campaign.sessionIdentity = session.identity; campaign.deviceAtLaunch = session.deviceAtLaunch ?? null; campaign.artifactStaging = session.artifactMapping?.stagingPlan ?? staging; campaign.artifactCopies = session.artifactMapping ? { files: session.artifactMapping.copies, bytes: session.artifactMapping.copiedBytes } : null;
    const project = await session.endpoint.call('POST', '/api/v1/projects', { name: 'S10 installed product terminal acceptance' });
    const rel = path.relative(await realpath(session.identity.state), await realpath(project.path)); assert(rel && !rel.startsWith('..') && !path.isAbsolute(rel), 'project escaped owned state');
    const inputs = {};
    const neededBindings = new Set(cells.flatMap(c => c.inputBindings.map(b => b.binding)));
    for (const cell of cells.filter(c => c.kind === 'training')) for (const item of inputSpec.training?.[cell.options.trainingMode]?.items ?? []) { neededBindings.add(item.binding); if (cell.options.trainingMode === 'edit') neededBindings.add(item.referenceBinding); }
    for (const binding of neededBindings) {
      inputs[binding] = await uploadBinding(session, project, inputSpec.images?.[binding]);
      const retainedDirectory = path.join(output, 'inputs'); await mkdir(retainedDirectory, { recursive: true });
      const retainedInput = path.join(retainedDirectory, `${safeId(binding)}.png`); await copyFile(inputs[binding].originalPath, retainedInput); assert.equal(await fileHash(retainedInput), inputs[binding].sha256, 'retained input must match uploaded source bytes'); inputs[binding].retainedInput = retainedInput;
    }
    for (const cell of cells) {
      const report = { ...cell, status: 'NOT_RUN', executionStatus: 'NOT_RUN', sourceSha: admission.sourceSha, inferenceMainSha: admission.inferenceMainSha, resourceAdmissionId: admission.admissionId, events: [], requests: [], jobs: [], adapters: [], outputs: [], observations: [], inputBindings: cell.inputBindings.map(b => ({ ...b, ...inputs[b.binding] })), timeoutSeconds: admission.timeoutSeconds, visualInspection: 'NOT_INSPECTED' };
      campaign.cases[campaign.cases.findIndex(c => c.caseId === cell.caseId)] = report;
      const target = registry.find(t => t.backend === cell.backend && t.modelId === cell.modelId);
      session.beginCase(cell);
      try {
        if (cell.status === 'BLOCKED') { const { response: refusal } = await ownedNegativeProbe(session, '/api/v1/image/jobs', imageRequest(cell, target, project.id, inputs), response => { assert.equal(response.status, 400, 'unprovisioned tier endpoint must refuse'); assert(JSON.stringify(response.body).includes(`${cell.tier} package has no verified catalog artifact`), 'tier refusal must name missing verified pins, not unrelated unsupported/missing-model errors'); }, report); report.rejection = refusal; throw new Blocked(cell.blocker); }
        if (!['installation-failure', 'reject', 'provision'].includes(cell.kind)) report.package = await assertPackage(session, cell, target, inputSpec.packages?.[cell.artifactKey]);
        await executeCell(session, cell, target, project, inputs, inputSpec, path.join(output, cell.caseId), report);
        report.executionStatus = 'PASSED'; report.semanticAcceptance = report.outputs.length ? 'PENDING' : 'NOT_APPLICABLE';
        report.status = report.outputs.length ? 'AWAITING_VISUAL_INSPECTION' : 'PASSED';
      } catch (e) { report.status = e instanceof Blocked || e instanceof ArtifactReadinessBlocked ? 'BLOCKED' : 'FAILED'; report.executionStatus = report.status; report.error = e.message; }
      await save(); session.alive();
      if (admission.queueReservation && report.executionStatus !== 'PASSED') throw new Error('queued campaign stops at its first failure; spent permits cannot be retried');
    }
    campaign.status = campaign.cases.some(c => c.status === 'FAILED') ? 'FAILED' : 'INCOMPLETE';
  } catch (error) { campaign.status = error instanceof Blocked || error instanceof ProvisioningBlocked || error instanceof ArtifactReadinessBlocked ? 'BLOCKED' : 'FAILED'; campaign.error = error.message;
    if (campaign.status === 'BLOCKED') for (const c of campaign.cases.filter(c => caseIds.includes(c.caseId) && c.status === 'NOT_RUN')) { c.status = 'BLOCKED'; c.blocker = error.message; }
  }
  finally { try { await session?.close(); } catch (error) { campaign.status = 'FAILED'; campaign.cleanupError = error.message; } finally { campaign.cleanup = session?.cleanup ?? []; campaign.permits = session?.permits ?? []; try { campaign.ledgerReceipt = await auditLedger({ approval: approvedPlan, backend: admission.backend, ledgerRoot: admission.ledgerRoot, inputSpecAddendum, host: admission.runner }); } catch (error) { campaign.status = 'FAILED'; campaign.ledgerError = error.message; } await save(); } }
  return { output, report: campaign };
}
export async function main(args = process.argv.slice(2)) {
  if (args[0] === 'prepare') {
    let formatter;
    const phase = args[1] === 'run' ? 'admission' : 'plan';
    try {
      const prep = await import('./flux3-image-klein-preparation.mjs'); formatter = prep.safePreparationFailure;
      assert((args[1] === 'plan' && args.length === 3) || (args[1] === 'run' && args.length === 4));
      const sources = await loadProductTargets(), input = JSON.parse(await readFile(args[2], 'utf8'));
      const result = args[1] === 'plan' ? prep.preparationPlan(input, sources) : await prep.runPreparation(input, JSON.parse(await readFile(args[3], 'utf8')), sources);
      console.log(JSON.stringify(result, null, 2));
    } catch (error) {
      console.log(JSON.stringify(formatter ? formatter(error, phase) : { schemaVersion: 1, kind: 'flux3-preparation-safe-error', phase, code: 'PREP_UNKNOWN_FAILURE' }));
      process.exitCode = 1;
    }
    return;
  }
  const sources = await loadProductTargets();
  if (!args.length || args[0] === 'plan' || args[0] === 'catalog') { assert(args.length <= 1); console.log(JSON.stringify(args[0] === 'catalog' ? terminalCatalog(sources) : terminalPlan(sources), null, 2)); return; }
  if (args[0] === 'produce-build') { assert(args.length === 2, 'produce-build requires one separately user-admitted build JSON'); const { produceTerminalBuild } = await import('./flux3-image-klein-terminal-build.mjs'); console.log(JSON.stringify(await produceTerminalBuild(JSON.parse(await readFile(args[1], 'utf8'))))); return; }
  if (args[0] === 'init-ledger') { assert([3, 4].includes(args.length) && ['mlx', 'candle'].includes(args[2]), 'init-ledger APPROVED_PLAN.json mlx|candle [HOST] is explicit CPU setup, never runtime reset'); const approval = JSON.parse(await readFile(args[1], 'utf8')), host = args[3]; console.log(JSON.stringify(await initializeLedger({ approval, backend: args[2], host, ledgerRoot: approval.ledgerRoots[ledgerLane(approval, args[2], host)] }))); return; }
  assert(args[0] === 'run' && [5, 6].includes(args.length), 'usage: node scripts/flux3-image-klein-terminal-driver.mjs [plan | catalog | init-ledger APPROVED_PLAN.json mlx|candle [HOST] | run ADMISSION.json INPUTS.json CASE_IDS.json APPROVED_PLAN.json [INPUT_SPEC_ADDENDUM.json]]');
  const [admission, inputSpec, caseIds, approvedPlan, inputSpecAddendum] = await Promise.all(args.slice(1).map(async f => JSON.parse(await readFile(f, 'utf8'))));
  const result = await runTerminal({ admission, inputSpec, caseIds, sources, approvedPlan, inputSpecAddendum }); console.log(JSON.stringify({ output: result.output, status: result.report.status }));
  if (result.report.status !== 'PASSED') process.exitCode = 1;
}
if (process.argv[1] && import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href) main().catch(e => { console.error(e.message); process.exitCode = 1; });
