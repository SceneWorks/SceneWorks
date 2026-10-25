// sc-26757: product evidence only. This module never loads a model or a GPU.
import { createHash } from 'node:crypto';
import { readFile, writeFile, mkdir, stat, readdir, unlink, open } from 'node:fs/promises';
import { resolve, relative, isAbsolute, dirname } from 'node:path';
import { execFileSync } from 'node:child_process';
import { inflateSync } from 'node:zlib';

export const VERSION = 'sc-26757-moss-edit-terminal/v5';
export const BACKEND_FILES = ['apiBinary', 'workerBinary', 'webPackage', 'packages', 'toolchain'];
const SHA = /^[a-f0-9]{64}$/;
export const sha = bytes => createHash('sha256').update(bytes).digest('hex');
export const canonical = value => JSON.stringify(value, (_, v) => v && typeof v === 'object' && !Array.isArray(v) ? Object.fromEntries(Object.entries(v).sort(([a], [b]) => a.localeCompare(b))) : v);
export const digest = value => sha(canonical(value));
export function requireThat(ok, message) { if (!ok) throw new Error(message); }
export const json = async path => JSON.parse(await readFile(path, 'utf8'));
export async function save(path, data) {
  await mkdir(dirname(path), { recursive: true });
  await writeFile(path, `${JSON.stringify(data, null, 2)}\n`, { flag: 'wx', mode: 0o600 });
}
export function safePath(root, name) {
  requireThat(typeof name === 'string' && name && !name.includes('\\') && !name.split('/').some(p => !p || p === '.' || p === '..') && !isAbsolute(name), 'unsafe relative evidence path');
  const path = resolve(root, name);
  requireThat(!relative(resolve(root), path).startsWith('..'), 'path escaped evidence root');
  return path;
}
export async function fileIdentity(path) {
  requireThat((await stat(path)).isFile(), 'identity requires a regular file');
  const file = await open(path, 'r');
  try {
    const before = await file.stat({ bigint: true });
    requireThat(before.isFile(), 'identity requires a regular file');
    requireThat(before.size <= BigInt(Number.MAX_SAFE_INTEGER), 'identity file size exceeds exact byte count');
    const hash = createHash('sha256'), buffer = Buffer.allocUnsafe(1024 * 1024);
    let byteLength = 0;
    for (;;) {
      const { bytesRead } = await file.read(buffer, 0, buffer.length, null);
      if (!bytesRead) break;
      byteLength += bytesRead;
      requireThat(BigInt(byteLength) <= before.size, 'identity file changed while hashing');
      hash.update(buffer.subarray(0, bytesRead));
    }
    const same = info => ['dev','ino','size','mtimeNs','ctimeNs'].every(key => info[key] === before[key]);
    requireThat(BigInt(byteLength) === before.size && same(await file.stat({ bigint: true })) && same(await stat(path, { bigint: true })), 'identity file changed while hashing');
    return { byteLength, sha256: hash.digest('hex') };
  } finally { await file.close(); }
}
export function treeIdentity(path) {
  const git = args => execFileSync('git', ['-C', path, ...args], { encoding: 'utf8' }).trim();
  requireThat(git(['status', '--porcelain', '--untracked-files=normal']) === '', 'terminal sources must be clean, including untracked files');
  return { head: git(['rev-parse', 'HEAD']), tree: git(['rev-parse', 'HEAD^{tree}']) };
}
export function validateInferencePins(manifests, cargoLock, revision) {
  for (const text of manifests) {
    const entries = text.split(/\r?\n/).map(line=>line.split('#')[0]).filter(line => /git\s*=\s*"https:\/\/github\.com\/SceneWorks\/inference(?:\.git)?"/.test(line));
    requireThat(entries.length && entries.every(line => line.match(/\brev\s*=\s*"([a-f0-9]{40})"/)?.[1] === revision), 'every SceneWorks inference manifest pin must equal the frozen paired revision');
  }
  const entries = cargoLock.split(/\r?\n/).filter(line => /^source\s*=\s*"git\+https:\/\/github\.com\/SceneWorks\/inference\?/.test(line));
  requireThat(entries.length && entries.every(line => line === `source = "git+https://github.com/SceneWorks/inference?rev=${revision}#${revision}"`), 'Cargo.lock inference source must equal the exact frozen pin');
}
export async function verifyFrozenSources(config, lock) {
  requireThat(lock.version === VERSION && ['mlx','cuda'].includes(config.localBackend), 'v5 candidate and explicit local backend required');
  requireThat(digest(await json(config.plan)) === lock.planSha256, 'plan changed after freeze');
  requireThat(digest(config.guardRequests) === lock.guardRequestsSha256, 'guard requests changed after freeze');
  for (const key of ['sceneworks', 'inference']) requireThat(canonical(treeIdentity(config.sources[key])) === canonical(lock.sources[key]), `${key} source changed after freeze`);
  const manifests = await Promise.all(['Cargo.toml','crates/sceneworks-worker/Cargo.toml','crates/sceneworks-memory-adapter/Cargo.toml'].map(file=>readFile(resolve(config.sources.sceneworks,file),'utf8')));
  validateInferencePins(manifests,await readFile(resolve(config.sources.sceneworks,'Cargo.lock'),'utf8'),lock.sources.inference.head);
  requireThat(execFileSync('git', ['-C', config.sources.inference, 'merge-base', '--is-ancestor', lock.sources.inference.head, config.inferenceMainRef], { encoding: 'utf8' }) === '', 'inference revision must be on the declared refreshed main ref');
  const files = { ...lock.files, ...lock.artifacts[config.localBackend].files };
  for (const [key, item] of Object.entries(files)) requireThat(canonical(await fileIdentity(config.files[key])) === canonical(item), `frozen ${key} changed`);
}
export function validateArtifactInventory(inventory, backend, sources) {
  requireThat(inventory?.backend === backend && canonical(inventory.sources) === canonical(sources), 'backend build inventory is not bound to the exact paired sources');
  requireThat(canonical(Object.keys(inventory.files ?? {}).sort()) === canonical([...BACKEND_FILES].sort()), 'backend inventory must contain all executable, web, package and toolchain identities');
  for (const item of Object.values(inventory.files)) requireThat(canonical(Object.keys(item).sort()) === canonical(['byteLength','sha256']) && Number.isSafeInteger(item.byteLength) && item.byteLength >= 0 && SHA.test(item.sha256 ?? ''), 'invalid path-independent artifact identity');
  return { backend, sources, files: inventory.files };
}
export function executionBinding(config, lock, files) {
  const backend = config.localBackend, endpoint = config.backends?.[backend];
  requireThat(['mlx','cuda'].includes(backend) && endpoint?.projectId && endpoint.workerId && endpoint.tokenEnv && endpoint.workerGpuId === (backend === 'cuda' ? '1' : 'mlx'), 'actual backend/project/worker/auth binding required');
  const url = new URL(endpoint.apiUrl);
  requireThat(['http:','https:'].includes(url.protocol) && !url.username && !url.password && !url.search && !url.hash, 'invalid bound API endpoint');
  return { version: VERSION, lockSha256: digest(lock), backend, endpoint, sources: lock.sources, files,
    paths: { sources: config.sources, files: config.files, plan: config.plan, evidenceRoot: config.evidenceRoot } };
}
export function validateBinding(binding, lock, backend) {
  requireThat(binding?.version === VERSION && binding.lockSha256 === digest(lock) && binding.backend === backend && canonical(binding.sources) === canonical(lock.sources), 'execution binding differs from frozen candidate/backend/sources');
  requireThat(canonical(binding.files) === canonical({ ...lock.files, ...lock.artifacts[backend].files }), 'execution binding substituted frozen artifacts');
  const endpoint = binding.endpoint;
  requireThat(endpoint?.projectId && endpoint.workerId && endpoint.tokenEnv && endpoint.workerGpuId === (backend === 'cuda' ? '1' : 'mlx'), 'execution binding lacks actual backend identity');
  const url = new URL(endpoint.apiUrl);
  requireThat(['http:','https:'].includes(url.protocol) && !url.username && !url.password && !url.search && !url.hash, 'invalid bound API endpoint');
  return binding;
}
export function validateAdmission(admission, lock, backend) {
  requireThat(admission.version === VERSION && admission.lockSha256 === digest(lock) && admission.backend === backend, 'admission is not bound to this frozen backend/candidate');
  requireThat(admission.device?.uuid && admission.device?.pciOrRegistryId && admission.device?.name, 'actual physical identity required');
  if (backend === 'cuda') requireThat(admission.device.physicalIndex === 1 && admission.device.logicalIndex === 0 && admission.cudaVisibleDevices === admission.device.uuid && admission.sceneWorksGpuId === '1' && admission.postCompile === true, 'CUDA must be freshly mapped physical GPU1, logical CUDA0, after compilation');
  else requireThat(backend === 'mlx' && admission.device.platform === 'apple-silicon', 'only native MLX or physical GPU1 CUDA admitted');
}

// Inert little-endian NPY: distinguish file digest from actual tensor-data digest.
export function makeNpy(shape, dtype, value) {
  requireThat(['float32', 'uint8'].includes(dtype) && shape.every(n => Number.isSafeInteger(n) && n > 0), 'invalid deterministic tensor');
  const n = shape.reduce((a, b) => a * b, 1), width = dtype === 'float32' ? 4 : 1;
  requireThat(n * width <= 512 * 1024 * 1024, 'tensor preparation exceeds artifact budget');
  let header = `{'descr': '${width === 4 ? '<f4' : '|u1'}', 'fortran_order': False, 'shape': (${shape.join(', ')},), }`;
  header += ' '.repeat((64 - ((10 + header.length + 1) % 64)) % 64) + '\n';
  const prefix = Buffer.from([0x93, 78, 85, 77, 80, 89, 1, 0, 0, 0]); prefix.writeUInt16LE(header.length, 8);
  const data = Buffer.alloc(n * width);
  for (let i = 0; i < n; i++) { const v = value(i); requireThat(Number.isFinite(v), 'nonfinite tensor'); if (width === 4) data.writeFloatLE(v, i * 4); else { requireThat(v === 0 || v === 1, 'mask must be binary'); data[i] = v; } }
  return Buffer.concat([prefix, Buffer.from(header), data]);
}
export function inspectNpy(bytes, expectedShape) {
  requireThat(bytes.subarray(0, 8).equals(Buffer.from([0x93, 78, 85, 77, 80, 89, 1, 0])), 'expected NPY v1');
  const size = bytes.readUInt16LE(8), text = bytes.subarray(10, 10 + size).toString('ascii');
  requireThat(size <= 4096 && /'fortran_order': False/.test(text), 'unsupported NPY header/order');
  const dtype = /'descr': '<f4'/.test(text) ? 'float32' : /'descr': '\|u1'/.test(text) ? 'uint8' : null;
  const shape = /'shape': \(([^)]*)\)/.exec(text)?.[1].split(',').map(s => s.trim()).filter(Boolean).map(Number);
  requireThat(dtype && shape?.length && shape.every(n => Number.isSafeInteger(n) && n > 0), 'unsupported NPY dtype/shape');
  const data = bytes.subarray(10 + size), elements = shape.reduce((a, b) => a * b, 1);
  requireThat(data.length === elements * (dtype === 'float32' ? 4 : 1), 'NPY length mismatch');
  if (expectedShape) requireThat(canonical(shape) === canonical(expectedShape) && dtype === 'float32', 'latent result must be single-sample F32 NCHW');
  for (let i = 0; i < elements; i++) requireThat(dtype === 'float32' ? Number.isFinite(data.readFloatLE(i * 4)) : data[i] <= 1, 'nonfinite tensor or nonbinary mask');
  return { dtype, shape, byteLength: data.length, sha256: sha(bytes), contentSha256: sha(data) };
}
export function decodeRgbPng(bytes, requireOpaque = false) {
  requireThat(bytes.subarray(0, 8).equals(Buffer.from([137,80,78,71,13,10,26,10])), 'source must be PNG');
  const width = bytes.readUInt32BE(16), height = bytes.readUInt32BE(20), color = bytes[25], channels = color === 2 ? 3 : color === 6 ? 4 : 0;
  requireThat(bytes[24] === 8 && channels && bytes[26] === 0 && bytes[27] === 0 && bytes[28] === 0 && width * height <= 32 * 1024 * 1024, 'source must be bounded noninterlaced RGB/RGBA8 PNG');
  const parts = []; for (let offset = 8; offset < bytes.length;) { const n = bytes.readUInt32BE(offset), type = bytes.subarray(offset + 4, offset + 8).toString(); requireThat(offset + 12 + n <= bytes.length, 'truncated PNG'); if (type === 'IDAT') parts.push(bytes.subarray(offset + 8, offset + 8 + n)); offset += n + 12; }
  const stride = width * channels, raw = inflateSync(Buffer.concat(parts), { maxOutputLength: (stride + 1) * height });
  requireThat(raw.length === (stride + 1) * height, 'PNG scanline length mismatch');
  const pixels = Buffer.alloc(stride * height);
  const paeth = (a,b,c) => { const p = a+b-c, pa = Math.abs(p-a), pb = Math.abs(p-b), pc = Math.abs(p-c); return pa <= pb && pa <= pc ? a : pb <= pc ? b : c; };
  for (let y = 0; y < height; y++) {
    const filter = raw[y * (stride + 1)]; requireThat(filter <= 4, 'unsupported PNG filter');
    for (let x = 0; x < stride; x++) {
      const i = y * stride + x, a = x >= channels ? pixels[i-channels] : 0, b = y ? pixels[i-stride] : 0, c = y && x >= channels ? pixels[i-stride-channels] : 0;
      pixels[i] = (raw[y*(stride+1)+x+1] + [0,a,b,Math.floor((a+b)/2),paeth(a,b,c)][filter]) & 255;
    }
  }
  if (requireOpaque && channels === 4) requireThat(pixels.every((value,index) => index % 4 !== 3 || value === 255), 'Editor scratch source must have fully opaque alpha');
  const rgb = channels === 3 ? pixels : Buffer.from(Array.from({ length: width * height * 3 }, (_, i) => pixels[Math.floor(i/3)*4+i%3]));
  return { width, height, pixels: rgb };
}
export function pngRgbIdentity(bytes, requireOpaque = false) { const {width,height,pixels} = decodeRgbPng(bytes, requireOpaque); return {width,height,pixelsSha256:sha(pixels)}; }
export function tensorData(bytes) { inspectNpy(bytes); return bytes.subarray(10 + bytes.readUInt16LE(8)); }
export function validateAdvancedDigests(base, request, tensors) {
  const advanced = request.advanced.mossImageAdvanced; if (!advanced) return;
  requireThat(base.conditioning?.length === 2, 'advanced native conditioning batch mismatch');
  for (let b = 0; b < 2; b++) for (const branch of ['positive', 'negative']) {
    const receipt = base.conditioning[b][branch];
    if (branch === 'negative' && advanced.negativePrompts) { requireThat(receipt?.source === 'text' && receipt.sha256 === sha(advanced.negativePrompts[b]), 'per-prompt negatives were not actually consumed'); continue; }
    if (branch === 'positive' && !advanced.positive) { requireThat(receipt?.source === 'text' && receipt.sha256 === sha(request.prompts[b]), 'seeded text conditioning row/order mismatch'); continue; }
    const values = tensorData(tensors[branch]), mask = tensorData(tensors[branch === 'positive' ? 'mask' : 'negativeMask']).subarray(b*3, (b+1)*3);
    requireThat(receipt?.source === 'embeddings' && receipt.sequence_length === 3 && receipt.valid_tokens === [...mask].reduce((a,n) => a+n, 0) && receipt.sha256 === sha(Buffer.concat([values.subarray(b*3*4096*4,(b+1)*3*4096*4),mask])), 'native supplied FP32 conditioning/mask digest mismatch');
  }
  if (advanced.noise) {
    const noise = tensorData(tensors.noise), stride = 128 * 4 * 4 * 4;
    requireThat(base.samples.every((s,i) => s.seed === null && s.noise_sha256 === sha(noise.subarray(i*stride,(i+1)*stride))), 'actual explicit-noise digest/sample order mismatch');
  } else requireThat(base.samples.every((s,i) => s.seed === request.seed + i && SHA.test(s.noise_sha256 ?? '')), 'seeded text sample seed/noise order mismatch');
}

export function primaryRows(inputs) {
  const transport = ['ui-full', 'mcp', 'ui-simple', 'rest', 'rest', 'ui-full', 'ui-editor', 'mcp'];
  let index = 0;
  return ['mlx', 'cuda'].flatMap(backend => ['base', 'sft'].flatMap(variant => ['single', 'four'].map(kind => {
    const task = inputs[`${kind}_case`];
    return { id: `quality-${backend}-${variant}-${kind}`, category: 'quality', backend, variant, transport: transport[index++], refs: task.references, request: { model: `moss_image_${variant}`, mode: 'edit_image', prompt: task.prompt, negativePrompt: '', width: task.width, height: task.height, count: 1, seed: 173, advanced: { steps: 30, guidanceScale: 4, schedulerShift: 5, mossImage: { acceleration: 'off', uncond_skip_layers: [] }, mossImageEdit: { reference_max_edge: 512, use_kv_cache: true } } }, anchors: kind === 'four' ? task.mandatory_preservation.concat(['Coherent image, no collage/duplicate limbs/objects', 'Correct ordinal left/right placement']) : ['Bottle body and cap deep red', 'Metallic silver collar and handle silhouette preserved', 'Droplets preserved', 'Both orange pieces and their placement preserved', 'Stone surface, lighting, shadows, framing and empty left space preserved', 'No added text/objects or global recolor'], expected: 'completed', output: 'images', outputCount: 1 };
  })));
}
export const LATENT_UPSCALE_GUARDS = ['latent-inline-upscale-real-esrgan', 'latent-inline-upscale-seedvr2'];
export function latentUpscaleGuard(name) {
  requireThat(LATENT_UPSCALE_GUARDS.includes(name), 'unknown latent inline upscale guard');
  return { refs: ['B'], request: { model: 'moss_image_sft', mode: 'edit_image', prompt: 'Make the bottle red.', width: 64, height: 64, count: 1, seed: 173, upscale: { enabled: true, engine: name.slice('latent-inline-upscale-'.length), factor: 2 }, advanced: { steps: 2, guidanceScale: 4, schedulerShift: 5, mossImage: { acceleration: 'off', uncond_skip_layers: [] }, mossImageEdit: { reference_max_edge: 32, use_kv_cache: true }, mossImageAdvanced: { output: 'latents' } } }, expectedStatus: 400, errorIncludes: ['MOSS latent output cannot use inline upscale', 'disable upscale or select images output'], noProviderLoadProbe: '' };
}
export function validateSeededRecipeSelection(row, request, history, origin, asset) {
  requireThat(history?.originRowId === row.replayOf && history.originJobId === origin.jobId && history.originSampleIndex === 2 && history.selectedAssetId === origin.outputs[2]?.assetId && history.selectedAssetId === asset.id && history.hydrationObserved === true && history.keepSeed === true, 'actual selected row-two asset/history/Keep seed binding required');
  requireThat(canonical(history.selectedRecipe) === canonical(asset.recipe), 'selected source recipe must remain the actual persisted row-two recipe');
  const recipe = asset.recipe, raw = recipe?.rawAdapterSettings;
  requireThat(recipe.prompt === request.prompts[1] && recipe.seed === request.seed + 2 && raw?.mossImageSampleIndex === 2 && canonical(raw.promptBatch) === canonical(request.prompts), 'source recipe lost row-two prompt/sample provenance or effective sample seed');
  requireThat(!raw.mossImageAdvanced?.noise && !request.advanced.mossImageAdvanced.noise && request.count === 2 && request.prompts.length === 2 && request.prompt === request.prompts[0] && request.seed === 173, 'seeded full-batch replay must restore prompt0/B2/count2/base seed without caller noise');
  requireThat(canonical(raw.referenceAssetIds) === canonical(request.referenceAssetIds) && canonical(raw.mossImageAdvanced) === canonical(request.advanced.mossImageAdvanced), 'seeded replay changed original ordered references, output or negative rows');
}
export function createPlan(inputs) {
  const rows = primaryRows(inputs);
  for (const backend of ['mlx', 'cuda']) {
    for (const variant of ['base', 'sft']) {
      const embeddingsNegative = (backend === 'mlx') === (variant === 'base');
      const advanced = { output: 'latents', positive: { values: '$positive', mask: '$mask' }, noise: '$noise', ...(embeddingsNegative ? { negative: { values: '$negative', mask: '$negativeMask' } } : { negativePrompts: ['blurred geometry', 'duplicated objects'] }) };
      const row = { id: `advanced-${backend}-${variant}`, category: 'advanced', backend, variant, transport: variant === 'base' ? 'rest' : 'mcp', refs: ['B', 'L'], request: { model: `moss_image_${variant}`, mode: 'edit_image', prompt: '', count: 2, width: 64, height: 64, seed: null, advanced: { steps: 2, guidanceScale: 4, schedulerShift: 5, mossImage: { acceleration: 'off', max_sequence_length: 3, training_shift: 2.5, uncond_skip_layers: [] }, mossImageEdit: { reference_max_edge: 32, use_kv_cache: true }, mossImageAdvanced: advanced } }, expected: 'completed', output: 'latents', outputCount: 4 };
      rows.push(row, { ...structuredClone(row), id: `${row.id}-replay`, category: 'replay', replayOf: row.id, action: variant === 'base' ? 'duplicate' : 'workflow', transport: variant === 'base' ? 'rest' : 'ui-full' });
    }
    const advancedCancel = structuredClone(rows.find(r => r.id === `advanced-${backend}-base`));
    Object.assign(advancedCancel,{id:`advanced-cancel-${backend}`,category:'advanced-cancel',transport:'rest',expected:'canceled',outputCount:0,cancelAtDenoise:true});
    advancedCancel.request.advanced.steps = 4;
    rows.push(advancedCancel,{...structuredClone(advancedCancel),id:`advanced-reentry-${backend}`,category:'advanced-reentry',expected:'completed',outputCount:4,cancelAtDenoise:false,action:'retry',replayOf:advancedCancel.id});
    const small = { model: 'moss_image_sft', mode: 'edit_image', prompt: inputs.single_case.prompt, count: 1, width: 64, height: 64, seed: 173, advanced: { steps: 4, guidanceScale: 4, schedulerShift: 5, mossImage: { acceleration: 'off', uncond_skip_layers: [] }, mossImageEdit: { reference_max_edge: 32, use_kv_cache: true } } };
    rows.push({ id: `cancel-${backend}`, category: 'cancel', backend, variant: 'sft', transport: 'rest', refs: ['B'], request: structuredClone(small), expected: 'canceled', outputCount: 0, cancelAtDenoise: true });
    for (const [kind, refs] of [['reentry', ['B']], ['changed', ['L']], ['reorder', ['L', 'B', 'P', 'G']], ['duplicates', ['B', 'L', 'B']]]) rows.push({ id: `${kind}-${backend}`, category: kind, backend, variant: 'sft', transport: 'rest', refs, request: structuredClone(small), expected: 'completed', output: 'images', outputCount: 1, ...(kind === 'reentry' ? { action: 'retry', replayOf: `cancel-${backend}` } : {}) });
    const cached = rows.find(r => r.id === `quality-${backend}-sft-single`);
    const uncached = structuredClone(cached); uncached.id = `cachefalse-${backend}`; uncached.category = 'cachefalse'; uncached.transport = 'rest'; uncached.compareTo = cached.id; uncached.request.advanced.mossImageEdit.use_kv_cache = false; delete uncached.anchors; rows.push(uncached);
    for (const mode of ['base', 'sft', 'unmerged_bf16', 'full_fp32']) {
      const ordinary = { model: `moss_image_${mode === 'base' ? 'base' : 'sft'}`, mode: 'text_to_image', prompt: 'A red ceramic bowl on a pale wooden table.', width: 64, height: 64, count: 1, seed: 173, advanced: { steps: 2, mossImage: { acceleration: ['base', 'sft'].includes(mode) ? 'off' : mode } } };
      if (['base', 'sft'].includes(mode)) Object.assign(ordinary.advanced, { guidanceScale: 4, schedulerShift: 5 });
      rows.push({ id: `ordinary-${backend}-${mode.replaceAll('_','-')}`, category: 'regression', backend, variant: mode === 'base' ? 'base' : 'sft', transport: 'rest', refs: [], request: ordinary, expected: 'completed', output: 'images', outputCount: 1 });
    }
    rows.push({ id: `return-edit-${backend}`, category: 'isolation', backend, variant: 'sft', transport: 'rest', refs: ['B'], request: structuredClone(small), expected: 'completed', output: 'images', outputCount: 1 });
  }
  // v3 appends these rows; every v2 row/request/rubric remains unchanged.
  for (const backend of ['mlx', 'cuda']) for (const output of ['images', 'latents']) {
    const variant = (backend === 'mlx') === (output === 'images') ? 'base' : 'sft';
    const prompts = ['Make the bottle deep red.', 'Make the bottle cobalt blue.'];
    const row = { id: `seeded-${backend}-${output}`, category: 'seeded-text', backend, variant, transport: 'rest', refs: ['B', 'L'], request: { model: `moss_image_${variant}`, mode: 'edit_image', prompt: prompts[0], prompts, count: 2, width: 64, height: 64, seed: 173, advanced: { steps: 2, guidanceScale: 4, schedulerShift: 5, mossImage: { acceleration: 'off', max_sequence_length: 128, training_shift: 2.5, uncond_skip_layers: [] }, mossImageEdit: { reference_max_edge: 32, use_kv_cache: true }, mossImageAdvanced: { output, negativePrompts: ['blurred geometry', 'duplicated objects'] } } }, expected: 'completed', output, outputCount: 4 };
    rows.push(row, { ...structuredClone(row), id: `${row.id}-replay`, category: 'seeded-text-replay', transport: 'ui-full', action: 'seeded-recipe', replayOf: row.id, originSampleIndex: 2 });
  }
  return { version: VERSION, state: 'DRAFT_REQUIRES_COORDINATOR_FREEZE', inputsSha256: digest(inputs), qualityScope: inputs.provenance, typographyScope: inputs.four_case.typography_scope, rows, renderExecutions: rows.length, requiredGuardCases: ['missing-adapter', 'corrupt-adapter', 'missing-reference', 'foreign-reference', 'corrupt-tensor', 'foreign-tensor', 'missing-tensor', 'embeddings-nonfinite', 'embeddings-shape', 'mask-nonbinary', 'mask-empty-row', 'sequence-over-limit', 'noise-shape', 'negative-batch-mismatch', 'negative-broadcast-conflict', 'positive-text-conflict', 'output-spelling', 'training-shift-zero', 'training-shift-nonfinite', 'host-budget-refusal', 'device-budget-before-load', 'edit-dmd', 'edit-slg', 'edit-skip', 'edit-strength', 'edit-lora', 'edit-dtype', ...LATENT_UPSCALE_GUARDS], requiredExternal: ['numeric-fixture-carry-forward', 'cache-invalidation-fixture-carry-forward', 'package-install-select-ui', 'advanced-controls-ui', 'history-hydration-ui', 'portable-workflow-reacquisition-ui', 'publication-cancel-error-cleanup', 'owned-cache-cleanup-reentry', 'memory-measurement', 'ordinary-dmd-isolation'], cacheComparisonPolicy: null };
}
export function validatePlan(plan, inputs) {
  requireThat(plan.version === VERSION && plan.inputsSha256 === digest(inputs), 'wrong protocol/input manifest');
  const required = primaryRows(inputs);
  for (const row of required) requireThat(canonical(plan.rows.find(r => r.id === row.id)) === canonical(row), `fixed primary quality cell changed: ${row.id}`);
  requireThat(plan.rows.length === plan.renderExecutions && new Set(plan.rows.map(r => r.id)).size === plan.rows.length, 'execution counts/unique rows disagree');
  for (const row of plan.rows) requireThat(/^[a-z0-9-]+$/.test(row.id) && ['mlx', 'cuda'].includes(row.backend) && row.request && row.expected && Number.isSafeInteger(row.outputCount), 'invalid execution row');
  const baseline = createPlan(inputs);
  requireThat(canonical(plan.rows) === canonical(baseline.rows), 'row allocation changes require a versioned protocol source change before freeze');
  requireThat(canonical(plan.requiredGuardCases) === canonical(baseline.requiredGuardCases) && canonical(plan.requiredExternal) === canonical(baseline.requiredExternal), 'required acceptance coverage removed');
  requireThat(plan.cacheComparisonPolicy?.metric === 'rgb8-mae/v1' && Number.isFinite(plan.cacheComparisonPolicy.tolerance) && plan.cacheComparisonPolicy.tolerance >= 0 && plan.cacheComparisonPolicy.tolerance <= 1 && plan.cacheComparisonPolicy.implementationSha256?.match(SHA), 'freeze source-owned rgb8-mae/v1, normalized tolerance and implementation before rendering');
}

export class Api {
  constructor(url, token = '') { this.base = new URL(url); requireThat(['http:', 'https:'].includes(this.base.protocol) && !this.base.username && !this.base.password, 'invalid API URL'); this.token = token; }
  async request(path, { method = 'GET', body, rawJson, observation, max = 32 * 1024 * 1024, raw = false } = {}) {
    requireThat(path.startsWith('/api/v1/'), 'only product API routes allowed');
    const headers = this.token ? { 'X-SceneWorks-Token': this.token } : {};
    requireThat(rawJson === undefined || (body === undefined && typeof rawJson === 'string' && Buffer.byteLength(rawJson) <= 1024 * 1024), 'raw JSON requires one bounded exact lexical body');
    if (rawJson !== undefined) { headers['Content-Type'] = 'application/json'; body = rawJson; }
    else if (body && !(body instanceof FormData)) { headers['Content-Type'] = 'application/json'; body = JSON.stringify(body); }
    const response = await fetch(new URL(path, this.base), { method, headers, body, redirect: 'error', signal: AbortSignal.timeout(60_000) });
    const chunks = []; let total = 0;
    for await (const chunk of response.body) { total += chunk.length; if (total > max) { await response.body.cancel().catch(() => {}); throw new Error('product response exceeded evidence bound'); } chunks.push(chunk); }
    const bytes = Buffer.concat(chunks);
    if(observation)Object.assign(observation,{status:response.status,contentType:response.headers.get('content-type'),byteLength:bytes.length,sha256:sha(bytes),body:bytes.toString('utf8')});
    if (!response.ok) { const error = new Error(`product API returned HTTP ${response.status}`); error.status = response.status; error.responseBody = bytes.toString('utf8'); throw error; }
    return raw ? bytes : JSON.parse(bytes.toString('utf8'));
  }
}

// Actual existing Streamable HTTP MCP service, including session negotiation and
// SSE progress. No tool implementation or alternate inference path lives here.
export class Mcp {
  constructor(url, token, onMessage) {
    this.next = 1; this.url = new URL('/mcp', url); this.token = token; this.onMessage = onMessage;
  }
  async send(message, timeout) {
    const headers = { Accept: 'application/json, text/event-stream', 'Content-Type': 'application/json' };
    if (this.token) headers['X-SceneWorks-Token'] = this.token;
    if (this.session) headers['Mcp-Session-Id'] = this.session;
    if (this.protocol) headers['MCP-Protocol-Version'] = this.protocol;
    const response = await fetch(this.url, { method: 'POST', headers, body: JSON.stringify(message), redirect: 'error', signal: AbortSignal.timeout(timeout) });
    requireThat(response.ok, `actual MCP returned HTTP ${response.status}`);
    this.session ??= response.headers.get('mcp-session-id');
    if ([202, 204].includes(response.status)) return;
    let answer;
    const consume = item => { this.onMessage?.(item); if (item.id === message.id) { requireThat(!item.error, 'MCP protocol error'); answer = item.result; } };
    if ((response.headers.get('content-type') ?? '').includes('application/json')) {
      const chunks = []; let length = 0; for await (const chunk of response.body) { length += chunk.length; requireThat(length <= 64 * 1024 * 1024, 'MCP JSON exceeds bound'); chunks.push(chunk); }
      consume(JSON.parse(Buffer.concat(chunks).toString('utf8')));
    } else {
      requireThat((response.headers.get('content-type') ?? '').includes('text/event-stream'), 'unsupported MCP response transport');
      let buffer = ''; const decoder = new TextDecoder();
      for await (const chunk of response.body) {
        buffer += decoder.decode(chunk, { stream: true }); requireThat(Buffer.byteLength(buffer) <= 64 * 1024 * 1024, 'MCP SSE event exceeds bound');
        let match;
        while ((match = /\r?\n\r?\n/.exec(buffer))) {
          const frame = buffer.slice(0, match.index); buffer = buffer.slice(match.index + match[0].length);
          const data = frame.split(/\r?\n/).filter(line => line.startsWith('data:')).map(line => line.slice(5).trimStart()).join('\n');
          if (data) consume(JSON.parse(data));
        }
        if (answer !== undefined) break;
      }
    }
    if (message.id !== undefined) requireThat(answer !== undefined, 'MCP response ended without matching request ID; reconcile any actual job before retry');
    return answer;
  }
  notify(method, params) { return this.send({ jsonrpc: '2.0', method, params }, 60_000); }
  call(method, params, timeout = 60_000) { return this.send({ jsonrpc: '2.0', id: this.next++, method, params }, timeout); }
  async initialize() { const result = await this.call('initialize', { protocolVersion: '2025-03-26', capabilities: {}, clientInfo: { name: 'sc-26757-terminal', version: '1' } }); this.protocol = result.protocolVersion; await this.notify('notifications/initialized', {}); }
  close() {} // No process or resource ownership to release. The product owns its session.
}
export function mcpArgs(request) {
  const { advanced = {}, ...args } = structuredClone(request);
  for (const key of ['steps', 'guidanceScale', 'schedulerShift', 'sampler', 'cfgInterval', 'mossImage', 'mossImageEdit', 'mossImageAdvanced']) if (Object.hasOwn(advanced, key)) args[key] = advanced[key];
  return { ...args, includeWorkflow: true };
}
export function mcpJson(result) {
  requireThat(!result.isError, 'MCP tool returned an error');
  const parsed = (result.content ?? []).filter(c => c.type === 'text').map(c => { try { return JSON.parse(c.text); } catch { return null; } }).filter(Boolean);
  return parsed.find(v => v.jobId) ?? result.structuredContent ?? null;
}
export function replaceInputs(row, uploaded, projectId) {
  const request = structuredClone(row.request); request.projectId = projectId;
  if (row.refs.length) request.referenceAssetIds = row.refs.map(id => { requireThat(uploaded[id]?.id, `missing uploaded ${id}`); return uploaded[id].id; });
  const replace = value => {
    if (typeof value === 'string' && value.startsWith('$')) { const asset = uploaded[value.slice(1)]; requireThat(asset?.type === 'tensor' && asset.tensor?.sha256, 'missing authorized tensor'); return { assetId: asset.id, sha256: asset.tensor.sha256 }; }
    if (Array.isArray(value)) return value.map(replace);
    if (value && typeof value === 'object') return Object.fromEntries(Object.entries(value).map(([k, v]) => [k, replace(v)]));
    return value;
  };
  return replace(request);
}
export function assertRequestBinding(expected, actual) {
  const matches = (wanted, observed) => wanted && typeof wanted === 'object' && !Array.isArray(wanted) ? observed && typeof observed === 'object' && Object.entries(wanted).every(([key, value]) => matches(value, observed[key])) : canonical(wanted) === canonical(observed);
  for (const [key, value] of Object.entries(expected)) requireThat(matches(value, actual[key]), `actual job changed frozen request field ${key}`);
  // The API persists an omitted broadcast prompt as "". Only a nonempty
  // broadcast competes with the authored per-prompt negatives.
  if (expected.advanced?.mossImageAdvanced?.negativePrompts) requireThat(!Object.hasOwn(actual,'negativePrompt') || actual.negativePrompt === '', 'per-prompt negatives gained an unsupported broadcast field');
  requireThat(!actual.advanced?.mossImageEditExecutionReport && !actual.advanced?.mossImageExecutionReport && actual.advanced?.mossImageSampleIndex === undefined, 'request imported an execution receipt/sample index');
}
// The Editor stages its unchanged working raster as a new project asset. Only this
// allocated row may bind that observed upload to the frozen original first slot.
// Keep the actual POST/job and origin history intact; never normalize their IDs.
export function validateEditorSource({ row, expected, capture, intent, original, uploadBytes, servedBytes, asset, origin, originJob, historyJob, originAsset, historyAsset }) {
  requireThat(row.id === 'quality-cuda-sft-single' && row.transport === 'ui-editor' && canonical(row.refs) === canonical(['B']), 'scratch source substitution is restricted to the allocated Editor row');
  const upload = capture.sourceUpload, actual = capture.network?.request, history = capture.history;
  requireThat(capture.version === VERSION && capture.rowId === row.id && capture.observer && SHA.test(intent.lockSha256 ?? '') && SHA.test(intent.bindingSha256 ?? '') && capture.lockSha256 === intent.lockSha256 && capture.bindingSha256 === intent.bindingSha256 && capture.network.method === 'POST' && capture.network.path === '/api/v1/image/jobs' && capture.network.response?.id, 'Editor POST/capture differs from the actual reserved candidate/backend intent');
  requireThat(upload?.inputId === 'B' && upload.network?.method === 'POST' && upload.network.path === `/api/v1/projects/${encodeURIComponent(expected.projectId)}/assets` && upload.network.response?.id === asset.id, 'missing actual authorized Editor upload request/response');
  requireThat(asset.projectId === expected.projectId && asset.type === 'image' && asset.file?.path && asset.id !== expected.referenceAssetIds[0] && !asset.lineage?.jobId, 'Editor scratch must be a newly uploaded project raster, not a generated output');
  const wire=upload.network.request,exactWire=upload.byteObserver==='playwright-request-post-data'&&wire?.observer===upload.byteObserver&&wire.fileSha256===upload.sha256;
  const assetReadback=upload.byteObserver==='authenticated-product-asset-readback'&&wire?.observer==='playwright-request-metadata-and-product-response'&&wire.fileSha256===null&&wire.postDataState==='file-bytes-unavailable-in-observer'&&wire.contentType==='multipart/form-data';
  requireThat(upload.sha256===sha(uploadBytes)&&sha(servedBytes)===upload.sha256&&(exactWire||assetReadback),'Editor upload/served bytes differ from the actual observed request or authenticated asset readback');
  const identity = pngRgbIdentity(uploadBytes, true);
  requireThat(canonical(identity) === canonical({ width:original.width,height:original.height,pixelsSha256:original.pixelsSha256 }) && canonical(pngRgbIdentity(servedBytes,true)) === canonical(identity), 'Editor scratch changed frozen original RGB content or geometry');
  const times = [intent.startedAt, upload.observedAt, capture.network.observedAt, capture.capturedAt].map(Date.parse);
  requireThat(times.every(Number.isFinite) && times.every((time,index) => index === 0 || time >= times[index-1]) && Date.parse(asset.createdAt) >= times[0] && Date.parse(asset.createdAt) <= times[2], 'Editor upload/job chronology is not bound to the reserved UI intent');
  requireThat(actual?.projectId === expected.projectId && canonical(actual.referenceAssetIds) === canonical([asset.id]) && (!actual.sourceAssetId || actual.sourceAssetId === asset.id) && (!actual.referenceAssetId || actual.referenceAssetId === asset.id), 'Editor POST changed effective original reference slots');
  const bound = structuredClone(expected); bound.referenceAssetIds = [asset.id];
  assertRequestBinding(bound, actual);
  requireThat(history?.originJobId === origin.jobId && history.originAssetId === origin.outputs[0].assetId && history.originSampleIndex === 0 && history.hydrationObserved === true, 'Editor must hydrate the actual allocated MLX history job/asset/sample');
  requireThat(originJob.id === origin.jobId && historyJob.id === origin.jobId && historyJob.status === 'completed' && historyJob.projectId === originJob.projectId && canonical(historyJob.payload) === canonical(originJob.payload) && canonical(historyJob.result) === canonical(originJob.result), 'authenticated Editor history read-back differs from the actual original job');
  requireThat(originAsset.id === history.originAssetId && historyAsset.id === originAsset.id && historyAsset.projectId === originAsset.projectId && canonical(historyAsset.recipe) === canonical(originAsset.recipe) && historyAsset.lineage?.jobId === origin.jobId, 'authenticated original history asset/recipe identity changed');
  requireThat(canonical(history.selectedRecipe) === canonical(originAsset.recipe), 'Editor hydration did not retain the actual origin recipe');
  return { inputId:'B', assetId:asset.id, uploadSha256:upload.sha256, ...identity, originJobId:origin.jobId, originAssetId:originAsset.id, actualRequestSha256:digest(actual), semanticRequestSha256:digest(expected) };
}
export function checkJob(row, request, job, workerGpuId = row.backend === 'cuda' ? '1' : 'mlx') {
  requireThat(job.id && job.projectId === request.projectId, 'job/project identity mismatch');
  assertRequestBinding(request, job.payload);
  // JobsStore keeps the first backend label. Pin progress uses worker settings.gpu_id;
  // generation events use mlx/candle. Only the native report proves execution_device.
  requireThat(typeof workerGpuId === 'string' && workerGpuId && (row.backend !== 'cuda' || workerGpuId === '1') && [row.backend === 'cuda' ? 'candle' : 'mlx', workerGpuId].includes(job.backend), 'queue backend label differs from the frozen worker launch contract');
  requireThat(job.status === row.expected, `expected ${row.expected}, got ${job.status}`);
  requireThat(job.status !== 'completed' || job.result?.outputPublication?.state !== 'pending', 'completed output publication still pending');
  const assets = job.result?.assets ?? [];
  requireThat(assets.length === row.outputCount, 'canonical published output count mismatch');
  if (row.expected !== 'completed') { requireThat(!(job.result?.assetWrites?.length)&&!job.result?.mossImageEditExecutionReport && !JSON.stringify(job.result).includes('mossImageEditExecutionReport'), 'canceled/error job published assets or an execution report'); return; }
  requireThat(job.workerId && job.startedAt && job.completedAt, 'no actual worker execution timestamps/identity');
  requireThat(new Set(assets.map(a => a.id)).size === assets.length && assets.every(a => a.id), 'duplicate/missing published asset IDs');
}
export function isDenoiseStep(job) {
  const step = /(?:^|\s)step ([0-9]+)\/([0-9]+)(?:\.|\s|$)/.exec(job.message ?? '');
  return job.status === 'running' && job.stage === 'generating' && Number(job.progress) > 0 && !!step && Number(step[1]) > 0 && Number(step[1]) < Number(step[2]);
}
export function sanitizedReport(report, edit = true) {
  const result = structuredClone(report);
  if (edit) { result.adapter_path = 'adapter_model.safetensors'; result.base = sanitizedReport(result.base,false); }
  else for (const adapter of result.adapter_reports ?? []) adapter.adapter_path = 'lora.safetensors';
  return result;
}
export function validateExportedWorkflow(workflow, report, sampleIndex) {
  requireThat(workflow && workflow.sceneworksWorkflow && workflow.mode === 'edit_image' && workflow.advanced, 'invalid actual exported workflow envelope');
  requireThat(canonical(workflow.advanced.mossImageEditExecutionReport) === canonical(sanitizedReport(report)), 'portable workflow provenance differs from the original sanitized native report');
  if (workflow.advanced.mossImageExecutionReport) requireThat(canonical(workflow.advanced.mossImageExecutionReport) === canonical(sanitizedReport(report.base,false)), 'portable backbone provenance mismatch');
  requireThat(workflow.advanced.mossImageSampleIndex === sampleIndex, 'portable workflow sample provenance mismatch');
}
export function mcpResultAssets(result, jobId, outputs, origin) {
  const summary = mcpJson(result);
  requireThat(summary?.jobId === jobId && summary.status === 'completed' && summary.assets?.length === outputs.length, 'MCP summary is not the actual complete asset set');
  const links = (result.content ?? []).filter(c => c.type === 'resource_link');
  const expectedUris = [];
  const checkUrl = url => { const parsed = new URL(url); requireThat(parsed.origin === origin && parsed.pathname.startsWith('/api/v1/projects/'), 'MCP resource is outside the actual product API'); return parsed.href; };
  const matched = outputs.map(output => {
    const asset = summary.assets.find(a => a.id === output.assetId);
    requireThat(asset && asset.type === 'tensor' && asset.tensor?.sha256 === output.sha256 && asset.mimeType === 'application/x-npy', 'MCP tensor asset ID/media identity mismatch');
    requireThat(output.tensor && ['dtype','shape','byteLength'].every(key => canonical(asset.tensor[key]) === canonical(output.tensor[key])), 'MCP tensor media metadata differs from actual REST artifact');
    const mediaUrl = checkUrl(asset.url), companion = asset.workflowCompanion;
    requireThat(companion?.sha256?.match(SHA) && companion.url, 'missing actual tensor workflow companion');
    const companionUrl = checkUrl(companion.url);
    requireThat(mediaUrl !== companionUrl && [mediaUrl,companionUrl].every(uri => links.filter(l => l.uri === uri).length === 1), 'MCP media/companion link cardinality mismatch');
    expectedUris.push(mediaUrl,companionUrl);
    return { assetId:output.assetId, mediaUrl, companionUrl, companionSha256:companion.sha256, mimeType:asset.mimeType };
  });
  requireThat(new Set(summary.assets.map(a => a.id)).size === outputs.length && links.length === expectedUris.length && links.every(l => expectedUris.includes(l.uri)), 'unexpected/duplicate MCP result resources');
  return matched;
}
export function rgbMae(left, right) {
  const a = decodeRgbPng(left), b = decodeRgbPng(right);
  requireThat(a.width === b.width && a.height === b.height, 'cache metric image geometry mismatch');
  let sum = 0; for (let i = 0; i < a.pixels.length; i++) sum += Math.abs(a.pixels[i]-b.pixels[i]);
  return {value:sum/(a.pixels.length*255),pixelSha256:[sha(a.pixels),sha(b.pixels)],width:a.width,height:a.height};
}
export function cacheComparison(policy, lockSha256, pair, buffers) {
  requireThat(policy.metric === 'rgb8-mae/v1' && pair.length === 2 && buffers.length === 2 && pair.every((item,i) => item.outputSha256 === sha(buffers[i]) && item.rowId && item.jobId && SHA.test(item.requestSha256)), 'cache metric input provenance mismatch');
  const measured = rgbMae(...buffers);
  return {version:VERSION,lockSha256,policySha256:digest(policy),metric:policy.metric,implementationSha256:policy.implementationSha256,pair,...measured};
}

// Every possible submission has immutable start + resolution records. A failed or
// crashed attempt stops later dispatch on that backend until explicit reconciliation.
const optionalJson = async path => json(path).catch(error => { if (error.code === 'ENOENT') return null; throw error; });
export async function assertDispatchClear(root, backend) {
  const directory = resolve(root,'attempts',backend);
  const names = await readdir(directory).catch(error => { if (error.code === 'ENOENT') return []; throw error; });
  for (const name of names.filter(n => n.endsWith('.start.json'))) {
    const start = await json(resolve(directory,name)), stem = name.slice(0,-'.start.json'.length);
    const resolution = await optionalJson(resolve(directory,`${stem}.resolution.json`));
    const parserProof=resolution?.state==='SOURCE_PARSER_REFUSAL'&&backend==='mlx'&&stem==='guard-training-shift-nonfinite'?await optionalJson(resolve(directory,`${stem}.reconciliation.json`)):null;
    const parserFailure=parserProof?await optionalJson(resolve(directory,`${stem}.failure.json`)):null;
    const parserValid=parserProof?.version===VERSION&&parserProof.kind==='source-parser-refusal-before-dispatch'&&parserProof.backend===backend&&parserProof.case==='training-shift-nonfinite'&&parserProof.attemptSha256===digest(start)&&parserProof.lockSha256===start.lockSha256&&parserProof.bindingSha256===start.bindingSha256&&parserProof.jobId===null&&parserFailure?.jobId===null&&parserFailure?.httpStatus===422&&parserProof.failureSha256===digest(parserFailure)&&resolution.proofSha256===digest(parserProof);
    const correctionProof=resolution?.state==='SOURCE_GUARD_INPUT_CORRECTION'&&backend==='mlx'&&stem==='guard-edit-dmd'?await optionalJson(resolve(directory,`${stem}.reconciliation.json`)):null;
    const correctionFailure=correctionProof?await optionalJson(resolve(directory,`${stem}.failure.json`)):null;
    const correctionValid=correctionProof?.version===VERSION&&correctionProof.kind==='source-guard-input-correction-before-dispatch'&&correctionProof.backend===backend&&correctionProof.case==='edit-dmd'&&correctionProof.attemptSha256===digest(start)&&correctionProof.lockSha256===start.lockSha256&&correctionProof.bindingSha256===start.bindingSha256&&correctionProof.jobId===null&&correctionFailure?.jobId===null&&correctionFailure?.httpStatus===400&&correctionProof.failureSha256===digest(correctionFailure)&&resolution.proofSha256===digest(correctionProof);
    if (!(resolution?.attemptSha256 === digest(start) && (['TERMINAL_VERIFIED','EXPECTED_REFUSAL','UPLOAD_COMPLETE','COORDINATOR_RECONCILED'].includes(resolution.state)||parserValid||correctionValid))) {
      const error = new Error(`unresolved ${backend} submission attempt ${stem}; authoritative reconciliation required`);
      error.code='MOSS_AWAITING_RESOLUTION';throw error;
    }
  }
}
export async function reserveAttempt(root, backend, id, fields) {
  requireThat(/^[a-z0-9-]+$/.test(id), 'unsafe attempt ID');
  const directory = resolve(root,'attempts',backend); await mkdir(directory,{recursive:true,mode:0o700});
  const mutex = resolve(directory,'dispatch.lock');
  await writeFile(mutex,id,{flag:'wx',mode:0o600});
  try {
    await assertDispatchClear(root,backend);
    const start = {version:VERSION,backend,id,startedAt:new Date().toISOString(),...fields};
    await save(resolve(directory,`${id}.start.json`),start);
    return {directory,mutex,id,start};
  } catch(error) { await unlink(mutex); throw error; }
}
export async function resumeUiAttempt(root,backend,id,fields) {
  const directory=resolve(root,'attempts',backend),mutex=resolve(directory,'dispatch.lock'),start=await json(resolve(directory,`${id}.start.json`));
  requireThat(start.transport === 'ui' && start.backend === backend && start.id === id && start.lockSha256 === fields.lockSha256 && start.bindingSha256 === fields.bindingSha256 && start.requestSha256 === fields.requestSha256 && await readFile(mutex,'utf8') === id,'UI adoption does not own the reserved exact request');
  requireThat(!await optionalJson(resolve(directory,`${id}.resolution.json`)) && !await optionalJson(resolve(directory,`${id}.failure.json`)),'UI attempt already resolved/failed; never submit again');
  return {directory,mutex,id,start};
}
async function releaseAttempt(ticket) { requireThat(await readFile(ticket.mutex,'utf8') === ticket.id,'dispatch lock ownership changed'); await unlink(ticket.mutex); }
export async function resolveAttempt(ticket,state,extra={}) {
  requireThat(['TERMINAL_VERIFIED','EXPECTED_REFUSAL','UPLOAD_COMPLETE'].includes(state),'invalid automatic attempt resolution');
  await save(resolve(ticket.directory,`${ticket.id}.resolution.json`),{version:VERSION,attemptSha256:digest(ticket.start),state,...extra}); await releaseAttempt(ticket);
}
export async function failAttempt(ticket,error,jobId=null) {
  await save(resolve(ticket.directory,`${ticket.id}.failure.json`),{version:VERSION,attemptSha256:digest(ticket.start),state:'UNRESOLVED',jobId,message:error.message,httpStatus:error.status??null,observedAt:new Date().toISOString()});
  await releaseAttempt(ticket);
}
export async function reconcileAttempt(root,backend,id,proof,lockSha256) {
  const directory = resolve(root,'attempts',backend), start = await json(resolve(directory,`${id}.start.json`)), failure = await optionalJson(resolve(directory,`${id}.failure.json`));
  requireThat(proof.version === VERSION && proof.lockSha256 === lockSha256 && proof.attemptSha256 === digest(start) && proof.coordinator && proof.backend === backend && proof.observedAt && Date.parse(proof.observedAt) >= Date.parse(start.startedAt) && proof.state === 'NO_ACTIVE_OWNED_JOB' && proof.rawEvidence?.length && Array.isArray(proof.jobs), 'invalid authoritative attempt reconciliation');
  requireThat(proof.jobs.every(job => job.id && ['completed','failed','canceled','interrupted'].includes(job.status) && (job.status !== 'completed' || job.result?.outputPublication?.state !== 'pending')), 'reconciliation still has an active/unpublished completed job');
  if (failure?.jobId) requireThat(proof.jobs.some(j => j.id === failure.jobId),'known unresolved job was not reconciled');
  else requireThat(proof.authoritativeSearch?.requestSha256 === start.requestSha256 && proof.authoritativeSearch.since === start.startedAt && proof.authoritativeSearch.complete === true,'unknown submission requires complete actual request/time-bound authoritative search');
  for (const item of proof.rawEvidence) requireThat((await fileIdentity(safePath(root,item.file))).sha256 === item.sha256,'reconciliation raw evidence changed');
  await save(resolve(directory,`${id}.reconciliation.json`),proof);
  await save(resolve(directory,`${id}.resolution.json`),{version:VERSION,attemptSha256:digest(start),state:'COORDINATOR_RECONCILED',proofSha256:digest(proof)});
  const mutex = resolve(directory,'dispatch.lock'); const owner = await readFile(mutex,'utf8').catch(error => { if(error.code === 'ENOENT') return null; throw error; });
  if (owner === id) await unlink(mutex); else requireThat(owner === null,'another attempt owns the dispatch lock');
}
export function findEditReport(asset, job) {
  return asset.recipe?.rawAdapterSettings?.mossImageEditExecutionReport ?? asset.rawAdapterSettings?.mossImageEditExecutionReport ?? job.result?.mossImageEditExecutionReport ?? null;
}
export function validateEditReport(row, request, report, originals) {
  requireThat(report && report.base && Array.isArray(report.references), 'missing actual native edit provider report');
  requireThat(report.adapter_repository === 'chinoll/Moss-Image-Edit' && report.adapter_revision === '3be9c40037e1b89b17aa85ff6f8ee805872aa558' && report.adapter_sha256 === '17a48b8c605dcffdedb92a2b887d18cf316505be90802717669fb6556098dcdf' && report.config_sha256 === 'b6b471cf7bc77b0874af03be618f8a06b6997d92b35de12c47a06b292ade2669' && report.applied_projections === 252 && report.edit_contract === 'ref-prefix-only-t0__target-text-original-moss-v1', 'provider did not use the exact published edit adapter');
  requireThat(report.references.length === row.refs.length, 'report reference cardinality mismatch');
  requireThat(report.reference_max_edge === request.advanced.mossImageEdit.reference_max_edge && report.use_kv_cache === request.advanced.mossImageEdit.use_kv_cache, 'effective reference/cache controls differ');
  const base = report.base.data ?? report.base;
  requireThat(base.execution_device === row.backend && base.resolved?.variant === row.variant && base.resolved.steps === request.advanced.steps && base.resolved.guidance === request.advanced.guidanceScale && base.resolved.shift === request.advanced.schedulerShift && base.resolved.training_shift === (request.advanced.mossImage.training_shift ?? 1) && base.resolved.max_sequence_length === (request.advanced.mossImage.max_sequence_length ?? 4096), 'native effective variant/device/scalars differ from authored request');
  if (originals) for (const [index, ref] of report.references.entries()) {
    const original = originals[row.refs[index]], edge = request.advanced.mossImageEdit.reference_max_edge, limit = Math.floor(edge/16)*16, scale = Math.min(1,limit/Math.max(original.width,original.height));
    const align = n => Math.min(limit,Math.max(16,Math.floor(n*scale/16+0.5)*16));
    requireThat(ref.pixels_sha256 === original.pixelsSha256 && canonical(ref.original_dimensions) === canonical([original.width,original.height]) && canonical(ref.encoded_dimensions) === canonical([align(original.width),align(original.height)]) && SHA.test(ref.resized_pixels_sha256) && SHA.test(ref.latent_sha256), 'ordered original/reference resize/latent identity mismatch');
  }
  requireThat(base.output_kind === row.output && canonical(base.latent_shape) === canonical([row.outputCount, 128, request.height / 16, request.width / 16]), 'provider output kind/shape mismatch');
  requireThat(base.samples?.length === row.outputCount, 'provider sample count mismatch');
  if (request.advanced.mossImageAdvanced?.noise) requireThat(base.samples.every(s => s.seed === null && SHA.test(s.noise_sha256 ?? '')), 'caller noise must retain seed:null and native noise digests');
  else requireThat(base.samples.every((s,i) => s.seed === request.seed + i && SHA.test(s.noise_sha256 ?? '')), 'generated native seed/noise receipt differs from frozen seed policy');
  for (const [i, sample] of base.samples.entries()) requireThat(sample.index === undefined || sample.index === i, 'provider sample order mismatch');
  return base;
}
export function memorySummary(samples, jobId) {
  requireThat(samples.version === VERSION && samples.jobId === jobId && SHA.test(samples.samplerSha256 ?? '') && samples.device?.uuid, 'unbound memory sampler evidence');
  const entries = samples.samples;
  requireThat(Array.isArray(entries) && entries.length >= 3 && entries.every(s => Number.isFinite(s.usedBytes) && s.usedBytes >= 0 && Number.isFinite(Date.parse(s.at))), 'invalid whole-device memory samples');
  for (const phase of ['baseline', 'active', 'drain']) requireThat(entries.some(s => s.phase === phase), `missing memory ${phase}`);
  requireThat(entries.every((s,i) => i === 0 || Date.parse(entries[i-1].at) < Date.parse(s.at)) && entries[0].phase === 'baseline' && entries.at(-1).phase === 'drain', 'memory sample time/phase ordering invalid');
  return { kind: 'sampled-whole-device-usage-not-admission-estimate', baselineBytes: entries.find(s => s.phase === 'baseline').usedBytes, sampledPeakBytes: Math.max(...entries.map(s => s.usedBytes)), drainBytes: entries.findLast(s => s.phase === 'drain').usedBytes, sampleCount: entries.length, samplerSha256: samples.samplerSha256 };
}
export function reviewVerdict(row, review, outputs) {
  requireThat(review.version === VERSION && review.rowId === row.id && review.reviewer && review.reviewedAt && review.outputSha256?.length === outputs.length && canonical(review.outputSha256) === canonical(outputs.map(o => o.sha256)), 'quality review not bound to actual outputs');
  requireThat(Array.isArray(review.anchors) && canonical(review.anchors.map(a => a.name)) === canonical(row.anchors), 'quality review omitted/changed a frozen anchor');
  requireThat(review.anchors.every(a => ['PASS', 'FAIL'].includes(a.verdict) && a.observation), 'every visual anchor needs explicit observation and PASS/FAIL');
  return review.anchors.every(a => a.verdict === 'PASS') ? 'PASS' : 'FAIL';
}
