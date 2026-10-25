#!/usr/bin/env node
// sc-26691: explicit, source-bound terminal execution. No ordinary CI/model execution.
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { createReadStream, createWriteStream } from 'node:fs';
import { mkdir, readFile, realpath, rename, stat, writeFile, copyFile } from 'node:fs/promises';
import { spawn } from 'node:child_process';
import { promisify } from 'node:util';
import { execFile } from 'node:child_process';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
const exec = promisify(execFile);
export const SOURCE = { repo: 'WarmBloodAban/Minimax-h3_Singularity', revision: 'a56850c870825d6a8ae501a595c4dbbbc77bbc7e' };
export const ARTIFACTS = [
  'Minimax-h3_Singularity_ref2va_v2_int8.safetensors',
  'Minimax-h3_Singularity_ref2va_v1.3_int8.safetensors',
  'Minimax-h3_Singularity_ref2va_Pruned_v1.3_int8.safetensors',
  'Minimax-h3_Singularity_ref2va_v1.3_Pruned_w4a8.safetensors',
];
// Mechanism coverage, not a claim that every checkpoint supports every operation.
export const SURFACE = {
  numeric: ['tensor-mapping', 'convrot', 'pruned-interpolation', 'w4a8'],
  capability: ['text', 'image', 'first-last-frame', 'last-frame-only', 'video', 'ordered-multimodal', 'audio-reference', 'hybrid', 'turbo', 'lora', 'lokr'],
  generalization: ['renamed', 'independent-finetune', 'native-single', 'sharded-diffusers', 'dense-f16', 'dense-bf16', 'dense-f32', 'existing-families', 'ltx-add-select-generate', 'compatible-base'],
  lifecycle: ['managed-linked-equivalence', 'restart-replay', 'source-drift', 'interruption-retry', 'removal', 'cancel', 'generation-source-prep-cancel', 'relink-rescan', 'leases', 'duplicate', 'bounded-loading', 'cold-warm-staged', 'provenance-license'],
};
export function requiredCases() {
  return ['mlx', 'cuda'].flatMap(backend => [
    ...ARTIFACTS.map((artifact, i) => ({ id: `${backend}.artifact.${i}`, backend, group: 'artifact', artifact })),
    ...Object.entries(SURFACE).flatMap(([group, checks]) => checks.map(check => ({ id: `${backend}.${group}.${check}`, backend, group }))),
  ]);
}
const absolutePath = v => typeof v === 'string' && (path.posix.isAbsolute(v) || path.win32.isAbsolute(v));
const sha40 = v => assert.match(v ?? '', /^[a-f0-9]{40}$/);
const sha64 = v => assert.match(v ?? '', /^[a-f0-9]{64}$/);
const positive = v => assert.ok(Number.isFinite(v) && v > 0, 'positive measured value required');
const json = async p => JSON.parse(await readFile(p, 'utf8'));
// Final delivery links arrive after hardware execution; they are not execution inputs.
const digest = ({ delivery: _delivery, ...inputs }) => createHash('sha256').update(JSON.stringify(inputs)).digest('hex');
export const executionPlanDigest = digest;
export async function hashFile(p) {
  const h = createHash('sha256');
  for await (const chunk of createReadStream(p)) h.update(chunk);
  return h.digest('hex');
}
async function save(p, value) {
  await writeFile(`${p}.tmp`, JSON.stringify(value, null, 2) + '\n', { flag: 'w' });
  await rename(`${p}.tmp`, p);
}
export function validatePlan(p) {
  assert.equal(p.version, 1);
  sha40(p.sceneworks); sha40(p.inference);
  assert.deepEqual(p.source, SOURCE);
  assert.ok(absolutePath(p.output), 'external absolute output required');
  assert.ok(p.responsibleTask === 'sc-26691');
  assert.ok(typeof p.nextAuthorizedAction === 'string' && p.nextAuthorizedAction.length > 10);
  assert.equal(p.cases.length, requiredCases().length, 'complete acceptance inventory required');
  assert.equal(new Set(p.cases.map(c => c.id)).size, p.cases.length);
  for (const expected of requiredCases()) {
    const c = p.cases.find(c => c.id === expected.id);
    for (const [k, v] of Object.entries(expected)) assert.equal(c?.[k], v, `missing/mismatched ${expected.id}.${k}`);
    assert.ok(Number.isInteger(c.timeoutSeconds) && c.timeoutSeconds > 0 && c.timeoutSeconds <= 7200);
    assert.ok(Array.isArray(c.command) && c.command.length > 0 && c.command.every(s => typeof s === 'string'));
    assert.ok(Array.isArray(c.inputs) && c.inputs.length > 0);
    for (const input of c.inputs) { sha64(input.sha256); assert.ok(absolutePath(input.path)); }
    assert.ok(typeof c.evidence === 'string' && c.evidence.length > 0);
    positive(c.loadHostBudgetBytes); positive(c.loadDeviceBudgetBytes);
    if (c.runtime?.scenario) {
      assert.ok(absolutePath(c.runtime.scenario) && c.inputs.some(i => i.path === c.runtime.scenario && i.role === 'scenario'), 'actual bound API scenario required');
      assert.ok(Number.isInteger(c.runtime.port) && c.runtime.port > 1024 && c.runtime.port <= 65535 && typeof c.runtime.gpuId === 'string', 'owned service port/device configuration required');
    }
    if (c.group === 'numeric' && c.numericProof?.kind === 'native-pinned-fixture') {
      sha40(c.referenceRevision);
      assert.ok(c.numericProof.tests?.length && c.inputs.some(i => i.role === 'numeric-fixture') && c.inputs.some(i => i.role === 'reference'), 'actual native comparisons and pinned fixture/reference inputs required');
    } else if (c.group === 'numeric') {
      sha40(c.referenceRevision); compareNumeric([0], [0], c.tolerance);
      assert.ok(Array.isArray(c.referenceCommand) && c.referenceCommand.length > 0 && c.referenceCommand.every(s => typeof s === 'string'));
      assert.ok(typeof c.referenceOutput === 'string' && c.referenceOutput.length > 0);
    }
    if (c.group === 'capability') { positive(c.minimumControlDelta); assert.ok(['video', 'audio'].includes(c.controlStream)); }
  }
  return p;
}
export async function sourcePair(p, root, inferenceRoot) {
  for (const [dir, revision] of [[root, p.sceneworks], [inferenceRoot, p.inference]]) {
    assert.equal((await exec('git', ['rev-parse', 'HEAD'], { cwd: dir })).stdout.trim(), revision, 'source revision drift');
    assert.equal((await exec('git', ['status', '--porcelain'], { cwd: dir })).stdout.trim(), '', 'source must stay clean');
  }
  const cargo = await readFile(path.join(root, 'crates/sceneworks-worker/Cargo.toml'), 'utf8');
  assert.ok(cargo.includes(`rev = "${p.inference}"`), 'permanent paired inference pin required');
  await exec('git', ['merge-base', '--is-ancestor', p.inference, 'origin/main'], { cwd: inferenceRoot });
  const out = await realpath(p.output);
  for (const dir of [root, inferenceRoot]) {
    const rel = path.relative(await realpath(dir), out);
    assert.ok(rel.startsWith(`..${path.sep}`) || path.isAbsolute(rel), 'outputs must be outside source trees');
  }
}
export async function verifyInputs(c, allowedRoots) {
  const roots = allowedRoots ? await Promise.all(allowedRoots.map(async root => ({ logical: path.resolve(root), actual: await realpath(root) }))) : undefined;
  const below = (root, file) => { const rel = path.relative(root, file); return rel && !rel.startsWith('..') && !path.isAbsolute(rel); };
  for (const input of c.inputs) {
    if (roots) { const candidate = path.resolve(input.path); assert.ok(roots.some(r => below(r.logical, candidate) || below(r.actual, candidate)), 'input outside task staging or paired source roots'); const actual = await realpath(candidate); assert.ok(roots.some(r => below(r.actual, actual)), 'input symlink escapes task/source roots'); }
    assert.equal(await hashFile(input.path), input.sha256, `content changed: ${input.path}`);
  }
}
export function verifyFrozenArtifact(c, catalog) {
  assert.equal(catalog.sha, SOURCE.revision, 'wrong frozen HF revision');
  const file = catalog.siblings?.find(s => s.rfilename === c.artifact);
  sha64(file?.lfs?.sha256);
  const input = c.inputs.find(i => i.role === 'checkpoint' && i.artifact === c.artifact);
  assert.equal(input?.sha256, file.lfs.sha256, 'checkpoint is not the frozen source artifact');
  return file;
}
async function fetchFrozenCatalog(c) {
  const url = `https://huggingface.co/api/models/${SOURCE.repo}/revision/${SOURCE.revision}?blobs=true`;
  const r = await fetch(url, { signal: AbortSignal.timeout(30000) });
  assert.ok(r.ok, `frozen source metadata HTTP ${r.status}; no substitution permitted`);
  const catalog = await r.json(); verifyFrozenArtifact(c, catalog); return catalog;
}
export function compareNumeric(candidate, reference, tolerance) {
  assert.ok(Array.isArray(candidate) && candidate.length > 0 && candidate.length === reference.length);
  assert.ok(Number.isFinite(tolerance.atol) && tolerance.atol >= 0 && Number.isFinite(tolerance.rtol) && tolerance.rtol >= 0);
  let maxAbs = 0;
  candidate.forEach((v, i) => {
    const r = reference[i];
    assert.ok(Number.isFinite(v) && Number.isFinite(r));
    maxAbs = Math.max(maxAbs, Math.abs(v - r));
    assert.ok(Math.abs(v - r) <= tolerance.atol + tolerance.rtol * Math.abs(r), `numeric mismatch at ${i}`);
  });
  return { elements: candidate.length, maxAbs };
}
export function executedTests(log) {
  const matches = [...log.matchAll(/test result: ok\. (\d+) passed; (\d+) failed; (\d+) ignored;/g)];
  assert.ok(matches.length > 0, 'missing actual test execution');
  let passed = 0;
  for (const m of matches) { assert.equal(+m[2], 0); assert.equal(+m[3], 0, 'ignored tests are not evidence'); passed += +m[1]; }
  assert.ok(passed > 0, 'zero executed tests');
  return passed;
}
// ffprobe metadata is not decoding. Decode both streams completely; corrupt/missing AV fails.
export async function decodeAV(file, command = exec) {
  positive((await stat(file)).size);
  const { stdout } = await command('ffprobe', ['-v', 'error', '-count_frames', '-show_streams', '-of', 'json', file]);
  const streams = JSON.parse(stdout).streams;
  for (const type of ['video', 'audio']) {
    const stream = streams.find(s => s.codec_type === type);
    assert.ok(stream, `missing decoded ${type}`);
    positive(Number(stream.nb_read_frames)); positive(Number(stream.duration));
  }
  await command('ffmpeg', ['-v', 'error', '-xerror', '-i', file, '-map', '0:v:0', '-map', '0:a:0', '-f', 'null', '-'], { timeout: 300000, maxBuffer: 1048576 });
  return { sha256: await hashFile(file), streams, video: streams.find(s => s.codec_type === 'video'), audio: streams.find(s => s.codec_type === 'audio') };
}
// ---------------------------------------------------------------------------
// Request-bound AV acceptance (terminal review finding 14).
//
// `decodeAV` proves the file decodes; it does NOT prove the file is the media
// the request asked for. A complete 4-frame clip decodes and is "positive", so
// every length/geometry trap below used to pass. These are ports of the SAME
// resolved ladders the engine is handed, not a second implementation:
//
//   * temporal lattice and raw frame count:
//     crates/sceneworks-core/src/video_request.rs
//       `raw_frame_count` (427-438) rounds `duration * fps`,
//       `video_frame_count` (486-496) selects the family arm,
//       `MINIMAX_H3_LEGAL_FRAME_COUNTS` (523-525) = 124..345 step 17,
//       `MINIMAX_H3_FPS` (529) = 24,
//       `minimax_h3_frame_count` (556-563) aligns UP to the next rung,
//       `ltx_frame_count` (573-587) nearest 8k+1 (min 9),
//       `wan_frame_count` (593-596) floor 4k+1 (min 5),
//       `mochi_frame_count` (610-620) nearest 6k+1 (min 7).
//   * spatial grid: `dimension_multiple_of` (1721-1730) defaults to 32;
//     `config/manifests/builtin.models.jsonc` declares 32 for MiniMax-H3
//     (the SIZE_MULTIPLE the VAE/DiT enforce) and 64 for LTX-2.3/2.5.
//   * audio coverage: MiniMax-H3 is 32 kHz stereo and the engine fits the
//     waveform to the picture at `round(frames / fps * 32000)` samples per
//     channel (inference `mlx-gen-minimax-h3/src/audio_config.rs:48`,
//     `AUDIO_SAMPLE_RATE = 32_000`; SceneWorks
//     `crates/sceneworks-worker/src/video_jobs/tests.rs:15144-15166`).
//   * mux rule: `audio_mux_args` (video_jobs/mod.rs:1691-1767) encodes AAC and
//     bounds the file at `frames / fps` with `-t`, so decoded audio may differ
//     from the picture only by the codec's own frame/padding quantum — one AAC
//     frame (1024 samples) — not by an arbitrary slack.
// ---------------------------------------------------------------------------
export const MINIMAX_H3_LEGAL_FRAME_COUNTS = Object.freeze([124, 141, 158, 175, 192, 209, 226, 243, 260, 277, 294, 311, 328, 345]);
export const MINIMAX_H3_FPS = 24;
export const MINIMAX_H3_AUDIO_SAMPLE_RATE = 32_000;
export const MINIMAX_H3_AUDIO_CHANNELS = 2;
const DEFAULT_SPATIAL_STRIDE = 32; // core `DEFAULT_DIMENSION_MULTIPLE`
const LTX_SPATIAL_STRIDE = 64;     // LTX-2.3/2.5 `requiresDimensionsMultipleOf`
const isLtxModel = model => typeof model === 'string' && model.startsWith('ltx');
const isMochiModel = model => typeof model === 'string' && model.startsWith('mochi');
const isMinimaxH3Model = model => typeof model === 'string' && model.startsWith('minimax_h3');
export const rawFrameCount = (duration, fps) => Math.max(1, Math.round(Number(duration) * Number(fps)));
export const minimaxH3FrameCount = raw => MINIMAX_H3_LEGAL_FRAME_COUNTS.find(legal => legal >= raw) ?? MINIMAX_H3_LEGAL_FRAME_COUNTS[MINIMAX_H3_LEGAL_FRAME_COUNTS.length - 1];
export function ltxFrameCount(raw) {
  const frameCount = Math.max(9, raw), lower = frameCount - ((frameCount - 1) % 8), upper = lower + 8;
  if (lower < 9) return upper;
  return frameCount - lower <= upper - frameCount ? lower : upper;
}
export const wanFrameCount = raw => Math.max(5, Math.max(1, raw) - ((Math.max(1, raw) - 1) % 4));
export function mochiFrameCount(raw) {
  const frameCount = Math.max(7, raw), lower = frameCount - ((frameCount - 1) % 6), upper = lower + 6;
  return frameCount - lower <= upper - frameCount ? lower : upper;
}
// Mirrors `video_frame_count`: LTX, Mochi and MiniMax-H3 are explicit arms and the
// fallback is Wan, exactly as the Rust ladder documents.
export function videoFrameCount(model, raw) {
  if (isLtxModel(model)) return ltxFrameCount(raw);
  if (isMochiModel(model)) return mochiFrameCount(raw);
  if (isMinimaxH3Model(model)) return minimaxH3FrameCount(raw);
  return wanFrameCount(raw);
}
// True only for a value the engine's own lattice admits (`17n+5` + 5-15 s clamp
// for H3, `8k+1`, `6k+1`, `4k+1` for the others).
export function isLegalFrameCount(model, frames) {
  if (!Number.isInteger(frames) || frames <= 0) return false;
  if (isLtxModel(model)) return frames >= 9 && (frames - 1) % 8 === 0;
  if (isMochiModel(model)) return frames >= 7 && (frames - 1) % 6 === 0;
  if (isMinimaxH3Model(model)) return MINIMAX_H3_LEGAL_FRAME_COUNTS.includes(frames);
  return frames >= 5 && (frames - 1) % 4 === 0;
}
const spatialStride = (model, contract) => Number.isInteger(contract.spatialStride) && contract.spatialStride > 0
  ? contract.spatialStride
  : isLtxModel(model) ? LTX_SPATIAL_STRIDE : DEFAULT_SPATIAL_STRIDE;
// Encoder frame quanta that legitimately move a stream's decoded duration/start.
// AAC/MP3/AC-3 have fixed samples-per-frame; Opus defaults to 20 ms packets; PCM
// and the lossless codecs encode sample-exact durations. `null` means "no
// source-derived rule", which the validator refuses rather than guessing.
export function audioFrameSamples(codec, sampleRate) {
  switch (codec) {
    case 'aac': return 1024;
    case 'mp3': return 1152;
    case 'ac3': case 'eac3': return 1536;
    case 'opus': return Math.round(Number(sampleRate) * 0.02);
    case 'pcm_s16le': case 'pcm_s24le': case 'pcm_s32le': case 'pcm_f32le': case 'pcm_f64le': case 'flac': case 'alac': return 1;
    default: return null;
  }
}
const finiteNumber = (value, what) => {
  const number = typeof value === 'string' && value.trim() !== '' ? Number(value) : value;
  assert.ok(typeof number === 'number' && Number.isFinite(number), `${what} required`);
  return number;
};
// "24/1" -> 24. `0/0` (VFR / unknown) returns null so callers can refuse it.
function rational(value) {
  const match = /^(\d+)\/(\d+)$/.exec(String(value ?? ''));
  if (!match || Number(match[2]) === 0) return null;
  return Number(match[1]) / Number(match[2]);
}
function resolvedAudioRequirement(model, contract) {
  const declared = contract.audio;
  if (declared !== undefined) {
    assert.ok(declared && typeof declared === 'object', 'resolved contract audio must be an object');
    assert.equal(typeof declared.required, 'boolean', 'resolved contract audio.required must be declared');
    return declared;
  }
  if (isMinimaxH3Model(model)) return { required: true, sampleRate: MINIMAX_H3_AUDIO_SAMPLE_RATE, channels: MINIMAX_H3_AUDIO_CHANNELS, codec: 'aac' };
  if (isLtxModel(model)) return { required: true }; // 24 kHz core / 48 kHz with BWE: caller must resolve it.
  return { required: false };
}
/**
 * Pure request-bound acceptance for a decoded media job.
 *
 * @param {object} decoded            `decodeAV` result: `{ streams }`.
 * @param {object} request            queued job payload (model, fps, width, height, duration).
 * @param {object} resolvedContract   resolved model contract:
 *   { model, fps, width, height, frameCount?, spatialStride?, maxPixels?,
 *     audio?: { required, sampleRate?, channels?, codec?, frameSamples? } }
 *
 * Throws unless the decoded streams are exactly the media the request resolved
 * to. Returns the resolved lattice facts as the verification receipt.
 */
export function validateAVAgainstRequest(decoded, request, resolvedContract) {
  assert.ok(decoded && Array.isArray(decoded.streams), 'decoded AV streams required');
  assert.ok(request && typeof request === 'object', 'resolved request required');
  assert.ok(resolvedContract && typeof resolvedContract === 'object', 'resolved AV contract required');
  const model = resolvedContract.model ?? request.model;
  assert.ok(typeof model === 'string' && model.length > 0, 'resolved model id required');
  const fps = finiteNumber(resolvedContract.fps ?? request.fps, 'resolved fps');
  assert.ok(Number.isInteger(fps) && fps > 0, 'resolved fps must be a positive integer');
  assert.equal(finiteNumber(request.fps, 'request fps'), fps, 'request fps differs from the resolved contract');
  const width = finiteNumber(resolvedContract.width ?? request.width, 'resolved width');
  const height = finiteNumber(resolvedContract.height ?? request.height, 'resolved height');
  for (const [axis, value] of [['width', width], ['height', height]]) {
    assert.ok(Number.isInteger(value) && value > 0, `resolved ${axis} must be a positive integer`);
    assert.equal(finiteNumber(request[axis], `request ${axis}`), value, `request ${axis} differs from the resolved contract`);
  }
  const duration = finiteNumber(request.duration, 'request duration');
  assert.ok(duration > 0, 'request duration must be positive');
  // The engine REFUSES out-of-window H3 durations rather than clamping them
  // (`duration_limit_error`); the manifest's 5.1667..14.375 menu is 4-decimal
  // rounding of 124/24..345/24, hence the 1e-4 menu-rounding epsilon.
  const lastH3Rung = MINIMAX_H3_LEGAL_FRAME_COUNTS[MINIMAX_H3_LEGAL_FRAME_COUNTS.length - 1];
  const hardMin = resolvedContract.hardMinDuration ?? (isMinimaxH3Model(model) ? MINIMAX_H3_LEGAL_FRAME_COUNTS[0] / MINIMAX_H3_FPS : undefined);
  const hardMax = resolvedContract.hardMaxDuration ?? (isMinimaxH3Model(model) ? lastH3Rung / MINIMAX_H3_FPS : undefined);
  if (hardMin !== undefined) assert.ok(duration >= finiteNumber(hardMin, 'contract hardMinDuration') - 1e-4, `request duration ${duration}s is below the renderable minimum`);
  if (hardMax !== undefined) assert.ok(duration <= finiteNumber(hardMax, 'contract hardMaxDuration') + 1e-6, `request duration ${duration}s is above the renderable maximum; the engine refuses rather than shortens`);
  const stride = spatialStride(model, resolvedContract);
  for (const [axis, value] of [['width', width], ['height', height]]) assert.equal(value % stride, 0, `resolved ${axis} ${value} is off the model's ${stride}px spatial grid`);
  if (resolvedContract.maxPixels !== undefined) {
    const maxPixels = finiteNumber(resolvedContract.maxPixels, 'resolved maxPixels');
    assert.ok(maxPixels > 0 && width * height <= maxPixels, `resolved canvas ${width}x${height} exceeds the model's ${maxPixels}px area budget`);
  }
  // The request's own lattice snap is authoritative; a supplied contract frame
  // count can only AGREE with it, never loosen it.
  const frameCount = videoFrameCount(model, rawFrameCount(duration, fps));
  assert.ok(isLegalFrameCount(model, frameCount), `resolved frame count ${frameCount} is not on ${model}'s temporal lattice`);
  if (resolvedContract.frameCount !== undefined) assert.equal(finiteNumber(resolvedContract.frameCount, 'resolved frameCount'), frameCount, 'resolved frameCount is not the request lattice snap');
  const video = decoded.streams.find(stream => stream.codec_type === 'video');
  assert.ok(video, 'actual decoded video stream required');
  const decodedFrameCount = finiteNumber(video.nb_read_frames, 'decoded video frame count');
  assert.ok(Number.isInteger(decodedFrameCount) && decodedFrameCount > 0, 'decoded video frame count must be a positive integer');
  assert.equal(decodedFrameCount, frameCount, `decoded video holds ${decodedFrameCount} frames but the request resolved to ${frameCount}`);
  if (video.nb_frames !== undefined && video.nb_frames !== 'N/A') assert.equal(finiteNumber(video.nb_frames, 'container frame count'), frameCount, 'container frame count differs from the resolved lattice');
  const avgFps = rational(video.avg_frame_rate), realFps = rational(video.r_frame_rate);
  assert.ok(avgFps !== null, 'decoded average frame rate required');
  assert.ok(realFps !== null, 'decoded real frame rate required (constant-rate stream)');
  assert.ok(Math.abs(avgFps - fps) < 1e-9, `decoded average fps ${avgFps} differs from the resolved ${fps}`);
  assert.ok(Math.abs(realFps - fps) < 1e-9, `decoded real fps ${realFps} differs from the resolved ${fps} (not a constant-rate stream)`);
  for (const [axis, expected] of [['width', width], ['height', height]]) assert.equal(finiteNumber(video[axis], `decoded ${axis}`), expected, `decoded ${axis} differs from the resolved ${expected}`);
  const frameDuration = 1 / fps;
  const expectedDuration = frameCount / fps;
  const decodedDuration = finiteNumber(video.duration, 'decoded video duration');
  assert.ok(decodedDuration > 0, 'decoded video duration must be positive');
  // The container's timestamp quantum is one video frame; the exact frame count
  // above is what makes a short clip impossible, this only binds its timestamps.
  assert.ok(Math.abs(decodedDuration - expectedDuration) <= frameDuration + 1e-9, `decoded video lasts ${decodedDuration}s but the resolved ${frameCount} frames at ${fps}fps are ${expectedDuration}s`);
  const videoStart = finiteNumber(video.start_time ?? 0, 'decoded video start time');
  assert.ok(Math.abs(videoStart) <= frameDuration + 1e-9, `decoded video starts at ${videoStart}s, not the request origin`);
  const requirement = resolvedAudioRequirement(model, resolvedContract);
  const audio = decoded.streams.find(stream => stream.codec_type === 'audio');
  if (requirement.required) assert.ok(audio, 'missing decoded audio stream required by the resolved contract');
  const report = { model, fps, width, height, frameCount, durationSeconds: expectedDuration, video: { duration: decodedDuration, startTime: videoStart, codec: video.codec_name ?? null }, audio: null };
  if (!audio) return report;
  const codec = audio.codec_name;
  assert.ok(typeof codec === 'string' && codec.length > 0, 'decoded audio codec required');
  const sampleRate = finiteNumber(audio.sample_rate, 'decoded audio sample rate');
  const channels = finiteNumber(audio.channels, 'decoded audio channels');
  assert.ok(Number.isInteger(sampleRate) && sampleRate > 0, 'decoded audio sample rate must be a positive integer');
  assert.ok(Number.isInteger(channels) && channels > 0, 'decoded audio channels must be a positive integer');
  if (requirement.sampleRate !== undefined) assert.equal(sampleRate, finiteNumber(requirement.sampleRate, 'contract audio sample rate'), 'decoded audio sample rate differs from the resolved contract');
  if (requirement.channels !== undefined) assert.equal(channels, finiteNumber(requirement.channels, 'contract audio channels'), 'decoded audio channels differ from the resolved contract');
  if (requirement.codec !== undefined) assert.equal(codec, requirement.codec, 'decoded audio codec differs from the resolved contract');
  const declaredFrameSamples = requirement.frameSamples !== undefined ? finiteNumber(requirement.frameSamples, 'contract audio frameSamples') : undefined;
  const frameSamples = declaredFrameSamples ?? audioFrameSamples(codec, sampleRate);
  assert.ok(Number.isInteger(frameSamples) && frameSamples > 0, `no source-derived padding rule for audio codec ${codec}; declare contract.audio.frameSamples`);
  // One codec frame + one sample is the real padding quantum after `-t` trims to
  // the picture. Anything shorter means the soundtrack does not cover the clip;
  // anything longer means it was not bounded to the picture.
  const audioTolerance = (frameSamples + 1) / sampleRate;
  const expectedAudioSamples = Math.round(expectedDuration * sampleRate);
  const audioDuration = finiteNumber(audio.duration, 'decoded audio duration');
  assert.ok(audioDuration > 0, 'decoded audio duration must be positive');
  assert.ok(Math.abs(audioDuration - expectedDuration) <= audioTolerance, `decoded audio lasts ${audioDuration}s but the resolved picture is ${expectedDuration}s (${expectedAudioSamples} samples; tolerance ${audioTolerance}s)`);
  const audioStart = finiteNumber(audio.start_time ?? 0, 'decoded audio start time');
  assert.ok(Math.abs(audioStart - videoStart) <= Math.max(audioTolerance, frameDuration) + 1e-9, `decoded audio starts at ${audioStart}s while video starts at ${videoStart}s`);
  report.audio = { codec, sampleRate, channels, duration: audioDuration, startTime: audioStart, expectedSamples: expectedAudioSamples, toleranceSeconds: audioTolerance, frameSamples };
  return report;
}
// The single call a completed-media job should make: decode (real ffprobe count +
// full ffmpeg decode in production) and then bind it to the resolved request.
export async function decodeAndValidateAV(file, request, resolvedContract, command = exec) {
  const decoded = await decodeAV(file, command);
  return { decoded, verification: validateAVAgainstRequest(decoded, request, resolvedContract) };
}
export function sensitivityDecision(interventionDelta, repeatedBaselineDelta, minimumDelta, maximumBaselineDelta) {
  positive(minimumDelta);
  assert.ok(Number.isFinite(maximumBaselineDelta) && maximumBaselineDelta >= 0, 'predeclared baseline bound required');
  for (const v of [interventionDelta, repeatedBaselineDelta]) assert.ok(Number.isFinite(v) && v >= 0);
  assert.ok(repeatedBaselineDelta <= maximumBaselineDelta, 'nondeterministic baseline exceeds declared bound; unresolved sensitivity');
  const threshold = Math.max(minimumDelta, 3 * repeatedBaselineDelta);
  assert.ok(interventionDelta > threshold, 'conditioning/adapters ignored or below baseline noise floor');
  return { interventionDelta, repeatedBaselineDelta, threshold };
}
export async function compareControlMedia(baseline, repeated, intervention, stream, minimumDelta, maximumBaselineDelta, decode = decodeAV, delta = decodedDelta) {
  assert.ok(['video', 'audio'].includes(stream));
  const media = await Promise.all([baseline, repeated, intervention].map(file => decode(file)));
  const noise = await delta(baseline, repeated, stream), changed = await delta(baseline, intervention, stream);
  return { ...sensitivityDecision(changed, noise, minimumDelta, maximumBaselineDelta), stream, media };
}
export async function decodedDelta(a, b, stream, command = exec) {
  const samples = async file => {
    const args = stream === 'video'
      ? ['-an', '-vf', 'fps=1,scale=16:16', '-pix_fmt', 'rgb24', '-f', 'rawvideo']
      : ['-vn', '-ac', '1', '-ar', '8000', '-f', 'f32le'];
    const { stdout } = await command('ffmpeg', ['-v', 'error', '-xerror', '-i', file, ...args, 'pipe:1'], { encoding: 'buffer', timeout: 300000, maxBuffer: 1048576 });
    if (stream === 'video') return [...stdout].map(v => v / 255);
    assert.equal(stdout.length % 4, 0);
    return Array.from({ length: stdout.length / 4 }, (_, i) => stdout.readFloatLE(i * 4));
  };
  const x = await samples(a), y = await samples(b);
  assert.ok(x.length > 0 && x.length === y.length, 'control geometry/duration differs');
  let squared = 0;
  x.forEach((v, i) => { assert.ok(Number.isFinite(v) && Number.isFinite(y[i])); squared += (v - y[i]) ** 2; });
  return Math.sqrt(squared / x.length);
}
const contained = (base, name) => {
  const p = path.resolve(base, name), rel = path.relative(base, p);
  assert.ok(rel !== '' && !rel.startsWith('..') && !path.isAbsolute(rel), 'evidence path escapes case directory');
  return p;
};
export function validatePeakMetrics(metrics, c) {
  if(metrics?.kind==='owned-worker-epochs'){
    assert.ok(metrics.epochs?.length>=2&&new Set(metrics.epochs.map(epoch=>epoch.pid)).size===metrics.epochs.length,'distinct actual worker epochs required');
    for(const epoch of metrics.epochs){assert.ok(!epoch.metrics.kind,'epoch measurements must be directly attributed observations');validatePeakMetrics(epoch.metrics,c);for(const observation of Object.values(epoch.metrics.observations)){assert.equal(observation.attribution.pid,epoch.pid);for(const sample of observation.samples)assert.equal(sample.pid,epoch.pid,'epoch sample belongs to a different PID');}}
    for(const key of ['hostPeakBytes','devicePeakBytes','loadHostPeakBytes','loadDevicePeakBytes'])assert.equal(metrics[key],Math.max(...metrics.epochs.map(epoch=>epoch.metrics[key])),'aggregate must be the maximum across distinct process epochs, not their sum');
    return;
  }
  const hostMethods = ['sampled-process-rss', 'os-rss-high-water'];
  const deviceMethods = ['sampled-process-cuda-used', 'sampled-device-used', 'sampled-native-allocated', 'native-allocated-high-water', 'mlx-allocator-cumulative-high-water'];
  for (const [name, bytesKey] of [['host', 'hostPeakBytes'], ['device', 'devicePeakBytes'], ['loadHost', 'loadHostPeakBytes'], ['loadDevice', 'loadDevicePeakBytes']]) {
    positive(metrics?.[bytesKey]);
    const o = metrics.observations?.[name], device = name.toLowerCase().includes('device');
    assert.ok((device ? deviceMethods : hostMethods).includes(o?.method), 'honest peak observation method required');
    assert.equal(o.sampled, o.method.startsWith('sampled-'));
    if (o.method === 'sampled-process-cuda-used') assert.equal(c.backend, 'cuda');
    if (o.method === 'mlx-allocator-cumulative-high-water') assert.equal(c.backend, 'mlx');
    const start = Date.parse(o.window?.start), end = Date.parse(o.window?.end);
    assert.ok(Number.isFinite(start) && Number.isFinite(end) && end >= start, 'observed peak window required');
    assert.equal(o.window.phase, name.startsWith('load') ? 'loading' : 'case');
    assert.ok((name.startsWith('load') ? ['full-case-load-inclusive', 'native-loading-windows'] : ['full-case']).includes(o.window.coverage), 'source acquisition alone does not cover deferred native loading');
    if (o.window.coverage === 'full-case-load-inclusive') assert.ok(o.bounds?.includes('full-case-load-inclusive-upper-bound'), 'whole-case bound is not an exact loading-phase peak');
    assert.ok(o.attribution?.scope && o.attribution?.meaning, 'peak attribution and exact meaning required');
    if (device) assert.ok(o.attribution.deviceId, 'device attribution required');
    if (o.method === 'sampled-device-used') {
      assert.equal(o.attribution.scope, 'device'); assert.equal(o.attribution.includesOtherProcesses, true);
      assert.ok(o.bounds?.includes('observed-device-global-bound'), 'device-global bound is not model/process allocation evidence');
    } else assert.ok(Number.isSafeInteger(o.attribution.pid) && o.attribution.pid > 0, 'process attribution required');
    assert.ok(Array.isArray(o.blindspots) && o.blindspots.every(v => typeof v === 'string' && v.length > 0));
    assert.ok(Array.isArray(o.samples) && o.samples.length > 0, 'actual peak observations required');
    if (o.sampled) { positive(o.intervalMs); assert.ok(o.samples.length >= 2 && o.blindspots.length > 0, 'sampled peaks must disclose sampling blindspots'); }
    if (o.method === 'mlx-allocator-cumulative-high-water' || o.method === 'os-rss-high-water') {
      assert.equal(o.attribution.scope, 'process-lifetime');
      assert.equal(o.attribution.includesPriorAllocations, true, 'cumulative highwater is not window-local');
    }
    for (const sample of o.samples) {
      const at = Date.parse(sample.at);
      assert.ok(at >= start && at <= end && Number.isFinite(sample.bytes) && sample.bytes >= 0, 'sample outside observation window or invalid bytes');
    }
    assert.equal(o.samples.reduce((peak, sample) => Math.max(peak, sample.bytes), 0), metrics[bytesKey], 'peak differs from retained observations');
  }
  assert.ok(metrics.loadHostPeakBytes <= c.loadHostBudgetBytes && metrics.loadDevicePeakBytes <= c.loadDeviceBudgetBytes, 'loading exceeds declared budgets');
}
export function normalizeLoadEvent(raw) {
  const fields = raw?.fields ?? raw;
  assert.equal(fields?.event, 'h3_import_prepared_load');
  assert.equal(fields.seal_validated, true);
  assert.ok(['Cold', 'Warm'].includes(fields.cache_state));
  const parse = value => typeof value === 'string' ? JSON.parse(value) : value;
  const optionNumber = value => {
    if (value === null || value === 'None') return null;
    const number = typeof value === 'number' ? value : Number(/^Some\(([^)]+)\)$/.exec(value ?? '')?.[1]);
    assert.ok(Number.isFinite(number), 'missing/invalid observed sampling field'); return number;
  };
  assert.ok(fields.backend && fields.backend !== 'cpu', 'CPU fallback cannot prove native H3 acceptance');
  return { kind: 'prepared-native-load-closure', jobId: fields.jobId, providerId: fields.provider, backend: fields.native_backend, deviceLabel: fields.backend, identity: parse(fields.identity), prepared_sources: parse(fields.prepared_sources), seed: fields.seed, operation: fields.operation, steps: optionNumber(fields.steps), schedulerShift: optionNumber(fields.scheduler_shift), cacheState: fields.cache_state };
}
export async function validateEvidence(p, c, dir, log, decode = decodeAV, delta = decodedDelta) {
  if(c.id.endsWith('.generation-source-prep-cancel')){const {requireGenerationSourceCancelContract}=await import('./h3-finetune-terminal-runtime.mjs');requireGenerationSourceCancelContract();}
  const canonical = requiredCases().find(item => item.id === c.id);
  assert.ok(canonical, 'unknown acceptance case');
  for (const [key, value] of Object.entries(canonical)) assert.equal(c[key], value, 'case reclassification cannot weaken acceptance');
  const e = await json(contained(dir, c.evidence));
  assert.equal(e.status, 'executed');
  assert.notEqual(e.proof_kind, 'unit_fake', 'unit_fake cannot be campaign evidence');
  assert.equal(e.sceneworks, p.sceneworks); assert.equal(e.inference, p.inference);
  assert.equal(e.caseId, c.id); assert.equal(e.backend, c.backend);
  assert.ok(e.hardware?.device && e.hardware?.driver, 'physical hardware identity required');
  const modelExecution = ['artifact', 'capability'].includes(c.group) || /\.(bounded-loading|cold-warm-staged)$/.test(c.id);
  if (modelExecution) {
    // Honest post-acquisition sealed closure, not a per-tensor read census or intended plan.
    assert.ok(e.runtimeIdentity?.checkpointSha256, 'actual loaded checkpoint content identity required');
    assert.ok(c.inputs.some(i => i.role === 'checkpoint' && i.sha256 === e.runtimeIdentity.checkpointSha256), 'wrong/stock checkpoint loaded');
    for (const key of ['topology', 'quantization', 'components', 'adapters', 'operation', 'samplingPolicy', 'planId']) assert.ok(e.runtimeIdentity[key] != null, `missing runtime ${key}`);
    for (const component of e.runtimeIdentity.components) assert.ok(c.inputs.some(i => i.sha256 === component.sha256), 'unverified runtime component');
    sha64(e.runtimeIdentity.semanticDigest);
    const closure = normalizeLoadEvent(e.loadEvent);
    assert.equal(closure.kind, 'prepared-native-load-closure');
    assert.equal(closure.jobId, e.completedJob?.id); assert.ok(closure.jobId);
    assert.equal(e.completedJob.status, 'completed'); assert.equal(closure.backend, c.backend);
    assert.ok(closure.providerId && closure.providerId === e.completedJob.providerId);
    assert.equal(closure.operation, e.runtimeIdentity.operation); assert.equal(closure.seed, e.runtimeIdentity.seed);
    assert.equal(closure.steps, e.runtimeIdentity.samplingPolicy.steps);
    assert.equal(closure.schedulerShift, e.runtimeIdentity.samplingPolicy.schedulerShift);
    assert.equal(closure.identity.planId, e.runtimeIdentity.planId);
    assert.equal(closure.identity.semanticDigest, e.runtimeIdentity.semanticDigest);
    assert.ok(closure.identity.sourceFiles.some(i => i.sha256 === e.runtimeIdentity.checkpointSha256), 'missing sealed checkpoint identity');
    assert.ok(closure.prepared_sources.length > 0, 'missing native prepared closure');
    for (const input of closure.identity.sourceFiles) assert.ok(c.inputs.some(i => i.sha256 === input.sha256), 'native source closure differs from verified inputs');
    for (const input of closure.prepared_sources) {
      // Imports may create native derivatives during this case; don't invent their hashes in
      // the pre-run plan. Verify their actual bytes after acquisition, confined to task state.
      const resolved = await realpath(input.path), relative = path.relative(await realpath(p.output), resolved);
      const declared = c.inputs.some(i => path.resolve(i.path) === path.resolve(input.path) && i.sha256 === input.sha256);
      assert.ok(declared || (relative !== '' && !relative.startsWith('..') && !path.isAbsolute(relative)), 'undeclared prepared source outside task state');
      assert.equal(await hashFile(input.path), input.sha256, 'prepared source content changed');
    }
    validatePeakMetrics(e.metrics, c);
  } else assert.ok(c.inputs.some(i => i.sha256 === e.testInputSha256), 'executed test input identity required');
  const files = [];
  for (const name of e.files ?? []) { const f = contained(dir, name); files.push({ path: name, sha256: await hashFile(f) }); }
  assert.ok(files.length > 0, 'missing supporting evidence');
  let av, numeric;
  if (['artifact', 'capability'].includes(c.group)) {
    assert.equal(e.completedJob.output, e.av, 'successful output not bound to native job');
    av = await decode(contained(dir, e.av));
    if (c.group === 'artifact') {
      assert.deepEqual(e.source, SOURCE); assert.equal(e.artifact, c.artifact);
      const catalogPath = path.join(dir, 'source-catalog.json');
      verifyFrozenArtifact(c, await json(catalogPath));
      files.push({ path: 'source-catalog.json', sha256: await hashFile(catalogPath) });
      assert.equal(e.runtimeIdentity.checkpointSha256, c.inputs.find(i => i.artifact === c.artifact)?.sha256);
    } else {
      // Paired intervention: same checkpoint/seed, changed conditioning/routing/adapter.
      assert.ok(e.control?.intervention, 'causal conditioning control required');
      assert.equal(e.control.checkpointSha256, e.runtimeIdentity.checkpointSha256);
      assert.ok(Number.isSafeInteger(e.runtimeIdentity.seed));
      assert.equal(e.control.seed, e.runtimeIdentity.seed);
      const controlAV = await decode(contained(dir, e.control.av));
      assert.notEqual(controlAV.sha256, av.sha256, 'conditioning/adapters ignored');
      // Compute a scale-sensitive delta from decoded media, not a producer-claimed scalar/hash.
      const measuredDelta = await delta(contained(dir, e.av), contained(dir, e.control.av), c.controlStream);
      assert.ok(measuredDelta >= c.minimumControlDelta, 'conditioning/adapters ignored');
      files.push({ path: e.control.av, sha256: await hashFile(contained(dir, e.control.av)), measuredDelta, stream: c.controlStream });
    }
  } else {
    if (c.group !== 'numeric' && e.apiTranscript) {
      const transcript = await json(contained(dir, e.apiTranscript));
      assert.equal(transcript.caseId, c.id);
      const assertions = transcript.transcript.filter(r => r.assertion);
      assert.ok(assertions.length > 0, 'no API assertions executed');
      for (const a of assertions) assert.deepEqual(a.actual, a.expected);
      assert.ok(log.includes(`H3_API_ASSERTIONS_EXECUTED=${assertions.length}`), 'API driver did not complete');
      files.push({ path: e.apiTranscript, sha256: await hashFile(contained(dir, e.apiTranscript)) });
    } else executedTests(log);
    if (c.group === 'numeric' && c.numericProof?.kind === 'native-pinned-fixture') {
      const { validateComparisonProof } = await import('./h3-finetune-terminal-runtime.mjs');
      const proof = await json(contained(dir, e.nativeComparison));
      assert.equal(proof.sceneworks, p.sceneworks); assert.equal(proof.inference, p.inference);
      contained(await realpath(p.output), await realpath(proof.binary.path));
      assert.equal(await hashFile(proof.binary.path), proof.binary.sha256, 'compiled native binary changed');
      for (const input of [...proof.fixtures, ...proof.referenceBlobs]) assert.equal(await hashFile(input.path), input.sha256);
      let nativeLog = '';
      for (const run of proof.runs) { const text = await readFile(contained(dir, run.log), 'utf8'); nativeLog += text + '\n'; }
      for (const item of proof.supporting ?? []) assert.equal(executedTests(await readFile(contained(dir, item.log), 'utf8')), item.passed);
      numeric = validateComparisonProof(c, proof, nativeLog);
      files.push({ path: e.nativeComparison, sha256: await hashFile(contained(dir, e.nativeComparison)) });
    } else if (c.group === 'numeric') {
      sha40(e.reference?.revision); sha64(e.reference?.sha256);
      assert.equal(e.reference.revision, c.referenceRevision, 'reference revision drift');
      assert.ok(c.inputs.some(i => i.role === 'reference' && i.sha256 === e.reference.sha256), 'unpinned executable reference');
      assert.equal(e.numeric.reference, c.referenceOutput);
      const candidatePath = contained(dir, e.numeric.candidate), referencePath = contained(dir, e.numeric.reference);
      const referenceRun = await json(path.join(dir, 'reference-execution.json'));
      assert.deepEqual(referenceRun.command, c.referenceCommand);
      assert.equal(referenceRun.revision, c.referenceRevision);
      assert.equal(referenceRun.outputSha256, await hashFile(referencePath), 'executable reference output was replaced');
      numeric = compareNumeric(await json(candidatePath), await json(referencePath), c.tolerance);
      files.push({ path: e.numeric.candidate, sha256: await hashFile(candidatePath) }, { path: e.numeric.reference, sha256: await hashFile(referencePath) });
    }
  }
  let runtime;
  if (e.runtimeBindings) {
    assert.equal(e.runtimeBindings, 'source-job-output.json');
    const { verifyAPIBindings } = await import('./h3-finetune-terminal-runtime.mjs');
    runtime = await verifyAPIBindings(p, c, dir, decode, delta);
  }
  return { evidenceSha256: await hashFile(contained(dir, c.evidence)), processLogSha256: createHash('sha256').update(log).digest('hex'), files, ...(av ? { av } : {}), ...(numeric ? { numeric } : {}), ...(runtime ? { runtime } : {}) };
}
// No shell interpolation. The process must implement cooperative timeout/cancel (native GPU
// execution must not be killed mid-command-buffer); Actions is the final job timeout boundary.
export async function runProcess(argv, { cwd, env, logPath, timeoutSeconds }) {
  const output = createWriteStream(logPath, { flags: 'wx' });
  await new Promise((resolve, reject) => { output.once('open', resolve); output.once('error', reject); });
  const child = spawn(argv[0], argv.slice(1), { cwd, env, stdio: ['ignore', 'pipe', 'pipe'], shell: false });
  child.stdout.pipe(output, { end: false }); child.stderr.pipe(output, { end: false });
  let expired = false;
  const timer = setTimeout(() => { expired = true; }, timeoutSeconds * 1000);
  try {
    const code = await new Promise((resolve, reject) => { child.on('error', reject); child.on('close', resolve); });
    assert.ok(!expired, 'case exceeded timeout; failed, never accepted on resume');
    assert.equal(code, 0, 'case process failed');
  } finally { clearTimeout(timer); await new Promise(resolve => output.end(resolve)); }
}
export async function runCase(p, id, root, inferenceRoot) {
  assert.notEqual(p.proof_kind, 'unit_fake', 'unit_fake plan is not campaign authorization');
  assert.equal(process.env.GITHUB_ACTIONS, 'true', 'native campaign execution is Actions-only');
  assert.equal(process.env.H3_TERMINAL_AUTHORIZED_PAIR, `${p.sceneworks}:${p.inference}`, 'explicit final pair authorization required');
  validatePlan(p); await sourcePair(p, root, inferenceRoot);
  const c = p.cases.find(c => c.id === id); assert.ok(c, 'unknown case');
  if (c.backend === 'mlx') { assert.equal(process.platform, 'darwin'); assert.equal(process.env.RUNNER_NAME, 'nax-macos-2'); }
  else assert.equal(process.platform, 'win32');
  await verifyInputs(c, [p.output, root, inferenceRoot]);
  const dir = contained(p.output, c.id); await mkdir(dir, { recursive: true });
  const identity = { planSha256: digest(p), caseId: id, command: c.command, sceneworks: p.sceneworks, inference: p.inference, runId: process.env.GITHUB_RUN_ID, runAttempt: process.env.GITHUB_RUN_ATTEMPT, responsibleTask: p.responsibleTask, timeoutSeconds: c.timeoutSeconds, nextAuthorizedAction: p.nextAuthorizedAction };
  const statePath = path.join(dir, 'execution.json');
  let previous;
  try { previous = await json(statePath); } catch (e) { if (e.code !== 'ENOENT') throw e; }
  if (previous) {
    assert.equal(previous.planSha256, identity.planSha256, 'resume plan drift');
    assert.equal(previous.status, 'passed', 'interrupted/failed case requires a new output directory, not silent overwrite');
    const verified = await validateEvidence(p, c, dir, await readFile(path.join(dir, 'process.log'), 'utf8'));
    assert.deepEqual(verified, previous.verified, 'resume evidence drift');
    return previous;
  }
  await save(statePath, { ...identity, status: 'running', started: new Date().toISOString() });
  try {
    if(c.id.endsWith('.generation-source-prep-cancel')){const {requireGenerationSourceCancelContract}=await import('./h3-finetune-terminal-runtime.mjs');requireGenerationSourceCancelContract();}
    if (c.group === 'artifact') await save(path.join(dir, 'source-catalog.json'), await fetchFrozenCatalog(c));
    const options = { cwd: root, env: { ...process.env, H3_CASE_DIR: dir, H3_CASE_ID: id, H3_INFERENCE_ROOT: inferenceRoot, H3_TIMEOUT_SECONDS: String(c.timeoutSeconds) }, timeoutSeconds: c.timeoutSeconds };
    if (c.group === 'numeric' && c.numericProof?.kind !== 'native-pinned-fixture') {
      await runProcess(c.referenceCommand, { ...options, logPath: path.join(dir, 'reference.log') });
      await save(path.join(dir, 'reference-execution.json'), { command: c.referenceCommand, revision: c.referenceRevision, outputSha256: await hashFile(contained(dir, c.referenceOutput)) });
    }
    let command = c.command;
    if (c.runtime?.scenario || c.numericProof?.kind === 'native-pinned-fixture' || c.fixtureProof?.kind === 'source-bound-assertions') {
      const adapterPlan = path.join(dir, 'adapter-plan.json'); await save(adapterPlan, p);
      command = [process.execPath, path.join(root, 'scripts/h3-finetune-terminal-runtime.mjs'), c.numericProof?.kind === 'native-pinned-fixture' ? 'numeric' : c.fixtureProof?.kind === 'source-bound-assertions' ? 'fixture' : 'api', adapterPlan, id, root, inferenceRoot];
      await save(path.join(dir, 'adapter-command.json'), { command, responsibleTask: p.responsibleTask });
    }
    await runProcess(command, { ...options, logPath: path.join(dir, 'process.log') });
    await sourcePair(p, root, inferenceRoot); await verifyInputs(c, [p.output, root, inferenceRoot]);
    const verified = await validateEvidence(p, c, dir, await readFile(path.join(dir, 'process.log'), 'utf8'));
    const receipt = { ...identity, status: 'passed', completed: new Date().toISOString(), verified };
    await save(statePath, receipt); return receipt;
  } catch (error) { await save(statePath, { ...identity, status: 'failed', error: error.message }); throw error; }
}
export function validateDelivery(p) {
  const d = p.delivery;
  assert.ok(d, 'missing integrated delivery records');
  const link = v => assert.match(v ?? '', /^https:\/\//);
  assert.ok(Array.isArray(d.requiredCI) && d.requiredCI.length > 0);
  for (const check of d.requiredCI) { assert.ok([p.sceneworks, p.inference].includes(check.headSha), 'CI source mismatch'); assert.equal(check.conclusion, 'success'); link(check.url); }
  // The combined epic/production review precedes the permanent pin, so its source
  // pair need not be the later stationary campaign pair. Bind any intervening delta
  // to the focused final-pair review instead of demanding a second whole-branch review.
  const review = d.featureEndReview;
  sha40(review?.sceneworks); sha40(review?.inference);
  assert.equal(review.scope, 'epic-requirements-and-production-delta');
  assert.equal(review.unresolvedFindings, 0); link(review.url);
  if (review.sceneworks !== p.sceneworks || review.inference !== p.inference || d.lateDeltaReview) {
    const late = d.lateDeltaReview;
    assert.ok(late, 'changed reviewed pair requires focused late-delta review');
    assert.equal(late.scope, 'late-pin-generated-repair-delta');
    assert.equal(late.baseSceneworks, review.sceneworks); assert.equal(late.baseInference, review.inference);
    assert.equal(late.sceneworks, p.sceneworks); assert.equal(late.inference, p.inference);
    assert.equal(late.unresolvedFindings, 0); link(late.url);
  }
  assert.equal(d.inferenceMain?.sha, p.inference); link(d.inferenceMain.url);
  sha40(d.sceneworksMain?.sha); assert.equal(d.sceneworksMain.sourceCommit, p.sceneworks); assert.equal(d.sceneworksMain.inferencePin, p.inference); link(d.sceneworksMain.url);
  assert.ok(Array.isArray(d.shortcutReadbacks) && d.shortcutReadbacks.some(r => r.id === 26679) && d.shortcutReadbacks.some(r => r.id === 26691));
  for (const r of d.shortcutReadbacks) { assert.ok(r.observedState && r.readAt); link(r.url); }
  assert.ok(Array.isArray(d.limitations) && d.limitations.length === 0, 'unresolved limitations are not a pass');
  // These are recorded readbacks, not an authorization to merge or mutate the tracker.
}
export function validateBackendReadback(p, backend, r, { allowUnitFake = false } = {}) {
  assert.equal(r.kind, 'backend-content-readback');
  assert.ok(r.proof_kind === 'campaign_content_reverified' || (allowUnitFake && r.proof_kind === 'unit_fake'), 'unit_fake readback is not campaign evidence');
  assert.equal(r.backend, backend); validatePlan(r.plan); if (!allowUnitFake) assert.notEqual(r.plan.proof_kind, 'unit_fake');
  assert.equal(r.plan.sceneworks, p.sceneworks); assert.equal(r.plan.inference, p.inference); assert.deepEqual(r.plan.source, p.source);
  assert.equal(r.planSha256, digest(r.plan)); assert.ok(r.verificationRunId && r.verificationRunAttempt);
  const expected = requiredCases().filter(c => c.backend === backend);
  assert.ok(allowUnitFake||!expected.some(c=>c.id.endsWith('.generation-source-prep-cancel')),'BLOCKED: native cold-preparation cancellation evidence contract is not implemented; backend cannot be certified complete');
  assert.equal(r.cases.length, expected.length); assert.equal(new Set(r.cases.map(c => c.id)).size, expected.length);
  for (const c of expected) {
    const row = r.cases.find(v => v.id === c.id), planned = r.plan.cases.find(v => v.id === c.id);
    assert.ok(row && row.runId && row.runAttempt && row.verified?.files?.length); sha64(row.executionSha256);
    assert.deepEqual(row.sourceInputs, planned.inputs); assert.equal(row.planSha256, r.planSha256);
  }
  return r;
}
export async function backendComplete(p, backend) {
  const selected = p.cases.filter(v => v.backend === backend); assert.ok(selected.length);
  for (const c of selected) {
    if(c.id.endsWith('.generation-source-prep-cancel'))return false; // Actual observer/ack contract is pending; labels or forged success rows cannot complete a backend.
    let r; try { r = await json(path.join(contained(p.output, c.id), 'execution.json')); } catch (error) { if (error.code === 'ENOENT') return false; throw error; }
    if (r.status !== 'passed' || r.planSha256 !== digest(p)) return false;
  }
  return true;
}
export async function verifyBackend(p, backend, root, inferenceRoot) {
  validatePlan(p); const { authorized } = await import('./h3-finetune-terminal-runtime.mjs');
  const selected = p.cases.filter(c => c.backend === backend); assert.ok(selected.length); authorized(p, selected[0]);
  await sourcePair(p, root, inferenceRoot); const cases = [];
  for (const c of selected) {
    await verifyInputs(c, [p.output, root, inferenceRoot]); const dir = contained(p.output, c.id), executionPath = path.join(dir, 'execution.json'), r = await json(executionPath);
    assert.equal(r.status, 'passed'); assert.equal(r.planSha256, digest(p)); assert.ok(r.runId && r.runAttempt);
    const verified = await validateEvidence(p, c, dir, await readFile(path.join(dir, 'process.log'), 'utf8'));
    assert.deepEqual(verified, r.verified);
    cases.push({ id: c.id, runId: r.runId, runAttempt: r.runAttempt, planSha256: r.planSha256, executionSha256: await hashFile(executionPath), sourceInputs: c.inputs, verified });
  }
  await sourcePair(p, root, inferenceRoot);
  const r = { kind: 'backend-content-readback', proof_kind: 'campaign_content_reverified', backend, plan: p, planSha256: digest(p), verificationRunId: process.env.GITHUB_RUN_ID, verificationRunAttempt: process.env.GITHUB_RUN_ATTEMPT, readAt: new Date().toISOString(), cases };
  validateBackendReadback(p, backend, r);
  const filename = path.join(p.output, `backend-readback-${backend}.json`);
  try { await writeFile(filename, JSON.stringify(r, null, 2) + '\n', { flag: 'wx' }); }
  catch (error) { if (error.code !== 'EEXIST') throw error; const previous = validateBackendReadback(p, backend, await json(filename)); assert.deepEqual(previous.cases, r.cases); return { path: filename, readback: previous }; }
  return { path: filename, readback: r };
}
export async function verifyPortableCase(p, c, dir, row) {
  if(c.id.endsWith('.generation-source-prep-cancel')){const {requireGenerationSourceCancelContract}=await import('./h3-finetune-terminal-runtime.mjs');requireGenerationSourceCancelContract();}
  assert.equal(await hashFile(path.join(dir, 'execution.json')), row.executionSha256, 'downloaded execution record differs from backend readback');
  const execution = await json(path.join(dir, 'execution.json')); assert.equal(execution.status, 'passed'); assert.equal(execution.planSha256, row.planSha256); assert.equal(execution.runId, row.runId); assert.equal(execution.runAttempt, row.runAttempt); assert.deepEqual(execution.verified, row.verified);
  const v = row.verified, log = await readFile(path.join(dir, 'process.log'), 'utf8');
  assert.equal(createHash('sha256').update(log).digest('hex'), v.processLogSha256);
  const e = await json(contained(dir, c.evidence)); assert.equal(e.status, 'executed'); assert.notEqual(e.proof_kind, 'unit_fake');
  assert.deepEqual([e.sceneworks, e.inference], [p.sceneworks, p.inference]); assert.equal(e.caseId, c.id); assert.equal(e.backend, c.backend);
  assert.equal(await hashFile(contained(dir, c.evidence)), v.evidenceSha256);
  for (const file of v.files) assert.equal(await hashFile(contained(dir, file.path)), file.sha256, 'downloaded supporting evidence drift');
  if (e.av) { const media = await decodeAV(contained(dir, e.av)); if (v.av) assert.equal(media.sha256, v.av.sha256); }
  if (c.group === 'artifact') verifyFrozenArtifact(c, await json(contained(dir, 'source-catalog.json')));
  if (e.runtimeBindings) {
    const { verifyAPIBindings } = await import('./h3-finetune-terminal-runtime.mjs');
    assert.deepEqual(await verifyAPIBindings(p, c, dir, decodeAV, decodedDelta, { contentReadback: true }), v.runtime);
  } else if (c.group === 'capability') {
    await decodeAV(contained(dir, e.control.av));
    assert.ok(await decodedDelta(contained(dir, e.av), contained(dir, e.control.av), c.controlStream) >= c.minimumControlDelta);
  }
  if (c.group === 'numeric' && c.numericProof?.kind === 'native-pinned-fixture') {
    const { validateComparisonProof } = await import('./h3-finetune-terminal-runtime.mjs');
    const proof = await json(contained(dir, e.nativeComparison)); let actual = '';
    for (const run of proof.runs) actual += await readFile(contained(dir, run.log), 'utf8') + '\n';
    assert.deepEqual(validateComparisonProof(c, proof, actual), v.numeric);
  } else if (c.group === 'numeric') compareNumeric(await json(contained(dir, e.numeric.candidate)), await json(contained(dir, e.numeric.reference)), c.tolerance);
  else if (e.apiTranscript) { const t = await json(contained(dir, e.apiTranscript)); assert.equal(t.caseId, c.id); const a = t.transcript.filter(r => r.assertion); assert.ok(a.length); for (const item of a) assert.deepEqual(item.actual, item.expected); assert.ok(log.includes(`H3_API_ASSERTIONS_EXECUTED=${a.length}`)); }
  else if (!['artifact','capability'].includes(c.group)) executedTests(log);
  return { id: c.id, inputContentReverifiedOn: c.backend, evidenceReverifiedLocally: true };
}
export async function verifyCampaign(p) {
  validatePlan(p);
  if (p.backendReadbacks) {
    for (const backend of ['mlx', 'cuda']) {
      const spec = p.backendReadbacks[backend]; assert.ok(spec?.path && spec.caseRoot, 'both native backend content readbacks and retained case files required');
      const r = validateBackendReadback(p, backend, await json(spec.path));
      for (const row of r.cases) await verifyPortableCase(r.plan, r.plan.cases.find(c => c.id === row.id), contained(spec.caseRoot, row.id), row);
    }
    validateDelivery(p);
    return { status: 'hardware-evidence-complete-delivery-recorded', sceneworks: p.sceneworks, inference: p.inference, cases: requiredCases().length, inputVerification: 'actual native host readbacks; all retained evidence/AV reverified locally' };
  }
  for (const c of p.cases) {
    await verifyInputs(c);
    const dir = contained(p.output, c.id), r = await json(path.join(dir, 'execution.json'));
    assert.equal(r.status, 'passed', `not executed: ${c.id}`); assert.equal(r.planSha256, digest(p));
    assert.ok(r.runId && r.runAttempt, 'missing Actions execution identity');
    assert.deepEqual(await validateEvidence(p, c, dir, await readFile(path.join(dir, 'process.log'), 'utf8')), r.verified);
  }
  // Hardware evidence alone is not integrated delivery / tracker completion.
  validateDelivery(p);
  return { status: 'hardware-evidence-complete-delivery-recorded', sceneworks: p.sceneworks, inference: p.inference, cases: p.cases.length };
}
async function terminalCLI() {
  const [mode, planFile, ...args] = process.argv.slice(2);
  try {
    if (mode === 'list') console.log(JSON.stringify({ source: SOURCE, cases: requiredCases() }, null, 2));
    else if (mode === 'compare-control') {
      assert.equal(args.length, 5, 'compare-control BASE REPEAT INTERVENTION video|audio MIN_DELTA MAX_BASELINE_DELTA');
      console.log(JSON.stringify(await compareControlMedia(planFile, args[0], args[1], args[2], Number(args[3]), Number(args[4])), null, 2));
    } else if (mode === 'case-plan') {
      const { concretePlan } = await import('./h3-finetune-terminal-cases.mjs');
      console.log(JSON.stringify(concretePlan(planFile, args[0], args[1]), null, 2));
    } else if (mode === 'prepare-cases') {
      const { prepareCases } = await import('./h3-finetune-terminal-cases.mjs');
      console.log(JSON.stringify(await prepareCases(await json(planFile), args[0]), null, 2));
    } else if (mode === 'prepare-lokr') {
      const { prepareControlledLoKr } = await import('./h3-finetune-terminal-cases.mjs');
      console.log(JSON.stringify(await prepareControlledLoKr(await json(planFile), args[0]), null, 2));
    } else if (mode === 'bind-plan') {
      const { finalizeCampaignPlan } = await import('./h3-finetune-terminal-cases.mjs');
      console.log(JSON.stringify(await finalizeCampaignPlan(await json(planFile), args[0]), null, 2));
    } else {
      const p = validatePlan(await json(planFile));
      if (mode === 'run-case') {
        const receipt = await runCase(p, args[0], args[1], args[2]), c = p.cases.find(v => v.id === args[0]);
        if (await backendComplete(p, c.backend)) {
          const filename = path.join(p.output, `backend-readback-${c.backend}.json`); let previous;
          try { previous = validateBackendReadback(p, c.backend, await json(filename)); } catch (error) { if (error.code !== 'ENOENT') throw error; }
          if (previous) assert.equal(previous.planSha256, digest(p)); else await verifyBackend(p, c.backend, args[1], args[2]);
          await copyFile(filename, path.join(contained(p.output, c.id), 'backend-readback.json'));
        }
        console.log(JSON.stringify(receipt));
      } else if (mode === 'verify-backend') console.log(JSON.stringify(await verifyBackend(p, args[0], args[1], args[2])));
      else if (mode === 'verify') console.log(JSON.stringify(await verifyCampaign(p)));
      else if (mode === 'check-plan') console.log(`complete inventory: ${p.cases.length} cases; no execution claimed`);
      else throw new Error('usage: h3-finetune-terminal.mjs list | case-plan SW_ROOT INFERENCE_ROOT STAGE_ROOT | prepare-cases PARAMETERS NEW_STAGE_ROOT | bind-plan PARAMETERS PREPARED_ROOT | prepare-lokr SPEC NEW_DIRECTORY | compare-control BASE REPEAT INTERVENTION video|audio MIN_DELTA MAX_BASELINE_DELTA | check-plan PLAN | run-case PLAN CASE SW_ROOT INFERENCE_ROOT | verify-backend PLAN mlx|cuda SW_ROOT INFERENCE_ROOT | verify PLAN');
    }
  } catch (e) { console.error(e.message); process.exitCode = 1; }
}
if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) terminalCLI();
