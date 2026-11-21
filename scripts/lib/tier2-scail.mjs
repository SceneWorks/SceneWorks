// Candidate-only plural SCAIL-2 product witnesses. All learned work belongs to the owned worker.
import assert from "node:assert/strict";
import { execFile } from "node:child_process";
import { readFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { promisify } from "node:util";
import { fileSha256, sha256 } from "./file-sha256.mjs";

const run = promisify(execFile);
const thumbs = fileURLToPath(new URL("../../apps/web/public/style-thumbs/", import.meta.url));
const portraits = [
  ["full-body-photo", "48eb36652563147a989e991d010aa7af08c534512919dadc07b84eba0f084487"],
  ["photoshoot", "c59541f34ad76b5faed05b78f11996c2ebdc1cd9dfb81d15d50057a356950591"],
  ["supermodel-photoshoot", "2cda939c046ceaa962d8d5a20aa1842c062b8600505b2b27d863e368eab6fa94"],
  ["linkedin-pro-portrait", "66f7bce2a2a4583f28616df1fe66e45010f83aedd4baff3a3dbf36b5fca545f3"],
  ["gq-photoshoot", "70f21ad362d9cc2287cbe156dceb710b2f83968792f982a291c86a3d1b2df201"],
  ["backlit-portraiture", "4bad4c5b6331368e727b28c14025df4253f9476f769d715c98c2740bc0c822cc"],
];
export const SCAIL_REVISION = "ce88cfdb1008f395e9c820e525e6db7b6695f7b3";
// These are the six files used by the native fit profile, in its source-seal order.
const weights = {
  "config.json": "7ccd156e99a51a80f93143acfc53a1f1edd0ca0f0f095cbdf35cc60345729061",
  "tokenizer.json": "6e197b4d3dbd71da14b4eb255f4fa91c9c1f2068b20a2de2472967ca3d22602b",
  "dit.safetensors": "1a46f805cf22dc74efeab0d4eae6a21ee48f4234281c4d09cf0755378661f614",
  "t5_encoder.safetensors": "70e44fce2fef5e7bf70a519f65a6dfa3a16a000f489c8c492c913eea77167f0d",
  "clip.safetensors": "90ddb7ff2d56e640b8ef991f498ec1f597fd86794e5c4790eb91622c5f551a5d",
  "vae.safetensors": "42159a8b571dbeb3ea40327b88a6161a5342c0511202af7c031360629757163d",
};
export const scailMembers = Object.keys(weights).map((name) => "q4/" + name).sort();
export const scailArtifactSha256 = sha256(Buffer.concat([Buffer.from("scail2-plural-weights-v1"),
  ...Object.values(weights).map((digest) => Buffer.from(digest, "hex"))]));
export function verifyScailClaim(files) {
  const actual = files.map(({ relativePath, sha256 }) => [relativePath.replaceAll("\\", "/"), sha256]).sort();
  assert.deepEqual(actual, Object.entries(weights).map(([name, digest]) => ["q4/" + name, "sha256:" + digest]).sort(),
    "claimed SCAIL weights differ from the fit fixture");
}
export const SAM3 = { repo: "SceneWorks/sam3-mlx", revision: "3ed10b164a755c00b1a4d671dde95719c127e1a7",
  files: { "model.safetensors": "6d06f0a5f84e435071fe6603e61d0b4cc7b40e0d39d487cfd4d67d8cc11cc14a",
    "tokenizer.json": "6d9109cc838977f3ca94a379eec36aecc7c807e1785cd729660ca2fc0171fb35" } };
// The two native MLX boundaries and the five native CUDA counts qualify these product witnesses.
export function scailFitEvidence(pool, backend) {
  const platform = backend === "mlx" ? "macos" : "windows";
  assert.ok(Array.isArray(pool), "invalid packaged SCAIL fit rows");
  const rows = pool.filter((row) => row.platform === platform && row.witness?.axes?.backend === backend
    && row.witness.axes.artifact_sha256 === scailArtifactSha256 && row.witness.axes.effective_tier === "q4"
    && row.witness.axes.artifact_tier === "q4" && row.witness.axes.width === 832 && row.witness.axes.height === 480
    && row.device && row.runtime && /^[a-f0-9]{64}$/.test(row.evidence_sha256));
  const counts = backend === "mlx" ? [2, 6] : [2, 3, 4, 5, 6];
  const complete = rows.filter((row) => counts.every((count) => rows.some((other) =>
    other.device === row.device && other.runtime === row.runtime && other.witness.axes.references === count)));
  assert.ok(complete.length, `no packaged native SCAIL q4 evidence covering the required counts on the same ${platform} host configuration`);
  return complete.map((row) => {
    const acceleratorBytes = backend === "mlx" ? row.witness.gross_unified_bytes : row.witness.gross_device_bytes;
    assert.ok(Number.isSafeInteger(acceleratorBytes) && acceleratorBytes > 0, "native SCAIL device measurement missing");
    return { device: row.device, runtime: row.runtime, references: row.witness.axes.references, sha256: row.evidence_sha256, acceleratorBytes };
  });
}
// A cold worker has zero provider credit. Cap below every qualifying six-reference capture on
// its device/runtime so an alternative, smaller quotation cannot admit the negative request.
export function scailRefusalCapGiB(events, evidence) {
  const cold = events.find((e) => e.stage === "cold_load" && e.admitted);
  const bound = evidence.find((row) => row.sha256 === cold?.evidenceSha256);
  assert.ok(bound && cold.providerCreditBytes === 0, "no accepted cold physical capture");
  const alternatives = evidence.filter((row) => row.device === bound.device && row.runtime === bound.runtime && row.references >= 6);
  assert.ok(alternatives.length, "missing six-reference physical capture");
  return (Math.min(cold.requiredAcceleratorBytes, ...alternatives.map((row) => row.acceleratorBytes)) - 1) / (1024 ** 3);
}
export async function verifySam3(dir) {
  for (const [file, digest] of Object.entries(SAM3.files)) assert.equal(await fileSha256(path.join(dir, file)), digest, `wrong pinned SAM3 ${file}`);
}

export const scailRoutes = ["mlx", "cuda"].flatMap((backend) => [2, 3, 4, 5, 6, "refuse"].map((count) => {
  const refuse = count === "refuse", references = refuse ? 6 : count;
  return { id: `${backend}-scail2-q4-animate-${references}${refuse ? "-insufficient" : ""}`, scailCandidate: true,
    backend, references, refuse, engineFrames: 45, savedFrames: backend === "mlx" ? 48 : 45, group: null,
    model: "scail2_14b", mode: "animate_character", kind: "video", tier: "q4", endpoint: "/api/v1/video/jobs", claimMinutes: 10, timeoutMinutes: 45,
    body: { model: "scail2_14b", mode: "animate_character", prompt: "The people shift their weight and turn to the camera.",
      negativePrompt: "", width: 832, height: 480, fps: 16, duration: 3, seed: 33402, loras: [],
      advanced: { mlxQuantize: 4, steps: 4, guidanceScale: 5, sampler: "default", lightning: false } } };
}));
export function scailBinding(route, sources) {
  return { referenceAssetIds: sources ? sources.references.map((item) => item.id)
    : portraits.slice(0, route.references).map(([name]) => `<${name}>`), sourceClipAssetId: sources?.clip.id ?? "<driving-clip>" };
}

export async function scailMedia(route, dir, ffmpeg) {
  const inputs = [], refs = [];
  const ff = (args) => run(ffmpeg, ["-hide_banner", "-loglevel", "error", "-nostdin", "-n", ...args], { timeout: 120_000, windowsHide: true });
  for (const [name, digest] of portraits.slice(0, route.references)) {
    const source = path.join(thumbs, name + ".png");
    assert.equal(await fileSha256(source), digest, `reference changed: ${name}`);
    const dest = path.join(dir, name + ".png");
    await ff(["-i", source, "-vf", "scale=480:480,pad=832:480:(ow-iw)/2:0:color=gray", "-frames:v", "1", "-threads", "1", dest]);
    refs.push(dest); inputs.push("-i", source);
  }
  // Match the number of visible people to the ordered reference count. A small pan produces
  // distinct driving frames using only the photographic samples shipped with SceneWorks.
  const cols = route.references <= 3 ? route.references : 3, tile = route.references <= 2 ? 400 : 224;
  const labels = refs.map((_, i) => `[c${i}]`);
  const tiles = refs.map((_, i) => `[${i}:v]scale=${tile}:${tile}${labels[i]}`);
  const layout = refs.map((_, i) => `${(i % cols) * tile}_${Math.floor(i / cols) * tile}`).join("|");
  const filter = [...tiles, `${labels.join("")}xstack=inputs=${refs.length}:layout=${layout}:fill=gray,pad=832:480:(ow-iw)/2:(oh-ih)/2:color=gray,zoompan=z=1.04:x='(iw-iw/zoom)*on/44':y='(ih-ih/zoom)/2':d=45:fps=16:s=832x480`].join(";");
  const clip = path.join(dir, "driving.mp4");
  await ff([...inputs, "-filter_complex_threads", "1", "-filter_complex", filter, "-frames:v", "45", "-an", "-c:v", "libx264",
    "-pix_fmt", "yuv420p", "-threads", "1", "-map_metadata", "-1", clip]);
  return { refs, clip };
}

export async function uploadScailMedia(route, endpoint, project, media, decodePng) {
  const upload = async (file, kind, mime) => {
    const bytes = await readFile(file), form = new FormData(), name = path.basename(file);
    form.append("file", new Blob([bytes], { type: mime }), name);
    const response = await endpoint.exchange("POST", `/api/v1/projects/${project}/assets`, form, { multipart: true });
    const asset = response.body, rel = asset?.file?.path;
    assert.ok(response.status < 300 && asset?.id && asset.projectId === project && asset.type === kind && typeof rel === "string"
      && rel.startsWith("assets/") && !rel.includes("\\") && rel.split("/").every((s) => s && s !== "." && s !== ".."), `invalid imported asset: ${name}`);
    const served = await endpoint.served(`/api/v1/projects/${project}/files/${rel.split("/").map(encodeURIComponent).join("/")}`);
    const digest = sha256(bytes);
    assert.equal(sha256(served.bytes), digest, `served input differs: ${name}`);
    if (kind === "image") {
      const { width, height, rgba } = decodePng(bytes);
      assert.deepEqual([width, height], [832, 480]);
      const rgb = Buffer.allocUnsafe(width * height * 3);
      for (let i = 0, j = 0; i < rgba.length; i += 4, j += 3) { rgb[j] = rgba[i]; rgb[j+1] = rgba[i+1]; rgb[j+2] = rgba[i+2]; }
      return { id: asset.id, name, sha256: digest, rgbSha256: sha256(rgb) };
    }
    return { id: asset.id, name, sha256: digest };
  };
  const references = [];
  for (const ref of media.refs) references.push(await upload(ref, "image", "image/png"));
  const clip = await upload(media.clip, "video", "video/mp4");
  assert.equal(new Set(references.map((r) => r.id)).size, route.references);
  return { references, clip };
}

export function scailFitEvents(text, route, jobId, admitted, sources, evidence) {
  const events = [];
  for (const line of text.replaceAll(/\u001b\[[0-9;]*m/g, "").split("\n")) {
    if (!line.includes("scail2_plural_fit")) continue;
    let entry;
    try { entry = JSON.parse(line); } catch { continue; }
    const fields = entry;
    if (fields?.event !== "scail2_plural_fit" || fields.jobId !== jobId) continue;
    assert.equal(fields.referenceRgbSha256, sources.references.map((r) => r.rgbSha256).join(","), "worker reference order or RGB pixels differ");
    const backend = route.backend === "mlx" ? "mlx" : "candle";
    assert.equal(fields.backend, backend);
    assert.equal(fields.tier, "q4");
    assert.equal(fields.artifactTier, "q4");
    assert.equal(fields.artifactSha256, scailArtifactSha256);
    assert.ok(evidence.some((row) => row.sha256 === fields.evidenceSha256), "worker fit is unbound to the packaged physical capture");
    for (const [key, expected] of Object.entries({ width: 832, height: 480, frames: 45, referenceCount: route.references }))
      assert.equal(Number(fields[key]), expected, key);
    const receipt = fields.carrierReceipt;
    assert.ok(receipt?.startsWith(`scail2-carrier-v2:backend-${backend}:task-animation:composite-832x480:characters-${route.references}:`));
    const pairs = Array.from({ length: route.references }, (_, i) => `reference-${i}-1:0:832x480:reference-mask-${i}-1:0:832x480`).join(":");
    const frames = Array.from({ length: 45 }, (_, i) => `${i}:832x480`).join(",");
    assert.ok(receipt.includes(`:${pairs}:driving-frames-45:${frames}:driving-mask-45:${frames}:`), "incorrect ordered character or driving axes");
    assert.match(receipt, /frame-0\+scail2-carrier-request-seal-v2-[a-f0-9]{64}$/);
    for (const key of ["requiredAcceleratorBytes", "availableAcceleratorBytes", "requiredPhysicalHostBytes", "availablePhysicalHostBytes"]) {
      assert.ok(Number.isSafeInteger(fields[key]) && (key.startsWith("available") ? fields[key] >= 0 : fields[key] > 0), key);
    }
    assert.ok(Number.isSafeInteger(fields.providerCreditBytes) && fields.providerCreditBytes >= 0, "provider credit unavailable");
    if (fields.stage === "cold_load") assert.equal(fields.providerCreditBytes, 0);
    const fits = fields.requiredAcceleratorBytes <= fields.availableAcceleratorBytes && fields.requiredPhysicalHostBytes <= fields.availablePhysicalHostBytes;
    assert.equal(fields.admitted, fits);
    events.push(fields);
  }
  if (admitted) {
    assert.ok(events.some((e) => e.stage === "cold_load" && e.admitted === true) && events.some((e) => e.stage === "execution" && e.admitted === true), "missing positive cold-load or execution budget evidence");
    assert.equal(new Set(events.filter((e) => e.admitted).map((e) => e.evidenceSha256)).size, 1, "worker changed its admitted physical evidence");
    assert.equal(new Set(events.map((e) => e.carrierReceipt)).size, 1, "admission and execution received different carriers");
  } else {
    assert.ok(events.length && events.every((e) => e.stage === "cold_load" && e.admitted === false) && events.some((e) => e.requiredAcceleratorBytes > e.availableAcceleratorBytes), "no observed accelerator-domain refusal");
  }
  return events;
}

export function verifyScailOutput(route, content, receipt, jobId, outputId, recipe, sources, metrics) {
  const backend = route.backend === "mlx" ? "mlx" : "candle";
  assert.equal(content.method, "ffmpeg_rgba_frames+s16le_audio+ffprobe_streams");
  const video = content.shape.video;
  assert.deepEqual([video.width, video.height, video.avgFrameRate, video.decodedFrames], [832, 480, "16/1", route.savedFrames]);
  assert.equal(content.shape.audio, null);
  const stride = 832 * 480 * 4;
  const digests = Array.from({ length: route.savedFrames }, (_, i) => sha256(content.buffer.subarray(i * stride, (i + 1) * stride)));
  assert.equal(content.buffer.length, stride * route.savedFrames);
  let varied = false;
  for (let i = 4; i < stride && !varied; i += 4) varied = content.buffer[i] !== content.buffer[0] || content.buffer[i+1] !== content.buffer[1] || content.buffer[i+2] !== content.buffer[2];
  assert.ok(varied, "first decoded frame has a single color");
  assert.ok(new Set(digests).size > 1, "all decoded frames are identical");
  assert.equal(receipt?.version, 1); assert.equal(receipt.jobId, jobId); assert.equal(receipt.acceptedBackend, backend);
  assert.equal(receipt.outputs.length, 1);
  const output = receipt.outputs[0]; assert.deepEqual(output.artifactIds, [outputId]);
  const actual = output.execution ?? receipt.execution;
  assert.equal(actual?.state, "resolved");
  const identity = actual.identity; assert.equal(identity?.provider_id, "scail2_14b"); assert.equal(identity.backend, backend);
  assert.equal(identity.schedule.state, "resolved"); assert.equal(identity.schedule.value.length, 1);
  const schedule = identity.schedule.value[0];
  const digest = schedule.digest ?? schedule.schedule;
  assert.equal(digest.count, 5); assert.match(digest.sha256, /^[a-f0-9]{64}$/);
  assert.equal(identity.sampling.state, "resolved"); assert.equal(identity.sampling.value.length, 1);
  const [sample] = identity.sampling.value;
  assert.equal(backend === "mlx" ? sample.solver : sample, backend === "mlx" ? "UniPC" : "wan_flow_unipc_bh2_order2_v1");
  assert.equal(identity.randomness?.kind, "draws"); assert.equal(identity.randomness.draws.length, 1);
  assert.equal(identity.randomness.draws[0].seed, route.body.seed);
  assert.equal(recipe.adapter, backend + "_scail2"); assert.equal(recipe.prompt, route.body.prompt);
  assert.deepEqual(recipe.loras, []);
  assert.deepEqual([recipe.model, recipe.mode, recipe.seed], [route.model, route.mode, route.body.seed]);
  assert.equal(recipe.rawAdapterSettings?.realModelInference, true);
  assert.equal(recipe.rawAdapterSettings.frameCount, route.savedFrames);
  const saved = recipe.normalizedSettings;
  const binding = scailBinding(route, sources);
  assert.equal(saved?.sourceClipAssetId, binding.sourceClipAssetId); assert.deepEqual(saved?.referenceAssetIds, binding.referenceAssetIds);
  for (const [key, expected] of Object.entries({ model: route.model, backend: route.backend, steps: 4, guidanceScale: 5, seed: route.body.seed, width: 832, height: 480 }))
    assert.equal(metrics[key], expected, key);
  assert.ok(Number.isFinite(metrics.totalMs) && metrics.totalMs > 0);
  return { provider: identity.provider_id, backend, schedule: digest, sampling: identity.sampling.value, randomness: identity.randomness,
    decodedFrameSha256: digests, workerTotalMs: metrics.totalMs, workerPeakMemoryBytes: metrics.peakMemoryBytes ?? null };
}
