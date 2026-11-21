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
const digestPattern = /^[a-f0-9]{64}$/;
function byteCount(value) {
  assert.ok(Number.isSafeInteger(value) && value >= 0, "invalid physical byte count");
  return value;
}

// Mirrors the worker's evidence::Envelope::price. Product fixtures use the captured geometry,
// four steps and uniformly sized inputs, so their native price has no resize or schedule delta.
function physicalPrice(envelope, floor) {
  const observed = byteCount(envelope.observed_peak_bytes);
  assert.ok(observed > 0, "missing physical observation");
  let price;
  if (envelope.basis === "guarded_ceiling") {
    const ceiling = byteCount(envelope.guard_ceiling_bytes);
    assert.ok(ceiling >= observed, "guard ceiling below observation");
    price = byteCount(ceiling + byteCount(envelope.unsampled_overshoot_upper_bytes));
  } else {
    assert.equal(envelope.basis, "measured_aggregate");
    const classified = byteCount(envelope.classified_peak_bytes);
    assert.ok(classified >= observed && digestPattern.test(envelope.review_sha256), "invalid reviewed aggregate");
    price = byteCount(classified + byteCount(envelope.co_live_extra_bytes) + byteCount(envelope.uncertainty_allowance_bytes));
  }
  assert.ok(price >= byteCount(floor), "physical price below native floor");
  return price;
}

// One native single-window request. Scail2FitProfile::from_plan packs the 12 latent planes,
// the quarter-resolution pose and the ordered references on the 16px patch grid.
function verifyProductWindow(axes, mlx) {
  const patches = (832 / 16) * (480 / 16);
  const tokens = patches * (12 + axes.references) + (patches / 4) * 12;
  const chunks = Math.ceil(tokens / 8192);
  const scoreStride = 40 * tokens;
  // Candle's native AttentionBudget gives each query block at most 1e9 score elements.
  const rows = mlx ? 0 : Math.floor(1_000_000_000 / scoreStride);
  assert.deepEqual(axes.windows, [{
    latent_frames: 12, history_frames: 0, decoded_frames: mlx ? 48 : 45, tokens,
    mlx_unsplit_ffn: mlx, mlx_ffn_chunks: chunks, mlx_ffn_tail: tokens - (chunks - 1) * 8192,
    self_attention_query_rows: rows, self_attention_score_elements: rows * scoreStride,
  }], "capture has a different native SCAIL window");
}

// Use the same two immutable approvals compiled by the candidate worker before deriving a
// standalone ceiling from its package. The worker independently checks these when it quotes.
export function scailPackageDigests(index, coverage, source) {
  function approved(bytes, constant) {
    const declarations = [...source.matchAll(new RegExp(`\\bconst ${constant}: &str\\s*=\\s*"([a-f0-9]{64})"\\s*;`, "g"))];
    assert.equal(declarations.length, 1, `missing or ambiguous SCAIL approval ${constant}`);
    const digest = sha256(bytes);
    assert.equal(digest, declarations[0][1], `candidate SCAIL package differs from ${constant}`);
    return digest;
  }
  return { indexSha256: approved(index, "APPROVED_INDEX_SHA256"), coverageSha256: approved(coverage, "APPROVED") };
}

export function scailFitEvidence(pool, backend, { compiledPin, coverage, closures, host }) {
  const mlx = backend === "mlx", platform = mlx ? "macos" : "windows";
  const counts = mlx ? [2, 6] : [2, 3, 4, 5, 6];
  assert.match(compiledPin, /^[a-f0-9]{40}$/);
  assert.ok(Array.isArray(pool) && Array.isArray(coverage), "invalid packaged SCAIL fit rows");
  const closure = closures?.providers?.[(mlx ? "mlx" : "candle") + ":scail2_14b"];
  const rows = [];
  for (const row of pool) {
    if (row.platform !== platform) continue;
    if (row.inference_revision !== compiledPin && !coverage.some((c) =>
      c.capture_revision === row.inference_revision && c.target_revision === compiledPin && c.backend === backend
      && closures?.inferenceRevision === compiledPin && digestPattern.test(c.tree_diff_sha256)
      && digestPattern.test(c.provider_digest) && c.provider_digest === closure?.digest
      && digestPattern.test(c.shared_contract_digest) && c.shared_contract_digest === closure?.sharedContractDigest)) continue;
    const a = row.axes;
    if (a?.backend !== backend || a.artifact_sha256 !== scailArtifactSha256 || a.effective_tier !== "q4"
      || a.artifact_tier !== "q4" || a.width !== 832 || a.height !== 480 || !Number.isInteger(a.references) || a.references < 2 || a.references > 6) continue;
    const accepted = row.recovery_certificate_sha256 === undefined ? row.accepted === true :
      !mlx && row.accepted === false && a.references === 6 && digestPattern.test(row.recovery_certificate_sha256)
      && row.artifact_id === 14719969057 && row.receipt_sha256 === "750313a3cdb89dd4a0e4e49b31935949301b309fe9410f2bf0ce47cb4628575f"
      && row.inference_revision === "7f6c4cca3fc053030bd71f0bfce6886adb762eb6";
    assert.ok(accepted && row.post_execution_verified === true && row.guard_exit_code === 0
      && Number.isSafeInteger(row.artifact_id) && row.artifact_id > 0 && digestPattern.test(row.receipt_sha256)
      && digestPattern.test(row.guard_policy_sha256) && /^[a-f0-9]{40}$/.test(row.inference_revision)
      && row.device?.trim() && row.runtime?.trim(), "invalid projected physical capture");
    assert.deepEqual(row.prompt_tokens, [512, 512]);
    verifyProductWindow(a, mlx);
    assert.ok(a.classifier_free_guidance === true && a.history_order === 4 && a.mlx_dpmpp === false
      && a.source_bytes === (2 * a.references + 90) * 832 * 480 * 3
      && a.driving_stack_bytes === 45 * 832 * 480 * 24 && a.source_resize_bytes === 0 && a.schedule_bytes === 5 * 64
      && a.load_policy === (mlx ? "None" : "Some(Q4)") + "/Bf16/Resident/EagerMaterialization"
      && Array.isArray(a.execution_policy), "capture does not cover the fixed product inputs");
    if (host && (row.device !== host.device || row.runtime !== host.runtime ||
      !a.execution_policy.every(({ name, value }) => (host.env[name] ?? null) === value))) continue;
    const b = row.bounds;
    assert.equal(b.measurement, mlx ? "darwin_lifetime_footprint" : "windows_job_and_wddm");
    if (row.recovery_certificate_sha256 !== undefined || b.device_interval !== undefined) {
      assert.ok(!mlx && b.device_interval === "load_through_completed_rgb" && b.device.basis === "measured_aggregate",
        "missing empirical CUDA output interval");
    }
    const adapter = b.capture_adapter;
    if (!mlx && host) assert.ok(adapter && host.captureAdapter && adapter.cuda_uuid === host.captureAdapter.cuda_uuid
      && adapter.luid === host.captureAdapter.luid && adapter.node_mask === host.captureAdapter.node_mask,
      "no packaged native proof on a single worker-eligible CUDA adapter");
    if (mlx) assert.ok(byteCount(b.mlx_decode_budget_bytes) > 0, "missing decoder budget");
    const price = {
      grossHostBytes: mlx ? 0 : physicalPrice(b.host, a.floor_host_bytes),
      grossDeviceBytes: mlx ? 0 : physicalPrice(b.device, a.floor_device_bytes),
      grossUnifiedBytes: mlx ? physicalPrice(b.unified, a.floor_unified_bytes) : 0,
    };
    if (mlx) assert.ok(b.mlx_decode_budget_bytes <= price.grossUnifiedBytes, "decoder budget exceeds unified price");
    // The worker authenticates the index, hardware and compatible compiled pin before quoting.
    const nativePolicy = JSON.stringify([row.device, row.runtime, ...(!mlx ? [adapter.cuda_uuid, adapter.luid, adapter.node_mask] : []), a.load_policy, a.execution_policy.map(({ name, value }) => [name, value])]);
    // Each Windows count has its own complete measurement. Every cross-revision donor above
    // must identify the compiled pin's provider and shared contract before counts can be joined.
    const campaign = JSON.stringify([nativePolicy, mlx ? row.inference_revision : null, row.guard_policy_sha256, b.mlx_decode_budget_bytes ?? null]);
    rows.push({ device: row.device, runtime: row.runtime, ...(!mlx ? { captureAdapter: adapter } : {}), nativePolicy, campaign, references: a.references,
      ...(row.recovery_certificate_sha256 ? {recoveryCertificateSha256: row.recovery_certificate_sha256} : {}),
      inferenceRevision: row.inference_revision, sha256: row.receipt_sha256, artifactId: row.artifact_id, price, acceleratorBytes: mlx ? price.grossUnifiedBytes : price.grossDeviceBytes });
  }
  const complete = rows.filter((row) => counts.every((n) => rows.some((r) => r.campaign === row.campaign && r.references === n)));
  assert.ok(complete.length, `no packaged native SCAIL q4 evidence covering the required counts on the same ${platform} policy`);
  // Windows acceptance claims five exact counts under a single guard and device policy.
  if (!mlx) {
    for (const row of complete) {
      assert.ok(rows.every((other) => other.nativePolicy !== row.nativePolicy || other.campaign === row.campaign),
        "worker-eligible CUDA counts have different capture guards");
    }
    return complete;
  }
  // The worker can also try an independently reviewed, cheaper quotation for this native policy.
  // Keep such rows available for exact donor matching and the isolated negative budget.
  return rows.filter((row) => complete.some((r) => r.nativePolicy === row.nativePolicy));
}

function donor(event, evidence, references) {
  // For these verified windows a larger MLX tail dominates the smaller one in the same four
  // unsplit FFN chunks. On CUDA a larger token count has fewer query rows and cannot dominate.
  const covers = event.backend === "mlx" ? event.evidenceReferences >= references : event.evidenceReferences === references;
  const matched = evidence.filter((row) => covers && row.artifactId === event.evidenceArtifactId
    && row.sha256 === event.evidenceSha256 && row.references === event.evidenceReferences);
  assert.equal(matched.length, 1, "worker must identify one physical capture covering the requested reference count");
  const [row] = matched;
  for (const [key, price] of Object.entries(row.price)) assert.equal(event[key], price, `worker ${key} differs from the projected price`);
  return row;
}

// An insufficient run starts cold. Its budget must also refuse a smaller eligible quotation.
export function scailRefusalCapGiB(events, evidence) {
  const cold = events.find((e) => e.stage === "cold_load" && e.admitted);
  assert.ok(cold?.providerCreditBytes === 0 && cold.referenceCount === 6, "no admitted cold six-reference capture");
  const bound = donor(cold, evidence, 6);
  const alternatives = evidence.filter((row) => row.nativePolicy === bound.nativePolicy && row.references === 6);
  return (Math.min(cold.requiredAcceleratorBytes, ...alternatives.map((row) => row.acceleratorBytes)) - 1) / (1024 ** 3);
}
// The caller restricted these authenticated rows to the actual device, runtime and worker
// policy. Count six can only use a six-reference donor on either backend.
export function scailPackagedRefusalCapBytes(evidence) {
  const prices = evidence.filter((r) => r.references === 6).map((r) => byteCount(r.acceleratorBytes));
  assert.ok(prices.length && Math.min(...prices) > 1, "no usable six-reference accelerator price on this host");
  return Math.min(...prices) - 1;
}

export function verifyScailRefusal(fit, capBytes, message) {
  assert.ok(fit.length && fit.every((e) => e.availableAcceleratorBytes <= capBytes), "worker refusal did not use the selected accelerator ceiling");
  // The worker returns the last refused quotation when all candidate donors fail.
  const last = fit.at(-1);
  // On Candle the cold gate returns Unsupported through the provider cache's load boundary.
  // scailFitEvents has already bound the logged backend to the selected route.
  const prefix = last.backend === "candle" ? "video load failed: unsupported: " : "";
  assert.ok(last.requiredAcceleratorBytes > last.availableAcceleratorBytes &&
    message === `${prefix}SCAIL-2 plural fit requires ${last.requiredAcceleratorBytes} accelerator bytes, only ${last.availableAcceleratorBytes} are available`,
    "missing SCAIL budget refusal for the observed byte counts");
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
    const measured = donor(fields, evidence, route.references);
    for (const [key, expected] of Object.entries({ width: 832, height: 480, frames: 45, referenceCount: route.references }))
      assert.equal(Number(fields[key]), expected, key);
    const parts = fields.carrierReceipt?.split("+");
    assert.equal(parts?.length, 3, "incomplete or unexpected SCAIL admission overlay");
    const [providerMode, carrier, seal] = parts;
    assert.equal(providerMode, "provider_video_mode:animation");
    const pairs = Array.from({ length: route.references }, (_, i) => `reference-${i}-1:0:832x480:reference-mask-${i}-1:0:832x480`).join(":");
    const frames = Array.from({ length: 45 }, (_, i) => `${i}:832x480`).join(",");
    assert.equal(carrier, `scail2-carrier-v2:backend-${backend}:task-animation:composite-832x480:characters-${route.references}:${pairs}` +
      `:driving-frames-45:${frames}:driving-mask-45:${frames}:strength-3f800000:mode-FaceOnly:frame-0`);
    assert.match(seal, /^scail2-carrier-request-seal-v2-[a-f0-9]{64}$/);
    for (const key of ["requiredAcceleratorBytes", "availableAcceleratorBytes", "requiredPhysicalHostBytes", "availablePhysicalHostBytes"]) {
      assert.ok(Number.isSafeInteger(fields[key]) && (key.startsWith("available") ? fields[key] >= 0 : fields[key] > 0), key);
    }
    assert.ok(Number.isSafeInteger(fields.providerCreditBytes) && fields.providerCreditBytes >= 0, "provider credit unavailable");
    if (fields.stage === "cold_load") assert.equal(fields.providerCreditBytes, 0);
    assert.ok(fields.requiredAcceleratorBytes >= measured.acceleratorBytes - fields.providerCreditBytes, "worker underpriced accelerator allocation");
    const physical = backend === "mlx" ? measured.price.grossUnifiedBytes - fields.providerCreditBytes : measured.price.grossHostBytes;
    assert.ok(fields.requiredPhysicalHostBytes >= physical - (2 * route.references + 90) * 832 * 480 * 3, "worker underpriced physical host allocation");
    const fits = fields.requiredAcceleratorBytes <= fields.availableAcceleratorBytes && fields.requiredPhysicalHostBytes <= fields.availablePhysicalHostBytes;
    assert.equal(fields.admitted, fits);
    events.push(fields);
  }
  if (admitted) {
    assert.ok(events.some((e) => e.stage === "cold_load" && e.admitted === true) && events.some((e) => e.stage === "execution" && e.admitted === true), "missing positive cold-load or execution budget evidence");
    assert.equal(new Set(events.filter((e) => e.admitted).map((e) => JSON.stringify([e.evidenceArtifactId, e.evidenceSha256, e.evidenceReferences]))).size, 1, "worker changed its admitted physical evidence");
  } else {
    assert.ok(events.length && events.every((e) => e.stage === "cold_load" && e.admitted === false) && events.some((e) => e.requiredAcceleratorBytes > e.availableAcceleratorBytes), "no observed accelerator-domain refusal");
  }
  assert.equal(new Set(events.map((e) => e.carrierReceipt)).size, 1, "fit events received different carriers");
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
