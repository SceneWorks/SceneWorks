import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";

export const KREA_ALLOCATION_POOL_BYTES = 64 * 1024 ** 3;
export const KREA_ALLOCATION_MINUTES = 55;
export const KREA_WIDE_IM2COL_PIN = "cef3e31c7af51003892c1e99f1856c36ccb93b06";
export const KREA_ALLOCATION_BASE_SIZE = 1120;
export const KREA_ALLOCATION_FINAL_SIZE = 2240;
const HOOK_FILE = new URL("./ci/krea-allocation-probe.rs.in", import.meta.url);
const OBSERVE = `    let krea_allocation_enabled = generator.descriptor().id == "krea_2_turbo"
        && std::env::var_os("SCENEWORKS_CUDA_PROOF_POOL_BYTES").is_some();
    let (krea_alloc_width, krea_alloc_height) = (request.width, request.height);
    if krea_allocation_enabled {
        tracing::info!(
            event = "sgaf336_krea_allocation_admission",
            width = krea_alloc_width,
            height = krea_alloc_height,
            reference_count = request.image_reference_count(),
            context_present = context.is_some(),
            context_width = context.map(|context| context.geometry.width).unwrap_or(0),
            context_height = context.map(|context| context.geometry.height).unwrap_or(0),
            context_reference_count = context.map(|context| context.geometry.reference_count).unwrap_or(0),
            context_revision = context.map(|context| context.evidence_revision.as_str()).unwrap_or(""),
            predicted_additional_bytes = context.map(|context| context.predicted_peak_bytes).unwrap_or(0),
            available_additional_bytes = context.map(|context| context.budget.effective_bytes()).unwrap_or(0),
            "Actual Candle Krea provider pass and its additional-device-byte admission basis"
        );
    }
    let mut observe_krea_allocation = |progress| {
        if krea_allocation_enabled {
            match &progress {
                gen_core::Progress::Step { current: 1, .. } => krea_allocation_pool(krea_alloc_width, krea_alloc_height, "first_step_after_conditioning"),
                gen_core::Progress::Decoding => krea_allocation_pool(krea_alloc_width, krea_alloc_height, "before_decode"),
                _ => {}
            }
        }
        on_progress(progress);
    };
    let on_progress = &mut observe_krea_allocation;
`;

function replaceOnce(source, anchor, replacement) {
  assert.equal(source.split(anchor).length, 2, `Krea allocation needs one ${anchor.trim()} source boundary`);
  return source.replace(anchor, replacement);
}

export async function kreaAllocationOverlay(source, anchor) {
  const [prefix, rest] = anchor.split("\n");
  let result = replaceOnce(source, anchor, `${prefix}\n${OBSERVE}${rest}`);
  const passReturn = (indent) => `${indent}if krea_allocation_enabled && result.is_ok() {
${indent}    krea_allocation_pool(krea_alloc_width, krea_alloc_height, "after_provider_return");
${indent}}`;
  result = replaceOnce(result, "        return run(generator, request, None, on_progress);", `        let result = run(generator, request, None, on_progress);\n${passReturn("        ")}\n        return result;`);
  const scopeReturn = "    if matches!(result, Err(gen_core::Error::ProcessPoisoned(_))) {";
  result = replaceOnce(result, scopeReturn, `${passReturn("    ")}\n${scopeReturn}`);
  return `${result}\n${await readFile(HOOK_FILE, "utf8")}`;
}

export function kreaAllocationReceipt(log, memory, hostCapBytes) {
  const entries = log.split(/\r?\n/).flatMap((line) => {
    try {
      const { fields } = JSON.parse(line);
      return fields ? [fields] : [];
    } catch { return []; }
  });
  const admissions = entries.filter(({ event }) => event === "sgaf336_krea_allocation_admission");
  assert.deepEqual(admissions.map(({ width, height, reference_count }) => [width, height, reference_count]), [[KREA_ALLOCATION_BASE_SIZE, KREA_ALLOCATION_BASE_SIZE, 0], [KREA_ALLOCATION_FINAL_SIZE, KREA_ALLOCATION_FINAL_SIZE, 1]], "Candle worker did not expose both actual provider pass shapes");
  for (const pass of admissions) {
    assert.ok(pass.context_present === true && pass.context_revision === "imported-krea-hires-floor", "actual Candle context is missing");
    assert.deepEqual([pass.context_width, pass.context_height, pass.context_reference_count], [pass.width, pass.height, pass.reference_count], "Candle context geometry or reference count differs from the provider pass");
    const { predicted_additional_bytes, available_additional_bytes } = pass;
    assert.ok(Number.isSafeInteger(predicted_additional_bytes) && predicted_additional_bytes > 0 && Number.isSafeInteger(available_additional_bytes) && predicted_additional_bytes <= available_additional_bytes, "Candle additional-device-byte price exceeds admission");
  }
  const applied = entries.filter(({ event, engine }) => event === "generation_memory_policy_applied" && engine === "krea_2_turbo");
  assert.equal(applied.length, 2, "candidate must apply both real provider scopes");
  assert.ok(applied.every(({ scope_configured }) => scope_configured === true), "candidate rendered without a configured Candle scope");

  assert.ok(!entries.some(({ event }) => event === "sgaf336_krea_allocation_pool_unavailable"), "native pool snapshot is unavailable");
  const pool = entries.filter(({ event }) => event === "sgaf336_krea_allocation_pool");
  const boundaries = ["first_step_after_conditioning", "before_decode", "after_provider_return"];
  const expected = [KREA_ALLOCATION_BASE_SIZE, KREA_ALLOCATION_FINAL_SIZE].flatMap((size) => boundaries.map((boundary) => [size, size, boundary]));
  assert.deepEqual(pool.map(({ width, height, boundary }) => [width, height, boundary]), expected, "wanted a cumulative pool measurement at each pass boundary");
  let usedHigh = 0, reservedHigh = 0;
  for (const entry of pool) {
    const { logical_ordinal, cap_bytes, used_bytes, used_high_bytes, reserved_bytes, reserved_high_bytes } = entry;
    assert.equal(logical_ordinal, 0);
    assert.equal(cap_bytes, KREA_ALLOCATION_POOL_BYTES);
    assert.ok([used_bytes, used_high_bytes, reserved_bytes, reserved_high_bytes].every((value) => Number.isSafeInteger(value) && value >= 0));
    assert.ok(used_high_bytes > 0 && used_bytes <= used_high_bytes && reserved_bytes <= reserved_high_bytes && used_high_bytes <= reserved_high_bytes && reserved_high_bytes <= cap_bytes, "private pool high-water and its cap do not agree");
    assert.ok(used_high_bytes >= usedHigh && reserved_high_bytes >= reservedHigh, "cumulative CUDA high-water counters regressed");
    usedHigh = used_high_bytes;
    reservedHigh = reserved_high_bytes;
  }
  const { gpuUuid, hostLifetimePeakBytes, ownedGpuPeakBytes, ownedGpuSamples, ownedGpuErrors, ownedGpuUnavailable } = memory;
  assert.ok(gpuUuid && Number.isSafeInteger(ownedGpuPeakBytes) && ownedGpuPeakBytes > 0 && ownedGpuSamples >= 3 && ownedGpuErrors === 0 && !ownedGpuUnavailable, "CUDA PID/UUID attribution is incomplete or N/A");
  assert.ok(Number.isSafeInteger(hostLifetimePeakBytes) && hostLifetimePeakBytes > 0 && hostLifetimePeakBytes <= hostCapBytes, "Windows worker lifetime working set is unavailable or exceeds the worker host containment limit");
  return { status: "observed", gpuUuid, admissions, pool, ownedGpuPeakBytes, ownedGpuSamples, hostLifetimePeakBytes, hostCapBytes, poolCapBytes: KREA_ALLOCATION_POOL_BYTES,
    budgetBasis: "Candle predicted additional bytes versus available additional bytes, private pool absolute counters versus its own cap, and Windows worker working set versus the worker containment limit. The sampled PID/UUID device peak is reported without applying additional-device-byte admission as an absolute limit." };
}
