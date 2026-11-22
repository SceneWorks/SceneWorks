import assert from "node:assert/strict";
import test from "node:test";
import { KREA_WDDM_SOURCE, kreaWindowsDedicatedPeak } from "./krea-windows-worker-memory.mjs";
import { kreaWindowsMemoryFixture } from "./krea-windows-worker-memory-fixture.mjs";

test("owned dedicated replay selects the actual peak and excludes missing or cached polls", () => {
  const result = kreaWindowsDedicatedPeak(kreaWindowsMemoryFixture());
  assert.equal(result.peakBytes, 80);
  assert.equal(result.samples, 3);
  assert.equal(result.source, KREA_WDDM_SOURCE);
});

test("device byte observations require the launched worker, leased adapter and approved source", () => {
  for (const [mutate, error] of [
    [(memory) => { memory.wddmDedicated.workerPid++; }, /launched worker/],
    [(memory) => { memory.wddmDedicated.rows[0].identity.executable += ".bad"; }, /launched worker/],
    [(memory) => { memory.wddmDedicated.rows[0].identity.created_filetime = 0; }, /FILETIME/],
    [(memory) => { memory.wddmDedicated.rows[0].adapter.node_mask = 4; }, /leased CUDA UUID and node/],
    [(memory) => { memory.wddmDedicated.rows[0].adapter.cuda_uuid += "a"; }, /leased CUDA UUID and node/],
    [(memory) => { memory.wddmDedicated.source = "gpu-global-used"; }, /incomplete/],
    [(memory) => { memory.ownedGpuIdentityError = "unleased device"; }, /incomplete/],
  ]) {
    const memory = kreaWindowsMemoryFixture();
    mutate(memory);
    assert.throws(() => kreaWindowsDedicatedPeak(memory), error);
  }
});

test("missing active polls and gaps, cached provider values, and stale terminal receipts fail closed", () => {
  for (const [mutate, error] of [
    [(rows) => { rows[3].event = "missing"; }, /disappeared while active/],
    [(rows) => { rows[5].elapsed_ms = 2000; }, /poll missing/],
    [(rows) => { rows[3].read_started_ms = 250; }, /sampling skew/],
    [(rows) => { rows[4].filetime = rows[5].filetime; }, /cached read/],
    [(rows) => { rows[7].filetime = rows[5].filetime; }, /invalid .*reading/],
    [(rows) => { rows[6].not_before_filetime = String(BigInt(rows[7].filetime) + 1n); }, /predates/],
    [(rows) => { rows[7].read_started_ms = rows[6].elapsed_ms - 1; }, /predates/],
    [(rows) => { rows.pop(); }, /terminal.*incomplete/],
    [(rows) => { rows[5].dedicated_bytes = 0; }, /positive samples.*incomplete/],
  ]) {
    const memory = kreaWindowsMemoryFixture();
    mutate(memory.wddmDedicated.rows);
    assert.throws(() => kreaWindowsDedicatedPeak(memory), error);
  }
});
