import test from "node:test";
import assert from "node:assert/strict";
import {
  loadShippingProfiles,
  verifyGenerationRouteProfiles,
} from "./check-generation-route-profiles.mjs";

test("all five checked-in sga.capabilities/2 shipping profiles verify cleanly", () => {
  const profiles = loadShippingProfiles();
  const summary = verifyGenerationRouteProfiles(profiles);
  assert.equal(summary.schemaVersion, "sga.capabilities/2");
  assert.equal(summary.iosMediaRoutes, 7);
  assert.equal(summary.iosLlmOnlyRoutes, 0);
  assert.ok(summary.candleCpuRoutes > 50);
  assert.ok(summary.candleCudaRoutes > 50);
  assert.ok(summary.macosMlxRoutes > 50);
});

test("rejects missing StarVector vector route or media leak into ios-llm-only", () => {
  const missingVector = structuredClone(loadShippingProfiles());
  delete missingVector["candle-cuda"].routes.starvector_8b;
  assert.throws(
    () => verifyGenerationRouteProfiles(missingVector),
    /candle-cuda missing starvector_8b/,
  );

  const leakedIos = structuredClone(loadShippingProfiles());
  leakedIos["ios-llm-only"].routes.sdxl = { modality: "image" };
  assert.throws(
    () => verifyGenerationRouteProfiles(leakedIos),
    /ios-llm-only must expose 0 media routes/,
  );
});

test("rejects forbidden pin provenance or wrong schema version", () => {
  const tainted = structuredClone(loadShippingProfiles());
  tainted["macos-mlx"].generatedFrom = { inferenceRevision: "a".repeat(40) };
  assert.throws(() => verifyGenerationRouteProfiles(tainted), /must be pin-free/);

  const wrongSchema = structuredClone(loadShippingProfiles());
  wrongSchema["candle-cpu"].schema_version = "sga.capabilities/1";
  assert.throws(() => verifyGenerationRouteProfiles(wrongSchema), /wrong schema_version/);
});
