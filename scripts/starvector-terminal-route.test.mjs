import assert from "node:assert/strict";
import test from "node:test";
import { vectorRequest } from "./starvector-terminal-route.mjs";

test("route runner can only construct the typed project-owned vector_generate request", () => {
  const request = vectorRequest({ projectId: "project", sourceAssetId: "asset", model: "starvector_1b", prompt: "icon" });
  assert.deepEqual(request, { projectId: "project", projectName: undefined, mode: "image_to_svg", model: "starvector_1b", sourceAssetId: "asset", prompt: "icon", sampling: undefined, detailBudget: undefined });
  assert.throws(() => vectorRequest({ projectId: "project", model: "starvector_1b" }), /sourceAssetId/);
});
