// Pure fixture-contract tests for the sc-26687 verification harness.
// They exercise the mock middleware with fake request/response objects: no server
// starts, no port binds, no browser runs, no network. The shapes asserted here are the
// canonical API contracts (crates/sceneworks-core/src/contracts.rs), because the hosted
// run proved the real shell crashes behind an ErrorBoundary when a fixture asset lacks
// `createdAt` (App.jsx sorts assets with sortNewest).
import { describe, expect, it } from "vitest";

import {
  ASSETS,
  IMAGE_EMPTY_REFS_ASSET,
  IMPORTED_MODEL_REF,
  importedModel,
  MODELS,
  REPAIR_MODEL,
  VIDEO_REPLAY_ASSET,
  verifyMiddleware,
} from "../../verify-fixtures.mjs";

function fakeRequest(url, method = "GET", body = null) {
  const req = {
    url,
    method,
    headers: {},
    on(event, callback) {
      // readBody resolves on "end"; answering synchronously keeps the middleware fully
      // in-process (no event loop, no listener, no socket).
      if (event === "end" || event === "close") callback();
      return req;
    },
  };
  if (body != null) req.bodyText = JSON.stringify(body);
  return req;
}

function fakeResponse() {
  const res = {
    statusCode: 0,
    headers: {},
    body: null,
    ended: false,
    setHeader(key, value) {
      res.headers[key] = value;
    },
    end(chunk) {
      if (chunk != null) res.body = chunk;
      res.ended = true;
    },
    write() {},
  };
  return res;
}

async function dispatch(path, { method = "GET", body = null } = {}) {
  const nextCalls = [];
  const res = fakeResponse();
  const promise = verifyMiddleware()(fakeRequest(path, method, body), res, () => nextCalls.push(path));
  await promise;
  return { res, nextCalls };
}

async function jsonAt(path, options) {
  const { res } = await dispatch(path, options);
  return { status: res.statusCode, body: JSON.parse(res.body) };
}

describe("fixture asset contract", () => {
  it("exposes the canonical project-store asset shape on every asset", () => {
    expect(ASSETS.length).toBeGreaterThan(0);
    for (const asset of ASSETS) {
      expect(asset.schemaVersion, asset.id).toBe(1);
      expect(asset.id, asset.id).toEqual(expect.any(String));
      expect(asset.projectId, asset.id).toBe("project-1");
      expect(asset.type, asset.id).toEqual(expect.any(String));
      expect(asset.displayName, asset.id).toEqual(expect.any(String));
      expect(Number.isNaN(Date.parse(asset.createdAt)), `${asset.id}.createdAt`).toBe(false);
      expect(asset.file.path, asset.id).toEqual(expect.any(String));
      expect(asset.file.mimeType, asset.id).toEqual(expect.any(String));
      expect(asset.status.favorite, asset.id).toBe(false);
      expect(asset.status.rating, asset.id).toBe(0);
      expect(asset.status.rejected, asset.id).toBe(false);
      expect(asset.status.trashed, asset.id).toBe(false);
      expect(Array.isArray(asset.lineage.parents), asset.id).toBe(true);
      expect(asset.generationSetId === null || typeof asset.generationSetId === "string", asset.id).toBe(true);
      for (const key of ["mode", "model", "adapter", "prompt", "negativePrompt", "seed", "loras", "normalizedSettings", "rawAdapterSettings"]) {
        expect(asset.recipe, `${asset.id}.${key}`).toHaveProperty(key);
      }
    }
  });

  it("keeps the replay recipes and imported identity intact", () => {
    expect(VIDEO_REPLAY_ASSET.generationSet.recipe.importedModelRef).toEqual(IMPORTED_MODEL_REF);
    expect(VIDEO_REPLAY_ASSET.generationSet.recipe.normalizedSettings).toMatchObject({
      sourceAssetId: null,
      lastFrameAssetId: "last",
    });
    expect(IMAGE_EMPTY_REFS_ASSET.generationSet.recipe.normalizedSettings.referenceAssetIds).toEqual([]);
    expect(IMAGE_EMPTY_REFS_ASSET.generationSet.recipe.rawAdapterSettings.referenceAssetIds).toEqual(["stale"]);
  });

  it("serves the last-only prerequisites on both imported partitions", () => {
    for (const partition of ["fl2va", "hybrid"]) {
      const model = importedModel(partition);
      expect(model.family).toBe("minimax-h3");
      expect(model.baseModel).toBe("minimax_h3");
      expect(model.importSourceShape).toBe("transformer_directory");
      expect(model.importPlan.checkpointId).toEqual(expect.any(String));
      expect(model.minimaxH3Partition).toBe(partition);
      expect(model.capabilities).toContain("image_to_video");
    }
    expect(REPAIR_MODEL.id).toBe("my_h3");
  });
});

describe("typed fixture endpoints", () => {
  it("answers the mac capability probe with the typed contract", async () => {
    const { status, body } = await jsonAt("/api/v1/capabilities/mac");
    expect(status).toBe(200);
    expect(body.macGatingActive).toBe(false);
    expect(body.candleGatingActive).toBe(false);
    expect(body.platform).toBe("linux");
  });

  it("answers the host memory probe with the typed pair", async () => {
    const { status, body } = await jsonAt("/api/v1/host-capabilities");
    expect(status).toBe(200);
    expect(body.platform).toBe("linux");
    expect(body.memoryKind).toBe("dedicated");
    expect(typeof body.memoryGb).toBe("number");
  });

  it("answers ready-empty lists for registered-work surfaces", async () => {
    for (const path of [
      "/api/v1/workers",
      "/api/v1/recipe-presets",
      "/api/v1/prompt-batches",
      "/api/v1/projects/project-1/characters",
      "/api/v1/projects/project-1/person-tracks",
    ]) {
      const { status, body } = await jsonAt(path);
      expect(status, path).toBe(200);
      expect(Array.isArray(body), path).toBe(true);
    }
    const { status, body } = await jsonAt("/api/v1/models/library-roots");
    expect(status).toBe(200);
    expect(Array.isArray(body.roots)).toBe(true);
    expect(body.roots[0].rootId).toEqual(expect.any(String));
  });

  it("answers the model-cache snapshot with its typed entries array", async () => {
    const { status, body } = await jsonAt("/api/v1/model-cache");
    expect(status).toBe(200);
    expect(body.entryCount).toBe(0);
    expect(Array.isArray(body.entries)).toBe(true);
  });

  it("serves the canonical asset list with createdAt on every asset", async () => {
    const { status, body } = await jsonAt("/api/v1/projects/project-1/assets");
    expect(status).toBe(200);
    expect(body).toHaveLength(ASSETS.length);
    for (const asset of body) {
      expect(Number.isNaN(Date.parse(asset.createdAt)), asset.id).toBe(false);
    }
  });

  it("serves the canonical project shape", async () => {
    const { status, body } = await jsonAt("/api/v1/projects");
    expect(status).toBe(200);
    expect(body[0]).toMatchObject({ schemaVersion: 1, id: "project-1", name: "Noir" });
    expect(Number.isNaN(Date.parse(body[0].createdAt))).toBe(false);
  });
});

describe("closed fixture boundary", () => {
  it("404s unknown /api paths and never forwards them", async () => {
    const { res, nextCalls } = await dispatch("/api/v1/definitely-not-a-real-endpoint");
    expect(res.statusCode).toBe(404);
    expect(nextCalls).toEqual([]);
  });

  it("passes non-API paths through to the Vite server", async () => {
    const { res, nextCalls } = await dispatch("/index.html");
    expect(res.ended).toBe(false);
    expect(nextCalls).toEqual(["/index.html"]);
  });

  it("keeps MODELS a closed catalog containing the stable repair id", () => {
    expect(MODELS.map((model) => model.id)).toContain("my_h3");
  });
});