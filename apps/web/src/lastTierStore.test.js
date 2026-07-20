import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

// The store mirrors every pick to the durable server copy (epic 10721 R1) so it survives a desktop
// relaunch. Mock the API so the tests can assert the write-through without a live server.
vi.mock("./api.js", () => ({ apiFetch: vi.fn(() => Promise.resolve({})) }));

import { apiFetch } from "./api.js";
import { readLastTier, seedLastTiersFromServer, writeLastTier } from "./lastTierStore.js";
import { defaultTierSelection } from "./quantTier.js";

// A download-matrix model whose `installed` tiers are present; unlisted declared tiers are missing.
function matrixModel({ tiers = ["q4", "q8", "bf16"], installed = [], id = "model_x" } = {}) {
  return {
    id,
    hasVariantMatrix: true,
    variants: tiers.map((tier) => ({
      variant: tier,
      // No `default: true` — the real catalog never emits it, so the sticky must win on its own.
      installState: installed.includes(tier) ? "installed" : "missing",
    })),
  };
}

const STORAGE_KEY = "sceneworks-last-tier";

beforeEach(() => {
  window.localStorage.clear();
  apiFetch.mockClear();
});

afterEach(() => {
  window.localStorage.clear();
});

describe("lastTierStore — (screen, model) key", () => {
  it("round-trips a written tier for the same (screen, model)", () => {
    writeLastTier("image", "model_x", "q4");
    expect(readLastTier("image", "model_x")).toBe("q4");
  });

  it("returns null when nothing has been stored for (screen, model)", () => {
    expect(readLastTier("image", "model_x")).toBe(null);
  });

  it("keys are independent per MODEL — model Y is unaffected by a pick on model X", () => {
    writeLastTier("image", "model_x", "q4");
    expect(readLastTier("image", "model_x")).toBe("q4");
    expect(readLastTier("image", "model_y")).toBe(null);
  });

  it("keys are independent per SCREEN — Video/Character are unaffected by an Image pick", () => {
    writeLastTier("image", "model_x", "q4");
    expect(readLastTier("image", "model_x")).toBe("q4");
    expect(readLastTier("video", "model_x")).toBe(null);
    expect(readLastTier("character", "model_x")).toBe(null);
  });

  it("the same model on two screens holds two independent picks", () => {
    writeLastTier("image", "model_x", "q4");
    writeLastTier("video", "model_x", "q8");
    expect(readLastTier("image", "model_x")).toBe("q4");
    expect(readLastTier("video", "model_x")).toBe("q8");
  });

  it("a later explicit pick overwrites the earlier sticky for that (screen, model)", () => {
    writeLastTier("image", "model_x", "q4");
    writeLastTier("image", "model_x", "q8");
    expect(readLastTier("image", "model_x")).toBe("q8");
  });

  it("ignores writes with a falsy screen, model, or tier", () => {
    writeLastTier("", "model_x", "q4");
    writeLastTier("image", "", "q4");
    writeLastTier("image", "model_x", "");
    expect(readLastTier("image", "model_x")).toBe(null);
  });

  it("returns null for a read with a falsy screen or model", () => {
    writeLastTier("image", "model_x", "q4");
    expect(readLastTier("", "model_x")).toBe(null);
    expect(readLastTier("image", "")).toBe(null);
  });
});

describe("lastTierStore — persistence (survives app restart)", () => {
  it("persists to localStorage and re-reads after a simulated restart", () => {
    writeLastTier("image", "model_x", "q4");
    // A restart drops all in-memory state. readLastTier reads localStorage fresh every call, so a
    // read AFTER the write — with no module-level cache in between — is the restart-equivalent.
    expect(readLastTier("image", "model_x")).toBe("q4");
    // And the value genuinely lives in the persisted blob, not just in memory.
    const persisted = JSON.parse(window.localStorage.getItem(STORAGE_KEY));
    expect(persisted).toEqual({ image: { model_x: "q4" } });
  });

  it("survives corrupt/absent storage by falling back to null", () => {
    window.localStorage.setItem(STORAGE_KEY, "{not json");
    expect(readLastTier("image", "model_x")).toBe(null);
  });
});

// The sticky rung is base-agnostic: it is passed to `defaultTierSelection` as `lastUsed`, which
// honors it above the model's base default (currently q8 — epic 10721 / sc-10726 — for both
// download-matrix and convert-at-install mlxTiers models) whenever the sticky tier is still
// installed, and otherwise falls through to that base and clamps to installed.
describe("lastTierStore — precedence with defaultTierSelection", () => {
  it("sticky beats the base default when the sticky tier is installed (download-matrix)", () => {
    // q4, q8 both installed; the model's base default is q8, but the user's sticky is q4.
    const model = matrixModel({ installed: ["q4", "q8"] });
    writeLastTier("image", model.id, "q4");
    expect(defaultTierSelection(model, readLastTier("image", model.id))).toBe("q4");
  });

  it("clamps a sticky to installed — sticky bf16 but only q4 installed → q4 (base clamps too)", () => {
    const model = matrixModel({ installed: ["q4"] });
    writeLastTier("image", model.id, "bf16");
    // The sticky bf16 is not installed, so defaultTierSelection ignores it and falls to the base
    // default (q8); q8 is not installed either, so it clamps to the only installed tier, q4.
    expect(defaultTierSelection(model, readLastTier("image", model.id))).toBe("q4");
  });

  it("clamps a sticky q8 to q4 when only q4 is installed (acceptance: sticky q8, q4-only → q4)", () => {
    const model = matrixModel({ installed: ["q4"] });
    writeLastTier("image", model.id, "q8");
    expect(defaultTierSelection(model, readLastTier("image", model.id))).toBe("q4");
  });

  it("with no sticky, defaultTierSelection uses the q8 base default (download-matrix)", () => {
    const model = matrixModel({ installed: ["q4", "q8"] });
    expect(readLastTier("image", model.id)).toBe(null);
    expect(defaultTierSelection(model, readLastTier("image", model.id))).toBe("q8");
  });

  it("applies to convert-at-install (mlxTiers) models — sticky q4 beats the q8 base default", () => {
    // Convert-at-install models surface installed tiers via `mlxTiers` (S5) instead of a variant
    // matrix and preselect q8 by default. The same sticky store seeds them: pick q4, and q4 wins.
    const convertModel = { id: "anima", mlxTiers: ["q4", "q8", "bf16"] };
    expect(defaultTierSelection(convertModel, readLastTier("image", convertModel.id))).toBe("q8");
    writeLastTier("image", convertModel.id, "q4");
    expect(defaultTierSelection(convertModel, readLastTier("image", convertModel.id))).toBe("q4");
  });
});

describe("lastTierStore — durable server persistence (epic 10721 R1)", () => {
  it("mirrors each pick to the durable ui-preferences copy as the full merged map", () => {
    writeLastTier("image", "sana_sprint_1600m", "bf16");
    writeLastTier("image", "flux_dev", "q4");
    writeLastTier("video", "wan_ti2v_5b", "q8");

    // Every write PUTs the FULL map (so the server can replace wholesale — no deep-merge needed).
    expect(apiFetch).toHaveBeenCalledTimes(3);
    const [path, token, opts] = apiFetch.mock.calls.at(-1);
    expect(path).toBe("/api/v1/ui-preferences");
    expect(token).toBe(""); // public route
    expect(opts.method).toBe("PUT");
    expect(JSON.parse(opts.body)).toEqual({
      perModelTier: {
        image: { sana_sprint_1600m: "bf16", flux_dev: "q4" },
        video: { wan_ti2v_5b: "q8" },
      },
    });
  });

  it("does not re-PUT when the pick is unchanged", () => {
    writeLastTier("image", "model_x", "q8");
    apiFetch.mockClear();
    writeLastTier("image", "model_x", "q8"); // same value — no-op
    expect(apiFetch).not.toHaveBeenCalled();
  });

  it("seedLastTiersFromServer primes the cache so a prior-session pick is read back after relaunch", () => {
    // Simulate a fresh launch: localStorage empty, server returns a stored map.
    expect(readLastTier("image", "sana_sprint_1600m")).toBe(null);
    seedLastTiersFromServer({ image: { sana_sprint_1600m: "bf16" } });
    expect(readLastTier("image", "sana_sprint_1600m")).toBe("bf16");
    // Seeding is a read-only cache prime — it must not echo a redundant PUT back to the server.
    expect(apiFetch).not.toHaveBeenCalled();
  });
});
