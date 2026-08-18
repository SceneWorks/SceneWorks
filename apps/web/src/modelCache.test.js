import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

// Only the transport is mocked: everything else under test is pure projection over the API's own
// DTOs, and the point of these tests is that the projections stay faithful to what the backend
// actually said (sc-19711, epic 19703).
vi.mock("./api.js", () => ({ apiFetch: vi.fn(() => Promise.resolve({})) }));

import { apiFetch } from "./api.js";
import { ACCESS_TOKEN_KEY } from "./accessToken.js";
import {
  CACHE_COVERAGE,
  MODEL_AVAILABILITY,
  availabilityBadge,
  bytesToGib,
  cacheEligibilityBadge,
  canHoldLocalCopy,
  daysToSeconds,
  describeMissingLocalCopy,
  describeDisableConsequence,
  describeLimitConsequence,
  describeRemovalPreview,
  entriesForModel,
  describeEntryState,
  fetchModelCache,
  gibToBytes,
  hasTransitionalEntries,
  isTransitionalEntry,
  policyNeedsRestart,
  previewCacheRemoval,
  removalIsAllowed,
  removeCacheEntry,
  secondsToDays,
  setCacheEntryPin,
} from "./modelCache.js";

const GIB = 1024 * 1024 * 1024;
const DAY = 24 * 60 * 60;

beforeEach(() => {
  window.localStorage.clear();
  apiFetch.mockClear();
  apiFetch.mockImplementation(() => Promise.resolve({}));
});

afterEach(() => {
  window.localStorage.clear();
});

// ---- availability badges ---------------------------------------------------------

describe("availabilityBadge", () => {
  // One badge per typed state the ONE shared resolver (sc-19708) can emit. Enumerated
  // exhaustively rather than spot-checked so adding a sixth wire state without a badge reds here
  // instead of silently rendering "unknown state" in production.
  it("labels every typed availability state distinctly", () => {
    const states = Object.values(MODEL_AVAILABILITY);
    expect(states).toHaveLength(5);
    const texts = states.map((state) => availabilityBadge({ modelAvailability: state }).text);
    expect(texts).toEqual([
      "local copy",
      "on external library",
      "library disconnected",
      "incomplete",
      "not installed",
    ]);
    // Every badge carries an explanatory title, and no two states collapse onto one label.
    for (const state of states) {
      expect(availabilityBadge({ modelAvailability: state }).title).toBeTruthy();
    }
    expect(new Set(texts).size).toBe(states.length);
  });

  // Warning tone is reserved for the two states a user must act on. `external_ready` is normal
  // operation and must NOT be alarmed; `local_ready` is the good case.
  it("tones only the actionable states as warnings", () => {
    const toneOf = (state) => availabilityBadge({ modelAvailability: state }).tone;
    expect(toneOf(MODEL_AVAILABILITY.LOCAL_READY)).toBe("installed");
    expect(toneOf(MODEL_AVAILABILITY.EXTERNAL_READY)).toBe("");
    expect(toneOf(MODEL_AVAILABILITY.INSTALLED_EXTERNAL_UNAVAILABLE)).toBe("warning");
    expect(toneOf(MODEL_AVAILABILITY.INCOMPLETE)).toBe("warning");
    expect(toneOf(MODEL_AVAILABILITY.MISSING)).toBe("");
  });

  // The rule that makes this module safe to point at a newer backend: an unrecognized state is
  // LABELLED, never coerced into a reassuring one. A future "evicting" state must not read as
  // "on external library".
  it("labels an unrecognized state instead of defaulting it to a reassuring one", () => {
    const badge = availabilityBadge({ modelAvailability: "quantum_superposition" });
    expect(badge.text).toBe("unknown state");
    expect(badge.tone).toBe("warning");
    expect(badge.title).toContain("quantum_superposition");
  });

  // No judgement at all is different from an unknown judgement: a row the resolver never annotated
  // renders no badge rather than an alarming one.
  it("renders no badge when the row carries no judgement", () => {
    expect(availabilityBadge({})).toBeNull();
    expect(availabilityBadge({ modelAvailability: "" })).toBeNull();
    expect(availabilityBadge({ modelAvailability: null })).toBeNull();
    expect(availabilityBadge(null)).toBeNull();
    expect(availabilityBadge(undefined)).toBeNull();
  });
});

describe("canHoldLocalCopy", () => {
  // The resolved cache only ever holds copies of models whose authoritative source is an external
  // library. `incomplete` and `missing` have nothing to copy, so offering local-copy controls on
  // them would be an untrue affordance.
  it("is true only for the three states that can back a local copy", () => {
    expect(canHoldLocalCopy({ modelAvailability: MODEL_AVAILABILITY.LOCAL_READY })).toBe(true);
    expect(canHoldLocalCopy({ modelAvailability: MODEL_AVAILABILITY.EXTERNAL_READY })).toBe(true);
    expect(
      canHoldLocalCopy({ modelAvailability: MODEL_AVAILABILITY.INSTALLED_EXTERNAL_UNAVAILABLE }),
    ).toBe(true);
    expect(canHoldLocalCopy({ modelAvailability: MODEL_AVAILABILITY.INCOMPLETE })).toBe(false);
    expect(canHoldLocalCopy({ modelAvailability: MODEL_AVAILABILITY.MISSING })).toBe(false);
    expect(canHoldLocalCopy({ modelAvailability: "something_new" })).toBe(false);
    expect(canHoldLocalCopy(null)).toBe(false);
  });
});

// ---- local-copy coverage -----------------------------------------------------------

// `cacheEligibility` exactly as `LocalCacheEligibility` serializes it (sc-19712 F-5): a typed
// coverage, a typed reason, and a prose detail that is supporting copy only.
const eligibility = (coverage, reason, detail) => ({
  schemaVersion: 1,
  coverage,
  reason: reason ?? null,
  detail: detail ?? null,
});

describe("cacheEligibilityBadge", () => {
  // A model whose local copy covers it needs no caveat — badging every model would make the
  // exclusions invisible again by drowning them.
  it("renders no badge for full coverage", () => {
    expect(cacheEligibilityBadge({ cacheEligibility: eligibility(CACHE_COVERAGE.FULL) })).toBeNull();
  });

  // The real `qwen_image` / `acestep_v15_turbo` case: the primary is cached, the optional
  // co-requisite never is, so a request needing it still reads from the library.
  it("warns that a partial local copy covers only part of the model", () => {
    const badge = cacheEligibilityBadge({
      cacheEligibility: eligibility(
        CACHE_COVERAGE.PARTIAL,
        "optional_components_excluded",
        "Optional components of this model are never copied locally, so a request that needs one still reads from the model library.",
      ),
    });
    expect(badge.text).toBe("partial local copy");
    expect(badge.tone).toBe("warning");
    // The backend's own sentence is what the user hovers, not a second explanation invented here.
    expect(badge.title).toContain("Optional components of this model are never copied locally");
  });

  // The real `SceneWorks/qwen-image-mlx` case: no recorded snapshot revision, so the whole
  // repository is unserveable from the local tier however many bytes are copied.
  it("warns that no local copy is possible when coverage is none", () => {
    const badge = cacheEligibilityBadge({
      cacheEligibility: eligibility(
        CACHE_COVERAGE.NONE,
        "unpinned_revision",
        "No snapshot revision was recorded for SceneWorks/qwen-image-mlx, so no local copy of this model can be used and it will always load from the model library.",
      ),
    });
    expect(badge.text).toBe("no local copy possible");
    expect(badge.tone).toBe("warning");
    expect(badge.title).toContain("SceneWorks/qwen-image-mlx");
  });

  // A null field means "no external requirement closure" — not applicable, NOT "full". Either way
  // there is nothing to warn about, so no badge; the distinction matters only in that neither may
  // be treated as an exclusion.
  it("renders no badge when the row carries no eligibility judgement", () => {
    expect(cacheEligibilityBadge({ cacheEligibility: null })).toBeNull();
    expect(cacheEligibilityBadge({})).toBeNull();
    expect(cacheEligibilityBadge({ cacheEligibility: { schemaVersion: 1 } })).toBeNull();
    expect(cacheEligibilityBadge({ cacheEligibility: { coverage: "" } })).toBeNull();
    expect(cacheEligibilityBadge(null)).toBeNull();
    expect(cacheEligibilityBadge(undefined)).toBeNull();
  });

  // The same rule the availability badge follows: a coverage value this build doesn't know must be
  // LABELLED, never dropped (which would read as "fully cacheable").
  it("labels an unrecognized coverage instead of hiding it", () => {
    const badge = cacheEligibilityBadge({ cacheEligibility: eligibility("mostly") });
    expect(badge).not.toBeNull();
    expect(badge.text).toBe("unknown local-copy coverage");
    expect(badge.tone).toBe("warning");
    expect(badge.title).toContain("mostly");
  });

  // With no detail sentence on the wire, the badge still explains itself rather than showing an
  // empty tooltip.
  it("falls back to its own explanation when the backend sent no detail", () => {
    const badge = cacheEligibilityBadge({
      cacheEligibility: eligibility(CACHE_COVERAGE.NONE, "unpinned_revision", null),
    });
    expect(badge.title).toBeTruthy();
    expect(badge.title).toContain("model library");
  });
});

describe("describeMissingLocalCopy", () => {
  // The promise is kept exactly where it is true.
  it("promises a copy for a fully cacheable model and for a row with no judgement", () => {
    const promised = "SceneWorks makes one the next time it loads this model";
    expect(describeMissingLocalCopy({ cacheEligibility: eligibility(CACHE_COVERAGE.FULL) })).toContain(
      promised,
    );
    expect(describeMissingLocalCopy({ cacheEligibility: null })).toContain(promised);
    expect(describeMissingLocalCopy({})).toContain(promised);
  });

  // The defect itself: this sentence used to promise a local copy for a model that can never have
  // one. It must now say the opposite, and carry the reason.
  it("states plainly that an excluded model cannot be given a local copy", () => {
    const text = describeMissingLocalCopy({
      cacheEligibility: eligibility(
        CACHE_COVERAGE.NONE,
        "unpinned_revision",
        "No snapshot revision was recorded for SceneWorks/qwen-image-mlx, so no local copy of this model can be used and it will always load from the model library.",
      ),
    });
    expect(text).toContain("can’t be given a local copy");
    expect(text).toContain("No snapshot revision was recorded");
    expect(text).not.toContain("SceneWorks makes one the next time");
  });

  it("says a partial copy covers only part of the model", () => {
    const text = describeMissingLocalCopy({
      cacheEligibility: eligibility(
        CACHE_COVERAGE.PARTIAL,
        "optional_components_excluded",
        "Optional components of this model are never copied locally, so a request that needs one still reads from the model library.",
      ),
    });
    expect(text).toContain("would cover only part of it");
    expect(text).toContain("Optional components");
    expect(text).not.toContain("SceneWorks makes one the next time");
  });

  // An unrecognized coverage promises nothing and claims nothing — it neither guarantees a copy
  // nor asserts an exclusion this build cannot actually read.
  it("neither promises nor denies a copy for an unrecognized coverage", () => {
    const text = describeMissingLocalCopy({ cacheEligibility: eligibility("mostly") });
    expect(text).not.toContain("SceneWorks makes one the next time");
    expect(text).not.toContain("can’t be given a local copy");
    expect(text).toContain("mostly");
  });
});

// ---- transport --------------------------------------------------------------------

describe("model-cache transport", () => {
  // The whole surface is gated on remote-auth deployments. A token assertion that pins "" passes
  // with the fix reverted and therefore says nothing (sc-15136), so every one of these pins a
  // NON-EMPTY token.
  it("sends the stored access token on every route", async () => {
    window.localStorage.setItem(ACCESS_TOKEN_KEY, "lan-password-1");
    await fetchModelCache();
    await previewCacheRemoval("key-a");
    await removeCacheEntry("key-a");
    await setCacheEntryPin("key-a", true);
    expect(apiFetch).toHaveBeenCalledTimes(4);
    for (const call of apiFetch.mock.calls) {
      expect(call[1]).toBe("lan-password-1");
    }
  });

  it("reads status with a plain GET — no body, no method override", async () => {
    apiFetch.mockResolvedValue({ entryCount: 0 });
    await expect(fetchModelCache()).resolves.toEqual({ entryCount: 0 });
    expect(apiFetch).toHaveBeenCalledWith("/api/v1/model-cache", "");
  });

  it("posts the cache key to the removal-preview route", async () => {
    await previewCacheRemoval("sha256:abc");
    expect(apiFetch).toHaveBeenCalledWith("/api/v1/model-cache/removal-preview", "", {
      method: "POST",
      body: JSON.stringify({ cacheKey: "sha256:abc" }),
    });
  });

  it("posts the cache key to the removal route", async () => {
    await removeCacheEntry("sha256:abc");
    expect(apiFetch).toHaveBeenCalledWith("/api/v1/model-cache/remove", "", {
      method: "POST",
      body: JSON.stringify({ cacheKey: "sha256:abc" }),
    });
  });

  // Both directions, because a pin route that ignored the flag would still look correct in a
  // one-way test.
  it("posts both pin directions", async () => {
    await setCacheEntryPin("sha256:abc", true);
    expect(apiFetch).toHaveBeenLastCalledWith("/api/v1/model-cache/pin", "", {
      method: "POST",
      body: JSON.stringify({ cacheKey: "sha256:abc", pinned: true }),
    });
    await setCacheEntryPin("sha256:abc", false);
    expect(apiFetch).toHaveBeenLastCalledWith("/api/v1/model-cache/pin", "", {
      method: "POST",
      body: JSON.stringify({ cacheKey: "sha256:abc", pinned: false }),
    });
  });
});

// ---- projections ------------------------------------------------------------------

describe("entriesForModel", () => {
  const status = {
    entries: [
      { cacheKey: "a", modelIds: ["flux_dev", "flux_krea"] },
      { cacheKey: "b", modelIds: ["z_image_turbo"] },
      { cacheKey: "c", modelIds: [] },
      { cacheKey: "d" },
    ],
  };

  it("selects the entries the BACKEND joined to this model, including shared ones", () => {
    expect(entriesForModel(status, "flux_dev").map((entry) => entry.cacheKey)).toEqual(["a"]);
    expect(entriesForModel(status, "flux_krea").map((entry) => entry.cacheKey)).toEqual(["a"]);
    expect(entriesForModel(status, "z_image_turbo").map((entry) => entry.cacheKey)).toEqual(["b"]);
  });

  // A row with no join (corrupt residue the API still lists, per its own comment) must never be
  // attributed to an arbitrary model.
  it("never attributes an unjoined or id-less entry to a model", () => {
    expect(entriesForModel(status, "unrelated")).toEqual([]);
    expect(entriesForModel(status, "")).toEqual([]);
    expect(entriesForModel(status, undefined)).toEqual([]);
  });

  // A failed status read leaves `status` null. That must yield "no information", which the screen
  // renders as no local-copy block — not an empty-but-confident "no local copies".
  it("returns nothing when the status read failed", () => {
    expect(entriesForModel(null, "flux_dev")).toEqual([]);
    expect(entriesForModel({}, "flux_dev")).toEqual([]);
    expect(entriesForModel({ entries: "not-an-array" }, "flux_dev")).toEqual([]);
  });
});

describe("unit conversions", () => {
  it("round-trips GiB and days through the wire units", () => {
    expect(gibToBytes(64)).toBe(64 * GIB);
    expect(bytesToGib(64 * GIB)).toBe(64);
    expect(daysToSeconds(14)).toBe(14 * DAY);
    expect(secondsToDays(14 * DAY)).toBe(14);
  });

  // The inputs are user-typed, so every non-positive / non-numeric form must fail closed to 0 —
  // which the commit handlers then refuse — rather than producing NaN on the wire.
  it("fails closed to zero for every non-positive or unparseable input", () => {
    for (const bad of [0, -1, "", " ", "abc", null, undefined, NaN, Infinity]) {
      expect(gibToBytes(bad)).toBe(0);
      expect(daysToSeconds(bad)).toBe(0);
      expect(bytesToGib(bad)).toBe(0);
      expect(secondsToDays(bad)).toBe(0);
    }
  });

  it("rounds fractional entry to whole wire units", () => {
    expect(gibToBytes("1.5")).toBe(Math.round(1.5 * GIB));
    expect(daysToSeconds("0.5")).toBe(Math.round(0.5 * DAY));
  });
});

// ---- the restart-bound policy comparison ------------------------------------------

describe("policyNeedsRestart", () => {
  const persisted = { enabled: true, maxBytes: 64 * GIB, inactivitySeconds: 14 * DAY };

  // The design fact this whole card is built on: the cache policy is desktop-shell-owned and
  // restart-bound. The shell persists it; the three sidecars captured theirs at spawn. So the UI
  // compares persisted-vs-running and only then claims "restart to apply".
  it("is false while the persisted policy matches the policy actually running", () => {
    expect(policyNeedsRestart(persisted, { ...persisted })).toBe(false);
  });

  it("is true for a divergence in ANY of the three fields", () => {
    expect(policyNeedsRestart(persisted, { ...persisted, enabled: false })).toBe(true);
    expect(policyNeedsRestart(persisted, { ...persisted, maxBytes: 32 * GIB })).toBe(true);
    expect(policyNeedsRestart(persisted, { ...persisted, inactivitySeconds: 7 * DAY })).toBe(true);
  });

  // Absent knowledge is not a divergence: before either side loads, claiming "restart to apply"
  // would be a false statement about state nobody has read yet.
  it("claims nothing while either side is still unknown", () => {
    expect(policyNeedsRestart(null, persisted)).toBe(false);
    expect(policyNeedsRestart(persisted, null)).toBe(false);
    expect(policyNeedsRestart(null, null)).toBe(false);
    expect(policyNeedsRestart(undefined, undefined)).toBe(false);
  });
});

// ---- consequence copy --------------------------------------------------------------

describe("describeDisableConsequence", () => {
  // Turning the cache off stops NEW copies. It does not sweep. Saying otherwise would promise a
  // deletion the backend never performs.
  it("never claims disabling deletes anything", () => {
    const text = describeDisableConsequence({ entryCount: 3, usedBytes: 12 * GIB });
    expect(text).toContain("stops making new local copies");
    expect(text).toContain("stay on disk");
    expect(text).toContain("3 existing copies");
    expect(text).toContain("12.0 GiB");
    expect(text).not.toMatch(/delet|remove them now|free up/i);
  });

  // Agreement across the WHOLE sentence, not just the noun: the original read
  // "The 1 existing copy … stay on disk — remove them … reclaim them."
  it("agrees in number throughout the sentence for a lone copy", () => {
    const text = describeDisableConsequence({ entryCount: 1, usedBytes: GIB });
    expect(text).toContain("1 existing copy");
    expect(text).toContain("stays on disk");
    expect(text).toContain("remove it per model");
    expect(text).toContain("reclaim it.");
    // Only the clauses that refer to the ONE existing copy must be singular; the leading
    // "stops making new local copies" is a general statement and stays plural.
    expect(text).not.toMatch(/existing copies|stay on disk|\bthem\b/);
  });

  it("says plainly that nothing changes when the cache is empty", () => {
    expect(describeDisableConsequence({ entryCount: 0, usedBytes: 0 })).toContain(
      "changes nothing on disk",
    );
    expect(describeDisableConsequence(null)).toContain("changes nothing on disk");
  });
});

describe("describeLimitConsequence", () => {
  // Raising the limit, or staying at or above current usage, has no consequence to warn about.
  it("is silent when the new limit already fits current usage", () => {
    expect(describeLimitConsequence({ usedBytes: 10 * GIB, reclaimableBytes: 10 * GIB }, 20 * GIB)).toBe("");
    expect(describeLimitConsequence({ usedBytes: 10 * GIB, reclaimableBytes: 10 * GIB }, 10 * GIB)).toBe("");
    expect(describeLimitConsequence({ usedBytes: 10 * GIB, reclaimableBytes: 10 * GIB }, 0)).toBe("");
  });

  // Nothing is swept at the moment of the change — cleanup runs on the worker's idle checkpoints,
  // so the copy must be future tense and bounded by what is actually reclaimable.
  it("bounds the promise by the reclaimable bytes, in the future tense", () => {
    const text = describeLimitConsequence(
      { usedBytes: 20 * GIB, reclaimableBytes: 20 * GIB },
      12 * GIB,
    );
    expect(text).toContain("next automatic cleanup");
    expect(text).toContain("up to 8.0 GiB");
    expect(text).toContain("12.0 GiB");
    // 8 GiB over and 20 GiB reclaimable: nothing is left stranded above the limit.
    expect(text).not.toContain("stay over the limit");
  });

  // The honest partial case: the overage exceeds what may be taken, so the remainder stays above
  // the limit and the copy must say so rather than implying the limit will be met.
  it("admits the shortfall when pinned or in-use copies keep usage above the limit", () => {
    const text = describeLimitConsequence(
      { usedBytes: 20 * GIB, reclaimableBytes: 2 * GIB },
      12 * GIB,
    );
    expect(text).toContain("up to 2.0 GiB");
    expect(text).toContain("remaining 6.0 GiB");
    expect(text).toContain("stay over the limit");
  });

  // Zero reclaimable is not "cleanup will handle it" — it is "nothing can be reclaimed".
  it("says nothing can be reclaimed when every copy is kept", () => {
    const text = describeLimitConsequence({ usedBytes: 20 * GIB, reclaimableBytes: 0 }, 12 * GIB);
    expect(text).toContain("every copy is kept");
    expect(text).toContain("nothing can be reclaimed automatically");
    expect(text).not.toContain("will remove up to");
  });
});

// ---- removal preview copy -----------------------------------------------------------

describe("describeRemovalPreview", () => {
  const base = {
    cacheKey: "k",
    state: "complete",
    reclaimableBytes: 4 * GIB,
    pins: { kind: "known", artifactPinned: false, owners: [] },
    sourceUnavailableWarning: null,
    blocked: null,
  };

  it("leads with the measured bytes and reassures that the source is untouched", () => {
    const lines = describeRemovalPreview(base);
    expect(lines[0]).toBe("Removing this local copy frees 4.0 GiB.");
    expect(lines.join(" ")).toContain("original files in the model library are not touched");
  });

  // The refusal must lead, and the "frees N" promise must NOT also appear — a dialog that says
  // both is claiming a removal the store would refuse.
  it("leads with the refusal and never also promises the reclaim", () => {
    const lines = describeRemovalPreview({ ...base, blocked: "a runtime lease is active" });
    expect(lines[0]).toContain("can't be removed right now");
    expect(lines[0]).toContain("a runtime lease is active");
    expect(lines.join(" ")).not.toContain("frees 4.0 GiB");
  });

  // THE distinction this module exists to preserve. `Unknown` means the store could not read the
  // entry's pin state. Rendering that as "not pinned" would be a fabricated reassurance.
  it("says 'couldn't determine' for an unknown pin answer, never 'not pinned'", () => {
    const lines = describeRemovalPreview({ ...base, pins: { kind: "unknown" } });
    const text = lines.join(" ");
    expect(text).toContain("couldn't determine whether this copy is being kept");
    expect(text).not.toMatch(/not pinned|isn't pinned|can be removed automatically/i);
    // And it must not silently borrow the Known arm's sentences.
    expect(text).not.toContain("Allow automatic removal first");
  });

  it("names the user's own keep-locally mark and how to clear it", () => {
    const lines = describeRemovalPreview({
      ...base,
      pins: { kind: "known", artifactPinned: true, owners: [] },
    });
    expect(lines.join(" ")).toContain("You marked this copy to keep locally");
    expect(lines.join(" ")).toContain("Allow automatic removal first");
  });

  it("counts the loaded models still holding the copy, with agreement", () => {
    expect(
      describeRemovalPreview({
        ...base,
        pins: { kind: "known", artifactPinned: false, owners: ["flux_dev"] },
      }).join(" "),
    ).toContain("1 loaded model is still holding this copy");
    expect(
      describeRemovalPreview({
        ...base,
        pins: { kind: "known", artifactPinned: false, owners: ["flux_dev", "z_image_turbo"] },
      }).join(" "),
    ).toContain("2 loaded models are still holding this copy");
  });

  // The acceptance criterion: manual removal must clearly state whether it leaves the model
  // unusable until the external library returns.
  it("warns that removal leaves the model unusable when the source is unreachable", () => {
    const text = describeRemovalPreview({
      ...base,
      sourceUnavailableWarning: "/Volumes/Models is not mounted",
    }).join(" ");
    expect(text).toContain("becomes unusable until its library is reconnected");
    expect(text).toContain("/Volumes/Models is not mounted");
    // The reassuring sentence is mutually exclusive with the warning — both would contradict.
    expect(text).not.toContain("are not touched");
  });

  it("describes nothing at all without a preview", () => {
    expect(describeRemovalPreview(null)).toEqual([]);
    expect(describeRemovalPreview(undefined)).toEqual([]);
  });
});

describe("removalIsAllowed", () => {
  it("permits only an unblocked preview", () => {
    expect(removalIsAllowed({ blocked: null })).toBe(true);
    expect(removalIsAllowed({ blocked: "" })).toBe(true);
    expect(removalIsAllowed({ blocked: "a lease is active" })).toBe(false);
  });

  // No preview means no evidence the removal would succeed. Fail closed.
  it("refuses without a preview", () => {
    expect(removalIsAllowed(null)).toBe(false);
    expect(removalIsAllowed(undefined)).toBe(false);
  });
});

// ---- entry state, progress and actionable failure --------------------------------

// The states the store is still moving on its own are what a screen has to keep watching. Getting
// this set wrong in either direction is a real defect: too narrow and a copy in flight looks
// frozen forever, too wide and a settled cache polls for nothing.
describe("isTransitionalEntry / hasTransitionalEntries", () => {
  it.each([
    ["pending", true],
    ["materializing", true],
    ["evicting", true],
    ["complete", false],
    ["interrupted", false],
    ["corrupt", false],
  ])("classifies %s as transitional=%s", (state, expected) => {
    expect(isTransitionalEntry({ state })).toBe(expected);
  });

  // An unrecognized state is NOT treated as still-moving. Polling a state this build cannot
  // interpret would be a guess, and it would never terminate.
  it("does not treat an unrecognized state as still moving", () => {
    expect(isTransitionalEntry({ state: "quantizing" })).toBe(false);
    expect(isTransitionalEntry(null)).toBe(false);
    expect(isTransitionalEntry({})).toBe(false);
  });

  it("reports a snapshot as converging only while one of its entries is", () => {
    expect(hasTransitionalEntries({ entries: [{ state: "complete" }] })).toBe(false);
    expect(
      hasTransitionalEntries({ entries: [{ state: "complete" }, { state: "materializing" }] }),
    ).toBe(true);
    // No answer is not "converging": a failed or absent read must not start a poll loop.
    expect(hasTransitionalEntries({ entries: [] })).toBe(false);
    expect(hasTransitionalEntries(null)).toBe(false);
    expect(hasTransitionalEntries({})).toBe(false);
  });
});

describe("describeEntryState", () => {
  // A complete entry's line is about what happens to it next, and it must flip with the pin.
  it("describes a complete copy by whether it is kept", () => {
    const unpinned = describeEntryState({ state: "complete", pinned: false });
    expect(unpinned.text).toContain("removed automatically");
    expect(unpinned.progress).toBe(false);
    expect(unpinned.failure).toBe(false);
    expect(describeEntryState({ state: "complete", pinned: true }).text).toContain("Kept");
  });

  it.each([
    ["pending", "Queued"],
    ["materializing", "Copying now"],
    ["evicting", "Removing now"],
  ])("marks %s as progress and says what is happening", (state, fragment) => {
    const described = describeEntryState({ state });
    expect(described.progress).toBe(true);
    expect(described.failure).toBe(false);
    expect(described.text).toContain(fragment);
  });

  // The point of a failure line is that it is ACTIONABLE. A bare state name tells a user neither
  // whether the model still works nor what to press, which is exactly the gap these replace.
  it("gives an interrupted copy a remedy rather than a state name", () => {
    const described = describeEntryState({ state: "interrupted" });
    expect(described.failure).toBe(true);
    expect(described.progress).toBe(false);
    expect(described.text).toContain("can't be used");
    expect(described.text).toMatch(/remove it now to reclaim the space/i);
  });

  it("says a corrupt copy is unrepairable and that the model still loads", () => {
    const described = describeEntryState({ state: "corrupt" });
    expect(described.failure).toBe(true);
    expect(described.text).toContain("can't be used or repaired");
    expect(described.text).toContain("still loads from its own library");
  });

  // Same rule the availability badge follows: a state this build does not know is labelled as
  // unknown, never coerced into a reassuring one.
  it("labels an unrecognized state instead of implying the copy is fine", () => {
    const described = describeEntryState({ state: "quantizing" });
    expect(described.failure).toBe(true);
    expect(described.text).toContain("quantizing");
    expect(described.text).toMatch(/doesn’t recognize/);
    expect(described.text).not.toMatch(/can be removed automatically|Kept/);
  });
});
