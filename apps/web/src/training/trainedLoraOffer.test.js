import { afterEach, beforeEach, describe, expect, it } from "vitest";

import {
  characterLoraAttachPayload,
  isRegisteredAdapterTrainingJob,
  readTrainedLoraOfferDecisions,
  rememberTrainedLoraOfferDecision,
  trainedLoraOfferCandidate,
} from "./trainedLoraOffer.js";

// sc-24815. The offer is the only place the app writes a character↔LoRA link on the
// user's behalf, and `attach_lora` PREPENDS a new row per call rather than upserting,
// so these tests are mostly about when the answer must be `null`.

const NOW = Date.parse("2026-10-02T12:00:00.000Z");
const RECENT = "2026-10-02T11:50:00.000Z";

function adapterJob(overrides = {}) {
  return {
    id: "job-train-1",
    type: "lora_train",
    status: "completed",
    projectId: "project-a",
    completedAt: RECENT,
    payload: {
      datasetId: "dataset-1",
      outputName: "Mira v3",
      manifestEntry: {
        id: "mira_v3",
        name: "Mira v3",
        scope: "project",
        family: "z-image",
        triggerWords: ["mira"],
        source: { provider: "training", path: "loras/mira_v3" },
      },
    },
    result: { loraRegistered: true, loraId: "mira_v3" },
    ...overrides,
  };
}

function context(overrides = {}) {
  return {
    jobs: [adapterJob()],
    datasets: [{ id: "dataset-1", name: "Mira Set", characterId: "char-mira" }],
    characters: [{ id: "char-mira", name: "Mira", loras: [] }],
    loras: [],
    projectId: "project-a",
    now: NOW,
    ...overrides,
  };
}

describe("isRegisteredAdapterTrainingJob", () => {
  it("accepts a completed run whose adapter registered", () => {
    expect(isRegisteredAdapterTrainingJob(adapterJob())).toBe(true);
  });

  it("rejects a base-checkpoint run, which reports under its own result key", () => {
    const job = adapterJob({
      result: { baseCheckpointRegistered: true, baseCheckpointId: "mira_full" },
    });
    expect(isRegisteredAdapterTrainingJob(job)).toBe(false);
  });

  it("rejects a failed registration, a dry run, a running job and a control overlay", () => {
    expect(
      isRegisteredAdapterTrainingJob(adapterJob({ result: { loraRegistered: false, loraRegistrationError: "no adapter" } })),
    ).toBe(false);
    expect(isRegisteredAdapterTrainingJob(adapterJob({ result: {} }))).toBe(false);
    expect(isRegisteredAdapterTrainingJob(adapterJob({ status: "running" }))).toBe(false);
    expect(isRegisteredAdapterTrainingJob(adapterJob({ type: "control_training" }))).toBe(false);
  });
});

describe("trainedLoraOfferCandidate", () => {
  it("names the trained LoRA and the character that owns the dataset", () => {
    const candidate = trainedLoraOfferCandidate(context());
    expect(candidate).toMatchObject({
      jobId: "job-train-1",
      projectId: "project-a",
      loraId: "mira_v3",
      loraName: "Mira v3",
      characterId: "char-mira",
      characterName: "Mira",
    });
  });

  it("surfaces no offer for a dataset with no character", () => {
    expect(trainedLoraOfferCandidate(context({ datasets: [{ id: "dataset-1", characterId: "" }] }))).toBeNull();
    expect(trainedLoraOfferCandidate(context({ datasets: [{ id: "dataset-1" }] }))).toBeNull();
    expect(trainedLoraOfferCandidate(context({ datasets: [] }))).toBeNull();
  });

  it("surfaces no offer for a base-checkpoint run on a character dataset", () => {
    expect(
      trainedLoraOfferCandidate(
        context({ jobs: [adapterJob({ result: { baseCheckpointRegistered: true, baseCheckpointId: "mira_full" } })] }),
      ),
    ).toBeNull();
  });

  it("surfaces no offer once that job id is decided, but still offers an older undecided run", () => {
    const older = adapterJob({ id: "job-train-0", completedAt: "2026-10-02T10:00:00.000Z" });
    expect(trainedLoraOfferCandidate(context({ decidedJobIds: new Set(["job-train-1"]) }))).toBeNull();
    expect(
      trainedLoraOfferCandidate(context({ jobs: [adapterJob(), older], decidedJobIds: new Set(["job-train-1"]) }))?.jobId,
    ).toBe("job-train-0");
  });

  it("ignores another project's run and a run whose character is gone", () => {
    expect(trainedLoraOfferCandidate(context({ projectId: "project-b" }))).toBeNull();
    expect(trainedLoraOfferCandidate(context({ characters: [] }))).toBeNull();
    expect(trainedLoraOfferCandidate(context({ characters: [{ id: "char-mira", name: "Mira", archived: true }] }))).toBeNull();
  });

  it("does not resurrect a run that finished outside the offer window", () => {
    const stale = adapterJob({ completedAt: "2026-09-01T11:50:00.000Z" });
    expect(trainedLoraOfferCandidate(context({ jobs: [stale] }))).toBeNull();
    // A missing timestamp is not read as "just now" either — the offer fails closed.
    expect(trainedLoraOfferCandidate(context({ jobs: [{ ...stale, completedAt: null, updatedAt: null, createdAt: null }] }))).toBeNull();
  });

  it("builds the same payload Character Studio's manual attach submits", () => {
    const candidate = trainedLoraOfferCandidate(context());
    expect(candidate.payload).toEqual({
      loraId: "mira_v3",
      name: "Mira v3",
      sourcePath: "loras/mira_v3",
      triggerWords: ["mira"],
      defaultWeight: 1.0,
      compatibility: { families: ["z-image"] },
      scope: "project",
    });
  });

  it("prefers the refreshed catalog entry over the staged manifest entry", () => {
    const catalogLora = {
      id: "mira_v3",
      name: "Mira v3 (renamed)",
      installedPath: "/models/loras/mira_v3.safetensors",
      triggerWords: ["mira", "m"],
      defaultWeight: 0.8,
      families: ["z-image"],
      scope: "global",
    };
    const candidate = trainedLoraOfferCandidate(context({ loras: [catalogLora] }));
    expect(candidate.payload).toEqual({
      loraId: "mira_v3",
      name: "Mira v3 (renamed)",
      sourcePath: "/models/loras/mira_v3.safetensors",
      triggerWords: ["mira", "m"],
      defaultWeight: 0.8,
      compatibility: { families: ["z-image"] },
      scope: "global",
    });
  });

  it("still offers when the catalog has not refreshed, from the run's manifest entry", () => {
    const job = adapterJob({ payload: { datasetId: "dataset-1" } });
    const candidate = trainedLoraOfferCandidate(context({ jobs: [job] }));
    expect(candidate.loraName).toBe("mira_v3");
    expect(candidate.payload.sourcePath).toBeNull();
  });
});

describe("characterLoraAttachPayload", () => {
  it("falls back to the id for a blank name and refuses an id-less source", () => {
    expect(characterLoraAttachPayload({ id: "mira_v3", name: "   " }).name).toBe("mira_v3");
    expect(characterLoraAttachPayload({ name: "No id" })).toBeNull();
    expect(characterLoraAttachPayload(null)).toBeNull();
  });
});

describe("trained LoRA offer decisions", () => {
  beforeEach(() => {
    window.localStorage.clear();
  });

  afterEach(() => {
    window.localStorage.clear();
  });

  it("persists a decided job id per project so a reload cannot re-ask", () => {
    rememberTrainedLoraOfferDecision("project-a", "job-train-1");
    expect(readTrainedLoraOfferDecisions("project-a")).toEqual(["job-train-1"]);
    expect(readTrainedLoraOfferDecisions("project-b")).toEqual([]);
    // A reload reads the same record, and the candidate then comes back null.
    expect(
      trainedLoraOfferCandidate(
        context({ decidedJobIds: new Set(readTrainedLoraOfferDecisions("project-a")) }),
      ),
    ).toBeNull();
  });

  it("is idempotent and bounded", () => {
    rememberTrainedLoraOfferDecision("project-a", "job-train-1");
    rememberTrainedLoraOfferDecision("project-a", "job-train-1");
    expect(readTrainedLoraOfferDecisions("project-a")).toEqual(["job-train-1"]);
    for (let index = 0; index < 250; index += 1) {
      rememberTrainedLoraOfferDecision("project-a", `job-${index}`);
    }
    expect(readTrainedLoraOfferDecisions("project-a").length).toBe(200);
  });

  it("swallows a blocked storage layer and an unreadable blob", () => {
    window.localStorage.setItem("sceneworks-trained-lora-offer:project-a", "{not json");
    expect(readTrainedLoraOfferDecisions("project-a")).toEqual([]);
    const original = Object.getOwnPropertyDescriptor(globalThis, "localStorage");
    Object.defineProperty(globalThis, "localStorage", {
      configurable: true,
      get() {
        throw new Error("storage disabled");
      },
    });
    try {
      expect(readTrainedLoraOfferDecisions("project-a")).toEqual([]);
      expect(() => rememberTrainedLoraOfferDecision("project-a", "job-train-1")).not.toThrow();
    } finally {
      Object.defineProperty(globalThis, "localStorage", original);
    }
  });
});
