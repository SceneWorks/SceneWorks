import { describe, it, expect, vi } from "vitest";
import {
  createModelLibraryGate,
  modelLibraryContext,
  modelLibraryContextForModel,
  modelLibraryUnavailable,
  rethrowUnlessPrompted,
  setModelLibraryHandler,
  ModelLibraryPrompted,
  MODEL_LIBRARY_UNAVAILABLE_CODE,
} from "./modelLibrary.js";

function unavailableError(overrides = {}) {
  return {
    code: MODEL_LIBRARY_UNAVAILABLE_CODE,
    message: "Model 'z-image' is installed on an external model library…",
    context: {
      schemaVersion: 1,
      availability: "installed_external_unavailable",
      modelId: "z-image",
      modelName: "Z-Image",
      configuredLibraryPath: "/Volumes/Models/hf/hub",
      expectedLibraryPath: "/Volumes/Models/hf/hub",
      expectedVolumeId: "macos-volume:abc",
      ...overrides,
    },
  };
}

// A probe whose answer the test releases by hand, so a second event can be injected while the
// first attempt is genuinely still in flight.
function deferredProbe() {
  let release;
  const gate = new Promise((resolve) => {
    release = resolve;
  });
  const probe = vi.fn(() => gate);
  return { probe, release: (value) => release(value) };
}

describe("modelLibraryContext", () => {
  it("recognizes only the typed rejection, never a message that reads like one", () => {
    expect(modelLibraryContext(unavailableError())?.modelId).toBe("z-image");
    expect(
      modelLibraryContext({
        message: "external model library is currently unavailable",
        status: 503,
      }),
    ).toBeNull();
    // Code without the typed payload is not enough to open a prompt that must name a library.
    expect(
      modelLibraryContext({ code: MODEL_LIBRARY_UNAVAILABLE_CODE }),
    ).toBeNull();
    // A different typed availability is a different story (missing/incomplete keep their
    // established download path).
    expect(
      modelLibraryContext(unavailableError({ availability: "missing" })),
    ).toBeNull();
  });

  it("reads a catalog row's availability from the seam rather than re-deriving it", () => {
    const row = {
      id: "z-image",
      name: "Z-Image",
      installState: "installed",
      modelAvailability: "installed_external_unavailable",
      modelResolution: {
        schemaVersion: 1,
        configuredLibraryPath: "/Volumes/Models/hf/hub",
        expectedLibrary: {
          canonicalPath: "/Volumes/Models/hf/hub",
          physicalIdentity: { volumeId: "macos-volume:abc" },
        },
      },
    };
    expect(modelLibraryUnavailable(row)).toBe(true);
    expect(modelLibraryContextForModel(row)).toMatchObject({
      modelId: "z-image",
      modelName: "Z-Image",
      expectedLibraryPath: "/Volumes/Models/hf/hub",
      expectedVolumeId: "macos-volume:abc",
    });
    // A complete local model is never prompted about.
    expect(
      modelLibraryUnavailable({ id: "local", modelAvailability: "local_ready" }),
    ).toBe(false);
    expect(
      modelLibraryContextForModel({ id: "local", modelAvailability: "local_ready" }),
    ).toBeNull();
  });
});

describe("model library gate", () => {
  it("resumes the blocked action exactly once when the library comes back", async () => {
    const action = vi.fn(async () => "job-1");
    const probe = vi.fn(async () => ({ available: true }));
    const gate = createModelLibraryGate({ probe });

    expect(gate.block(unavailableError().context, action)).toBe(true);
    expect(gate.getState().status).toBe("blocked");
    expect(action).not.toHaveBeenCalled();

    await expect(gate.retry()).resolves.toBe("job-1");
    expect(action).toHaveBeenCalledTimes(1);
    expect(gate.getState()).toMatchObject({ status: "idle", context: null });

    // The pending slot is empty: a retry after the resume can never re-fire it.
    await gate.retry();
    expect(action).toHaveBeenCalledTimes(1);
  });

  it("does not double-submit when a reconnect event lands while a retry is in flight", async () => {
    const action = vi.fn(async () => "job-1");
    const { probe, release } = deferredProbe();
    const gate = createModelLibraryGate({ probe });
    gate.block(unavailableError().context, action);

    const first = gate.retry();
    // The drive-watcher fires again, and the user clicks the button, both mid-probe.
    const second = gate.retry({ auto: true });
    const third = gate.retry();
    release({ available: true });
    await Promise.all([first, second, third]);

    expect(probe).toHaveBeenCalledTimes(1);
    expect(action).toHaveBeenCalledTimes(1);
  });

  it("keeps exactly one pending action when a second rejection arrives while blocked", async () => {
    const first = vi.fn(async () => "job-1");
    const second = vi.fn(async () => "job-2");
    const gate = createModelLibraryGate({ probe: async () => ({ available: true }) });

    gate.block(unavailableError().context, first);
    // The second click is owned by the gate (no second dialog, no second error banner) but its
    // action is dropped rather than queued behind the first.
    expect(gate.block(unavailableError().context, second)).toBe(true);
    await gate.retry();

    expect(first).toHaveBeenCalledTimes(1);
    expect(second).not.toHaveBeenCalled();
  });

  it("stays blocked, and keeps the action runnable, when the library is still missing", async () => {
    const action = vi.fn(async () => "job-1");
    const probe = vi
      .fn()
      .mockResolvedValueOnce({ available: false })
      .mockResolvedValueOnce({ available: true });
    const gate = createModelLibraryGate({ probe });
    gate.block(unavailableError().context, action);

    await gate.retry();
    expect(action).not.toHaveBeenCalled();
    expect(gate.getState()).toMatchObject({ status: "blocked" });
    expect(gate.getState().hint).toBeTruthy();

    await gate.retry();
    expect(action).toHaveBeenCalledTimes(1);
  });

  it("reports a probe failure without losing the pending action, and never as a raw throw", async () => {
    const action = vi.fn(async () => "job-1");
    const probe = vi
      .fn()
      .mockRejectedValueOnce(new Error("Request failed with 500"))
      .mockResolvedValueOnce({ available: true });
    const gate = createModelLibraryGate({ probe });
    gate.block(unavailableError().context, action);

    await expect(gate.retry()).resolves.toBeNull();
    expect(gate.getState()).toMatchObject({ status: "blocked" });
    expect(gate.getState().error).toBeTruthy();

    await gate.retry();
    expect(action).toHaveBeenCalledTimes(1);
  });

  it("leaves nothing queued on cancel — including a retry already in flight", async () => {
    const action = vi.fn(async () => "job-1");
    const { probe, release } = deferredProbe();
    const gate = createModelLibraryGate({ probe });
    gate.block(unavailableError().context, action);

    const inFlight = gate.retry();
    gate.cancel();
    expect(gate.getState()).toMatchObject({ status: "idle", context: null });
    // The drive really did come back — but the user already cancelled, so nothing may submit.
    release({ available: true });
    await inFlight;
    expect(action).not.toHaveBeenCalled();

    await gate.retry();
    expect(action).not.toHaveBeenCalled();
  });

  it("validates, then persists, then adopts — and queues nothing afterwards", async () => {
    const action = vi.fn(async () => "job-1");
    const target = {
      libraryRoot: "/Volumes/Models 1/hf/hub",
      hfHome: "/Volumes/Models 1/hf",
      adopted: false,
    };
    const order = [];
    const validate = vi.fn(async () => {
      order.push("validate");
      return target;
    });
    const persist = vi.fn(async () => {
      order.push("persist");
      return null;
    });
    const adopt = vi.fn(async () => {
      order.push("adopt");
    });
    const gate = createModelLibraryGate({
      probe: async () => ({ available: false }),
      validate,
      persist,
      adopt,
    });
    gate.block(unavailableError().context, action);

    await expect(gate.relocate("/Volumes/Models 1/hf")).resolves.toBe(target);
    // Validation runs BEFORE either durable write, so an ordinary refusal never leaves the two
    // copies of the location disagreeing.
    expect(order).toEqual(["validate", "persist", "adopt"]);
    expect(validate).toHaveBeenCalledWith("/Volumes/Models 1/hf");
    expect(persist).toHaveBeenCalledWith(target);
    expect(gate.getState()).toMatchObject({ status: "idle", relocated: target });
    // Relocation takes effect on the next launch, so resuming now would only be refused again:
    // the action is dropped, exactly like cancel, rather than left queued.
    expect(action).not.toHaveBeenCalled();
    await gate.retry();
    expect(action).not.toHaveBeenCalled();
  });

  it("keeps the prompt open with actionable guidance when a relocation is rejected", async () => {
    const action = vi.fn(async () => "job-1");
    const rejection = new Error(
      "That folder does not contain a SceneWorks model library.",
    );
    const validate = vi.fn().mockRejectedValueOnce(rejection);
    const persist = vi.fn();
    const adopt = vi.fn();
    const gate = createModelLibraryGate({
      probe: async () => ({ available: true }),
      validate,
      persist,
      adopt,
    });
    gate.block(unavailableError().context, action);

    await expect(gate.relocate("/Users/me/Pictures")).resolves.toBeNull();
    // Nothing was written anywhere, which is the entire point of validating first.
    expect(persist).not.toHaveBeenCalled();
    expect(adopt).not.toHaveBeenCalled();
    expect(gate.getState()).toMatchObject({ status: "blocked" });
    expect(gate.getState().error).toBe(rejection.message);
    // The original action survived the rejected attempt.
    await gate.retry();
    expect(action).toHaveBeenCalledTimes(1);
  });

  // The two durable writes live in different places (the shell's settings file, the server's
  // binding ledger). If the second fails, the first must be undone — and whatever the user is told
  // has to match the state they are actually in.
  it("undoes the persisted location when the server's re-bind fails, and says so", async () => {
    const action = vi.fn(async () => "job-1");
    const target = { libraryRoot: "/new/hub", hfHome: "/new" };
    const undo = vi.fn(async () => {});
    const persist = vi.fn(async () => undo);
    const adopt = vi.fn().mockRejectedValueOnce(new Error("the library went away"));
    const gate = createModelLibraryGate({
      probe: async () => ({ available: true }),
      validate: async () => target,
      persist,
      adopt,
    });
    gate.block(unavailableError().context, action);

    await expect(gate.relocate("/new")).resolves.toBeNull();
    expect(undo).toHaveBeenCalledTimes(1);
    expect(gate.getState().error).toContain("previous location is still in use");
    expect(gate.getState()).toMatchObject({ status: "blocked" });
    // The blocked submission is still resumable — nothing was lost by the failed attempt.
    await gate.retry();
    expect(action).toHaveBeenCalledTimes(1);
  });

  it("says the previous location could NOT be restored when the undo also fails", async () => {
    const undo = vi.fn().mockRejectedValueOnce(new Error("settings file is read-only"));
    const gate = createModelLibraryGate({
      probe: async () => ({ available: true }),
      validate: async () => ({ libraryRoot: "/new/hub", hfHome: "/new" }),
      persist: async () => undo,
      adopt: async () => {
        throw new Error("re-bind failed");
      },
    });
    gate.block(unavailableError().context, vi.fn());

    await expect(gate.relocate("/new")).resolves.toBeNull();
    expect(gate.getState().error).toContain("could not restore the previous location");
    expect(gate.getState().error).toContain("Settings");
  });

  it("clears a stale error when a later retry answers with a hint", async () => {
    const gate = createModelLibraryGate({
      probe: vi
        .fn()
        .mockRejectedValueOnce(new Error("Request failed with 500"))
        .mockResolvedValueOnce({ available: false }),
    });
    gate.block(unavailableError().context, vi.fn());

    await gate.retry();
    expect(gate.getState().error).toBeTruthy();
    await gate.retry();
    expect(gate.getState().error).toBe("");
    expect(gate.getState().hint).toBeTruthy();
  });

  // The throwing submission paths (training) must not print a second surface beside the dialog,
  // and must still propagate every other failure untouched.
  it("converts a typed rejection into a silent throw for propagating call sites", async () => {
    const action = vi.fn(async () => "training-job-1");
    const gate = createModelLibraryGate({ probe: async () => ({ available: true }) });
    const unregister = setModelLibraryHandler(gate.block);

    expect(() => rethrowUnlessPrompted(unavailableError(), action)).toThrow(
      ModelLibraryPrompted,
    );
    // An empty message is what makes `setError(err.message)` CLEAR rather than print.
    try {
      rethrowUnlessPrompted(unavailableError(), action);
    } catch (error) {
      expect(error.message).toBe("");
    }
    expect(gate.getState()).toMatchObject({ status: "blocked" });
    await gate.retry();
    expect(action).toHaveBeenCalledTimes(1);

    const ordinary = new Error("network down");
    expect(() => rethrowUnlessPrompted(ordinary, action)).toThrow(ordinary);
    unregister();
  });

  it("notifies subscribers on every settled transition", async () => {
    const listener = vi.fn();
    const gate = createModelLibraryGate({ probe: async () => ({ available: true }) });
    const unsubscribe = gate.subscribe(listener);
    gate.block(unavailableError().context, async () => "job-1");
    await gate.retry();
    expect(listener).toHaveBeenCalled();
    unsubscribe();
    const seen = listener.mock.calls.length;
    gate.block(unavailableError().context, async () => "job-2");
    expect(listener.mock.calls.length).toBe(seen);
  });
});
