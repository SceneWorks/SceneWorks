import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  CHECKPOINT_LIBRARY_NOT_PERMITTED_CODE,
  CHECKPOINT_LIBRARY_REJECTED_CODE,
  LINKED_NEEDS_RELINK,
  LINKED_NEEDS_RESCAN,
  LINKED_READY,
  MANAGED_SOURCES,
  OWNERSHIP_CHOICES,
  OWNERSHIP_LINKED,
  OWNERSHIP_MANAGED,
  approveLibraryRoot,
  checkpointLibraryRefusal,
  describeRefusal,
  duplicateCheckpointIds,
  duplicateWarningText,
  fetchLibraryRoots,
  isLocalOnlyRefusal,
  linkedCorrection,
  linkedImportBody,
  linkedStatusIndex,
  managedImportBody,
  managedSourceProblem,
  modelCheckpointId,
  modelLinkedStatus,
  modelOwnership,
  modelProvenance,
  removalCopy,
  removeLibraryRoot,
  rescanLibraryCheckpoint,
  rootRemovalCopy,
  scanLibraryRoot,
  updateLibraryRoot,
} from "./checkpointLibrary.js";

const fetchMock = vi.fn();

beforeEach(() => {
  fetchMock.mockReset();
  fetchMock.mockResolvedValue({ ok: true, status: 200, json: async () => ({}) });
  vi.stubGlobal("fetch", fetchMock);
});

afterEach(() => {
  vi.unstubAllGlobals();
});

function lastRequest() {
  const [url, init] = fetchMock.mock.calls.at(-1);
  return { url, init, body: init?.body ? JSON.parse(init.body) : null };
}

describe("ownership choices", () => {
  it("offers exactly the two named ownership choices, linked first", () => {
    expect(OWNERSHIP_CHOICES.map((choice) => choice.id)).toEqual([OWNERSHIP_LINKED, OWNERSHIP_MANAGED]);
    expect(OWNERSHIP_CHOICES.map((choice) => choice.label)).toEqual([
      "Use existing model library",
      "Add to SceneWorks",
    ]);
  });

  it("offers all five managed inputs, keyed by the wire discriminants", () => {
    expect(MANAGED_SOURCES.map((source) => source.kind)).toEqual([
      "upload",
      "localPath",
      "url",
      "huggingFace",
      "civitai",
    ]);
  });
});

describe("transport", () => {
  it("addresses every library-root route the API serves", async () => {
    await fetchLibraryRoots("tok");
    expect(lastRequest().url).toContain("/api/v1/models/library-roots");

    await approveLibraryRoot("tok", { path: "/Volumes/Models", label: "Drive" });
    expect(lastRequest()).toMatchObject({
      init: { method: "POST" },
      body: { path: "/Volumes/Models", label: "Drive" },
    });

    await updateLibraryRoot("tok", "root-a b", { path: "/Volumes/New" });
    expect(lastRequest().url).toContain("/api/v1/models/library-roots/root-a%20b");
    expect(lastRequest().init.method).toBe("PATCH");
    expect(lastRequest().body).toEqual({ path: "/Volumes/New" });

    await removeLibraryRoot("tok", "root-a");
    expect(lastRequest().init.method).toBe("DELETE");

    await scanLibraryRoot("tok", "root-a");
    expect(lastRequest().url).toContain("/api/v1/models/library-roots/root-a/scan");

    await rescanLibraryCheckpoint("tok", "root-a", "sdxl/model.safetensors");
    expect(lastRequest().url).toContain("/api/v1/models/library-roots/root-a/rescan");
    expect(lastRequest().body).toEqual({ relativePath: "sdxl/model.safetensors" });
  });

  it("omits an absent label rather than sending null", async () => {
    await approveLibraryRoot("tok", { path: "/Volumes/Models" });
    expect(lastRequest().body).toEqual({ path: "/Volumes/Models" });
  });
});

describe("request bodies", () => {
  it("names a linked import by root id and relative path and nothing else", () => {
    const body = linkedImportBody({ rootId: "root-a", relativePath: "sdxl/model.safetensors", name: "My SDXL" });
    expect(body).toEqual({
      linkedRootId: "root-a",
      linkedRelativePath: "sdxl/model.safetensors",
      type: "image",
      name: "My SDXL",
    });
    // The route refuses a linked body that also names a transfer, and `ownershipMode: "linked"` is
    // not a served value.
    expect(body).not.toHaveProperty("ownershipMode");
    expect(body).not.toHaveProperty("source");
    expect(body).not.toHaveProperty("repo");
    expect(body).not.toHaveProperty("sourceUrl");
  });

  it("builds one discriminated source per managed input", () => {
    expect(managedImportBody({ kind: "upload" })).toMatchObject({
      ownershipMode: OWNERSHIP_MANAGED,
      source: { kind: "upload" },
    });
    expect(managedImportBody({ kind: "upload" }).source).not.toHaveProperty("stagedPath");
    expect(managedImportBody({ kind: "localPath", path: "/tmp/x" }).source).toEqual({
      kind: "localPath",
      path: "/tmp/x",
    });
    expect(managedImportBody({ kind: "url", url: "https://x/y.safetensors", expectedSha256: "ab" }).source).toEqual({
      kind: "url",
      url: "https://x/y.safetensors",
      expectedSha256: "ab",
    });
    expect(managedImportBody({ kind: "huggingFace", repo: "org/m", revision: "main" }).source).toEqual({
      kind: "huggingFace",
      repo: "org/m",
      revision: "main",
    });
    expect(managedImportBody({ kind: "civitai", url: "https://civitai.com/x", modelVersionId: "9", fileId: "3" }).source).toEqual({
      kind: "civitai",
      url: "https://civitai.com/x",
      modelVersionId: "9",
      fileId: "3",
    });
  });

  it("refuses to build a body for an unknown source instead of defaulting to one", () => {
    expect(() => managedImportBody({ kind: "ftp" })).toThrow(/Unknown managed import source/);
  });

  it("names the missing field per source", () => {
    expect(managedSourceProblem({ kind: "upload" })).toMatch(/file/i);
    expect(managedSourceProblem({ kind: "upload", file: {} })).toBe("");
    expect(managedSourceProblem({ kind: "huggingFace", repo: "  " })).toMatch(/owner\/name/);
    expect(managedSourceProblem({ kind: "huggingFace", repo: "org/m" })).toBe("");
    expect(managedSourceProblem({ kind: "civitai", url: "" })).toMatch(/Civitai/);
    expect(managedSourceProblem({})).toMatch(/where the checkpoint comes from/);
  });
});

describe("refusals", () => {
  const rejected = {
    code: CHECKPOINT_LIBRARY_REJECTED_CODE,
    context: { reason: "source-drifted" },
    message: "[checkpoint-plan:source-drifted] recorded ab… now cd…",
  };

  it("surfaces the store's own code and sentence", () => {
    const refusal = checkpointLibraryRefusal(rejected);
    expect(refusal).toMatchObject({ reason: "source-drifted", action: "rescan" });
    expect(refusal.message).toContain("[checkpoint-plan:source-drifted]");
  });

  it("is null for a code without a reason and for a reason without the code", () => {
    expect(checkpointLibraryRefusal({ code: CHECKPOINT_LIBRARY_REJECTED_CODE })).toBeNull();
    expect(checkpointLibraryRefusal({ context: { reason: "source-drifted" } })).toBeNull();
    expect(checkpointLibraryRefusal({ code: "some_other_code", context: { reason: "x" } })).toBeNull();
  });

  it("maps root-unavailable to relink and source-missing to rescan", () => {
    expect(checkpointLibraryRefusal({ ...rejected, context: { reason: "root-unavailable" } }).action).toBe("relink");
    expect(checkpointLibraryRefusal({ ...rejected, context: { reason: "source-missing" } }).action).toBe("rescan");
    // A refusal with no one-click correction still carries its reason — it is explained, not hidden.
    const contract = checkpointLibraryRefusal({ ...rejected, context: { reason: "unrunnable-source" } });
    expect(contract.action).toBeNull();
    expect(contract.reason).toBe("unrunnable-source");
  });

  it("never collapses a typed refusal into a generic message", () => {
    const described = describeRefusal(rejected);
    expect(described.message).toContain("[checkpoint-plan:source-drifted]");
    expect(described.message).not.toMatch(/something went wrong/i);
    expect(described.reason).toBe("source-drifted");
  });

  it("keeps an untyped failure's own message", () => {
    expect(describeRefusal(new Error("network down")).message).toBe("network down");
    expect(describeRefusal(null).message).toBe("The request could not be completed.");
  });

  it("recognises the local-only refusal", () => {
    expect(isLocalOnlyRefusal({ code: CHECKPOINT_LIBRARY_NOT_PERMITTED_CODE })).toBe(true);
    expect(isLocalOnlyRefusal(rejected)).toBe(false);
  });
});

describe("linked state", () => {
  it("gives Needs Relink and Needs Rescan a corrective action, not a missing verdict", () => {
    const relink = linkedCorrection({ state: LINKED_NEEDS_RELINK, detail: "[checkpoint-plan:root-unavailable] …" });
    expect(relink).toMatchObject({ action: "relink", label: "Relink library", headline: "Needs relink" });
    expect(relink.summary).not.toMatch(/missing|not installed|uninstalled/i);
    expect(relink.detail).toContain("[checkpoint-plan:root-unavailable]");

    const rescan = linkedCorrection({ state: LINKED_NEEDS_RESCAN, detail: "" });
    expect(rescan).toMatchObject({ action: "rescan", label: "Rescan checkpoint", headline: "Needs rescan" });
    expect(rescan.summary).not.toMatch(/missing|not installed|uninstalled/i);
  });

  it("has no correction for a ready checkpoint", () => {
    expect(linkedCorrection({ state: LINKED_READY })).toBeNull();
    expect(linkedCorrection(null)).toBeNull();
  });

  it("indexes both matched candidates and unmatched persisted checkpoints", () => {
    const index = linkedStatusIndex([
      {
        candidates: [{ status: { checkpointId: "linked/root-a/a.safetensors", state: LINKED_READY } }, { status: null }],
        unmatched: [{ checkpointId: "linked/root-a/gone.safetensors", state: LINKED_NEEDS_RESCAN }],
      },
      null,
    ]);
    expect(index.get("linked/root-a/a.safetensors").state).toBe(LINKED_READY);
    // The disappeared one must still be findable: it is a catalog row the user can act on.
    expect(index.get("linked/root-a/gone.safetensors").state).toBe(LINKED_NEEDS_RESCAN);
  });

  it("reads ownership off the persisted checkpoint id", () => {
    expect(modelOwnership({ importPlan: { checkpointId: "linked/root-a/x.safetensors" } })).toBe(OWNERSHIP_LINKED);
    expect(modelOwnership({ importPlan: { checkpointId: "managed/install-1" } })).toBe(OWNERSHIP_MANAGED);
    expect(modelOwnership({})).toBeNull();
    expect(modelCheckpointId({ importPlan: { checkpointId: "" } })).toBeNull();
  });

  it("finds a linked row's state and leaves a managed row alone", () => {
    const index = linkedStatusIndex([
      { candidates: [{ status: { checkpointId: "linked/root-a/x.safetensors", state: LINKED_NEEDS_RELINK } }] },
    ]);
    expect(modelLinkedStatus({ importPlan: { checkpointId: "linked/root-a/x.safetensors" } }, index).state).toBe(
      LINKED_NEEDS_RELINK,
    );
    expect(modelLinkedStatus({ importPlan: { checkpointId: "managed/install-1" } }, index)).toBeNull();
  });
});

describe("provenance", () => {
  it("names a linked row's library and a fetched row's source", () => {
    expect(
      modelProvenance({ source: { provider: "linked-library", rootId: "root-a", relativePath: "sdxl/m.safetensors" } }),
    ).toMatchObject({ label: "Linked library", reference: "sdxl/m.safetensors", rootId: "root-a" });
    expect(modelProvenance({ source: { provider: "huggingface", repo: "org/m" } })).toMatchObject({
      label: "Hugging Face",
      reference: "org/m",
    });
    expect(modelProvenance({ source: { provider: "civitai", url: "https://civitai.com/x" } }).label).toBe("Civitai");
  });

  it("is null when the row records none, rather than printing 'unknown'", () => {
    expect(modelProvenance({})).toBeNull();
    expect(modelProvenance({ source: {} })).toBeNull();
  });
});

describe("duplicates", () => {
  it("reads the ids off the completed job result", () => {
    expect(duplicateCheckpointIds({ result: { duplicateCheckpointIds: ["managed/a", "", 4] } })).toEqual(["managed/a"]);
    expect(duplicateCheckpointIds({ result: {} })).toEqual([]);
    expect(duplicateWarningText({ result: { duplicateCheckpointIds: ["managed/a"] } })).toContain("managed/a");
    expect(duplicateWarningText({ result: { duplicateCheckpointIds: ["a", "b"] } })).toContain("2 checkpoints");
    expect(duplicateWarningText({})).toBe("");
  });
});

describe("removal copy", () => {
  it("promises the opposite things for linked and managed, and never crosses them", () => {
    const linked = removalCopy({ name: "My SDXL", importPlan: { checkpointId: "linked/root-a/x.safetensors" } });
    expect(linked.ownership).toBe(OWNERSHIP_LINKED);
    expect(linked.confirmLabel).toBe("Remove");
    expect(linked.message).toMatch(/never opened, moved or deleted/);
    expect(linked.message).not.toMatch(/removes those files from this machine/);

    const managed = removalCopy({ name: "Fetched", importPlan: { checkpointId: "managed/install-1" } });
    expect(managed.ownership).toBe(OWNERSHIP_MANAGED);
    expect(managed.confirmLabel).toBe("Delete");
    expect(managed.message).toMatch(/removes those files from this machine/);
    expect(managed.message).not.toMatch(/never opened, moved or deleted/);
  });

  it("treats a row with no plan as managed, which is the safe default for a delete warning", () => {
    expect(removalCopy({ name: "Legacy" }).ownership).toBe(OWNERSHIP_MANAGED);
  });

  it("says a forgotten library keeps every file", () => {
    const copy = rootRemovalCopy({ displayLabel: "Drive" }, { candidates: [{ status: {} }, { status: null }] });
    expect(copy.message).toContain("Drive");
    expect(copy.message).toContain("1 checkpoint");
    expect(copy.message).toMatch(/left exactly as they are/);
    expect(copy.confirmLabel).toBe("Forget library");
  });
});
