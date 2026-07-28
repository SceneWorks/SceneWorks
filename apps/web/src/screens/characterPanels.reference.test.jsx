import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { CharacterGenerationPanel } from "./characterPanels.jsx";

// sc-8851 ([F-049]): the reference-selection effect used to unconditionally snap
// `referenceAssetId` back to approvedReferences[0] on every render where the
// `approvedReferences` identity changed. useCharacters replaces that array (new
// identity, same contents) on every character mutation — including the panel's own
// upload flow (importAsset -> addCharacterReference), which sets the freshly uploaded
// id. The unconditional reset immediately undid that pick, so users generated against
// reference #1 instead of the reference they chose. The fix mirrors ImageStudio's
// guard: keep the current id while it is still an approved reference, and only fall
// back to [0] when the current id is absent/invalid. onUpload also now approves the
// reference before selecting it, so the selection lands on a valid (already-refetched)
// id rather than one the guard would reject.

const MODEL = {
  id: "instantid_realvisxl",
  name: "InstantID RealVisXL",
  family: "instantid",
  ui: { viewAngles: ["front", "left", "right"] },
};

// Minimal mode object exercising every mode.* field the panel reads, with a stub
// controller so we don't drag the real angle/pose controllers (and their fields)
// into a test that is only about reference selection.
const TEST_MODE = {
  eyebrow: "Angle set",
  defaultNegativePrompt: "",
  identityStructureMode: "angleSet",
  referenceRole: "angle-set-reference",
  referenceNoun: "angle-set reference",
  jobPredicate: () => false,
  title: () => "Turnaround",
  seedPrompt: () => "a prompt",
  renderIntro: () => null,
  useController: () => ({
    controls: null,
    advancedExtras: {},
    isReady: true,
    canSubmit: true,
    submitLabel: "Generate",
    expectedThumbnailCount: () => 0,
  }),
};

const character = { id: "char_1", name: "Ada", description: "a scientist" };

// Fresh reference objects each call so the array (and its element) identities differ
// on every "refetch", exactly as useCharacters produces after a mutation.
function refs(...assetIds) {
  return assetIds.map((assetId) => ({ assetId, role: "angle-set-reference", asset: null }));
}

// A stateful host that owns approvedReferences + selectedCharacter and passes an
// addCharacterReference that mutates that state — mirroring the App/useCharacters
// contract where a mutation triggers a refetch and hands the panel a new array. The
// ref exposes setters so a test can drive a refetch / character switch declaratively
// without a nested root.render (which pollutes React act state across tests).
function Host({ initialRefs, initialCharacter, importAsset, onAddCharacterReference, apiRef }) {
  const [approvedReferences, setApprovedReferences] = React.useState(initialRefs);
  const [selectedCharacter, setSelectedCharacter] = React.useState(initialCharacter);

  apiRef.current = {
    setApprovedReferences: (next) => setApprovedReferences(next),
    setSelectedCharacter: (next) => setSelectedCharacter(next),
  };

  const addCharacterReference = React.useCallback(
    async (characterId, payload) => {
      onAddCharacterReference?.(characterId, payload);
      // The refetch: fold the newly approved reference in under a fresh array identity.
      setApprovedReferences((current) => refs(...current.map((r) => r.assetId), payload.assetId));
      return {};
    },
    [onAddCharacterReference],
  );

  return (
    <CharacterGenerationPanel
      mode={TEST_MODE}
      selectedCharacter={selectedCharacter}
      model={MODEL}
      models={[MODEL]}
      catalog={[]}
      approvedReferences={approvedReferences}
      assets={[]}
      createImageJob={vi.fn()}
      importAsset={importAsset ?? vi.fn()}
      addCharacterReference={addCharacterReference}
      latestAssets={[]}
      imageLocalJobs={[]}
      loras={[]}
      rememberLocalGenerationJob={vi.fn()}
    />
  );
}

describe("CharacterGenerationPanel reference selection (sc-8851)", () => {
  let container;
  let root;

  beforeEach(() => {
    global.IS_REACT_ACT_ENVIRONMENT = true;
    // Some shared children portal to document.body; render into a body-attached node.
    container = document.createElement("div");
    document.body.appendChild(container);
    root = createRoot(container);
  });

  afterEach(async () => {
    await act(async () => root.unmount());
    container.remove();
  });

  async function mount(hostProps) {
    const apiRef = React.createRef();
    await act(async () => root.render(<Host {...hostProps} apiRef={apiRef} />));
    return apiRef;
  }

  // The active reference is the one whose thumbnail button is aria-pressed.
  function pickedId() {
    const active = container.querySelector('.reference-thumb[aria-pressed="true"]');
    return active?.getAttribute("aria-label")?.replace(/^Use /, "").replace(/ as the .*$/, "") ?? null;
  }

  function thumbFor(assetId) {
    return [...container.querySelectorAll(".reference-thumb")].find((button) =>
      button.getAttribute("aria-label")?.startsWith(`Use ${assetId} `),
    );
  }

  async function click(element) {
    await act(async () => {
      element.dispatchEvent(new window.MouseEvent("click", { bubbles: true }));
    });
  }

  it("defaults to the first approved reference on mount", async () => {
    await mount({ initialRefs: refs("ref_a", "ref_b", "ref_c"), initialCharacter: character });
    expect(pickedId()).toBe("ref_a");
  });

  it("preserves the picked id across a same-character refetch (new array identity, same contents)", async () => {
    const api = await mount({
      initialRefs: refs("ref_a", "ref_b", "ref_c"),
      initialCharacter: character,
    });

    // User picks reference #2.
    await click(thumbFor("ref_b"));
    expect(pickedId()).toBe("ref_b");

    // Character mutation -> useCharacters hands a brand-new approvedReferences array
    // with the same contents. Selection must NOT snap back to ref_a.
    await act(async () => api.current.setApprovedReferences(refs("ref_a", "ref_b", "ref_c")));
    expect(pickedId()).toBe("ref_b");
  });

  it("keeps a newly uploaded reference selected after the mutation-triggered refetch", async () => {
    // The panel's own upload flow end to end: importAsset resolves to a new asset, the
    // panel approves it via addCharacterReference (Host folds it into approvedReferences,
    // mimicking the refetch), and only then selects it. The pick must survive.
    const importAsset = vi.fn(async () => ({ id: "ref_new" }));
    const onAdd = vi.fn();
    await mount({
      initialRefs: refs("ref_a", "ref_b", "ref_c"),
      initialCharacter: character,
      importAsset,
      onAddCharacterReference: onAdd,
    });

    const fileInput = container.querySelector('input[type="file"]');
    const file = new File(["x"], "ref.png", { type: "image/png" });
    Object.defineProperty(fileInput, "files", { value: [file], configurable: true });
    await act(async () => {
      fileInput.dispatchEvent(new window.Event("change", { bubbles: true }));
    });

    expect(importAsset).toHaveBeenCalled();
    expect(onAdd).toHaveBeenCalledWith(
      "char_1",
      expect.objectContaining({ assetId: "ref_new", approved: true }),
    );
    // The upload-then-select flow survives the refetch: the panel ends up on the reference
    // the user just uploaded, not snapped back to reference #1.
    expect(pickedId()).toBe("ref_new");
  });

  it("falls back to [0] when the picked id disappears from approvedReferences", async () => {
    const api = await mount({
      initialRefs: refs("ref_a", "ref_b", "ref_c"),
      initialCharacter: character,
    });

    await click(thumbFor("ref_c"));
    expect(pickedId()).toBe("ref_c");

    // ref_c is removed (e.g. deleted/unapproved elsewhere). Fall back to the new [0].
    await act(async () => api.current.setApprovedReferences(refs("ref_b", "ref_a")));
    expect(pickedId()).toBe("ref_b");
  });

  it("resets to the new character's first reference on a genuine character switch", async () => {
    const api = await mount({
      initialRefs: refs("ref_a", "ref_b", "ref_c"),
      initialCharacter: character,
    });

    await click(thumbFor("ref_c"));
    expect(pickedId()).toBe("ref_c");

    // Switch to a different character whose references share none of the previous ids.
    await act(async () => {
      api.current.setSelectedCharacter({ id: "char_2", name: "Bly", description: "an artist" });
      api.current.setApprovedReferences(refs("ref_x", "ref_y"));
    });
    expect(pickedId()).toBe("ref_x");
  });
});

describe("CharacterGenerationPanel catalog axes (sc-15441)", () => {
  let container;
  let root;

  const guided = { ...MODEL, id: "guided-image" };
  const cfgFree = {
    ...MODEL,
    id: "cfg-free-image",
    image: { supportsGuidance: false, supportsNegativePrompt: false },
  };
  const noNegative = {
    ...MODEL,
    id: "no-negative-image",
    image: { supportsGuidance: true, supportsNegativePrompt: false },
  };
  const noGuidance = {
    ...MODEL,
    id: "no-guidance-image",
    image: { supportsGuidance: false, supportsNegativePrompt: true },
  };

  beforeEach(() => {
    global.IS_REACT_ACT_ENVIRONMENT = true;
    container = document.createElement("div");
    document.body.appendChild(container);
    root = createRoot(container);
  });

  afterEach(async () => {
    await act(async () => root.unmount());
    container.remove();
  });

  function labelStartingWith(text) {
    return [...container.querySelectorAll("label")].find((label) =>
      label.textContent.trim().startsWith(text),
    );
  }

  async function mount(createImageJob, models = [guided, cfgFree]) {
    await act(async () =>
      root.render(
        <CharacterGenerationPanel
          mode={{ ...TEST_MODE, defaultNegativePrompt: "curated baseline" }}
          selectedCharacter={character}
          model={guided}
          models={models}
          catalog={[]}
          approvedReferences={refs("ref_a")}
          assets={[]}
          createImageJob={createImageJob}
          importAsset={vi.fn()}
          addCharacterReference={vi.fn()}
          latestAssets={[]}
          imageLocalJobs={[]}
          loras={[]}
          rememberLocalGenerationJob={vi.fn()}
        />,
      ),
    );
    await act(async () =>
      container.querySelector(".advanced-section-toggle").dispatchEvent(
        new window.MouseEvent("click", { bubbles: true }),
      ),
    );
  }

  async function setControl(control, value) {
    await act(async () => {
      const prototype = control instanceof window.HTMLTextAreaElement
        ? window.HTMLTextAreaElement.prototype
        : window.HTMLInputElement.prototype;
      Object.getOwnPropertyDescriptor(prototype, "value").set.call(control, value);
      control.dispatchEvent(new window.Event("input", { bubbles: true }));
    });
  }

  async function selectModel(id) {
    const select = labelStartingWith("Backbone").querySelector("select");
    await act(async () => {
      Object.getOwnPropertyDescriptor(window.HTMLSelectElement.prototype, "value").set.call(select, id);
      select.dispatchEvent(new window.Event("change", { bubbles: true }));
    });
  }

  async function generate() {
    const button = [...container.querySelectorAll("button")].find(
      (candidate) => candidate.textContent.trim() === "Generate",
    );
    await act(async () => button.dispatchEvent(new window.MouseEvent("click", { bubbles: true })));
  }

  it("sends both values for an undeclared image model", async () => {
    const createImageJob = vi.fn(async () => ({ id: "job-1" }));
    await mount(createImageJob);
    await setControl(labelStartingWith("Guidance").querySelector("input"), "6.5");
    await setControl(labelStartingWith("Negative prompt").querySelector("textarea"), "blurry");
    await generate();

    const payload = createImageJob.mock.calls[0][0];
    expect(payload.model).toBe(guided.id);
    expect(payload.negativePrompt).toBe("blurry");
    expect(payload.advanced.guidanceScale).toBe(6.5);
  });

  it("suppresses non-empty retained values after switching to a CFG-free image model", async () => {
    const createImageJob = vi.fn(async () => ({ id: "job-1" }));
    await mount(createImageJob);
    await setControl(labelStartingWith("Guidance").querySelector("input"), "7.5");
    await setControl(
      labelStartingWith("Negative prompt").querySelector("textarea"),
      "blurry, low quality",
    );
    await selectModel(cfgFree.id);

    expect(labelStartingWith("Guidance")).toBeUndefined();
    expect(labelStartingWith("Negative prompt")).toBeUndefined();
    await generate();

    const payload = createImageJob.mock.calls[0][0];
    expect(payload.model).toBe(cfgFree.id);
    expect(payload.negativePrompt).toBe("");
    expect(payload.advanced).not.toHaveProperty("guidanceScale");
  });

  it("keeps Guidance while suppressing only Negative prompt after a mixed-axis switch", async () => {
    const createImageJob = vi.fn(async () => ({ id: "job-1" }));
    await mount(createImageJob, [guided, noNegative]);
    await setControl(labelStartingWith("Guidance").querySelector("input"), "7.5");
    await setControl(labelStartingWith("Negative prompt").querySelector("textarea"), "typed negative");
    await selectModel(noNegative.id);

    expect(labelStartingWith("Guidance")).toBeTruthy();
    expect(labelStartingWith("Negative prompt")).toBeUndefined();
    await generate();

    const payload = createImageJob.mock.calls[0][0];
    expect(payload.model).toBe(noNegative.id);
    expect(payload.negativePrompt).toBe("");
    expect(payload.advanced.guidanceScale).toBe(7.5);
  });

  it("keeps Negative prompt while suppressing only Guidance after a mixed-axis switch", async () => {
    const createImageJob = vi.fn(async () => ({ id: "job-1" }));
    await mount(createImageJob, [guided, noGuidance]);
    await setControl(labelStartingWith("Guidance").querySelector("input"), "7.5");
    await setControl(labelStartingWith("Negative prompt").querySelector("textarea"), "typed negative");
    await selectModel(noGuidance.id);

    expect(labelStartingWith("Guidance")).toBeUndefined();
    expect(labelStartingWith("Negative prompt")).toBeTruthy();
    await generate();

    const payload = createImageJob.mock.calls[0][0];
    expect(payload.model).toBe(noGuidance.id);
    expect(payload.negativePrompt).toBe("typed negative");
    expect(payload.advanced).not.toHaveProperty("guidanceScale");
  });
});
