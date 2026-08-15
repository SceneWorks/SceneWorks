import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it } from "vitest";

import {
  CharacterAdvancedOptions,
  useCharacterAdvancedOptions,
} from "./CharacterAdvancedOptions.jsx";

// The Angles/Poses PiD toggle (epic 7840, sc-8372): it mirrors the Image Studio checkbox and is
// gated on `pidToggleVisible(model, catalog)` — shown only when the active model declares a PiD
// backbone (`ui.pid.checkpointId`) AND that checkpoint is installed in the catalog. When shown and
// checked it folds `usePid: true` into the advanced payload; otherwise the key is absent.

const instantId = { id: "instantid_realvisxl", ui: { pid: { checkpointId: "pid_sdxl" } } };
const nonEligible = { id: "kolors", ui: {} };
const installed = [{ id: "pid_sdxl", installState: "installed" }];
const missing = [{ id: "pid_sdxl", installState: "missing" }];

// Hosts the hook + presentational panel, opens the advanced section, and renders the current
// `buildAdvanced()` output so the test can assert what the job payload would carry.
function Harness({ model, catalog, baseWidth, baseHeight, defaultNegativePrompt = "" }) {
  const state = useCharacterAdvancedOptions(model, { catalog, defaultNegativePrompt });
  React.useEffect(() => state.setOpen(true), []); // eslint-disable-line react-hooks/exhaustive-deps
  return (
    <>
      <CharacterAdvancedOptions state={state} baseWidth={baseWidth} baseHeight={baseHeight} />
      <output data-testid="adv">{JSON.stringify(state.buildAdvanced())}</output>
      <output data-testid="negative">
        {state.supportsNegativePrompt ? state.negativePrompt : ""}
      </output>
      <output data-testid="raw-negative">{state.negativePrompt}</output>
    </>
  );
}

describe("CharacterAdvancedOptions PiD toggle (sc-8372)", () => {
  let container;
  let root;

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

  const pidCheckbox = () => container.querySelector(".pid-decoder-toggle input[type=checkbox]");
  const pidHint = () => container.querySelector(".pid-decode-hint");
  const pidTargetSelect = () => container.querySelector(".pid-target-select select");
  const advanced = () => JSON.parse(container.querySelector('[data-testid="adv"]').textContent);

  it("hides the toggle when the model has no PiD backbone", async () => {
    await act(async () => root.render(<Harness model={nonEligible} catalog={installed} />));
    expect(pidCheckbox()).toBe(null);
    expect(advanced().usePid).toBeUndefined();
  });

  it("hides the toggle when the PiD checkpoint is not installed", async () => {
    await act(async () => root.render(<Harness model={instantId} catalog={missing} />));
    expect(pidCheckbox()).toBe(null);
    expect(advanced().usePid).toBeUndefined();
  });

  it("shows the toggle and emits advanced.usePid when eligible, installed, and checked", async () => {
    await act(async () => root.render(<Harness model={instantId} catalog={installed} />));
    const box = pidCheckbox();
    expect(box).not.toBe(null);
    // Off by default → no usePid in the payload.
    expect(advanced().usePid).toBeUndefined();
    // Checking it folds usePid:true in.
    await act(async () => box.click());
    expect(advanced().usePid).toBe(true);
  });

  // sc-10144: the high-res decode heads-up. Character panels render at a fixed 1024² base, which the
  // default 4K PiD tier super-resolves 4× to 4096² — a multi-minute decode we warn about so it never
  // reads as hung. The fast 2K tier (~2048² output) never warns.
  it("shows the multi-minute heads-up at the default 4K tier for the 1024² panel base", async () => {
    await act(async () =>
      root.render(<Harness model={instantId} catalog={installed} baseWidth={1024} baseHeight={1024} />),
    );
    // No hint until PiD is on.
    expect(pidHint()).toBe(null);
    await act(async () => pidCheckbox().click());
    const hint = pidHint();
    expect(hint).not.toBe(null);
    expect(hint.textContent).toContain("4096×4096");
    expect(hint.textContent).toContain("not stuck");
  });

  it("drops the heads-up on the fast 2K tier", async () => {
    await act(async () =>
      root.render(<Harness model={instantId} catalog={installed} baseWidth={1024} baseHeight={1024} />),
    );
    await act(async () => pidCheckbox().click());
    expect(pidHint()).not.toBe(null);
    // Switch the output tier to 2K → the decode caps to ~2048², fast → no heads-up.
    await act(async () => {
      const select = pidTargetSelect();
      select.value = "2k";
      select.dispatchEvent(new Event("change", { bubbles: true }));
    });
    expect(pidHint()).toBe(null);
  });

  it("keeps Guidance and Negative prompt for an image model with no axis declarations", async () => {
    await act(async () =>
      root.render(<Harness model={{ id: "guided" }} catalog={[]} />),
    );
    expect(container.textContent).toContain("Guidance");
    expect(container.textContent).toContain("Negative prompt");
  });

  it("hides and suppresses retained Guidance and Negative prompt after a CFG-free model switch", async () => {
    const guided = { id: "guided" };
    const cfgFree = {
      id: "cfg-free",
      image: { supportsGuidance: false, supportsNegativePrompt: false },
    };
    await act(async () =>
      root.render(<Harness model={guided} catalog={[]} defaultNegativePrompt="curated baseline" />),
    );

    const guidance = [...container.querySelectorAll("label")].find((label) =>
      label.textContent.trim().startsWith("Guidance"),
    )?.querySelector("input");
    const negative = [...container.querySelectorAll("label")].find((label) =>
      label.textContent.trim().startsWith("Negative prompt"),
    )?.querySelector("textarea");
    await act(async () => {
      const inputSetter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, "value").set;
      inputSetter.call(guidance, "7.5");
      guidance.dispatchEvent(new window.Event("input", { bubbles: true }));
      const textareaSetter = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, "value").set;
      textareaSetter.call(negative, "blurry, low quality");
      negative.dispatchEvent(new window.Event("input", { bubbles: true }));
    });
    expect(advanced().guidanceScale).toBe(7.5);

    await act(async () =>
      root.render(<Harness model={cfgFree} catalog={[]} defaultNegativePrompt="curated baseline" />),
    );
    expect(container.textContent).not.toContain("Guidance");
    expect(container.textContent).not.toContain("Negative prompt");
    expect(advanced()).not.toHaveProperty("guidanceScale");
    expect(container.querySelector('[data-testid="negative"]').textContent).toBe("");
  });

  it("does not seed a hidden negative prompt on a CFG-free model", async () => {
    const cfgFree = {
      id: "cfg-free",
      image: { supportsGuidance: false, supportsNegativePrompt: false },
    };
    await act(async () =>
      root.render(<Harness model={cfgFree} catalog={[]} defaultNegativePrompt="curated baseline" />),
    );
    expect(container.querySelector('[data-testid="raw-negative"]').textContent).toBe("");
    await act(async () =>
      root.render(<Harness model={cfgFree} catalog={[]} defaultNegativePrompt="updated baseline" />),
    );
    expect(container.querySelector('[data-testid="raw-negative"]').textContent).toBe("");
    expect(container.querySelector('[data-testid="negative"]').textContent).toBe("");
    expect(container.textContent).not.toContain("Negative prompt");
  });

  it("keeps Guidance while independently hiding and suppressing Negative prompt", async () => {
    const guided = { id: "guided" };
    const noNegative = {
      id: "no-negative",
      image: { supportsGuidance: true, supportsNegativePrompt: false },
    };
    await act(async () =>
      root.render(<Harness model={guided} catalog={[]} defaultNegativePrompt="baseline" />),
    );
    const guidance = [...container.querySelectorAll("label")].find((label) =>
      label.textContent.trim().startsWith("Guidance"),
    )?.querySelector("input");
    await act(async () => {
      Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, "value").set.call(guidance, "6.5");
      guidance.dispatchEvent(new window.Event("input", { bubbles: true }));
    });

    await act(async () =>
      root.render(<Harness model={noNegative} catalog={[]} defaultNegativePrompt="hidden baseline" />),
    );
    expect(container.textContent).toContain("Guidance");
    expect(container.textContent).not.toContain("Negative prompt");
    expect(advanced().guidanceScale).toBe(6.5);
    expect(container.querySelector('[data-testid="negative"]').textContent).toBe("");
    expect(container.querySelector('[data-testid="raw-negative"]').textContent).toBe("baseline");
  });

  it("keeps Negative prompt while independently hiding and suppressing Guidance", async () => {
    const guided = { id: "guided" };
    const noGuidance = {
      id: "no-guidance",
      image: { supportsGuidance: false, supportsNegativePrompt: true },
    };
    await act(async () =>
      root.render(<Harness model={guided} catalog={[]} defaultNegativePrompt="baseline" />),
    );
    const guidance = [...container.querySelectorAll("label")].find((label) =>
      label.textContent.trim().startsWith("Guidance"),
    )?.querySelector("input");
    await act(async () => {
      Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, "value").set.call(guidance, "6.5");
      guidance.dispatchEvent(new window.Event("input", { bubbles: true }));
    });

    await act(async () =>
      root.render(<Harness model={noGuidance} catalog={[]} defaultNegativePrompt="visible baseline" />),
    );
    expect(container.textContent).not.toContain("Guidance");
    expect(container.textContent).toContain("Negative prompt");
    expect(advanced()).not.toHaveProperty("guidanceScale");
    expect(container.querySelector('[data-testid="raw-negative"]').textContent).toBe("visible baseline");
  });

  it("emits manifest variationStrength as trueCfgScale and suppresses the inert IP scale", async () => {
    const qwenEdit = {
      id: "qwen_image_edit_2511",
      ui: {
        hideReferenceStrength: true,
        variationStrength: {
          label: "Prompt strength",
          default: 4,
          min: 1,
          max: 10,
          step: 0.5,
        },
      },
    };
    await act(async () => root.render(<Harness model={qwenEdit} catalog={[]} />));

    expect(container.textContent).not.toContain("Reference strength");
    expect(container.textContent).toContain("Prompt strength");
    expect(advanced()).toMatchObject({ trueCfgScale: 4 });
    expect(advanced()).not.toHaveProperty("ipAdapterScale");

    const slider = container.querySelector(".variation-strength input");
    await act(async () => {
      Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, "value").set.call(slider, "6");
      slider.dispatchEvent(new window.Event("input", { bubbles: true }));
    });
    expect(advanced().trueCfgScale).toBe(6);
  });

  it("resets trueCfgScale to the next model's declared default", async () => {
    const qwenEdit = {
      id: "qwen_image_edit_2511",
      ui: {
        hideReferenceStrength: true,
        variationStrength: { label: "Prompt strength", default: 4, min: 1, max: 10, step: 0.5 },
      },
    };
    const senseNova = {
      id: "sensenova_u1_8b",
      ui: {
        hideReferenceStrength: true,
        variationStrength: { label: "Reference strength", default: 1.5, min: 1, max: 4, step: 0.1 },
      },
    };
    await act(async () => root.render(<Harness model={qwenEdit} catalog={[]} />));
    const slider = container.querySelector(".variation-strength input");
    await act(async () => {
      Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, "value").set.call(slider, "7");
      slider.dispatchEvent(new window.Event("input", { bubbles: true }));
    });
    expect(advanced().trueCfgScale).toBe(7);

    await act(async () => root.render(<Harness model={senseNova} catalog={[]} />));
    expect(advanced()).toMatchObject({ trueCfgScale: 1.5 });
    expect(advanced()).not.toHaveProperty("ipAdapterScale");
  });
});
