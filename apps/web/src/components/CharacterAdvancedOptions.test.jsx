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
function Harness({ model, catalog, baseWidth, baseHeight }) {
  const state = useCharacterAdvancedOptions(model, { catalog });
  React.useEffect(() => state.setOpen(true), []); // eslint-disable-line react-hooks/exhaustive-deps
  return (
    <>
      <CharacterAdvancedOptions state={state} baseWidth={baseWidth} baseHeight={baseHeight} />
      <output data-testid="adv">{JSON.stringify(state.buildAdvanced())}</output>
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
});
