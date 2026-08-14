import React, { act } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { SetupWizard } from "./SetupWizard.jsx";
import { mountRoot, unmountRoot } from "../testUtils/dom.js";

// sc-17227 BLOCKER 1, Setup Wizard surface.
//
// First-run onboarding lists every `isOfferable` model with a checkbox and a "Download N selected"
// button that fires `onDownloadModel` for each tick. It renders NO licence UI — no terms, no
// acknowledgment box — so before this change a brand-new user could bulk-queue MiniMax-H3 without
// ever being shown the MiniMax H3 Community License, which binds them personally (§II
// non-transferable, §I.9 "Licensee" is whoever uses the Works).
//
// That was survivable for every previously gated model because Hugging Face answered 401 without a
// saved credential, so the weights never landed. `MiniMaxAI/MiniMax-H3` is PUBLIC: nothing upstream
// refuses it, so the acknowledgment is the only gate.
//
// The fix is to not OFFER what the wizard cannot gate. Scoped to the standalone
// `requiresLicenseAcknowledgment` flag rather than to `gated`, so the credential-gated models that
// have been offered here for years are untouched — that scoping is asserted below, not assumed.

const ACK_MODEL = {
  id: "minimax_h3",
  name: "MiniMax-H3",
  type: "video",
  installState: "missing",
  downloadable: true,
  recommended: true,
  requiresLicenseAcknowledgment: true,
  licenseUrl: "https://huggingface.co/MiniMaxAI/MiniMax-H3",
  downloads: [{ provider: "huggingface", repo: "SceneWorks/minimax-h3-mlx", files: ["q4/*"] }],
};

const GATED_MODEL = {
  id: "flux1_dev",
  name: "FLUX.1 [dev]",
  type: "image",
  installState: "missing",
  downloadable: true,
  recommended: true,
  gated: true,
  credentialHost: "huggingface.co",
  downloads: [{ provider: "huggingface", repo: "black-forest-labs/FLUX.1-dev", files: ["*.safetensors"] }],
};

const PLAIN_MODEL = {
  id: "z_image",
  name: "Z-Image",
  type: "image",
  installState: "missing",
  downloadable: true,
  recommended: true,
  downloads: [{ provider: "huggingface", repo: "Tongyi-MAI/Z-Image-Turbo", files: ["*.safetensors"] }],
};

describe("SetupWizard license gate (sc-17227)", () => {
  let container;
  let root;
  let onDownloadModel;

  async function render(models) {
    await act(async () => {
      root.render(
        <SetupWizard
          jobs={[]}
          macCapabilities={{ macGatingActive: false }}
          models={models}
          onComplete={vi.fn()}
          onCreateProject={vi.fn()}
          onDownloadModel={onDownloadModel}
          onOpenQueue={vi.fn()}
        />,
      );
    });
  }

  function offeredNames() {
    return [...container.querySelectorAll(".setup-wizard-model-name")].map((node) =>
      node.firstChild?.textContent ?? node.textContent,
    );
  }

  async function downloadSelected() {
    const button = [...container.querySelectorAll(".setup-wizard-actions button")].find((node) =>
      node.textContent.startsWith("Download"),
    );
    expect(button, "the Download selected button").toBeTruthy();
    await act(async () => {
      button.dispatchEvent(new window.MouseEvent("click", { bubbles: true }));
    });
    return button;
  }

  beforeEach(() => {
    global.IS_REACT_ACT_ENVIRONMENT = true;
    window.localStorage.clear();
    onDownloadModel = vi.fn(async () => ({ id: "job-1" }));
    ({ container, root } = mountRoot());
  });

  afterEach(async () => {
    await unmountRoot(root, container);
    window.localStorage.clear();
    vi.clearAllMocks();
  });

  it("does not offer a model that requires a license acknowledgment", async () => {
    await render([ACK_MODEL, PLAIN_MODEL]);
    expect(offeredNames()).toEqual(["Z-Image"]);
    expect(container.textContent).not.toContain("MiniMax-H3");
    // It is `recommended`, so before this change it would have been PRE-TICKED and queued by the
    // wizard's own default selection without the user doing anything.
    const checkboxes = [...container.querySelectorAll(".setup-wizard-model input")];
    expect(checkboxes).toHaveLength(1);
    expect(checkboxes[0].checked).toBe(true);

    await downloadSelected();
    expect(onDownloadModel).toHaveBeenCalledTimes(1);
    expect(onDownloadModel).toHaveBeenCalledWith(expect.objectContaining({ id: "z_image" }));
    expect(onDownloadModel).not.toHaveBeenCalledWith(expect.objectContaining({ id: "minimax_h3" }));
  });

  it("still offers a credential-gated model, whose 401 backstop is unchanged", async () => {
    // The exclusion is deliberately NOT keyed on `gated`: FLUX.1 [dev] has been offered on this
    // screen for years and its download fails at Hugging Face without a saved token, so no weights
    // land unacknowledged. Widening the rule here would be an unrelated regression.
    await render([GATED_MODEL, PLAIN_MODEL]);
    expect(offeredNames().sort()).toEqual(["FLUX.1 [dev]", "Z-Image"]);
    await downloadSelected();
    expect(onDownloadModel).toHaveBeenCalledWith(expect.objectContaining({ id: "flux1_dev" }));
  });

  it("says the catalog is empty rather than listing an ungateable model alone", async () => {
    await render([ACK_MODEL]);
    expect(container.querySelector(".setup-wizard-empty")).toBeTruthy();
    const button = await downloadSelected();
    expect(button.disabled).toBe(true);
    expect(onDownloadModel).not.toHaveBeenCalled();
  });
});
