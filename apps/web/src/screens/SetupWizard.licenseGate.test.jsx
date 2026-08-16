import React, { act } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const { apiFetchMock } = vi.hoisted(() => ({ apiFetchMock: vi.fn() }));
vi.mock("../api.js", () => ({
  apiFetch: (...args) => apiFetchMock(...args),
  isAbortError: () => false,
}));

import { useModelsAndLoras } from "../hooks/useModelsAndLoras.js";
import { SetupWizard } from "./SetupWizard.jsx";
import { mountRoot, unmountRoot } from "../testUtils/dom.js";

// sc-17227 BLOCKER 1, Setup Wizard surface — re-grounded by the sc-17137 review.
//
// First-run onboarding lists every `isOfferable` model with a checkbox and a "Download N selected"
// button. Two rules compose here, and these tests drive the REAL composition (the wizard wired to
// the real `useModelsAndLoras` hook, asserting which POSTs leave the client) rather than a mocked
// `onDownloadModel`, because the two halves used to be tested against contradictory accounts:
//
//   * A model whose only gate is the standalone `requiresLicenseAcknowledgment` flag is NOT
//     offered at all (`offerableWithoutLicenseUi`): its repo is public, the acknowledgment is the
//     only thing between a bulk-queued first run and weights whose licence binds the user
//     personally (MiniMax H3 Community License §II / §I.9), and the Models screen shows the full
//     terms.
//   * A credential-`gated` model IS offered — but `gated` implies the acknowledgment
//     (`requiresLicenseAcknowledgment`), and `createModelDownloadJob` refuses ANY unacknowledged
//     licence-bearing download CLIENT-SIDE, before a request exists. The old account here ("the
//     401 backstop is unchanged") was false: Hugging Face is never asked. So the wizard renders
//     the licence gate on gated rows — notice, links, acknowledgment checkbox — and a model left
//     unacknowledged is refused BY NAME instead of being marked "Download started".

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
  licenseNotice: "FLUX.1 [dev] Non-Commercial License: outputs may not be used commercially.",
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

describe("SetupWizard license gate (sc-17227 / sc-17137 review)", () => {
  let container;
  let root;
  let errors;

  // The wizard wired to the REAL choke point: `onDownloadModel` is the hook's own
  // `createModelDownloadJob`, so what these tests observe is whether a POST leaves the client.
  function ComposedWizard({ models }) {
    const api = useModelsAndLoras({
      token: "tok",
      activeProject: { id: "proj-1" },
      activeProjectRef: { current: { id: "proj-1" } },
      setError: (value) => errors.push(value),
      setLoraError: () => {},
      setJobs: () => {},
      setActiveView: () => {},
      refreshData: async () => {},
      refreshDataWithLoraOverlay: async () => {},
    });
    return (
      <SetupWizard
        jobs={[]}
        macCapabilities={{ macGatingActive: false }}
        models={models}
        onComplete={vi.fn()}
        onCreateProject={vi.fn()}
        onDownloadModel={api.createModelDownloadJob}
        onOpenQueue={vi.fn()}
      />
    );
  }

  async function render(models) {
    await act(async () => {
      root.render(<ComposedWizard models={models} />);
    });
  }

  function offeredNames() {
    return [...container.querySelectorAll(".setup-wizard-model-name")].map((node) =>
      node.firstChild?.textContent ?? node.textContent,
    );
  }

  function rowMeta(name) {
    const row = [...container.querySelectorAll(".setup-wizard-model")].find((node) =>
      node.textContent.includes(name),
    );
    expect(row, `the ${name} row`).toBeTruthy();
    return row.querySelector(".setup-wizard-model-meta").textContent;
  }

  /** Every `POST …/download` that actually left the client. */
  function downloadPosts() {
    return apiFetchMock.mock.calls.filter(([path]) => path.endsWith("/download"));
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

  async function tickLicenseAck() {
    const checkbox = container.querySelector(".setup-wizard-license .model-license-ack input");
    expect(checkbox, "the wizard's license acknowledgment checkbox").toBeTruthy();
    await act(async () => {
      // Drive the checkbox through React's value tracker (a bare `.checked =` assignment is
      // pre-recorded by React and the onChange is skipped).
      Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, "checked").set.call(
        checkbox,
        true,
      );
      checkbox.dispatchEvent(new window.Event("click", { bubbles: true }));
    });
    return checkbox;
  }

  beforeEach(() => {
    global.IS_REACT_ACT_ENVIRONMENT = true;
    window.localStorage.clear();
    apiFetchMock.mockReset();
    apiFetchMock.mockResolvedValue({ id: "job-1", type: "model_download" });
    errors = [];
    ({ container, root } = mountRoot());
  });

  afterEach(async () => {
    await unmountRoot(root, container);
    window.localStorage.clear();
    vi.clearAllMocks();
  });

  it("does not offer a model that requires a standalone license acknowledgment", async () => {
    await render([ACK_MODEL, PLAIN_MODEL]);
    expect(offeredNames()).toEqual(["Z-Image"]);
    expect(container.textContent).not.toContain("MiniMax-H3");
    // It is `recommended`, so before sc-17227 it would have been PRE-TICKED and queued by the
    // wizard's own default selection without the user doing anything.
    const checkboxes = [...container.querySelectorAll(".setup-wizard-model input")];
    expect(checkboxes).toHaveLength(1);
    expect(checkboxes[0].checked).toBe(true);

    await downloadSelected();
    expect(downloadPosts().map(([path]) => path)).toEqual(["/api/v1/models/z_image/download"]);
  });

  it("refuses an unacknowledged gated model BY NAME — no request, no 'Download started'", async () => {
    await render([GATED_MODEL, PLAIN_MODEL]);
    // Offered, as it always was, and now with the licence gate rendered on its row.
    expect(offeredNames().sort()).toEqual(["FLUX.1 [dev]", "Z-Image"]);
    const gate = container.querySelector(".setup-wizard-license");
    expect(gate, "the wizard's licence gate block").toBeTruthy();
    expect(gate.textContent).toContain("Gated download");
    // The manifest's statement of what is being accepted (sc-17227) rides along.
    expect(gate.textContent).toContain("FLUX.1 [dev] Non-Commercial License");

    await downloadSelected();
    // Only the plain model's POST leaves the client. The gated one is refused BEFORE any request —
    // there is no Hugging Face 401 backstop in this path, which is exactly why the wizard must
    // handle the refusal itself.
    expect(downloadPosts().map(([path]) => path)).toEqual(["/api/v1/models/z_image/download"]);
    // The refused row is named, and is NOT claimed started.
    const refusal = container.querySelector(".setup-wizard-refusals");
    expect(refusal, "the refusal notice").toBeTruthy();
    expect(refusal.textContent).toContain(
      "FLUX.1 [dev] was not downloaded — accept its license above first.",
    );
    expect(rowMeta("FLUX.1 [dev]")).not.toBe("Download started");
    expect(rowMeta("Z-Image")).toBe("Download started");
    // The row stays pending, so the user can accept and retry without re-ticking anything.
    const retry = [...container.querySelectorAll(".setup-wizard-actions button")].find((node) =>
      node.textContent.startsWith("Download"),
    );
    expect(retry.disabled).toBe(false);
  });

  it("downloads a gated model once its license is accepted in the wizard, asserting the ack to the API", async () => {
    await render([GATED_MODEL]);
    const checkbox = await tickLicenseAck();
    expect(checkbox.checked).toBe(true);
    // The acknowledgment persists to the SAME store the choke point reads, so it also unblocks
    // every other surface (and survives into the Models screen's own checkbox).
    expect(window.localStorage.getItem("sceneworks-license-ack:flux1_dev")).toBe("true");

    await downloadSelected();
    const posts = downloadPosts();
    expect(posts).toHaveLength(1);
    expect(posts[0][0]).toBe("/api/v1/models/flux1_dev/download");
    // The API refuses this same request without the flag, so assert the BODY, not just the POST.
    expect(JSON.parse(posts[0][2].body)).toEqual({
      requestedGpu: "auto",
      licenseAcknowledged: true,
    });
    expect(rowMeta("FLUX.1 [dev]")).toBe("Download started");
    expect(container.querySelector(".setup-wizard-refusals")).toBeNull();
  });

  it("seeds a prior acknowledgment from the store, so a returning user is not re-asked", async () => {
    window.localStorage.setItem("sceneworks-license-ack:flux1_dev", "true");
    await render([GATED_MODEL]);
    const checkbox = container.querySelector(".setup-wizard-license .model-license-ack input");
    expect(checkbox.checked).toBe(true);
    await downloadSelected();
    expect(downloadPosts()).toHaveLength(1);
    expect(rowMeta("FLUX.1 [dev]")).toBe("Download started");
  });

  it("names a row whose download the API refused instead of claiming it started", async () => {
    apiFetchMock.mockRejectedValueOnce(new Error("disk full"));
    await render([PLAIN_MODEL]);
    await downloadSelected();
    // The hook returned null (and set the app-level error, which this overlay hides), so the
    // wizard must say itself that the row did not start.
    expect(errors).toContain("disk full");
    expect(rowMeta("Z-Image")).not.toBe("Download started");
    expect(container.querySelector(".setup-wizard-refusals").textContent).toContain(
      "Z-Image download did not start.",
    );
  });

  it("says the catalog is empty rather than listing an ungateable model alone", async () => {
    await render([ACK_MODEL]);
    expect(container.querySelector(".setup-wizard-empty")).toBeTruthy();
    const button = await downloadSelected();
    expect(button.disabled).toBe(true);
    expect(downloadPosts()).toHaveLength(0);
  });
});
