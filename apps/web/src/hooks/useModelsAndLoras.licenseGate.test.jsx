// sc-17227 BLOCKER 1 — the licence acknowledgment enforced at the CHOKE POINT.
//
// The gate used to live only on the Models screen's card. Three other surfaces start a download
// without ever rendering it: the Simple UI's model manager, the first-run Setup Wizard (which has
// since grown its own gate UI for the gated models it offers — SetupWizard.licenseGate.test.jsx
// drives that composition), and the studio availability gates (`ModelAvailabilityGate`, whose
// offers come from `downloadOffersFor`, which falls back to ALL eligible models when no
// recommended one is eligible). All of them call `createModelDownloadJob`, so that is where the
// gate has to bind.
//
// Why this became load-bearing with MiniMax-H3: every previously gated model was ALSO credential-
// gated, and Hugging Face answers 401 without a saved token, so an unacknowledged download failed
// no matter which button started it. `MiniMaxAI/MiniMax-H3` is a PUBLIC repo — the checkbox is the
// only gate, and a surface that skips it lands the weights.
//
// These tests drive the REAL hook (and, for the availability gate, the REAL component wired to the
// real hook), so what is asserted is whether a POST leaves the client — not whether a helper
// returned false.
import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const { apiFetchMock } = vi.hoisted(() => ({ apiFetchMock: vi.fn() }));
vi.mock("../api.js", () => ({
  apiFetch: (...args) => apiFetchMock(...args),
  isAbortError: () => false,
}));

import { ModelAvailabilityGate } from "../components/ModelAvailabilityGate.jsx";
import { downloadOffersFor } from "../modelEligibility.js";
import { useModelsAndLoras } from "./useModelsAndLoras.js";

const ACK_MODEL = {
  id: "minimax_h3",
  name: "MiniMax-H3",
  type: "video",
  installState: "missing",
  downloadable: true,
  requiresLicenseAcknowledgment: true,
};
const GATED_MODEL = {
  id: "flux1_dev",
  name: "FLUX.1 [dev]",
  type: "image",
  installState: "missing",
  downloadable: true,
  gated: true,
  credentialHost: "huggingface.co",
};
const PLAIN_MODEL = {
  id: "z_image",
  name: "Z-Image",
  type: "image",
  installState: "missing",
  downloadable: true,
};

let container;
let root;
let hookApi;
let errors;
let loraErrors;

function Harness({ children }) {
  hookApi = useModelsAndLoras({
    token: "tok",
    activeProject: { id: "proj-1" },
    activeProjectRef: { current: { id: "proj-1" } },
    setError: (value) => errors.push(value),
    setLoraError: (value) => loraErrors.push(value),
    setJobs: () => {},
    setActiveView: () => {},
    refreshData: async () => {},
    refreshDataWithLoraOverlay: async () => {},
  });
  return children ?? null;
}

/** Every `POST …/download` the hook actually issued. */
function downloadPosts() {
  return apiFetchMock.mock.calls.filter(([path]) => path.endsWith("/download"));
}

function acknowledge(modelId) {
  window.localStorage.setItem(`sceneworks-license-ack:${modelId}`, "true");
}

beforeEach(async () => {
  global.IS_REACT_ACT_ENVIRONMENT = true;
  apiFetchMock.mockReset();
  apiFetchMock.mockResolvedValue({ id: "job-1", type: "model_download" });
  window.localStorage.clear();
  errors = [];
  loraErrors = [];
  container = document.createElement("div");
  document.body.appendChild(container);
  root = createRoot(container);
});

afterEach(async () => {
  await act(async () => root.unmount());
  container.remove();
  window.localStorage.clear();
});

async function mount(children) {
  await act(async () => {
    root.render(<Harness>{children}</Harness>);
  });
}

describe("createModelDownloadJob license-acknowledgment choke point (sc-17227)", () => {
  it("refuses an unacknowledged download and issues no request at all", async () => {
    await mount();
    let job;
    await act(async () => {
      job = await hookApi.createModelDownloadJob(ACK_MODEL);
    });
    expect(job).toBeNull();
    expect(downloadPosts()).toHaveLength(0);
    // Assert WHICH refusal the user is shown, not merely that something was set: the message has
    // to name the screen that can take the acknowledgment, because the surfaces this fires on
    // (Simple UI, availability gate, workflow drop) have no licence UI of their own.
    expect(errors).toEqual([
      "MiniMax-H3 requires accepting its license first. Open Models and accept the license on the MiniMax-H3 card before downloading.",
    ]);
  });

  it("allows it once accepted, and asserts the acknowledgment to the API", async () => {
    acknowledge("minimax_h3");
    await mount();
    await act(async () => {
      await hookApi.createModelDownloadJob(ACK_MODEL, { variant: "q4" });
    });
    const posts = downloadPosts();
    expect(posts).toHaveLength(1);
    expect(posts[0][0]).toBe("/api/v1/models/minimax_h3/download");
    // The API refuses this same request without the flag, so the client must send it. Assert the
    // BODY, not just that a POST happened — omitting the field would 403 server-side.
    expect(JSON.parse(posts[0][2].body)).toEqual({
      requestedGpu: "auto",
      variant: "q4",
      licenseAcknowledged: true,
    });
    expect(errors).toEqual([""]);
  });

  it("covers a credential-gated model too, and leaves an unlicensed model's body unchanged", async () => {
    // `gated` implies the acknowledgment, so the refusal fires HERE, client-side, before any
    // request exists — there is no "unchanged 401 backstop" in this path (the account the Setup
    // Wizard test used to carry): Hugging Face is never asked. Every surface that offers a gated
    // model therefore has to be able to take the acknowledgment (the Models screen's card, and
    // now the wizard's own gate) or the refusal below is what its user hits.
    await mount();
    await act(async () => {
      await hookApi.createModelDownloadJob(GATED_MODEL);
    });
    expect(downloadPosts()).toHaveLength(0);
    // Refused BY NAME, pointing at a surface that can take the acknowledgment.
    expect(errors.at(-1)).toBe(
      "FLUX.1 [dev] requires accepting its license first. Open Models and accept the license on the FLUX.1 [dev] card before downloading.",
    );

    // A model with no licence requirement must not gain a gate or a body field — the flag is sent
    // only where it is required, so no other download's request shape changes.
    await act(async () => {
      await hookApi.createModelDownloadJob(PLAIN_MODEL);
    });
    const posts = downloadPosts();
    expect(posts).toHaveLength(1);
    expect(posts[0][0]).toBe("/api/v1/models/z_image/download");
    expect(JSON.parse(posts[0][2].body)).toEqual({ requestedGpu: "auto" });
  });

  // The availability-gate surface. `downloadOffersFor` falls back to every eligible model when no
  // recommended one is eligible, so an acknowledgment model reaches the offer list and its
  // Download button is enabled — the component itself has no licence concept and should not grow
  // one. Driving the REAL component through the REAL hook is what proves the choke point covers it.
  it("blocks a ModelAvailabilityGate offer, then lets it through once accepted", async () => {
    const offers = downloadOffersFor([ACK_MODEL], (model) => model.type === "video", {});
    expect(offers.map((model) => model.id)).toEqual(["minimax_h3"]);

    function GateHarness() {
      return (
        <ModelAvailabilityGate
          jobs={[]}
          onDownload={hookApi?.createModelDownloadJob}
          offers={offers}
          title="Video Studio needs a model"
        />
      );
    }
    await mount(<GateHarness />);
    // Second render so the harness's child sees the hook api from the first pass.
    await mount(<GateHarness />);

    const button = [...container.querySelectorAll(".model-availability-offer button")][0];
    expect(button, "the offer's Download button").toBeTruthy();
    expect(button.disabled).toBe(false);
    await act(async () => {
      button.dispatchEvent(new window.MouseEvent("click", { bubbles: true }));
    });
    expect(downloadPosts()).toHaveLength(0);
    expect(errors.at(-1)).toContain("requires accepting its license first");

    acknowledge("minimax_h3");
    await act(async () => {
      button.dispatchEvent(new window.MouseEvent("click", { bubbles: true }));
    });
    expect(downloadPosts()).toHaveLength(1);
    expect(JSON.parse(downloadPosts()[0][2].body).licenseAcknowledged).toBe(true);
  });
});

// The LoRA half of the same choke point (sc-17227 review MAJOR 1). `POST /api/v1/loras/:id/download`
// gates on the repo the catalog row resolves to, so a catalog LoRA whose `source.repo` is a
// restricted repo is refused server-side exactly as a model download is. This hook is the only path
// every LoRA-download surface takes (the Models screen, the Simple UI's model manager, the studio
// LoRA pickers' Update buttons, the editor's on-demand fetch, and the dropped-workflow panel's
// missing-requirement install), so it is where the gate binds — and without it the refusal is a bare
// 403 with no checkbox anywhere in the product to clear it.
//
// Binding here is necessary but not sufficient: the gate reads the row it is HANDED, so a caller
// that hands it an `{ id }` stub disables it just as surely as no gate at all. The dropped-workflow
// panel did exactly that until sc-17227 review MAJOR 1; `useWorkflowDrop.test.jsx` pins that caller
// resolving both kinds of row to their real catalog entry first.
//
// A LoRA row cannot declare the requirement itself: the server stamps `licenseAcknowledgmentModelId`
// / `…ModelName` onto the row from the UNFILTERED model manifest, naming the model whose card carries
// the licence text and the acknowledgment.
const GATED_LORA = {
  id: "restricted_lora",
  name: "Restricted LoRA",
  source: { provider: "huggingface", repo: "MiniMaxAI/MiniMax-H3" },
  licenseAcknowledgmentModelId: "minimax_h3",
  licenseAcknowledgmentModelName: "MiniMax-H3",
};
const PLAIN_LORA = { id: "plain_lora", name: "Plain LoRA" };

describe("createLoraDownloadJob license-acknowledgment choke point (sc-17227)", () => {
  it("refuses an unacknowledged gated LoRA and issues no request at all", async () => {
    await mount();
    let job;
    await act(async () => {
      job = await hookApi.createLoraDownloadJob(GATED_LORA);
    });
    expect(job).toBeNull();
    expect(downloadPosts()).toHaveLength(0);
    // The refusal must name the MODEL card that can take the acknowledgment — a message naming the
    // LoRA would send the user to a row that renders no licence UI, which is the dead end this
    // whole gate exists to avoid.
    expect(loraErrors).toEqual([
      "MiniMax-H3 requires accepting its license first. Open Models and accept the license on the MiniMax-H3 card before downloading.",
    ]);
  });

  it("lets it through once the gating model's licence is accepted, and asserts it to the API", async () => {
    acknowledge("minimax_h3");
    await mount();
    await act(async () => {
      await hookApi.createLoraDownloadJob(GATED_LORA);
    });
    const posts = downloadPosts();
    expect(posts).toHaveLength(1);
    expect(posts[0][0]).toBe("/api/v1/loras/restricted_lora/download");
    // Assert the BODY: the server refuses this exact request without the field, so a POST that
    // merely happened would still be a 403 the user cannot clear.
    expect(JSON.parse(posts[0][2].body)).toEqual({
      requestedGpu: "auto",
      licenseAcknowledged: true,
    });
  });

  it("leaves an ordinary LoRA's gate and request body untouched", async () => {
    await mount();
    await act(async () => {
      await hookApi.createLoraDownloadJob(PLAIN_LORA);
    });
    const posts = downloadPosts();
    expect(posts).toHaveLength(1);
    expect(posts[0][0]).toBe("/api/v1/loras/plain_lora/download");
    expect(JSON.parse(posts[0][2].body)).toEqual({ requestedGpu: "auto" });
    expect(loraErrors).toEqual([""]);
  });

  // Accepting a DIFFERENT model's licence must not open this one: the ack is read under the id the
  // server stamped, not "some acknowledgment exists".
  it("is not cleared by an unrelated model's acknowledgment", async () => {
    acknowledge("z_image");
    await mount();
    await act(async () => {
      await hookApi.createLoraDownloadJob(GATED_LORA);
    });
    expect(downloadPosts()).toHaveLength(0);
    expect(loraErrors.at(-1)).toContain("requires accepting its license first");
  });
});
