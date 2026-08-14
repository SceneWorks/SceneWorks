// sc-17227 BLOCKER 1 — the licence acknowledgment enforced at the CHOKE POINT.
//
// The gate used to live only on the Models screen's card. Three other surfaces start a download
// without ever rendering it: the Simple UI's model manager, the first-run Setup Wizard, and the
// studio availability gates (`ModelAvailabilityGate`, whose offers come from `downloadOffersFor`,
// which falls back to ALL eligible models when no recommended one is eligible). All of them call
// `createModelDownloadJob`, so that is where the gate has to bind.
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

function Harness({ children }) {
  hookApi = useModelsAndLoras({
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
    await mount();
    await act(async () => {
      await hookApi.createModelDownloadJob(GATED_MODEL);
    });
    expect(downloadPosts()).toHaveLength(0);

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
