import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { FakeEventSource, response, settle } from "./main.testSupport.jsx";
import { click } from "./testUtils/dom.js";
import { resetStartupTimingForTests } from "./startupTiming.js";
import { resetAccessTokenForTests } from "./accessToken.js";

// Capture the pre-submit hook from the one image-job creator shared by every image studio. That
// proves the request is held before its POST without coupling this suite to a particular form.
const hoisted = vi.hoisted(() => ({ beforeCreate: null }));

vi.mock("./createJob.js", async (importOriginal) => {
  const actual = await importOriginal();
  return {
    ...actual,
    makeCreateJob(options) {
      if (options.definition === actual.CREATE_JOB_DEFINITIONS.image) {
        hoisted.beforeCreate = options.beforeCreate;
      }
      return actual.makeCreateJob(options);
    },
  };
});

const { App } = await import("./main.jsx");

const MODAL_TITLE = "Include the recipe in generated images?";

function fakeApi(storedPreferences, failurePlan = { workflowPuts: 0 }) {
  return vi.fn((url, init) => {
    const path = new URL(url).pathname;
    if (path.endsWith("/health")) {
      return Promise.resolve(response({ status: "ok", authRequired: false }));
    }
    if (path.endsWith("/access")) {
      return Promise.resolve(response({ authRequired: false }));
    }
    if (path.endsWith("/jobs/events/ticket")) {
      return Promise.resolve(response({ ticket: "stream-ticket" }));
    }
    if (path.endsWith("/projects")) {
      return Promise.resolve(response([{ id: "project-default", name: "Default Project" }]));
    }
    if (path.endsWith("/ui-preferences")) {
      if (init?.method === "PUT") {
        const patch = JSON.parse(init.body);
        if (
          Object.hasOwn(patch, "workflowEmbedNoticeSeen") &&
          failurePlan.workflowPuts > 0
        ) {
          failurePlan.workflowPuts -= 1;
          return Promise.reject(new Error("offline"));
        }
        return Promise.resolve(response({ ...storedPreferences, ...patch }));
      }
      if (failurePlan.preferencesGet) {
        return failurePlan.preferencesGet.then((preferences) => response(preferences));
      }
      return Promise.resolve(response(storedPreferences));
    }
    return Promise.resolve(response([]));
  });
}

function preferenceWrites() {
  return global.fetch.mock.calls
    .filter(
      ([url, init]) =>
        new URL(url).pathname.endsWith("/ui-preferences") && init?.method === "PUT",
    )
    .map(([, init]) => JSON.parse(init.body));
}

const workflowDecisionWrites = () =>
  preferenceWrites().filter(
    (patch) =>
      Object.hasOwn(patch, "embedWorkflowInImages") ||
      Object.hasOwn(patch, "workflowEmbedNoticeSeen"),
  );

const modalIsShowing = () => document.body.textContent.includes(MODAL_TITLE);
const buttonNamed = (name) =>
  [...document.body.querySelectorAll("button")].find(
    (button) => button.textContent.trim() === name,
  );

describe("App - the one-time embedded-workflow choice (sc-16390)", () => {
  let container;
  let root;

  beforeEach(() => {
    global.IS_REACT_ACT_ENVIRONMENT = true;
    container = document.createElement("div");
    document.body.appendChild(container);
    FakeEventSource.instances = [];
    window.EventSource = FakeEventSource;
    window.localStorage.clear();
    resetAccessTokenForTests();
    window.innerWidth = 1440;
    hoisted.beforeCreate = null;
  });

  afterEach(async () => {
    await act(async () => root?.unmount());
    container.remove();
    resetStartupTimingForTests();
    window.localStorage.clear();
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  async function renderApp(storedPreferences = {}, failurePlan) {
    global.fetch = fakeApi(storedPreferences, failurePlan);
    root = createRoot(container);
    await act(async () => {
      root.render(<App />);
    });
    await settle();
  }

  async function startGeneration() {
    expect(hoisted.beforeCreate, "App must provide the image pre-submit hook").toBeTypeOf(
      "function",
    );
    let gate;
    await act(async () => {
      gate = hoisted.beforeCreate({ prompt: "roses" });
    });
    await settle();
    return { gate };
  }

  it("does not interrupt the app before a generation is attempted", async () => {
    await renderApp();
    expect(modalIsShowing()).toBe(false);
  });

  it("holds the first generation until Ok atomically enables embedding and records the choice", async () => {
    await renderApp();
    const { gate } = await startGeneration();
    expect(modalIsShowing()).toBe(true);
    expect(workflowDecisionWrites()).toEqual([]);
    expect(
      [...document.body.querySelector("[role=dialog]").querySelectorAll("button")].map(
        (button) => button.textContent.trim(),
      ),
    ).toEqual(["Go to settings", "No thanks!", "Ok, got it"]);
    const dialog = document.body.querySelector("[role=dialog]");
    const description = document.getElementById(dialog.getAttribute("aria-describedby"));
    expect(description?.textContent).toContain("model, and LoRA details");

    await click(buttonNamed("Ok, got it"));
    await settle();

    await expect(gate).resolves.toBe(true);
    expect(modalIsShowing()).toBe(false);
    expect(preferenceWrites()).toContainEqual({
      embedWorkflowInImages: true,
      workflowEmbedNoticeSeen: true,
    });
    const second = await startGeneration();
    await expect(second.gate).resolves.toBe(true);
    expect(modalIsShowing()).toBe(false);
  });

  it("persists No thanks atomically and resumes the held generation with embedding off", async () => {
    await renderApp();
    const { gate } = await startGeneration();

    await click(buttonNamed("No thanks!"));
    await settle();

    await expect(gate).resolves.toBe(true);
    expect(preferenceWrites()).toContainEqual({
      embedWorkflowInImages: false,
      workflowEmbedNoticeSeen: true,
    });
  });

  it("opens Sharing and cancels the held request without recording a decision", async () => {
    await renderApp();
    const { gate } = await startGeneration();

    await click(buttonNamed("Go to settings"));
    await settle();

    await expect(gate).resolves.toBe(false);
    expect(workflowDecisionWrites()).toEqual([]);
    expect(document.body.textContent).toContain("Include the recipe in generated images");
    expect(document.activeElement?.textContent).toBe("Sharing");

    const { gate: nextGate } = await startGeneration();
    expect(modalIsShowing()).toBe(true);
    await click(buttonNamed("No thanks!"));
    await expect(nextGate).resolves.toBe(true);
  });

  it("never asks again when the durable decision marker is already present", async () => {
    await renderApp({ workflowEmbedNoticeSeen: true });
    const { gate } = await startGeneration();
    await expect(gate).resolves.toBe(true);
    expect(modalIsShowing()).toBe(false);
  });

  it("waits for the durable preference GET before deciding whether to show the modal", async () => {
    let releasePreferences;
    const preferencesGet = new Promise((resolve) => {
      releasePreferences = resolve;
    });
    await renderApp({}, { preferencesGet, workflowPuts: 0 });

    const { gate } = await startGeneration();
    expect(modalIsShowing()).toBe(false);

    await act(async () => {
      releasePreferences({ embedWorkflowInImages: false, workflowEmbedNoticeSeen: true });
    });
    await settle();

    await expect(gate).resolves.toBe(true);
    expect(modalIsShowing()).toBe(false);
    expect(workflowDecisionWrites()).toEqual([]);
  });

  it("asks an upgraded install that has no decision marker", async () => {
    await renderApp({ theme: "dark", accent: "coral", defaultGenerationQuality: "q8" });
    await startGeneration();
    expect(modalIsShowing()).toBe(true);
  });

  it("asks for an explicit decision even when the current setting is off", async () => {
    await renderApp({ embedWorkflowInImages: false });
    await startGeneration();
    expect(modalIsShowing()).toBe(true);
  });

  it("keeps the choice open after a persistence failure and retries the same action", async () => {
    const failurePlan = { workflowPuts: 3 };
    await renderApp({}, failurePlan);
    const { gate } = await startGeneration();

    await click(buttonNamed("No thanks!"));
    await new Promise((resolve) => setTimeout(resolve, 1300));
    await settle();

    expect(modalIsShowing()).toBe(true);
    expect(document.body.textContent).toContain("could not save your choice");
    let resolved = false;
    gate.then(() => {
      resolved = true;
    });
    await Promise.resolve();
    expect(resolved).toBe(false);

    await click(buttonNamed("No thanks!"));
    await settle();
    await expect(gate).resolves.toBe(true);
  });

  it("opens and focuses Sharing inside the Simple shell too", async () => {
    window.innerWidth = 390;
    await renderApp();
    const { gate } = await startGeneration();

    await click(buttonNamed("Go to settings"));
    await settle();

    await expect(gate).resolves.toBe(false);
    expect(document.body.querySelector(".su-group-title")?.textContent).toBe("Appearance");
    expect(document.activeElement?.textContent).toBe("Sharing");
  });
});
