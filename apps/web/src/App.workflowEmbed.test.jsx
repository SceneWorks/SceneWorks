import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { FakeEventSource, response, settle } from "./main.testSupport.jsx";
import { click } from "./testUtils/dom.js";
import { resetStartupTimingForTests } from "./startupTiming.js";
import { resetAccessTokenForTests } from "./accessToken.js";

// The first-run embedded-workflow disclosure (sc-15953), asserted through the REAL App shell.
//
// The observable event is "a generation was accepted while embedding is on", which App hangs off
// the `afterCreate` hook of the ONE image-job creator every studio goes through. Rather than
// driving a whole studio form to reach it, this captures the hook App actually hands that creator
// and calls it — so what is under test is App's wiring, its once-only rule and its persistence,
// without a test that would break the next time a studio's submit button moved.
const hoisted = vi.hoisted(() => ({ afterCreate: null }));

vi.mock("./createJob.js", async (importOriginal) => {
  const actual = await importOriginal();
  return {
    ...actual,
    makeCreateJob(options) {
      if (options.definition === actual.CREATE_JOB_DEFINITIONS.image) {
        hoisted.afterCreate = options.afterCreate;
      }
      return actual.makeCreateJob(options);
    },
  };
});

const { App } = await import("./main.jsx");

const NOTICE_TITLE = "Your generated images carry their recipe";

function fakeApi(storedPreferences) {
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
        return Promise.resolve(response({ ...storedPreferences, ...JSON.parse(init.body) }));
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

const noticeIsShowing = () => document.body.textContent.includes(NOTICE_TITLE);

describe("App — the one-time embedded-workflow disclosure (sc-15953)", () => {
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
    hoisted.afterCreate = null;
  });

  afterEach(async () => {
    await act(async () => root?.unmount());
    container.remove();
    resetStartupTimingForTests();
    window.localStorage.clear();
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  async function renderApp(storedPreferences = {}) {
    global.fetch = fakeApi(storedPreferences);
    root = createRoot(container);
    await act(async () => {
      root.render(<App />);
    });
    await settle();
  }

  async function generate() {
    expect(hoisted.afterCreate, "App must hand the image-job creator an afterCreate hook").toBeTypeOf(
      "function",
    );
    await act(async () => {
      hoisted.afterCreate({ id: "job-1" });
    });
    await settle();
  }

  it("is not on screen until a generation is submitted", async () => {
    // It is a disclosure about what a generated file will carry, so it has nothing to say to
    // someone who has not generated anything.
    await renderApp();
    expect(noticeIsShowing()).toBe(false);
  });

  it("appears on the first generation, is dismissible, and never comes back", async () => {
    await renderApp();
    await generate();
    expect(noticeIsShowing()).toBe(true);

    const dismiss = [...document.body.querySelectorAll("button")].find(
      (button) => button.textContent.trim() === "Got it",
    );
    expect(dismiss, "a dismiss button").toBeTruthy();
    await click(dismiss);
    await settle();

    expect(noticeIsShowing()).toBe(false);
    // Durable, not a browser flag: the desktop shell's origin changes each launch and would take
    // localStorage with it, re-showing the notice forever.
    expect(preferenceWrites()).toContainEqual({ workflowEmbedNoticeSeen: true });

    // A second, third and fourth generation in the same session must not bring it back.
    await generate();
    await generate();
    expect(noticeIsShowing()).toBe(false);
  });

  it("never appears for a user who has already been told, across a relaunch", async () => {
    // The stored flag is what a restart restores. This is the case that would regress into
    // "fires on every launch" if the flag lived only in localStorage.
    await renderApp({ workflowEmbedNoticeSeen: true });
    await generate();
    expect(noticeIsShowing()).toBe(false);
  });

  it("fires once for an install that was already generating before the feature shipped", async () => {
    // The upgrade case. An existing install has no stored flag — which reads as "never told" —
    // and embedding defaults ON, so this is exactly the user the disclosure exists for. A
    // mechanism keyed on "this is a fresh install" would stay silent here.
    await renderApp({ theme: "dark", accent: "coral", defaultGenerationQuality: "q8" });
    await generate();
    expect(noticeIsShowing()).toBe(true);
  });

  it("stays silent when the user has turned embedding off", async () => {
    // Nothing is being disclosed, because nothing is being embedded.
    await renderApp({ embedWorkflowInImages: false });
    await generate();
    expect(noticeIsShowing()).toBe(false);
  });

  it("does not fire for a POST the API refused", async () => {
    // `afterCreate` only runs on a job the server accepted; a failed submit writes no file and
    // must not burn the one disclosure the user gets.
    await renderApp();
    await act(async () => {
      hoisted.afterCreate(null);
    });
    await settle();
    expect(noticeIsShowing()).toBe(false);
  });
});
