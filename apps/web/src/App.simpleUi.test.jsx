import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { App } from "./main.jsx";
import { FakeEventSource, response, settle } from "./main.testSupport.jsx";
import { click } from "./testUtils/dom.js";
import { resetStartupTimingForTests } from "./startupTiming.js";

// Simple UI ⇄ full workspace switching at the App level (design handoff).
//
// The constraint this suite pins: the EXISTING workspace is what an install opens on and
// what it keeps rendering — the Simple UI only appears when the user asks for it (or on a
// phone). A regression that flipped the default would swap every user's UI silently, so
// the "unchanged by default" assertion is the first test here.

function sidebarSwitchButtons() {
  return [...document.body.querySelectorAll(".sidebar-footer .su-switch button")];
}

describe("App — Simple/Advanced shell switching", () => {
  let container;
  let root;

  beforeEach(() => {
    global.IS_REACT_ACT_ENVIRONMENT = true;
    container = document.createElement("div");
    document.body.appendChild(container);
    FakeEventSource.instances = [];
    window.EventSource = FakeEventSource;
    window.localStorage.clear();
    // Desktop-width viewport: the phone rule must not be what decides these cases.
    window.innerWidth = 1440;
    global.fetch = vi.fn((url) => {
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
        return Promise.resolve(response({}));
      }
      return Promise.resolve(response([]));
    });
  });

  afterEach(async () => {
    await act(async () => root?.unmount());
    container.remove();
    resetStartupTimingForTests();
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  async function renderApp() {
    root = createRoot(container);
    await act(async () => {
      root.render(<App />);
    });
    await settle();
  }

  it("opens on the FULL workspace, unchanged, for an untouched install", async () => {
    await renderApp();
    expect(container.querySelector("main.app")).toBeTruthy();
    expect(container.querySelector(".workspace")).toBeTruthy();
    expect(container.querySelector(".su-root")).toBeNull();
  });

  it("records Assets ready only after the advanced Library surface commits", async () => {
    let now = 0;
    const mark = vi.fn();
    vi.stubGlobal("performance", {
      clearMarks: vi.fn(),
      clearMeasures: vi.fn(),
      mark,
      measure: vi.fn(),
      now: () => ++now,
    });
    resetStartupTimingForTests();

    await renderApp();

    expect(mark.mock.calls.map(([name]) => name)).toEqual([
      "sceneworks.access-resolved",
      "sceneworks.media-authorization-settled",
      "sceneworks.projects-committed",
      "sceneworks.active-project-selected",
      "sceneworks.asset-request-start",
      "sceneworks.asset-request-settled",
      "sceneworks.assets-ready-render",
    ]);
    expect(container.querySelector(".workspace")).toBeTruthy();
  });

  it("adds the Simple/Advanced switch to the workspace sidebar, on Advanced", async () => {
    await renderApp();
    const [simple, advanced] = sidebarSwitchButtons();
    expect(simple.textContent).toBe("Simple");
    expect(advanced.textContent).toBe("Advanced");
    expect(advanced.className).toContain("active");
    expect(simple.className).not.toContain("active");
  });

  it("swaps to the Simple shell when the sidebar switch is set to Simple, and back", async () => {
    await renderApp();
    await click(sidebarSwitchButtons()[0]);

    expect(container.querySelector(".su-root")).toBeTruthy();
    expect(container.querySelector("main.app")).toBeNull();
    expect(container.querySelector(".su-topbar-title strong").textContent).toBe("Image Studio");

    // Back again through the Simple shell's own footer switch — same component.
    const advanced = [...container.querySelectorAll(".su-nav-footer .su-switch button")].find(
      (node) => node.textContent === "Advanced",
    );
    await click(advanced);
    expect(container.querySelector("main.app")).toBeTruthy();
    expect(container.querySelector(".su-root")).toBeNull();
  });

  it("opens on the Simple shell when the durable preference says so", async () => {
    global.fetch = vi.fn((url) => {
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
        return Promise.resolve(response({ simpleUi: true }));
      }
      return Promise.resolve(response([]));
    });

    await renderApp();
    expect(container.querySelector(".su-root")).toBeTruthy();
    // …and the durable value is mirrored into the instant-paint cache for the next launch.
    expect(window.localStorage.getItem("sceneworks-simple-ui-default")).toBe("true");
  });

  it("forces the Simple shell on a phone viewport and locks the switch", async () => {
    window.innerWidth = 390;
    await renderApp();
    expect(container.querySelector(".su-root")).toBeTruthy();
    const [simple, advanced] = [
      ...container.querySelectorAll(".su-nav-footer .su-switch button"),
    ];
    expect(simple.disabled).toBe(true);
    expect(advanced.disabled).toBe(true);
  });
});
