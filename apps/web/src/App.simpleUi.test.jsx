import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { App } from "./main.jsx";
import { FakeEventSource, response, settle } from "./main.testSupport.jsx";
import { click, setInput } from "./testUtils/dom.js";
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

  it.each([1440, 390])("switches and creates workspaces in Simple mode at %i px", async (width) => {
    window.innerWidth = width;
    vi.stubGlobal("ResizeObserver", class {
      constructor(callback) { this.callback = callback; }
      observe() { this.callback([{ contentRect: { width } }]); }
      disconnect() {}
    });
    const existingFetch = global.fetch;
    global.fetch = vi.fn((url, options) => {
      if (width === 390 && new URL(url).pathname.endsWith("/ui-preferences") &&
          (options?.method ?? "GET") === "GET") {
        return Promise.resolve(response({ sidebarCollapsed: true }));
      }
      if (new URL(url).pathname.endsWith("/projects")) {
        return Promise.resolve(response(options?.method === "POST"
          ? { id: "project-new", name: JSON.parse(options.body).name }
          : [
            { id: "project-default", name: "Default Project" },
            { id: "project-other", name: "Other Workspace" },
          ]));
      }
      return existingFetch(url, options);
    });
    await renderApp();
    if (width === 1440) await click(sidebarSwitchButtons()[0]);
    else await click(container.querySelector('[aria-label="Open menu"]'));

    const switcher = () => container.querySelector(".su-nav .project-switcher");
    expect(switcher().compareDocumentPosition(container.querySelector(".su-nav-group")) &
      Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
    await click(switcher().querySelector(".project-pill"));
    await click([...switcher().querySelectorAll('[role="option"]')].find(
      (node) => node.textContent === "Other Workspace",
    ));
    await settle();
    if (width === 390) await click(container.querySelector('[aria-label="Open menu"]'));
    expect(switcher().querySelector(".project-pill strong").textContent).toBe("Other Workspace");
    expect(container.querySelector(".su-root")).toBeTruthy();
    expect(global.fetch.mock.calls.some(([url]) =>
      new URL(url).pathname === "/api/v1/projects/project-other/assets",
    )).toBe(true);

    await click(switcher().querySelector(".project-pill"));
    await click(switcher().querySelector(".project-menu-item-new"));
    await act(async () => setInput(switcher().querySelector("input"), "New Workspace"));
    await act(async () => switcher().querySelector("form").dispatchEvent(
      new window.Event("submit", { bubbles: true, cancelable: true }),
    ));
    await settle();
    if (width === 390) await click(container.querySelector('[aria-label="Open menu"]'));
    expect(switcher().querySelector(".project-pill strong").textContent).toBe("New Workspace");
    expect(container.querySelector(".su-root")).toBeTruthy();
    if (width === 390) {
      expect(container.querySelector(".sidebar-toggle")).toBeNull();
      expect(global.fetch.mock.calls.some(([url, options]) =>
        new URL(url).pathname.endsWith("/ui-preferences") && options?.method === "PUT" &&
        Object.hasOwn(JSON.parse(options.body), "sidebarCollapsed"),
      )).toBe(false);
    }
  });

  it("keeps workspace drafts separate when switching within Simple mode", async () => {
    const existingFetch = global.fetch;
    global.fetch = vi.fn((url, options) => new URL(url).pathname.endsWith("/projects")
      ? Promise.resolve(response([
        { id: "project-default", name: "Default Project" },
        { id: "project-other", name: "Other Workspace" },
      ]))
      : existingFetch(url, options));
    await renderApp();
    await click(sidebarSwitchButtons()[0]);
    const prompt = () => container.querySelector("#su-image-prompt");
    async function typePrompt(value) {
      await act(async () => {
        Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, "value")
          .set.call(prompt(), value);
        prompt().dispatchEvent(new window.Event("input", { bubbles: true }));
      });
    }
    async function selectWorkspace(name) {
      await click(container.querySelector(".su-nav .project-pill"));
      await click([...container.querySelectorAll('.su-nav [role="option"]')].find(
        (node) => node.textContent === name,
      ));
      await settle();
    }

    await typePrompt("Default workspace draft");
    await selectWorkspace("Other Workspace");
    expect(prompt().value).toBe("");
    await typePrompt("Other workspace draft");
    await selectWorkspace("Default Project");
    expect(prompt().value).toBe("Default workspace draft");
    await selectWorkspace("Other Workspace");
    expect(prompt().value).toBe("Other workspace draft");

    await click([...container.querySelectorAll(".su-nav-item")].find(
      (node) => node.textContent === "Assets",
    ));
    await selectWorkspace("Default Project");
    expect(container.querySelector(".su-topbar-title strong").textContent).toBe("Assets");
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
    expect(container.querySelector(".sidebar-toggle")).toBeNull();
    expect(container.querySelector(".su-sidebar-collapsed")).toBeNull();
  });
});
