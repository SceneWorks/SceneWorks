import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { App } from "./main.jsx";
import { FakeEventSource, response, settle } from "./main.testSupport.jsx";
import { click } from "./testUtils/dom.js";
import { resetStartupTimingForTests } from "./startupTiming.js";

// sc-15136 — the durable UI-preference writes must be AUTHENTICATED on the wire.
//
// `/api/v1/ui-preferences` is method-asymmetric on the server (sc-8869, F-067): the GET is public so
// the theme can paint before the access gate, but the PUT writes `ui-preferences.json` to disk and is
// gated. App.jsx sent all three of its writes (theme, accent, the Simple-UI default) with an empty
// token and swallowed the 401 with `.catch(() => {})`, so on any remote-auth deployment those
// preferences never reached disk — invisibly, because each also writes a localStorage cache. None of
// the three had any token coverage before sc-15136, which is how all three shipped that way.
//
// These run against the REAL transport (only `fetch` is faked) rather than a mocked `apiFetch`, so the
// assertion is on the header that actually leaves the client. A mocked-transport test could pass while
// `apiFetch` dropped the token.

const TOKEN = "lan-password-1";

function fakeApi() {
  return vi.fn((url) => {
    const path = new URL(url).pathname;
    if (path.endsWith("/health")) {
      return Promise.resolve(response({ status: "ok", authRequired: true }));
    }
    if (path.endsWith("/access")) {
      return Promise.resolve(response({ authRequired: true }));
    }
    if (path.endsWith("/auth/verify")) {
      return Promise.resolve(response({ ok: true }));
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
}

// Every PUT the client made to the preferences route, as { body, token }.
function preferenceWrites() {
  return global.fetch.mock.calls
    .filter(([url, init]) => new URL(url).pathname.endsWith("/ui-preferences") && init?.method === "PUT")
    .map(([, init]) => ({
      body: JSON.parse(init.body),
      // `headers` is a Headers instance built by apiFetch.
      token: init.headers.get("X-SceneWorks-Token"),
    }));
}

describe("App — durable UI preference writes carry the access token (sc-15136)", () => {
  let container;
  let root;

  beforeEach(() => {
    global.IS_REACT_ACT_ENVIRONMENT = true;
    container = document.createElement("div");
    document.body.appendChild(container);
    FakeEventSource.instances = [];
    window.EventSource = FakeEventSource;
    window.localStorage.clear();
    // A remote browser that already unlocked the host: the verified password is the live token.
    window.localStorage.setItem("sceneworks-token", TOKEN);
    window.innerWidth = 1440;
    global.fetch = fakeApi();
  });

  afterEach(async () => {
    await act(async () => root?.unmount());
    container.remove();
    resetStartupTimingForTests();
    window.localStorage.clear();
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

  const themeToggle = () =>
    [...document.body.querySelectorAll("button")].find((button) =>
      (button.getAttribute("title") ?? "").startsWith("Switch to "),
    );

  it("sends the token on the theme PUT, so a remote theme change reaches ui-preferences.json", async () => {
    await renderApp();
    const toggle = themeToggle();
    expect(toggle, "theme toggle").toBeTruthy();

    await click(toggle);
    await settle();

    const writes = preferenceWrites();
    expect(writes.length).toBeGreaterThan(0);
    // Pinned to the real token, never to "": a "" assertion here would pass with the fix reverted.
    expect(writes.at(-1)).toEqual({ body: { theme: "dark" }, token: TOKEN });
  });

  it("sends the token on the accent PUT", async () => {
    await renderApp();
    // The topbar accent picker collapses to a trigger; the alternatives live in the dropdown.
    const trigger = document.body.querySelector(".accent-picker .accent-swatch.active");
    expect(trigger, "accent picker trigger").toBeTruthy();
    await click(trigger);

    const swatch = document.body.querySelector('.accent-picker-menu [role="option"]');
    expect(swatch, "an alternative accent swatch").toBeTruthy();

    await click(swatch);
    await settle();

    const writes = preferenceWrites();
    expect(writes.length).toBeGreaterThan(0);
    expect(writes.at(-1).token).toBe(TOKEN);
    expect(Object.keys(writes.at(-1).body)).toEqual(["accent"]);
  });

  it("sends the token on the Simple-UI-default PUT", async () => {
    // The third App-owned write, and the one with the longest reach: the toggle lives in the Simple
    // shell's Settings screen, whose handler is App's `changeSimpleUiDefault`.
    await renderApp();
    await click([...document.body.querySelectorAll(".sidebar-footer .su-switch button")][0]);
    await settle();

    const settingsNav = [...document.body.querySelectorAll("button")].find(
      (button) => button.textContent.trim() === "Settings",
    );
    expect(settingsNav, "Simple shell Settings nav").toBeTruthy();
    await click(settingsNav);
    await settle();

    const toggle = document.body.querySelector('[aria-label="Use Simple UI by default"]');
    expect(toggle, "Use Simple UI by default toggle").toBeTruthy();
    await click(toggle);
    await settle();

    const writes = preferenceWrites();
    expect(writes.length).toBeGreaterThan(0);
    expect(writes.at(-1)).toEqual({ body: { simpleUi: true }, token: TOKEN });
  });

  it("never sends the token on the pre-auth GET, which is public by design", async () => {
    // The launch-time read runs before the gate has a token and the server keeps it public to
    // avoid a theme flash. Authenticating it would be harmless but is deliberately not done —
    // this pins the asymmetry so a future "just send it everywhere" change is a conscious one.
    await renderApp();
    const gets = global.fetch.mock.calls.filter(
      ([url, init]) =>
        new URL(url).pathname.endsWith("/ui-preferences") && (init?.method ?? "GET") === "GET",
    );
    expect(gets.length).toBeGreaterThan(0);
    for (const [, init] of gets) {
      expect(init.headers.get("X-SceneWorks-Token")).toBe(null);
    }
  });
});
