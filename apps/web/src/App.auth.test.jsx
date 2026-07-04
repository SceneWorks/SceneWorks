import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { App } from "./main.jsx";
import { getMediaTicket, setMediaTicket } from "./api.js";
import { FakeEventSource, response, errorResponse, settle, buttonInside, changeField } from "./main.testSupport.jsx";

describe("SceneWorks app shell", () => {
  let container;
  let root;

  beforeEach(() => {
    global.IS_REACT_ACT_ENVIRONMENT = true;
    container = document.createElement("div");
    document.body.appendChild(container);
    FakeEventSource.instances = [];
    window.EventSource = FakeEventSource;
    window.localStorage.clear();
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
      return Promise.resolve(response([]));
    });
  });

  afterEach(() => {
    act(() => {
      root?.unmount();
    });
    container.remove();
    vi.restoreAllMocks();
  });

  // call apiFetch directly (Logs, Image Editor, Pose Library, useUserPoseLoader)
  // read it from context; before the fix they got `undefined` and never sent the
  // X-SceneWorks-Token header, breaking every pairing-token deployment.
  it("provides the pairing token through context so screens send X-SceneWorks-Token", async () => {
    window.localStorage.setItem("sceneworks-token", "pair-tok-123");
    const logsTokens = [];
    const baseFetch = global.fetch.getMockImplementation();
    global.fetch = vi.fn((url, options = {}) => {
      const path = new URL(url).pathname;
      if (path.endsWith("/api/v1/logs")) {
        logsTokens.push(new Headers(options.headers).get("X-SceneWorks-Token"));
        return Promise.resolve(response([]));
      }
      return baseFetch(url, options);
    });

    root = createRoot(container);
    await act(async () => {
      root.render(<App />);
    });
    await settle();

    await act(async () => {
      [...document.body.querySelectorAll("button")].find((button) => button.textContent === "Logs").click();
    });
    await settle();

    expect(logsTokens.length).toBeGreaterThan(0);
    expect(logsTokens.every((value) => value === "pair-tok-123")).toBe(true);
  });

  // sc-8808 (F-007): the login gate's password box keeps its own draft state.
  // Before the fix it wrote every keystroke straight into the live `token`, so
  // the first character flipped `authenticated` and the [authenticated, token]
  // effects fired refreshData + the SSE ticket POST per keystroke, each 401ing
  // with a partial password and filling the notices band mid-typing.
  function mockAuthRequiredFetch({ verifyOk, requests, tokens }) {
    global.fetch = vi.fn((url, options = {}) => {
      const path = new URL(url).pathname;
      requests?.push({ path, method: options.method ?? "GET" });
      tokens?.push(new Headers(options.headers ?? {}).get("X-SceneWorks-Token"));
      if (path.endsWith("/health")) {
        return Promise.resolve(response({ status: "ok", authRequired: true }));
      }
      if (path.endsWith("/access")) {
        return Promise.resolve(response({ authRequired: true }));
      }
      if (path.endsWith("/auth/verify")) {
        return Promise.resolve(response({ ok: verifyOk }));
      }
      if (path.endsWith("/jobs/events/ticket")) {
        return Promise.resolve(response({ ticket: "stream-ticket" }));
      }
      if (path.endsWith("/projects")) {
        return Promise.resolve(response([{ id: "project-default", name: "Default Project" }]));
      }
      return Promise.resolve(response([]));
    });
  }

  it("does not fire API calls or SSE connections while typing the login password", async () => {
    const requests = [];
    mockAuthRequiredFetch({ verifyOk: true, requests });

    root = createRoot(container);
    await act(async () => {
      root.render(<App />);
    });
    await settle();

    const gateInput = document.body.querySelector("#token");
    expect(gateInput).not.toBeNull();
    // Auth is required and no token is saved: no protected load, no SSE ticket.
    expect(requests.every((request) => !request.path.endsWith("/projects"))).toBe(true);
    expect(requests.every((request) => !request.path.endsWith("/jobs/events/ticket"))).toBe(true);
    expect(FakeEventSource.instances.length).toBe(0);

    const requestCountBeforeTyping = requests.length;
    for (const partial of ["s", "se", "sec", "secr", "secre", "secret"]) {
      await changeField(gateInput, partial);
      await settle();
    }

    // Typing is pure local state: zero requests, zero EventSource churn, no notices.
    expect(requests.length).toBe(requestCountBeforeTyping);
    expect(FakeEventSource.instances.length).toBe(0);
    expect(document.body.querySelectorAll(".notice.error").length).toBe(0);
    expect(gateInput.value).toBe("secret");
  });

  it("keeps the gate up with an inline error when the password is wrong", async () => {
    const requests = [];
    mockAuthRequiredFetch({ verifyOk: false, requests });

    root = createRoot(container);
    await act(async () => {
      root.render(<App />);
    });
    await settle();

    await changeField(document.body.querySelector("#token"), "wrong-password");
    await act(async () => {
      buttonInside(document.body, "Unlock").click();
    });
    await settle();

    // Wrong password: inline error inside the gate, token never persisted,
    // no protected loads and no SSE connection attempted.
    expect(document.body.querySelector(".auth-band")?.textContent).toContain("Incorrect password. Try again.");
    expect(window.localStorage.getItem("sceneworks-token")).toBeNull();
    expect(document.body.querySelector("#token")).not.toBeNull();
    expect(requests.filter((request) => request.path.endsWith("/auth/verify")).length).toBe(1);
    expect(requests.every((request) => !request.path.endsWith("/projects"))).toBe(true);
    expect(FakeEventSource.instances.length).toBe(0);
  });

  it("unlocks and loads data exactly once after the password verifies", async () => {
    const requests = [];
    const tokens = [];
    mockAuthRequiredFetch({ verifyOk: true, requests, tokens });

    root = createRoot(container);
    await act(async () => {
      root.render(<App />);
    });
    await settle();

    // Trailing whitespace exercises the trim: the stored token must be exact.
    await changeField(document.body.querySelector("#token"), "correct-password ");
    await act(async () => {
      buttonInside(document.body, "Unlock").click();
    });
    await settle();

    expect(window.localStorage.getItem("sceneworks-token")).toBe("correct-password");
    // The gate keys off the token state, so it drops after verification.
    expect(document.body.querySelector("#token")).toBeNull();
    expect(document.body.querySelector(".auth-band")).toBeNull();
    // The [authenticated, token] effect performs the initial load exactly once
    // (no duplicate refresh from the submit handler), with the verified token.
    const projectLoads = requests
      .map((request, index) => ({ ...request, headerToken: tokens[index] }))
      .filter((request) => request.path.endsWith("/projects"));
    expect(projectLoads.length).toBe(1);
    expect(projectLoads[0].headerToken).toBe("correct-password");
    // SSE comes up once, via the ticket POST, after authentication.
    expect(requests.filter((request) => request.path.endsWith("/jobs/events/ticket")).length).toBe(1);
    expect(FakeEventSource.instances.length).toBe(1);
    expect(FakeEventSource.instances[0].url).toContain("ticket=stream-ticket");
  });

  // sc-9063 (follow-up to sc-8810 / F-008): a media-ticket mint that keeps failing
  // in remote-auth mode must not block ALL data loading. `ready` waits for the
  // first mint attempt to SETTLE, not to succeed: on failure the lists/metadata
  // still load (thumbnails degrade to placeholders), a distinct notice names the
  // media-ticket problem, and the notice clears once a backoff retry lands.
  describe("media-ticket mint failure (sc-9063)", () => {
    function mockRemoteAuthFetch({ requests = [], mint }) {
      global.fetch = vi.fn((url, options = {}) => {
        const path = new URL(url).pathname;
        requests.push({ path, method: options.method ?? "GET" });
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
        if (path.endsWith("/files/ticket")) {
          return Promise.resolve(
            mint()
              ? response({ ticket: "media-ticket-1", expiresInSeconds: 300 })
              : errorResponse(500, "mint exploded"),
          );
        }
        if (path.endsWith("/projects")) {
          return Promise.resolve(response([{ id: "project-default", name: "Default Project" }]));
        }
        return Promise.resolve(response([]));
      });
      return requests;
    }

    afterEach(() => {
      setMediaTicket("");
    });

    it("still loads core data and shows a media-ticket notice when the mint persistently fails", async () => {
      window.localStorage.setItem("sceneworks-token", "remote-token");
      const requests = mockRemoteAuthFetch({ mint: () => false });

      root = createRoot(container);
      await act(async () => {
        root.render(<App />);
      });
      await settle();

      // The mint was attempted and failed…
      expect(requests.some((request) => request.path.endsWith("/files/ticket"))).toBe(true);
      expect(getMediaTicket()).toBe("");
      // …but core data still loads (pre-sc-9063 the whole app stayed empty).
      expect(requests.some((request) => request.path.endsWith("/projects"))).toBe(true);
      // And a distinct notice names the media-ticket problem.
      const noticeTexts = [...document.body.querySelectorAll(".notice.error")].map((node) => node.textContent);
      expect(noticeTexts.some((text) => text.includes("media ticket"))).toBe(true);
    });

    it("clears the media-ticket notice once a backoff retry succeeds", async () => {
      vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout"] });
      try {
        window.localStorage.setItem("sceneworks-token", "remote-token");
        let mintOk = false;
        mockRemoteAuthFetch({ mint: () => mintOk });

        root = createRoot(container);
        await act(async () => {
          root.render(<App />);
        });
        await settle();

        expect(getMediaTicket()).toBe("");
        const failedTexts = [...document.body.querySelectorAll(".notice.error")].map((node) => node.textContent);
        expect(failedTexts.some((text) => text.includes("media ticket"))).toBe(true);

        // The first backoff retry fires after 1s; let it succeed this time.
        mintOk = true;
        await act(async () => {
          vi.advanceTimersByTime(1000);
        });
        await settle();

        expect(getMediaTicket()).toBe("media-ticket-1");
        const remainingTexts = [...document.body.querySelectorAll(".notice.error")].map((node) => node.textContent);
        expect(remainingTexts.some((text) => text.includes("media ticket"))).toBe(false);
      } finally {
        vi.useRealTimers();
      }
    });

    it("still gates the initial data load on a successful mint (sc-8810 unchanged)", async () => {
      window.localStorage.setItem("sceneworks-token", "remote-token");
      const requests = mockRemoteAuthFetch({ mint: () => true });

      root = createRoot(container);
      await act(async () => {
        root.render(<App />);
      });
      await settle();

      expect(getMediaTicket()).toBe("media-ticket-1");
      const mintIndex = requests.findIndex((request) => request.path.endsWith("/files/ticket"));
      const projectsIndex = requests.findIndex((request) => request.path.endsWith("/projects"));
      expect(mintIndex).toBeGreaterThanOrEqual(0);
      expect(projectsIndex).toBeGreaterThan(mintIndex);
      expect(document.body.querySelectorAll(".notice.error").length).toBe(0);
    });

    it("lock resets the settled gate and clears the notice; unlock waits for the new mint", async () => {
      window.localStorage.setItem("sceneworks-token", "remote-token");
      // Mint behavior across the phases: fail on mount, then hang after unlock so
      // we can observe the gap between re-login and the new mint settling.
      let mintBehavior = "fail";
      let releaseMint = null;
      const requests = [];
      global.fetch = vi.fn((url, options = {}) => {
        const path = new URL(url).pathname;
        requests.push({ path, method: options.method ?? "GET" });
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
        if (path.endsWith("/files/ticket")) {
          if (mintBehavior === "fail") {
            return Promise.resolve(errorResponse(500, "mint exploded"));
          }
          return new Promise((resolve) => {
            releaseMint = () => resolve(response({ ticket: "media-ticket-2", expiresInSeconds: 300 }));
          });
        }
        if (path.endsWith("/projects")) {
          return Promise.resolve(response([{ id: "project-default", name: "Default Project" }]));
        }
        return Promise.resolve(response([]));
      });

      root = createRoot(container);
      await act(async () => {
        root.render(<App />);
      });
      await settle();

      // sc-9063 baseline: mint failed, data loaded anyway, notice is up.
      expect(requests.some((request) => request.path.endsWith("/projects"))).toBe(true);
      const failedTexts = [...document.body.querySelectorAll(".notice.error")].map((node) => node.textContent);
      expect(failedTexts.some((text) => text.includes("media ticket"))).toBe(true);

      // Lock: the gate returns and the failure state must not leak onto it — the
      // "Retrying in the background" notice clears (the backoff was stopped by the
      // mint effect's cleanup, so the message would be a lie).
      await act(async () => {
        buttonInside(document.body, "Lock").click();
      });
      await settle();
      expect(document.body.querySelector("#token")).not.toBeNull();
      expect(document.body.querySelectorAll(".notice.error").length).toBe(0);

      // Unlock while the new mint hangs: mediaTicketFailed was reset on lock, so
      // `ready` must wait for the NEW mint to settle — no data load in the gap
      // (sc-8810's mint-before-data ordering applies to re-login too).
      mintBehavior = "pending";
      const requestCountAtUnlock = requests.length;
      await changeField(document.body.querySelector("#token"), "remote-token");
      await act(async () => {
        buttonInside(document.body, "Unlock").click();
      });
      await settle();
      const afterUnlock = requests.slice(requestCountAtUnlock);
      expect(afterUnlock.some((request) => request.path.endsWith("/files/ticket"))).toBe(true);
      expect(afterUnlock.some((request) => request.path.endsWith("/projects"))).toBe(false);

      // The new mint settles: data loads with the fresh ticket already in place.
      await act(async () => {
        releaseMint();
      });
      await settle();
      expect(getMediaTicket()).toBe("media-ticket-2");
      expect(
        requests.slice(requestCountAtUnlock).some((request) => request.path.endsWith("/projects")),
      ).toBe(true);
      expect(document.body.querySelectorAll(".notice.error").length).toBe(0);
    });

    it("leaves auth-off (desktop/loopback) mode untouched: no mint, data loads, no notice", async () => {
      const requests = [];
      global.fetch = vi.fn((url, options = {}) => {
        const path = new URL(url).pathname;
        requests.push({ path, method: options.method ?? "GET" });
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
        return Promise.resolve(response([]));
      });

      root = createRoot(container);
      await act(async () => {
        root.render(<App />);
      });
      await settle();

      expect(requests.some((request) => request.path.endsWith("/files/ticket"))).toBe(false);
      expect(requests.some((request) => request.path.endsWith("/projects"))).toBe(true);
      expect(getMediaTicket()).toBe("");
      expect(document.body.querySelectorAll(".notice.error").length).toBe(0);
    });
  });

});
