import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { App, eventUrl } from "./main.jsx";

class FakeEventSource {
  constructor(url) {
    this.url = url;
    this.listeners = {};
  }

  addEventListener(event, handler) {
    this.listeners[event] = handler;
  }

  close() {}
}

function response(payload) {
  return {
    ok: true,
    json: async () => payload,
  };
}

async function settle() {
  await act(async () => {
    await Promise.resolve();
    await Promise.resolve();
  });
}

describe("SceneWorks app shell", () => {
  let container;
  let root;

  beforeEach(() => {
    global.IS_REACT_ACT_ENVIRONMENT = true;
    container = document.createElement("div");
    document.body.appendChild(container);
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

  it("renders the app navigation against mocked API calls", async () => {
    root = createRoot(container);
    await act(async () => {
      root.render(<App />);
    });
    await settle();

    expect(container.textContent).toContain("Library");
    expect(container.textContent).toContain("Queue");
  });

  it("switches Replace Person to the replacement-capable video model", async () => {
    root = createRoot(container);
    await act(async () => {
      root.render(<App />);
    });
    await settle();

    await act(async () => {
      [...container.querySelectorAll("button")].find((button) => button.textContent === "Video").click();
    });
    await settle();
    await act(async () => {
      [...container.querySelectorAll("button")].find((button) => button.textContent === "Replace Person").click();
    });
    await settle();

    expect(container.textContent).toContain("Wan2.2");
    expect(container.textContent).toContain("V1 placeholder tracking");
  });

  it("adds the SSE ticket as a query parameter", () => {
    expect(eventUrl("/api/v1/jobs/events", "stream-ticket")).toContain("ticket=stream-ticket");
  });

  it("filters stale and placeholder-only GPU workers from the queue view", async () => {
    global.fetch.mockImplementation((url) => {
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
      if (path.endsWith("/workers")) {
        return Promise.resolve(
          response([
            {
              id: "python-gpu-0",
              gpuId: "0",
              gpuName: "Fixture GPU 0",
              status: "idle",
              capabilities: ["placeholder", "gpu", "image_generate"],
            },
            {
              id: "rust-gpu-1",
              gpuId: "1",
              gpuName: "Rust placeholder GPU",
              status: "idle",
              capabilities: ["placeholder", "gpu", "nvidia"],
            },
            {
              id: "stale-gpu-2",
              gpuId: "2",
              gpuName: "Stale GPU",
              status: "offline",
              capabilities: ["placeholder", "gpu", "image_generate"],
            },
            {
              id: "rust-cpu",
              gpuId: "cpu",
              gpuName: "Rust CPU utility worker",
              status: "idle",
              capabilities: ["placeholder", "cpu", "model_download"],
            },
          ]),
        );
      }
      return Promise.resolve(response([]));
    });

    root = createRoot(container);
    await act(async () => {
      root.render(<App />);
    });
    await settle();

    await act(async () => {
      [...container.querySelectorAll("button")].find((button) => button.textContent === "Queue").click();
    });
    await settle();

    expect(container.textContent).toContain("Fixture GPU 0");
    expect(container.textContent).toContain("Rust CPU utility worker");
    expect(container.textContent).not.toContain("Rust placeholder GPU");
    expect(container.textContent).not.toContain("Stale GPU");
    expect([...container.querySelector("#queue-gpu").options].map((option) => option.value)).toEqual(["auto", "0"]);
  });
});
