import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { click } from "../testUtils/dom.js";

// SettingsScreen computes `isDesktop` from window.__TAURI__ at module load, so we
// set the Tauri bridge and re-import the module fresh in each test.
async function changeField(input, value) {
  await act(async () => {
    const setter = Object.getOwnPropertyDescriptor(input.constructor.prototype, "value")?.set;
    setter?.call(input, value);
    input.dispatchEvent(
      new window.Event(input.tagName === "SELECT" ? "change" : "input", { bubbles: true }),
    );
  });
}

describe("SettingsScreen service credentials", () => {
  let container;
  let root;
  let invoke;
  let credentials;
  let SettingsScreen;

  beforeEach(async () => {
    global.IS_REACT_ACT_ENVIRONMENT = true;
    credentials = [];
    invoke = vi.fn(async (command, args) => {
      switch (command) {
        case "get_app_settings":
          return {};
        case "get_gpu_info":
          return { platform: "windows", devices: [] };
        case "list_credentials":
          return credentials;
        case "set_credential":
          credentials = [
            {
              host: args.host.replace(/^https?:\/\//i, "").split("/")[0].toLowerCase(),
              label: args.label,
              scheme: args.scheme,
              present: true,
            },
          ];
          return credentials;
        case "delete_credential":
          credentials = credentials.filter((credential) => credential.host !== args.host);
          return credentials;
        default:
          return null;
      }
    });
    window.__TAURI__ = { core: { invoke } };
    vi.resetModules();
    ({ SettingsScreen } = await import("./SettingsScreen.jsx"));
    container = document.createElement("div");
    document.body.appendChild(container);
    root = createRoot(container);
  });

  afterEach(async () => {
    await act(async () => {
      root.unmount();
    });
    container.remove();
    delete window.__TAURI__;
    vi.restoreAllMocks();
  });

  async function render() {
    await act(async () => {
      root.render(<SettingsScreen />);
    });
    // Flush the initial refresh() Promise.all.
    await act(async () => {});
  }

  it("lists a stored credential by host without exposing the token", async () => {
    credentials = [{ host: "huggingface.co", label: "Hugging Face", scheme: "bearer", present: true }];
    await render();
    expect(invoke).toHaveBeenCalledWith("list_credentials", undefined);
    expect(container.textContent).toContain("Hugging Face");
    expect(container.textContent).toContain("huggingface.co");
  });

  it("flags a recorded credential whose token is missing from the keychain", async () => {
    credentials = [{ host: "civitai.com", label: "Civit.ai", scheme: "query", present: false }];
    await render();
    expect(container.textContent).toContain("token missing");
  });

  it("saves a new credential via set_credential", async () => {
    await render();
    await changeField(container.querySelector('[aria-label="Credential host"]'), "https://Civitai.com");
    await changeField(container.querySelector('[aria-label="Credential label"]'), "Civit.ai");
    await changeField(container.querySelector('[aria-label="Authentication scheme"]'), "query");
    await changeField(container.querySelector('[aria-label="Credential token"]'), "key123");
    await click(container.querySelector(".settings-credential-form button"));
    expect(invoke).toHaveBeenCalledWith("set_credential", {
      host: "https://Civitai.com",
      label: "Civit.ai",
      scheme: "query",
      token: "key123",
    });
    expect(container.textContent).toContain("Civit.ai");
    expect(container.textContent).toContain("civitai.com");
    expect(container.textContent).not.toContain("key123");
  });

  it("removes a credential via delete_credential", async () => {
    credentials = [{ host: "civitai.com", label: "Civit.ai", scheme: "query", present: true }];
    await render();
    await click(container.querySelector(".settings-credential button"));
    expect(invoke).toHaveBeenCalledWith("delete_credential", { host: "civitai.com" });
  });
});

describe("SettingsScreen server (REST) mode", () => {
  let container;
  let root;
  let apiFetch;
  let SettingsScreen;

  beforeEach(async () => {
    global.IS_REACT_ACT_ENVIRONMENT = true;
    delete window.__TAURI__;
    apiFetch = vi.fn(async (_path, _token, options) => {
      if (options?.method === "PUT") {
        return [{ host: "civitai.com", label: "Civit.ai", scheme: "query", present: true }];
      }
      if (options?.method === "DELETE") {
        return [];
      }
      return [{ host: "huggingface.co", label: "Hugging Face", scheme: "bearer", present: true }];
    });
    vi.resetModules();
    vi.doMock("../api.js", () => ({
      apiFetch,
      isAbortError: () => false,
      API_BASE_URL: "",
      eventUrl: () => "",
    }));
    ({ SettingsScreen } = await import("./SettingsScreen.jsx"));
    container = document.createElement("div");
    document.body.appendChild(container);
    root = createRoot(container);
  });

  afterEach(async () => {
    await act(async () => {
      root.unmount();
    });
    container.remove();
    vi.doUnmock("../api.js");
    vi.restoreAllMocks();
  });

  async function render() {
    await act(async () => {
      root.render(<SettingsScreen />);
    });
    await act(async () => {});
  }

  it("lists credentials over REST and hides the desktop-only cards", async () => {
    await render();
    expect(apiFetch).toHaveBeenCalledWith("/api/v1/credentials", expect.anything());
    expect(container.textContent).toContain("huggingface.co");
    expect(container.textContent).not.toContain("Data directory");
    expect(container.textContent).not.toContain("Detected GPU");
  });

  it("saves a credential via PUT to the API", async () => {
    await render();
    await changeField(container.querySelector('[aria-label="Credential host"]'), "civitai.com");
    await changeField(container.querySelector('[aria-label="Credential token"]'), "key123");
    await click(container.querySelector(".settings-credential-form button"));
    expect(apiFetch).toHaveBeenCalledWith(
      "/api/v1/credentials",
      expect.anything(),
      expect.objectContaining({ method: "PUT" }),
    );
  });

  // epic 4484 stories 10/12: a remote browser hides the Tauri-only cards (including
  // the desktop-only Remote Access controls) but keeps the inference-worker restart,
  // which routes over REST instead of the Tauri command.
  it("hides desktop-only cards and restarts the worker over REST", async () => {
    await render();
    expect(container.textContent).not.toContain("Remote access (LAN)");
    expect(container.textContent).not.toContain("Data directory");
    expect(container.textContent).not.toContain("Detected GPU");
    expect(container.textContent).not.toContain("Setup wizard");
    expect(container.textContent).toContain("Inference worker");
    const restartButton = [...container.querySelectorAll("button")].find(
      (button) => button.textContent.trim() === "Restart worker",
    );
    expect(restartButton).toBeTruthy();
    await click(restartButton);
    expect(apiFetch).toHaveBeenCalledWith(
      "/api/v1/worker/restart",
      expect.anything(),
      expect.objectContaining({ method: "POST" }),
    );
  });
});

describe("SettingsScreen remote access (desktop)", () => {
  let container;
  let root;
  let invoke;
  let remote;
  let SettingsScreen;

  beforeEach(async () => {
    global.IS_REACT_ACT_ENVIRONMENT = true;
    remote = {
      enabled: false,
      port: 8787,
      passwordSet: false,
      lanAddress: "192.168.1.50",
      lanCandidates: ["192.168.1.50"],
      url: "http://192.168.1.50:8787",
      defaultPort: 8787,
      platform: "macos",
    };
    invoke = vi.fn(async (command, args) => {
      switch (command) {
        case "get_app_settings":
          return {};
        case "get_gpu_info":
          return { platform: "macos", devices: [] };
        case "list_credentials":
          return [];
        case "get_remote_access":
          return remote;
        case "set_remote_access_password":
          remote = { ...remote, passwordSet: true };
          return remote;
        case "clear_remote_access_password":
          remote = { ...remote, passwordSet: false, enabled: false };
          return remote;
        case "set_remote_access":
          remote = { ...remote, enabled: args.enabled, port: args.port };
          return remote;
        default:
          return null;
      }
    });
    window.__TAURI__ = { core: { invoke } };
    vi.resetModules();
    ({ SettingsScreen } = await import("./SettingsScreen.jsx"));
    container = document.createElement("div");
    document.body.appendChild(container);
    root = createRoot(container);
  });

  afterEach(async () => {
    await act(async () => {
      root.unmount();
    });
    container.remove();
    delete window.__TAURI__;
    vi.restoreAllMocks();
  });

  async function render() {
    await act(async () => {
      root.render(<SettingsScreen />);
    });
    await act(async () => {});
  }

  const button = (label) =>
    [...container.querySelectorAll("button")].find((b) => b.textContent.trim() === label);

  it("shows the section, the LAN URL, and blocks enabling without a password", async () => {
    await render();
    expect(container.textContent).toContain("Remote access (LAN)");
    expect(container.textContent).toContain("http://192.168.1.50:8787");
    expect(button("Enable remote access").disabled).toBe(true);
  });

  it("sets a password, then can enable remote access on the chosen port", async () => {
    await render();
    await changeField(
      container.querySelector('[aria-label="Remote access password"]'),
      "lan-pass",
    );
    await click(button("Set password"));
    expect(invoke).toHaveBeenCalledWith("set_remote_access_password", { password: "lan-pass" });
    // Re-rendered with passwordSet → the enable button is now allowed.
    expect(button("Enable remote access").disabled).toBe(false);
    await click(button("Enable remote access"));
    expect(invoke).toHaveBeenCalledWith("set_remote_access", { enabled: true, port: 8787 });
  });
});

// epic 7819 Phase 2: live-apply the GPU memory target (sc-7824, no worker restart) and show live
// MLX memory telemetry (sc-7825) on macOS with a known unified-memory total.
describe("SettingsScreen GPU memory (desktop macOS)", () => {
  let container;
  let root;
  let invoke;
  let SettingsScreen;
  const GIB = 1024 * 1024 * 1024;

  beforeEach(async () => {
    global.IS_REACT_ACT_ENVIRONMENT = true;
    invoke = vi.fn(async (command) => {
      switch (command) {
        case "get_app_settings":
          return {}; // no cap → slider starts at 100% (Off)
        case "get_gpu_info":
          return { platform: "macos", devices: ["Apple M3 Max"], unifiedMemoryMb: 131072 };
        case "list_credentials":
          return [];
        case "get_remote_access":
          return null;
        case "get_gpu_telemetry":
          return { activeBytes: 20 * GIB, peakBytes: 40 * GIB, cacheBytes: 2 * GIB, limitBytes: 64 * GIB };
        case "set_gpu_memory_limit":
          return { gpuMemoryLimitFraction: 0.5 };
        default:
          return null;
      }
    });
    window.__TAURI__ = { core: { invoke } };
    vi.resetModules();
    ({ SettingsScreen } = await import("./SettingsScreen.jsx"));
    container = document.createElement("div");
    document.body.appendChild(container);
    root = createRoot(container);
  });

  afterEach(async () => {
    await act(async () => {
      root.unmount();
    });
    container.remove();
    delete window.__TAURI__;
    vi.restoreAllMocks();
  });

  async function render() {
    await act(async () => {
      root.render(<SettingsScreen />);
    });
    await act(async () => {});
  }

  const slider = () =>
    container.querySelector(
      '[aria-label="GPU memory target for SceneWorks (percent of unified memory)"]',
    );

  it("shows live MLX memory telemetry from the worker", async () => {
    await render();
    expect(invoke).toHaveBeenCalledWith("get_gpu_telemetry", undefined);
    expect(container.textContent).toContain("MLX memory");
    expect(container.textContent).toContain("Active: 20.0 GB");
    expect(container.textContent).toContain("Peak: 40.0 GB");
    expect(container.textContent).toContain("Limit: 64 GB");
  });

  it("applies the GPU memory target live, without restarting the worker", async () => {
    await render();
    const input = slider();
    expect(input).toBeTruthy();
    await changeField(input, "50");
    await act(async () => {
      input.dispatchEvent(new window.MouseEvent("mouseup", { bubbles: true }));
    });
    expect(invoke).toHaveBeenCalledWith("set_gpu_memory_limit", { fraction: 0.5 });
    expect(invoke).not.toHaveBeenCalledWith("restart_worker");
    expect(container.textContent).toContain("applies within a couple of seconds");
  });
});
