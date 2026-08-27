import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { click } from "../testUtils/dom.js";

// SettingsScreen computes `isDesktop` from window.__TAURI__ at module load, so we
// set the Tauri bridge and re-import the module fresh in each test.
//
// The redesigned screen (Settings-page handoff) reads theme + the worker registry off the
// app context and splits its content across four tabs, so every test mounts it inside a
// legacy combined <AppContext.Provider> and clicks through to the tab under test.
const APP_CONTEXT = { theme: "light", changeTheme: vi.fn(), visibleWorkers: [], macCapabilities: {} };

// AppContext has to be re-imported alongside the screen after vi.resetModules(): a statically
// imported context object is a DIFFERENT module instance from the one the freshly-imported
// screen consumes, so its Provider wouldn't be seen and useAppContext() would throw.
let AppContext;

// Sequential on purpose: AppContext.js is a dependency of the screen, and importing both
// concurrently races the freshly-reset module registry — the screen can end up bound to the
// real ../api.js instead of the test's vi.doMock'd one.
async function loadScreen() {
  const { SettingsScreen } = await import("./SettingsScreen.jsx");
  ({ AppContext } = await import("../context/AppContext.js"));
  return SettingsScreen;
}

// Mount the screen inside the app context and flush the initial refresh().
async function renderScreen(root, SettingsScreen, props = {}, context = APP_CONTEXT) {
  await act(async () => {
    root.render(
      <AppContext.Provider value={context}>
        <SettingsScreen {...props} />
      </AppContext.Provider>,
    );
  });
  await act(async () => {});
}

async function changeField(input, value) {
  await act(async () => {
    const setter = Object.getOwnPropertyDescriptor(input.constructor.prototype, "value")?.set;
    setter?.call(input, value);
    input.dispatchEvent(
      new window.Event(input.tagName === "SELECT" ? "change" : "input", { bubbles: true }),
    );
  });
}

// Click the tab whose label matches, so the assertions below run against its body.
async function openTab(container, label) {
  const tab = [...container.querySelectorAll(".mode-tab")].find(
    (button) => button.textContent.trim() === label,
  );
  expect(tab, `tab "${label}"`).toBeTruthy();
  await click(tab);
}

const buttonByText = (container, label) =>
  [...container.querySelectorAll("button")].find((button) => button.textContent.trim() === label);

// Settings → Storage → Model library (the Hugging Face cache home). The row tells the user where
// SceneWorks is ACTUALLY reading models from (override, else the platform default — never a bare
// "Default location"), and Change… hands off to App's relocation sequence (sc-19709), which is
// injected as `onChangeModelLibrary` so the screen never re-implements the two-write ordering.
describe("SettingsScreen model library location", () => {
  let container;
  let root;
  let invoke;
  let SettingsScreen;
  let appSettings;

  beforeEach(async () => {
    global.IS_REACT_ACT_ENVIRONMENT = true;
    appSettings = {};
    invoke = vi.fn(async (command) => {
      switch (command) {
        case "get_app_settings":
          return appSettings;
        case "get_storage_setup":
          return {
            dataDirDefault: "/Users/me/Library/Application Support/SceneWorks",
            hfHome: appSettings.hfHome ?? null,
            hfHomeDefault: "/Users/me/.cache/huggingface",
            // The shell resolves what the sidecars will actually receive: ambient env, then the
            // persisted override, then the default. The mock mirrors that (no ambient here).
            hfHomeActive: appSettings.hfHome ?? "/Users/me/.cache/huggingface",
            hfHomeFromEnvironment: false,
            storageConfigured: true,
            setupCompleted: true,
          };
        case "get_gpu_info":
          return { platform: "macos", devices: [] };
        case "list_credentials":
          return [];
        default:
          return null;
      }
    });
    window.__TAURI__ = { core: { invoke } };
    vi.resetModules();
    SettingsScreen = await loadScreen();
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

  async function render(props) {
    await renderScreen(root, SettingsScreen, props);
    await openTab(container, "Settings");
  }

  it("shows the platform default cache home when no override is set", async () => {
    await render({ onChangeModelLibrary: vi.fn() });
    expect(container.textContent).toContain("Model library (Hugging Face cache)");
    expect(container.textContent).toContain("/Users/me/.cache/huggingface");
    expect(container.textContent).not.toContain("Default location/Users/me/.cache");
  });

  it("shows the persisted override and reveals it", async () => {
    appSettings = { hfHome: "/Volumes/Models/huggingface" };
    await render({ onChangeModelLibrary: vi.fn() });
    expect(container.textContent).toContain("/Volumes/Models/huggingface");
    const reveal = [...container.querySelectorAll("button")].filter(
      (button) => button.textContent.trim() === "Reveal in Finder",
    );
    // Data directory (disabled: no override) then the model library.
    expect(reveal).toHaveLength(2);
    await click(reveal[1]);
    expect(invoke).toHaveBeenCalledWith("reveal_in_os", { path: "/Volumes/Models/huggingface" });
  });

  it("relocates through the injected sequence, re-reads settings and asks for a restart", async () => {
    const onChangeModelLibrary = vi.fn(async () => {
      appSettings = { hfHome: "/Volumes/Models/huggingface" };
      return { hfHome: "/Volumes/Models/huggingface", libraryRoot: "/Volumes/Models/huggingface/hub" };
    });
    await render({ onChangeModelLibrary });
    const change = [...container.querySelectorAll("button")].filter(
      (button) => button.textContent.trim() === "Change…",
    );
    expect(change).toHaveLength(2);
    await click(change[1]);
    expect(onChangeModelLibrary).toHaveBeenCalledTimes(1);
    expect(container.textContent).toContain("/Volumes/Models/huggingface");
    expect(container.textContent).toContain("Model library updated — restart SceneWorks to apply.");
  });

  it("shows the refusal inline and keeps the current location when the folder is rejected", async () => {
    const onChangeModelLibrary = vi.fn(async () => {
      throw new Error("That folder does not contain a SceneWorks model library.");
    });
    await render({ onChangeModelLibrary });
    const change = [...container.querySelectorAll("button")].filter(
      (button) => button.textContent.trim() === "Change…",
    );
    await click(change[1]);
    // The refusal renders INSIDE the model library row — the top-of-panel status line can be
    // scrolled out of view, which made a refusal look like a silent no-op.
    const row = [...container.querySelectorAll(".settings-row-title")]
      .find((title) => title.textContent.includes("Model library"))
      .closest("div").parentElement;
    expect(row.querySelector('[role="status"]').textContent).toContain(
      "That folder does not contain a SceneWorks model library.",
    );
    expect(container.textContent).toContain("/Users/me/.cache/huggingface");
    expect(container.textContent).not.toContain("restart SceneWorks to apply");
  });

  // `tauri dev` / a shell-profile export: an ambient `HF_HOME` beats the persisted override at
  // spawn, so the row must show the path actually in use and must not offer a change that would
  // silently never take effect.
  it("shows the environment-owned location and disables Change…", async () => {
    appSettings = { hfHome: "/Volumes/Models/huggingface" };
    const fromEnv = invoke.getMockImplementation();
    invoke.mockImplementation(async (command) => {
      if (command === "get_storage_setup") {
        return {
          ...(await fromEnv(command)),
          hfHomeActive: "/tmp/dev-hf-home",
          hfHomeFromEnvironment: true,
        };
      }
      return fromEnv(command);
    });
    await render({ onChangeModelLibrary: vi.fn() });
    expect(container.textContent).toContain("/tmp/dev-hf-home");
    expect(container.textContent).toContain("HF_HOME");
    expect(container.textContent).toContain("environment variable");
    const change = [...container.querySelectorAll("button")].filter(
      (button) => button.textContent.trim() === "Change…",
    );
    expect(change[1].disabled).toBe(true);
  });

  it("does nothing when the picker is dismissed", async () => {
    const onChangeModelLibrary = vi.fn(async () => null);
    await render({ onChangeModelLibrary });
    const change = [...container.querySelectorAll("button")].filter(
      (button) => button.textContent.trim() === "Change…",
    );
    await click(change[1]);
    expect(onChangeModelLibrary).toHaveBeenCalledTimes(1);
    expect(container.textContent).not.toContain("restart SceneWorks to apply");
  });
});

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
    SettingsScreen = await loadScreen();
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

  // Credentials live on the Settings tab.
  async function render() {
    await renderScreen(root, SettingsScreen);
    await openTab(container, "Settings");
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
    // The scheme <select> is now a two-chip radiogroup.
    await click(
      [...container.querySelectorAll('[aria-label="Authentication scheme"] button')].find(
        (chip) => chip.textContent.trim() === "Query token",
      ),
    );
    await changeField(container.querySelector('[aria-label="Credential token"]'), "key123");
    await click(buttonByText(container, "Add"));
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

  it("surfaces an actionable Linux Secret Service save failure", async () => {
    invoke.mockImplementation(async (command) => {
      if (command === "get_app_settings") return {};
      if (command === "get_gpu_info") return { platform: "linux", devices: [] };
      if (command === "list_credentials") return [];
      if (command === "set_credential") {
        throw new Error(
          "Couldn't save the credential because Linux Secret Service is unavailable or locked. " +
            "Start and unlock GNOME Keyring or KWallet, make sure a D-Bus user session is running, then try again.",
        );
      }
      return null;
    });
    await render();
    expect(container.textContent).toContain("Secret Service (GNOME Keyring or KWallet)");

    await changeField(container.querySelector('[aria-label="Credential host"]'), "huggingface.co");
    await changeField(container.querySelector('[aria-label="Credential token"]'), "hf_secret");
    await click(buttonByText(container, "Add"));

    expect(container.querySelector(".settings-status").textContent).toContain(
      "Start and unlock GNOME Keyring or KWallet",
    );
    expect(container.textContent).not.toContain("Credential saved.");
    expect(container.textContent).not.toContain("hf_secret");
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
    SettingsScreen = await loadScreen();
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
    await renderScreen(root, SettingsScreen);
  }

  it("lists credentials over REST and hides the desktop-only groups", async () => {
    await render();
    await openTab(container, "Settings");
    expect(apiFetch).toHaveBeenCalledWith("/api/v1/credentials", expect.anything());
    expect(container.textContent).toContain("huggingface.co");
    expect(container.textContent).not.toContain("Data directory");
    await openTab(container, "Device");
    expect(container.textContent).not.toContain("GPU memory");
  });

  it("saves a credential via PUT to the API", async () => {
    await render();
    await openTab(container, "Settings");
    await changeField(container.querySelector('[aria-label="Credential host"]'), "civitai.com");
    await changeField(container.querySelector('[aria-label="Credential token"]'), "key123");
    await click(buttonByText(container, "Add"));
    expect(apiFetch).toHaveBeenCalledWith(
      "/api/v1/credentials",
      expect.anything(),
      expect.objectContaining({ method: "PUT" }),
    );
  });

  // epic 4484 stories 10/12: a remote browser hides the Tauri-only groups (including
  // the desktop-only Remote Access controls, whose whole tab disappears) but keeps the
  // inference-worker restart, which routes over REST instead of the Tauri command.
  it("hides desktop-only groups and restarts the worker over REST", async () => {
    await render();
    expect(
      [...container.querySelectorAll(".mode-tab")].map((tab) => tab.textContent.trim()),
    ).toEqual(["Appearance", "Settings", "Device"]);
    await openTab(container, "Settings");
    expect(container.textContent).not.toContain("Data directory");
    await openTab(container, "Device");
    expect(container.textContent).not.toContain("Setup wizard");
    expect(container.textContent).toContain("Inference worker");
    await click(buttonByText(container, "Restart"));
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
    SettingsScreen = await loadScreen();
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
    await renderScreen(root, SettingsScreen);
    await openTab(container, "Server");
  }

  const remoteToggle = () => container.querySelector('[aria-label="Remote access"]');

  it("shows the section, the LAN URL, and blocks enabling without a password", async () => {
    await render();
    expect(container.textContent).toContain("Remote access (LAN)");
    expect(container.textContent).toContain("http://192.168.1.50:8787");
    expect(remoteToggle().disabled).toBe(true);
  });

  it("sets a password, then can enable remote access on the chosen port", async () => {
    await render();
    await changeField(
      container.querySelector('[aria-label="Remote access password"]'),
      "lan-pass",
    );
    await click(buttonByText(container, "Save"));
    expect(invoke).toHaveBeenCalledWith("set_remote_access_password", { password: "lan-pass" });
    // Re-rendered with passwordSet → the toggle is now allowed.
    expect(remoteToggle().disabled).toBe(false);
    await click(remoteToggle());
    expect(invoke).toHaveBeenCalledWith("set_remote_access", { enabled: true, port: 8787 });
    // …and flipping it back disables it.
    await click(remoteToggle());
    expect(invoke).toHaveBeenCalledWith("set_remote_access", { enabled: false, port: 8787 });
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
  let telemetry;

  beforeEach(async () => {
    global.IS_REACT_ACT_ENVIRONMENT = true;
    telemetry = { activeBytes: 20 * GIB, peakBytes: 40 * GIB, cacheBytes: 2 * GIB, limitBytes: 64 * GIB };
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
          return telemetry;
        case "set_gpu_memory_limit":
          return { gpuMemoryLimitFraction: 0.5 };
        default:
          return null;
      }
    });
    window.__TAURI__ = { core: { invoke } };
    vi.resetModules();
    SettingsScreen = await loadScreen();
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

  // The GPU controls live on the Device tab.
  async function render() {
    await renderScreen(root, SettingsScreen);
    await openTab(container, "Device");
  }

  const slider = () =>
    container.querySelector(
      '[aria-label="GPU memory target for SceneWorks (percent of unified memory)"]',
    );

  it("shows live MLX memory telemetry from the worker", async () => {
    await render();
    expect(invoke).toHaveBeenCalledWith("get_gpu_telemetry", undefined);
    expect(container.textContent).toContain("Live MLX memory");
    expect(container.textContent).toContain("20.0 GiB");
    expect(container.textContent).toContain("40.0 GiB");
    expect(container.textContent).toContain("64 GiB");
  });

  it("marks missing telemetry fields unavailable while preserving a reported zero", async () => {
    telemetry = { activeBytes: 0, limitBytes: 0 };
    await render();
    expect(container.textContent).toContain("0.0 GiB");
    expect(container.textContent).toContain("0 GiB");
    expect(container.textContent).toContain("Unavailable");
    const rows = [...container.querySelectorAll(".settings-kv")];
    expect(rows.find((row) => row.textContent.includes("Active"))?.textContent).toContain("0.0 GiB");
    expect(rows.find((row) => row.textContent.includes("Peak"))?.textContent).toContain("Unavailable");
  });

  it("summarises this machine from the detected GPU and the worker registry", async () => {
    await render();
    expect(container.textContent).toContain("Engine");
    expect(container.textContent).toContain("Unified memory");
    expect(container.textContent).toContain("128 GiB");
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

// Global "default generation quality" setting (epic 10721 / sc-10728). Persisted like theme/accent:
// a durable server copy (PUT /api/v1/ui-preferences) plus a localStorage instant-paint cache, so it
// survives a desktop relaunch (the shell's 127.0.0.1:<port> origin changes each launch and wipes
// origin-keyed localStorage). Renders in both desktop and web modes. These run in web (REST) mode with
// api.js mocked so refresh() resolves cleanly and the write-through PUT is observable.
describe("SettingsScreen default generation quality (sc-10728)", () => {
  const STORAGE_KEY = "sceneworks-default-generation-quality";
  let container;
  let root;
  let SettingsScreen;
  let apiFetch;

  beforeEach(async () => {
    global.IS_REACT_ACT_ENVIRONMENT = true;
    delete window.__TAURI__;
    window.localStorage.clear();
    vi.resetModules();
    apiFetch = vi.fn(async () => []);
    vi.doMock("../api.js", () => ({
      apiFetch,
      isAbortError: () => false,
      API_BASE_URL: "",
      eventUrl: () => "",
    }));
    SettingsScreen = await loadScreen();
    container = document.createElement("div");
    document.body.appendChild(container);
    root = createRoot(container);
  });

  afterEach(async () => {
    await act(async () => {
      root.unmount();
    });
    container.remove();
    window.localStorage.clear();
    vi.doUnmock("../api.js");
    vi.restoreAllMocks();
  });

  // Default quality lives on the Appearance tab, which is the landing tab.
  async function render() {
    await renderScreen(root, SettingsScreen);
  }

  const qualityChips = () =>
    [...container.querySelectorAll('[aria-label="Default generation quality"] button')];
  const selectedQuality = () =>
    qualityChips().find((chip) => chip.getAttribute("aria-checked") === "true");
  const chooseQuality = async (label) =>
    click(qualityChips().find((chip) => chip.textContent.trim() === label));

  it("renders the control defaulting to Auto when nothing is stored (epic 10721 R3)", async () => {
    await render();
    expect(container.textContent).toContain("Default quality");
    // Auto leads the row, alongside the explicit tiers.
    expect(qualityChips().map((chip) => chip.textContent.trim())).toEqual([
      "Auto",
      "bf16",
      "Q8",
      "Q4",
    ]);
    expect(selectedQuality().textContent.trim()).toBe("Auto");
  });

  it("writes a change to the localStorage instant-paint cache and confirms it in the status line (cache path)", async () => {
    await render();
    await chooseQuality("bf16");
    expect(window.localStorage.getItem(STORAGE_KEY)).toBe("bf16");
    expect(selectedQuality().textContent.trim()).toBe("bf16");
    expect(container.textContent).toContain("High fidelity (bf16)");
  });

  it("writes the change through to the durable server copy via PUT /api/v1/ui-preferences (durable path)", async () => {
    // The acceptance criterion is "persists across sessions". On the desktop shell the localStorage
    // cache alone does NOT survive a relaunch (the origin changes), so the change must also PUT to the
    // durable ui-preferences store — same contract as theme/accent in App.jsx. Sending only this field
    // relies on the endpoint MERGING, so theme/accent can't be clobbered.
    //
    // The PUT must carry the access token (sc-15136): it is a GATED route — only the pre-auth GET is
    // public — so the credential-less write this used to send 401'd on every remote-auth deployment and
    // the preference never reached disk. A non-empty token is pinned on purpose: this assertion
    // used to read `""`, which is what let the defect sit green here — it passes either way.
    window.localStorage.setItem("sceneworks-token", "lan-password-1");
    await render();
    await chooseQuality("Q4");
    expect(apiFetch).toHaveBeenCalledWith(
      "/api/v1/ui-preferences",
      "lan-password-1",
      expect.objectContaining({
        method: "PUT",
        body: JSON.stringify({ defaultGenerationQuality: "q4" }),
      }),
    );
  });

  it("rolls back the optimistic cache and reports a failed durable preference write", async () => {
    apiFetch.mockImplementation(async (path, _token, options) => {
      if (path === "/api/v1/ui-preferences" && options?.method === "PUT") {
        throw new Error("disk full");
      }
      return [];
    });
    await render();
    await chooseQuality("Q4");
    expect(selectedQuality().textContent.trim()).toBe("Auto");
    expect(window.localStorage.getItem(STORAGE_KEY)).toBe("auto");
    expect(container.textContent).toContain("Could not save default generation quality");
    expect(container.textContent).not.toContain("Default generation quality set to");
  });

  it("does not let an older failed PUT roll back a newer successful quality choice", async () => {
    const writes = [];
    apiFetch.mockImplementation((path, _token, options) => {
      if (path === "/api/v1/ui-preferences" && options?.method === "PUT") {
        return new Promise((resolve, reject) => writes.push({ resolve, reject }));
      }
      return Promise.resolve([]);
    });
    await render();

    await chooseQuality("Q4");
    await chooseQuality("bf16");
    expect(writes).toHaveLength(2);

    await act(async () => {
      writes[1].resolve({});
      await Promise.resolve();
    });
    await act(async () => {
      writes[0].reject(new Error("older disk failure"));
      await Promise.resolve();
    });

    expect(selectedQuality().textContent.trim()).toBe("bf16");
    expect(window.localStorage.getItem(STORAGE_KEY)).toBe("bf16");
    expect(container.textContent).toContain("Default generation quality set to High fidelity (bf16)");
    expect(container.textContent).not.toContain("older disk failure");
  });

  it("seeds the control from the instant-paint cache on mount (cache round-trip)", async () => {
    // App.jsx re-primes this cache from the durable server copy on launch (covered in App.shell.test.jsx);
    // here we assert the control reflects whatever the cache holds on mount.
    window.localStorage.setItem(STORAGE_KEY, "q4");
    await render();
    expect(selectedQuality().textContent.trim()).toBe("Q4");
  });
});

// The settings brought over from Simple mode (Settings-page handoff): theme, accent and the
// Simple-mode default all write through to the SAME stores their Simple-mode counterparts use.
describe("SettingsScreen appearance settings", () => {
  let container;
  let root;
  let SettingsScreen;
  let changeTheme;

  beforeEach(async () => {
    global.IS_REACT_ACT_ENVIRONMENT = true;
    delete window.__TAURI__;
    vi.resetModules();
    vi.doMock("../api.js", () => ({
      apiFetch: vi.fn(async () => []),
      isAbortError: () => false,
      API_BASE_URL: "",
      eventUrl: () => "",
    }));
    SettingsScreen = await loadScreen();
    changeTheme = vi.fn();
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

  async function render(props) {
    await renderScreen(root, SettingsScreen, props, {
      theme: "light",
      changeTheme,
      visibleWorkers: [],
      macCapabilities: {},
    });
  }

  it("drives the app-wide theme through changeTheme", async () => {
    await render({});
    const dark = [...container.querySelectorAll('[aria-label="Theme"] button')].find(
      (button) => button.textContent.trim() === "Dark",
    );
    await click(dark);
    expect(changeTheme).toHaveBeenCalledWith("dark");
  });

  it("reports the selected accent and forwards a change", async () => {
    const onAccentChange = vi.fn();
    await render({ accent: "coral", onAccentChange });
    const swatches = container.querySelectorAll('[aria-label="Accent"] button');
    expect(swatches).toHaveLength(7);
    expect(container.querySelector('[aria-label="Coral"]').getAttribute("aria-checked")).toBe("true");
    await click(container.querySelector('[aria-label="Violet"]'));
    expect(onAccentChange).toHaveBeenCalledWith("violet");
  });

  it("toggles the Simple-mode default", async () => {
    const onSimpleDefaultChange = vi.fn();
    await render({ simpleDefault: false, onSimpleDefaultChange });
    const toggle = container.querySelector('[aria-label="Use Simple mode by default"]');
    expect(toggle.getAttribute("aria-checked")).toBe("false");
    await click(toggle);
    expect(onSimpleDefaultChange).toHaveBeenCalledWith(true);
  });

  it("drops the sidebar-switch sentence when the viewport locks the app to Simple", async () => {
    await render({ lockedToSimple: true });
    expect(container.textContent).toContain("Phones always use it.");
    expect(container.textContent).not.toContain("the sidebar switch overrides it");
  });
});

describe("SettingsScreen embedded-workflow toggle (sc-15953)", () => {
  let container;
  let root;
  let SettingsScreen;

  beforeEach(async () => {
    global.IS_REACT_ACT_ENVIRONMENT = true;
    delete window.__TAURI__;
    vi.resetModules();
    vi.doMock("../api.js", () => ({
      apiFetch: vi.fn(async () => []),
      isAbortError: () => false,
      API_BASE_URL: "",
      eventUrl: () => "",
    }));
    SettingsScreen = await loadScreen();
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

  async function openSharing(props) {
    await renderScreen(root, SettingsScreen, props);
    await openTab(container, "Settings");
  }

  const toggle = () =>
    container.querySelector('[aria-label="Include the recipe in generated images"]');

  it("is discoverable under Sharing and reflects the preference", async () => {
    await openSharing({ embedWorkflow: true });
    expect(container.textContent).toContain("Sharing");
    expect(toggle().getAttribute("aria-checked")).toBe("true");
  });

  it("forwards a change in both directions", async () => {
    const onEmbedWorkflowChange = vi.fn();
    await openSharing({ embedWorkflow: true, onEmbedWorkflowChange });
    await click(toggle());
    expect(onEmbedWorkflowChange).toHaveBeenCalledWith(false);

    onEmbedWorkflowChange.mockClear();
    await openSharing({ embedWorkflow: false, onEmbedWorkflowChange });
    await click(toggle());
    expect(onEmbedWorkflowChange).toHaveBeenCalledWith(true);
  });

  it("names every path-exempt prose field rather than summarizing them", async () => {
    // The claim a user acts on before sharing. Rendered from the list that
    // `the_settings_copy_names_exactly_the_path_exempt_prose_fields` (Rust) pins against the
    // contract doc, so this asserts the UI actually shows what that list holds.
    const { EMBEDDED_PROSE_FIELDS } = await import("../workflowEmbed.js");
    await openSharing({ embedWorkflow: true });
    for (const [, label] of EMBEDDED_PROSE_FIELDS) {
      expect(container.textContent).toContain(label);
    }
  });

  it("does not imply that turning it off cleans images already on disk", async () => {
    await openSharing({ embedWorkflow: true });
    const copy = container.textContent;
    expect(copy).toContain("Turning this off applies from the next generation");
    expect(copy).toContain("Images already on disk keep the block they were written with");
    // The wording that would be a lie. Nothing rewrites an existing asset.
    expect(copy).not.toContain("removes it from");
    expect(copy).not.toMatch(/clean(s|ed)? (up )?(your )?existing/i);
  });

  it("links to the contract document", async () => {
    await openSharing({ embedWorkflow: true });
    const link = [...container.querySelectorAll("a")].find((anchor) =>
      anchor.href.includes("workflow-share-envelope.md"),
    );
    expect(link).toBeTruthy();
    expect(link.getAttribute("rel")).toContain("noopener");
  });
});
