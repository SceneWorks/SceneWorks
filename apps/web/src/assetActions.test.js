import { afterEach, describe, expect, it, vi } from "vitest";

// assetActions derives `isDesktop` from window.__TAURI__ at module load (via
// runtime.js), so — like SettingsScreen.test.jsx — we set/delete the Tauri bridge
// and re-import the module fresh in each test to exercise both branches.
const IMAGE_ASSET = {
  id: "asset_img",
  type: "image",
  displayName: "Mira",
  projectId: "project_1",
  file: { path: "assets/images/Mira.png", mimeType: "image/png" },
};
const VIDEO_ASSET = {
  id: "asset_vid",
  type: "video",
  displayName: "Clip",
  projectId: "project_1",
  file: { path: "assets/videos/Clip.mp4", mimeType: "video/mp4" },
};

async function importDesktop(invoke) {
  window.__TAURI__ = { core: { invoke } };
  vi.resetModules();
  return import("./assetActions.js");
}

async function importBrowser() {
  delete window.__TAURI__;
  vi.resetModules();
  return import("./assetActions.js");
}

afterEach(() => {
  delete window.__TAURI__;
  vi.restoreAllMocks();
  // The last-save-dir feature (sc-8737) persists to localStorage; clear it so tests
  // that assert "no startDir before any save" aren't contaminated by an earlier save.
  try {
    globalThis.localStorage?.clear();
  } catch {
    // no-op
  }
});

describe("suggestedFilename (sc-8727)", () => {
  it("keeps the display name when it already has the right extension", async () => {
    const { suggestedFilename } = await importBrowser();
    expect(suggestedFilename({ displayName: "Mira.png", file: { path: "a/Mira.png", mimeType: "image/png" } })).toBe("Mira.png");
  });

  it("appends the source-path extension when the display name has none (image)", async () => {
    const { suggestedFilename } = await importBrowser();
    expect(suggestedFilename(IMAGE_ASSET)).toBe("Mira.png");
  });

  it("appends the source-path extension for a video asset", async () => {
    const { suggestedFilename } = await importBrowser();
    expect(suggestedFilename(VIDEO_ASSET)).toBe("Clip.mp4");
  });

  it("derives the extension from the mime type when the path has none", async () => {
    const { suggestedFilename } = await importBrowser();
    const asset = { displayName: "shot", file: { path: "assets/images/shot", mimeType: "image/jpeg" } };
    expect(suggestedFilename(asset)).toBe("shot.jpg");
  });

  it("falls back to the id, then a generic name, and leaves off an unknown extension", async () => {
    const { suggestedFilename } = await importBrowser();
    expect(suggestedFilename({ id: "asset_9", file: {} })).toBe("asset_9");
    expect(suggestedFilename({})).toBe("asset");
  });
});

describe("saveAssetAs — desktop (sc-8727)", () => {
  it("resolves the path then saves with the suggested filename (image)", async () => {
    const invoke = vi.fn(async (command) => {
      if (command === "resolve_asset_path") return "/data/project_1/assets/images/Mira.png";
      if (command === "save_asset_as") return "/Users/me/Desktop/Mira.png";
      return null;
    });
    const { saveAssetAs } = await importDesktop(invoke);

    const dest = await saveAssetAs(IMAGE_ASSET);

    expect(invoke).toHaveBeenNthCalledWith(1, "resolve_asset_path", {
      projectId: "project_1",
      relativePath: "assets/images/Mira.png",
    });
    expect(invoke).toHaveBeenNthCalledWith(2, "save_asset_as", {
      sourcePath: "/data/project_1/assets/images/Mira.png",
      suggestedFilename: "Mira.png",
    });
    expect(dest).toBe("/Users/me/Desktop/Mira.png");
  });

  it("works for a video asset too", async () => {
    const invoke = vi.fn(async (command) => {
      if (command === "resolve_asset_path") return "/data/project_1/assets/videos/Clip.mp4";
      if (command === "save_asset_as") return "/Users/me/Desktop/Clip.mp4";
      return null;
    });
    const { saveAssetAs } = await importDesktop(invoke);

    await saveAssetAs(VIDEO_ASSET);

    expect(invoke).toHaveBeenNthCalledWith(2, "save_asset_as", {
      sourcePath: "/data/project_1/assets/videos/Clip.mp4",
      suggestedFilename: "Clip.mp4",
    });
  });

  it("returns null quietly when the user cancels the save dialog", async () => {
    const invoke = vi.fn(async (command) => {
      if (command === "resolve_asset_path") return "/data/project_1/assets/images/Mira.png";
      return null; // save_asset_as returns null on cancel
    });
    const { saveAssetAs } = await importDesktop(invoke);

    await expect(saveAssetAs(IMAGE_ASSET)).resolves.toBeNull();
  });
});

describe("saveAssetAs — last-used save directory (sc-8737)", () => {
  // A save_asset_as mock that records the args each invocation received, so we can
  // assert whether/what startDir was passed across successive saves.
  function invokeRecording(destination) {
    const saveCalls = [];
    const invoke = vi.fn(async (command, args) => {
      if (command === "resolve_asset_path") return "/data/project_1/assets/images/Mira.png";
      if (command === "save_asset_as") {
        saveCalls.push(args);
        return destination;
      }
      return null;
    });
    return { invoke, saveCalls };
  }

  it("passes no startDir before any save has succeeded", async () => {
    const { invoke, saveCalls } = invokeRecording("/Users/me/Pictures/Mira.png");
    const { saveAssetAs } = await importDesktop(invoke);

    await saveAssetAs(IMAGE_ASSET);

    expect(saveCalls).toHaveLength(1);
    expect(saveCalls[0]).not.toHaveProperty("startDir");
  });

  it("opens the next Save As in the previously-saved POSIX directory", async () => {
    const { invoke, saveCalls } = invokeRecording("/Users/me/Pictures/Mira.png");
    const { saveAssetAs } = await importDesktop(invoke);

    await saveAssetAs(IMAGE_ASSET); // first save records /Users/me/Pictures
    await saveAssetAs(VIDEO_ASSET); // second save should be seeded with it

    expect(saveCalls).toHaveLength(2);
    expect(saveCalls[0]).not.toHaveProperty("startDir");
    expect(saveCalls[1].startDir).toBe("/Users/me/Pictures");
  });

  it("handles a Windows destination path (backslash separators)", async () => {
    const { invoke, saveCalls } = invokeRecording("C:\\Users\\me\\Pictures\\Mira.png");
    const { saveAssetAs } = await importDesktop(invoke);

    await saveAssetAs(IMAGE_ASSET);
    await saveAssetAs(VIDEO_ASSET);

    expect(saveCalls[1].startDir).toBe("C:\\Users\\me\\Pictures");
  });

  it("persists the last directory across a module reload via localStorage", async () => {
    const first = invokeRecording("/Users/me/Exports/Mira.png");
    const firstMod = await importDesktop(first.invoke);
    await firstMod.saveAssetAs(IMAGE_ASSET);

    // Re-import fresh (as a page reload would) — the module-level fallback is gone,
    // but localStorage should still carry the directory.
    const second = invokeRecording("/Users/me/Exports/Clip.mp4");
    const secondMod = await importDesktop(second.invoke);
    await secondMod.saveAssetAs(VIDEO_ASSET);

    expect(second.saveCalls[0].startDir).toBe("/Users/me/Exports");
  });

  it("does not persist a startDir when the save is cancelled", async () => {
    const invoke = vi.fn(async (command) => {
      if (command === "resolve_asset_path") return "/data/project_1/assets/images/Mira.png";
      return null; // cancelled
    });
    const { saveAssetAs } = await importDesktop(invoke);

    await saveAssetAs(IMAGE_ASSET); // cancelled → nothing remembered

    // A subsequent save still has no startDir.
    const { invoke: invoke2, saveCalls } = invokeRecording("/Users/me/Desktop/Clip.mp4");
    const { saveAssetAs: saveAssetAs2 } = await importDesktop(invoke2);
    await saveAssetAs2(VIDEO_ASSET);
    expect(saveCalls[0]).not.toHaveProperty("startDir");
  });

  it("falls back to an in-memory value when localStorage is unavailable", async () => {
    const original = Object.getOwnPropertyDescriptor(globalThis, "localStorage");
    Object.defineProperty(globalThis, "localStorage", {
      configurable: true,
      get() {
        throw new Error("localStorage disabled");
      },
    });
    try {
      const { invoke, saveCalls } = invokeRecording("/Users/me/Pictures/Mira.png");
      const { saveAssetAs } = await importDesktop(invoke);
      await saveAssetAs(IMAGE_ASSET);
      await saveAssetAs(VIDEO_ASSET);
      // Storage threw on both read and write; the module-level fallback still carries it.
      expect(saveCalls[1].startDir).toBe("/Users/me/Pictures");
    } finally {
      if (original) {
        Object.defineProperty(globalThis, "localStorage", original);
      }
    }
  });
});

describe("saveAssetAs — browser (sc-8727)", () => {
  it("triggers an <a download> and never calls invoke", async () => {
    const { saveAssetAs } = await importBrowser();

    const clicked = [];
    const clickSpy = vi
      .spyOn(window.HTMLAnchorElement.prototype, "click")
      .mockImplementation(function mockClick() {
        clicked.push({ href: this.href, download: this.download });
      });

    const dest = await saveAssetAs(IMAGE_ASSET);

    expect(clickSpy).toHaveBeenCalledTimes(1);
    expect(clicked[0].download).toBe("Mira.png");
    expect(clicked[0].href).toContain(
      "/api/v1/projects/project_1/files/assets/images/Mira.png",
    );
    expect(dest).toBeNull();
    // No anchor left dangling in the DOM.
    expect(document.querySelector("a[download]")).toBeNull();
  });

  it("works for a video asset in the browser", async () => {
    const { saveAssetAs } = await importBrowser();
    const clicked = [];
    vi.spyOn(window.HTMLAnchorElement.prototype, "click").mockImplementation(function mockClick() {
      clicked.push({ href: this.href, download: this.download });
    });

    await saveAssetAs(VIDEO_ASSET);

    expect(clicked[0].download).toBe("Clip.mp4");
    expect(clicked[0].href).toContain("/api/v1/projects/project_1/files/assets/videos/Clip.mp4");
  });
});

describe("revealAsset (sc-8727)", () => {
  it("resolves the path then reveals it in the OS (desktop)", async () => {
    const invoke = vi.fn(async (command) => {
      if (command === "resolve_asset_path") return "/data/project_1/assets/images/Mira.png";
      return null;
    });
    const { revealAsset } = await importDesktop(invoke);

    await revealAsset(IMAGE_ASSET);

    expect(invoke).toHaveBeenNthCalledWith(1, "resolve_asset_path", {
      projectId: "project_1",
      relativePath: "assets/images/Mira.png",
    });
    expect(invoke).toHaveBeenNthCalledWith(2, "reveal_in_os", {
      path: "/data/project_1/assets/images/Mira.png",
    });
  });

  it("throws in browser mode (desktop-only)", async () => {
    const { revealAsset } = await importBrowser();
    await expect(revealAsset(IMAGE_ASSET)).rejects.toThrow(/desktop app/i);
  });
});
