// @vitest-environment node
// jsdom's File has no arrayBuffer(); Node's (like every shipped webview's) does.
import { describe, expect, it, vi } from "vitest";
import { saveJsonExport } from "./saveExportFile.js";

// sc-6554: JSON exports (prompt batches, film reference packs, production/compiled plans) take
// the same native save-dialog path as Image Editor exports on desktop, instead of a `blob:`
// `<a download>` whose filename WKWebView/WebKitGTK may ignore.
describe("saveJsonExport", () => {
  it("hands the shell the pretty-printed JSON bytes and the intended filename on desktop", async () => {
    const invoke = vi.fn(async () => "/Users/me/Desktop/plan.json");
    const saved = await saveJsonExport("plan.json", { shots: [1] }, { desktop: true, invoke });

    expect(saved).toBe("/Users/me/Desktop/plan.json");
    expect(invoke).toHaveBeenCalledTimes(1);
    const [command, args] = invoke.mock.calls[0];
    expect(command).toBe("save_image_export");
    expect(args.suggestedFilename).toBe("plan.json");
    expect(new TextDecoder().decode(new Uint8Array(args.imageBytes))).toBe(
      `${JSON.stringify({ shots: [1] }, null, 2)}\n`,
    );
  });

  it("resolves null when the user cancels the desktop dialog", async () => {
    const invoke = vi.fn(async () => null);
    await expect(saveJsonExport("plan.json", {}, { desktop: true, invoke })).resolves.toBeNull();
  });

  it("falls back to an <a download> of the named file in a browser", async () => {
    const anchor = { click: vi.fn(), remove: vi.fn() };
    const documentRef = {
      createElement: vi.fn(() => anchor),
      body: { appendChild: vi.fn() },
    };
    const urlApi = { createObjectURL: vi.fn(() => "blob:plan"), revokeObjectURL: vi.fn() };

    await saveJsonExport("plan.json", {}, { desktop: false, documentRef, urlApi });

    expect(anchor.download).toBe("plan.json");
    expect(anchor.href).toBe("blob:plan");
    expect(anchor.click).toHaveBeenCalledTimes(1);
    expect(urlApi.revokeObjectURL).toHaveBeenCalledWith("blob:plan");
  });
});
