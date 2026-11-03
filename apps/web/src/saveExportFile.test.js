// @vitest-environment node
// jsdom's File has no arrayBuffer(); Node's (like every shipped webview's) does.
import { describe, expect, it, vi } from "vitest";
import { saveJsonExport, saveExportFile } from "./saveExportFile.js";

// sc-6554: JSON exports (prompt batches, film reference packs, production/compiled plans) take
// the same native save-dialog path as Image Editor exports on desktop, instead of a `blob:`
// `<a download>` whose filename WKWebView/WebKitGTK may ignore.
describe("saveJsonExport", () => {
  it("hands the shell the pretty-printed JSON bytes and the intended filename on desktop", async () => {
    const invoke = vi.fn(async () => "/Users/me/Desktop/plan.json");
    const saved = await saveJsonExport("plan.json", { shots: [1] }, { desktop: true, invoke });

    expect(saved).toBe("/Users/me/Desktop/plan.json");
    expect(invoke).toHaveBeenCalledTimes(3);
    const [command, args] = invoke.mock.calls[0];
    expect(command).toBe("begin_image_export");
    expect(args.suggestedFilename).toBe("plan.json");
    const [chunkCommand, bytes, options] = invoke.mock.calls[1];
    expect(chunkCommand).toBe("save_image_export");
    expect(bytes).toBeInstanceOf(ArrayBuffer);
    expect(options.headers["x-sceneworks-export-id"]).toBe("/Users/me/Desktop/plan.json");
    expect(new TextDecoder().decode(bytes)).toBe(`${JSON.stringify({ shots: [1] }, null, 2)}\n`);
    expect(args.totalBytes).toBe(bytes.byteLength);

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
    await new Promise((resolve) => setTimeout(resolve, 0));
    expect(urlApi.revokeObjectURL).toHaveBeenCalledWith("blob:plan");
  });
});

it("rejects oversize exports before reading bytes or invoking native IPC", async () => {
  const invoke = vi.fn();
  const file = { size: 256 * 1024 * 1024 + 1, slice: vi.fn() };
  await expect(saveExportFile(file, { desktop: true, invoke })).rejects.toThrow("256 MB");
  expect(invoke).not.toHaveBeenCalled();
  expect(file.slice).not.toHaveBeenCalled();
});
it("streams binary chunks and cancels native staging on failure", async () => {
  const file = new File([new Uint8Array(1024 * 1024 + 7)], "large.png");
  const invoke = vi.fn(async (command) => command === "begin_image_export" ? "id" : "/saved");
  await saveExportFile(file, { desktop: true, invoke });
  const chunks = invoke.mock.calls.filter(([command]) => command === "save_image_export");
  expect(chunks.map(([, bytes]) => bytes.byteLength)).toEqual([1024 * 1024, 7]);
  const failure = new Error("disk full");
  const failing = vi.fn(async (command) => {
    if (command === "begin_image_export") return "failed";
    if (command === "save_image_export") throw failure;
  });
  await expect(saveExportFile(file, { desktop: true, invoke: failing })).rejects.toBe(failure);
  expect(failing).toHaveBeenLastCalledWith("cancel_image_export", { exportId: "failed" });
});
