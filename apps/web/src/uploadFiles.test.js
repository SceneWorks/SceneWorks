import { describe, expect, it, vi } from "vitest";
import { isImageUpload, uploadFiles } from "./uploadFiles.js";

describe("uploadFiles", () => {
  it("bounds concurrency, reports progress and keeps per-file failures", async () => {
    let active = 0;
    let maximum = 0;
    const progress = vi.fn();
    const files = ["a", "b", "c", "d"].map((name) => ({ name }));
    const results = await uploadFiles(files, async (file) => {
      maximum = Math.max(maximum, ++active);
      await Promise.resolve();
      active--;
      if (file.name === "b") throw new Error("failed");
      return { id: file.name };
    }, { concurrency: 2, onProgress: progress });
    expect(maximum).toBe(2);
    expect(results.map((item) => item.value?.id ?? item.error.message)).toEqual(["a", "failed", "c", "d"]);
    expect(progress).toHaveBeenLastCalledWith(4, 4);
  });
  it("does not schedule or report after abort", async () => {
    const controller = new AbortController();
    const progress = vi.fn();
    const upload = vi.fn(async () => { controller.abort(); return { id: "a" }; });
    await uploadFiles([{ name: "a" }, { name: "b" }], upload, { signal: controller.signal, concurrency: 1, onProgress: progress });
    expect(upload).toHaveBeenCalledTimes(1);
    expect(progress).not.toHaveBeenCalled();
    expect(isImageUpload({ name: "PHOTO.JPG", type: "" })).toBe(true);
  });
});
