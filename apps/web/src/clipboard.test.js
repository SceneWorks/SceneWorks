import { describe, expect, it, vi } from "vitest";
import { writeClipboardText } from "./clipboard.js";

describe("writeClipboardText WebKit compatibility", () => {
  it("uses the asynchronous Clipboard API when available", async () => {
    const writeText = vi.fn().mockResolvedValue(undefined);
    await expect(
      writeClipboardText("hello", {
        clipboard: { writeText },
        documentRef: document,
      }),
    ).resolves.toBe(true);
    expect(writeText).toHaveBeenCalledWith("hello");
  });

  it("falls back to a selected textarea when WebKit denies clipboard access", async () => {
    const writeText = vi.fn().mockRejectedValue(new Error("denied"));
    const execCommand = vi.fn(() => true);
    const documentRef = Object.create(document);
    Object.defineProperty(documentRef, "body", { value: document.body });
    documentRef.createElement = document.createElement.bind(document);
    documentRef.execCommand = execCommand;

    await expect(
      writeClipboardText("fallback", {
        clipboard: { writeText },
        documentRef,
      }),
    ).resolves.toBe(true);
    expect(execCommand).toHaveBeenCalledWith("copy");
    expect(document.querySelector("textarea")).toBeNull();
  });

  it("reports failure when neither clipboard path is available", async () => {
    await expect(
      writeClipboardText("nope", {
        clipboard: null,
        documentRef: null,
      }),
    ).resolves.toBe(false);
  });
});
