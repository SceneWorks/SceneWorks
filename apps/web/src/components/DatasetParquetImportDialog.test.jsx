import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { DatasetParquetImportDialog } from "./DatasetParquetImportDialog.jsx";

function fieldByLabel(text, selector = "input") {
  const label = [...document.body.querySelectorAll("label")].find((candidate) =>
    candidate.textContent.includes(text),
  );
  return label?.querySelector(selector);
}

function change(element, value) {
  act(() => {
    const setter = Object.getOwnPropertyDescriptor(
      Object.getPrototypeOf(element),
      "value",
    )?.set;
    setter.call(element, value);
    element.dispatchEvent(new Event("input", { bubbles: true }));
    element.dispatchEvent(new Event("change", { bubbles: true }));
  });
}

describe("DatasetParquetImportDialog", () => {
  let container;
  let root;

  beforeEach(() => {
    global.IS_REACT_ACT_ENVIRONMENT = true;
    container = document.createElement("div");
    document.body.appendChild(container);
  });

  afterEach(() => {
    act(() => root?.unmount());
    container.remove();
    vi.restoreAllMocks();
  });

  it("turns a caption category preset into include/exclude terms", async () => {
    const onRun = vi.fn(async () => ({ id: "job_1" }));
    root = createRoot(container);
    act(() =>
      root.render(
        <DatasetParquetImportDialog onClose={vi.fn()} onRun={onRun} />,
      ),
    );

    change(fieldByLabel("Parquet file or folder"), "D:\\laion2B-en-aesthetic-metadata");
    change(fieldByLabel("Caption category preset", "select"), "faces");
    change(fieldByLabel("Exclude any caption term"), "logo, illustration");
    expect(fieldByLabel("Images to import").max).toBe("100000");
    change(fieldByLabel("Images to import"), "100000");
    await act(async () => {
      document.body.querySelector('button[type="submit"]').click();
    });

    expect(onRun).toHaveBeenCalledTimes(1);
    expect(onRun.mock.calls[0][0]).toMatchObject({
      sourcePath: "D:\\laion2B-en-aesthetic-metadata",
      urlColumn: "URL",
      captionColumn: "TEXT",
      maxItems: 100000,
      minEdge: 512,
      concurrency: 16,
      captionIncludes: ["face", "facial", "portrait", "headshot", "selfie", "close-up", "close up"],
      captionExcludes: ["logo", "illustration"],
    });
  });
});
