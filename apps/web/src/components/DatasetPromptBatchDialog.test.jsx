import React, { act } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { click, mountRoot, settle, unmountRoot } from "../testUtils/dom.js";
import { AppContext } from "../context/AppContext.js";
import { VISION_CAPTION_MODEL_REPO } from "../constants.js";

vi.mock("../api/trainingResources.js", () => ({
  trainingApi: { defaultDescribePrompt: vi.fn(), batchDescribeJobs: vi.fn() },
}));
vi.mock("../api/jobsResources.js", () => ({ jobsApi: { item: vi.fn(), cancel: vi.fn() } }));
vi.mock("../pollJob.js", () => ({ abortableDelay: vi.fn(() => Promise.resolve()) }));
vi.mock("../saveExportFile.js", () => ({ saveExportFile: vi.fn(async () => null) }));

const { trainingApi } = await import("../api/trainingResources.js");
const { jobsApi } = await import("../api/jobsResources.js");
const { saveExportFile } = await import("../saveExportFile.js");
const { DatasetPromptBatchDialog } = await import("./DatasetPromptBatchDialog.jsx");

const DEFAULT_PROMPT = "You are a meticulous scene describer.\n## LABELS";
const dataset = {
  id: "ds_1",
  name: "Film prints",
  version: 3,
  items: [{ id: "item_a" }, { id: "item_b" }, { id: "item_c" }],
};
const installed = (id, name) => ({ id, name, installState: "installed" });

function described(itemId, width, height, overrides = {}) {
  return {
    itemId,
    imagePath: `images/${itemId}.jpg`,
    assetId: null,
    width,
    height,
    description: `A woman, woman_1, stands (${itemId}).`,
    style: "1960s color negative print",
    camera: "35mm SLR",
    composition: "Medium shot",
    lighting: "Window light",
    environment: "A kitchen",
    failed: false,
    error: null,
    ...overrides,
  };
}

function snapshot(status, items, extra = {}) {
  const failedCount = items.filter((item) => item.failed).length;
  return {
    id: "job_1",
    type: "dataset_batch_describe",
    status,
    progress: extra.progress ?? 0.5,
    result: {
      itemCount: extra.itemCount ?? 3,
      processedCount: items.length,
      describedCount: items.length - failedCount,
      failedCount,
      canceled: status === "canceled",
      items,
    },
  };
}

// jobsApi.item returns pending promises the test resolves one poll at a time.
let polls;
function nextPoll(value) {
  const resolve = polls.shift();
  if (!resolve) throw new Error("no poll pending");
  resolve(value);
}

function textarea(label) {
  return [...document.body.querySelectorAll("label")]
    .find((node) => node.firstChild?.textContent?.trim() === label)
    ?.querySelector("textarea, input");
}
function button(text) {
  return [...document.body.querySelectorAll("button")].find((node) => node.textContent.trim() === text);
}
function setText(element, value) {
  const proto = element instanceof window.HTMLTextAreaElement ? window.HTMLTextAreaElement : window.HTMLInputElement;
  Object.getOwnPropertyDescriptor(proto.prototype, "value").set.call(element, value);
  element.dispatchEvent(new window.Event("input", { bubbles: true }));
}

describe("DatasetPromptBatchDialog", () => {
  let container;
  let root;
  let context;

  beforeEach(() => {
    global.IS_REACT_ACT_ENVIRONMENT = true;
    ({ container, root } = mountRoot());
    polls = [];
    trainingApi.defaultDescribePrompt.mockReset().mockResolvedValue({ prompt: DEFAULT_PROMPT, maxChars: 32000 });
    trainingApi.batchDescribeJobs.mockReset().mockResolvedValue({ id: "job_1", status: "queued", progress: 0 });
    jobsApi.item.mockReset().mockImplementation(() => new Promise((resolve) => polls.push(resolve)));
    jobsApi.cancel.mockReset().mockResolvedValue({});
    saveExportFile.mockClear();
    context = {
      token: "t",
      activeProject: { id: "proj_1" },
      models: [
        installed("vision_caption_qwen3vl_8b", "Vision Captioner (Qwen3-VL-8B)"),
        { id: "joycaption_beta_one", name: "JoyCaption (beta one)", installState: "missing" },
      ],
      jobs: [],
      createModelDownloadJob: vi.fn(async () => ({ id: "dl_1" })),
      jobAction: vi.fn(async () => ({})),
      setActiveView: vi.fn(),
      createPromptBatch: vi.fn(async (payload) => ({ id: "film_prints_prompts", ...payload })),
      sendPromptBatchToImage: vi.fn(),
    };
  });

  afterEach(async () => {
    await unmountRoot(root, container);
  });

  async function render(props = {}) {
    await act(async () => {
      root.render(
        <AppContext.Provider value={context}>
          <DatasetPromptBatchDialog dataset={dataset} onClose={vi.fn()} renderThumbnail={(id) => <img alt={`thumb ${id}`} />} {...props} />
        </AppContext.Provider>,
      );
    });
    await settle();
  }

  async function runWith(items, status = "completed") {
    await click(button("Describe images"));
    await settle();
    await act(async () => nextPoll(snapshot(status, items, { progress: 1 })));
    await settle();
  }

  it("prefills the describing prompt with the default text before the first run", async () => {
    await render();
    expect(textarea("Describing prompt").value).toBe(DEFAULT_PROMPT);
    expect(button("Describe images").disabled).toBe(false);
  });

  it("shows a model that is not installed as unavailable with a download route", async () => {
    await render();
    const joy = document.body.querySelector('input[value="joycaption_beta_one"]');
    expect(joy.closest("label").textContent).toContain("not installed");
    await click(joy);
    expect(document.body.textContent).toContain("Describing the dataset needs a model");
    expect(button("Describe images").disabled).toBe(true);
    await click(button("Download"));
    expect(context.createModelDownloadJob).toHaveBeenCalledWith(expect.objectContaining({ id: "joycaption_beta_one" }));
    await click(button("Open the Model Manager"));
    expect(context.setActiveView).toHaveBeenCalledWith("Models");
  });

  it("sends no prompt for the default text and the edited text otherwise", async () => {
    await render();
    await runWith([described("item_a", 300, 400)]);
    expect(JSON.parse(trainingApi.batchDescribeJobs.mock.calls[0][3].body)).toEqual({
      model: VISION_CAPTION_MODEL_REPO,
      requestedGpu: "auto",
    });
    await act(async () => setText(textarea("Describing prompt"), "Describe only the clothing."));
    await click(button("Describe all again"));
    await settle();
    expect(JSON.parse(trainingApi.batchDescribeJobs.mock.calls[1][3].body).prompt).toBe("Describe only the clothing.");
  });

  it("shows per-image progress and keeps finished rows when canceled", async () => {
    await render();
    await click(button("Describe images"));
    await settle();
    await act(async () => nextPoll(snapshot("running", [described("item_a", 300, 400)], { progress: 0.37 })));
    await settle();
    expect(document.body.textContent).toContain("Describing: 1 of 3 images");
    expect(document.body.querySelector("progress").getAttribute("value")).toBe("37");
    expect(document.body.querySelectorAll(".dataset-prompt-batch-row")).toHaveLength(1);

    await click(button("Cancel"));
    expect(context.jobAction).toHaveBeenCalledWith(expect.objectContaining({ id: "job_1" }), "cancel");
    await act(async () => nextPoll(snapshot("canceled", [described("item_a", 300, 400), described("item_b", 400, 300)])));
    await settle();
    expect(document.body.querySelectorAll(".dataset-prompt-batch-row")).toHaveLength(2);
    expect(document.body.textContent).toContain("Canceled. Kept 2 finished images.");
    expect(button("Cancel")).toBeUndefined();
  });

  it("shows exact size, snapped ratio and a flagged row's error", async () => {
    await render();
    await runWith([
      described("item_a", 2400, 3200),
      described("item_b", null, null, { failed: true, error: "Unreadable image.", description: "" }),
    ]);
    const rows = document.body.querySelectorAll(".dataset-prompt-batch-row");
    expect(rows[0].textContent).toContain("2400×3200 · 3:4");
    expect(rows[0].querySelector("img").getAttribute("alt")).toBe("thumb item_a");
    expect(rows[1].classList.contains("failed")).toBe(true);
    expect(rows[1].textContent).toContain("Flagged: Unreadable image.");
    expect(document.body.textContent).toContain("1 flagged");
  });

  it("re-runs a single image and merges its result into that row", async () => {
    await render();
    await runWith([described("item_a", 300, 400), described("item_b", 400, 300)]);
    const rowB = () => document.body.querySelector('[data-item-id="item_b"]');
    await click([...rowB().querySelectorAll("button")].find((node) => node.textContent === "Re-run"));
    await settle();
    expect(JSON.parse(trainingApi.batchDescribeJobs.mock.calls[1][3].body).itemIds).toEqual(["item_b"]);
    await act(async () =>
      nextPoll(snapshot("completed", [described("item_b", 400, 300, { description: "A man, man_1, sits." })], { itemCount: 1 })),
    );
    await settle();
    const rows = document.body.querySelectorAll(".dataset-prompt-batch-row");
    expect([...rows].map((row) => row.dataset.itemId)).toEqual(["item_a", "item_b"]);
    expect(rowB().querySelector("textarea").value).toBe("A man, man_1, sits.");
    expect(document.body.querySelector('[data-item-id="item_a"] textarea').value).toContain("(item_a)");
  });

  it("saves the formatter blocks with ratio prefixes and pixel sizes, then opens Image Studio", async () => {
    await render();
    await runWith([
      described("item_a", 2400, 3200),
      described("item_b", 4000, 2250),
      described("item_c", null, null, { failed: true, error: "Bad image." }),
    ]);
    await act(async () => setText(document.body.querySelector('[data-item-id="item_a"] textarea'), "A girl, girl_1, waves."));
    await click(button("Save prompt batch"));
    await settle();
    const payload = context.createPromptBatch.mock.calls[0][0];
    expect(payload.name).toBe("Film prints prompts");
    expect(payload.prompts).toHaveLength(2);
    expect(payload.prompts[0]).toBe(
      "[3:4]\nDescription: A girl, {{girl_1}}, waves.\nStyle: 1960s color negative print\nCamera: 35mm SLR\n" +
        "Composition: Medium shot\nLighting: Window light\nEnvironment: A kitchen",
    );
    expect(payload.prompts[1].startsWith("[16:9]\nDescription: A woman, {{woman_1}}")).toBe(true);
    expect(payload.variables).toEqual([{ key: "girl_1", values: [] }, { key: "woman_1", values: [] }]);
    expect(payload.sourceImages).toEqual([
      { itemId: "item_a", width: 2400, height: 3200, ratio: "3:4" },
      { itemId: "item_b", width: 4000, height: 2250, ratio: "16:9" },
    ]);
    expect(payload.source).toMatchObject({ kind: "training_dataset", datasetId: "ds_1", datasetVersion: 3 });

    await click(button("Open in Image Studio"));
    expect(context.sendPromptBatchToImage).toHaveBeenCalledWith(expect.objectContaining({ id: "film_prints_prompts" }));
  });

  it("exports the batch as the standard prompt-batch .json", async () => {
    await render();
    await runWith([described("item_a", 300, 400)]);
    await click(button("Export .json"));
    const file = saveExportFile.mock.calls[0][0];
    expect(file.name).toBe("film-prints-prompts.json");
    const text = await new Promise((resolve) => {
      const reader = new window.FileReader();
      reader.onload = () => resolve(reader.result);
      reader.readAsText(file);
    });
    const exported = JSON.parse(text);
    expect(exported).toMatchObject({ sceneworksPromptBatch: 1, name: "Film prints prompts" });
    expect(exported.prompts).toHaveLength(1);
    expect(exported.prompts[0].startsWith("[3:4]\n")).toBe(true);
  });

  it("blocks a run while the dataset has unsaved changes", async () => {
    await render({ dirty: true });
    expect(document.body.textContent).toContain("Save the dataset first");
    expect(button("Describe images").disabled).toBe(true);
  });
});
