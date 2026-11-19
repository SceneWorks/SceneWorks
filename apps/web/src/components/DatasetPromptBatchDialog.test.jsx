import React, { act } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { click, mountRoot, settle, unmountRoot } from "../testUtils/dom.js";
import { AppContext } from "../context/AppContext.js";
import { VISION_CAPTION_MODEL_REPO } from "../constants.js";
import { formatTierBytes } from "../formatting.js";

vi.mock("../api/trainingResources.js", () => ({
  trainingApi: {
    defaultDescribePrompt: vi.fn(),
    batchDescribeJobs: vi.fn(),
    listBatchDescribeJobs: vi.fn(),
    describePromptPresets: vi.fn(),
    describePromptPreset: vi.fn(),
  },
}));
vi.mock("../api/jobsResources.js", () => ({ jobsApi: { item: vi.fn(), cancel: vi.fn(), resultItems: vi.fn() } }));
vi.mock("../pollJob.js", () => ({ abortableDelay: vi.fn(() => Promise.resolve()) }));
vi.mock("../saveExportFile.js", () => ({ saveExportFile: vi.fn(async () => null) }));
vi.mock("../appConfirm.jsx", () => ({ appConfirm: vi.fn(async () => true) }));

const { trainingApi } = await import("../api/trainingResources.js");
const { jobsApi } = await import("../api/jobsResources.js");
const { abortableDelay } = await import("../pollJob.js");
const { saveExportFile } = await import("../saveExportFile.js");
const { appConfirm } = await import("../appConfirm.jsx");
const { DESCRIBE_MODELS, DatasetPromptBatchDialog, RESTORE_JOBS_PAGE, rowsNotFitted, VISION_DESCRIBE_MODEL_REPO } = await import("./DatasetPromptBatchDialog.jsx");
const { default: candidateCatalog } = await import("../testUtils/catalogBaseline/candidate-catalog.json");
const { readFileSync } = await import("node:fs");

// The dialog's 27B tier radios must match what the catalog and the shared contract offer: the same
// tier keys in the same order (core QWEN38_27B_DESCRIBER_TIERS, every one a catalog row); sizes come
// from the catalog at render time, never from a label.
describe("Qwen3.8 27B tier choices", () => {
  it("match the core tiers and every tier has a catalog row", () => {
    const core = readFileSync(`${process.cwd()}/../../crates/sceneworks-core/src/dataset_batch_describe.rs`, "utf8");
    const coreTiers = [...core.match(/QWEN38_27B_DESCRIBER_TIERS: \[&str; \d+\] = \[([^\]]*)\]/)[1].matchAll(/"([a-z0-9]+)"/g)].map((m) => m[1]);
    const entry = candidateCatalog.models.find((model) => model.id === "vision_describe_qwen3_8_27b");
    const tiers = DESCRIBE_MODELS.find((model) => model.id === "vision_describe_qwen3_8_27b").tiers;
    expect(tiers.map(([key]) => key)).toEqual(coreTiers);
    for (const [key, label] of tiers) {
      expect(entry.downloads.some((row) => row.variant === key), key).toBe(true);
      // Sizes are never written into the label: the dialog prints each tier's catalog size.
      expect(label, key).not.toMatch(/GB|GiB|\(/);
    }
  });
});
const { formatPromptBatch } = await import("../promptBatchDescribe.js");

const DEFAULT_PROMPT = "You are a meticulous scene describer.\n## LABELS";
const JOY_REPO = "fancyfeast/llama-joycaption-beta-one-hf-llava";
const JOY_DEFAULT_PROMPT = "You are a meticulous scene describer. Describe only what is visible.\nLABELS: ...";
const DEFAULTS = {
  "sensenova/SenseNova-Loop-9B": { prompt: DEFAULT_PROMPT, maxChars: 16000 },
  [VISION_CAPTION_MODEL_REPO]: { prompt: DEFAULT_PROMPT, maxChars: 32000 },
  [JOY_REPO]: { prompt: JOY_DEFAULT_PROMPT, maxChars: 3000 },
  [VISION_DESCRIBE_MODEL_REPO]: { prompt: DEFAULT_PROMPT, maxChars: 32000 },
};
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
      model: extra.model ?? VISION_CAPTION_MODEL_REPO,
      ...(extra.variant ? { variant: extra.variant } : {}),
      items,
    },
  };
}

// jobsApi.item returns pending promises the test resolves one poll at a time. Like the server, a
// snapshot's result carries only counts: its `items` become the job's stored items, which
// jobsApi.resultItems pages back out.
let polls;
let presetStore;
let storedItems;
function nextPoll(value) {
  const resolve = polls.shift();
  if (!resolve) throw new Error("no poll pending");
  if (Array.isArray(value?.result?.items)) {
    storedItems = value.result.items;
    const { items: _items, ...result } = value.result;
    resolve({ ...value, result });
  } else {
    resolve(value);
  }
}
function failPoll(error = new Error("network down")) {
  const resolve = polls.shift();
  if (!resolve) throw new Error("no poll pending");
  resolve(Promise.reject(error));
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

// The dataset's describe history, newest first, served like the API: `unfinished` keeps jobs not
// yet in a terminal status; `offset`/`limit` page it.
const TERMINAL = new Set(["completed", "failed", "canceled", "interrupted"]);
function serveHistory(jobs) {
  trainingApi.listBatchDescribeJobs.mockImplementation(async (_p, _d, _t, { offset = 0, limit, unfinished = false } = {}) => {
    const listed = unfinished ? jobs.filter((job) => !TERMINAL.has(job.status)) : jobs;
    return listed.slice(offset, limit ? offset + limit : undefined);
  });
}

describe("DatasetPromptBatchDialog", () => {
  let container;
  let root;
  let context;

  beforeEach(() => {
    global.IS_REACT_ACT_ENVIRONMENT = true;
    ({ container, root } = mountRoot());
    polls = [];
    trainingApi.defaultDescribePrompt
      .mockReset()
      .mockImplementation(async (_token, _options, model) => ({ model, ...DEFAULTS[model || VISION_CAPTION_MODEL_REPO] }));
    trainingApi.batchDescribeJobs.mockReset().mockResolvedValue({ id: "job_1", status: "queued", progress: 0 });
    // No earlier runs unless a test lists some.
    trainingApi.listBatchDescribeJobs.mockReset();
    serveHistory([]);
    // A tiny in-memory preset store standing in for the server catalog.
    presetStore = [{ id: "default", name: "Default", prompt: DEFAULT_PROMPT, scope: "builtin", builtin: true }];
    trainingApi.describePromptPresets.mockReset().mockImplementation(async (_token, options = {}, model = "") => {
      if (options.method === "POST") {
        const body = JSON.parse(options.body);
        const created = { id: body.name.toLowerCase().replace(/[^a-z0-9]+/g, "_"), scope: "global", ...body };
        presetStore = [...presetStore, created];
        return created;
      }
      // The server's built-in default is the selected model's default instruction.
      return presetStore.map((preset) =>
        preset.builtin ? { ...preset, prompt: DEFAULTS[model || VISION_CAPTION_MODEL_REPO].prompt } : preset,
      );
    });
    trainingApi.describePromptPreset.mockReset().mockImplementation(async (id) => {
      presetStore = presetStore.filter((preset) => preset.id !== id);
      return {};
    });
    jobsApi.item.mockReset().mockImplementation(() => new Promise((resolve) => polls.push(resolve)));
    jobsApi.cancel.mockReset().mockResolvedValue({});
    storedItems = [];
    jobsApi.resultItems.mockReset().mockImplementation(async (_jobId, _token, { offset = 0 } = {}) => ({
      items: storedItems.slice(offset),
      total: storedItems.length,
      nextOffset: null,
    }));
    abortableDelay.mockClear();
    saveExportFile.mockClear();
    appConfirm.mockReset().mockResolvedValue(true);
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
      promptBatches: [],
      createPromptBatch: vi.fn(async (payload) => ({ id: "film_prints_prompts", ...payload })),
      updatePromptBatch: vi.fn(async (id, payload) => ({ id, ...payload })),
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
      { itemId: "item_a", width: 2400, height: 3200, ratio: "3:4", model: VISION_CAPTION_MODEL_REPO },
      { itemId: "item_b", width: 4000, height: 2250, ratio: "16:9", model: VISION_CAPTION_MODEL_REPO },
    ]);
    expect(payload.source).toMatchObject({
      kind: "training_dataset",
      datasetId: "ds_1",
      datasetVersion: 3,
      model: VISION_CAPTION_MODEL_REPO,
    });

    await click(button("Open in Image Studio"));
    expect(context.sendPromptBatchToImage).toHaveBeenCalledWith(expect.objectContaining({ id: "film_prints_prompts" }));
  });

  it("shows a Pose field after Description and saves one Pose line per person; a six-field result has an empty pose", async () => {
    await render();
    const pose =
      "woman_1 — Body line: upright; Stance: standing; Facing: front\nman_1 — Body line: horizontal; Stance: lying on their back; Facing: up";
    await runWith([
      described("item_a", 2400, 3200, { pose }),
      // Stored before pose existed: no pose key at all.
      described("item_b", 4000, 2250),
    ]);
    const labels = (itemId) =>
      [...document.body.querySelectorAll(`[data-item-id="${itemId}"] label`)].map((node) => node.firstChild.textContent);
    expect(labels("item_a").slice(0, 3)).toEqual(["Description", "Pose", "Style"]);
    const textareas = (itemId) => [...document.body.querySelectorAll(`[data-item-id="${itemId}"] textarea`)];
    expect(textareas("item_a")[1].value).toBe(pose);
    expect(textareas("item_b")[1].value).toBe("");
    await click(button("Save prompt batch"));
    await settle();
    const payload = context.createPromptBatch.mock.calls[0][0];
    expect(payload.prompts[0]).toBe(
      "[3:4]\nDescription: A woman, {{woman_1}}, stands (item_a).\n" +
        "Pose: {{woman_1}} — Body line: upright; Stance: standing; Facing: front\n" +
        "Pose: {{man_1}} — Body line: horizontal; Stance: lying on their back; Facing: up\n" +
        "Style: 1960s color negative print\nCamera: 35mm SLR\nComposition: Medium shot\nLighting: Window light\nEnvironment: A kitchen",
    );
    expect(payload.prompts[1]).not.toContain("Pose:");
    expect(payload.variables.map((variable) => variable.key)).toEqual(["woman_1", "man_1"]);
  });

  it("shows a cut-off notice on a described row and keeps the row in the saved batch", async () => {
    await render();
    const notice =
      "The reply reached the length limit: the pose covers only the first 2 people in this image; the other fields are complete.";
    await runWith([described("item_a", 2400, 3200, { pose: "woman_1 — Body line: upright", notice }), described("item_b", 4000, 2250)]);
    const rowNote = (itemId, kind) => document.body.querySelector(`[data-item-id="${itemId}"] [data-row-note="${kind}"]`);
    expect(rowNote("item_a", "cut-off").textContent).toBe(notice);
    expect(rowNote("item_b", "cut-off")).toBeNull();
    await click(button("Save prompt batch"));
    await settle();
    expect(context.createPromptBatch.mock.calls[0][0].prompts).toHaveLength(2);
  });

  it("warns, without blocking Save, on a row whose prompt is over the generation limit", async () => {
    await render();
    await runWith([described("item_a", 2400, 3200, { environment: "x".repeat(3900) }), described("item_b", 4000, 2250)]);
    const tooLong = (itemId) => document.body.querySelector(`[data-item-id="${itemId}"] [data-row-note="too-long"]`);
    expect(tooLong("item_a").textContent).toMatch(/^This prompt is 4,\d{3} characters; image generation accepts at most 4,000/);
    expect(tooLong("item_b")).toBeNull();
    // Editing the row under the limit clears the warning.
    const environment = [...document.body.querySelectorAll('[data-item-id="item_a"] textarea')].at(-1);
    await act(async () => setText(environment, "A kitchen"));
    expect(tooLong("item_a")).toBeNull();
    await act(async () => setText(environment, "y".repeat(3950)));
    expect(button("Save prompt batch").disabled).toBe(false);
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

  it("reads only the new stored items on each poll, in pages, and never from the job result (SW-7082)", async () => {
    await render();
    await click(button("Describe images"));
    await settle();
    await act(async () => nextPoll(snapshot("running", [described("item_a", 300, 400)])));
    await settle();
    expect(jobsApi.resultItems).toHaveBeenLastCalledWith("job_1", "t", { offset: 0 }, expect.anything());
    // The next page comes back in two server pages.
    jobsApi.resultItems.mockImplementationOnce(async () => ({ items: [described("item_b", 400, 300)], total: 3, nextOffset: 2 }));
    await act(async () =>
      nextPoll(snapshot("completed", [described("item_a", 300, 400), described("item_b", 400, 300), described("item_c", 300, 300)])),
    );
    await settle();
    const offsets = jobsApi.resultItems.mock.calls.map((call) => call[2].offset);
    expect(offsets).toEqual([0, 1, 2]);
    const rows = document.body.querySelectorAll(".dataset-prompt-batch-row");
    expect([...rows].map((row) => row.dataset.itemId)).toEqual(["item_a", "item_b", "item_c"]);
  });

  it("retries a failed poll with a bounded backoff and keeps going (SW-7085 review)", async () => {
    await render();
    await click(button("Describe images"));
    await settle();
    await act(async () => failPoll());
    await settle();
    await act(async () => failPoll());
    await settle();
    expect(abortableDelay.mock.calls.map((call) => call[0])).toEqual(expect.arrayContaining([1000, 2000]));
    await act(async () => nextPoll(snapshot("completed", [described("item_a", 300, 400)], { progress: 1 })));
    await settle();
    expect(document.body.querySelectorAll(".dataset-prompt-batch-row")).toHaveLength(1);
    expect(document.body.textContent).toContain("Described: 1 of 3 images");
    expect(button("Describe all again").disabled).toBe(false);
    expect(document.body.textContent).not.toContain("Lost contact");
  });

  it("stops monitoring after the last retry without freezing, then resumes the same job (SW-7085 review)", async () => {
    await render();
    await click(button("Describe images"));
    await settle();
    await act(async () => nextPoll(snapshot("running", [described("item_a", 300, 400)])));
    await settle();
    // Every read fails: the first try and each retry.
    for (let attempt = 0; attempt <= 4; attempt += 1) {
      await act(async () => failPoll());
      await settle();
    }
    expect(polls).toHaveLength(0);
    expect(document.body.textContent).toContain("Lost contact with the describe job");
    expect(document.body.querySelectorAll(".dataset-prompt-batch-row")).toHaveLength(1);
    // The job may still be running, so no second job can start, but the dialog is not stuck.
    expect(button("Describing…").disabled).toBe(true);
    expect(button("Cancel")).toBeDefined();

    await click(button("Resume monitoring"));
    await settle();
    expect(polls).toHaveLength(1);
    expect(jobsApi.item).toHaveBeenLastCalledWith("job_1", "t", expect.anything());
    await act(async () =>
      nextPoll(snapshot("completed", [described("item_a", 300, 400), described("item_b", 400, 300)], { progress: 1 })),
    );
    await settle();
    expect(document.body.querySelectorAll(".dataset-prompt-batch-row")).toHaveLength(2);
    expect(jobsApi.resultItems.mock.calls.at(-1)[2].offset).toBe(1);
    expect(document.body.textContent).not.toContain("Lost contact");
    expect(button("Describe all again").disabled).toBe(false);
    expect(trainingApi.batchDescribeJobs).toHaveBeenCalledTimes(1);
  });

  it("cancels a job whose monitoring stopped and resumes monitoring to collect the outcome", async () => {
    await render();
    await click(button("Describe images"));
    await settle();
    for (let attempt = 0; attempt <= 4; attempt += 1) {
      await act(async () => failPoll());
      await settle();
    }
    expect(document.body.textContent).toContain("Lost contact");
    await click(button("Cancel"));
    await settle();
    expect(context.jobAction).toHaveBeenCalledWith(expect.objectContaining({ id: "job_1" }), "cancel");
    await act(async () => nextPoll(snapshot("canceled", [described("item_a", 300, 400)])));
    await settle();
    expect(document.body.textContent).toContain("Canceled. Kept 1 finished image.");
    expect(button("Cancel")).toBeUndefined();
  });

  it("starts ONE job when Describe is clicked twice before the request returns (SW-7085 review)", async () => {
    let created;
    trainingApi.batchDescribeJobs.mockReset().mockImplementation(
      () => new Promise((resolve) => { created = resolve; }),
    );
    await render();
    const describe = button("Describe images");
    await act(async () => {
      describe.click();
      describe.click();
    });
    await settle();
    expect(trainingApi.batchDescribeJobs).toHaveBeenCalledTimes(1);
    expect(button("Describe images").disabled).toBe(true);
    await act(async () => created({ id: "job_1", status: "queued", progress: 0 }));
    await settle();
    await act(async () => nextPoll(snapshot("completed", [described("item_a", 300, 400), described("item_b", 400, 300)])));
    await settle();
    // Every run control is blocked while a start is in flight: the per-row re-run and describe-remaining too.
    trainingApi.batchDescribeJobs.mockReset().mockImplementation(
      () => new Promise((resolve) => { created = resolve; }),
    );
    const rerun = [...document.body.querySelector('[data-item-id="item_b"]').querySelectorAll("button")]
      .find((node) => node.textContent === "Re-run");
    const remaining = button("Describe remaining (1)");
    await act(async () => {
      rerun.click();
      remaining.click();
      button("Describe all again").click();
    });
    await settle();
    expect(trainingApi.batchDescribeJobs).toHaveBeenCalledTimes(1);
    expect(JSON.parse(trainingApi.batchDescribeJobs.mock.calls[0][3].body).itemIds).toEqual(["item_b"]);
    expect(button("Describe remaining (1)").disabled).toBe(true);
    expect(button("Describe all again").disabled).toBe(true);
  });

  function presetPicker() {
    return [...document.body.querySelectorAll("label")]
      .find((node) => node.firstChild?.textContent?.trim() === "Saved prompt")
      ?.querySelector("select");
  }
  async function choosePreset(id) {
    const select = presetPicker();
    await act(async () => {
      Object.getOwnPropertyDescriptor(window.HTMLSelectElement.prototype, "value").set.call(select, id);
      select.dispatchEvent(new window.Event("change", { bubbles: true }));
    });
  }

  it("saves an edited prompt as a named preset and restores its exact text when reopened (SW-7086)", async () => {
    await render();
    const edited = "  Describe only the clothing.\n\nNever the faces.  ";
    await act(async () => setText(textarea("Describing prompt"), edited));
    await act(async () => setText(textarea("Save as"), "Clothing only"));
    await click(button("Save prompt"));
    await settle();
    expect(trainingApi.describePromptPresets).toHaveBeenCalledWith("t", expect.objectContaining({
      method: "POST",
      body: JSON.stringify({ name: "Clothing only", prompt: edited }),
    }));

    // Reopen: a fresh dialog lists the saved preset; selecting it restores the exact text.
    await unmountRoot(root, container);
    ({ container, root } = mountRoot());
    await render();
    const options = [...presetPicker().options].map((option) => [option.value, option.textContent]);
    expect(options).toEqual([["default", "Default (built-in)"], ["clothing_only", "Clothing only"]]);
    expect(textarea("Describing prompt").value).toBe(DEFAULT_PROMPT);
    await choosePreset("clothing_only");
    expect(textarea("Describing prompt").value).toBe(edited);
    await runWith([described("item_a", 300, 400)]);
    expect(JSON.parse(trainingApi.batchDescribeJobs.mock.calls[0][3].body).prompt).toBe(edited);
  });

  it("lists the built-in default, protects it, and selecting it sends the default instruction (SW-7086)", async () => {
    presetStore.push({ id: "mine", name: "Mine", prompt: "Describe the shoes.", scope: "global" });
    await render();
    await choosePreset("mine");
    expect(button("Delete").disabled).toBe(false);
    await choosePreset("default");
    expect(button("Delete").disabled).toBe(true);
    expect(textarea("Describing prompt").value).toBe(DEFAULT_PROMPT);
    await runWith([described("item_a", 300, 400)]);
    expect(JSON.parse(trainingApi.batchDescribeJobs.mock.calls[0][3].body)).not.toHaveProperty("prompt");

    // The server refuses writes to the built-in; the refusal is shown, not swallowed.
    trainingApi.describePromptPresets.mockImplementationOnce(async () => {
      throw new Error("The built-in default describing prompt cannot be changed or deleted.");
    });
    await act(async () => setText(textarea("Save as"), "Default"));
    await click(button("Save prompt"));
    await settle();
    expect(document.body.textContent).toContain("cannot be changed or deleted");
  });

  it("describes only the remaining images after a cancel and keeps edited rows", async () => {
    await render();
    await runWith(
      [described("item_a", 300, 400), described("item_b", null, null, { failed: true, error: "Unreadable image." })],
      "canceled",
    );
    const rowA = () => document.body.querySelector('[data-item-id="item_a"] textarea');
    await act(async () => setText(rowA(), "A girl, girl_1, waves."));
    await click(button("Describe remaining (2)"));
    await settle();
    expect(appConfirm).not.toHaveBeenCalled();
    expect(JSON.parse(trainingApi.batchDescribeJobs.mock.calls[1][3].body).itemIds).toEqual(["item_b", "item_c"]);
    await act(async () =>
      nextPoll(snapshot("completed", [described("item_b", 400, 300), described("item_c", 300, 300)], { itemCount: 2 })),
    );
    await settle();
    const rows = document.body.querySelectorAll(".dataset-prompt-batch-row");
    expect([...rows].map((row) => row.dataset.itemId)).toEqual(["item_a", "item_b", "item_c"]);
    expect(rowA().value).toBe("A girl, girl_1, waves.");
    expect(document.body.querySelector('[data-item-id="item_b"]').classList.contains("failed")).toBe(false);
    expect(button("Describe remaining (0)")).toBeUndefined();
    expect([...document.body.querySelectorAll("button")].some((node) => node.textContent.startsWith("Describe remaining"))).toBe(false);
  });

  it("asks before Describe all again discards edited fields", async () => {
    await render();
    await runWith([described("item_a", 300, 400)]);
    // No edits: no confirmation.
    await click(button("Describe all again"));
    await settle();
    expect(appConfirm).not.toHaveBeenCalled();
    expect(trainingApi.batchDescribeJobs).toHaveBeenCalledTimes(2);
    await act(async () => nextPoll(snapshot("completed", [described("item_a", 300, 400)], { progress: 1 })));
    await settle();

    await act(async () => setText(document.body.querySelector('[data-item-id="item_a"] textarea'), "Edited."));
    appConfirm.mockResolvedValueOnce(false);
    await click(button("Describe all again"));
    await settle();
    expect(appConfirm).toHaveBeenCalledTimes(1);
    expect(trainingApi.batchDescribeJobs).toHaveBeenCalledTimes(2);
    expect(document.body.querySelector('[data-item-id="item_a"] textarea').value).toBe("Edited.");

    await click(button("Describe all again"));
    await settle();
    expect(appConfirm).toHaveBeenCalledTimes(2);
    expect(trainingApi.batchDescribeJobs).toHaveBeenCalledTimes(3);
    expect(JSON.parse(trainingApi.batchDescribeJobs.mock.calls[2][3].body)).not.toHaveProperty("itemIds");
  });

  function joyInstalled() {
    context.models = [
      installed("vision_caption_qwen3vl_8b", "Vision Captioner (Qwen3-VL-8B)"),
      installed("joycaption_beta_one", "JoyCaption (beta one)"),
    ];
  }
  const chooseModel = (id) => click(document.body.querySelector(`input[value="${id}"]`));

  it("prefills JoyCaption's own default and sends no prompt for it", async () => {
    joyInstalled();
    await render();
    await chooseModel("joycaption_beta_one");
    await settle();
    expect(trainingApi.defaultDescribePrompt).toHaveBeenLastCalledWith("t", expect.anything(), JOY_REPO);
    expect(textarea("Describing prompt").value).toBe(JOY_DEFAULT_PROMPT);
    await runWith([described("item_a", 300, 400)]);
    expect(JSON.parse(trainingApi.batchDescribeJobs.mock.calls[0][3].body)).toEqual({ model: JOY_REPO, requestedGpu: "auto" });

    // Back to the Vision Captioner: the untouched default swaps back.
    await chooseModel("vision_caption_qwen3vl_8b");
    await settle();
    expect(textarea("Describing prompt").value).toBe(DEFAULT_PROMPT);
  });

  it("keeps an edited prompt when the model changes", async () => {
    joyInstalled();
    await render();
    await act(async () => setText(textarea("Describing prompt"), "Describe only the clothing."));
    await chooseModel("joycaption_beta_one");
    await settle();
    expect(textarea("Describing prompt").value).toBe("Describe only the clothing.");
    await runWith([described("item_a", 300, 400)]);
    expect(JSON.parse(trainingApi.batchDescribeJobs.mock.calls[0][3].body)).toMatchObject({
      model: JOY_REPO,
      prompt: "Describe only the clothing.",
    });
  });

  it("blocks a prompt over the selected model's limit with a validation error before the API", async () => {
    joyInstalled();
    await render();
    const long = "x".repeat(3001);
    await act(async () => setText(textarea("Describing prompt"), long));
    // Within the Vision Captioner's limit.
    expect(document.body.querySelector(".validation-chip")).toBeNull();
    expect(button("Describe images").disabled).toBe(false);

    await chooseModel("joycaption_beta_one");
    await settle();
    expect(textarea("Describing prompt").value).toBe(long);
    const chip = document.body.querySelector(".validation-chip.tone-error");
    expect(chip.textContent).toContain("at most 3,000");
    expect(textarea("Describing prompt").getAttribute("aria-invalid")).toBe("true");
    expect(button("Describe images").disabled).toBe(true);
    await click(button("Describe images"));
    expect(trainingApi.batchDescribeJobs).not.toHaveBeenCalled();
  });

  it("resolves the built-in preset to the selected model's default and flags a too-long preset (SW-7086)", async () => {
    joyInstalled();
    presetStore.push({ id: "long", name: "Long", prompt: "y".repeat(3500), scope: "global" });
    await render();
    await chooseModel("joycaption_beta_one");
    await settle();
    await choosePreset("long");
    expect(document.body.querySelector(".validation-chip.tone-error").textContent).toContain("at most 3,000");
    expect(button("Describe images").disabled).toBe(true);

    await choosePreset("default");
    expect(textarea("Describing prompt").value).toBe(JOY_DEFAULT_PROMPT);
    expect(document.body.querySelector(".validation-chip")).toBeNull();
    await runWith([described("item_a", 300, 400)]);
    expect(JSON.parse(trainingApi.batchDescribeJobs.mock.calls[0][3].body)).not.toHaveProperty("prompt");
  });

  it("records the model that described each row, not the one selected at save time", async () => {
    joyInstalled();
    await render();
    await runWith([described("item_a", 300, 400), described("item_b", 400, 300)]);
    // Describe the remaining image with JoyCaption, then switch back to the Vision Captioner before saving.
    await chooseModel("joycaption_beta_one");
    await settle();
    await click(button("Describe remaining (1)"));
    await settle();
    expect(JSON.parse(trainingApi.batchDescribeJobs.mock.calls[1][3].body)).toMatchObject({
      model: JOY_REPO,
      itemIds: ["item_c"],
    });
    await act(async () =>
      nextPoll(snapshot("completed", [described("item_c", 400, 300)], { itemCount: 1, model: JOY_REPO })),
    );
    await settle();
    await chooseModel("vision_caption_qwen3vl_8b");
    await settle();
    await click(button("Save prompt batch"));
    await settle();
    const payload = context.createPromptBatch.mock.calls[0][0];
    expect(payload.sourceImages.map((image) => [image.itemId, image.model])).toEqual([
      ["item_a", VISION_CAPTION_MODEL_REPO],
      ["item_b", VISION_CAPTION_MODEL_REPO],
      ["item_c", JOY_REPO],
    ]);
    expect(payload.source.model).toBe("mixed");
    // Neither model has tiers: no image records one.
    expect(payload.sourceImages.some((image) => "variant" in image)).toBe(false);
    expect(payload.source.variant).toBeNull();
  });

  // The 27B describer with per-tier install state: `installedTiers` lists the tiers present. The
  // model-level size label describes the default (q4) tier; each variant carries its own bytes,
  // except bf16 here.
  const TIER_BYTES = { q4: 19995477764, q8: 32371297322 };
  function describerWithTiers(...installedTierKeys) {
    return {
      id: "vision_describe_qwen3_8_27b",
      name: "Vision Describer (Qwen3.8-27B)",
      installState: installedTierKeys.length ? "installed" : "missing",
      downloadSizeLabel: "18.6 GB",
      hasVariantMatrix: true,
      variants: ["q4", "q8", "bf16"].map((variant) => ({
        variant,
        installState: installedTierKeys.includes(variant) ? "installed" : "missing",
        downloadSizeBytes: TIER_BYTES[variant] ?? null,
      })),
    };
  }
  const tierInput = (key) => document.body.querySelector(`input[name="dataset-prompt-batch-tier"][value="${key}"]`);
  const rerunButton = (itemId) =>
    [...document.body.querySelectorAll(`[data-item-id="${itemId}"] button`)].find((node) => node.textContent === "Re-run");

  it.each(["failed", "canceled"])("lets a missing Loop tier retry after %s or a refused enqueue", async (status) => {
    const entry = { ...describerWithTiers("q8"), id: "vision_sensenova_loop_9b", name: "SenseNova-Loop-9B" };
    const job = { id: "loop-download", type: "model_download", status: "queued",
      payload: { modelId: entry.id, variant: "bf16" } };
    context.models = [...context.models, entry];
    context.createModelDownloadJob.mockResolvedValueOnce(null).mockResolvedValueOnce(job);
    await render();
    await chooseModel(entry.id);
    await click(tierInput("bf16"));
    await click(button("Download"));
    expect(button("Download").disabled).toBe(false);
    expect(document.body.textContent).toContain("Could not start the model download");
    await click(button("Download"));
    expect(button("queued").disabled).toBe(true);
    context.jobs = [
      { ...job, id: "other-tier", status: "running", payload: { ...job.payload, variant: "q8" } },
      { ...job, status, error: "The BF16 download did not finish." },
    ];
    await render();
    expect(document.body.textContent).toContain("The BF16 download did not finish");
    expect(button("Download").disabled).toBe(false);
    await click(button("Download"));
    expect(context.createModelDownloadJob).toHaveBeenCalledTimes(3);
    expect(context.createModelDownloadJob).toHaveBeenLastCalledWith(entry, { variant: "bf16" });
  });

  it("submits and saves Loop provenance at an installed tier", async () => {
    const model = "sensenova/SenseNova-Loop-9B";
    const entry = { ...describerWithTiers("q8"), id: "vision_sensenova_loop_9b", name: "SenseNova-Loop-9B" };
    entry.variants = entry.variants.filter((row) => row.variant !== "q4");
    context.models = [...context.models, entry];
    await render();
    await chooseModel(entry.id);
    await settle();
    expect(tierInput("q4")).toBeNull();
    expect(tierInput("q8").checked).toBe(true);
    expect(trainingApi.defaultDescribePrompt).toHaveBeenLastCalledWith("t", expect.anything(), model);
    await click(button("Describe images"));
    await settle();
    expect(JSON.parse(trainingApi.batchDescribeJobs.mock.calls[0][3].body)).toEqual({
      model, variant: "q8", requestedGpu: "auto",
    });
    await act(async () => nextPoll(snapshot("completed", [
      described("item_a", 300, 400),
      described("item_b", 300, 400, { failed: true, error: "Malformed JSON." }),
    ], { model, variant: "q8" })));
    await settle();
    await chooseModel("vision_caption_qwen3vl_8b");
    await click(rerunButton("item_b"));
    await settle();
    expect(JSON.parse(trainingApi.batchDescribeJobs.mock.calls[1][3].body)).toMatchObject({
      model, variant: "q8", itemIds: ["item_b"],
    });
    await act(async () => nextPoll(snapshot("completed", [described("item_b", 300, 400)], { model, variant: "q8" })));
    await settle();
    await click(button("Save prompt batch"));
    expect(context.createPromptBatch.mock.calls[0][0].sourceImages.map(({ model, variant }) => [model, variant]))
      .toEqual([[model, "q8"], [model, "q8"]]);
  });

  it("offers the 27B Vision Describer with a tier choice that defaults to q4 (SW-7122)", async () => {
    context.models = [...context.models, describerWithTiers("q4")];
    await render();
    expect(document.body.querySelector('input[value="vision_caption_qwen3vl_8b"]').checked).toBe(true);
    expect(document.body.querySelector('input[value="vision_describe_qwen3_8_27b"]').closest("label").textContent).toContain(
      "Vision Describer (Qwen3.8-27B)",
    );
    expect(tierInput("q4")).toBeNull();

    await chooseModel("vision_describe_qwen3_8_27b");
    await settle();
    expect([...document.body.querySelectorAll('input[name="dataset-prompt-batch-tier"]')].map((input) => input.closest("label").textContent)).toEqual([
      `q4 (${formatTierBytes(TIER_BYTES.q4)})`,
      `q8 (${formatTierBytes(TIER_BYTES.q8)}) — not installed`,
      // No size served for bf16 in this catalog: none rather than a guessed one.
      "Full precision — not installed",
    ]);
    // The tier choice says what it costs: full precision is about half the describe throughput.
    expect(document.body.querySelector('[data-tier-hint="speed"]').textContent).toContain(
      "Full precision takes about twice as long per image",
    );
    expect(tierInput("q4").checked).toBe(true);
    // The prompt editor follows the 27B's own default, as for the Vision Captioner.
    expect(trainingApi.defaultDescribePrompt).toHaveBeenLastCalledWith("t", expect.anything(), VISION_DESCRIBE_MODEL_REPO);
    expect(textarea("Describing prompt").value).toBe(DEFAULT_PROMPT);
    await runWith([described("item_a", 300, 400)]);
    expect(JSON.parse(trainingApi.batchDescribeJobs.mock.calls[0][3].body)).toEqual({
      model: VISION_DESCRIBE_MODEL_REPO,
      variant: "q4",
      requestedGpu: "auto",
    });

    await chooseModel("vision_caption_qwen3vl_8b");
    await settle();
    expect(tierInput("q4")).toBeNull();
    // The Vision Captioner's run sends no variant.
    await click(button("Describe all again"));
    await settle();
    expect(JSON.parse(trainingApi.batchDescribeJobs.mock.calls[1][3].body)).toEqual({
      model: VISION_CAPTION_MODEL_REPO,
      requestedGpu: "auto",
    });
  });

  it("shows a tier that is not installed as unavailable and downloads only that tier (SW-7122)", async () => {
    const entry = describerWithTiers("q4");
    context.models = [...context.models, entry];
    context.jobs = [
      { id: "dl_q4", type: "model_download", status: "queued", payload: { modelId: entry.id, variant: "q4" } },
      { id: "dl_q8", type: "model_download", status: "running", payload: { modelId: entry.id, variant: "q8" } },
    ];
    await render();
    await chooseModel("vision_describe_qwen3_8_27b");
    await settle();
    expect(document.body.querySelector(".required-models-notice")).toBeNull();
    expect(button("Describe images").disabled).toBe(false);

    await click(tierInput("bf16"));
    await settle();
    const notice = document.body.querySelector(".required-models-notice");
    expect(notice.textContent).toContain("Describing the dataset needs a model");
    expect(notice.textContent).toContain("Vision Describer (Qwen3.8-27B) Full precision");
    // No per-tier size for bf16: no size rather than the default tier's.
    expect(notice.querySelector("small")).toBeNull();
    expect(button("Describe images").disabled).toBe(true);
    await click(button("Download"));
    expect(context.createModelDownloadJob).toHaveBeenCalledWith(entry, { variant: "bf16" });

    // The progress shown is that tier's download, never another tier's, with that tier's size.
    await click(tierInput("q8"));
    await settle();
    const q8Notice = document.body.querySelector(".required-models-notice");
    // One figure in one unit: the radio and the download panel print the same catalog size.
    expect(q8Notice.textContent).toContain("Vision Describer (Qwen3.8-27B) q8");
    expect(q8Notice.querySelector("small").textContent).toBe(formatTierBytes(TIER_BYTES.q8));
    expect(tierInput("q8").closest("label").textContent).toContain(`(${formatTierBytes(TIER_BYTES.q8)})`);
    const labels = [...q8Notice.querySelectorAll("button")].map((node) => node.textContent.trim());
    expect(labels).toContain("running");
    expect(labels).not.toContain("queued");
    expect(labels).not.toContain("Download");
  });

  it("lets each missing tier be downloaded after another tier's Download was clicked (SW-7122 review)", async () => {
    const entry = describerWithTiers("q4");
    context.models = [...context.models, entry];
    await render();
    await chooseModel("vision_describe_qwen3_8_27b");
    await settle();
    await click(tierInput("bf16"));
    await settle();
    await click(button("Download"));
    await settle();
    expect(button("Queued").disabled).toBe(true);

    await click(tierInput("q8"));
    await settle();
    expect(button("Queued")).toBeUndefined();
    expect(button("Download").disabled).toBe(false);
    await click(button("Download"));
    expect(context.createModelDownloadJob).toHaveBeenLastCalledWith(entry, { variant: "q8" });
  });

  it("records each image's tier from the live job result in the saved batch (SW-7122 review)", async () => {
    context.models = [...context.models, describerWithTiers("q4", "q8")];
    await render();
    await chooseModel("vision_describe_qwen3_8_27b");
    await settle();
    await click(button("Describe images"));
    await settle();
    await act(async () =>
      nextPoll(snapshot("completed", [described("item_a", 300, 400)], { model: VISION_DESCRIBE_MODEL_REPO, variant: "q4" })),
    );
    await settle();
    await click(tierInput("q8"));
    await settle();
    await click(button("Describe remaining (2)"));
    await settle();
    await act(async () =>
      nextPoll(
        snapshot("completed", [described("item_b", 400, 300), described("item_c", 300, 300)], {
          itemCount: 2,
          model: VISION_DESCRIBE_MODEL_REPO,
          variant: "q8",
        }),
      ),
    );
    await settle();
    await click(button("Save prompt batch"));
    await settle();
    const payload = context.createPromptBatch.mock.calls[0][0];
    expect(payload.sourceImages.map((image) => [image.itemId, image.model, image.variant])).toEqual([
      ["item_a", VISION_DESCRIBE_MODEL_REPO, "q4"],
      ["item_b", VISION_DESCRIBE_MODEL_REPO, "q8"],
      ["item_c", VISION_DESCRIBE_MODEL_REPO, "q8"],
    ]);
    // One model at two tiers: the model is single, the tier is not.
    expect(payload.source.model).toBe(VISION_DESCRIBE_MODEL_REPO);
    expect(payload.source.variant).toBe("mixed");
  });

  it("blocks a re-run whose recorded tier the model does not offer instead of substituting (SW-7122 review)", async () => {
    context.models = [...context.models, describerWithTiers("q4", "q8", "bf16")];
    serveHistory([
      {
        id: "job_27b",
        status: "completed",
        result: { processedCount: 2, itemCount: 2, model: VISION_DESCRIBE_MODEL_REPO, variant: "q2" },
      },
      { id: "job_old", status: "completed", result: { processedCount: 1, itemCount: 1, model: VISION_DESCRIBE_MODEL_REPO } },
    ]);
    serveStoredRuns({ job_27b: [described("item_a", 300, 400), described("item_b", 300, 400)], job_old: [described("item_c", 300, 400)] });
    await render();
    expect(rerunButton("item_a").disabled).toBe(true);
    expect(rerunButton("item_a").title).toContain("“q2”");
    // An older row with no recorded tier re-runs on the default tier.
    expect(rerunButton("item_c").disabled).toBe(false);
    await click(rerunButton("item_c"));
    await settle();
    expect(JSON.parse(trainingApi.batchDescribeJobs.mock.calls[0][3].body)).toMatchObject({
      model: VISION_DESCRIBE_MODEL_REPO,
      variant: "q4",
      itemIds: ["item_c"],
    });
  });

  it("loads each model's default once, and a failure does not clear or borrow another's (SW-7122 review)", async () => {
    joyInstalled();
    trainingApi.defaultDescribePrompt.mockImplementation(async (_token, _options, model) => {
      if (model === JOY_REPO) throw new Error("joy default unavailable");
      return { model, ...DEFAULTS[model] };
    });
    serveHistory([
      { id: "job_joy", status: "completed", result: { processedCount: 1, itemCount: 1, model: JOY_REPO } },
      { id: "job_27b", status: "completed", result: { processedCount: 1, itemCount: 1, model: VISION_DESCRIBE_MODEL_REPO, variant: "q4" } },
    ]);
    serveStoredRuns({ job_joy: [described("item_a", 300, 400)], job_27b: [described("item_b", 300, 400)] });
    context.models = [...context.models, describerWithTiers("q4")];
    await render();
    const repos = trainingApi.defaultDescribePrompt.mock.calls.map((call) => call[2]);
    expect(repos.filter((repo) => repo === VISION_CAPTION_MODEL_REPO)).toHaveLength(1);
    expect(repos.filter((repo) => repo === VISION_DESCRIBE_MODEL_REPO)).toHaveLength(1);
    // JoyCaption's failure belongs to its rows' re-run, not to the selected model's editor.
    expect(document.body.textContent).not.toContain("joy default unavailable");
    expect(rerunButton("item_a").disabled).toBe(true);
    expect(rerunButton("item_a").title).toBe("joy default unavailable");
    expect(rerunButton("item_b").disabled).toBe(false);
  });

  it("re-runs a row with the model and tier that described it (SW-7122)", async () => {
    joyInstalled();
    context.models = [...context.models, describerWithTiers("q4", "q8")];
    await render();
    await chooseModel("vision_describe_qwen3_8_27b");
    await settle();
    await click(tierInput("q8"));
    await settle();
    await click(button("Describe images"));
    await settle();
    await act(async () =>
      nextPoll(
        snapshot("completed", [described("item_a", 300, 400), described("item_b", 400, 300)], {
          model: VISION_DESCRIBE_MODEL_REPO,
          variant: "q8",
        }),
      ),
    );
    await settle();

    // Another model selected: the re-run still sends the row's own model and tier.
    await chooseModel("joycaption_beta_one");
    await settle();
    await click(rerunButton("item_b"));
    await settle();
    expect(JSON.parse(trainingApi.batchDescribeJobs.mock.calls[1][3].body)).toEqual({
      model: VISION_DESCRIBE_MODEL_REPO,
      variant: "q8",
      requestedGpu: "auto",
      itemIds: ["item_b"],
    });
    await act(async () => nextPoll(snapshot("completed", [described("item_b", 400, 300)], { itemCount: 1, model: JOY_REPO })));
    await settle();

    // item_b is now JoyCaption's: its re-run sends no variant, even with the 27B selected.
    await chooseModel("vision_describe_qwen3_8_27b");
    await settle();
    await click(rerunButton("item_b"));
    await settle();
    expect(JSON.parse(trainingApi.batchDescribeJobs.mock.calls[2][3].body)).toEqual({
      model: JOY_REPO,
      requestedGpu: "auto",
      itemIds: ["item_b"],
    });
  });

  it("says on each row which model and tier its re-run uses, whatever is selected above", async () => {
    context.models = [...context.models, describerWithTiers("q4")];
    serveHistory([{ id: "job_q4", status: "completed", result: { processedCount: 1, itemCount: 1, model: VISION_DESCRIBE_MODEL_REPO, variant: "q4" } }]);
    serveStoredRuns({ job_q4: [described("item_a", 300, 400)] });
    await render();
    // The 27B at q8, which is not installed, selected above: the notice is about that selection.
    await chooseModel("vision_describe_qwen3_8_27b");
    await settle();
    await click(tierInput("q8"));
    await settle();
    expect(document.body.querySelector(".required-models-notice").textContent).toContain("Vision Describer (Qwen3.8-27B) q8");
    // The row's re-run reproduces the row's own describer and tier, and says so.
    const note = document.body.querySelector('[data-item-id="item_a"] [data-row-note="rerun-with"]');
    expect(note.textContent).toBe("Re-runs with Vision Describer (Qwen3.8-27B) q4");
    expect(rerunButton("item_a").disabled).toBe(false);
    await click(rerunButton("item_a"));
    await settle();
    expect(JSON.parse(trainingApi.batchDescribeJobs.mock.calls.at(-1)[3].body)).toMatchObject({
      model: VISION_DESCRIBE_MODEL_REPO,
      variant: "q4",
      itemIds: ["item_a"],
    });
  });

  it("marks the 27B not installed only when no tier is, and a missing tier on its own radio", async () => {
    const modelLabel = () => document.body.querySelector('input[value="vision_describe_qwen3_8_27b"]').closest("label").textContent;
    context.models = [...context.models, describerWithTiers("q4")];
    await render();
    await chooseModel("vision_describe_qwen3_8_27b");
    await settle();
    await click(tierInput("q8"));
    await settle();
    expect(modelLabel()).not.toContain("not installed");
    expect(tierInput("q8").closest("label").textContent).toContain("— not installed");
  });

  it("marks the 27B not installed when no tier is installed", async () => {
    context.models = [...context.models, describerWithTiers()];
    await render();
    expect(document.body.querySelector('input[value="vision_describe_qwen3_8_27b"]').closest("label").textContent).toContain(
      "— not installed",
    );
  });

  it("says the model files are being checked when the start request is slow", async () => {
    let answer;
    trainingApi.batchDescribeJobs.mockImplementationOnce(() => new Promise((resolve) => { answer = resolve; }));
    await render();
    await click(button("Describe images"));
    await settle();
    const note = () => document.body.querySelector('[data-run-note="checking-files"]');
    expect(note().textContent).toContain("Checking the model files before the job starts.");
    await act(async () => answer({ id: "job_1", status: "queued", progress: 0 }));
    await settle();
    expect(note()).toBeNull();
  });

  it("checks a re-run against the row's own tier install and prompt limit (SW-7122)", async () => {
    joyInstalled();
    context.models = [...context.models, describerWithTiers("q4")];
    serveHistory([
      { id: "job_joy", status: "completed", result: { processedCount: 1, itemCount: 1, model: JOY_REPO } },
      // Restored from the payload's variant: a tier this install no longer has.
      {
        id: "job_27b",
        status: "completed",
        payload: { model: VISION_DESCRIBE_MODEL_REPO, variant: "bf16" },
        result: { processedCount: 1, itemCount: 1, model: VISION_DESCRIBE_MODEL_REPO },
      },
      { id: "job_vc", status: "completed", result: { processedCount: 1, itemCount: 1, model: VISION_CAPTION_MODEL_REPO } },
    ]);
    serveStoredRuns({
      job_joy: [described("item_a", 300, 400)],
      job_27b: [described("item_b", 300, 400)],
      job_vc: [described("item_c", 300, 400)],
    });
    await render();
    expect(rerunButton("item_b").disabled).toBe(true);
    expect(rerunButton("item_b").title).toBe("Vision Describer (Qwen3.8-27B) Full precision is not installed.");
    expect(rerunButton("item_a").disabled).toBe(false);
    expect(rerunButton("item_c").disabled).toBe(false);

    // Over JoyCaption's limit but within the Vision Captioner's: only the JoyCaption row is blocked.
    await act(async () => setText(textarea("Describing prompt"), "x".repeat(3001)));
    await settle();
    expect(rerunButton("item_a").disabled).toBe(true);
    expect(rerunButton("item_a").title).toContain("at most 3,000");
    expect(rerunButton("item_c").disabled).toBe(false);
  });

  it("blocks a run while the dataset has unsaved changes", async () => {
    await render({ dirty: true });
    expect(document.body.textContent).toContain("Save the dataset first");
    expect(button("Describe images").disabled).toBe(true);
  });
  // Earlier runs stored per job: `{ [jobId]: items }`, served by jobsApi.resultItems.
  function serveStoredRuns(byJob) {
    jobsApi.resultItems.mockReset().mockImplementation(async (jobId, _token, { offset = 0 } = {}) => {
      const items = byJob[jobId] ?? [];
      return { items: items.slice(offset), total: items.length, nextOffset: null };
    });
  }
  function rowText(itemId) {
    return document.body.querySelector(`[data-item-id="${itemId}"] textarea`)?.value;
  }

  it("restores earlier runs on open: newest success per image, interrupted run reported", async () => {
    serveHistory([
      // Newest: interrupted by a restart after one image; item_b was flagged.
      { id: "job_new", status: "interrupted", result: { processedCount: 2, itemCount: 3, model: JOY_REPO } },
      { id: "job_old", status: "completed", result: { processedCount: 3, itemCount: 3, model: VISION_CAPTION_MODEL_REPO } },
    ]);
    serveStoredRuns({
      job_new: [
        described("item_a", 300, 400, { description: "Newest a." }),
        described("item_b", 400, 300, { failed: true, error: "Bad reply." }),
      ],
      job_old: [
        described("item_a", 300, 400, { description: "Old a." }),
        described("item_b", 400, 300, { description: "Old b." }),
        described("item_gone", 400, 300, { description: "Removed from the dataset." }),
      ],
    });
    await render();
    const rows = [...document.body.querySelectorAll(".dataset-prompt-batch-row")].map((row) => row.dataset.itemId);
    expect(rows).toEqual(["item_a", "item_b"]);
    expect(rowText("item_a")).toBe("Newest a.");
    // The newer run flagged item_b; the older success is kept instead.
    expect(rowText("item_b")).toBe("Old b.");
    expect(document.body.textContent).toContain("Restored 2 described images from earlier runs.");
    expect(document.body.textContent).toContain("The last run stopped (interrupted) after 2 of 3 images.");
    expect(button("Describe remaining (1)")).toBeTruthy();
    await click(button("Save prompt batch"));
    await settle();
    const payload = context.createPromptBatch.mock.calls[0][0];
    expect(payload.sourceImages.map((image) => [image.itemId, image.model])).toEqual([
      ["item_a", JOY_REPO],
      ["item_b", VISION_CAPTION_MODEL_REPO],
    ]);
  });

  it("resumes monitoring a run that is still going when the dialog reopens", async () => {
    serveHistory([
      { id: "job_1", status: "running", progress: 0.3, result: { processedCount: 1, itemCount: 3, model: VISION_CAPTION_MODEL_REPO } },
    ]);
    storedItems = [described("item_a", 300, 400)];
    await render();
    expect(rowText("item_a")).toContain("(item_a)");
    expect(button("Cancel")).toBeTruthy();
    await act(async () => nextPoll(snapshot("completed", [described("item_a", 300, 400), described("item_b", 400, 300)], { progress: 1 })));
    await settle();
    expect([...document.body.querySelectorAll(".dataset-prompt-batch-row")].map((row) => row.dataset.itemId)).toEqual([
      "item_a",
      "item_b",
    ]);
    // The resumed poll read only the item it had not seen.
    expect(jobsApi.resultItems.mock.calls.at(-1)[2]).toEqual({ offset: 1 });
  });

  it("replaces a saved batch with the same name after a confirm instead of failing", async () => {
    context.promptBatches = [
      {
        id: "film_prints_prompts",
        name: "Film Prints Prompts ",
        scope: "global",
        prompts: ["old"],
        variables: [{ key: "woman_1", values: ["Ada"] }, { key: "man_1", values: ["Bo"] }],
      },
    ];
    await render();
    await runWith([described("item_a", 300, 400)]);
    appConfirm.mockResolvedValueOnce(false);
    await click(button("Save prompt batch"));
    await settle();
    expect(context.updatePromptBatch).not.toHaveBeenCalled();
    expect(context.createPromptBatch).not.toHaveBeenCalled();
    await click(button("Save prompt batch"));
    await settle();
    expect(context.createPromptBatch).not.toHaveBeenCalled();
    expect(context.updatePromptBatch).toHaveBeenCalledWith("film_prints_prompts", expect.objectContaining({ name: "Film prints prompts" }), "global");
    // The filled-in value for a key the new prompts still use is kept.
    expect(context.updatePromptBatch.mock.calls[0][1].variables).toEqual([{ key: "woman_1", values: ["Ada"] }]);
    expect(button("Open in Image Studio")).toBeTruthy();
  });

  it("reads every run newer than a still-running older job before stopping", async () => {
    // Older job_1 (A, B) is still running; job_mid re-ran A; job_new (newest) re-ran C.
    serveHistory([
      { id: "job_new", status: "completed", result: { processedCount: 1, itemCount: 1 } },
      { id: "job_mid", status: "completed", result: { processedCount: 1, itemCount: 1 } },
      { id: "job_1", status: "running", result: { processedCount: 2, itemCount: 3 } },
    ]);
    serveStoredRuns({
      job_new: [described("item_c", 300, 400, { description: "Newest c." })],
      job_mid: [described("item_a", 300, 400, { description: "Re-run a." })],
      job_1: [described("item_a", 300, 400, { description: "Old a." }), described("item_b", 400, 300, { description: "Old b." })],
    });
    await render();
    expect(rowText("item_a")).toBe("Re-run a.");
    expect(rowText("item_b")).toBe("Old b.");
    expect(rowText("item_c")).toBe("Newest c.");
    expect(button("Cancel")).toBeTruthy();
  });

  it("keeps a newer run's result when a resumed older job later describes the same image", async () => {
    serveHistory([
      { id: "job_new", status: "completed", result: {} },
      { id: "job_1", status: "running", result: { processedCount: 1, itemCount: 3 } },
    ]);
    // job_1 (the polled job) has described only item_a so far; job_new already re-ran item_b.
    storedItems = [described("item_a", 300, 400, { description: "Old a." })];
    const newer = [described("item_b", 400, 300, { description: "Newer b." })];
    jobsApi.resultItems.mockImplementation(async (jobId, _token, { offset = 0 } = {}) => {
      const items = jobId === "job_new" ? newer : storedItems;
      return { items: items.slice(offset), total: items.length, nextOffset: null };
    });
    await render();
    expect(rowText("item_b")).toBe("Newer b.");
    await act(async () =>
      nextPoll(
        snapshot(
          "completed",
          [
            described("item_a", 300, 400, { description: "Old a." }),
            described("item_b", 400, 300, { description: "Older b." }),
            described("item_c", 300, 400, { description: "Old c." }),
          ],
          { progress: 1 },
        ),
      ),
    );
    await settle();
    expect(rowText("item_b")).toBe("Newer b.");
    expect(rowText("item_c")).toBe("Old c.");
  });

  it("pages back through the history until every image is covered", async () => {
    // A full page of newer single-image re-runs of item_a, then the original full run on page 2.
    const reruns = Array.from({ length: RESTORE_JOBS_PAGE }, (_, index) => ({ id: `rerun_${index}`, status: "completed", result: {} }));
    serveHistory([...reruns, { id: "job_full", status: "completed", result: {} }]);
    serveStoredRuns({
      ...Object.fromEntries(reruns.map((run) => [run.id, [described("item_a", 300, 400, { description: "Re-run a." })]])),
      job_full: [described("item_a", 300, 400), described("item_b", 400, 300, { description: "Full b." }), described("item_c", 300, 400, { description: "Full c." })],
    });
    await render();
    expect(rowText("item_a")).toBe("Re-run a.");
    expect(rowText("item_b")).toBe("Full b.");
    expect(rowText("item_c")).toBe("Full c.");
    expect(trainingApi.listBatchDescribeJobs.mock.calls.filter((call) => !call[3].unfinished).map((call) => call[3].offset)).toEqual([
      0,
      RESTORE_JOBS_PAGE,
    ]);
  });

  it("resumes a running job that sits past the first page of history", async () => {
    const reruns = Array.from({ length: RESTORE_JOBS_PAGE }, (_, index) => ({ id: `rerun_${index}`, status: "canceled", result: {} }));
    serveHistory([...reruns, { id: "job_1", status: "running", progress: 0.3, result: { processedCount: 1, itemCount: 3 } }]);
    storedItems = [described("item_a", 300, 400)];
    jobsApi.resultItems.mockImplementation(async (jobId, _token, { offset = 0 } = {}) => {
      const items = jobId === "job_1" ? storedItems : [];
      return { items: items.slice(offset), total: items.length, nextOffset: null };
    });
    await render();
    expect(button("Cancel")).toBeTruthy();
    await act(async () => nextPoll(snapshot("completed", [described("item_a", 300, 400), described("item_b", 400, 300)], { progress: 1 })));
    await settle();
    expect(rowText("item_b")).toContain("(item_b)");
  });

  it("drops a restored result whose image was cropped or repointed since", async () => {
    const changed = {
      ...dataset,
      items: [
        { id: "item_a", path: "images/item_a.png", contentHash: "hash-a" },
        { id: "item_b", path: "images/item_b_cropped.png", width: 300, height: 300 },
        { id: "item_c", path: "images/item_c.png", contentHash: "hash-c2" },
      ],
    };
    serveHistory([
      {
        id: "job_old",
        status: "completed",
        payload: {
          items: [
            { itemId: "item_a", relativePath: "images/item_a.png", contentHash: "hash-a" },
            // An older job without hashes: the path tells item_b changed.
            { itemId: "item_b", relativePath: "images/item_b.png" },
            { itemId: "item_c", relativePath: "images/item_c.png", contentHash: "hash-c1" },
          ],
        },
        result: { processedCount: 3, itemCount: 3 },
      },
    ]);
    serveStoredRuns({
      job_old: [described("item_a", 300, 400), described("item_b", 400, 300), described("item_c", 300, 400)],
    });
    await render({ dataset: changed });
    expect([...document.body.querySelectorAll(".dataset-prompt-batch-row")].map((row) => row.dataset.itemId)).toEqual(["item_a"]);
    expect(button("Describe remaining (2)")).toBeTruthy();
  });

  it("asks before closing with unsaved edits", async () => {
    const onClose = vi.fn();
    await render({ onClose });
    await runWith([described("item_a", 300, 400)]);
    await act(async () => setText(document.body.querySelector('[data-item-id="item_a"] textarea'), "Edited."));
    appConfirm.mockResolvedValueOnce(false);
    await click(button("Close"));
    await settle();
    expect(onClose).not.toHaveBeenCalled();
    await click(button("Close"));
    await settle();
    expect(onClose).toHaveBeenCalledTimes(1);
  });
  // ── Fit to (SW-7163) ──────────────────────────────────────────────────────────────────────────

  const KREA = { id: "krea_2_turbo", name: "Krea 2 Turbo", installState: "missing", image: { maxPromptTokens: 1024 } };
  const KREA_RAW = { id: "krea_2_raw", name: "Krea 2 Raw", installState: "missing", image: { maxPromptTokens: 1024 } };
  function condensedFor(target, over = {}) {
    return {
      target,
      limitTokens: 1024,
      text: "A woman, woman_1, stands by a window.",
      tokens: 612,
      tokenCount: "exact",
      estimateReason: null,
      characters: 37,
      over: false,
      failed: false,
      notice: null,
      ...over,
    };
  }
  async function chooseFitTo(value) {
    const select = [...document.body.querySelectorAll("label")]
      .find((node) => node.firstChild?.textContent?.trim() === "Fit to")
      .querySelector("select");
    await act(async () => {
      select.value = value;
      select.dispatchEvent(new window.Event("change", { bubbles: true }));
    });
    await settle();
  }
  const rowNote = (itemId, kind) => document.body.querySelector(`[data-item-id="${itemId}"] [data-row-note="${kind}"]`);
  const condensedBox = (itemId) =>
    [...document.body.querySelectorAll(`[data-item-id="${itemId}"] label`)]
      .find((node) => node.firstChild?.textContent?.trim() === "Condensed prompt")
      ?.querySelector("textarea");

  it("offers the catalog's prompt-limited models as Fit to targets and sends the choice with a describe", async () => {
    context.models = [...context.models, KREA, { id: "z_image_turbo", name: "Z-Image Turbo", image: {} }];
    await render();
    const fitToLabel = [...document.body.querySelectorAll("label")].find((node) => node.firstChild?.textContent?.trim() === "Fit to");
    const options = [...fitToLabel.querySelectorAll("option")].map((node) => [node.value, node.textContent]);
    expect(options).toEqual([
      ["", "Off — save the seven fields"],
      ["krea_2_turbo", "Krea 2 Turbo — 1,024 tokens"],
    ]);
    await chooseFitTo("krea_2_turbo");
    expect(trainingApi.batchDescribeJobs).not.toHaveBeenCalled();
    await runWith([described("item_a", 300, 400, { condensed: condensedFor("krea_2_turbo") })]);
    expect(JSON.parse(trainingApi.batchDescribeJobs.mock.calls[0][3].body)).toEqual({
      model: VISION_CAPTION_MODEL_REPO,
      requestedGpu: "auto",
      fitTo: "krea_2_turbo",
    });
    expect(condensedBox("item_a").value).toBe("A woman, woman_1, stands by a window.");
    expect(rowNote("item_a", "condensed-size").textContent).toBe("612 of 1,024 tokens.");
    expect(textarea("Description").value).toBe("A woman, woman_1, stands (item_a).");

    await click(button("Save prompt batch"));
    await settle();
    const saved = context.createPromptBatch.mock.calls[0][0];
    expect(saved.prompts).toEqual(["[3:4]\nA woman, {{woman_1}}, stands by a window."]);
    expect(saved.variables).toEqual([{ key: "woman_1", values: [] }]);

    // Off: the seven fields again, with no job.
    await chooseFitTo("");
    expect(trainingApi.batchDescribeJobs).toHaveBeenCalledTimes(1);
    expect(condensedBox("item_a")).toBeUndefined();
    await click(button("Save prompt batch"));
    await settle();
    expect(context.createPromptBatch.mock.calls.at(-1)[0].prompts[0].startsWith("[3:4]\nDescription: A woman")).toBe(true);
  });

  it("changing the target condenses only the described Vision Captioner rows and describes nothing", async () => {
    context.models = [...context.models, KREA, KREA_RAW];
    serveHistory([
      { id: "job_joy", status: "completed", result: { processedCount: 1, itemCount: 1, model: JOY_REPO } },
      {
        id: "job_vision",
        status: "completed",
        payload: { fitTo: "krea_2_turbo" },
        result: { processedCount: 2, itemCount: 2, model: VISION_CAPTION_MODEL_REPO },
      },
    ]);
    serveStoredRuns({
      job_joy: [described("item_b", 400, 300, { description: "A man, man_1, by JoyCaption." })],
      job_vision: [
        described("item_a", 300, 400, { condensed: condensedFor("krea_2_turbo") }),
        described("item_c", null, null, { failed: true, error: "Bad image." }),
      ],
    });
    await render();
    // Later polls read the condense-only job's stored items.
    jobsApi.resultItems.mockReset().mockImplementation(async (_jobId, _token, { offset = 0 } = {}) => ({
      items: storedItems.slice(offset),
      total: storedItems.length,
      nextOffset: null,
    }));
    // The newest run (JoyCaption) had no target, so the dialog starts with Fit to off.
    await chooseFitTo("krea_2_turbo");
    expect(trainingApi.batchDescribeJobs).not.toHaveBeenCalled();
    expect(condensedBox("item_a").value).toBe("A woman, woman_1, stands by a window.");
    expect(rowNote("item_b", "not-condensed").textContent).toContain("Described by JoyCaption beta-one, which has no text step");

    await chooseFitTo("krea_2_raw");
    const body = JSON.parse(trainingApi.batchDescribeJobs.mock.calls[0][3].body);
    expect(body).toEqual({
      model: VISION_CAPTION_MODEL_REPO,
      requestedGpu: "auto",
      fitTo: "krea_2_raw",
      condenseRows: [
        {
          itemId: "item_a",
          width: 300,
          height: 400,
          notice: null,
          fields: {
            description: "A woman, woman_1, stands (item_a).",
            pose: "",
            style: "1960s color negative print",
            camera: "35mm SLR",
            composition: "Medium shot",
            lighting: "Window light",
            environment: "A kitchen",
          },
        },
      ],
    });
    await act(async () =>
      nextPoll(
        snapshot("completed", [described("item_a", 300, 400, { condensed: condensedFor("krea_2_raw", { text: "woman_1 at a window." }) })], {
          itemCount: 1,
        }),
      ),
    );
    await settle();
    expect(condensedBox("item_a").value).toBe("woman_1 at a window.");
    expect([...document.body.querySelectorAll(".dataset-prompt-batch-row")].map((row) => row.dataset.itemId)).toEqual([
      "item_a",
      "item_b",
      "item_c",
    ]);
    expect(button("Describe all again")).toBeTruthy();
  });

  it("condenses a 27B row with the model and tier that described it", async () => {
    context.models = [...context.models, KREA, describerWithTiers("q4", "q8")];
    serveHistory([
      { id: "job_q8", status: "completed", result: { processedCount: 1, itemCount: 1, model: VISION_DESCRIBE_MODEL_REPO, variant: "q8" } },
      { id: "job_q4", status: "completed", result: { processedCount: 1, itemCount: 1, model: VISION_DESCRIBE_MODEL_REPO, variant: "q4" } },
    ]);
    serveStoredRuns({ job_q8: [described("item_a", 300, 400)], job_q4: [described("item_b", 400, 300)] });
    await render();
    jobsApi.resultItems.mockReset().mockImplementation(async () => ({ items: [], total: 0, nextOffset: null }));
    await chooseFitTo("krea_2_turbo");
    // One condense-only job per tier, each naming the tier its rows were described with.
    const first = JSON.parse(trainingApi.batchDescribeJobs.mock.calls[0][3].body);
    expect(first).toMatchObject({ model: VISION_DESCRIBE_MODEL_REPO, variant: "q8", fitTo: "krea_2_turbo" });
    expect(first.condenseRows.map((row) => row.itemId)).toEqual(["item_a"]);
    await act(async () => nextPoll(snapshot("completed", [], { itemCount: 1, model: VISION_DESCRIBE_MODEL_REPO, variant: "q8" })));
    await settle();
    const second = JSON.parse(trainingApi.batchDescribeJobs.mock.calls[1][3].body);
    expect(second).toMatchObject({ model: VISION_DESCRIBE_MODEL_REPO, variant: "q4", fitTo: "krea_2_turbo" });
    expect(second.condenseRows.map((row) => row.itemId)).toEqual(["item_b"]);
  });

  it("marks a condensed prompt stale after a field edit and holds Save (not Export) until it is condensed again", async () => {
    context.models = [...context.models, KREA];
    await render();
    await chooseFitTo("krea_2_turbo");
    await runWith([described("item_a", 300, 400, { condensed: condensedFor("krea_2_turbo") })]);
    expect(button("Condense for Krea 2 Turbo (1)")).toBeUndefined();
    await act(async () => setText(textarea("Lighting"), "Hard noon light"));
    expect(rowNote("item_a", "condense-stale").textContent).toContain("Fields changed since this was condensed");
    expect(condensedBox("item_a")).toBeUndefined();
    const note = document.body.querySelector('[data-save-note="not-fitted"]').textContent.replace(/\s+/g, " ");
    expect(note).toContain("1 row has no usable condensed prompt for Krea 2 Turbo");
    expect(note).toContain("Condense it, or turn Fit to off to save the seven fields.");
    expect(note).toContain("Export writes it as the seven fields.");
    expect(button("Save prompt batch").disabled).toBe(true);
    // Export still writes what the table holds.
    expect(button("Export .json").disabled).toBe(false);
    await click(button("Condense for Krea 2 Turbo (1)"));
    await settle();
    const body = JSON.parse(trainingApi.batchDescribeJobs.mock.calls[1][3].body);
    expect(body.condenseRows.map((row) => [row.itemId, row.fields.lighting])).toEqual([["item_a", "Hard noon light"]]);
    expect(body.model).toBe(VISION_CAPTION_MODEL_REPO);
    // The one-row condense reads as one row, never the whole dataset, and is not a describe.
    expect(document.body.textContent).toContain("Condensing: 0 of 1 image");
    expect(button("Describing…")).toBeUndefined();
    // Turning Fit to off is the other way out: the seven fields save.
    await act(async () => nextPoll(snapshot("failed", [], { itemCount: 1 })));
    await settle();
    await chooseFitTo("");
    expect(button("Save prompt batch").disabled).toBe(false);
  });

  it("shows an edited prompt's live characters, never its old size, and measures it without generating", async () => {
    context.models = [...context.models, KREA];
    await render();
    await chooseFitTo("krea_2_turbo");
    await runWith([described("item_a", 300, 400, { condensed: condensedFor("krea_2_turbo", { labelMentions: 1 }) })]);
    expect(rowNote("item_a", "condensed-size").textContent).toBe("612 of 1,024 tokens, leaving room for 1 variable value.");
    await act(async () => setText(condensedBox("item_a"), "woman_1 by the window, edited."));
    const size = rowNote("item_a", "condensed-size").textContent;
    expect(size).toContain("of 4,000 characters; tokens not measured since your edit.");
    expect(size).not.toContain("612");
    expect(document.body.querySelector('[data-save-note="unmeasured"]').textContent).toContain(
      "1 condensed prompt was edited and not measured since.",
    );
    expect(button("Save prompt batch").disabled).toBe(false);
    await click(button("Measure"));
    await settle();
    const body = JSON.parse(trainingApi.batchDescribeJobs.mock.calls[1][3].body);
    expect(body).toMatchObject({ measureOnly: true, fitTo: "krea_2_turbo" });
    expect(body.condenseRows[0]).toMatchObject({ itemId: "item_a", condensedText: "woman_1 by the window, edited." });
  });

  // Restored rows on the krea target: item_a condensed (its prompt is then edited), item_b not.
  async function restoreFittedRun(models, result) {
    context.models = models;
    serveHistory([{ id: "job_fit", status: "completed", payload: { fitTo: "krea_2_turbo" }, result: { processedCount: 2, itemCount: 2, ...result } }]);
    serveStoredRuns({
      job_fit: [described("item_a", 300, 400, { condensed: condensedFor("krea_2_turbo") }), described("item_b", 400, 300)],
    });
    await render();
    await act(async () => setText(condensedBox("item_a"), "woman_1 by the window, edited."));
  }

  it("measures an edited prompt with no describer installed, while Condense stays blocked", async () => {
    await restoreFittedRun(
      [{ id: "vision_caption_qwen3vl_8b", name: "Vision Captioner (Qwen3-VL-8B)", installState: "missing" }, KREA],
      { model: VISION_CAPTION_MODEL_REPO },
    );
    expect(button("Condense for Krea 2 Turbo (1)").disabled).toBe(true);
    expect(button("Measure").disabled).toBe(false);
    await click(button("Measure"));
    await settle();
    const body = JSON.parse(trainingApi.batchDescribeJobs.mock.calls.at(-1)[3].body);
    expect(body).toMatchObject({ model: VISION_CAPTION_MODEL_REPO, measureOnly: true, fitTo: "krea_2_turbo" });
    expect(body.condenseRows.map((row) => row.itemId)).toEqual(["item_a"]);
  });

  it("measures a 27B row whose tier was uninstalled, while Condense for it stays blocked", async () => {
    await restoreFittedRun([...context.models, KREA, describerWithTiers("q4")], {
      model: VISION_DESCRIBE_MODEL_REPO,
      variant: "q8",
    });
    const condense = button("Condense for Krea 2 Turbo (1)");
    expect(condense.disabled).toBe(true);
    expect(condense.title).toBe("Vision Describer (Qwen3.8-27B) q8 is not installed, so its rows cannot be condensed.");
    expect(button("Measure").disabled).toBe(false);
    await click(button("Measure"));
    await settle();
    const body = JSON.parse(trainingApi.batchDescribeJobs.mock.calls.at(-1)[3].body);
    // The tier only names how the row was described; a measure-only job loads nothing.
    expect(body).toEqual({
      model: VISION_DESCRIBE_MODEL_REPO,
      variant: "q8",
      requestedGpu: "auto",
      fitTo: "krea_2_turbo",
      condenseRows: [expect.objectContaining({ itemId: "item_a", condensedText: "woman_1 by the window, edited." })],
      measureOnly: true,
    });
  });

  it("remembers a cleared Fit to for the dataset even though the newest run had a target", async () => {
    context.models = [...context.models, KREA];
    window.localStorage.removeItem("sceneworks.datasetPromptBatch.fitTo.ds_1");
    serveHistory([
      { id: "job_v", status: "completed", payload: { fitTo: "krea_2_turbo" }, result: { processedCount: 1, itemCount: 1, model: VISION_CAPTION_MODEL_REPO } },
    ]);
    serveStoredRuns({ job_v: [described("item_a", 300, 400, { condensed: condensedFor("krea_2_turbo") })] });
    await render();
    expect(condensedBox("item_a")).toBeTruthy();
    await chooseFitTo("");
    await unmountRoot(root, container);
    ({ container, root } = mountRoot());
    await render();
    expect(condensedBox("item_a")).toBeUndefined();
    window.localStorage.removeItem("sceneworks.datasetPromptBatch.fitTo.ds_1");
  });

  it("restores a run while an image is provisional and shows its condensed prompt once it is final", async () => {
    context.models = [...context.models, KREA];
    window.localStorage.setItem("sceneworks.datasetPromptBatch.fitTo.ds_1", "krea_2_turbo");
    serveHistory([
      { id: "job_1", status: "running", payload: { fitTo: "krea_2_turbo" }, result: { processedCount: 0, itemCount: 1, model: VISION_CAPTION_MODEL_REPO } },
    ]);
    // Stored before its condensing step: provisional, not yet counted.
    storedItems = [described("item_a", 300, 400, { provisional: true })];
    await render();
    expect(rowNote("item_a", "not-condensed")).toBeTruthy();
    await act(async () =>
      nextPoll(snapshot("completed", [described("item_a", 300, 400, { condensed: condensedFor("krea_2_turbo") })], { itemCount: 1 })),
    );
    await settle();
    expect(condensedBox("item_a").value).toBe("A woman, woman_1, stands by a window.");
    window.localStorage.removeItem("sceneworks.datasetPromptBatch.fitTo.ds_1");
  });

  it("re-reads a provisional row an ordinary poll saw until it is final", async () => {
    context.models = [...context.models, KREA];
    await render();
    await chooseFitTo("krea_2_turbo");
    await click(button("Describe images"));
    await settle();
    const first = snapshot("running", [
      described("item_a", 300, 400, { condensed: condensedFor("krea_2_turbo") }),
      described("item_b", 400, 300, { provisional: true }),
    ]);
    first.result.processedCount = 1;
    await act(async () => nextPoll(first));
    await settle();
    expect(rowNote("item_b", "not-condensed")).toBeTruthy();
    await act(async () =>
      nextPoll(
        snapshot("completed", [
          described("item_a", 300, 400, { condensed: condensedFor("krea_2_turbo") }),
          described("item_b", 400, 300, { condensed: condensedFor("krea_2_turbo", { text: "woman_1 at the door." }) }),
        ]),
      ),
    );
    await settle();
    expect(condensedBox("item_b").value).toBe("woman_1 at the door.");
    expect(jobsApi.resultItems).toHaveBeenCalledWith("job_1", "t", { offset: 1, limit: 1 }, expect.anything());
    expect(button("Save prompt batch").disabled).toBe(false);
  });

  it("clears a remembered Fit to the catalog no longer offers and never sends it", async () => {
    for (const [stored, models] of [
      // A removed model.
      ["user_model_gone", [...context.models, KREA]],
      // A model whose limit is below the 512-token minimum.
      ["small_model", [...context.models, KREA, { id: "small_model", name: "Small", image: { maxPromptTokens: 128 } }]],
    ]) {
      context.models = models;
      trainingApi.batchDescribeJobs.mockClear();
      window.localStorage.setItem("sceneworks.datasetPromptBatch.fitTo.ds_1", stored);
      await unmountRoot(root, container);
      ({ container, root } = mountRoot());
      await render();
      expect(window.localStorage.getItem("sceneworks.datasetPromptBatch.fitTo.ds_1")).toBe("");
      await click(button("Describe images"));
      await settle();
      expect(JSON.parse(trainingApi.batchDescribeJobs.mock.calls[0][3].body).fitTo).toBeUndefined();
      await act(async () => nextPoll(snapshot("completed", [described("item_a", 300, 400)], { progress: 1 })));
      await settle();
    }
    window.localStorage.removeItem("sceneworks.datasetPromptBatch.fitTo.ds_1");
  });

  // A job that stops while condensing leaves the image's description checkpointed but uncounted;
  // the terminal poll drains it so the row shows its fields and says condensing failed.
  async function stopWhileCondensing(status, items, processedCount) {
    context.models = [...context.models, KREA];
    await render();
    await chooseFitTo("krea_2_turbo");
    await click(button("Describe images"));
    await settle();
    const stopped = snapshot(status, items, { itemCount: 3 });
    stopped.result.processedCount = processedCount;
    await act(async () => nextPoll(stopped));
    await settle();
  }

  it("shows the described fields and a plain notice when condensing fails on the first image", async () => {
    await stopWhileCondensing("failed", [described("item_a", 300, 400, { provisional: true })], 0);
    expect(document.body.querySelector('[data-item-id="item_a"] textarea').value).toBe("A woman, woman_1, stands (item_a).");
    expect(rowNote("item_a", "condense-interrupted").textContent).toContain("Condensing failed for this image");
    expect(button("Save prompt batch").disabled).toBe(true);
  });

  it("shows every described image when condensing fails on the third", async () => {
    await stopWhileCondensing(
      "failed",
      [
        described("item_a", 300, 400, { condensed: condensedFor("krea_2_turbo") }),
        described("item_b", 400, 300, { condensed: condensedFor("krea_2_turbo") }),
        described("item_c", 400, 300, { provisional: true }),
      ],
      2,
    );
    expect([...document.body.querySelectorAll(".dataset-prompt-batch-row")].map((row) => row.dataset.itemId)).toEqual([
      "item_a",
      "item_b",
      "item_c",
    ]);
    expect(condensedBox("item_a")).toBeTruthy();
    expect(rowNote("item_c", "condense-interrupted")).toBeTruthy();
  });

  it("keeps an image whose condensing was canceled, with its fields", async () => {
    await stopWhileCondensing("canceled", [described("item_a", 300, 400, { provisional: true })], 1);
    expect(document.body.querySelector('[data-item-id="item_a"] textarea').value).toBe("A woman, woman_1, stands (item_a).");
    expect(rowNote("item_a", "condense-interrupted")).toBeTruthy();
  });

  it("shows an estimate as about, a result still over the limit as over, and a failed condense plainly", async () => {
    context.models = [...context.models, KREA];
    await render();
    await chooseFitTo("krea_2_turbo");
    await runWith([
      described("item_a", 300, 400, {
        condensed: condensedFor("krea_2_turbo", { tokenCount: "estimate", tokens: 700, estimateReason: "Krea 2 Turbo is not installed" }),
      }),
      described("item_b", 400, 300, {
        condensed: condensedFor("krea_2_turbo", {
          tokenCount: "estimate",
          tokens: 1100,
          over: true,
          notice: "Still over the limit after one retry (about 1100 of 870 tokens). Shorten it before generating.",
        }),
      }),
      described("item_c", 400, 300, {
        condensed: condensedFor("krea_2_turbo", {
          failed: true,
          text: "",
          notice: "The condensed prompt leaves out woman_1 even after one retry, so the seven fields are used.",
        }),
      }),
    ]);
    const sizeA = rowNote("item_a", "condensed-size");
    expect(sizeA.textContent).toBe("about 700 of 1,024 tokens (estimated: Krea 2 Turbo is not installed).");
    expect(sizeA.classList.contains("inline-warning")).toBe(false);
    const sizeB = rowNote("item_b", "condensed-size");
    expect(sizeB.classList.contains("inline-warning")).toBe(true);
    expect(sizeB.textContent).toMatch(/^Over the limit: about 1,100 of 1,024 tokens \(estimated\)\. Still over the limit after one retry/);
    expect(rowNote("item_c", "condense-failed").textContent).toContain("leaves out woman_1");
    expect(condensedBox("item_c")).toBeUndefined();
    // A failed row never silently saves its seven fields under an active target.
    expect(button("Save prompt batch").disabled).toBe(true);
    expect(document.body.querySelector('[data-save-note="not-fitted"]').textContent).toContain(
      "1 row has no usable condensed prompt",
    );
    // The over row saves its condensed prompt and is reported as over, never as fitting.
    expect(document.body.querySelector('[data-save-note="over"]').textContent).toContain(
      "1 condensed prompt is still over the limit",
    );
    await chooseFitTo("");
    await click(button("Save prompt batch"));
    await settle();
    const prompts = context.createPromptBatch.mock.calls[0][0].prompts;
    expect(prompts.every((prompt) => prompt.includes("\nDescription: "))).toBe(true);
  });

  it("condenses over-limit rows again, condenses one row on demand, and filters to the rows needing attention", async () => {
    context.models = [...context.models, KREA];
    await render();
    await chooseFitTo("krea_2_turbo");
    await runWith([
      described("item_a", 300, 400, { condensed: condensedFor("krea_2_turbo") }),
      described("item_b", 400, 300, { condensed: condensedFor("krea_2_turbo", { tokens: 1100, over: true }) }),
      described("item_c", 400, 300, { condensed: condensedFor("krea_2_turbo", { failed: true, text: "" }) }),
    ]);
    const shown = () => [...document.body.querySelectorAll(".dataset-prompt-batch-row")].map((row) => row.dataset.itemId);
    const filter = document.body.querySelector('[data-save-note="attention-filter"] input');
    expect(document.body.querySelector('[data-save-note="attention-filter"]').textContent).toContain("2 rows that need attention");
    await act(async () => filter.click());
    expect(shown()).toEqual(["item_b", "item_c"]);

    const calls = trainingApi.batchDescribeJobs.mock.calls.length;
    await click(button("Condense over-limit again (1)"));
    await settle();
    const over = JSON.parse(trainingApi.batchDescribeJobs.mock.calls[calls][3].body);
    expect(over).toMatchObject({ fitTo: "krea_2_turbo", model: VISION_CAPTION_MODEL_REPO });
    expect(over.condenseRows.map((row) => row.itemId)).toEqual(["item_b"]);
    await act(async () => nextPoll(snapshot("completed", [], { itemCount: 1 })));
    await settle();

    const again = document.body.querySelector('[data-item-id="item_c"] [data-row-action="condense-again"]');
    expect(again.disabled).toBe(false);
    await click(again);
    await settle();
    const one = JSON.parse(trainingApi.batchDescribeJobs.mock.calls[calls + 1][3].body);
    expect(one.condenseRows.map((row) => row.itemId)).toEqual(["item_c"]);
  });

  it("an emptied condensed prompt blocks Save with the same message as a stale row", async () => {
    context.models = [...context.models, KREA];
    await render();
    await chooseFitTo("krea_2_turbo");
    await runWith([described("item_a", 300, 400, { condensed: condensedFor("krea_2_turbo") })]);
    expect(button("Save prompt batch").disabled).toBe(false);
    await act(async () => setText(condensedBox("item_a"), "   "));
    expect(rowNote("item_a", "condensed-empty").textContent).toContain("The condensed prompt is empty");
    expect(button("Save prompt batch").disabled).toBe(true);
    expect(button("Export .json").disabled).toBe(false);
    expect(document.body.querySelector('[data-save-note="not-fitted"]').textContent.replace(/\s+/g, " ")).toContain(
      "Condense it, or turn Fit to off to save the seven fields.",
    );
    await act(async () => setText(condensedBox("item_a"), "woman_1 by the window."));
    expect(button("Save prompt batch").disabled).toBe(false);
  });

  it("decides every row state with the one fitted definition the formatter uses", () => {
    const krea = (over = {}) => condensedFor("krea_2_turbo", over);
    const vision = VISION_CAPTION_MODEL_REPO;
    const row = (itemId, over) => ({ itemId, model: vision, width: 300, height: 400, failed: false, fields: { description: "woman_1" }, ...over });
    const rows = [
      row("fitted", { condensed: krea() }),
      row("over", { condensed: krea({ over: true }) }),
      row("edited", { condensed: krea({ edited: true, text: "woman_1 edited." }) }),
      row("failed", { condensed: krea({ failed: true, text: "" }) }),
      row("stale", { condensed: krea(), fieldsEdited: true }),
      row("emptied", { condensed: krea({ edited: true, text: "  " }) }),
      row("provisional", { provisional: true, condensed: null }),
      row("other_target", { condensed: condensedFor("krea_2_raw") }),
      row("joycaption", { model: JOY_REPO, condensed: null }),
      row("flagged", { failed: true, condensed: null }),
    ];
    expect(rowsNotFitted(rows, "krea_2_turbo").map((r) => r.itemId)).toEqual(["failed", "stale", "emptied", "provisional", "other_target"]);
    // Everything the predicate passes saves a condensed block; the rest are not saved under it.
    const { text } = formatPromptBatch(rows.filter((r) => !rowsNotFitted(rows, "krea_2_turbo").includes(r)), { fitTo: "krea_2_turbo" });
    const blocks = text.split("\n---\n");
    expect(blocks.filter((block) => block.includes("\nDescription: "))).toHaveLength(1); // the JoyCaption row
    expect(rowsNotFitted(rows, "")).toEqual([]);
  });

  it("warns on the over-4,000-character limit using the block that will be saved", async () => {
    context.models = [...context.models, KREA];
    await render();
    await chooseFitTo("krea_2_turbo");
    await runWith([described("item_a", 2400, 3200, { environment: "x".repeat(3900), condensed: condensedFor("krea_2_turbo") })]);
    expect(rowNote("item_a", "too-long")).toBeNull();
    await chooseFitTo("");
    expect(rowNote("item_a", "too-long")).toBeTruthy();
  });

  it("says why Fit to does nothing for JoyCaption and sends no target with it", async () => {
    context.models = [installed("vision_caption_qwen3vl_8b", "Vision Captioner"), installed("joycaption_beta_one", "JoyCaption"), KREA];
    await render();
    await chooseFitTo("krea_2_turbo");
    await click(document.body.querySelector('input[value="joycaption_beta_one"]'));
    await settle();
    expect(document.body.textContent).toContain("JoyCaption rows are not condensed: Fit to needs the Vision Captioner's text step.");
    await runWith([described("item_a", 300, 400)], "completed");
    expect(JSON.parse(trainingApi.batchDescribeJobs.mock.calls[0][3].body).fitTo).toBeUndefined();
  });
});
