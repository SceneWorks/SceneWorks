import React, { act } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { click, mountRoot, settle, unmountRoot } from "../testUtils/dom.js";
import { AppContext } from "../context/AppContext.js";
import { VISION_CAPTION_MODEL_REPO, VISION_DESCRIBE_MODEL_REPO } from "../constants.js";
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
const { DatasetPromptBatchDialog, RESTORE_JOBS_PAGE } = await import("./DatasetPromptBatchDialog.jsx");

const DEFAULT_PROMPT = "You are a meticulous scene describer.\n## LABELS";
const JOY_REPO = "fancyfeast/llama-joycaption-beta-one-hf-llava";
const JOY_DEFAULT_PROMPT = "You are a meticulous scene describer. Describe only what is visible.\nLABELS: ...";
const DEFAULTS = {
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
      "q4 (20 GB)",
      "q8 (32 GB) — not installed",
      "Full precision (56 GB) — not installed",
    ]);
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
    expect(notice.textContent).toContain("Vision Describer (Qwen3.8-27B) Full precision (56 GB)");
    // No per-tier size for bf16: no size rather than the default tier's.
    expect(notice.querySelector("small")).toBeNull();
    expect(button("Describe images").disabled).toBe(true);
    await click(button("Download"));
    expect(context.createModelDownloadJob).toHaveBeenCalledWith(entry, { variant: "bf16" });

    // The progress shown is that tier's download, never another tier's, with that tier's size.
    await click(tierInput("q8"));
    await settle();
    const q8Notice = document.body.querySelector(".required-models-notice");
    expect(q8Notice.textContent).toContain("q8 (32 GB)");
    expect(q8Notice.querySelector("small").textContent).toBe(formatTierBytes(TIER_BYTES.q8));
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
    expect(rerunButton("item_b").title).toBe("Vision Describer (Qwen3.8-27B) Full precision (56 GB) is not installed.");
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
});
