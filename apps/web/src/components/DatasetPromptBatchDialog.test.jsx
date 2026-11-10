import React, { act } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { click, mountRoot, settle, unmountRoot } from "../testUtils/dom.js";
import { AppContext } from "../context/AppContext.js";
import { VISION_CAPTION_MODEL_REPO } from "../constants.js";

vi.mock("../api/trainingResources.js", () => ({
  trainingApi: {
    defaultDescribePrompt: vi.fn(),
    batchDescribeJobs: vi.fn(),
    describePromptPresets: vi.fn(),
    describePromptPreset: vi.fn(),
  },
}));
vi.mock("../api/jobsResources.js", () => ({ jobsApi: { item: vi.fn(), cancel: vi.fn() } }));
vi.mock("../pollJob.js", () => ({ abortableDelay: vi.fn(() => Promise.resolve()) }));
vi.mock("../saveExportFile.js", () => ({ saveExportFile: vi.fn(async () => null) }));
vi.mock("../appConfirm.jsx", () => ({ appConfirm: vi.fn(async () => true) }));

const { trainingApi } = await import("../api/trainingResources.js");
const { jobsApi } = await import("../api/jobsResources.js");
const { saveExportFile } = await import("../saveExportFile.js");
const { appConfirm } = await import("../appConfirm.jsx");
const { DatasetPromptBatchDialog } = await import("./DatasetPromptBatchDialog.jsx");

const DEFAULT_PROMPT = "You are a meticulous scene describer.\n## LABELS";
const JOY_REPO = "fancyfeast/llama-joycaption-beta-one-hf-llava";
const JOY_DEFAULT_PROMPT = "You are a meticulous scene describer. Describe only what is visible.\nLABELS: ...";
const DEFAULTS = {
  [VISION_CAPTION_MODEL_REPO]: { prompt: DEFAULT_PROMPT, maxChars: 32000 },
  [JOY_REPO]: { prompt: JOY_DEFAULT_PROMPT, maxChars: 3000 },
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
      items,
    },
  };
}

// jobsApi.item returns pending promises the test resolves one poll at a time.
let polls;
let presetStore;
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
    trainingApi.defaultDescribePrompt
      .mockReset()
      .mockImplementation(async (_token, _options, model) => ({ model, ...DEFAULTS[model || VISION_CAPTION_MODEL_REPO] }));
    trainingApi.batchDescribeJobs.mockReset().mockResolvedValue({ id: "job_1", status: "queued", progress: 0 });
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
    // Re-run one image with JoyCaption, then switch back to the Vision Captioner before saving.
    await chooseModel("joycaption_beta_one");
    await settle();
    const rowB = document.body.querySelector('[data-item-id="item_b"]');
    await click([...rowB.querySelectorAll("button")].find((node) => node.textContent === "Re-run"));
    await settle();
    await act(async () =>
      nextPoll(snapshot("completed", [described("item_b", 400, 300)], { itemCount: 1, model: JOY_REPO })),
    );
    await settle();
    await chooseModel("vision_caption_qwen3vl_8b");
    await settle();
    await click(button("Save prompt batch"));
    await settle();
    const payload = context.createPromptBatch.mock.calls[0][0];
    expect(payload.sourceImages.map((image) => [image.itemId, image.model])).toEqual([
      ["item_a", VISION_CAPTION_MODEL_REPO],
      ["item_b", JOY_REPO],
    ]);
    expect(payload.source.model).toBe("mixed");
  });

  it("blocks a run while the dataset has unsaved changes", async () => {
    await render({ dirty: true });
    expect(document.body.textContent).toContain("Save the dataset first");
    expect(button("Describe images").disabled).toBe(true);
  });
});
